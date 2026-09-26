import pytest
import torch
from torch import nn
from transformers import Qwen2Config, Qwen2ForCausalLM

from src.router_training.model import (
    LayerRouter,
    TeacherForcedRouterQwen,
    windowed_router_logits,
)
from src.router_training.loss import focal_loss


def tiny_router_model():
    torch.manual_seed(3)
    config = Qwen2Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=24,
        num_hidden_layers=28,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=32,
    )
    return TeacherForcedRouterQwen(Qwen2ForCausalLM(config))


def test_production_router_architecture_initialization_and_count():
    torch.manual_seed(4)
    router = LayerRouter()
    assert router.input.in_features == 1536
    assert router.input.out_features == 128
    assert router.output.in_features == 128
    assert router.output.out_features == 3
    assert router.activation.approximate == "tanh"
    assert torch.count_nonzero(router.input.bias) == 0
    assert torch.count_nonzero(router.output.bias) == 0
    for weight in (router.input.weight, router.output.weight):
        fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(weight)
        bound = (6 / (fan_in + fan_out)) ** 0.5
        assert weight.abs().max() <= bound
        assert weight.std() > 0

    parameter_count = sum(p.numel() for p in router.parameters()) * 28
    assert parameter_count == 5_519_444


def test_exactly_28_routers_and_only_they_are_trainable():
    model = tiny_router_model()
    assert len(model.routers) == 28
    assert all(not parameter.requires_grad for parameter in model.base_model.parameters())
    assert all(parameter.requires_grad for parameter in model.routers.parameters())
    trainable = {name for name, p in model.named_parameters() if p.requires_grad}
    assert trainable and all(name.startswith("routers.") for name in trainable)


def test_routers_inherit_external_base_dtype():
    base = tiny_router_model().base_model.to(dtype=torch.float64)
    model = TeacherForcedRouterQwen(base)
    assert {parameter.dtype for parameter in model.routers.parameters()} == {torch.float64}


class RecordingRouter(nn.Module):
    def __init__(self):
        super().__init__()
        self.inputs = None

    def forward(self, values):
        self.inputs = values
        return torch.stack((values[:, 0], values[:, 0] * 2, -values[:, 0]), dim=-1)


def test_exact_window_means_and_logits_not_probabilities():
    hidden = torch.arange(1, 13, dtype=torch.float32).reshape(1, 6, 2)
    mask = torch.tensor([[0, 1, 1, 1, 1, 1]])
    router = RecordingRouter()
    actual = windowed_router_logits(hidden, mask, router, num_windows=2)
    valid = hidden[0, 1:5].reshape(2, 2, 2)
    expected_means = valid.mean(dim=1)
    torch.testing.assert_close(router.inputs, expected_means)
    expected = router(expected_means).mean(dim=0, keepdim=True)
    torch.testing.assert_close(actual, expected)
    probability_average = router(expected_means).softmax(-1).mean(0)
    assert not torch.allclose(actual[0], probability_average)


def _layer_call_counts(model, labels):
    counts = [0] * 28
    handles = [layer.register_forward_hook(
        lambda _module, _args, _output, index=index: counts.__setitem__(index, counts[index] + 1)
    ) for index, layer in enumerate(model.base_model.model.layers)]
    try:
        model(torch.tensor([[1, 2, 3]]), router_labels=labels, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    return counts


def test_teacher_forced_skip_execute_repeat_and_predictions_are_ignored():
    model = tiny_router_model()
    # Force every prediction to SKIP; teacher labels must still control execution.
    with torch.no_grad():
        for router in model.routers:
            router.input.weight.zero_()
            router.input.bias.zero_()
            router.output.weight.zero_()
            router.output.bias.copy_(torch.tensor([100.0, -100.0, -100.0]))

    ones = torch.ones((1, 28), dtype=torch.long)
    assert _layer_call_counts(model, ones) == [1] * 28
    skipped = ones.clone()
    skipped[0, 7] = 0
    expected = [1] * 28
    expected[7] = 0
    assert _layer_call_counts(model, skipped) == expected
    repeated = ones.clone()
    repeated[0, 12] = 2
    expected = [1] * 28
    expected[12] = 2
    assert _layer_call_counts(model, repeated) == expected


def test_next_router_observes_teacher_forced_previous_hidden_state():
    model = tiny_router_model()
    observed = []
    handles = [model.routers[index].register_forward_pre_hook(
        lambda _module, args, index=index: observed.append((index, args[0].detach().clone()))
    ) for index in (0, 1)]
    try:
        model(torch.tensor([[1, 2, 3]]), router_labels=torch.ones((1, 28), dtype=torch.long))
    finally:
        for handle in handles:
            handle.remove()
    assert [item[0] for item in observed] == [0, 1]
    assert not torch.equal(observed[0][1], observed[1][1])


def test_only_router_gradients_exist_after_backward():
    model = tiny_router_model()
    targets = torch.ones((1, 28), dtype=torch.long)
    output = model(torch.tensor([[1, 2, 3]]), router_labels=targets)
    focal_loss(output.router_logits, targets, [1, 1, 1]).backward()
    assert any(parameter.grad is not None for parameter in model.routers.parameters())
    assert all(parameter.grad is None for parameter in model.base_model.parameters())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_fp16_cuda_forward():
    model = tiny_router_model().to(device="cuda", dtype=torch.float16)
    output = model(
        torch.tensor([[1, 2, 3]], device="cuda"),
        router_labels=torch.ones((1, 28), dtype=torch.long, device="cuda"),
    )
    assert output.router_logits.dtype == torch.float16
    assert output.router_logits.is_cuda
