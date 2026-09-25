import random
import types

import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from src.model.routed_qwen import (
    LayerAction,
    enable_qwen2_routing,
    generate_with_route,
    labels_to_path,
    path_to_labels,
    validate_path,
)


def tiny_model(num_layers=4):
    torch.manual_seed(7)
    config = Qwen2Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=num_layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
    )
    return Qwen2ForCausalLM(config).eval()


def test_identity_path_preserves_parameters_and_logits_exactly():
    model = tiny_model()
    input_ids = torch.tensor([[1, 7, 3, 9]])
    parameter_objects = tuple(id(p) for p in model.parameters())
    storage_pointers = tuple(p.data_ptr() for p in model.parameters())
    state_keys = tuple(model.state_dict())
    with torch.inference_mode():
        vanilla = model(input_ids, use_cache=False).logits
    enable_qwen2_routing(model)
    with torch.inference_mode():
        routed = model(input_ids, use_cache=False).logits
    assert torch.equal(vanilla, routed)
    assert parameter_objects == tuple(id(p) for p in model.parameters())
    assert storage_pointers == tuple(p.data_ptr() for p in model.parameters())
    assert state_keys == tuple(model.state_dict())
    assert all(not p.requires_grad for p in model.parameters())


def _hook_counts(model, path):
    counts = [0] * model.config.num_hidden_layers
    handles = []
    for index, layer in enumerate(model.model.layers):
        handles.append(layer.register_forward_hook(lambda _m, _a, _o, i=index: counts.__setitem__(i, counts[i] + 1)))
    try:
        model.model.layer_indices = path
        with torch.inference_mode():
            model(torch.tensor([[1, 2, 3]]), use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    return counts


def test_execution_hooks_default_skip_and_repeat():
    model = enable_qwen2_routing(tiny_model())
    assert _hook_counts(model, [0, 1, 2, 3]) == [1, 1, 1, 1]
    assert _hook_counts(model, [0, 2, 3]) == [1, 0, 1, 1]
    assert _hook_counts(model, [0, 1, 1, 2, 3]) == [1, 2, 1, 1]


def test_round_trip_thousands_of_legal_routes():
    rng = random.Random(42)
    for _ in range(10_000):
        labels = [LayerAction(rng.randrange(3)) for _ in range(28)]
        assert path_to_labels(labels_to_path(labels)) == labels


@pytest.mark.parametrize(
    "path",
    [
        [-1],
        [28],
        [0, 2, 1],
        [4, 4, 4],
        [0.0],
        [True],
        "0,1,2",
        None,
    ],
)
def test_invalid_routes_are_rejected(path):
    with pytest.raises((TypeError, ValueError)):
        validate_path(path)


def test_label_validation_and_all_skip_path():
    assert labels_to_path([LayerAction.SKIP] * 28) == []
    with pytest.raises(ValueError):
        labels_to_path([LayerAction.EXECUTE] * 27)
    with pytest.raises(ValueError):
        labels_to_path([3] * 28)


def test_repeat_with_kv_cache_is_explicitly_rejected():
    model = enable_qwen2_routing(tiny_model())
    model.model.layer_indices = [0, 1, 1, 2, 3]
    with pytest.raises(ValueError, match="KV caching is unsafe"):
        model(torch.tensor([[1, 2]]), use_cache=True)


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 2

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return messages[0]["content"]

    def __call__(self, text, return_tensors="pt"):
        return {"input_ids": torch.tensor([[1, 5, 6]]), "attention_mask": torch.ones(1, 3, dtype=torch.long)}

    def decode(self, ids, skip_special_tokens=True):
        return "result"


def test_generation_is_deterministic_and_restores_route_after_success():
    model = enable_qwen2_routing(tiny_model())
    full = list(range(4))
    route = [0, 1, 1, 3]

    def deterministic_generate(self, input_ids, attention_mask, **kwargs):
        assert kwargs["do_sample"] is False
        assert kwargs["temperature"] is None
        assert kwargs["top_p"] is None
        assert kwargs["use_cache"] is False
        self(input_ids, attention_mask=attention_mask, use_cache=False)
        return torch.cat((input_ids, torch.tensor([[11]])), dim=1)

    model.generate = types.MethodType(deterministic_generate, model)
    outputs = [generate_with_route(model, FakeTokenizer(), "question", route) for _ in range(5)]
    assert outputs == ["result"] * 5
    assert model.model.layer_indices == full


def test_route_restored_after_generation_exception_then_default_is_full():
    model = enable_qwen2_routing(tiny_model())
    tokenizer = FakeTokenizer()

    def exploding_generate(self, **kwargs):
        raise RuntimeError("synthetic generation failure")

    model.generate = types.MethodType(exploding_generate, model)
    with pytest.raises(RuntimeError, match="synthetic"):
        generate_with_route(model, tokenizer, "question", [0, 2, 3])
    assert model.model.layer_indices == [0, 1, 2, 3]

    observed = []
    handles = [layer.register_forward_hook(lambda _m, _a, _o, i=i: observed.append(i))
               for i, layer in enumerate(model.model.layers)]

    def working_generate(self, input_ids, attention_mask, **kwargs):
        self(input_ids, attention_mask=attention_mask, use_cache=False)
        return torch.cat((input_ids, torch.tensor([[11]])), dim=1)

    try:
        model.generate = types.MethodType(working_generate, model)
        generate_with_route(model, tokenizer, "question")
    finally:
        for handle in handles:
            handle.remove()
    assert observed == [0, 1, 2, 3]
    assert model.model.layer_indices == [0, 1, 2, 3]

