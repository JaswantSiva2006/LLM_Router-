"""End-to-end validation CLI for routed Qwen2.5 execution."""

from __future__ import annotations

import argparse
import random
import sys
from collections.abc import Callable

import torch
from transformers import AutoTokenizer, Qwen2ForCausalLM

from src.model.routed_qwen import (
    LayerAction,
    enable_qwen2_routing,
    generate_with_route,
    labels_to_path,
    path_to_labels,
    validate_path,
)


def _chat_inputs(tokenizer, question: str, device: torch.device) -> dict[str, torch.Tensor]:
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": question}], tokenize=False, add_generation_prompt=True
    )
    return {key: value.to(device) for key, value in tokenizer(rendered, return_tensors="pt").items()}


def _direct_generate(model, tokenizer, question: str, max_new_tokens: int) -> torch.Tensor:
    inputs = _chat_inputs(tokenizer, question, next(model.parameters()).device)
    with torch.inference_mode():
        return model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        ).cpu()


class Reporter:
    def __init__(self):
        self.failures = 0

    def check(self, name: str, operation: Callable[[], None]) -> None:
        try:
            operation()
        except Exception as exc:
            self.failures += 1
            print(f"FAIL {name}: {type(exc).__name__}: {exc}")
        else:
            print(f"PASS {name}")


def validate_model(model_name: str, max_new_tokens: int = 24) -> int:
    report = Reporter()
    if not torch.cuda.is_available():
        print("FAIL environment: CUDA is required (target: NVIDIA T4, float16)")
        return 1
    device = torch.device("cuda")
    print(f"INFO device: {torch.cuda.get_device_name(device)}; dtype: float16")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = Qwen2ForCausalLM.from_pretrained(model_name, dtype=torch.float16).to(device).eval()
    if model.config.num_hidden_layers != 28:
        print(f"FAIL architecture: expected 28 layers, got {model.config.num_hidden_layers}")
        print("SUMMARY FAIL: 0/8 invariants passed")
        return 1
    print("PASS architecture: 28 transformer layers")
    question = "What is 17 plus 25? Reply with only the final result."
    inputs = _chat_inputs(tokenizer, question, device)

    with torch.inference_mode():
        vanilla_logits = model(**inputs, use_cache=False).logits.detach().cpu()
    vanilla_ids = _direct_generate(model, tokenizer, question, max_new_tokens)
    parameter_ids = tuple(id(parameter) for parameter in model.parameters())
    storage = tuple(parameter.data_ptr() for parameter in model.parameters())
    state_keys = tuple(model.state_dict())
    enable_qwen2_routing(model)
    full = list(range(28))

    def weight_identity() -> None:
        assert parameter_ids == tuple(id(parameter) for parameter in model.parameters())
        assert storage == tuple(parameter.data_ptr() for parameter in model.parameters())
        assert state_keys == tuple(model.state_dict())
        assert all(not parameter.requires_grad for parameter in model.parameters())

    report.check("pretrained weights unchanged and frozen", weight_identity)

    def identity_path() -> None:
        model.model.layer_indices = full
        with torch.inference_mode():
            routed_logits = model(**inputs, use_cache=False).logits.detach().cpu()
        torch.testing.assert_close(routed_logits, vanilla_logits, rtol=1e-3, atol=1e-3)
        assert torch.equal(_direct_generate(model, tokenizer, question, max_new_tokens), vanilla_ids)

    report.check("identity path logits and greedy output", identity_path)

    def hook_counts(path: list[int]) -> list[int]:
        counts = [0] * 28
        handles = [layer.register_forward_hook(
            lambda _module, _args, _output, i=i: counts.__setitem__(i, counts[i] + 1)
        ) for i, layer in enumerate(model.model.layers)]
        try:
            model.model.layer_indices = path
            with torch.inference_mode():
                model(**inputs, use_cache=False)
        finally:
            for handle in handles:
                handle.remove()
            model.model.layer_indices = full
        return counts

    def execution_hooks() -> None:
        assert hook_counts(full) == [1] * 28
        skipped = full.copy()
        skipped.remove(4)
        expected_skip = [1] * 28
        expected_skip[4] = 0
        assert hook_counts(skipped) == expected_skip
        repeated = full.copy()
        repeated.insert(10, 9)
        expected_repeat = [1] * 28
        expected_repeat[9] = 2
        assert hook_counts(repeated) == expected_repeat

    report.check("execution hooks: default, skip, repeat", execution_hooks)

    def determinism() -> None:
        route = full.copy()
        route.remove(4)
        route.insert(route.index(9) + 1, 9)
        outputs = [generate_with_route(model, tokenizer, question, route, max_new_tokens=max_new_tokens) for _ in range(5)]
        assert len(set(outputs)) == 1

    report.check("greedy determinism across 5 runs", determinism)

    def round_trip() -> None:
        rng = random.Random(20261025)
        for _ in range(10_000):
            labels = [LayerAction(rng.randrange(3)) for _ in range(28)]
            assert path_to_labels(labels_to_path(labels)) == labels

    report.check("10,000 label/path round trips", round_trip)

    def invalid_routes() -> None:
        invalid = ([-1], [28], [0, 2, 1], [9, 9, 9], [0.5], [True], "0,1", None)
        for route in invalid:
            try:
                validate_path(route)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                continue
            raise AssertionError(f"invalid route was accepted: {route!r}")

    report.check("invalid routes rejected", invalid_routes)

    def restoration() -> None:
        class ExplodingTokenizer:
            def apply_chat_template(self, *args, **kwargs):
                raise RuntimeError("intentional validation exception")

        try:
            generate_with_route(model, ExplodingTokenizer(), question, [0, 2, 3], max_new_tokens=1)
        except RuntimeError:
            pass
        else:
            raise AssertionError("intentional exception was not raised")
        assert model.model.layer_indices == full
        assert hook_counts(full) == [1] * 28
        generate_with_route(model, tokenizer, question, max_new_tokens=1)
        assert model.model.layer_indices == full

    report.check("route restoration after exception", restoration)
    print(f"SUMMARY {'PASS' if report.failures == 0 else 'FAIL'}: {8 - report.failures}/8 invariants passed")
    return int(report.failures != 0)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--max-new-tokens", type=int, default=24)
    args = parser.parse_args(argv)
    return validate_model(args.model, args.max_new_tokens)


if __name__ == "__main__":
    sys.exit(main())
