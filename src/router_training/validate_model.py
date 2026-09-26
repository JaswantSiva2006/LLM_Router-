"""Kaggle validation CLI for the teacher-forced Qwen router model."""

from __future__ import annotations

import argparse

import torch

from src.data.gsm8k import build_generation_prompt
from src.router_training.model import NUM_LAYERS, QWEN_HIDDEN_SIZE, TeacherForcedRouterQwen


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--precision", choices=("fp16", "fp32"), default="fp16")
    args = parser.parse_args(argv)
    if args.precision == "fp16" and not torch.cuda.is_available():
        parser.error("fp16 validation requires CUDA; run this command later on a Kaggle T4")

    from transformers import AutoTokenizer, Qwen2ForCausalLM

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if args.precision == "fp16" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    base = Qwen2ForCausalLM.from_pretrained(args.model, dtype=dtype).to(device)
    if base.config.num_hidden_layers != NUM_LAYERS or base.config.hidden_size != QWEN_HIDDEN_SIZE:
        parser.error(
            f"expected Qwen2.5-1.5B dimensions ({NUM_LAYERS} layers, hidden size "
            f"{QWEN_HIDDEN_SIZE}); got ({base.config.num_hidden_layers}, {base.config.hidden_size})"
        )
    model = TeacherForcedRouterQwen(base).to(device).eval()
    encoded = tokenizer(
        build_generation_prompt("What is 2 plus 2?"), return_tensors="pt"
    ).to(device)
    labels = torch.ones((1, NUM_LAYERS), dtype=torch.long, device=device)
    with torch.inference_mode():
        output = model(**encoded, router_labels=labels, use_cache=False)
    print(
        f"validated router_logits={tuple(output.router_logits.shape)} "
        f"router_parameters={model.router_parameter_count:,} dtype={dtype} device={device}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
