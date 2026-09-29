"""Kaggle T4x2 preflight for the existing Dr.LLM router trainer."""

from __future__ import annotations

import argparse
import gc
from pathlib import Path

import torch
import transformers
from transformers import AutoTokenizer, Qwen2ForCausalLM

from src.router_training.dataset import load_router_dataset
from src.router_training.loss import focal_loss
from src.router_training.model import NUM_LAYERS, TeacherForcedRouterQwen
from src.router_training.stats import count_labels, effective_number_weights
from src.router_training.train import (
    TokenizedRouterDataset,
    assert_trainable_gradients_fp32,
    build_optimizer,
    make_collator,
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--expected-samples", type=int, default=2916)
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    args = parser.parse_args(argv)

    torch_release = torch.__version__.split("+", 1)[0]
    torch_major_minor = tuple(int(part) for part in torch_release.split(".")[:2])
    _require(torch_major_minor >= (2, 2), f"torch>=2.2 is required, found {torch.__version__}")
    _require(
        transformers.__version__ == "5.2.0",
        f"transformers==5.2.0 is required, found {transformers.__version__}",
    )
    _require(torch.cuda.is_available(), "CUDA is unavailable; enable the Kaggle T4 x2 accelerator")
    _require(torch.cuda.device_count() == 2, f"expected exactly 2 GPUs, found {torch.cuda.device_count()}")
    _require(torch.distributed.is_nccl_available(), "NCCL is unavailable; DDP cannot start")
    print(f"torch={torch.__version__} transformers={transformers.__version__} cuda={torch.version.cuda}")
    for index in range(torch.cuda.device_count()):
        properties = torch.cuda.get_device_properties(index)
        print(
            f"gpu[{index}]={properties.name} "
            f"vram={properties.total_memory / (1024 ** 3):.2f} GiB"
        )

    _require(args.data.is_file(), f"dataset not found: {args.data}")
    records = load_router_dataset(args.data, allow_small_dataset=True)
    _require(
        len(records) == args.expected_samples,
        f"expected {args.expected_samples} records, found {len(records)} in {args.data}",
    )
    counts = count_labels(records)
    weights = effective_number_weights(counts)
    print(f"dataset={args.data.resolve()} samples={len(records)} labels={counts['total']}")
    print(f"class_counts={counts}")
    print(f"class_weights={weights}")

    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = Qwen2ForCausalLM.from_pretrained(args.model, dtype=torch.float16).to(device)
    model = TeacherForcedRouterQwen(base).to(device)
    model.routers.to(device=device, dtype=torch.float32)
    _require(len(model.routers) == NUM_LAYERS, f"expected {NUM_LAYERS} routers")
    _require(all(not parameter.requires_grad for parameter in model.base_model.parameters()), "base model is not frozen")
    _require(all(parameter.requires_grad for parameter in model.routers.parameters()), "a router parameter is frozen")

    trainable = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    assert all("router" in name.lower() for name, _ in trainable)
    assert all(parameter.dtype == torch.float32 for _, parameter in trainable)

    optimizer = build_optimizer(model)
    optimized = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    router_parameters = {id(parameter) for parameter in model.routers.parameters()}
    _require(optimized == router_parameters, "optimizer contains parameters other than routers")

    item = TokenizedRouterDataset(records)[0]
    batch = make_collator(tokenizer)([item])
    batch = {key: value.to(device) for key, value in batch.items()}
    alpha = torch.tensor(weights, dtype=torch.float32, device=device)
    watched = next(model.routers.parameters())
    before = watched.detach().clone()
    scaler = torch.amp.GradScaler("cuda")
    model.train()
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        output = model(**batch, use_cache=False)
        loss = focal_loss(output.router_logits, batch["router_labels"], alpha, gamma=2.0)
    _require(bool(torch.isfinite(loss)), f"smoke loss is not finite: {loss.item()}")
    scaler.scale(loss).backward()
    assert_trainable_gradients_fp32(model)
    scaler.step(optimizer)
    scaler.update()
    _require(not torch.equal(before, watched.detach()), "router weights did not change in smoke step")
    _require(all(parameter.grad is None for parameter in model.base_model.parameters()), "base model received gradients")
    print(
        f"single_gpu_smoke=PASS router_logits={tuple(output.router_logits.shape)} "
        f"loss={loss.item():.8f} trainable_router_parameters={model.router_parameter_count:,}"
    )

    del output, loss, batch, optimizer, model, base, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
