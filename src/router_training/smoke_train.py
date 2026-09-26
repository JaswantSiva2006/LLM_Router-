"""Brief Kaggle GPU overfit test for the frozen-Qwen router training path."""

from __future__ import annotations

import argparse
import itertools

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, Qwen2ForCausalLM

from src.router_training.dataset import load_router_dataset
from src.router_training.loss import focal_loss
from src.router_training.model import TeacherForcedRouterQwen
from src.router_training.stats import count_labels, effective_number_weights
from src.router_training.train import TokenizedRouterDataset, build_optimizer, make_collator


@torch.no_grad()
def evaluate_loss(model, loader, alpha, gamma: float, device: torch.device) -> float:
    model.eval()
    losses = []
    for batch in loader:
        batch = {key: value.to(device) for key, value in batch.items()}
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            output = model(**batch, use_cache=False)
            loss = focal_loss(output.router_logits, batch["router_labels"], alpha, gamma)
        losses.append(loss.float().item())
    return sum(losses) / len(losses)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--allow-small-dataset", action="store_true")
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--precision", choices=("fp16",), default="fp16")
    parser.add_argument("--max-steps", type=int, default=16)
    args = parser.parse_args(argv)
    if not torch.cuda.is_available():
        parser.error("the fp16 smoke train requires a CUDA GPU; run it on Kaggle")
    if args.samples <= 0 or args.max_steps <= 0:
        parser.error("--samples and --max-steps must be positive")

    device = torch.device("cuda")
    records = load_router_dataset(args.data, allow_small_dataset=args.allow_small_dataset)
    if args.samples > len(records):
        parser.error(f"requested {args.samples} samples, but the dataset contains {len(records)}")
    records = records[: args.samples]
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = Qwen2ForCausalLM.from_pretrained(args.model, dtype=torch.float16).to(device)
    model = TeacherForcedRouterQwen(base).to(device)
    optimizer = build_optimizer(model)
    weights = effective_number_weights(count_labels(records))
    alpha = torch.tensor(weights, dtype=torch.float16, device=device)
    loader = DataLoader(
        TokenizedRouterDataset(records), batch_size=1, shuffle=False,
        collate_fn=make_collator(tokenizer),
    )

    base_versions = tuple(parameter._version for parameter in model.base_model.parameters())
    router_before = {key: value.detach().clone() for key, value in model.routers.state_dict().items()}
    initial_loss = evaluate_loss(model, loader, alpha, 2.0, device)
    scaler = torch.amp.GradScaler("cuda")
    model.train()
    cycle = itertools.cycle(loader)
    for _ in range(args.max_steps):
        batch = {key: value.to(device) for key, value in next(cycle).items()}
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            output = model(**batch, use_cache=False)
            loss = focal_loss(output.router_logits, batch["router_labels"], alpha, 2.0)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    final_loss = evaluate_loss(model, loader, alpha, 2.0, device)

    routers_changed = any(
        not torch.equal(value, router_before[key])
        for key, value in model.routers.state_dict().items()
    )
    base_unchanged = base_versions == tuple(
        parameter._version for parameter in model.base_model.parameters()
    ) and all(parameter.grad is None for parameter in model.base_model.parameters())
    checks = {
        "loss decreased": final_loss < initial_loss,
        "routers changed": routers_changed,
        "Qwen weights unchanged": base_unchanged,
    }
    print(f"initial loss: {initial_loss:.8f}")
    print(f"final loss:   {final_loss:.8f}")
    for name, passed in checks.items():
        print(f"{name}: {'PASS' if passed else 'FAIL'}")
    passed = all(checks.values())
    print(f"ROUTER SMOKE TRAIN: {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
