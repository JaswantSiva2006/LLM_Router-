"""Audit final-dataset and Kaggle hardware readiness for router training."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
from collections import OrderedDict
from pathlib import Path

import torch
from torch import nn
from torch.optim.lr_scheduler import LambdaLR

from src.router_training.dataset import load_router_dataset
from src.router_training.loss import focal_loss
from src.router_training.model import (
    NUM_LAYERS,
    QWEN_HIDDEN_SIZE,
    TeacherForcedRouterQwen,
    windowed_router_logits,
)
from src.router_training.stats import count_labels, effective_number_weights
from src.router_training.train import (
    build_optimizer,
    load_training_state,
    save_training_state,
)


CHECK_NAMES = (
    "dataset exists", "sample count = 4000", "unique ids", "unique questions",
    "112000 router labels", "class counts", "effective-number weights", "focal loss",
    "28 routers", "base model frozen", "window pooling", "teacher forcing",
    "fp16 T4 compatibility", "DDP 2-GPU compatibility", "checkpoint/resume",
    "existing MCTS tests",
)


def inspect_dataset(path: str | Path) -> tuple[OrderedDict[str, bool], list]:
    path = Path(path)
    checks = OrderedDict((name, False) for name in CHECK_NAMES[:7])
    checks["dataset exists"] = path.is_file()
    if not path.is_file():
        return checks, []
    try:
        records = load_router_dataset(path, allow_small_dataset=True)
    except (ValueError, OSError):
        return checks, []
    checks["sample count = 4000"] = len(records) == 4000
    checks["unique ids"] = len({record.id for record in records}) == len(records)
    checks["unique questions"] = len({record.question for record in records}) == len(records)
    counts = count_labels(records)
    checks["112000 router labels"] = counts["total"] == 112000
    checks["class counts"] = all(counts[key] > 0 for key in ("n_skip", "n_execute", "n_repeat"))
    try:
        weights = effective_number_weights(counts)
        checks["effective-number weights"] = (
            all(math.isfinite(value) and value > 0 for value in weights)
            and math.isclose(sum(weights) / 3, 1.0, rel_tol=1e-12)
        )
    except ValueError:
        pass
    return checks, records


def check_focal_loss() -> bool:
    logits = torch.randn(2, 28, 3, requires_grad=True)
    targets = torch.randint(0, 3, (2, 28))
    loss = focal_loss(logits, targets, [1.0, 1.0, 1.0])
    loss.backward()
    return bool(torch.isfinite(loss) and logits.grad is not None and torch.isfinite(logits.grad).all())


def check_window_pooling() -> bool:
    hidden = torch.arange(1, 13, dtype=torch.float32).reshape(1, 6, 2)
    mask = torch.tensor([[0, 1, 1, 1, 1, 1]])

    class FirstCoordinate(nn.Module):
        def forward(self, values):
            return values[:, :1].expand(-1, 3)

    actual = windowed_router_logits(hidden, mask, FirstCoordinate(), num_windows=2)
    expected = hidden[0, 1:5].reshape(2, 2, 2).mean(1)[:, 0].mean().expand(1, 3)
    return torch.equal(actual, expected)


def check_existing_mcts_tests() -> bool:
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "tests/test_mcts.py", "tests/test_routed_qwen.py"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=environment, check=False,
    )
    return result.returncode == 0


def model_checks(model_name: str, precision: str) -> dict[str, bool]:
    results = {name: False for name in CHECK_NAMES[8:15]}
    if precision != "fp16" or not torch.cuda.is_available():
        return results
    device = torch.device("cuda")
    from transformers import AutoTokenizer, Qwen2ForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    base = Qwen2ForCausalLM.from_pretrained(model_name, dtype=torch.float16).to(device)
    model = TeacherForcedRouterQwen(base).to(device)
    results["28 routers"] = (
        len(model.routers) == NUM_LAYERS and base.config.hidden_size == QWEN_HIDDEN_SIZE
    )
    results["base model frozen"] = (
        all(not parameter.requires_grad for parameter in model.base_model.parameters())
        and all(parameter.requires_grad for parameter in model.routers.parameters())
    )
    results["window pooling"] = check_window_pooling()

    counts = [0] * NUM_LAYERS
    handles = [layer.register_forward_hook(
        lambda _module, _args, _output, index=index: counts.__setitem__(index, counts[index] + 1)
    ) for index, layer in enumerate(model.base_model.model.layers)]
    try:
        encoded = tokenizer("Question: What is 1 plus 1?", return_tensors="pt").to(device)
        labels = torch.ones((1, NUM_LAYERS), dtype=torch.long, device=device)
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
            output = model(**encoded, router_labels=labels, use_cache=False)
        results["teacher forcing"] = counts == [1] * NUM_LAYERS
        results["fp16 T4 compatibility"] = (
            output.router_logits.dtype == torch.float16
            and "T4" in torch.cuda.get_device_name(device).upper()
        )
    finally:
        for handle in handles:
            handle.remove()
    results["DDP 2-GPU compatibility"] = (
        torch.cuda.device_count() >= 2 and torch.distributed.is_nccl_available()
    )

    optimizer = build_optimizer(model)
    scheduler = LambdaLR(optimizer, lambda _: 1.0)
    metadata = {"dataset_sha256": "readiness", "class_weights": [1.0, 1.0, 1.0]}
    with tempfile.TemporaryDirectory() as directory:
        save_training_state(
            directory, model, optimizer, scheduler, None,
            epoch=1, global_step=1, metadata=metadata,
        )
        epoch, step, restored = load_training_state(
            Path(directory) / "training_checkpoint.pt", model, optimizer, scheduler, None,
            expected_dataset_sha256="readiness",
        )
        results["checkpoint/resume"] = (epoch, step, restored) == (1, 1, metadata)
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--precision", choices=("fp16",), default="fp16")
    args = parser.parse_args(argv)

    checks = OrderedDict((name, False) for name in CHECK_NAMES)
    dataset_results, records = inspect_dataset(args.data)
    checks.update(dataset_results)
    checks["focal loss"] = check_focal_loss()
    checks["existing MCTS tests"] = check_existing_mcts_tests()
    # Avoid a multi-GB model load when the final artifact is absent or invalid.
    if records and all(dataset_results.values()):
        try:
            checks.update(model_checks(args.model, args.precision))
        except Exception as exc:
            print(f"model readiness error: {exc}", file=sys.stderr)

    ready = all(checks.values())
    print(f"ROUTER TRAINING READY: {'YES' if ready else 'NO'}")
    for name, passed in checks.items():
        print(f"{name}: {'PASS' if passed else 'FAIL'}")
    return 0 if ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
