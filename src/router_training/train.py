"""Single-GPU and DDP training entry point for the Dr.LLM Qwen routers."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch
import torch.distributed as dist
from safetensors.torch import save_file
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, DistributedSampler, Subset
from transformers import AutoTokenizer, Qwen2ForCausalLM, get_cosine_schedule_with_warmup
from transformers import __version__ as transformers_version

from src.router_training.dataset import RouterTrainingRecord, load_router_dataset
from src.router_training.loss import focal_loss
from src.router_training.model import NUM_LAYERS, TeacherForcedRouterQwen
from src.router_training.stats import BETA, count_labels, effective_number_weights


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 25
    microbatch: int = 1
    gradient_accumulation: int = 16
    learning_rate: float = 1e-3
    weight_decay: float = 0.01
    warmup_steps: int = 500
    gamma: float = 2.0
    beta: float = BETA
    seed: int = 42
    precision: str = "fp16"


def global_effective_batch_size(microbatch: int, gradient_accumulation: int, world_size: int) -> int:
    if min(microbatch, gradient_accumulation, world_size) <= 0:
        raise ValueError("microbatch, gradient accumulation, and world size must be positive")
    return microbatch * gradient_accumulation * world_size


def deterministic_split_indices(size: int, val_fraction: float, seed: int) -> tuple[list[int], list[int]]:
    if size <= 0:
        raise ValueError("dataset must not be empty")
    if not 0 <= val_fraction < 1:
        raise ValueError("val_fraction must be in [0, 1)")
    validation_size = int(size * val_fraction)
    generator = torch.Generator().manual_seed(seed)
    shuffled = torch.randperm(size, generator=generator).tolist()
    return shuffled[validation_size:], shuffled[:validation_size]


def make_distributed_sampler(
    dataset: Dataset[Any], *, rank: int, world_size: int, shuffle: bool, seed: int
) -> DistributedSampler:
    return DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=shuffle, seed=seed, drop_last=False
    )


def build_optimizer(model: nn.Module, learning_rate: float = 1e-3, weight_decay: float = 0.01) -> AdamW:
    named = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    if not named:
        raise ValueError("model has no trainable router parameters")
    invalid = [name for name, _ in named if not (name.startswith("routers.") or ".routers." in name)]
    if invalid:
        raise RuntimeError(f"non-router parameters require gradients: {invalid[:3]}")
    frozen_violations = [
        name for name, parameter in model.named_parameters()
        if not (name.startswith("routers.") or ".routers." in name) and parameter.requires_grad
    ]
    if frozen_violations:
        raise RuntimeError("all Qwen parameters must be frozen")
    return AdamW([parameter for _, parameter in named], lr=learning_rate, weight_decay=weight_decay)


class TokenizedRouterDataset(Dataset[dict[str, Any]]):
    def __init__(self, records: Sequence[RouterTrainingRecord]) -> None:
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        return {"text": record.router_input, "labels": record.optimal_layer_config}


def make_collator(tokenizer: Any):
    def collate(items: list[dict[str, Any]]) -> dict[str, Tensor]:
        # Only canonical router prompts enter tokenization; provenance answers are absent.
        encoded = tokenizer(
            [item["text"] for item in items], padding=True, return_tensors="pt"
        )
        encoded["router_labels"] = torch.tensor(
            [item["labels"] for item in items], dtype=torch.long
        )
        return encoded
    return collate


class RouterMetrics:
    """Additive sufficient statistics, suitable for exact DDP all-reduction."""

    def __init__(self, device: torch.device | str = "cpu") -> None:
        self.device = torch.device(device)
        self.loss_sum = torch.zeros((), dtype=torch.float64, device=self.device)
        self.sample_count = torch.zeros((), dtype=torch.long, device=self.device)
        self.confusion = torch.zeros((3, 3), dtype=torch.long, device=self.device)
        self.per_layer_correct = torch.zeros(NUM_LAYERS, dtype=torch.long, device=self.device)
        self.path_correct = torch.zeros((), dtype=torch.long, device=self.device)
        self.predicted_executed = torch.zeros((), dtype=torch.float64, device=self.device)
        self.target_executed = torch.zeros((), dtype=torch.float64, device=self.device)

    @torch.no_grad()
    def update(self, loss: Tensor | float, logits: Tensor, targets: Tensor) -> None:
        predictions = logits.argmax(dim=-1)
        batch_size = targets.shape[0]
        self.loss_sum += torch.as_tensor(loss, dtype=torch.float64, device=self.device) * batch_size
        self.sample_count += batch_size
        flat = targets.reshape(-1) * 3 + predictions.reshape(-1)
        self.confusion += torch.bincount(flat, minlength=9).reshape(3, 3).to(self.device)
        self.per_layer_correct += (predictions == targets).sum(dim=0).to(self.device)
        self.path_correct += (predictions == targets).all(dim=1).sum().to(self.device)
        self.predicted_executed += predictions.sum(dtype=torch.float64).to(self.device)
        self.target_executed += targets.sum(dtype=torch.float64).to(self.device)

    def reduce(self) -> None:
        if dist.is_available() and dist.is_initialized():
            for tensor in self._tensors():
                dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

    def _tensors(self) -> list[Tensor]:
        return [
            self.loss_sum, self.sample_count, self.confusion, self.per_layer_correct,
            self.path_correct, self.predicted_executed, self.target_executed,
        ]

    def compute(self) -> dict[str, Any]:
        samples = int(self.sample_count.item())
        if samples == 0:
            raise ValueError("cannot compute metrics without samples")
        confusion = self.confusion.cpu()
        total_positions = int(confusion.sum().item())
        f1 = []
        for class_id in range(3):
            tp = confusion[class_id, class_id].item()
            fp = confusion[:, class_id].sum().item() - tp
            fn = confusion[class_id, :].sum().item() - tp
            denominator = 2 * tp + fp + fn
            f1.append(0.0 if denominator == 0 else 2 * tp / denominator)
        predicted_counts = confusion.sum(dim=0)
        target_counts = confusion.sum(dim=1)
        return {
            "router_loss": self.loss_sum.item() / samples,
            "overall_layer_accuracy": confusion.diag().sum().item() / total_positions,
            "macro_f1": sum(f1) / 3,
            "f1_skip": f1[0], "f1_execute": f1[1], "f1_repeat": f1[2],
            "predicted_class_proportions": (predicted_counts / total_positions).tolist(),
            "target_class_proportions": (target_counts / total_positions).tolist(),
            "mean_predicted_executed_layers": self.predicted_executed.item() / samples,
            "mean_target_executed_layers": self.target_executed.item() / samples,
            "exact_path_accuracy": self.path_correct.item() / samples,
            "per_layer_accuracy": (self.per_layer_correct.cpu() / samples).tolist(),
        }


def dataset_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def router_state_dict(model: nn.Module) -> dict[str, Tensor]:
    unwrapped = model.module if isinstance(model, DistributedDataParallel) else model
    return {key: value.detach().cpu() for key, value in unwrapped.routers.state_dict().items()}


def save_training_state(
    output_dir: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Any,
    *,
    epoch: int,
    global_step: int,
    metadata: dict[str, Any],
    rank: int = 0,
) -> None:
    if rank != 0:
        return
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    routers = router_state_dict(model)
    save_file(routers, destination / "routers_only.safetensors")
    torch.save(
        {
            "routers": routers,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict() if scaler is not None else None,
            "epoch": epoch,
            "global_step": global_step,
            "metadata": metadata,
            "python_rng_state": random.getstate(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
        destination / "training_checkpoint.pt",
    )


def load_training_state(
    checkpoint: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    scaler: Any,
    *,
    expected_dataset_sha256: str | None = None,
) -> tuple[int, int, dict[str, Any]]:
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    metadata = state["metadata"]
    if expected_dataset_sha256 and metadata["dataset_sha256"] != expected_dataset_sha256:
        raise ValueError("checkpoint dataset SHA256 does not match the current dataset")
    unwrapped = model.module if isinstance(model, DistributedDataParallel) else model
    unwrapped.routers.load_state_dict(state["routers"])
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    if scaler is not None and state["scaler"] is not None:
        scaler.load_state_dict(state["scaler"])
    random.setstate(state["python_rng_state"])
    torch.set_rng_state(state["torch_rng_state"])
    if torch.cuda.is_available() and state["cuda_rng_state"] is not None:
        torch.cuda.set_rng_state_all(state["cuda_rng_state"])
    return int(state["epoch"]), int(state["global_step"]), metadata


def _optimizer_update(
    optimizer: torch.optim.Optimizer, scheduler: Any, scaler: Any, model: nn.Module
) -> None:
    if scaler is not None:
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    alpha: Tensor,
    gamma: float,
    device: torch.device,
    precision: str,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any = None,
    scaler: Any = None,
    accumulation: int = 1,
) -> tuple[dict[str, Any], int]:
    training = optimizer is not None
    model.train(training)
    metrics = RouterMetrics(device)
    updates = 0
    if training:
        optimizer.zero_grad(set_to_none=True)
    for batch_index, batch in enumerate(loader):
        batch = {key: value.to(device) for key, value in batch.items()}
        boundary = (batch_index + 1) % accumulation == 0 or batch_index + 1 == len(loader)
        autocast = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if precision == "fp16" else
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if precision == "bf16" else contextlib.nullcontext()
        )
        grad_context = contextlib.nullcontext() if training else torch.no_grad()
        sync_context = (
            model.no_sync()
            if training and isinstance(model, DistributedDataParallel) and not boundary
            else contextlib.nullcontext()
        )
        with sync_context, grad_context, autocast:
            output = model(**batch, use_cache=False)
            loss = focal_loss(output.router_logits, batch["router_labels"], alpha, gamma)
            scaled_loss = loss / accumulation
            if training:
                if scaler is not None:
                    scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()
        metrics.update(loss.detach(), output.router_logits.detach(), batch["router_labels"])
        if training:
            if boundary:
                _optimizer_update(optimizer, scheduler, scaler, model)
                updates += 1
    metrics.reduce()
    return metrics.compute(), updates


def _distributed_context() -> tuple[int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        dist.init_process_group(backend="nccl")
    return rank, world_size, local_rank


def _validate_precision(precision: str) -> None:
    if precision == "fp16" and not torch.cuda.is_available():
        raise RuntimeError("--precision fp16 requires CUDA")
    if precision == "bf16" and (
        not torch.cuda.is_available() or not torch.cuda.is_bf16_supported()
    ):
        raise RuntimeError("--precision bf16 is not supported by this CUDA device")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--model-revision", default="main")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--microbatch", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=None)
    parser.add_argument("--precision", choices=("fp16", "bf16", "fp32"), default="fp16")
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--output", type=Path, default=Path("runs/router_train"))
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--allow-small-dataset", action="store_true")
    args = parser.parse_args(argv)

    rank, world_size, local_rank = _distributed_context()
    try:
        if rank != 0:
            from transformers.utils import logging as transformers_logging
            transformers_logging.set_verbosity_error()
            transformers_logging.disable_progress_bar()
        _validate_precision(args.precision)
        device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
        if device.type == "cuda":
            torch.cuda.set_device(device)
        accumulation = args.gradient_accumulation or (8 if world_size == 2 else 16)
        config = TrainConfig(
            epochs=args.epochs, microbatch=args.microbatch,
            gradient_accumulation=accumulation, precision=args.precision,
        )
        random.seed(config.seed)
        torch.manual_seed(config.seed + rank)

        records = load_router_dataset(args.data, allow_small_dataset=args.allow_small_dataset)
        train_indices, validation_indices = deterministic_split_indices(
            len(records), args.val_fraction, config.seed
        )
        train_records = [records[index] for index in train_indices]
        validation_records = [records[index] for index in validation_indices]
        class_counts = count_labels(train_records)
        class_weights = effective_number_weights(class_counts, beta=config.beta)
        alpha = torch.tensor(class_weights, dtype=torch.float32, device=device)

        dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.precision]
        tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.model_revision)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        base = Qwen2ForCausalLM.from_pretrained(
            args.model, revision=args.model_revision, dtype=dtype
        ).to(device)
        model: nn.Module = TeacherForcedRouterQwen(base).to(device)
        optimizer = build_optimizer(model, config.learning_rate, config.weight_decay)

        train_dataset = TokenizedRouterDataset(train_records)
        validation_dataset = TokenizedRouterDataset(validation_records)
        train_sampler = make_distributed_sampler(
            train_dataset, rank=rank, world_size=world_size, shuffle=True, seed=config.seed
        ) if world_size > 1 else None
        validation_sampler = make_distributed_sampler(
            validation_dataset, rank=rank, world_size=world_size, shuffle=False, seed=config.seed
        ) if world_size > 1 and validation_records else None
        collator = make_collator(tokenizer)
        train_loader = DataLoader(
            train_dataset, batch_size=config.microbatch, sampler=train_sampler,
            shuffle=train_sampler is None, collate_fn=collator,
        )
        validation_loader = DataLoader(
            validation_dataset, batch_size=config.microbatch, sampler=validation_sampler,
            shuffle=False, collate_fn=collator,
        ) if validation_records else None
        steps_per_epoch = math.ceil(len(train_loader) / config.gradient_accumulation)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, config.warmup_steps, steps_per_epoch * config.epochs
        )
        scaler = torch.amp.GradScaler("cuda", enabled=args.precision == "fp16")
        if world_size > 1:
            model = DistributedDataParallel(
                model, device_ids=[local_rank], output_device=local_rank,
                broadcast_buffers=False, find_unused_parameters=False,
            )

        sha256 = dataset_sha256(args.data)
        metadata = {
            "config": asdict(config), "class_counts": class_counts,
            "class_weights": class_weights, "gamma": config.gamma, "beta": config.beta,
            "dataset_sha256": sha256, "model": args.model,
            "model_revision": args.model_revision, "torch_version": torch.__version__,
            "transformers_version": transformers_version, "cuda_version": torch.version.cuda,
            "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "precision": args.precision, "seed": config.seed, "world_size": world_size,
            "effective_global_batch": global_effective_batch_size(
                config.microbatch, config.gradient_accumulation, world_size
            ),
        }
        start_epoch = global_step = 0
        if args.resume:
            start_epoch, global_step, saved_metadata = load_training_state(
                args.resume, model, optimizer, scheduler, scaler,
                expected_dataset_sha256=sha256,
            )
            if saved_metadata["class_weights"] != class_weights:
                raise ValueError("checkpoint class weights do not match the training split")

        if rank == 0:
            args.output.mkdir(parents=True, exist_ok=True)
            print(json.dumps(metadata, indent=2))
        for epoch in range(start_epoch, config.epochs):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            train_metrics, updates = _run_epoch(
                model, train_loader, alpha, config.gamma, device, config.precision,
                optimizer=optimizer, scheduler=scheduler, scaler=scaler,
                accumulation=config.gradient_accumulation,
            )
            global_step += updates
            result: dict[str, Any] = {"epoch": epoch + 1, "global_step": global_step, "train": train_metrics}
            if validation_loader is not None:
                validation_metrics, _ = _run_epoch(
                    model, validation_loader, alpha, config.gamma, device, config.precision
                )
                result["validation"] = validation_metrics
            if rank == 0:
                print(json.dumps(result))
                with (args.output / "metrics.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(result) + "\n")
            save_training_state(
                args.output, model, optimizer, scheduler, scaler,
                epoch=epoch + 1, global_step=global_step, metadata=metadata, rank=rank,
            )
        return 0
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(main())
