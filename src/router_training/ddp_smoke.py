"""Verify a two-rank CUDA/NCCL torchrun launch without loading the LLM."""

from __future__ import annotations

import os

import torch
import torch.distributed as dist


def main() -> int:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size != 2:
        raise RuntimeError(f"DDP smoke requires exactly 2 ranks, found {world_size}")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
        raise RuntimeError(f"DDP smoke requires exactly 2 CUDA GPUs, found {torch.cuda.device_count()}")
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    try:
        value = torch.tensor(float(dist.get_rank() + 1), device=local_rank)
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
        if value.item() != 3.0:
            raise RuntimeError(f"DDP all-reduce returned {value.item()}, expected 3.0")
        dist.barrier()
        if dist.get_rank() == 0:
            print("ddp_smoke=PASS world_size=2 backend=nccl")
    finally:
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
