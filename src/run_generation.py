"""Restartable one-process-per-GPU GSM8K MCTS generation runner.

This module deliberately avoids importing torch/transformers in the parent
process. Each spawned worker first restricts CUDA visibility to one physical
GPU, then imports and loads its own model copy.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Durably replace a JSON checkpoint in the same directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def read_valid_jsonl(path: Path, *, repair: bool = False) -> list[dict[str, Any]]:
    """Read complete JSONL records; optionally truncate a crash-torn tail."""
    if not path.exists():
        return []
    data = path.read_bytes()
    valid_end = 0
    records: list[dict[str, Any]] = []
    offset = 0
    for raw_line in data.splitlines(keepends=True):
        offset += len(raw_line)
        if not raw_line.endswith((b"\n", b"\r")):
            break
        try:
            item = json.loads(raw_line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            break
        if not isinstance(item, dict) or "question_id" not in item:
            break
        records.append(item)
        valid_end = offset
    if valid_end != len(data):
        if not repair:
            raise ValueError(f"invalid or incomplete JSONL tail in {path}")
        with path.open("r+b") as handle:
            handle.truncate(valid_end)
            handle.flush()
            os.fsync(handle.fileno())
    return records


def append_jsonl_durable(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
    with path.open("ab") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())


def partition_indices(start: int, count: int, worker_count: int) -> list[list[int]]:
    if start < 0 or count < 0 or worker_count <= 0:
        raise ValueError("start/count must be non-negative and worker_count positive")
    partitions = [[] for _ in range(worker_count)]
    for offset, index in enumerate(range(start, start + count)):
        partitions[offset % worker_count].append(index)
    return partitions


def load_requested_indices(path: Path) -> list[int]:
    indices = []
    for line_number, raw in enumerate(path.read_text("utf-8").splitlines(), 1):
        value = raw.strip()
        if not value or value.startswith("#"):
            continue
        suffix = value.rsplit(":", 1)[-1]
        try:
            index = int(suffix)
        except ValueError as exc:
            raise ValueError(f"invalid ID/index at {path}:{line_number}: {value!r}") from exc
        if index < 0:
            raise ValueError(f"negative index at {path}:{line_number}")
        indices.append(index)
    if not indices:
        raise ValueError("IDs file contains no IDs")
    if len(indices) != len(set(indices)):
        raise ValueError("IDs file contains duplicate IDs")
    return sorted(indices)


def _record_index(record: dict[str, Any]) -> int:
    question_id = str(record["question_id"])
    try:
        return int(question_id.rsplit(":", 1)[1])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"question_id does not end in a numeric index: {question_id!r}") from exc


def merge_worker_outputs(run_directory: Path, worker_count: int) -> list[dict[str, Any]]:
    """Validate, deduplicate, sort, and atomically write merged.jsonl."""
    by_id: dict[str, dict[str, Any]] = {}
    for worker_id in range(worker_count):
        for record in read_valid_jsonl(run_directory / f"worker_{worker_id}.jsonl", repair=True):
            question_id = str(record["question_id"])
            if question_id in by_id and by_id[question_id] != record:
                raise ValueError(f"conflicting records for {question_id}")
            by_id[question_id] = record
    merged = sorted(by_id.values(), key=lambda item: (_record_index(item), str(item["question_id"])))
    target = run_directory / "merged.jsonl"
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        for record in merged:
            handle.write((json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8"))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, target)
    return merged


def git_metadata() -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = bool(subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, check=True
        ).stdout.strip())
        return {"git_commit": commit, "git_dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"git_commit": None, "git_dirty": None}


def package_version(name: str) -> str | None:
    from importlib.metadata import PackageNotFoundError, version
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def resolve_model_revision(model: str, revision: str | None) -> str:
    """Resolve a Hub ref to an immutable SHA without downloading model weights."""
    from huggingface_hub import model_info
    return model_info(model, revision=revision or "main").sha


@dataclass(frozen=True)
class WorkerSpec:
    worker_id: int
    physical_gpu_id: int
    indices: tuple[int, ...]
    run_directory: str
    model: str
    model_revision: str
    dataset: str
    split: str
    num_simulations: int
    early_stop: str
    seed: int
    max_new_tokens: int


def _checkpoint_path(spec: WorkerSpec) -> Path:
    return Path(spec.run_directory) / f"worker_{spec.worker_id}.checkpoint.json"


def _output_path(spec: WorkerSpec) -> Path:
    return Path(spec.run_directory) / f"worker_{spec.worker_id}.jsonl"


def _save_worker_checkpoint(
    spec: WorkerSpec,
    *,
    status: str,
    attempted_ids: set[str],
    completed_ids: set[str],
    model_inferences: int,
    runtime_seconds: float,
    environment: dict[str, Any] | None = None,
    error: str | None = None,
) -> None:
    atomic_write_json(_checkpoint_path(spec), {
        "worker_id": spec.worker_id,
        "physical_gpu_id": spec.physical_gpu_id,
        "status": status,
        "attempted_ids": sorted(attempted_ids),
        "completed_ids": sorted(completed_ids),
        "questions_attempted": len(attempted_ids),
        "questions_completed": len(completed_ids),
        "model_inferences": model_inferences,
        "runtime_seconds": runtime_seconds,
        "environment": environment or {},
        "error": error,
        "updated_at": utc_now(),
    })


def generation_worker(spec_dict: dict[str, Any], stop_event: Any) -> None:
    """Spawn target. Heavy imports occur only after exclusive GPU binding."""
    spec = WorkerSpec(**spec_dict)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(spec.physical_gpu_id)
    output_path = _output_path(spec)
    existing = read_valid_jsonl(output_path, repair=True)
    completed_ids = {str(record["question_id"]) for record in existing}
    checkpoint_path = _checkpoint_path(spec)
    previous = json.loads(checkpoint_path.read_text("utf-8")) if checkpoint_path.exists() else {}
    attempted_ids = set(previous.get("attempted_ids", [])) | completed_ids
    completed_inferences = sum(
        int(record.get("search_metadata", {}).get("unique_paths_evaluated", 0)) for record in existing
    )
    # Preserve inference spent on a previously interrupted, incomplete question.
    model_inferences = max(int(previous.get("model_inferences", 0)), completed_inferences)
    prior_runtime = float(previous.get("runtime_seconds", 0.0))
    session_start = time.monotonic()
    environment: dict[str, Any] = {}

    def elapsed() -> float:
        return prior_runtime + (time.monotonic() - session_start)

    def handle_interrupt(_signum, _frame):
        stop_event.set()

    signal.signal(signal.SIGINT, handle_interrupt)

    try:
        import torch
        from datasets import load_dataset
        from transformers import AutoTokenizer, Qwen2ForCausalLM, __version__ as transformers_version

        from src.data.gsm8k import adapt_record
        from src.model.routed_qwen import enable_qwen2_routing, generate_with_route
        from src.search.mcts import MCTSConfig, MCTSSearch

        if not torch.cuda.is_available():
            raise RuntimeError(f"worker {spec.worker_id} cannot access assigned GPU {spec.physical_gpu_id}")
        torch.cuda.set_device(0)  # only the assigned physical GPU is visible
        environment = {
            "physical_gpu_id": spec.physical_gpu_id,
            "visible_device": "cuda:0",
            "gpu_name": torch.cuda.get_device_name(0),
            "torch_version": torch.__version__,
            "transformers_version": transformers_version,
            "cuda_version": torch.version.cuda,
            "model": spec.model,
            "model_revision": spec.model_revision,
        }
        torch.manual_seed(spec.seed + spec.worker_id)
        torch.cuda.manual_seed_all(spec.seed + spec.worker_id)
        tokenizer = AutoTokenizer.from_pretrained(spec.model, revision=spec.model_revision)
        model = Qwen2ForCausalLM.from_pretrained(
            spec.model, revision=spec.model_revision, dtype=torch.float16
        ).to("cuda:0")
        if model.config.num_hidden_layers != 28:
            raise ValueError(f"expected 28 model layers, got {model.config.num_hidden_layers}")
        enable_qwen2_routing(model)
        dataset = load_dataset(spec.dataset, "main", split=spec.split)
        _save_worker_checkpoint(
            spec, status="running", attempted_ids=attempted_ids, completed_ids=completed_ids,
            model_inferences=model_inferences, runtime_seconds=elapsed(), environment=environment,
        )
        for index in spec.indices:
            if stop_event.is_set():
                break
            record = adapt_record(dataset[index], index, spec.split)
            if record.dataset_id in completed_ids:
                continue
            attempted_ids.add(record.dataset_id)
            _save_worker_checkpoint(
                spec, status="running", attempted_ids=attempted_ids, completed_ids=completed_ids,
                model_inferences=model_inferences, runtime_seconds=elapsed(), environment=environment,
            )
            question_seed = spec.seed + index
            search = MCTSSearch(MCTSConfig(
                num_simulations=spec.num_simulations,
                seed=question_seed,
                early_stop=spec.early_stop,
            ))

            def evaluate(path: list[int]) -> str:
                nonlocal model_inferences
                response = generate_with_route(
                    model, tokenizer, record.generation_prompt, path,
                    max_new_tokens=spec.max_new_tokens,
                )
                model_inferences += 1
                return response

            result = search.run(
                question_id=record.dataset_id,
                question=record.question,
                gold_answer=record.gold_answer,
                evaluate_route=evaluate,
            )
            payload = result.to_dict()
            payload["dataset_index"] = index
            payload["worker_id"] = spec.worker_id
            payload["physical_gpu_id"] = spec.physical_gpu_id
            payload["question_seed"] = question_seed
            append_jsonl_durable(output_path, payload)
            completed_ids.add(record.dataset_id)
            _save_worker_checkpoint(
                spec, status="running", attempted_ids=attempted_ids, completed_ids=completed_ids,
                model_inferences=model_inferences, runtime_seconds=elapsed(), environment=environment,
            )
        final_status = "interrupted" if stop_event.is_set() else "complete"
        _save_worker_checkpoint(
            spec, status=final_status, attempted_ids=attempted_ids, completed_ids=completed_ids,
            model_inferences=model_inferences, runtime_seconds=elapsed(), environment=environment,
        )
    except BaseException as exc:
        _save_worker_checkpoint(
            spec, status="failed", attempted_ids=attempted_ids, completed_ids=completed_ids,
            model_inferences=model_inferences, runtime_seconds=elapsed(), environment=environment,
            error=f"{type(exc).__name__}: {exc}",
        )
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--revision", help="model revision/ref; resolved to an immutable Hub SHA")
    parser.add_argument("--dataset", default="openai/gsm8k", choices=["openai/gsm8k"])
    parser.add_argument("--split", choices=["train"], default="train")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int)
    parser.add_argument(
        "--ids-file", type=Path,
        help="sparse retry list containing GSM8K IDs or integer train indices",
    )
    parser.add_argument("--num-simulations", type=int, default=50)
    parser.add_argument("--early-stop", choices=["none", "paper"], default="none")
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.start < 0 or args.num_simulations <= 0:
        raise SystemExit("--start must be >=0 and --num-simulations must be >0")
    if args.ids_file is not None and args.count is not None:
        raise SystemExit("use either --ids-file or --count, not both")
    if args.ids_file is None and (args.count is None or args.count <= 0):
        raise SystemExit("--count must be >0 when --ids-file is not provided")
    if args.ids_file is not None:
        try:
            requested_indices = load_requested_indices(args.ids_file)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
    else:
        requested_indices = list(range(args.start, args.start + args.count))
    gpu_ids = [int(item.strip()) for item in args.gpus.split(",") if item.strip()]
    if not gpu_ids or len(set(gpu_ids)) != len(gpu_ids):
        raise SystemExit("--gpus must contain unique comma-separated physical GPU IDs")
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / "run_manifest.json"
    existing_manifest = json.loads(manifest_path.read_text("utf-8")) if manifest_path.exists() else None
    if existing_manifest and args.revision is None and existing_manifest.get("model") == args.model:
        revision = str(existing_manifest["model_revision"])
    else:
        revision = resolve_model_revision(args.model, args.revision)
    partitions = [[] for _ in gpu_ids]
    for offset, index in enumerate(requested_indices):
        partitions[offset % len(gpu_ids)].append(index)
    immutable = {
        "model": args.model,
        "model_revision": revision,
        "dataset": args.dataset,
        "dataset_config": "main",
        "split": args.split,
        "start": args.start if args.ids_file is None else None,
        "count": len(requested_indices),
        "requested_indices": requested_indices,
        "num_simulations": args.num_simulations,
        "early_stop": args.early_stop,
        "gpus": gpu_ids,
        "seed": args.seed,
        "max_new_tokens": args.max_new_tokens,
    }
    if existing_manifest:
        old = {key: existing_manifest.get(key) for key in immutable}
        if old != immutable:
            raise SystemExit("run directory manifest does not match requested configuration")
    manifest = {
        **immutable,
        **git_metadata(),
        "python_version": sys.version,
        "torch_version": package_version("torch"),
        "transformers_version": package_version("transformers"),
        "created_at": existing_manifest.get("created_at") if existing_manifest else utc_now(),
        "status": "running",
        "worker_count": len(gpu_ids),
        "worker_partitions": partitions,
        "run_started_at": utc_now(),
    }
    atomic_write_json(manifest_path, manifest)
    context = mp.get_context("spawn")
    stop_event = context.Event()
    specs = [WorkerSpec(
        worker_id=worker_id,
        physical_gpu_id=gpu_id,
        indices=tuple(partitions[worker_id]),
        run_directory=str(args.output.resolve()),
        model=args.model,
        model_revision=revision,
        dataset=args.dataset,
        split=args.split,
        num_simulations=args.num_simulations,
        early_stop=args.early_stop,
        seed=args.seed,
        max_new_tokens=args.max_new_tokens,
    ) for worker_id, gpu_id in enumerate(gpu_ids)]
    processes = [context.Process(
        target=generation_worker, args=(asdict(spec), stop_event), name=f"gpu-worker-{spec.worker_id}"
    ) for spec in specs]
    wall_start = time.monotonic()
    prior_wall_seconds = float(existing_manifest.get("wall_clock_seconds", 0.0)) if existing_manifest else 0.0
    interrupted = False
    for process in processes:
        process.start()
    try:
        while any(process.is_alive() for process in processes):
            for process in processes:
                process.join(timeout=0.5)
    except KeyboardInterrupt:
        interrupted = True
        stop_event.set()
        print("Interruption received; workers will checkpoint after the active question.", file=sys.stderr)
        for process in processes:
            process.join(timeout=30)
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join()
    merged = merge_worker_outputs(args.output, len(processes))
    wall_seconds = prior_wall_seconds + (time.monotonic() - wall_start)
    checkpoints = []
    for spec in specs:
        path = _checkpoint_path(spec)
        if path.exists():
            checkpoints.append(json.loads(path.read_text("utf-8")))
    failures = [process.exitcode for process in processes if process.exitcode not in (0, None)]
    manifest.update({
        "status": "interrupted" if interrupted else ("failed" if failures else "complete"),
        "run_finished_at": utc_now(),
        "wall_clock_seconds": wall_seconds,
        "questions_completed": len(merged),
        "model_inferences": sum(int(item.get("model_inferences", 0)) for item in checkpoints),
        "worker_checkpoints": checkpoints,
        "worker_exit_codes": [process.exitcode for process in processes],
    })
    atomic_write_json(manifest_path, manifest)
    print(f"Merged {len(merged)} completed questions into {args.output / 'merged.jsonl'}")
    print(f"Wall time: {wall_seconds:.1f}s; model inferences: {manifest['model_inferences']}")
    return 130 if interrupted else (1 if failures else 0)


if __name__ == "__main__":
    raise SystemExit(main())
