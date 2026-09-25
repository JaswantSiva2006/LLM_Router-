"""Build the canonical GSM8K MCTS supervision dataset after replay verification."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import statistics
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

from src.data.gsm8k import PROMPT_TEMPLATE
from src.grading.gsm8k_verifier import VerificationLabel, verify_gsm8k_answer
from src.model.routed_qwen import path_to_labels, validate_path
from src.run_generation import append_jsonl_durable, atomic_write_json, read_valid_jsonl, utc_now

ReplayFunction = Callable[[dict[str, Any]], str]


@dataclass(frozen=True)
class SourceCandidate:
    record: dict[str, Any]
    selected: dict[str, Any]
    manifest: dict[str, Any]
    run_directory: str

    @property
    def question_id(self) -> str:
        return str(self.record["question_id"])

    @property
    def path(self) -> list[int]:
        return [int(value) for value in self.selected["execution_path"]]


def discover_run_directories(roots: Iterable[Path]) -> list[Path]:
    found: set[Path] = set()
    for root in roots:
        root = root.resolve()
        if (root / "run_manifest.json").is_file():
            found.add(root)
        if root.is_dir():
            found.update(path.parent.resolve() for path in root.rglob("run_manifest.json"))
    return sorted(found, key=str)


def _load_run(run_directory: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = json.loads((run_directory / "run_manifest.json").read_text("utf-8"))
    merged = run_directory / "merged.jsonl"
    if merged.exists():
        return manifest, read_valid_jsonl(merged)
    by_id: dict[str, dict[str, Any]] = {}
    for path in sorted(run_directory.glob("worker_*.jsonl")):
        for record in read_valid_jsonl(path):
            question_id = str(record["question_id"])
            if question_id in by_id and by_id[question_id] != record:
                raise ValueError(f"conflicting records for {question_id} in {run_directory}")
            by_id[question_id] = record
    return manifest, sorted(by_id.values(), key=lambda item: str(item["question_id"]))


def validate_source_candidate(
    record: dict[str, Any], manifest: dict[str, Any], run_directory: Path
) -> tuple[SourceCandidate | None, str | None]:
    question_id = str(record.get("question_id", ""))
    if not question_id.startswith("openai/gsm8k:main:train:"):
        return None, "not_gsm8k_main_train"
    selected = record.get("selected_pi_star")
    if not isinstance(selected, dict):
        return None, "no_pi_star"
    if selected.get("verifier_status") != VerificationLabel.VERIFIED_CORRECT.value:
        return None, "pi_star_not_verified_correct"
    if selected.get("training_disposition") == "UNLABELED":
        return None, "pi_star_unlabeled"
    try:
        path = validate_path(selected.get("execution_path"), 28)
    except (TypeError, ValueError):
        return None, "illegal_pi_star_path"
    expected_actions = [int(label) for label in path_to_labels(path, 28)]
    if selected.get("action_vector") != expected_actions:
        return None, "action_vector_path_mismatch"
    baseline = record.get("baseline", {})
    baseline_status = baseline.get("verifier_status")
    if baseline_status == VerificationLabel.VERIFIED_CORRECT.value:
        if len(path) >= 28:
            return None, "baseline_correct_pi_star_not_shorter"
    elif baseline_status == VerificationLabel.VERIFIED_INCORRECT.value:
        pass
    else:
        return None, "baseline_not_decisive"
    if not isinstance(record.get("question"), str) or not record["question"].strip():
        return None, "missing_question"
    if not isinstance(record.get("gold_answer"), str):
        return None, "missing_gold_answer"
    if not manifest.get("model") or not manifest.get("model_revision"):
        return None, "missing_model_provenance"
    return SourceCandidate(record, selected, manifest, str(run_directory)), None


def collect_candidates(roots: Iterable[Path]) -> tuple[list[SourceCandidate], Counter, list[Path]]:
    run_directories = discover_run_directories(roots)
    if not run_directories:
        raise ValueError("no run_manifest.json files found under --runs")
    rejection_reasons: Counter = Counter()
    by_id: dict[str, SourceCandidate] = {}
    identity: dict[str, tuple[str, str]] = {}
    for run_directory in run_directories:
        manifest, records = _load_run(run_directory)
        for record in records:
            candidate, reason = validate_source_candidate(record, manifest, run_directory)
            if candidate is None:
                rejection_reasons[reason] += 1
                continue
            key = candidate.question_id
            content = (str(record["question"]), str(record["gold_answer"]))
            if key in identity and identity[key] != content:
                raise ValueError(f"conflicting question/gold content for duplicate ID {key}")
            identity[key] = content
            current = by_id.get(key)
            rank = (len(candidate.path), candidate.selected.get("num_repeated_layers", 0), tuple(candidate.path), candidate.run_directory)
            if current is None:
                by_id[key] = candidate
            else:
                current_rank = (len(current.path), current.selected.get("num_repeated_layers", 0), tuple(current.path), current.run_directory)
                if rank < current_rank:
                    by_id[key] = candidate
    candidates = sorted(by_id.values(), key=lambda item: item.question_id)
    return candidates, rejection_reasons, run_directories


def _replay_key(candidate: SourceCandidate, replay_identity: str) -> str:
    payload = json.dumps({
        "id": candidate.question_id,
        "path": candidate.path,
        "model": candidate.manifest["model"],
        "revision": candidate.manifest["model_revision"],
        "replay_identity": replay_identity,
    }, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _load_replay_cache(path: Path) -> dict[str, dict[str, Any]]:
    cache = {}
    for entry in read_valid_jsonl(path, repair=True):
        key = str(entry.get("replay_key", ""))
        if key and key not in cache:
            cache[key] = entry
    return cache


def _atomic_write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        for row in rows:
            handle.write((json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8"))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_write_gzip_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            for row in rows:
                compressed.write((json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8"))
        raw.flush()
        os.fsync(raw.fileno())
    os.replace(temporary, path)


def _distribution(configs: list[list[int]]) -> dict[str, Any]:
    counts = Counter(value for config in configs for value in config)
    total = sum(counts.values())
    return {
        "counts": {"skip": counts[0], "execute": counts[1], "repeat": counts[2]},
        "fractions": {
            "skip": counts[0] / total if total else 0.0,
            "execute": counts[1] / total if total else 0.0,
            "repeat": counts[2] / total if total else 0.0,
        },
    }


def finalize_dataset(
    *,
    run_roots: list[Path],
    target: int,
    output: Path,
    replay: ReplayFunction,
    replay_identity: str = "greedy_no_cache_max_new_tokens_24_v1",
) -> dict[str, Any]:
    if target <= 0:
        raise ValueError("target must be positive")
    candidates, source_rejections, run_directories = collect_candidates(run_roots)
    if len(candidates) < target:
        raise RuntimeError(
            f"only {len(candidates)} unique eligible samples exist; target is {target}. "
            "Run a second search pass; verification will not be weakened."
        )
    model_pairs = {(candidate.manifest["model"], candidate.manifest["model_revision"]) for candidate in candidates}
    if len(model_pairs) != 1:
        raise RuntimeError(f"eligible samples use mixed model revisions: {sorted(model_pairs)}")
    replay_path = output.with_name(f"{output.stem}_replay.jsonl")
    quarantine_path = output.with_name(f"{output.stem}_quarantine.jsonl")
    replay_cache = _load_replay_cache(replay_path)
    accepted: list[tuple[SourceCandidate, dict[str, Any]]] = []
    quarantined: list[dict[str, Any]] = []
    seen_questions: set[str] = set()
    for candidate in candidates:
        if len(accepted) == target:
            break
        if candidate.record["question"] in seen_questions:
            quarantined.append({
                "question_id": candidate.question_id,
                "reason": "duplicate_question_text",
                "run_directory": candidate.run_directory,
            })
            continue
        key = _replay_key(candidate, replay_identity)
        replay_entry = replay_cache.get(key)
        if replay_entry is None:
            raw_response = replay({
                "question_id": candidate.question_id,
                "question": candidate.record["question"],
                "gold_answer": candidate.record["gold_answer"],
                "generation_prompt": PROMPT_TEMPLATE.format(question=candidate.record["question"]),
                "execution_path": candidate.path,
                "model": candidate.manifest["model"],
                "model_revision": candidate.manifest["model_revision"],
            })
            review = verify_gsm8k_answer(candidate.record["gold_answer"], raw_response)
            replay_entry = {
                "replay_key": key,
                "question_id": candidate.question_id,
                "execution_path": candidate.path,
                "raw_model_response": raw_response,
                "verifier_status": review.label.value,
                "verifier_reason": review.reason,
                "strict_parser_result": review.strict_parser_result,
                "replayed_at": utc_now(),
            }
            append_jsonl_durable(replay_path, replay_entry)
            replay_cache[key] = replay_entry
        if replay_entry["verifier_status"] != VerificationLabel.VERIFIED_CORRECT.value:
            quarantined.append({
                **replay_entry,
                "reason": "replay_not_verified_correct",
                "source_run_directory": candidate.run_directory,
            })
            continue
        accepted.append((candidate, replay_entry))
        seen_questions.add(candidate.record["question"])
    if len(accepted) < target:
        _atomic_write_jsonl(quarantine_path, quarantined)
        raise RuntimeError(
            f"only {len(accepted)} samples passed independent replay; target is {target}; "
            f"{len(quarantined)} samples quarantined"
        )

    canonical = []
    provenance = []
    preservation = improvement = 0
    for candidate, replay_entry in accepted:
        actions = [int(label) for label in path_to_labels(candidate.path, 28)]
        canonical.append({
            "id": candidate.question_id,
            "question": candidate.record["question"],
            "answer": candidate.record["gold_answer"],
            "optimal_layer_config": actions,
        })
        baseline_status = candidate.record["baseline"]["verifier_status"]
        preservation += int(baseline_status == VerificationLabel.VERIFIED_CORRECT.value)
        improvement += int(baseline_status == VerificationLabel.VERIFIED_INCORRECT.value)
        provenance.append({
            "id": candidate.question_id,
            "source_run_directory": candidate.run_directory,
            "selected_execution_path": candidate.path,
            "baseline": candidate.record["baseline"],
            "selected_raw_output": candidate.selected["raw_model_response"],
            "selected_verifier_status": candidate.selected["verifier_status"],
            "selected_verifier_reason": candidate.selected["verifier_reason"],
            "replay": replay_entry,
            "search_metadata": candidate.record.get("search_metadata", {}),
            "model": candidate.manifest["model"],
            "model_revision": candidate.manifest["model_revision"],
            "software": {
                "torch_version": candidate.manifest.get("torch_version"),
                "transformers_version": candidate.manifest.get("transformers_version"),
                "git_commit": candidate.manifest.get("git_commit"),
            },
        })
    ids = [row["id"] for row in canonical]
    questions = [row["question"] for row in canonical]
    configs = [row["optimal_layer_config"] for row in canonical]
    round_trip_legal = all(
        [int(label) for label in path_to_labels(validate_path(candidate.path, 28), 28)] == config
        for (candidate, _), config in zip(accepted, configs)
    )
    summary = {
        "rows": len(canonical),
        "target": target,
        "unique_ids": len(set(ids)),
        "unique_questions": len(set(questions)),
        "all_config_lengths_28": all(len(config) == 28 for config in configs),
        "labels_subset_0_1_2": all(set(config) <= {0, 1, 2} for config in configs),
        "all_configs_round_trip_to_legal_paths": round_trip_legal,
        "test_split_ids": sum(":test:" in question_id for question_id in ids),
        "uncertain_selected_samples": 0,
        "replay_failures": 0,
        "quarantined_replay_failures": sum(
            item.get("reason") == "replay_not_verified_correct" for item in quarantined
        ),
        "skip_execute_repeat_distribution": _distribution(configs),
        "baseline_improvement_count": improvement,
        "baseline_preservation_count": preservation,
        "mean_selected_path_length": statistics.fmean(len(candidate.path) for candidate, _ in accepted),
        "source_eligible_unique_samples": len(candidates),
        "source_rejection_reasons": dict(sorted(source_rejections.items())),
        "source_run_directories": [str(path) for path in run_directories],
        "model": next(iter(model_pairs))[0],
        "model_revision": next(iter(model_pairs))[1],
        "replay_identity": replay_identity,
        "created_at": utc_now(),
    }
    invariants = (
        summary["rows"] == target
        and summary["unique_ids"] == target
        and summary["unique_questions"] == target
        and summary["all_config_lengths_28"]
        and summary["labels_subset_0_1_2"]
        and summary["all_configs_round_trip_to_legal_paths"]
        and summary["test_split_ids"] == 0
        and summary["uncertain_selected_samples"] == 0
        and summary["replay_failures"] == 0
    )
    if not invariants:
        raise RuntimeError(f"final dataset invariant failure: {summary}")
    provenance_path = output.with_name(f"{output.stem}_provenance.jsonl.gz")
    summary_path = output.with_name(f"{output.stem}_summary.json")
    _atomic_write_jsonl(output, canonical)
    _atomic_write_gzip_jsonl(provenance_path, provenance)
    atomic_write_json(summary_path, summary)
    _atomic_write_jsonl(quarantine_path, quarantined)
    return summary


def _build_replay(model_name: str, revision: str, max_new_tokens: int) -> ReplayFunction:
    import torch
    from transformers import AutoTokenizer, Qwen2ForCausalLM

    from src.model.routed_qwen import enable_qwen2_routing, generate_with_route

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for independent replay")
    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
    model = Qwen2ForCausalLM.from_pretrained(model_name, revision=revision, dtype=torch.float16).to("cuda:0")
    if model.config.num_hidden_layers != 28:
        raise ValueError(f"expected 28 model layers, got {model.config.num_hidden_layers}")
    enable_qwen2_routing(model)

    def replay(item: dict[str, Any]) -> str:
        return generate_with_route(
            model, tokenizer, item["generation_prompt"], item["execution_path"],
            max_new_tokens=max_new_tokens,
        )

    return replay


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--target", type=int, default=4000)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    args = parser.parse_args(argv)
    candidates, _rejections, _directories = collect_candidates(args.runs)
    if len(candidates) < args.target:
        raise SystemExit(
            f"FAIL: only {len(candidates)} eligible unique samples exist; target is {args.target}"
        )
    model_pairs = {(item.manifest["model"], item.manifest["model_revision"]) for item in candidates}
    if len(model_pairs) != 1:
        raise SystemExit(f"FAIL: mixed model revisions: {sorted(model_pairs)}")
    model_name, revision = next(iter(model_pairs))
    replay = _build_replay(model_name, revision, args.max_new_tokens)
    summary = finalize_dataset(
        run_roots=args.runs,
        target=args.target,
        output=args.output,
        replay=replay,
        replay_identity=f"greedy_no_cache_max_new_tokens_{args.max_new_tokens}_v1",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
