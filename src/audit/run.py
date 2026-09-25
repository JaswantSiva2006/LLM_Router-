"""Audit a completed or interrupted GSM8K MCTS run directory."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from src.grading.numeric import parse_numeric
from src.run_generation import merge_worker_outputs, read_valid_jsonl


def load_run(run_directory: Path) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    manifest_path = run_directory / "run_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"missing {manifest_path}")
    manifest = json.loads(manifest_path.read_text("utf-8"))
    worker_count = int(manifest.get("worker_count", len(manifest.get("gpus", []))))
    merged_path = run_directory / "merged.jsonl"
    records = read_valid_jsonl(merged_path) if merged_path.exists() else merge_worker_outputs(run_directory, worker_count)
    checkpoints = []
    for worker_id in range(worker_count):
        path = run_directory / f"worker_{worker_id}.checkpoint.json"
        if path.exists():
            checkpoints.append(json.loads(path.read_text("utf-8")))
    return manifest, records, checkpoints


def _strict_matches_gold(candidate: dict[str, Any], gold: str) -> bool:
    strict = candidate.get("strict_parser_result")
    strict_value, gold_value = parse_numeric(str(strict)) if strict is not None else None, parse_numeric(gold)
    return strict_value is not None and strict_value == gold_value


def compute_audit(
    manifest: dict[str, Any], records: list[dict[str, Any]], checkpoints: list[dict[str, Any]]
) -> dict[str, Any]:
    attempted_ids = {item for checkpoint in checkpoints for item in checkpoint.get("attempted_ids", [])}
    completed = len(records)
    baseline_correct = baseline_incorrect = pi_found = 0
    wrong_corrected = correct_shorter = 0
    path_lengths: list[int] = []
    layers_saved: list[int] = []
    repeats_with_any = 0
    skip_by_layer = Counter()
    repeat_by_layer = Counter()
    statuses = Counter()
    strict_false_negative = strict_false_positive = 0
    cache_avoided = total_inferences = 0

    for record in records:
        baseline = record["baseline"]
        baseline_is_correct = baseline["verifier_status"] == "VERIFIED_CORRECT"
        baseline_correct += int(baseline_is_correct)
        baseline_incorrect += int(baseline["verifier_status"] == "VERIFIED_INCORRECT")
        selected = record.get("selected_pi_star")
        if selected is not None:
            pi_found += 1
            path_lengths.append(int(selected["path_length"]))
            actions = [int(value) for value in selected["action_vector"]]
            skipped = [index for index, value in enumerate(actions) if value == 0]
            repeated = [index for index, value in enumerate(actions) if value == 2]
            skip_by_layer.update(skipped)
            repeat_by_layer.update(repeated)
            repeats_with_any += int(bool(repeated))
            if baseline_is_correct:
                correct_shorter += 1
                layers_saved.append(28 - int(selected["path_length"]))
            elif baseline["verifier_status"] == "VERIFIED_INCORRECT":
                wrong_corrected += 1
        for candidate in record.get("evaluated_candidates", []):
            status = candidate["verifier_status"]
            statuses[status] += 1
            strict_correct = _strict_matches_gold(candidate, str(record["gold_answer"]))
            conservative_correct = status == "VERIFIED_CORRECT"
            strict_false_negative += int(conservative_correct and not strict_correct)
            strict_false_positive += int(strict_correct and not conservative_correct)
        metadata = record.get("search_metadata", {})
        cache_avoided += int(metadata.get("evaluation_cache_hits", 0))
        total_inferences += int(metadata.get("unique_paths_evaluated", 0))

    wall_seconds = float(manifest.get("wall_clock_seconds") or 0.0)
    if wall_seconds <= 0 and checkpoints:
        wall_seconds = max(float(checkpoint.get("runtime_seconds", 0.0)) for checkpoint in checkpoints)
    hours = wall_seconds / 3600 if wall_seconds > 0 else 0.0
    accepted = pi_found
    denominator = accepted or 1
    actual_model_inferences = int(manifest.get("model_inferences", total_inferences))
    return {
        "questions_attempted": len(attempted_ids) if attempted_ids else completed,
        "questions_completed": completed,
        "baseline_verified_correct": baseline_correct,
        "baseline_verified_incorrect": baseline_incorrect,
        "pi_star_found": pi_found,
        "accepted_supervision_samples": accepted,
        "acceptance_rate": accepted / completed if completed else 0.0,
        "baseline_wrong_corrected_by_mcts": wrong_corrected,
        "baseline_correct_shorter_correct_route": correct_shorter,
        "mean_pi_star_path_length": statistics.fmean(path_lengths) if path_lengths else None,
        "median_pi_star_path_length": statistics.median(path_lengths) if path_lengths else None,
        "mean_layers_saved_baseline_correct": statistics.fmean(layers_saved) if layers_saved else None,
        "repeat_frequency": repeats_with_any / accepted if accepted else 0.0,
        "skip_frequency_by_layer": {str(index): skip_by_layer[index] / denominator for index in range(28)},
        "repeat_frequency_by_layer": {str(index): repeat_by_layer[index] / denominator for index in range(28)},
        "candidate_status_counts": {
            status: statuses[status]
            for status in ("VERIFIED_CORRECT", "VERIFIED_INCORRECT", "UNCERTAIN")
        },
        "strict_parser_false_negatives_recovered": strict_false_negative,
        "strict_parser_false_positives_rejected": strict_false_positive,
        "duplicate_routes_avoided_by_cache": cache_avoided,
        "average_unique_evaluations_per_question": total_inferences / completed if completed else 0.0,
        "questions_per_hour": completed / hours if hours else None,
        "model_inferences_per_hour": actual_model_inferences / hours if hours else None,
        "wall_clock_seconds": wall_seconds,
        "model_inferences": actual_model_inferences,
    }


def _display(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = parser.parse_args(argv)
    manifest, records, checkpoints = load_run(args.run_directory)
    audit = compute_audit(manifest, records, checkpoints)
    if args.json:
        print(json.dumps(audit, indent=2, sort_keys=True))
        return 0
    labels = {
        "questions_attempted": "questions attempted",
        "questions_completed": "questions completed",
        "baseline_verified_correct": "baseline verified correct",
        "baseline_verified_incorrect": "baseline verified incorrect",
        "pi_star_found": "pi* found",
        "accepted_supervision_samples": "accepted supervision samples",
        "acceptance_rate": "acceptance rate",
        "baseline_wrong_corrected_by_mcts": "baseline wrong -> corrected by MCTS",
        "baseline_correct_shorter_correct_route": "baseline correct -> shorter correct route",
        "mean_pi_star_path_length": "mean pi* path length",
        "median_pi_star_path_length": "median pi* path length",
        "mean_layers_saved_baseline_correct": "mean layers saved for baseline-correct cases",
        "repeat_frequency": "repeat frequency",
        "candidate_status_counts": "candidate status counts",
        "strict_parser_false_negatives_recovered": "strict-parser false negatives recovered",
        "strict_parser_false_positives_rejected": "strict-parser false positives rejected",
        "duplicate_routes_avoided_by_cache": "duplicate routes avoided by cache",
        "average_unique_evaluations_per_question": "average unique evaluations/question",
        "questions_per_hour": "questions/hour",
        "model_inferences_per_hour": "model inferences/hour",
    }
    for key, label in labels.items():
        print(f"{label}: {_display(audit[key])}")
    print("skip frequency by layer: " + json.dumps(audit["skip_frequency_by_layer"], sort_keys=True))
    print("repeat frequency by layer: " + json.dumps(audit["repeat_frequency_by_layer"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
