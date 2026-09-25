import json
from pathlib import Path

import pytest

from src.audit.run import compute_audit, main as audit_main
from src.export.audit_examples import main as examples_main, select_diverse
from src.run_generation import (
    append_jsonl_durable,
    atomic_write_json,
    merge_worker_outputs,
    partition_indices,
    read_valid_jsonl,
)


def candidate(path, status, strict=None, repeats=0):
    actions = [1, 1, 1, 1]
    if len(path) < 4:
        actions[1] = 0
    if repeats:
        actions[2] = 2
    return {
        "question_id": "q",
        "execution_path": path,
        "action_vector": actions,
        "raw_model_response": "response",
        "strict_parser_result": strict,
        "verifier_status": status,
        "verifier_reason": "reason",
        "normalized_candidate_answer": "2" if status == "VERIFIED_CORRECT" else None,
        "path_length": len(path),
        "num_skipped_layers": actions.count(0),
        "num_repeated_layers": repeats,
        "simulation_index": 0,
        "mcts_reward": float(status == "VERIFIED_CORRECT"),
        "training_disposition": "UNLABELED" if status == "UNCERTAIN" else "candidate",
    }


def record(index, baseline_status, selected, candidates, cache_hits=0):
    baseline = candidate([0, 1, 2, 3], baseline_status, "2" if baseline_status == "VERIFIED_CORRECT" else "12")
    question_id = f"openai/gsm8k:main:train:{index:05d}"
    baseline["question_id"] = question_id
    for item in candidates:
        item["question_id"] = question_id
    if selected:
        selected["question_id"] = question_id
    return {
        "question_id": question_id,
        "question": f"question {index}",
        "gold_answer": "2",
        "baseline": baseline,
        "evaluated_candidates": candidates,
        "selected_pi_star": selected,
        "search_metadata": {"unique_paths_evaluated": len(candidates), "evaluation_cache_hits": cache_hits},
    }


def test_partition_is_deterministic_round_robin():
    assert partition_indices(10, 7, 2) == [[10, 12, 14, 16], [11, 13, 15]]
    assert partition_indices(10, 7, 2) == partition_indices(10, 7, 2)


def test_append_repair_and_resume_ids(tmp_path):
    path = tmp_path / "worker_0.jsonl"
    append_jsonl_durable(path, {"question_id": "q0", "value": 1})
    with path.open("ab") as handle:
        handle.write(b'{"question_id":"torn"')
    with pytest.raises(ValueError):
        read_valid_jsonl(path)
    assert read_valid_jsonl(path, repair=True) == [{"question_id": "q0", "value": 1}]
    assert path.read_bytes().endswith(b"\n")


def test_atomic_checkpoint_replacement(tmp_path):
    path = tmp_path / "checkpoint.json"
    atomic_write_json(path, {"completed_ids": ["q0"]})
    atomic_write_json(path, {"completed_ids": ["q0", "q1"]})
    assert json.loads(path.read_text("utf-8"))["completed_ids"] == ["q0", "q1"]
    assert not list(tmp_path.glob("*.tmp"))


def test_merge_is_deterministic_and_workers_are_isolated(tmp_path):
    append_jsonl_durable(tmp_path / "worker_0.jsonl", record(2, "VERIFIED_INCORRECT", None, []))
    append_jsonl_durable(tmp_path / "worker_1.jsonl", record(1, "VERIFIED_INCORRECT", None, []))
    merged = merge_worker_outputs(tmp_path, 2)
    assert [item["question_id"] for item in merged] == [
        "openai/gsm8k:main:train:00001", "openai/gsm8k:main:train:00002"
    ]
    assert read_valid_jsonl(tmp_path / "worker_0.jsonl")[0]["question_id"].endswith("00002")
    assert read_valid_jsonl(tmp_path / "merged.jsonl") == merged


def test_merge_rejects_conflicting_duplicate_ids(tmp_path):
    first = record(0, "VERIFIED_INCORRECT", None, [])
    second = record(0, "VERIFIED_CORRECT", None, [])
    append_jsonl_durable(tmp_path / "worker_0.jsonl", first)
    append_jsonl_durable(tmp_path / "worker_1.jsonl", second)
    with pytest.raises(ValueError, match="conflicting"):
        merge_worker_outputs(tmp_path, 2)


def test_audit_metrics_and_strict_parser_disagreements():
    recovered = candidate([0, 2, 3], "VERIFIED_CORRECT", strict=None)
    false_positive = candidate([0, 1, 2, 3], "VERIFIED_INCORRECT", strict="2")
    uncertain = candidate([0, 1, 2], "UNCERTAIN", strict=None)
    repeated = candidate([0, 1, 2, 2, 3], "VERIFIED_CORRECT", strict="2", repeats=1)
    records = [
        record(0, "VERIFIED_INCORRECT", recovered, [recovered, false_positive], cache_hits=2),
        record(1, "VERIFIED_CORRECT", repeated, [repeated, uncertain], cache_hits=1),
    ]
    manifest = {"wall_clock_seconds": 3600}
    checkpoints = [{"attempted_ids": [item["question_id"] for item in records]}]
    audit = compute_audit(manifest, records, checkpoints)
    assert audit["questions_attempted"] == 2
    assert audit["questions_completed"] == 2
    assert audit["baseline_verified_correct"] == 1
    assert audit["baseline_verified_incorrect"] == 1
    assert audit["pi_star_found"] == 2
    assert audit["baseline_wrong_corrected_by_mcts"] == 1
    assert audit["baseline_correct_shorter_correct_route"] == 1
    assert audit["strict_parser_false_negatives_recovered"] == 1
    assert audit["strict_parser_false_positives_rejected"] == 1
    assert audit["candidate_status_counts"] == {
        "VERIFIED_CORRECT": 2, "VERIFIED_INCORRECT": 1, "UNCERTAIN": 1
    }
    assert audit["duplicate_routes_avoided_by_cache"] == 3
    assert audit["average_unique_evaluations_per_question"] == 2
    assert audit["questions_per_hour"] == 2
    assert audit["model_inferences_per_hour"] == 4
    assert audit["repeat_frequency"] == 0.5


def test_diverse_export_includes_accepted_and_rejected():
    accepted = candidate([0, 2, 3], "VERIFIED_CORRECT")
    uncertain = candidate([0, 1, 2], "UNCERTAIN")
    records = [
        record(0, "VERIFIED_INCORRECT", accepted, [accepted]),
        record(1, "VERIFIED_INCORRECT", None, [uncertain]),
        record(2, "VERIFIED_INCORRECT", None, [candidate([0, 1], "VERIFIED_INCORRECT")]),
    ]
    chosen = select_diverse(records, 3)
    categories = {category for category, _record in chosen}
    assert "accepted_baseline_wrong_corrected" in categories
    assert "rejected_with_uncertain" in categories
    assert "rejected_no_correct_route" in categories


def test_audit_and_example_clis_on_synthetic_run(tmp_path, capsys):
    selected = candidate([0, 2, 3], "VERIFIED_CORRECT")
    append_jsonl_durable(
        tmp_path / "merged.jsonl",
        record(0, "VERIFIED_INCORRECT", selected, [selected]),
    )
    atomic_write_json(tmp_path / "run_manifest.json", {
        "worker_count": 1,
        "gpus": [0],
        "wall_clock_seconds": 60,
    })
    assert audit_main([str(tmp_path)]) == 0
    audit_output = capsys.readouterr().out
    assert "questions completed: 1" in audit_output
    assert examples_main(["--run", str(tmp_path), "--n", "1"]) == 0
    examples_output = capsys.readouterr().out
    assert "EXAMPLE 1/1" in examples_output
    assert "accepted_baseline_wrong_corrected" in examples_output
