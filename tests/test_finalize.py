import gzip
import json

import pytest

from src.finalize import collect_candidates, finalize_dataset
from src.run_generation import append_jsonl_durable, atomic_write_json, load_requested_indices
from src.unresolved import unresolved_ids


def source_candidate(index, *, baseline="VERIFIED_INCORRECT", selected=True, path=None, status="VERIFIED_CORRECT"):
    path = path or list(range(27))
    actions = [1] * 28
    missing = set(range(28)) - set(path)
    for layer in missing:
        actions[layer] = 0
    for layer in set(path):
        if path.count(layer) == 2:
            actions[layer] = 2
    question_id = f"openai/gsm8k:main:train:{index:05d}"
    selected_record = None if not selected else {
        "question_id": question_id,
        "execution_path": path,
        "action_vector": actions,
        "raw_model_response": r"\boxed{2}",
        "strict_parser_result": "2",
        "verifier_status": status,
        "verifier_reason": "correct",
        "normalized_candidate_answer": "2",
        "path_length": len(path),
        "num_skipped_layers": actions.count(0),
        "num_repeated_layers": actions.count(2),
        "simulation_index": 1,
        "mcts_reward": 1.0,
        "training_disposition": "POSITIVE_CANDIDATE" if status == "VERIFIED_CORRECT" else "UNLABELED",
    }
    return {
        "question_id": question_id,
        "question": f"Unique question {index}?",
        "gold_answer": "2",
        "baseline": {"verifier_status": baseline, "raw_model_response": r"\boxed{12}"},
        "evaluated_candidates": [],
        "selected_pi_star": selected_record,
        "search_metadata": {"unique_paths_evaluated": 50},
    }


def make_run(tmp_path, records, *, indices=None, name="run"):
    directory = tmp_path / name
    directory.mkdir()
    indices = indices if indices is not None else list(range(len(records)))
    atomic_write_json(directory / "run_manifest.json", {
        "model": "Qwen/Qwen2.5-1.5B-Instruct",
        "model_revision": "abc123",
        "dataset": "openai/gsm8k",
        "split": "train",
        "start": min(indices) if indices else 0,
        "count": len(indices),
        "requested_indices": indices,
        "torch_version": "test",
        "transformers_version": "test",
        "git_commit": "deadbeef",
    })
    for record in records:
        append_jsonl_durable(directory / "merged.jsonl", record)
    return directory


def test_finalizer_replays_and_writes_minimal_canonical_artifacts(tmp_path):
    run = make_run(tmp_path, [source_candidate(0), source_candidate(1)])
    output = tmp_path / "artifacts" / "gsm8k_mcts_4k.jsonl"
    calls = []

    def replay(item):
        calls.append(item)
        return "The answer is 2."

    summary = finalize_dataset(run_roots=[run], target=2, output=output, replay=replay)
    rows = [json.loads(line) for line in output.read_text("utf-8").splitlines()]
    assert len(calls) == 2
    assert len(rows) == 2
    assert set(rows[0]) == {"id", "question", "answer", "optimal_layer_config"}
    assert summary["rows"] == summary["unique_ids"] == summary["unique_questions"] == 2
    assert summary["all_config_lengths_28"] is True
    assert summary["labels_subset_0_1_2"] is True
    assert summary["all_configs_round_trip_to_legal_paths"] is True
    assert summary["test_split_ids"] == 0
    assert summary["uncertain_selected_samples"] == 0
    assert summary["replay_failures"] == 0
    provenance = output.with_name("gsm8k_mcts_4k_provenance.jsonl.gz")
    with gzip.open(provenance, "rt", encoding="utf-8") as handle:
        assert len([json.loads(line) for line in handle]) == 2
    assert output.with_name("gsm8k_mcts_4k_summary.json").exists()


def test_replay_failure_is_quarantined_and_next_candidate_used(tmp_path):
    run = make_run(tmp_path, [source_candidate(0), source_candidate(1), source_candidate(2)])
    output = tmp_path / "artifacts" / "gsm8k_mcts_4k.jsonl"

    def replay(item):
        return "Final answer: 12" if item["question_id"].endswith("00000") else "Final answer: 2"

    summary = finalize_dataset(run_roots=[run], target=2, output=output, replay=replay)
    rows = [json.loads(line) for line in output.read_text("utf-8").splitlines()]
    assert {row["id"] for row in rows} == {
        "openai/gsm8k:main:train:00001", "openai/gsm8k:main:train:00002"
    }
    assert summary["replay_failures"] == 0
    assert summary["quarantined_replay_failures"] == 1
    quarantine = output.with_name("gsm8k_mcts_4k_quarantine.jsonl")
    assert json.loads(quarantine.read_text("utf-8").splitlines()[0])["reason"] == "replay_not_verified_correct"


def test_replay_cache_prevents_duplicate_replay_on_resume(tmp_path):
    run = make_run(tmp_path, [source_candidate(0)])
    output = tmp_path / "artifact.jsonl"
    calls = 0

    def replay(_item):
        nonlocal calls
        calls += 1
        return "The answer is 2."

    finalize_dataset(run_roots=[run], target=1, output=output, replay=replay)
    finalize_dataset(run_roots=[run], target=1, output=output, replay=replay)
    assert calls == 1


def test_fewer_than_target_fails_before_replay_or_publication(tmp_path):
    run = make_run(tmp_path, [source_candidate(0)])
    output = tmp_path / "final.jsonl"
    called = False

    def replay(_item):
        nonlocal called
        called = True
        return "The answer is 2."

    with pytest.raises(RuntimeError, match="only 1 unique eligible"):
        finalize_dataset(run_roots=[run], target=2, output=output, replay=replay)
    assert called is False
    assert not output.exists()


@pytest.mark.parametrize(
    "record",
    [
        source_candidate(0, selected=False),
        source_candidate(0, status="UNCERTAIN"),
        source_candidate(0, baseline="VERIFIED_CORRECT", path=list(range(28))),
        source_candidate(0, baseline="UNCERTAIN"),
    ],
)
def test_ineligible_sources_are_rejected(record, tmp_path):
    run = make_run(tmp_path, [record])
    candidates, reasons, _runs = collect_candidates([run])
    assert candidates == []
    assert sum(reasons.values()) == 1


def test_unresolved_includes_missing_and_ineligible_but_not_eligible(tmp_path):
    run = make_run(
        tmp_path,
        [source_candidate(0), source_candidate(1, selected=False)],
        indices=[0, 1, 2],
    )
    assert unresolved_ids([run]) == [
        "openai/gsm8k:main:train:00001",
        "openai/gsm8k:main:train:00002",
    ]


def test_sparse_ids_file_accepts_full_ids_and_indices(tmp_path):
    path = tmp_path / "ids.txt"
    path.write_text("# retry\nopenai/gsm8k:main:train:00009\n3\n", encoding="utf-8")
    assert load_requested_indices(path) == [3, 9]
    path.write_text("3\n3\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        load_requested_indices(path)

