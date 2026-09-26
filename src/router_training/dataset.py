"""Validated JSONL interface for Dr.LLM router supervision."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

from src.data.gsm8k import build_generation_prompt

EXPECTED_SAMPLES = 4000
NUM_LAYERS = 28
VALID_LABELS = frozenset((0, 1, 2))


@dataclass(frozen=True)
class RouterTrainingRecord:
    id: str
    question: str
    answer: str
    optimal_layer_config: tuple[int, ...]

    @property
    def router_input(self) -> str:
        """Canonical model input; the provenance answer is deliberately excluded."""
        return build_generation_prompt(self.question)


def _record_from_json(value: Any, *, line_number: int) -> RouterTrainingRecord:
    where = f"line {line_number}"
    if not isinstance(value, dict):
        raise ValueError(f"{where}: record must be a JSON object")

    required = {"id", "question", "answer", "optimal_layer_config"}
    missing = required.difference(value)
    if missing:
        raise ValueError(f"{where}: missing required fields: {', '.join(sorted(missing))}")

    record_id, question, answer = value["id"], value["question"], value["answer"]
    if not isinstance(record_id, str) or not record_id.strip():
        raise ValueError(f"{where}: id must be a non-empty string")
    if not isinstance(question, str) or not question.strip():
        raise ValueError(f"{where}: question must be a non-empty string")
    if not isinstance(answer, str):
        raise ValueError(f"{where}: answer must be a string")

    labels = value["optimal_layer_config"]
    if not isinstance(labels, list) or len(labels) != NUM_LAYERS:
        actual = len(labels) if isinstance(labels, list) else "non-list"
        raise ValueError(f"{where}: optimal_layer_config must contain exactly {NUM_LAYERS} labels (got {actual})")
    for index, label in enumerate(labels):
        if type(label) is not int or label not in VALID_LABELS:
            raise ValueError(
                f"{where}: label at layer {index} must be one of {{0, 1, 2}}; got {label!r}"
            )

    return RouterTrainingRecord(record_id, question, answer, tuple(labels))


def load_router_dataset(
    path: str | Path, *, allow_small_dataset: bool = False
) -> list[RouterTrainingRecord]:
    """Load and validate router data without requiring production artifacts at import."""
    data_path = Path(path)
    if not data_path.is_file():
        raise FileNotFoundError(
            f"Router dataset not found: {data_path}. Generate it first, or pass a fixture "
            "with --allow-small-dataset for development."
        )

    records: list[RouterTrainingRecord] = []
    with data_path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                raise ValueError(f"line {line_number}: blank lines are not valid JSONL records")
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"line {line_number}: invalid JSON: {exc.msg}") from exc
            records.append(_record_from_json(value, line_number=line_number))

    ids = [record.id for record in records]
    questions = [record.question for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError("dataset contains duplicate ids/records")
    if len(questions) != len(set(questions)):
        raise ValueError("dataset contains duplicate questions/records")
    if not allow_small_dataset and len(records) != EXPECTED_SAMPLES:
        raise ValueError(
            f"production dataset must contain exactly {EXPECTED_SAMPLES} samples; "
            f"found {len(records)} (use --allow-small-dataset only for development)"
        )
    return records


class RouterTrainingDataset(Sequence[RouterTrainingRecord]):
    """A lightweight sequence suitable for tokenization by a later training phase."""

    def __init__(self, path: str | Path, *, allow_small_dataset: bool = False) -> None:
        self.records = load_router_dataset(path, allow_small_dataset=allow_small_dataset)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> RouterTrainingRecord:
        return self.records[index]

    def __iter__(self) -> Iterator[RouterTrainingRecord]:
        return iter(self.records)
