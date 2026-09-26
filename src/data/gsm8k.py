"""Stable GSM8K training-record adapter for Qwen instruction generation."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from itertools import islice
from typing import Any, Iterable, Iterator

from src.grading.numeric import fraction_to_string, parse_numeric

DATASET_NAME = "openai/gsm8k"
DATASET_CONFIG = "main"
DEFAULT_SPLIT = "train"

PROMPT_TEMPLATE = """Question: {question}
ONLY return the final result in LaTeX with no words.
The result MUST be wrapped inside \\boxed{{...}}."""


def build_generation_prompt(question: str) -> str:
    """Build the canonical prompt used for GSM8K route search and training."""
    return PROMPT_TEMPLATE.format(question=question)


@dataclass(frozen=True)
class GSM8KRecord:
    dataset_id: str
    index: int
    question: str
    gold_reasoning: str
    gold_answer: str
    generation_prompt: str


def parse_gold_answer(answer: str) -> tuple[str, str]:
    """Split GSM8K rationale from its final ``#### number`` marker."""
    matches = list(re.finditer(r"(?m)^\s*####\s*(.*?)\s*$", answer))
    if not matches:
        raise ValueError("GSM8K answer has no final '#### ...' marker")
    match = matches[-1]
    raw_final = match.group(1).strip()
    parsed = parse_numeric(raw_final)
    if parsed is None:
        raise ValueError(f"GSM8K final answer is not a supported number: {raw_final!r}")
    reasoning = answer[: match.start()].rstrip()
    return reasoning, fraction_to_string(parsed)


def adapt_record(item: dict[str, Any], index: int, split: str = DEFAULT_SPLIT) -> GSM8KRecord:
    question = str(item["question"]).strip()
    reasoning, gold = parse_gold_answer(str(item["answer"]))
    return GSM8KRecord(
        dataset_id=f"{DATASET_NAME}:{DATASET_CONFIG}:{split}:{index:05d}",
        index=index,
        question=question,
        gold_reasoning=reasoning,
        gold_answer=gold,
        generation_prompt=build_generation_prompt(question),
    )


def iter_gsm8k(split: str = DEFAULT_SPLIT, *, streaming: bool = False) -> Iterator[GSM8KRecord]:
    """Load and adapt GSM8K without shuffling, preserving source row indices."""
    if split != DEFAULT_SPLIT:
        raise ValueError("this data-generation adapter only permits the GSM8K train split")
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("Install the 'datasets' package to load GSM8K") from exc
    dataset: Iterable[dict[str, Any]] = load_dataset(
        DATASET_NAME, DATASET_CONFIG, split=split, streaming=streaming
    )
    for index, item in enumerate(dataset):
        yield adapt_record(item, index, split)


def load_gsm8k(split: str = DEFAULT_SPLIT) -> list[GSM8KRecord]:
    return list(iter_gsm8k(split))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inspect", type=int, metavar="N", help="stream and print the first N adapted records")
    args = parser.parse_args(argv)
    if args.inspect is None:
        parser.error("--inspect N is required")
    if args.inspect < 0:
        parser.error("--inspect must be non-negative")
    for record in islice(iter_gsm8k(streaming=True), args.inspect):
        print(json.dumps(asdict(record), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
