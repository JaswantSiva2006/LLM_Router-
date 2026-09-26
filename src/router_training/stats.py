"""Class statistics and effective-number weights for router supervision."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .dataset import RouterTrainingRecord, load_router_dataset

BETA = 0.999


def _labels(record: RouterTrainingRecord | dict[str, Any]) -> Iterable[int]:
    if isinstance(record, RouterTrainingRecord):
        return record.optimal_layer_config
    return record["optimal_layer_config"]


def count_labels(dataset: Iterable[RouterTrainingRecord | dict[str, Any]]) -> dict[str, int]:
    """Count SKIP, EXECUTE, and REPEAT labels across every layer position."""
    counts = [0, 0, 0]
    for record in dataset:
        for label in _labels(record):
            if type(label) is not int or label not in (0, 1, 2):
                raise ValueError(f"invalid router label: {label!r}")
            counts[label] += 1
    return {
        "n_skip": counts[0],
        "n_execute": counts[1],
        "n_repeat": counts[2],
        "total": sum(counts),
    }


def effective_number_weights(
    counts: Iterable[int] | dict[str, int], *, beta: float = BETA
) -> list[float]:
    """Compute mean-one weights in [SKIP, EXECUTE, REPEAT] order."""
    if isinstance(counts, dict):
        values = [counts["n_skip"], counts["n_execute"], counts["n_repeat"]]
    else:
        values = list(counts)
    if len(values) != 3 or any(type(n) is not int or n <= 0 for n in values):
        raise ValueError("all three class counts must be positive integers")
    if not 0.0 < beta < 1.0:
        raise ValueError("beta must be between zero and one")

    # 1 - beta**n = -expm1(n * log(beta)), avoiding cancellation.
    raw = [(1.0 - beta) / -math.expm1(n * math.log(beta)) for n in values]
    mean = sum(raw) / len(raw)
    return [weight / mean for weight in raw]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--allow-small-dataset", action="store_true")
    args = parser.parse_args(argv)
    try:
        dataset = load_router_dataset(args.data, allow_small_dataset=args.allow_small_dataset)
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))
    counts = count_labels(dataset)
    result = {**counts, "alpha": effective_number_weights(counts), "beta": BETA}
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
