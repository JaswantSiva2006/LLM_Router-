"""Export planned GSM8K IDs that still lack an eligible selected route."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from src.finalize import _load_run, discover_run_directories, validate_source_candidate


def _id_index(question_id: str) -> int:
    try:
        return int(question_id.rsplit(":", 1)[1])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"malformed GSM8K ID: {question_id!r}") from exc


def unresolved_ids(run_roots: list[Path]) -> list[str]:
    planned: set[str] = set()
    eligible: set[str] = set()
    directories = discover_run_directories(run_roots)
    if not directories:
        raise ValueError("no run manifests found")
    for directory in directories:
        manifest, records = _load_run(directory)
        if manifest.get("dataset") != "openai/gsm8k" or manifest.get("split") != "train":
            continue
        indices = manifest.get("requested_indices")
        if indices is None:
            start, count = int(manifest["start"]), int(manifest["count"])
            indices = range(start, start + count)
        planned.update(f"openai/gsm8k:main:train:{int(index):05d}" for index in indices)
        for record in records:
            candidate, _reason = validate_source_candidate(record, manifest, directory)
            if candidate is not None:
                eligible.add(candidate.question_id)
    return sorted(planned - eligible, key=lambda item: (_id_index(item), item))


def atomic_write_lines(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for line in lines:
            handle.write(line + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    unresolved = unresolved_ids(args.runs)
    atomic_write_lines(args.output, unresolved)
    print(f"wrote {len(unresolved)} unresolved IDs to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

