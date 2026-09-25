"""Print diverse accepted and rejected MCTS examples for human inspection."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

from src.audit.run import load_run


def example_categories(record: dict[str, Any]) -> list[str]:
    categories = []
    baseline_correct = record["baseline"]["verifier_status"] == "VERIFIED_CORRECT"
    selected = record.get("selected_pi_star")
    if selected:
        categories.append("accepted_baseline_correct_shorter" if baseline_correct else "accepted_baseline_wrong_corrected")
        if int(selected.get("num_repeated_layers", 0)) > 0:
            categories.append("accepted_with_repeat")
        if selected.get("strict_parser_result") is None:
            categories.append("accepted_strict_parser_recovery")
    else:
        statuses = {candidate["verifier_status"] for candidate in record.get("evaluated_candidates", [])}
        categories.append("rejected_with_uncertain" if "UNCERTAIN" in statuses else "rejected_no_correct_route")
    return categories


def select_diverse(records: list[dict[str, Any]], n: int) -> list[tuple[str, dict[str, Any]]]:
    if n < 0:
        raise ValueError("n must be non-negative")
    buckets: dict[str, deque] = defaultdict(deque)
    for record in sorted(records, key=lambda item: str(item["question_id"])):
        for category in example_categories(record):
            buckets[category].append(record)
    order = (
        "accepted_baseline_wrong_corrected",
        "accepted_baseline_correct_shorter",
        "accepted_with_repeat",
        "accepted_strict_parser_recovery",
        "rejected_with_uncertain",
        "rejected_no_correct_route",
    )
    selected: list[tuple[str, dict[str, Any]]] = []
    seen: set[str] = set()
    while len(selected) < n:
        progress = False
        for category in order:
            while buckets[category] and str(buckets[category][0]["question_id"]) in seen:
                buckets[category].popleft()
            if buckets[category] and len(selected) < n:
                record = buckets[category].popleft()
                seen.add(str(record["question_id"]))
                selected.append((category, record))
                progress = True
        if not progress:
            break
    if len(selected) < n:
        for record in sorted(records, key=lambda item: str(item["question_id"])):
            question_id = str(record["question_id"])
            if question_id not in seen:
                selected.append((example_categories(record)[0], record))
                seen.add(question_id)
                if len(selected) == n:
                    break
    return selected


def compact_example(category: str, record: dict[str, Any]) -> dict[str, Any]:
    baseline = record["baseline"]
    selected = record.get("selected_pi_star")
    uncertain = next((candidate for candidate in record.get("evaluated_candidates", [])
                      if candidate["verifier_status"] == "UNCERTAIN"), None)
    return {
        "category": category,
        "question_id": record["question_id"],
        "question": record["question"],
        "gold_answer": record["gold_answer"],
        "baseline": {
            "path": baseline["execution_path"],
            "response": baseline["raw_model_response"],
            "strict_parser_result": baseline["strict_parser_result"],
            "verifier_status": baseline["verifier_status"],
            "verifier_reason": baseline["verifier_reason"],
        },
        "selected_pi_star": selected,
        "representative_uncertain_candidate": uncertain,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--n", type=int, default=20)
    args = parser.parse_args(argv)
    _manifest, records, _checkpoints = load_run(args.run)
    chosen = select_diverse(records, args.n)
    for index, (category, record) in enumerate(chosen, 1):
        print(f"=== EXAMPLE {index}/{len(chosen)} ===")
        print(json.dumps(compact_example(category, record), ensure_ascii=False, indent=2))
    if len(chosen) < args.n:
        print(f"Only {len(chosen)} completed questions were available (requested {args.n}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

