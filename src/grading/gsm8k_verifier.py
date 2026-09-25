"""Conservative, deterministic three-way verifier for GSM8K generations."""

from __future__ import annotations

import argparse
import re
from dataclasses import asdict, dataclass
from enum import Enum
from fractions import Fraction
from typing import Any

from src.grading.numeric import (
    evaluate_arithmetic,
    fraction_to_string,
    numeric_tokens,
    parse_numeric,
    parse_written_number,
)


class VerificationLabel(str, Enum):
    VERIFIED_CORRECT = "VERIFIED_CORRECT"
    VERIFIED_INCORRECT = "VERIFIED_INCORRECT"
    UNCERTAIN = "UNCERTAIN"


@dataclass(frozen=True)
class NumericCandidate:
    raw: str
    normalized: str
    start: int
    end: int
    source: str
    asserted: bool


@dataclass(frozen=True)
class ContextSpan:
    start: int
    end: int
    text: str
    source: str


@dataclass(frozen=True)
class ArithmeticCheck:
    expression: str
    lhs: str
    rhs: str
    valid: bool | None
    materially_supports_claim: bool


@dataclass(frozen=True)
class VerificationResult:
    label: VerificationLabel
    strict_parser_result: str | None
    numeric_candidates: tuple[NumericCandidate, ...]
    answer_context_spans: tuple[ContextSpan, ...]
    arithmetic_checks: tuple[ArithmeticCheck, ...]
    contradiction_flag: bool
    reason: str

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["label"] = self.label.value
        return result


_LITERAL = r"[+-]?(?:(?:\d{1,3}(?:,\d{3})+)|\d+)(?:\.\d+)?(?:\s*/\s*[+-]?\d+(?:\.\d+)?)?"
_WORDS = (
    r"(?:negative\s+|minus\s+)?(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|"
    r"eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|"
    r"thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|million|billion)"
    r"(?:[ -](?:and[ -])?(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|"
    r"twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|"
    r"forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|million|billion))*\b"
)
_VALUE = rf"(?P<value>{_LITERAL}|{_WORDS})"
_CONTEXT_PATTERNS = (
    ("gsm8k_marker", re.compile(rf"####\s*{_VALUE}", re.I)),
    ("answer_is", re.compile(rf"(?:the\s+)?answer\s+is\s*(?:[:=]\s*)?{_VALUE}", re.I)),
    ("final_answer", re.compile(rf"final\s+answer\s*(?:is|:|=)?\s*{_VALUE}", re.I)),
    ("therefore", re.compile(rf"(?:therefore|thus|hence)\s*[,,:]?\s*(?:the\s+answer\s+is\s*)?{_VALUE}", re.I)),
)
_BOXED = re.compile(r"\\boxed\s*\{((?:[^{}]|\{[^{}]*\})+)\}")
_MALFORMED_ASSERTION = re.compile(r"(?:final\s+answer|answer\s+is)\s*[:=]?\s*(?:\?+|[^\w\d+\-.\\$]+)\s*$", re.I)
_EQUATION = re.compile(
    r"(?<![\w])(?P<lhs>[()+\-]?\s*\d[\d,().\s+\-*/×÷]*?)\s*=\s*(?P<rhs>[+\-]?\d[\d,]*(?:\.\d+)?(?:\s*/\s*[+\-]?\d+)?)"
)


def _value(raw: str) -> Fraction | None:
    return parse_numeric(raw) or parse_written_number(raw)


def strict_boxed_parser(output: str) -> str | None:
    """Legacy-style strict parse, retained only as diagnostic provenance."""
    match = re.fullmatch(r"\s*\\boxed\s*\{((?:[^{}]|\{[^{}]*\})+)\}\s*[.!]?\s*", output, re.S)
    if not match:
        return None
    parsed = _value(match.group(1))
    return fraction_to_string(parsed) if parsed is not None else None


def _candidate(raw: str, start: int, end: int, source: str, asserted: bool) -> NumericCandidate | None:
    parsed = _value(raw)
    if parsed is None:
        return None
    return NumericCandidate(raw.strip(), fraction_to_string(parsed), start, end, source, asserted)


def extract_candidates(output: str) -> tuple[list[NumericCandidate], list[ContextSpan]]:
    """Extract all numeric provenance plus semantically asserted candidates."""
    candidates: list[NumericCandidate] = []
    contexts: list[ContextSpan] = []
    asserted_ranges: set[tuple[int, int, str]] = set()

    for match in _BOXED.finditer(output):
        raw = match.group(1).strip()
        inner_start = match.start(1)
        candidate = _candidate(raw, inner_start, match.end(1), "boxed", True)
        if candidate:
            candidates.append(candidate)
            asserted_ranges.add((candidate.start, candidate.end, candidate.normalized))
            contexts.append(ContextSpan(match.start(), match.end(), match.group(0), "boxed"))

    for source, pattern in _CONTEXT_PATTERNS:
        for match in pattern.finditer(output):
            raw = match.group("value")
            start, end = match.span("value")
            candidate = _candidate(raw, start, end, source, True)
            if candidate and (start, end, candidate.normalized) not in asserted_ranges:
                candidates.append(candidate)
                asserted_ranges.add((start, end, candidate.normalized))
                contexts.append(ContextSpan(match.start(), match.end(), match.group(0), source))

    # A bare numeric result on the final nonempty line is an answer assertion.
    end_match = re.search(rf"(?m)^\s*(?P<value>{_LITERAL})\s*[.!]?\s*$", output.rstrip())
    if end_match:
        start, end = end_match.span("value")
        candidate = _candidate(end_match.group("value"), start, end, "standalone_end", True)
        if candidate and (start, end, candidate.normalized) not in asserted_ranges:
            candidates.append(candidate)
            asserted_ranges.add((start, end, candidate.normalized))
            contexts.append(ContextSpan(end_match.start(), end_match.end(), end_match.group(0), "standalone_end"))

    # Preserve every other numeric token for auditability, but do not assert it.
    covered = {(candidate.start, candidate.end) for candidate in candidates}
    for raw, parsed, start, end in numeric_tokens(output):
        if (start, end) not in covered:
            candidates.append(NumericCandidate(raw, fraction_to_string(parsed), start, end, "numeric_mention", False))
    candidates.sort(key=lambda item: (item.start, item.end, not item.asserted))
    contexts.sort(key=lambda item: item.start)
    return candidates, contexts


def check_arithmetic(output: str, claimed: Fraction | None) -> list[ArithmeticCheck]:
    checks = []
    for match in _EQUATION.finditer(output):
        lhs_text, rhs_text = match.group("lhs").strip(), match.group("rhs").strip()
        lhs, rhs = evaluate_arithmetic(lhs_text), parse_numeric(rhs_text)
        valid = None if lhs is None or rhs is None else lhs == rhs
        material = claimed is not None and rhs == claimed
        checks.append(ArithmeticCheck(match.group(0), lhs_text, rhs_text, valid, material))
    return checks


def verify_gsm8k_answer(gold_answer: str, output: str) -> VerificationResult:
    gold = parse_numeric(gold_answer)
    if gold is None:
        raise ValueError(f"Unsupported gold answer: {gold_answer!r}")
    candidates, contexts = extract_candidates(output)
    asserted = [candidate for candidate in candidates if candidate.asserted]
    values = [parse_numeric(candidate.normalized) for candidate in asserted]
    contradiction = len(set(values)) > 1
    boxed_starts = len(re.findall(r"\\boxed\s*\{", output))
    malformed = bool(_MALFORMED_ASSERTION.search(output.rstrip())) or boxed_starts > len(_BOXED.findall(output))

    if not asserted:
        label = VerificationLabel.UNCERTAIN
        reason = "No unambiguous answer assertion was found."
        claimed = None
    else:
        final = asserted[-1]
        claimed = parse_numeric(final.normalized)
        if claimed != gold:
            label = VerificationLabel.VERIFIED_INCORRECT
            reason = f"The latest asserted answer is {final.normalized}, not the gold answer {fraction_to_string(gold)}."
        elif malformed and final.end < len(output.rstrip()):
            label = VerificationLabel.UNCERTAIN
            reason = "A correct candidate exists, but a later malformed answer assertion is ambiguous."
        else:
            label = VerificationLabel.VERIFIED_CORRECT
            reason = f"The latest clear answer assertion equals the gold answer {fraction_to_string(gold)}."

    checks = check_arithmetic(output, claimed)
    material = [check for check in checks if check.materially_supports_claim]
    if label == VerificationLabel.VERIFIED_CORRECT and any(check.valid is False for check in material):
        label = VerificationLabel.VERIFIED_INCORRECT
        reason = "An explicitly false arithmetic equality materially derives the claimed gold answer."
    elif label == VerificationLabel.VERIFIED_CORRECT and any(check.valid is None for check in material):
        label = VerificationLabel.UNCERTAIN
        reason = "Arithmetic materially supporting the claimed answer could not be interpreted safely."

    return VerificationResult(
        label=label,
        strict_parser_result=strict_boxed_parser(output),
        numeric_candidates=tuple(candidates),
        answer_context_spans=tuple(contexts),
        arithmetic_checks=tuple(checks),
        contradiction_flag=contradiction,
        reason=reason,
    )


def _self_test() -> None:
    cases = [
        ("2", "The answer is 2.", VerificationLabel.VERIFIED_CORRECT),
        ("2", "The final answer is 12.", VerificationLabel.VERIFIED_INCORRECT),
        ("72", r"\boxed{72}", VerificationLabel.VERIFIED_CORRECT),
        ("0.75", "Final answer: 3/4", VerificationLabel.VERIFIED_CORRECT),
        ("2", "1 + 2 = 2. The answer is 2.", VerificationLabel.VERIFIED_INCORRECT),
        ("2", r"The final answer is ???", VerificationLabel.UNCERTAIN),
    ]
    for gold, output, expected in cases:
        actual = verify_gsm8k_answer(gold, output).label
        assert actual == expected, (gold, output, expected, actual)
    print(f"gsm8k_verifier self-test: {len(cases)} passed, 0 failed")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    if not args.self_test:
        parser.error("--self-test is required")
    _self_test()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
