import pytest

from src.grading.gsm8k_verifier import VerificationLabel as L, verify_gsm8k_answer


@pytest.mark.parametrize(
    ("gold", "output", "expected"),
    [
        ("2", "Final answer: 12", L.VERIFIED_INCORRECT),
        ("2", "We start with 2 apples. Final answer: 12", L.VERIFIED_INCORRECT),
        ("2", "The answer is 2.", L.VERIFIED_CORRECT),
        ("2", "The answer is 2. Wait—final answer: 12.", L.VERIFIED_INCORRECT),
        ("72", r"\boxed{72}", L.VERIFIED_CORRECT),
        ("72", "Therefore, the final answer is seventy two.", L.VERIFIED_CORRECT),
        ("0.75", "The answer is 3/4.", L.VERIFIED_CORRECT),
        ("0.75", r"\boxed{\frac{3}{4}}", L.VERIFIED_CORRECT),
        ("1,200", "Final answer: 1,200", L.VERIFIED_CORRECT),
        ("-12", r"\boxed{-12}", L.VERIFIED_CORRECT),
        ("2", "The answer is 2. The next part of my explan", L.VERIFIED_CORRECT),
        ("2", "Since 1 + 2 = 2, the answer is 2.", L.VERIFIED_INCORRECT),
        ("2", "The final answer is ???", L.UNCERTAIN),
    ],
)
def test_required_review_cases(gold, output, expected):
    assert verify_gsm8k_answer(gold, output).label == expected


def test_incidental_numeric_mentions_are_not_answers_or_substring_matches():
    result = verify_gsm8k_answer("2", "I saw 12 items, 2.5 kg, and 200 birds.")
    assert result.label == L.UNCERTAIN
    assert all(not candidate.asserted for candidate in result.numeric_candidates)
    assert {c.normalized for c in result.numeric_candidates} == {"12", "5/2", "200"}


def test_strict_parser_does_not_gate_retention():
    result = verify_gsm8k_answer("2", "Some work. The answer is 2.")
    assert result.strict_parser_result is None
    assert result.label == L.VERIFIED_CORRECT


def test_later_contradiction_is_recorded_and_rejected():
    result = verify_gsm8k_answer("2", r"\boxed{2} Correction: final answer 12")
    assert result.contradiction_flag is True
    assert result.label == L.VERIFIED_INCORRECT


def test_true_arithmetic_and_direct_short_answer_are_accepted():
    assert verify_gsm8k_answer("2", "1 + 1 = 2. Therefore 2.").label == L.VERIFIED_CORRECT
    assert verify_gsm8k_answer("2", "2").label == L.VERIFIED_CORRECT


def test_provenance_fields_are_populated():
    result = verify_gsm8k_answer("2", r"1 + 1 = 2, so \boxed{2}")
    assert result.answer_context_spans
    assert result.arithmetic_checks[0].valid is True
    assert result.reason
    assert result.to_dict()["label"] == "VERIFIED_CORRECT"


def test_unclosed_box_is_uncertain_not_dropped_or_negative():
    result = verify_gsm8k_answer("2", r"\boxed{2")
    assert result.label == L.UNCERTAIN
    assert any(c.normalized == "2" for c in result.numeric_candidates)
