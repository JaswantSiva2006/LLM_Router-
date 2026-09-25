import math

import pytest

from src.grading.gsm8k_verifier import VerificationLabel
from src.model.routed_qwen import validate_path
from src.search.mcts import MCTSConfig, MCTSSearch, ucb_score


def config(**kwargs):
    defaults = dict(
        num_layers=4,
        num_simulations=40,
        max_path_length=8,
        random_child_probability=0.1,
        seed=42,
        early_stop="none",
    )
    defaults.update(kwargs)
    return MCTSConfig(**defaults)


def run(evaluator, **kwargs):
    return MCTSSearch(config(**kwargs)).run(
        question_id="q0",
        question="What is one plus one?",
        gold_answer="2",
        evaluate_route=evaluator,
    )


def test_ucb_calculation():
    actual = ucb_score(
        cumulative_reward=3.0,
        child_visits=4,
        parent_visits=10,
        path_length=3,
        num_layers=4,
        exploration_constant=1.8,
        length_penalty=3.0,
    )
    expected = 0.75 + 1.8 * math.sqrt(math.log(10) / 4) - 3.0 * (3 / 4)
    assert actual == pytest.approx(expected)
    assert math.isinf(ucb_score(
        cumulative_reward=0, child_visits=0, parent_visits=1,
        path_length=4, num_layers=4,
    ))


def test_cache_prevents_duplicate_inference():
    calls = []

    def evaluator(path):
        calls.append(tuple(path))
        return "Final answer: 12"

    result = run(evaluator, num_layers=2, max_path_length=4, num_simulations=50)
    assert len(calls) == len(set(calls))
    assert len(calls) == len(result.candidates)
    assert result.metadata["evaluation_cache_hits"] > 0


def test_search_is_deterministic_for_same_seed():
    def evaluator(path):
        return "The answer is 2." if len(path) <= 3 else "Final answer: 12"

    first = run(evaluator).to_dict()
    second = run(evaluator).to_dict()
    assert first == second


def test_shortest_verified_correct_route_wins_with_deterministic_ties():
    result = run(lambda path: "The answer is 2.")
    correct = [c for c in result.candidates if c.verifier_status == "VERIFIED_CORRECT" and c.path_length < 4]
    expected = min(correct, key=lambda c: (c.path_length, c.num_repeated_layers, c.execution_path))
    assert result.selected_pi_star == expected


def test_strict_parser_failure_does_not_block_pi_star():
    def evaluator(path):
        return "The answer is 2." if len(path) < 4 else "Final answer: 12"

    result = run(evaluator)
    assert result.selected_pi_star is not None
    assert result.selected_pi_star.strict_parser_result is None
    assert result.selected_pi_star.verifier_status == "VERIFIED_CORRECT"


def test_uncertain_route_is_unlabeled_and_never_selected():
    result = run(lambda path: "I calculated several values but the final answer is ???")
    assert result.selected_pi_star is None
    assert all(candidate.training_disposition == "UNLABELED" for candidate in result.candidates)
    assert all(candidate.mcts_reward == 0.0 for candidate in result.candidates)


def test_uncertain_baseline_never_yields_pi_star_even_if_an_edit_is_correct():
    full = [0, 1, 2, 3]

    def evaluator(path):
        return "The final answer is ???" if path == full else "The answer is 2."

    result = run(evaluator)
    assert result.baseline.verifier_status == "UNCERTAIN"
    assert any(candidate.verifier_status == "VERIFIED_CORRECT" for candidate in result.candidates)
    assert result.selected_pi_star is None


def test_baseline_correct_requires_strictly_shorter_correct_route():
    def evaluator(path):
        if path == [0, 1, 2, 3] or len(path) >= 4:
            return "The answer is 2."
        return "Final answer: 12"

    result = run(evaluator)
    assert result.baseline.verifier_status == "VERIFIED_CORRECT"
    assert result.selected_pi_star is None


def test_baseline_wrong_may_select_longer_repeated_route():
    def evaluator(path):
        repeated = len(path) > len(set(path))
        return "The answer is 2." if repeated and len(path) > 4 else "Final answer: 12"

    result = run(evaluator)
    assert result.baseline.verifier_status == "VERIFIED_INCORRECT"
    assert result.selected_pi_star is not None
    assert result.selected_pi_star.path_length > 4
    assert result.selected_pi_star.num_repeated_layers > 0


def test_no_illegal_path_can_enter_tree():
    result = run(lambda path: "Final answer: 12", num_simulations=150)
    for candidate in result.candidates:
        assert validate_path(candidate.execution_path, 4) == list(candidate.execution_path)
        assert candidate.path_length <= 8
        assert len(candidate.action_vector) == 4


def test_none_exhausts_budget_and_paper_can_stop_early():
    always_correct = lambda path: "The answer is 2."
    exhaustive = run(always_correct, num_simulations=20, early_stop="none")
    abbreviated = run(always_correct, num_simulations=20, early_stop="paper")
    assert exhaustive.metadata["num_simulations_completed"] == 20
    assert exhaustive.metadata["early_stopped"] is False
    assert abbreviated.metadata["num_simulations_completed"] < 20
    assert abbreviated.metadata["early_stopped"] is True


def test_raw_record_contains_required_provenance():
    result = run(lambda path: r"\boxed{2}", num_simulations=3)
    raw = result.to_dict()
    candidate = raw["evaluated_candidates"][0]
    required = {
        "question_id", "execution_path", "action_vector", "raw_model_response",
        "strict_parser_result", "verifier_status", "verifier_reason",
        "normalized_candidate_answer", "path_length", "num_skipped_layers",
        "num_repeated_layers", "simulation_index", "mcts_reward",
    }
    assert required <= candidate.keys()
    assert raw["search_metadata"]["second_pass_verification"] is True
