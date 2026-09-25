"""Dr.LLM-style length-aware MCTS with conservative GSM8K verification."""

from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Callable

from src.grading.gsm8k_verifier import (
    VerificationLabel,
    VerificationResult,
    verify_gsm8k_answer,
)
from src.model.routed_qwen import LayerAction, path_to_labels, validate_path

Route = tuple[int, ...]
RouteEvaluator = Callable[[list[int]], str]


class EarlyStopMode(str, Enum):
    NONE = "none"
    PAPER = "paper"


@dataclass(frozen=True)
class MCTSConfig:
    num_layers: int = 28
    num_simulations: int = 50
    exploration_constant: float = 1.8
    length_penalty: float = 3.0
    random_child_probability: float = 0.1
    max_path_length: int = 56
    skip_sizes: tuple[int, ...] = (1, 2)
    repeat_block_size: int = 1
    repeat_count: int = 1
    seed: int = 42
    early_stop: EarlyStopMode | str = EarlyStopMode.NONE

    def __post_init__(self) -> None:
        mode = EarlyStopMode(self.early_stop)
        object.__setattr__(self, "early_stop", mode)
        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if self.num_simulations <= 0:
            raise ValueError("num_simulations must be positive")
        if self.exploration_constant < 0 or self.length_penalty < 0:
            raise ValueError("UCB constants must be non-negative")
        if not 0 <= self.random_child_probability <= 1:
            raise ValueError("random_child_probability must be in [0, 1]")
        if self.max_path_length <= 0 or self.max_path_length > 2 * self.num_layers:
            raise ValueError("max_path_length must be in 1..2L")
        if self.skip_sizes != (1, 2):
            raise ValueError("this reproduction requires skip sizes (1, 2)")
        if self.repeat_block_size != 1 or self.repeat_count != 1:
            raise ValueError("this reproduction permits only one repetition of one layer")


@dataclass
class MCTSNode:
    path: Route
    parent: "MCTSNode | None" = None
    children: list["MCTSNode"] = field(default_factory=list)
    untried_paths: list[Route] = field(default_factory=list)
    visits: int = 0
    cumulative_reward: float = 0.0

    @property
    def mean_reward(self) -> float:
        return self.cumulative_reward / self.visits if self.visits else 0.0


@dataclass(frozen=True)
class EvaluatedPath:
    question_id: str
    execution_path: tuple[int, ...]
    action_vector: tuple[int, ...]
    raw_model_response: str
    strict_parser_result: str | None
    verifier_status: str
    verifier_reason: str
    normalized_candidate_answer: str | None
    path_length: int
    num_skipped_layers: int
    num_repeated_layers: int
    simulation_index: int
    mcts_reward: float
    training_disposition: str

    def to_dict(self) -> dict:
        result = asdict(self)
        result["execution_path"] = list(self.execution_path)
        result["action_vector"] = list(self.action_vector)
        return result


@dataclass(frozen=True)
class QuestionSearchResult:
    question_id: str
    question: str
    gold_answer: str
    baseline: EvaluatedPath
    candidates: tuple[EvaluatedPath, ...]
    selected_pi_star: EvaluatedPath | None
    metadata: dict

    def to_dict(self) -> dict:
        return {
            "question_id": self.question_id,
            "question": self.question,
            "gold_answer": self.gold_answer,
            "baseline": self.baseline.to_dict(),
            "evaluated_candidates": [candidate.to_dict() for candidate in self.candidates],
            "selected_pi_star": self.selected_pi_star.to_dict() if self.selected_pi_star else None,
            "search_metadata": self.metadata,
        }


def ucb_score(
    *,
    cumulative_reward: float,
    child_visits: int,
    parent_visits: int,
    path_length: int,
    num_layers: int,
    exploration_constant: float = 1.8,
    length_penalty: float = 3.0,
) -> float:
    """Length-aware UCB from Dr.LLM Section 4.2."""
    if child_visits == 0:
        return math.inf
    if parent_visits <= 0:
        raise ValueError("a visited child must have a visited parent")
    mean_reward = cumulative_reward / child_visits
    exploration = exploration_constant * math.sqrt(math.log(parent_visits) / child_visits)
    penalty = length_penalty * (path_length / num_layers)
    return mean_reward + exploration - penalty


def _successors(path: Route, config: MCTSConfig, rng: random.Random) -> list[Route]:
    """Return unique, validated routes reachable by one allowed edit."""
    successors: set[Route] = set()
    for start in range(len(path)):
        for size in config.skip_sizes:
            if start + size <= len(path):
                candidate = path[:start] + path[start + size :]
                try:
                    successors.add(tuple(validate_path(candidate, config.num_layers)))
                except (TypeError, ValueError):
                    pass
    if len(path) < config.max_path_length:
        counts = {layer_id: path.count(layer_id) for layer_id in set(path)}
        for position, layer_id in enumerate(path):
            if counts[layer_id] == 1:
                candidate = path[: position + 1] + (layer_id,) + path[position + 1 :]
                if len(candidate) <= config.max_path_length:
                    successors.add(tuple(validate_path(candidate, config.num_layers)))
    ordered = sorted(successors)
    rng.shuffle(ordered)
    return ordered


def _candidate_answer(review: VerificationResult) -> str | None:
    asserted = [candidate for candidate in review.numeric_candidates if candidate.asserted]
    return asserted[-1].normalized if asserted else None


def _disposition(status: VerificationLabel) -> str:
    if status == VerificationLabel.VERIFIED_CORRECT:
        return "POSITIVE_CANDIDATE"
    if status == VerificationLabel.VERIFIED_INCORRECT:
        return "NEGATIVE_CANDIDATE"
    return "UNLABELED"


def _record(
    question_id: str,
    path: Route,
    response: str,
    review: VerificationResult,
    simulation_index: int,
    num_layers: int,
    reward: float | None = None,
) -> EvaluatedPath:
    labels = path_to_labels(path, num_layers)
    actual_reward = float(review.label == VerificationLabel.VERIFIED_CORRECT) if reward is None else reward
    return EvaluatedPath(
        question_id=question_id,
        execution_path=path,
        action_vector=tuple(int(label) for label in labels),
        raw_model_response=response,
        strict_parser_result=review.strict_parser_result,
        verifier_status=review.label.value,
        verifier_reason=review.reason,
        normalized_candidate_answer=_candidate_answer(review),
        path_length=len(path),
        num_skipped_layers=sum(label == LayerAction.SKIP for label in labels),
        num_repeated_layers=sum(label == LayerAction.REPEAT for label in labels),
        simulation_index=simulation_index,
        mcts_reward=actual_reward,
        training_disposition=_disposition(review.label),
    )


class MCTSSearch:
    def __init__(self, config: MCTSConfig | None = None):
        self.config = config or MCTSConfig()

    def _select(self, root: MCTSNode, rng: random.Random) -> MCTSNode:
        node = root
        while not node.untried_paths and node.children:
            if rng.random() < self.config.random_child_probability:
                node = rng.choice(node.children)
            else:
                node = max(
                    node.children,
                    key=lambda child: (
                        ucb_score(
                            cumulative_reward=child.cumulative_reward,
                            child_visits=child.visits,
                            parent_visits=node.visits,
                            path_length=len(child.path),
                            num_layers=self.config.num_layers,
                            exploration_constant=self.config.exploration_constant,
                            length_penalty=self.config.length_penalty,
                        ),
                        tuple(-part for part in child.path),
                    ),
                )
        return node

    def _expand(self, node: MCTSNode, rng: random.Random) -> MCTSNode:
        if not node.untried_paths:
            return node
        path = node.untried_paths.pop()
        validate_path(path, self.config.num_layers)
        child = MCTSNode(path=path, parent=node)
        child.untried_paths = _successors(path, self.config, rng)
        node.children.append(child)
        return child

    @staticmethod
    def _backpropagate(node: MCTSNode, reward: float) -> None:
        while node is not None:
            node.visits += 1
            node.cumulative_reward += reward
            node = node.parent

    def run(
        self,
        *,
        question_id: str,
        question: str,
        gold_answer: str,
        evaluate_route: RouteEvaluator,
    ) -> QuestionSearchResult:
        rng = random.Random(self.config.seed)
        root_path = tuple(range(self.config.num_layers))
        root = MCTSNode(path=root_path)
        root.untried_paths = _successors(root_path, self.config, rng)
        cache: dict[Route, EvaluatedPath] = {}
        cache_hits = 0
        early_stopped = False

        def evaluate(path: Route, simulation_index: int) -> tuple[EvaluatedPath, bool]:
            nonlocal cache_hits
            if path in cache:
                cache_hits += 1
                return cache[path], True
            validate_path(path, self.config.num_layers)
            response = evaluate_route(list(path))
            review = verify_gsm8k_answer(gold_answer, response)
            record = _record(question_id, path, response, review, simulation_index, self.config.num_layers)
            cache[path] = record
            return record, False

        baseline, _ = evaluate(root_path, 0)
        self._backpropagate(root, baseline.mcts_reward)
        completed = 1
        best_online_length = len(root_path) if baseline.mcts_reward == 1.0 else math.inf

        for simulation_index in range(1, self.config.num_simulations):
            node = self._expand(self._select(root, rng), rng)
            candidate, _cached = evaluate(node.path, simulation_index)
            self._backpropagate(node, candidate.mcts_reward)
            completed += 1
            if candidate.mcts_reward == 1.0 and len(node.path) < best_online_length:
                best_online_length = len(node.path)
            if self.config.early_stop == EarlyStopMode.PAPER and candidate.mcts_reward == 1.0:
                # Reproduce the released script: W->C stops immediately; a correct
                # baseline stops once a correct route saves at least two layers.
                if baseline.mcts_reward == 0.0 or best_online_length + 2 <= self.config.num_layers:
                    early_stopped = True
                    break

        # Mandatory second pass: discard online labels and re-review every raw response.
        reviewed: list[EvaluatedPath] = []
        for candidate in sorted(cache.values(), key=lambda item: item.simulation_index):
            second_review = verify_gsm8k_answer(gold_answer, candidate.raw_model_response)
            reviewed.append(
                _record(
                    question_id,
                    candidate.execution_path,
                    candidate.raw_model_response,
                    second_review,
                    candidate.simulation_index,
                    self.config.num_layers,
                    reward=candidate.mcts_reward,
                )
            )
        by_path = {candidate.execution_path: candidate for candidate in reviewed}
        baseline = by_path[root_path]
        correct = [
            candidate for candidate in reviewed
            if candidate.verifier_status == VerificationLabel.VERIFIED_CORRECT.value
        ]
        if baseline.verifier_status == VerificationLabel.VERIFIED_CORRECT.value:
            eligible = [candidate for candidate in correct if candidate.path_length < self.config.num_layers]
        elif baseline.verifier_status == VerificationLabel.VERIFIED_INCORRECT.value:
            eligible = correct
        else:
            # An uncertain baseline cannot establish preservation or improvement,
            # so this question must not yield router supervision.
            eligible = []
        selected = min(
            eligible,
            key=lambda candidate: (
                candidate.path_length,
                candidate.num_repeated_layers,
                candidate.execution_path,
            ),
            default=None,
        )
        metadata = {
            "algorithm": "length_aware_mcts",
            "num_layers": self.config.num_layers,
            "num_simulations_requested": self.config.num_simulations,
            "num_simulations_completed": completed,
            "unique_paths_evaluated": len(reviewed),
            "evaluation_cache_hits": cache_hits,
            "exploration_constant": self.config.exploration_constant,
            "length_penalty": self.config.length_penalty,
            "random_child_probability": self.config.random_child_probability,
            "max_path_length": self.config.max_path_length,
            "skip_sizes": list(self.config.skip_sizes),
            "repeat_block_size": self.config.repeat_block_size,
            "repeat_count": self.config.repeat_count,
            "seed": self.config.seed,
            "early_stop": self.config.early_stop.value,
            "early_stopped": early_stopped,
            "second_pass_verification": True,
        }
        return QuestionSearchResult(
            question_id=question_id,
            question=question,
            gold_answer=gold_answer,
            baseline=baseline,
            candidates=tuple(reviewed),
            selected_pi_star=selected,
            metadata=metadata,
        )
