import json
import pytest
import torch
import torch.nn.functional as F

from src.data.gsm8k import build_generation_prompt
from src.router_training.dataset import load_router_dataset
from src.router_training.loss import focal_loss
from src.router_training.stats import count_labels, effective_number_weights


FIXTURE = "tests/fixtures/router_training_small.jsonl"


def record(question="A unique question", labels=None):
    return {
        "id": "gsm8k-train-test",
        "question": question,
        "answer": "SECRET_PROVENANCE_ANSWER",
        "optimal_layer_config": labels if labels is not None else [1] * 28,
    }


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_valid_sample_and_small_fixture_are_accepted(tmp_path):
    path = tmp_path / "one.jsonl"
    write_jsonl(path, [record()])
    assert len(load_router_dataset(path, allow_small_dataset=True)[0].optimal_layer_config) == 28
    assert len(load_router_dataset(FIXTURE, allow_small_dataset=True)) == 8


@pytest.mark.parametrize("bad", [None, "UNCERTAIN", -1, 3, True])
def test_invalid_labels_are_rejected(tmp_path, bad):
    path = tmp_path / "bad.jsonl"
    labels = [1] * 28
    labels[4] = bad
    write_jsonl(path, [record(labels=labels)])
    with pytest.raises(ValueError, match="label at layer 4"):
        load_router_dataset(path, allow_small_dataset=True)


def test_duplicate_questions_are_rejected_before_production_size_check(tmp_path):
    path = tmp_path / "duplicate.jsonl"
    first, second = record(), record()
    second["id"] = "gsm8k-train-test-2"
    write_jsonl(path, [first, second])
    with pytest.raises(ValueError, match="duplicate questions"):
        load_router_dataset(path)


def test_counts_and_weights():
    dataset = load_router_dataset(FIXTURE, allow_small_dataset=True)
    counts = count_labels(dataset)
    assert counts == {"n_skip": 15, "n_execute": 194, "n_repeat": 15, "total": 224}

    alpha = effective_number_weights(counts)
    expected_raw = [(1 - 0.999) / (1 - 0.999**n) for n in (15, 194, 15)]
    expected = [value / (sum(expected_raw) / 3) for value in expected_raw]
    assert alpha == pytest.approx(expected)
    assert sum(alpha) / 3 == pytest.approx(1.0)
    assert alpha[0] > alpha[1]


def test_gamma_zero_is_weighted_cross_entropy():
    torch.manual_seed(1)
    logits = torch.randn(2, 28, 3)
    targets = torch.randint(0, 3, (2, 28))
    alpha = torch.tensor([1.5, 0.5, 2.0])
    actual = focal_loss(logits, targets, alpha, gamma=0)
    expected = F.cross_entropy(logits.reshape(-1, 3), targets.reshape(-1), weight=alpha)
    # The contract averages weighted per-position losses (rather than CE's weight-normalized mean).
    expected = expected * alpha[targets].mean()
    assert actual == pytest.approx(expected)


def test_confidence_lowers_loss_and_gradient_flows():
    targets = torch.ones((1, 28), dtype=torch.long)
    weak = torch.zeros((1, 28, 3), requires_grad=True)
    confident = torch.zeros((1, 28, 3))
    confident[..., 1] = 8
    assert focal_loss(confident, targets, [1, 1, 1]) < focal_loss(weak, targets, [1, 1, 1])
    loss = focal_loss(weak, targets, [1, 1, 1])
    loss.backward()
    assert weak.grad is not None
    assert torch.isfinite(weak.grad).all()
    assert weak.grad.abs().sum() > 0


def test_answer_never_enters_router_input(tmp_path):
    path = tmp_path / "answer.jsonl"
    write_jsonl(path, [record(question="Only this question may be used")])
    loaded = load_router_dataset(path, allow_small_dataset=True)[0]
    assert loaded.router_input == build_generation_prompt(loaded.question)
    assert loaded.answer not in loaded.router_input
