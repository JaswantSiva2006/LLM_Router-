from dataclasses import dataclass

import pytest
import torch
from torch import nn
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, TensorDataset

from src.router_training.train import (
    RouterMetrics,
    _run_epoch,
    build_optimizer,
    deterministic_split_indices,
    global_effective_batch_size,
    load_training_state,
    make_distributed_sampler,
    save_training_state,
)


class TinyTrainModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_model = nn.Linear(2, 2)
        self.base_model.requires_grad_(False)
        self.routers = nn.ModuleList([nn.Linear(2, 3) for _ in range(28)])

    def forward(self, input_ids, router_labels, use_cache=False):
        features = input_ids.float()
        logits = torch.stack([router(features) for router in self.routers], dim=1)
        return type("Output", (), {"router_logits": logits})()


def loader(sample_count=5):
    inputs = torch.arange(sample_count * 2).reshape(sample_count, 2)
    labels = torch.ones((sample_count, 28), dtype=torch.long)
    dataset = TensorDataset(inputs, labels)
    return DataLoader(
        dataset,
        batch_size=1,
        collate_fn=lambda rows: {"input_ids": rows[0][0].unsqueeze(0), "router_labels": rows[0][1].unsqueeze(0)},
    )


class CountingScheduler:
    def __init__(self):
        self.steps = 0

    def step(self):
        self.steps += 1

    def state_dict(self):
        return {"steps": self.steps}

    def load_state_dict(self, state):
        self.steps = state["steps"]


def test_effective_batch_and_deterministic_split():
    assert global_effective_batch_size(1, 16, 1) == 16
    assert global_effective_batch_size(1, 8, 2) == 16
    first = deterministic_split_indices(4000, 0.1, 42)
    second = deterministic_split_indices(4000, 0.1, 42)
    assert first == second
    assert len(first[0]) == 3600 and len(first[1]) == 400
    assert set(first[0]).isdisjoint(first[1])


def test_distributed_sampler_shards_fixture_without_overlap():
    dataset = list(range(8))
    left = list(make_distributed_sampler(dataset, rank=0, world_size=2, shuffle=False, seed=42))
    right = list(make_distributed_sampler(dataset, rank=1, world_size=2, shuffle=False, seed=42))
    assert set(left).isdisjoint(right)
    assert sorted(left + right) == list(range(8))


def test_optimizer_only_contains_routers_and_accumulation_controls_scheduler():
    model = TinyTrainModel()
    optimizer = build_optimizer(model)
    optimized = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    assert optimized == {id(parameter) for parameter in model.routers.parameters()}
    scheduler = CountingScheduler()
    alpha = torch.tensor([1.0, 1.0, 1.0])
    original_alpha = alpha.clone()
    _, updates = _run_epoch(
        model, loader(5), alpha, 2.0, torch.device("cpu"), "fp32",
        optimizer=optimizer, scheduler=scheduler, accumulation=2,
    )
    assert updates == 3
    assert scheduler.steps == updates
    torch.testing.assert_close(alpha, original_alpha)


def test_checkpoint_resume_and_rank_zero_only(tmp_path):
    model = TinyTrainModel()
    optimizer = build_optimizer(model)
    scheduler = LambdaLR(optimizer, lambda _: 1.0)
    metadata = {"dataset_sha256": "abc", "class_weights": [1.0, 1.0, 1.0]}
    blocked = tmp_path / "blocked"
    save_training_state(
        blocked, model, optimizer, scheduler, None,
        epoch=1, global_step=3, metadata=metadata, rank=1,
    )
    assert not blocked.exists()

    output = tmp_path / "saved"
    original = {key: value.clone() for key, value in model.routers.state_dict().items()}
    save_training_state(
        output, model, optimizer, scheduler, None,
        epoch=2, global_step=7, metadata=metadata, rank=0,
    )
    with torch.no_grad():
        for parameter in model.routers.parameters():
            parameter.add_(10)
    epoch, step, loaded = load_training_state(
        output / "training_checkpoint.pt", model, optimizer, scheduler, None,
        expected_dataset_sha256="abc",
    )
    assert (epoch, step, loaded) == (2, 7, metadata)
    for key, value in model.routers.state_dict().items():
        torch.testing.assert_close(value, original[key])


def test_metrics_and_fake_distributed_aggregation(monkeypatch):
    metrics = RouterMetrics()
    targets = torch.tensor([[0, 1] * 14])
    logits = torch.full((1, 28, 3), -5.0)
    logits.scatter_(2, targets.unsqueeze(-1), 5.0)
    metrics.update(torch.tensor(0.25), logits, targets)
    computed = metrics.compute()
    assert computed["overall_layer_accuracy"] == 1.0
    assert computed["exact_path_accuracy"] == 1.0
    assert computed["mean_predicted_executed_layers"] == 14
    assert len(computed["per_layer_accuracy"]) == 28

    calls = []
    monkeypatch.setattr("src.router_training.train.dist.is_initialized", lambda: True)
    monkeypatch.setattr("src.router_training.train.dist.all_reduce", lambda tensor, op: calls.append(tensor))
    metrics.reduce()
    assert len(calls) == 7
