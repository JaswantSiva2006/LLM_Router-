from pathlib import Path

import pytest
import torch

from src.router_training.smoke_train import main as smoke_main
from src.router_training.verify_readiness import (
    CHECK_NAMES,
    check_focal_loss,
    check_window_pooling,
    inspect_dataset,
    main as readiness_main,
)


FIXTURE = Path("tests/fixtures/router_training_small.jsonl")


def test_fixture_readiness_checks_are_truthful_without_requiring_4k():
    checks, records = inspect_dataset(FIXTURE)
    assert len(records) == 8
    assert checks["dataset exists"]
    assert not checks["sample count = 4000"]
    assert checks["unique ids"]
    assert checks["unique questions"]
    assert not checks["112000 router labels"]
    assert checks["class counts"]
    assert checks["effective-number weights"]


def test_missing_dataset_is_reported_without_model_load(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(
        "src.router_training.verify_readiness.check_existing_mcts_tests", lambda: True
    )
    monkeypatch.setattr(
        "src.router_training.verify_readiness.model_checks",
        lambda *_: pytest.fail("model must not load when the dataset is absent"),
    )
    result = readiness_main([
        "--data", str(tmp_path / "not-generated.jsonl"),
        "--model", "Qwen/Qwen2.5-1.5B-Instruct", "--precision", "fp16",
    ])
    output = capsys.readouterr().out
    assert result == 1
    assert "ROUTER TRAINING READY: NO" in output
    assert "dataset exists: FAIL" in output
    assert all(f"{name}:" in output for name in CHECK_NAMES)


def test_readiness_math_self_checks():
    assert check_focal_loss()
    assert check_window_pooling()


@pytest.mark.skipif(torch.cuda.is_available(), reason="only exercises the no-CUDA diagnostic")
def test_smoke_train_fails_cleanly_without_cuda():
    with pytest.raises(SystemExit, match="2"):
        smoke_main([
            "--data", str(FIXTURE), "--allow-small-dataset",
            "--model", "Qwen/Qwen2.5-1.5B-Instruct", "--samples", "8",
            "--precision", "fp16",
        ])
