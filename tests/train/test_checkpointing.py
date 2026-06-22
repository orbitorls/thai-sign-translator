"""Tests for src/tsl/train/checkpointing.py.

Uses tiny toy models (nn.Linear(2, 2)) so tests are fast and dependency-free.
"""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.optim as optim

from tsl.train.checkpointing import (
    find_best_checkpoint,
    find_latest_checkpoint,
    load_checkpoint,
    save_checkpoint,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_components():
    """Return a minimal (model, optimizer, scheduler=None, scaler=None) tuple."""
    model = nn.Linear(2, 2)
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    return model, optimizer, None, None


def _make_components_with_scheduler():
    model = nn.Linear(2, 2)
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
    return model, optimizer, scheduler, None


# ---------------------------------------------------------------------------
# Round-trip: save then load restores step, epoch, metrics
# ---------------------------------------------------------------------------

def test_save_load_round_trip(tmp_path):
    model, optimizer, scheduler, scaler = _make_components()
    metrics = {"train_loss": 1.23, "val_chrf": 42.5}

    save_checkpoint(tmp_path, model, optimizer, scheduler, scaler, step=100, epoch=3, metrics=metrics)

    model2, optimizer2, scheduler2, scaler2 = _make_components()
    info = load_checkpoint(
        tmp_path / "ckpt_step00000100.pt",
        model2, optimizer2, scheduler2, scaler2,
    )

    assert info["step"] == 100
    assert info["epoch"] == 3
    assert info["metrics"]["val_chrf"] == pytest.approx(42.5)
    assert info["metrics"]["train_loss"] == pytest.approx(1.23)


def test_load_restores_model_weights(tmp_path):
    model, optimizer, _, _ = _make_components()
    # Set known weights.
    with torch.no_grad():
        model.weight.fill_(7.0)
        model.bias.fill_(3.0)

    save_checkpoint(tmp_path, model, optimizer, None, None, step=1, epoch=0, metrics={})

    model2 = nn.Linear(2, 2)
    optimizer2 = optim.SGD(model2.parameters(), lr=0.01)
    load_checkpoint(tmp_path / "ckpt_step00000001.pt", model2, optimizer2, None, None)

    assert torch.allclose(model2.weight, torch.full((2, 2), 7.0))
    assert torch.allclose(model2.bias, torch.full((2,), 3.0))


# ---------------------------------------------------------------------------
# RNG state restoration
# ---------------------------------------------------------------------------

def test_rng_states_restored(tmp_path):
    """After load_checkpoint, torch RNG produces the same sequence as right after save."""
    model, optimizer, _, _ = _make_components()

    # Fix a known RNG state before saving.
    torch.manual_seed(42)
    save_checkpoint(tmp_path, model, optimizer, None, None, step=5, epoch=0, metrics={})

    # Capture the "golden" sequence right after that manual_seed.
    torch.manual_seed(42)
    golden = torch.rand(8)

    # Scramble the RNG.
    torch.manual_seed(999)
    _ = torch.rand(100)

    # Restore via load_checkpoint.
    model2, optimizer2, _, _ = _make_components()
    load_checkpoint(tmp_path / "ckpt_step00000005.pt", model2, optimizer2, None, None)

    restored = torch.rand(8)
    assert torch.allclose(golden, restored), "Torch RNG state not correctly restored"


def test_python_rng_restored(tmp_path):
    model, optimizer, _, _ = _make_components()

    random.seed(1234)
    save_checkpoint(tmp_path, model, optimizer, None, None, step=10, epoch=0, metrics={})

    random.seed(1234)
    golden = [random.random() for _ in range(10)]

    random.seed(9999)
    _ = [random.random() for _ in range(100)]

    model2, optimizer2, _, _ = _make_components()
    load_checkpoint(tmp_path / "ckpt_step00000010.pt", model2, optimizer2, None, None)

    restored = [random.random() for _ in range(10)]
    assert golden == restored, "Python RNG state not correctly restored"


def test_numpy_rng_restored(tmp_path):
    model, optimizer, _, _ = _make_components()

    np.random.seed(777)
    save_checkpoint(tmp_path, model, optimizer, None, None, step=20, epoch=0, metrics={})

    np.random.seed(777)
    golden = np.random.rand(10).tolist()

    np.random.seed(5555)
    _ = np.random.rand(100)

    model2, optimizer2, _, _ = _make_components()
    load_checkpoint(tmp_path / "ckpt_step00000020.pt", model2, optimizer2, None, None)

    restored = np.random.rand(10).tolist()
    assert golden == pytest.approx(restored), "NumPy RNG state not correctly restored"


# ---------------------------------------------------------------------------
# Atomic write: temp file cleaned up after successful save
# ---------------------------------------------------------------------------

def test_atomic_write_no_leftover_tmp(tmp_path):
    model, optimizer, _, _ = _make_components()
    save_checkpoint(tmp_path, model, optimizer, None, None, step=50, epoch=1, metrics={})

    tmp_files = list(tmp_path.glob("*.tmp"))
    assert tmp_files == [], f"Leftover temp files found: {tmp_files}"


def test_atomic_write_final_file_exists(tmp_path):
    model, optimizer, _, _ = _make_components()
    path = save_checkpoint(tmp_path, model, optimizer, None, None, step=50, epoch=1, metrics={})

    assert path.exists()
    assert path.name == "ckpt_step00000050.pt"


# ---------------------------------------------------------------------------
# keep_last_k pruning
# ---------------------------------------------------------------------------

def test_keep_last_k_limits_checkpoint_count(tmp_path):
    model, optimizer, _, _ = _make_components()
    # Save 5 checkpoints with no val_chrf so the best-ckpt logic doesn't pin any.
    for step in range(1, 6):
        save_checkpoint(
            tmp_path, model, optimizer, None, None,
            step=step, epoch=0, metrics={}, keep_last_k=3,
        )

    remaining = sorted(tmp_path.glob("ckpt_step*.pt"))
    # Should keep exactly 3 (steps 3, 4, 5 — the last 3).
    assert len(remaining) == 3
    steps = [int(p.stem.split("step")[1]) for p in remaining]
    assert steps == [3, 4, 5]


def test_keep_last_k_retains_best_checkpoint(tmp_path):
    """The checkpoint with the highest val_chrf is kept even if it's not in last-k."""
    model, optimizer, _, _ = _make_components()

    # Step 1 has the best val_chrf.
    save_checkpoint(
        tmp_path, model, optimizer, None, None,
        step=1, epoch=0, metrics={"val_chrf": 99.0}, keep_last_k=2,
    )
    # Steps 2, 3, 4 have lower scores.
    for step in range(2, 5):
        save_checkpoint(
            tmp_path, model, optimizer, None, None,
            step=step, epoch=0, metrics={"val_chrf": float(step)}, keep_last_k=2,
        )

    remaining = sorted(tmp_path.glob("ckpt_step*.pt"))
    steps = [int(p.stem.split("step")[1]) for p in remaining]
    # Must include step 1 (best) and the last 2 (steps 3 and 4).
    assert 1 in steps, "Best checkpoint was pruned"
    assert 3 in steps
    assert 4 in steps


def test_keep_last_k_1_plus_best(tmp_path):
    """keep_last_k=1: only latest + best remain (2 files if they differ)."""
    model, optimizer, _, _ = _make_components()

    save_checkpoint(tmp_path, model, optimizer, None, None,
                    step=10, epoch=0, metrics={"val_chrf": 50.0}, keep_last_k=1)
    save_checkpoint(tmp_path, model, optimizer, None, None,
                    step=20, epoch=0, metrics={"val_chrf": 30.0}, keep_last_k=1)
    save_checkpoint(tmp_path, model, optimizer, None, None,
                    step=30, epoch=0, metrics={"val_chrf": 10.0}, keep_last_k=1)

    remaining = sorted(tmp_path.glob("ckpt_step*.pt"))
    steps = [int(p.stem.split("step")[1]) for p in remaining]
    assert 10 in steps, "Best checkpoint (step 10) was pruned"
    assert 30 in steps, "Latest checkpoint (step 30) was pruned"
    assert len(remaining) == 2


# ---------------------------------------------------------------------------
# find_latest_checkpoint
# ---------------------------------------------------------------------------

def test_find_latest_returns_highest_step(tmp_path):
    model, optimizer, _, _ = _make_components()
    for step in [5, 20, 3]:
        save_checkpoint(tmp_path, model, optimizer, None, None,
                        step=step, epoch=0, metrics={}, keep_last_k=10)

    latest = find_latest_checkpoint(tmp_path)
    assert latest is not None
    assert Path(latest).name == "ckpt_step00000020.pt"


def test_find_latest_returns_none_for_empty_dir(tmp_path):
    assert find_latest_checkpoint(tmp_path) is None


def test_find_latest_returns_none_for_nonexistent_dir(tmp_path):
    assert find_latest_checkpoint(tmp_path / "does_not_exist") is None


# ---------------------------------------------------------------------------
# find_best_checkpoint
# ---------------------------------------------------------------------------

def test_find_best_returns_highest_val_chrf(tmp_path):
    model, optimizer, _, _ = _make_components()
    scores = {1: 10.0, 2: 75.5, 3: 40.0}
    for step, score in scores.items():
        save_checkpoint(tmp_path, model, optimizer, None, None,
                        step=step, epoch=0, metrics={"val_chrf": score}, keep_last_k=10)

    best = find_best_checkpoint(tmp_path)
    assert best is not None
    assert Path(best).name == "ckpt_step00000002.pt"


def test_find_best_returns_none_for_empty_dir(tmp_path):
    assert find_best_checkpoint(tmp_path) is None


def test_find_best_returns_none_for_nonexistent_dir(tmp_path):
    assert find_best_checkpoint(tmp_path / "no_such_dir") is None


def test_find_best_returns_none_when_metric_absent(tmp_path):
    """No checkpoint has the requested metric → None."""
    model, optimizer, _, _ = _make_components()
    save_checkpoint(tmp_path, model, optimizer, None, None,
                    step=1, epoch=0, metrics={"train_loss": 0.5}, keep_last_k=10)

    assert find_best_checkpoint(tmp_path, metric="val_chrf") is None


def test_find_best_custom_metric(tmp_path):
    model, optimizer, _, _ = _make_components()
    for step, bleu in [(1, 5.0), (2, 20.0), (3, 15.0)]:
        save_checkpoint(tmp_path, model, optimizer, None, None,
                        step=step, epoch=0, metrics={"bleu": bleu}, keep_last_k=10)

    best = find_best_checkpoint(tmp_path, metric="bleu")
    assert best is not None
    assert Path(best).name == "ckpt_step00000002.pt"


def test_save_checkpoint_updates_latest_and_best_sidecars(tmp_path):
    model, optimizer, _, _ = _make_components()

    save_checkpoint(
        tmp_path, model, optimizer, None, None,
        step=100, epoch=0, metrics={"val_chrf": 11.0}, keep_last_k=2,
    )
    save_checkpoint(
        tmp_path, model, optimizer, None, None,
        step=200, epoch=0, metrics={"val_chrf": 9.0}, keep_last_k=2,
    )
    save_checkpoint(
        tmp_path, model, optimizer, None, None,
        step=300, epoch=0, metrics={"val_chrf": 12.5}, keep_last_k=2,
    )

    assert (tmp_path / "latest_checkpoint.txt").read_text(encoding="utf-8") == "ckpt_step00000300.pt"
    assert (tmp_path / "best_checkpoint.txt").read_text(encoding="utf-8") == "ckpt_step00000300.pt"


def test_save_checkpoint_keeps_best_sidecar_when_best_is_older(tmp_path):
    model, optimizer, _, _ = _make_components()

    save_checkpoint(
        tmp_path, model, optimizer, None, None,
        step=100, epoch=0, metrics={"val_chrf": 20.0}, keep_last_k=1,
    )
    save_checkpoint(
        tmp_path, model, optimizer, None, None,
        step=200, epoch=0, metrics={"val_chrf": 10.0}, keep_last_k=1,
    )

    assert (tmp_path / "latest_checkpoint.txt").read_text(encoding="utf-8") == "ckpt_step00000200.pt"
    assert (tmp_path / "best_checkpoint.txt").read_text(encoding="utf-8") == "ckpt_step00000100.pt"


# ---------------------------------------------------------------------------
# Scheduler state saved and restored
# ---------------------------------------------------------------------------

def test_scheduler_state_round_trip(tmp_path):
    model, optimizer, scheduler, _ = _make_components_with_scheduler()
    # Advance scheduler so its state is non-trivial.
    scheduler.step()
    scheduler.step()
    lr_before = optimizer.param_groups[0]["lr"]

    save_checkpoint(tmp_path, model, optimizer, scheduler, None,
                    step=7, epoch=2, metrics={})

    model2, optimizer2, scheduler2, _ = _make_components_with_scheduler()
    load_checkpoint(tmp_path / "ckpt_step00000007.pt", model2, optimizer2, scheduler2, None)

    lr_after = optimizer2.param_groups[0]["lr"]
    assert lr_after == pytest.approx(lr_before)


# ---------------------------------------------------------------------------
# Checkpoint dir created automatically
# ---------------------------------------------------------------------------

def test_checkpoint_dir_created_automatically(tmp_path):
    deep_dir = tmp_path / "a" / "b" / "c"
    model, optimizer, _, _ = _make_components()
    save_checkpoint(deep_dir, model, optimizer, None, None, step=1, epoch=0, metrics={})
    assert deep_dir.exists()
    assert (deep_dir / "ckpt_step00000001.pt").exists()
