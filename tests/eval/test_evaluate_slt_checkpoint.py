from __future__ import annotations

import os
import sys

import pytest


_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from scripts.evaluate_slt_checkpoint import main


def test_checkpoint_gate_returns_nonzero_when_not_ready(monkeypatch, tmp_path):
    checkpoint_dir = tmp_path / "ckpt"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "train_metrics.json").write_text('{"best_val_loss": 1.72}', encoding="utf-8")

    monkeypatch.setattr(
        "scripts.evaluate_slt_checkpoint._build_checkpoint_report",
        lambda **kwargs: {
            "ready": False,
            "failures": ["val chrF below threshold"],
            "overall_metrics": {"chrf": 13.05, "exact_match_pct": 0.0, "n": 25},
        },
    )

    exit_code = main(
        [
            "--checkpoint-dir",
            str(checkpoint_dir),
            "--stage",
            "tsl51",
            "--data-root",
            str(tmp_path / "data"),
        ]
    )

    assert exit_code == 1


def test_checkpoint_gate_returns_zero_when_ready(monkeypatch, tmp_path):
    checkpoint_dir = tmp_path / "ckpt"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "train_metrics.json").write_text('{"best_val_loss": 0.01}', encoding="utf-8")

    monkeypatch.setattr(
        "scripts.evaluate_slt_checkpoint._build_checkpoint_report",
        lambda **kwargs: {
            "ready": True,
            "failures": [],
            "overall_metrics": {"chrf": 100.0, "exact_match_pct": 100.0, "n": 25},
        },
    )

    exit_code = main(
        [
            "--checkpoint-dir",
            str(checkpoint_dir),
            "--stage",
            "tsl51",
            "--data-root",
            str(tmp_path / "data"),
        ]
    )

    assert exit_code == 0
