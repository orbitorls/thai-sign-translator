"""Tests for train_pose_t5.py — multi-stage Kaggle-resumable training script.

All tests use tiny synthetic data (numpy .npy files + manifest.csv) and a
monkeypatched tiny mT5 config so no network downloads are needed.
"""
from __future__ import annotations

import argparse
import csv
import os
import tempfile
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from transformers import MT5Config, MT5ForConditionalGeneration


# ---------------------------------------------------------------------------
# Tiny mT5 config (shared across tests)
# ---------------------------------------------------------------------------

TINY_CONFIG = MT5Config(
    d_model=32,
    num_heads=2,
    num_layers=1,
    d_ff=64,
    d_kv=16,
    vocab_size=250112,
    decoder_start_token_id=0,
    eos_token_id=1,
    pad_token_id=0,
)


def _tiny_mt5_factory(*args, **kwargs) -> MT5ForConditionalGeneration:
    """Return a tiny MT5 model without downloading anything."""
    return MT5ForConditionalGeneration(TINY_CONFIG)


# ---------------------------------------------------------------------------
# Fake HF Tokenizer
# ---------------------------------------------------------------------------

class _FakeHFTokenizer:
    """Minimal HF-tokenizer stand-in that tokenizes character-by-character."""

    pad_token_id = 0

    def __call__(self, texts, padding=True, return_tensors="pt"):
        if isinstance(texts, str):
            texts = [texts]
        rows = []
        for t in texts:
            # Use ord() % 1000 + 1 so we never produce pad_token_id=0
            rows.append([ord(c) % 1000 + 1 for c in t])
        if padding:
            max_len = max(len(r) for r in rows) if rows else 1
            rows = [r + [self.pad_token_id] * (max_len - len(r)) for r in rows]
        return {"input_ids": torch.tensor(rows, dtype=torch.long)}

    def decode(self, token_ids, skip_special_tokens=True):
        """Decode token ids back to a string (best-effort)."""
        chars = []
        for tid in token_ids:
            tid = int(tid)
            if tid == self.pad_token_id or tid <= 0:
                continue
            # Reverse: ord(c) % 1000 + 1 → character
            chars.append(chr((tid - 1) % 128))
        return "".join(chars)


# ---------------------------------------------------------------------------
# Synthetic data fixture
# ---------------------------------------------------------------------------

_FEAT_DIM = 312
_N_EXAMPLES = 6  # enough for a non-empty val split with 70/30 fracs


@pytest.fixture()
def synthetic_data_dir(tmp_path):
    """Create a synthetic manifest + .npy feature files under tmp_path.

    Returns the data root directory (str).
    """
    npy_dir = tmp_path / "features"
    npy_dir.mkdir()

    rows = []
    for i in range(_N_EXAMPLES):
        n_frames = 16 + i * 4  # different lengths: 16, 20, 24, 28, 32, 36
        arr = np.random.randn(n_frames, _FEAT_DIM).astype(np.float32)
        npy_path = npy_dir / f"seg_{i:03d}.npy"
        np.save(str(npy_path), arr)
        rows.append(
            {
                "segment_id": f"seg_{i:03d}",
                "npy_path": str(npy_path),
                "text": f"ข้อความที่ {i}",
                "video_id": f"vid_{i // 2}",  # 2 examples per video → non-empty val
                "split": "train",
                "source": "test_synth",
                "feature_layout_version": "v3-312",
            }
        )

    manifest_path = tmp_path / "manifest.csv"
    with open(manifest_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "segment_id",
                "npy_path",
                "text",
                "video_id",
                "split",
                "source",
                "feature_layout_version",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    return str(tmp_path)


@pytest.fixture()
def patch_mt5(monkeypatch):
    """Patch MT5ForConditionalGeneration.from_pretrained in pose_t5 module."""
    monkeypatch.setattr(
        "tsl.models.pose_t5.MT5ForConditionalGeneration.from_pretrained",
        _tiny_mt5_factory,
    )


@pytest.fixture()
def patch_tokenizer(monkeypatch):
    """Patch AutoTokenizer.from_pretrained in train_pose_t5 module."""
    monkeypatch.setattr(
        "tsl.train.train_pose_t5.AutoTokenizer.from_pretrained",
        lambda *a, **kw: _FakeHFTokenizer(),
    )


@pytest.fixture()
def base_args(synthetic_data_dir, tmp_path):
    """Base argparse.Namespace for training tests."""
    return argparse.Namespace(
        data_roots=synthetic_data_dir,
        out_dir=str(tmp_path / "ckpts"),
        base_model="google/mt5-small",
        epochs=1,
        batch_size=2,
        grad_accum=1,
        lr=1e-3,
        max_src_len=32,
        downsample_factor=2,
        num_encoder_layers=1,
        amp="false",
        resume="none",
        max_runtime_min=690,
        keep_checkpoints=3,
        seed=42,
        eval_steps=3,  # evaluate often so tests hit it within 3 steps
    )


# ---------------------------------------------------------------------------
# Helper: build a pre-populated checkpoint in out_dir
# ---------------------------------------------------------------------------

def _make_checkpoint(
    out_dir: str,
    model,
    step: int,
    monkeypatch,
    patch_mt5,
    patch_tokenizer,
) -> str:
    """Save a checkpoint at the given step and return its path."""
    from tsl.train.checkpointing import save_checkpoint

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min")
    path = save_checkpoint(
        out_dir,
        model,
        optimizer,
        scheduler,
        scaler=None,
        step=step,
        epoch=0,
        metrics={"val_loss": 5.0, "val_chrf": 0.0},
        keep_last_k=3,
    )
    return str(path)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestTrainRunsWithoutError:
    """Training can run 3 optimizer steps on tiny synthetic data."""

    def test_three_steps_no_error(
        self, patch_mt5, patch_tokenizer, base_args, tmp_path
    ):
        from tsl.train.train_pose_t5 import main

        # Set eval_steps=3 so we hit exactly one evaluation
        base_args.eval_steps = 3
        base_args.epochs = 2  # run 2 epochs to accumulate enough steps

        metrics = main(base_args)

        assert isinstance(metrics, dict)
        assert "global_step" in metrics
        assert metrics["global_step"] >= 0

    def test_output_dir_created(self, patch_mt5, patch_tokenizer, base_args):
        from tsl.train.train_pose_t5 import main

        main(base_args)

        assert os.path.isdir(base_args.out_dir)

    def test_metrics_json_written(self, patch_mt5, patch_tokenizer, base_args):
        from tsl.train.train_pose_t5 import main

        base_args.eval_steps = 3
        base_args.epochs = 2

        main(base_args)

        metrics_path = Path(base_args.out_dir) / "train_metrics.json"
        assert metrics_path.exists(), "train_metrics.json should be written"

        import json
        with open(metrics_path) as fh:
            data = json.load(fh)
        assert "global_step" in data
        assert "history" in data


class TestResumeFromCheckpoint:
    """Training resumes at the correct step when given a checkpoint."""

    def test_resume_restores_step(
        self, patch_mt5, patch_tokenizer, base_args, tmp_path
    ):
        from tsl.models.pose_t5 import PoseToTextT5
        from tsl.train.checkpointing import save_checkpoint, find_latest_checkpoint

        out_dir = str(tmp_path / "ckpts_resume")
        os.makedirs(out_dir, exist_ok=True)

        # Build a tiny model and save a checkpoint at step=7
        model = PoseToTextT5(
            input_dim=_FEAT_DIM,
            num_encoder_layers=1,
            downsample_factor=2,
            base_model_name="google/mt5-small",
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

        save_checkpoint(
            out_dir,
            model,
            optimizer,
            scheduler,
            scaler=None,
            step=7,
            epoch=0,
            metrics={"val_loss": 2.0, "val_chrf": 0.0},
            keep_last_k=3,
        )

        # Confirm find_latest_checkpoint sees it
        latest = find_latest_checkpoint(out_dir)
        assert latest is not None

        # Now train with resume=auto
        base_args.out_dir = out_dir
        base_args.resume = "auto"
        base_args.epochs = 1
        base_args.eval_steps = 9999  # never trigger eval, just check resume works

        from tsl.train.train_pose_t5 import main

        result = main(base_args)

        # After resuming from step=7, global_step should be >= 7
        assert result["global_step"] >= 7, (
            f"Expected global_step >= 7 after resuming, got {result['global_step']}"
        )

    def test_checkpoint_file_exists_after_resume(
        self, patch_mt5, patch_tokenizer, base_args, tmp_path
    ):
        from tsl.models.pose_t5 import PoseToTextT5
        from tsl.train.checkpointing import save_checkpoint, find_latest_checkpoint

        out_dir = str(tmp_path / "ckpts_resume2")
        os.makedirs(out_dir, exist_ok=True)

        model = PoseToTextT5(
            input_dim=_FEAT_DIM,
            num_encoder_layers=1,
            downsample_factor=2,
            base_model_name="google/mt5-small",
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

        save_checkpoint(
            out_dir, model, optimizer, scheduler,
            scaler=None, step=5, epoch=0,
            metrics={"val_loss": 3.0, "val_chrf": 0.0},
            keep_last_k=3,
        )

        base_args.out_dir = out_dir
        base_args.resume = "auto"

        from tsl.train.train_pose_t5 import main
        main(base_args)

        ckpts = list(Path(out_dir).glob("ckpt_step*.pt"))
        assert len(ckpts) >= 1, "At least one checkpoint should exist after training"


class TestMaxRuntimeExit:
    """max_runtime_min triggers clean exit with checkpoint saved."""

    def test_clean_exit_saves_checkpoint(
        self, patch_mt5, patch_tokenizer, base_args, tmp_path
    ):
        from tsl.train.train_pose_t5 import main

        out_dir = str(tmp_path / "ckpts_timeout")
        base_args.out_dir = out_dir
        base_args.max_runtime_min = 0  # immediate timeout
        base_args.epochs = 100  # would run forever without timeout
        base_args.eval_steps = 9999

        result = main(base_args)

        # Should exit cleanly with stopped_reason = "max_runtime"
        assert result.get("stopped_reason") == "max_runtime", (
            f"Expected 'max_runtime', got {result.get('stopped_reason')!r}"
        )

        # A checkpoint should have been saved
        ckpts = list(Path(out_dir).glob("ckpt_step*.pt"))
        assert len(ckpts) >= 1, "Checkpoint should be saved on timeout exit"

    def test_clean_exit_writes_metrics_json(
        self, patch_mt5, patch_tokenizer, base_args, tmp_path
    ):
        from tsl.train.train_pose_t5 import main

        out_dir = str(tmp_path / "ckpts_timeout2")
        base_args.out_dir = out_dir
        base_args.max_runtime_min = 0
        base_args.epochs = 100

        main(base_args)

        metrics_path = Path(out_dir) / "train_metrics.json"
        assert metrics_path.exists(), "train_metrics.json should be written on timeout"


class TestValLossComputation:
    """Val loss is computed correctly (finite, positive)."""

    def test_val_loss_is_finite(
        self, patch_mt5, patch_tokenizer, base_args, synthetic_data_dir, tmp_path
    ):
        from tsl.models.pose_t5 import PoseToTextT5
        from tsl.data.unified import load_manifest
        from tsl.eval.build_splits import split_by_video
        from tsl.train.train_pose_t5 import _compute_val_loss

        model = PoseToTextT5(
            input_dim=_FEAT_DIM,
            num_encoder_layers=1,
            downsample_factor=2,
            base_model_name="google/mt5-small",
        )
        device = torch.device("cpu")

        examples = load_manifest(synthetic_data_dir)
        splits = split_by_video(examples, {"train": 0.7, "val": 0.3}, seed=42)
        val_examples = splits.get("val", [])

        if not val_examples:
            pytest.skip("val split is empty for this seed; skip val_loss test")

        hf_tokenizer = _FakeHFTokenizer()

        val_loss = _compute_val_loss(
            model,
            val_examples,
            hf_tokenizer,
            batch_size=2,
            max_src_len=32,
            device=device,
            use_amp=False,
        )

        assert math.isfinite(val_loss), f"val_loss should be finite, got {val_loss}"
        assert val_loss > 0.0, "val_loss should be positive"

    def test_val_loss_empty_returns_inf(self, patch_mt5):
        from tsl.models.pose_t5 import PoseToTextT5
        from tsl.train.train_pose_t5 import _compute_val_loss

        model = PoseToTextT5(
            input_dim=_FEAT_DIM,
            num_encoder_layers=1,
            downsample_factor=2,
            base_model_name="google/mt5-small",
        )
        hf_tokenizer = _FakeHFTokenizer()

        val_loss = _compute_val_loss(
            model,
            [],  # empty val set
            hf_tokenizer,
            batch_size=2,
            max_src_len=32,
            device=torch.device("cpu"),
            use_amp=False,
        )

        assert val_loss == float("inf"), (
            f"Empty val set should return inf, got {val_loss}"
        )


class TestChrFSkippedWhenSacrebleuMissing:
    """chrF computation gracefully returns 0.0 if sacrebleu not importable."""

    def test_chrf_fallback(self, patch_mt5, monkeypatch):
        import sys

        # Remove sacrebleu from sys.modules to simulate it being absent
        sacrebleu_backup = sys.modules.get("sacrebleu")
        sys.modules["sacrebleu"] = None  # type: ignore[assignment]

        try:
            from tsl.models.pose_t5 import PoseToTextT5
            from tsl.train.train_pose_t5 import _compute_val_chrf

            model = PoseToTextT5(
                input_dim=_FEAT_DIM,
                num_encoder_layers=1,
                downsample_factor=2,
                base_model_name="google/mt5-small",
            )
            score = _compute_val_chrf(
                model,
                [],
                _FakeHFTokenizer(),
                max_src_len=32,
                device=torch.device("cpu"),
            )
            assert score == 0.0
        finally:
            if sacrebleu_backup is not None:
                sys.modules["sacrebleu"] = sacrebleu_backup
            else:
                del sys.modules["sacrebleu"]


class TestMainCallable:
    """main(args) can be called from tests (not just from CLI)."""

    def test_main_returns_dict(self, patch_mt5, patch_tokenizer, base_args):
        from tsl.train.train_pose_t5 import main

        result = main(base_args)
        assert isinstance(result, dict)

    def test_module_importable(self):
        """train_pose_t5 must be importable without side-effects."""
        import importlib

        mod = importlib.import_module("tsl.train.train_pose_t5")
        assert hasattr(mod, "main")
        assert hasattr(mod, "_build_parser")
        assert callable(mod.main)


# ---------------------------------------------------------------------------
# Import for math module (used in TestValLossComputation)
# ---------------------------------------------------------------------------
import math
