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

    def save_pretrained(self, output_dir):
        path = Path(output_dir) / "tokenizer_config.json"
        path.write_text("{}", encoding="utf-8")


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
        dropout=0.1,
        amp="false",
        device="auto",
        require_gpu=False,
        max_train_steps=0,
        resume="none",
        reset_progress_history=False,
        max_runtime_min=690,
        keep_checkpoints=3,
        weight_decay=0.01,
        early_stopping_patience=0,
        early_stopping_min_delta=0.0,
        early_stopping_metric="val_loss",
        seed=42,
        eval_steps=3,  # evaluate often so tests hit it within 3 steps
        checkpoint_steps=0,
        preload_train_features=False,
        split_policy="auto",
        focus_target_tokens="",
        focus_target_max_multiplier=1.0,
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


def _write_manifest_dataset(tmp_path, rows: list[dict[str, str]]) -> str:
    npy_dir = tmp_path / "features"
    npy_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows = []
    for idx, row in enumerate(rows):
        arr = np.random.randn(16 + idx, _FEAT_DIM).astype(np.float32)
        npy_path = npy_dir / f"seg_custom_{idx:03d}.npy"
        np.save(str(npy_path), arr)
        manifest_rows.append(
            {
                "segment_id": row["segment_id"],
                "npy_path": str(npy_path),
                "text": row["text"],
                "video_id": row["video_id"],
                "split": row["split"],
                "source": row.get("source", "test_synth"),
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
        writer.writerows(manifest_rows)
    return str(tmp_path)


# ---------------------------------------------------------------------------
# Tests


def test_build_scaler_falls_back_to_torch_cuda_amp(monkeypatch):
    from tsl.train import train_pose_t5 as module

    sentinel = object()

    monkeypatch.setattr(module.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(module, "_amp_dtype", lambda: torch.float16)
    monkeypatch.delattr(module.torch.amp, "GradScaler", raising=False)
    monkeypatch.setattr(module.torch.cuda.amp, "GradScaler", lambda: sentinel)

    assert module._build_scaler(True) is sentinel
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

    def test_run_status_json_written(self, patch_mt5, patch_tokenizer, base_args):
        from tsl.train.train_pose_t5 import main

        base_args.eval_steps = 3
        base_args.epochs = 2

        main(base_args)

        status_path = Path(base_args.out_dir) / "run_status.json"
        assert status_path.exists(), "run_status.json should be written during training"

        import json
        with open(status_path, encoding="utf-8") as fh:
            payload = json.load(fh)
        assert payload["phase"] in {"training", "evaluated", "completed", "early_stopping", "max_runtime"}
        assert payload["global_step"] >= 0
        assert "updated_at_unix" in payload

    def test_manifest_quality_report_written(self, patch_mt5, patch_tokenizer, base_args):
        from tsl.train.train_pose_t5 import main

        base_args.eval_steps = 9999
        main(base_args)

        report_path = Path(base_args.out_dir) / "manifest_quality.json"
        assert report_path.exists(), "manifest_quality.json should be written before training"

        import json
        with open(report_path, encoding="utf-8") as fh:
            report = json.load(fh)

        assert "passed" in report
        assert "overall" in report
        assert "by_source" in report
        assert report["overall"]["train_examples_per_target"] >= 0.0

    def test_fail_on_manifest_quality_aborts_before_training(
        self, patch_mt5, patch_tokenizer, base_args
    ):
        from tsl.train.train_pose_t5 import main

        base_args.required_sources = "tsl51,thaisignvis"
        base_args.fail_on_manifest_quality = "true"

        with pytest.raises(RuntimeError, match="Manifest-quality gates failed before training"):
            main(base_args)

    def test_source_sampling_report_written(self, patch_mt5, patch_tokenizer, base_args):
        from tsl.train.train_pose_t5 import main

        base_args.eval_steps = 9999
        main(base_args)

        report_path = Path(base_args.out_dir) / "source_sampling.json"
        assert report_path.exists(), "source_sampling.json should be written before training"

        import json
        with open(report_path, encoding="utf-8") as fh:
            report = json.load(fh)

        assert "enabled" in report
        assert "source_counts" in report

    def test_streaming_train_features_skips_preload_dataset(
        self, patch_mt5, patch_tokenizer, base_args, monkeypatch
    ):
        from tsl.train import train_pose_t5

        base_args.eval_steps = 9999
        base_args.preload_train_features = False

        def _boom(*args, **kwargs):
            raise AssertionError("preloaded dataset should not be constructed")

        monkeypatch.setattr(train_pose_t5, "_PreloadedDataset", _boom)

        metrics = train_pose_t5.main(base_args)

        assert isinstance(metrics, dict)
        assert metrics["global_step"] >= 0

    def test_tokenizer_files_written_with_final_export(
        self, patch_mt5, patch_tokenizer, base_args
    ):
        from tsl.train.train_pose_t5 import main

        base_args.eval_steps = 9999
        main(base_args)

        tokenizer_config = Path(base_args.out_dir) / "tokenizer_config.json"
        assert tokenizer_config.exists(), "tokenizer_config.json should be exported with the final model"

    def test_auto_split_policy_preserves_manifest_train_val_when_available(
        self, patch_mt5, patch_tokenizer, base_args, tmp_path
    ):
        from tsl.train.train_pose_t5 import main
        import json

        data_root = _write_manifest_dataset(
            tmp_path / "manifest_split_auto",
            [
                {"segment_id": "seg_a", "text": "alpha", "video_id": "vid_a", "split": "train"},
                {"segment_id": "seg_b", "text": "alpha", "video_id": "vid_b", "split": "train"},
                {"segment_id": "seg_c", "text": "alpha", "video_id": "vid_c", "split": "val"},
            ],
        )
        base_args.data_roots = data_root
        base_args.out_dir = str(tmp_path / "ckpts_manifest_auto")
        base_args.eval_steps = 9999
        base_args.epochs = 1
        base_args.split_policy = "auto"

        main(base_args)

        report = json.loads((Path(base_args.out_dir) / "manifest_quality.json").read_text(encoding="utf-8"))
        assert report["overall"]["train_examples"] == 2
        assert report["overall"]["val_examples"] == 1

    def test_manifest_split_policy_uses_manifest_assignments(
        self, patch_mt5, patch_tokenizer, base_args, tmp_path
    ):
        from tsl.train.train_pose_t5 import main
        import json

        data_root = _write_manifest_dataset(
            tmp_path / "manifest_split_forced",
            [
                {"segment_id": "seg_a", "text": "alpha", "video_id": "vid_a", "split": "train"},
                {"segment_id": "seg_b", "text": "alpha", "video_id": "vid_b", "split": "train"},
                {"segment_id": "seg_c", "text": "alpha", "video_id": "vid_c", "split": "val"},
            ],
        )
        base_args.data_roots = data_root
        base_args.out_dir = str(tmp_path / "ckpts_manifest_forced")
        base_args.eval_steps = 9999
        base_args.epochs = 1
        base_args.split_policy = "manifest"

        main(base_args)

        report = json.loads((Path(base_args.out_dir) / "manifest_quality.json").read_text(encoding="utf-8"))
        assert report["overall"]["train_examples"] == 2
        assert report["overall"]["val_examples"] == 1

    def test_manifest_quality_sources_can_gate_subset_of_sources(
        self, patch_mt5, patch_tokenizer, base_args, tmp_path
    ):
        from tsl.train.train_pose_t5 import main
        import json

        data_root = _write_manifest_dataset(
            tmp_path / "manifest_quality_subset",
            [
                {"segment_id": "seg_a", "text": "alpha", "video_id": "vid_a", "split": "train", "source": "tsl51"},
                {"segment_id": "seg_b", "text": "alpha", "video_id": "vid_b", "split": "train", "source": "tsl51"},
                {"segment_id": "seg_c", "text": "alpha", "video_id": "vid_c", "split": "val", "source": "tsl51"},
                {"segment_id": "seg_d", "text": "beta", "video_id": "vid_d", "split": "train", "source": "thaisignvis"},
                {"segment_id": "seg_e", "text": "gamma", "video_id": "vid_e", "split": "val", "source": "thaisignvis"},
            ],
        )
        base_args.data_roots = data_root
        base_args.out_dir = str(tmp_path / "ckpts_manifest_quality_subset")
        base_args.eval_steps = 9999
        base_args.epochs = 1
        base_args.split_policy = "manifest"
        base_args.required_sources = "tsl51"
        base_args.manifest_quality_sources = "tsl51"
        base_args.fail_on_manifest_quality = "true"

        main(base_args)

        report = json.loads((Path(base_args.out_dir) / "manifest_quality.json").read_text(encoding="utf-8"))
        assert report["passed"] is True
        assert report["gated_sources"] == ["tsl51"]


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

    def test_resume_best_prefers_metric_best_checkpoint(
        self, patch_mt5, patch_tokenizer, base_args, tmp_path, monkeypatch
    ):
        from tsl.models.pose_t5 import PoseToTextT5
        from tsl.train.checkpointing import save_checkpoint
        from tsl.train import train_pose_t5

        out_dir = str(tmp_path / "ckpts_resume_best")
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
            scaler=None, step=7, epoch=0,
            metrics={"val_loss": 5.0, "val_chrf": 12.0},
            keep_last_k=3,
        )
        save_checkpoint(
            out_dir, model, optimizer, scheduler,
            scaler=None, step=9, epoch=0,
            metrics={"val_loss": 4.0, "val_chrf": 10.0},
            keep_last_k=3,
        )

        seen: dict[str, str] = {}
        original_load_checkpoint = train_pose_t5.load_checkpoint

        def _recording_load_checkpoint(path, *args, **kwargs):
            seen["path"] = os.fspath(path)
            return original_load_checkpoint(path, *args, **kwargs)

        monkeypatch.setattr(train_pose_t5, "load_checkpoint", _recording_load_checkpoint)

        base_args.out_dir = out_dir
        base_args.resume = "best"
        base_args.epochs = 1
        base_args.eval_steps = 9999
        base_args.early_stopping_metric = "val_chrf"

        train_pose_t5.main(base_args)

        assert seen["path"].endswith("ckpt_step00000007.pt")

    def test_resume_best_state_prefers_best_model_state_without_optimizer_state(
        self, patch_mt5, patch_tokenizer, base_args, tmp_path, monkeypatch
    ):
        from tsl.models.pose_t5 import PoseToTextT5
        from tsl.train import train_pose_t5

        out_dir = str(tmp_path / "ckpts_resume_best_state")
        os.makedirs(out_dir, exist_ok=True)

        model = PoseToTextT5(
            input_dim=_FEAT_DIM,
            num_encoder_layers=1,
            downsample_factor=2,
            base_model_name="google/mt5-small",
        )
        torch.save(
            {
                "step": 11,
                "epoch": 2,
                "metrics": {"val_loss": 3.0, "val_chrf": 16.0},
                "model_state_dict": model.state_dict(),
            },
            Path(out_dir) / "best_model_state.pt",
        )

        seen: dict[str, str] = {}
        original_load_checkpoint = train_pose_t5.load_checkpoint

        def _recording_load_checkpoint(path, *args, **kwargs):
            seen["path"] = os.fspath(path)
            return original_load_checkpoint(path, *args, **kwargs)

        monkeypatch.setattr(train_pose_t5, "load_checkpoint", _recording_load_checkpoint)

        base_args.out_dir = out_dir
        base_args.resume = "best_state"
        base_args.epochs = 1
        base_args.eval_steps = 9999
        base_args.preload_train_features = False

        result = train_pose_t5.main(base_args)

        assert seen["path"].endswith("best_model_state.pt")
        assert result["global_step"] >= 11

    def test_resume_can_fail_when_zero_new_steps_are_taken(
        self, patch_mt5, patch_tokenizer, base_args, tmp_path
    ):
        from tsl.models.pose_t5 import PoseToTextT5
        from tsl.train import train_pose_t5

        out_dir = str(tmp_path / "ckpts_resume_noop")
        os.makedirs(out_dir, exist_ok=True)

        model = PoseToTextT5(
            input_dim=_FEAT_DIM,
            num_encoder_layers=1,
            downsample_factor=2,
            base_model_name="google/mt5-small",
        )
        torch.save(
            {
                "step": 11,
                "epoch": 2,
                "metrics": {"val_loss": 3.0, "val_chrf": 16.0},
                "model_state_dict": model.state_dict(),
            },
            Path(out_dir) / "best_model_state.pt",
        )

        base_args.out_dir = out_dir
        base_args.resume = "best_state"
        base_args.epochs = 1
        base_args.eval_steps = 9999
        base_args.allow_noop_resume = "false"

        with pytest.raises(RuntimeError, match="zero new optimizer steps"):
            train_pose_t5.main(base_args)


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

    def test_val_chrf_uses_runtime_decode_defaults(self, monkeypatch):
        from tsl.train.train_pose_t5 import _compute_val_chrf
        from tsl.inference.pose_t5_translator import PoseT5Translator

        seen: dict[str, object] = {}

        class _FakeModel:
            def eval(self):
                return self

            def generate(self, src, src_lengths, **kwargs):
                seen.update(kwargs)
                return torch.tensor([[1, 2, 3]], dtype=torch.long)

        class _FakeExample:
            features_path = "ignored.npy"
            target_text = "abc"

        monkeypatch.setattr("tsl.train.train_pose_t5.load_features", lambda *_a, **_k: np.ones((8, _FEAT_DIM), dtype=np.float32))

        score = _compute_val_chrf(
            _FakeModel(),
            [_FakeExample()],
            _FakeHFTokenizer(),
            max_src_len=32,
            device=torch.device("cpu"),
            sample_size=1,
        )

        assert score >= 0.0
        assert seen["max_new_tokens"] == PoseT5Translator.DEFAULT_MAX_NEW_TOKENS
        assert seen["num_beams"] == PoseT5Translator.DEFAULT_BEAM_SIZE
        assert seen["no_repeat_ngram_size"] == PoseT5Translator.DEFAULT_NO_REPEAT_NGRAM_SIZE
        assert seen["repetition_penalty"] == pytest.approx(PoseT5Translator.DEFAULT_REPETITION_PENALTY)
        assert seen["length_penalty"] == pytest.approx(PoseT5Translator.DEFAULT_LENGTH_PENALTY)
        assert seen["early_stopping"] is True


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


class TestTrainingControls:
    def test_parser_accepts_regularization_and_early_stopping_args(self):
        from tsl.train.train_pose_t5 import _build_parser

        parser = _build_parser()
        args = parser.parse_args(
            [
                "--data-roots",
                "synthetic",
                "--checkpoint-steps",
                "500",
                "--dropout",
                "0.3",
                "--weight-decay",
                "0.05",
                "--early-stopping-patience",
                "10",
                "--early-stopping-min-delta",
                "0.01",
                "--early-stopping-metric",
                "val_chrf",
                "--reset-progress-history",
            ]
        )

        assert args.checkpoint_steps == 500
        assert args.dropout == pytest.approx(0.3)
        assert args.weight_decay == pytest.approx(0.05)
        assert args.early_stopping_patience == 10
        assert args.early_stopping_min_delta == pytest.approx(0.01)
        assert args.early_stopping_metric == "val_chrf"
        assert args.reset_progress_history is True

    def test_cli_dropout_reaches_model_constructor(
        self, patch_mt5, patch_tokenizer, base_args, monkeypatch
    ):
        from tsl.train import train_pose_t5

        seen: dict[str, float] = {}
        original_ctor = train_pose_t5.PoseToTextT5

        def _recording_ctor(*args, **kwargs):
            seen["dropout"] = kwargs["encoder_dropout"]
            return original_ctor(*args, **kwargs)

        monkeypatch.setattr(train_pose_t5, "PoseToTextT5", _recording_ctor)
        base_args.dropout = 0.3
        base_args.eval_steps = 9999

        train_pose_t5.main(base_args)

        assert seen["dropout"] == pytest.approx(0.3)

    def test_cli_weight_decay_reaches_optimizer(
        self, patch_mt5, patch_tokenizer, base_args, monkeypatch
    ):
        from tsl.train import train_pose_t5

        seen: dict[str, float] = {}
        original_adamw = train_pose_t5.torch.optim.AdamW

        def _recording_adamw(params, *args, **kwargs):
            seen["weight_decay"] = kwargs.get("weight_decay")
            return original_adamw(params, *args, **kwargs)

        monkeypatch.setattr(train_pose_t5.torch.optim, "AdamW", _recording_adamw)
        base_args.weight_decay = 0.05
        base_args.eval_steps = 9999

        train_pose_t5.main(base_args)

        assert seen["weight_decay"] == pytest.approx(0.05)

    def test_early_stopping_stops_after_patience(
        self, patch_mt5, patch_tokenizer, base_args, monkeypatch, tmp_path
    ):
        from tsl.train import train_pose_t5

        losses = iter([4.0, 4.2, 4.25])
        saved_metrics: list[dict[str, float]] = []

        def _fake_val_loss(*args, **kwargs):
            return next(losses)

        def _fake_val_chrf(*args, **kwargs):
            return 0.0

        def _fake_save_checkpoint(*args, **kwargs):
            saved_metrics.append(kwargs["metrics"])
            return Path(args[0]) / f"ckpt_step{kwargs['step']:08d}.pt"

        monkeypatch.setattr(train_pose_t5, "_compute_val_loss", _fake_val_loss)
        monkeypatch.setattr(train_pose_t5, "_compute_val_chrf", _fake_val_chrf)
        monkeypatch.setattr(train_pose_t5, "save_checkpoint", _fake_save_checkpoint)

        base_args.out_dir = str(tmp_path / "ckpts_early_stop")
        base_args.eval_steps = 1
        base_args.epochs = 3
        base_args.early_stopping_patience = 2
        base_args.early_stopping_min_delta = 0.0

        result = train_pose_t5.main(base_args)

        assert result["stopped_reason"] == "early_stopping"
        assert len(result["history"]) == 3
        assert saved_metrics[-1]["val_loss"] == pytest.approx(4.25)

    def test_early_stopping_can_track_val_chrf(
        self, patch_mt5, patch_tokenizer, base_args, monkeypatch, tmp_path
    ):
        from tsl.train import train_pose_t5

        losses = iter([4.0, 4.0, 4.0])
        chrfs = iter([12.0, 11.5, 11.4])
        saved_metrics: list[dict[str, float]] = []

        def _fake_save_checkpoint(*args, **kwargs):
            saved_metrics.append(kwargs["metrics"])
            return Path(args[0]) / f"ckpt_step{kwargs['step']:08d}.pt"

        monkeypatch.setattr(train_pose_t5, "_compute_val_loss", lambda *a, **k: next(losses))
        monkeypatch.setattr(train_pose_t5, "_compute_val_chrf", lambda *a, **k: next(chrfs))
        monkeypatch.setattr(train_pose_t5, "save_checkpoint", _fake_save_checkpoint)

        base_args.out_dir = str(tmp_path / "ckpts_early_stop_chrf")
        base_args.eval_steps = 1
        base_args.epochs = 3
        base_args.early_stopping_patience = 2
        base_args.early_stopping_min_delta = 0.0
        base_args.early_stopping_metric = "val_chrf"

        result = train_pose_t5.main(base_args)

        assert result["stopped_reason"] == "early_stopping"
        assert [item["val_chrf"] for item in result["history"]] == pytest.approx([12.0, 11.5, 11.4])
        assert saved_metrics[-1]["val_chrf"] == pytest.approx(11.4)

    def test_checkpoint_steps_can_skip_intermediate_checkpoint_saves(
        self, patch_mt5, patch_tokenizer, base_args, monkeypatch, tmp_path
    ):
        from tsl.train import train_pose_t5

        saved_steps: list[int] = []

        def _fake_save_checkpoint(*args, **kwargs):
            saved_steps.append(kwargs["step"])
            return Path(args[0]) / f"ckpt_step{kwargs['step']:08d}.pt"

        monkeypatch.setattr(train_pose_t5, "_compute_val_loss", lambda *a, **k: 1.0)
        monkeypatch.setattr(train_pose_t5, "_compute_val_chrf", lambda *a, **k: 0.0)
        monkeypatch.setattr(train_pose_t5, "save_checkpoint", _fake_save_checkpoint)

        base_args.out_dir = str(tmp_path / "ckpts_checkpoint_steps")
        base_args.epochs = 1
        base_args.eval_steps = 1
        base_args.checkpoint_steps = 2

        train_pose_t5.main(base_args)

        assert saved_steps == [2]

    def test_metric_improvement_saves_checkpoint_between_intervals(
        self, patch_mt5, patch_tokenizer, base_args, monkeypatch, tmp_path
    ):
        from tsl.train import train_pose_t5

        saved_steps: list[int] = []
        saved_best_states: list[int] = []
        losses = iter([5.0, 4.8, 4.8])

        def _fake_save_checkpoint(*args, **kwargs):
            saved_steps.append(kwargs["step"])
            return Path(args[0]) / f"ckpt_step{kwargs['step']:08d}.pt"

        def _fake_save_best_model_state(*args, **kwargs):
            saved_best_states.append(kwargs["step"])
            return Path(args[0]) / "best_model_state.pt"

        monkeypatch.setattr(train_pose_t5, "_compute_val_loss", lambda *a, **k: next(losses))
        monkeypatch.setattr(train_pose_t5, "_compute_val_chrf", lambda *a, **k: 0.0)
        monkeypatch.setattr(train_pose_t5, "save_checkpoint", _fake_save_checkpoint)
        monkeypatch.setattr(train_pose_t5, "_save_best_model_state", _fake_save_best_model_state)

        base_args.out_dir = str(tmp_path / "ckpts_improved_between_intervals")
        base_args.epochs = 1
        base_args.eval_steps = 1
        base_args.checkpoint_steps = 5

        train_pose_t5.main(base_args)

        assert saved_steps == []
        assert saved_best_states == [1, 2]

    def test_metric_improvement_still_saves_best_state_on_checkpoint_cadence(
        self, patch_mt5, patch_tokenizer, base_args, monkeypatch, tmp_path
    ):
        from tsl.train import train_pose_t5

        saved_steps: list[int] = []
        saved_best_states: list[int] = []
        losses = iter([5.0, 4.8, 4.7])

        def _fake_save_checkpoint(*args, **kwargs):
            saved_steps.append(kwargs["step"])
            return Path(args[0]) / f"ckpt_step{kwargs['step']:08d}.pt"

        def _fake_save_best_model_state(*args, **kwargs):
            saved_best_states.append(kwargs["step"])
            return Path(args[0]) / "best_model_state.pt"

        monkeypatch.setattr(train_pose_t5, "_compute_val_loss", lambda *a, **k: next(losses))
        monkeypatch.setattr(train_pose_t5, "_compute_val_chrf", lambda *a, **k: 0.0)
        monkeypatch.setattr(train_pose_t5, "save_checkpoint", _fake_save_checkpoint)
        monkeypatch.setattr(train_pose_t5, "_save_best_model_state", _fake_save_best_model_state)

        base_args.out_dir = str(tmp_path / "ckpts_improved_on_cadence")
        base_args.epochs = 1
        base_args.eval_steps = 1
        base_args.checkpoint_steps = 1

        train_pose_t5.main(base_args)

        assert saved_steps == [1, 2, 3]
        assert saved_best_states == [1, 2, 3]

    def test_source_balanced_sampler_enables_for_multi_source_train_examples(self):
        from tsl.data.manifest import SignTextExample
        from tsl.train.train_pose_t5 import _build_source_balanced_sampler

        examples = [
            SignTextExample("a0", "src_a", "train", "f0.npy", "alpha"),
            SignTextExample("a1", "src_a", "train", "f1.npy", "beta"),
            SignTextExample("b0", "src_b", "train", "f2.npy", "gamma"),
        ]

        sampler, summary = _build_source_balanced_sampler(examples, "auto")

        assert sampler is not None
        assert summary["enabled"] is True
        assert summary["source_counts"] == {"src_a": 2, "src_b": 1}
        assert summary["weights_by_source"]["src_a"] == pytest.approx(0.5)
        assert summary["weights_by_source"]["src_b"] == pytest.approx(1.0)

    def test_source_balanced_sampler_can_focus_rare_target_tokens(self):
        from tsl.data.manifest import SignTextExample
        from tsl.train.train_pose_t5 import _build_source_balanced_sampler

        examples = [
            SignTextExample("a0", "src_a", "train", "f0.npy", "ฉัน กิน"),
            SignTextExample("a1", "src_a", "train", "f1.npy", "ฉัน เรียน"),
            SignTextExample("a2", "src_a", "train", "f2.npy", "แม่ กิน"),
        ]

        sampler, summary = _build_source_balanced_sampler(
            examples,
            "false",
            focus_target_tokens="ฉัน,แม่",
            focus_target_max_multiplier=3.0,
        )

        assert sampler is not None
        assert summary["enabled"] is True
        assert summary["reason"] == "focus_only"
        assert summary["focus_balance_enabled"] is True
        assert summary["focus_token_counts"] == {"ฉัน": 2, "แม่": 1}
        assert summary["focus_token_multipliers"]["ฉัน"] == pytest.approx(1.0)
        assert summary["focus_token_multipliers"]["แม่"] == pytest.approx(2.0)
        assert float(sampler.weights[2]) == pytest.approx(float(sampler.weights[0]) * 2.0)

    def test_restore_best_checkpoint_weights_prefers_best_model_state(
        self, patch_mt5, patch_tokenizer, tmp_path
    ):
        from tsl.models.pose_t5 import PoseToTextT5
        from tsl.train import train_pose_t5

        model = PoseToTextT5(
            input_dim=312,
            num_encoder_layers=2,
            encoder_dropout=0.1,
            downsample_factor=4,
            base_model_name="dummy",
            local_model_path="dummy",
        )
        original = {
            key: value.detach().clone()
            for key, value in model.state_dict().items()
        }
        replacement = {
            key: torch.zeros_like(value)
            for key, value in original.items()
        }
        torch.save(
            {
                "step": 123,
                "epoch": 4,
                "metrics": {"val_chrf": 9.9},
                "model_state_dict": replacement,
            },
            tmp_path / "best_model_state.pt",
        )

        train_pose_t5._restore_best_checkpoint_weights(tmp_path, model, "val_chrf")

        for key, value in model.state_dict().items():
            assert torch.equal(value, replacement[key])

    def test_progress_metrics_written_during_run(
        self, patch_mt5, patch_tokenizer, base_args, tmp_path
    ):
        from tsl.train import train_pose_t5

        base_args.out_dir = str(tmp_path / "ckpts_progress")
        base_args.eval_steps = 1
        base_args.epochs = 1

        train_pose_t5.main(base_args)

        metrics_path = Path(base_args.out_dir) / "train_metrics.json"
        payload = __import__("json").loads(metrics_path.read_text(encoding="utf-8"))

        assert payload["history"]
        assert payload["global_step"] >= 1
        assert payload["stopped_reason"] in {"running", "completed", "early_stopping", "max_runtime"}


class TestProgressMetricHelpers:
    def test_trim_history_to_step_discards_future_metrics(self):
        from tsl.train.train_pose_t5 import _trim_history_to_step

        history = [
            {"step": 100, "val_chrf": 10.0},
            {"step": 200, "val_chrf": 9.0},
            {"step": 300, "val_chrf": 8.0},
        ]

        assert _trim_history_to_step(history, 200) == history[:2]

    def test_count_evals_without_improvement_tracks_tail_streak(self):
        from tsl.train.train_pose_t5 import _count_evals_without_improvement

        history = [
            {"val_loss": 5.0},
            {"val_loss": 4.5},
            {"val_loss": 4.6},
            {"val_loss": 4.7},
        ]

        assert _count_evals_without_improvement(history, "val_loss", 0.0) == 2

    def test_count_evals_without_improvement_supports_max_metrics(self):
        from tsl.train.train_pose_t5 import _count_evals_without_improvement

        history = [
            {"val_chrf": 10.0},
            {"val_chrf": 12.0},
            {"val_chrf": 11.8},
            {"val_chrf": 11.7},
        ]

        assert _count_evals_without_improvement(history, "val_chrf", 0.0) == 2

    def test_resume_trims_future_history_before_early_stopping(
        self, patch_mt5, patch_tokenizer, base_args, tmp_path, monkeypatch
    ):
        from tsl.models.pose_t5 import PoseToTextT5
        from tsl.train.checkpointing import save_checkpoint
        from tsl.train import train_pose_t5

        out_dir = Path(tmp_path / "ckpts_resume_trim")
        out_dir.mkdir()

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
            step=1,
            epoch=0,
            metrics={"val_loss": 5.0, "val_chrf": 12.0},
            keep_last_k=3,
        )
        (out_dir / "train_metrics.json").write_text(
            __import__("json").dumps(
                {
                    "global_step": 3,
                    "stopped_reason": "running",
                    "history": [
                        {"step": 1, "epoch": 0, "val_chrf": 12.0, "val_loss": 5.0, "train_loss": 6.0},
                        {"step": 2, "epoch": 0, "val_chrf": 11.0, "val_loss": 5.1, "train_loss": 5.9},
                        {"step": 3, "epoch": 0, "val_chrf": 10.5, "val_loss": 5.2, "train_loss": 5.8},
                    ],
                }
            ),
            encoding="utf-8",
        )

        chrfs = iter([11.5, 11.4])
        monkeypatch.setattr(train_pose_t5, "_compute_val_loss", lambda *a, **k: 5.0)
        monkeypatch.setattr(train_pose_t5, "_compute_val_chrf", lambda *a, **k: next(chrfs))

        base_args.out_dir = str(out_dir)
        base_args.resume = "auto"
        base_args.epochs = 1
        base_args.eval_steps = 1
        base_args.early_stopping_patience = 2
        base_args.early_stopping_metric = "val_chrf"

        result = train_pose_t5.main(base_args)

        # If stale future history were reused, this would stop on the first new eval.
        assert [item["step"] for item in result["history"][-2:]] == [2, 3]

    def test_reset_progress_history_restarts_patience_from_resumed_checkpoint(
        self, patch_mt5, patch_tokenizer, base_args, tmp_path, monkeypatch
    ):
        from tsl.models.pose_t5 import PoseToTextT5
        from tsl.train.checkpointing import save_checkpoint
        from tsl.train import train_pose_t5

        out_dir = Path(tmp_path / "ckpts_reset_progress")
        out_dir.mkdir()

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
            step=3,
            epoch=0,
            metrics={"val_loss": 4.0, "val_chrf": 12.0},
            keep_last_k=3,
        )
        (out_dir / "train_metrics.json").write_text(
            __import__("json").dumps(
                {
                    "global_step": 6,
                    "stopped_reason": "running",
                    "history": [
                        {"step": 1, "epoch": 0, "val_chrf": 9.0, "val_loss": 5.0, "train_loss": 6.0},
                        {"step": 2, "epoch": 0, "val_chrf": 10.0, "val_loss": 4.5, "train_loss": 5.8},
                        {"step": 3, "epoch": 0, "val_chrf": 12.0, "val_loss": 4.0, "train_loss": 5.6},
                        {"step": 4, "epoch": 0, "val_chrf": 11.5, "val_loss": 4.1, "train_loss": 5.4},
                        {"step": 5, "epoch": 0, "val_chrf": 11.2, "val_loss": 4.2, "train_loss": 5.2},
                        {"step": 6, "epoch": 0, "val_chrf": 11.0, "val_loss": 4.3, "train_loss": 5.0},
                    ],
                }
            ),
            encoding="utf-8",
        )

        chrfs = iter([11.8, 11.7])
        monkeypatch.setattr(train_pose_t5, "_compute_val_loss", lambda *a, **k: 4.0)
        monkeypatch.setattr(train_pose_t5, "_compute_val_chrf", lambda *a, **k: next(chrfs))

        base_args.out_dir = str(out_dir)
        base_args.resume = "auto"
        base_args.reset_progress_history = True
        base_args.epochs = 1
        base_args.eval_steps = 1
        base_args.early_stopping_patience = 2
        base_args.early_stopping_metric = "val_chrf"

        result = train_pose_t5.main(base_args)

        assert result["history"][0]["step"] == 3
        assert [item["step"] for item in result["history"][-2:]] == [4, 5]

    def test_find_best_checkpoint_for_metric_supports_min_and_max(
        self, patch_mt5, patch_tokenizer, tmp_path
    ):
        from tsl.models.pose_t5 import PoseToTextT5
        from tsl.train.checkpointing import save_checkpoint
        from tsl.train.train_pose_t5 import _find_best_checkpoint_for_metric

        out_dir = tmp_path / "ckpts_best_metric"
        out_dir.mkdir()
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
            step=100,
            epoch=0,
            metrics={"val_loss": 4.0, "val_chrf": 11.0},
            keep_last_k=10,
        )
        save_checkpoint(
            out_dir,
            model,
            optimizer,
            scheduler,
            scaler=None,
            step=200,
            epoch=0,
            metrics={"val_loss": 3.5, "val_chrf": 10.0},
            keep_last_k=10,
        )

        assert _find_best_checkpoint_for_metric(out_dir, "val_loss").name == "ckpt_step00000200.pt"
        assert _find_best_checkpoint_for_metric(out_dir, "val_chrf").name == "ckpt_step00000100.pt"

    def test_finalize_training_exports_best_checkpoint_weights(
        self, patch_mt5, patch_tokenizer, tmp_path
    ):
        from tsl.models.pose_t5 import PoseToTextT5
        from tsl.train.checkpointing import save_checkpoint
        from tsl.train.train_pose_t5 import _finalize_training

        out_dir = tmp_path / "ckpts_finalize_best"
        out_dir.mkdir()

        model = PoseToTextT5(
            input_dim=_FEAT_DIM,
            num_encoder_layers=1,
            downsample_factor=2,
            base_model_name="google/mt5-small",
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

        with torch.no_grad():
            model.input_proj.weight.fill_(1.0)
        save_checkpoint(
            out_dir,
            model,
            optimizer,
            scheduler,
            scaler=None,
            step=100,
            epoch=0,
            metrics={"val_loss": 3.0, "val_chrf": 12.0},
            keep_last_k=10,
        )

        with torch.no_grad():
            model.input_proj.weight.fill_(2.0)
        save_checkpoint(
            out_dir,
            model,
            optimizer,
            scheduler,
            scaler=None,
            step=200,
            epoch=0,
            metrics={"val_loss": 4.0, "val_chrf": 10.0},
            keep_last_k=10,
        )

        with torch.no_grad():
            model.input_proj.weight.fill_(9.0)

        _finalize_training(
            out_dir=out_dir,
            model=model,
            hf_tokenizer=_FakeHFTokenizer(),
            global_step=200,
            stopped_reason="early_stopping",
            all_metrics=[],
            tracked_metric="val_chrf",
        )

        reloaded = PoseToTextT5.from_pretrained(str(out_dir), device="cpu")
        assert torch.allclose(reloaded.input_proj.weight, torch.ones_like(reloaded.input_proj.weight))

    def test_require_gpu_fails_before_loading_data(
        self, monkeypatch, base_args
    ):
        from tsl.train import train_pose_t5

        loaded_data = {"called": False}

        def _fail_if_called(*args, **kwargs):
            loaded_data["called"] = True
            raise AssertionError("data loading should not run when GPU is required but unavailable")

        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        monkeypatch.setattr(train_pose_t5, "_load_all_examples", _fail_if_called)
        base_args.require_gpu = True

        with pytest.raises(RuntimeError, match="CUDA is required"):
            train_pose_t5.main(base_args)

        assert loaded_data["called"] is False

    def test_nonfinite_loss_guard_rejects_nan(self):
        from tsl.train.train_pose_t5 import _assert_finite_loss

        with pytest.raises(RuntimeError, match="Non-finite training loss"):
            _assert_finite_loss(torch.tensor(float("nan")), global_step=3, epoch=1)

    def test_max_train_steps_stops_after_requested_optimizer_steps(
        self, patch_mt5, patch_tokenizer, base_args
    ):
        from tsl.train import train_pose_t5

        base_args.max_train_steps = 1
        base_args.eval_steps = 100

        result = train_pose_t5.main(base_args)

        assert result["stopped_reason"] == "max_train_steps"
        assert result["new_optimizer_steps"] == 1
        assert Path(base_args.out_dir, "run_status.json").is_file()
        assert Path(base_args.out_dir, "pose_t5_config.json").is_file()


# ---------------------------------------------------------------------------
# Import for math module (used in TestValLossComputation)
# ---------------------------------------------------------------------------
import math
