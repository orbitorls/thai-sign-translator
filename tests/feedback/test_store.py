"""Tests for user feedback contribution store."""
from __future__ import annotations

import csv
import json

import numpy as np
import pytest

from tsl.feedback.store import (
    ContributionStore,
    ContributionValidationError,
    DuplicateContributionError,
    MANIFEST_COLUMNS,
    MIN_FRAMES,
)


def _raw_frames(count: int = MIN_FRAMES) -> np.ndarray:
    return np.zeros((count, 543, 3), dtype=np.float32)


def _raw_frames_v4(count: int = MIN_FRAMES) -> np.ndarray:
    frames = np.zeros((count, 543, 4), dtype=np.float32)
    frames[:, :, 3] = 1.0
    return frames


def test_save_submission_writes_npy_manifest_and_meta(tmp_path, monkeypatch):
    monkeypatch.setattr("tsl.feedback.store.config.USER_CONTRIBUTIONS_DIR", str(tmp_path))
    store = ContributionStore()
    meta = store.save_teach(
        frames=_raw_frames().tolist(),
        label_text="สวัสดีครับ",
        user_hash="hash-1",
    )

    npy_path = tmp_path / "features" / f"{meta.segment_id}.npy"
    assert npy_path.is_file()
    arr = np.load(npy_path)
    assert arr.shape[1] == 312

    with (tmp_path / "manifest.csv").open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["segment_id"] == meta.segment_id
    assert rows[0]["text"] == "สวัสดีครับ"
    assert rows[0]["source"] == "user_feedback"
    assert rows[0]["kind"] == "teach"
    assert set(rows[0].keys()) == set(MANIFEST_COLUMNS)

    meta_rows = store.iter_meta()
    assert len(meta_rows) == 1
    assert meta_rows[0]["status"] == "pending"
    assert meta_rows[0]["user_hash"] == "hash-1"


def test_save_submission_accepts_543x4_and_writes_capture_quality(tmp_path, monkeypatch):
    monkeypatch.setattr("tsl.feedback.store.config.USER_CONTRIBUTIONS_DIR", str(tmp_path))
    store = ContributionStore()
    meta = store.save_teach(
        frames=_raw_frames_v4().tolist(),
        label_text="สวัสดีครับ",
        user_hash="hash-1",
        capture_quality={
            "fps": 24.0,
            "lighting_ok": True,
            "hand_present": True,
            "feature_schema": "raw_mediapipe_543x4",
            "camera_facing": "user",
        },
    )

    arr = np.load(tmp_path / "features" / f"{meta.segment_id}.npy")
    assert arr.shape[1] == 312
    row = store.iter_meta()[0]
    assert row["capture_quality"]["fps"] == 24.0
    assert row["capture_quality"]["feature_schema"] == "raw_mediapipe_543x4"


def test_dedup_rejects_identical_submission(tmp_path, monkeypatch):
    monkeypatch.setattr("tsl.feedback.store.config.USER_CONTRIBUTIONS_DIR", str(tmp_path))
    store = ContributionStore()
    frames = _raw_frames().tolist()
    store.save_correction(
        frames=frames,
        predicted_text="ผิด",
        corrected_text="ถูก",
        user_hash="hash-1",
    )
    with pytest.raises(DuplicateContributionError):
        store.save_correction(
            frames=frames,
            predicted_text="ผิด",
            corrected_text="ถูก",
            user_hash="hash-1",
        )


def test_validation_rejects_short_sequence(tmp_path, monkeypatch):
    monkeypatch.setattr("tsl.feedback.store.config.USER_CONTRIBUTIONS_DIR", str(tmp_path))
    store = ContributionStore()
    with pytest.raises(ContributionValidationError):
        store.save_teach(
            frames=_raw_frames(MIN_FRAMES - 1).tolist(),
            label_text="สั้น",
            user_hash="hash-1",
        )


def test_mark_used_updates_meta_status(tmp_path, monkeypatch):
    monkeypatch.setattr("tsl.feedback.store.config.USER_CONTRIBUTIONS_DIR", str(tmp_path))
    store = ContributionStore()
    meta = store.save_teach(
        frames=_raw_frames().tolist(),
        label_text="ทดสอบ",
        user_hash="hash-1",
    )
    assert store.pending_count() == 1
    updated = store.mark_used({meta.segment_id})
    assert updated == 1
    assert store.pending_count() == 0
    row = store.iter_meta()[0]
    assert row["status"] == "used"


def test_pending_segment_ids(tmp_path, monkeypatch):
    monkeypatch.setattr("tsl.feedback.store.config.USER_CONTRIBUTIONS_DIR", str(tmp_path))
    store = ContributionStore()
    meta = store.save_teach(
        frames=_raw_frames().tolist(),
        label_text="ทดสอบ",
        user_hash="hash-1",
    )
    assert store.pending_segment_ids() == {meta.segment_id}
    store.mark_used({meta.segment_id})
    assert store.pending_segment_ids() == set()


def test_stats_and_retrain_log(tmp_path, monkeypatch):
    monkeypatch.setattr("tsl.feedback.store.config.USER_CONTRIBUTIONS_DIR", str(tmp_path))
    store = ContributionStore()
    store.save_teach(
        frames=_raw_frames().tolist(),
        label_text="หนึ่ง",
        user_hash="hash-1",
    )
    store.write_retrain_log(
        {"completed_at": "2026-06-27T00:00:00Z", "model_version": "v1", "status": "completed"}
    )
    stats = store.stats()
    assert stats["total_count"] == 1
    assert stats["pending_count"] == 1
    assert stats["last_retrain_at"] == "2026-06-27T00:00:00Z"
    assert stats["feedback_version"] == "v1"


def test_delete_by_user_hash_and_rebuild_manifest(tmp_path, monkeypatch):
    monkeypatch.setattr("tsl.feedback.store.config.USER_CONTRIBUTIONS_DIR", str(tmp_path))
    store = ContributionStore()
    meta = store.save_teach(
        frames=_raw_frames().tolist(),
        label_text="ลบ",
        user_hash="hash-delete",
    )
    store.save_teach(
        frames=_raw_frames(15).tolist(),
        label_text="คงไว้",
        user_hash="hash-keep",
    )
    removed = store.delete_by_user_hash("hash-delete")
    assert removed == 1
    assert store.total_count() == 1
    assert not (tmp_path / "features" / f"{meta.segment_id}.npy").is_file()
    rebuilt = store.rebuild_manifest()
    assert rebuilt == 1
