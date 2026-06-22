from __future__ import annotations

import json

import pytest
import torch

from scripts.colab_publish_checkpoints import (
    _ensure_metadata,
    _best_checkpoint,
    _latest_checkpoint,
    _sync_staging,
    _verify_published_checkpoint,
    _write_state,
)


def test_latest_checkpoint_picks_highest_step(tmp_path):
    ckpt_dir = tmp_path / "ckpts"
    ckpt_dir.mkdir()
    (ckpt_dir / "ckpt_step00000500.pt").write_bytes(b"a")
    (ckpt_dir / "ckpt_step00001000.pt").write_bytes(b"b")

    latest = _latest_checkpoint(ckpt_dir)

    assert latest is not None
    assert latest.name == "ckpt_step00001000.pt"


def test_ensure_metadata_creates_directory_and_json(tmp_path):
    staging_dir = tmp_path / "publish"

    _ensure_metadata(staging_dir, "orbitorls/thai-sign-ckpt", "Thai Sign Ckpt")

    metadata = json.loads((staging_dir / "dataset-metadata.json").read_text(encoding="utf-8"))
    assert metadata["id"] == "orbitorls/thai-sign-ckpt"
    assert metadata["title"] == "Thai Sign Ckpt"


def test_sync_staging_keeps_only_latest_checkpoint(tmp_path):
    ckpt_dir = tmp_path / "ckpts"
    staging_dir = tmp_path / "publish"
    ckpt_dir.mkdir()
    staging_dir.mkdir()

    latest = ckpt_dir / "ckpt_step00001000.pt"
    latest.write_bytes(b"new")
    (ckpt_dir / "train.log").write_text("log", encoding="utf-8")
    (staging_dir / "ckpt_step00000500.pt").write_bytes(b"old")

    _sync_staging(ckpt_dir, staging_dir, latest)

    assert not (staging_dir / "ckpt_step00000500.pt").exists()
    assert (staging_dir / "ckpt_step00001000.pt").read_bytes() == b"new"
    assert (staging_dir / "latest_checkpoint.txt").read_text(encoding="utf-8") == latest.name


def test_best_checkpoint_prefers_highest_val_chrf(tmp_path):
    ckpt_dir = tmp_path / "ckpts"
    ckpt_dir.mkdir()
    low = ckpt_dir / "ckpt_step00001000.pt"
    high = ckpt_dir / "ckpt_step00001500.pt"
    torch.save({"metrics": {"val_chrf": 9.7}}, low)
    torch.save({"metrics": {"val_chrf": 13.6}}, high)

    best = _best_checkpoint(ckpt_dir)

    assert best is not None
    assert best.name == "ckpt_step00001500.pt"


def test_verify_published_checkpoint_retries_until_visible(monkeypatch):
    calls = {"count": 0}

    def fake_contains(_dataset_id: str, checkpoint_name: str) -> bool:
        calls["count"] += 1
        return calls["count"] >= 3 and checkpoint_name == "ckpt_step00002000.pt"

    monkeypatch.setattr("scripts.colab_publish_checkpoints._dataset_contains_checkpoint", fake_contains)

    _verify_published_checkpoint(
        "orbitorls/thai-sign-ckpt",
        "ckpt_step00002000.pt",
        retries=3,
        delay_sec=0,
    )

    assert calls["count"] == 3


def test_verify_published_checkpoint_raises_when_missing(monkeypatch):
    monkeypatch.setattr(
        "scripts.colab_publish_checkpoints._dataset_contains_checkpoint",
        lambda *_args, **_kwargs: False,
    )

    with pytest.raises(RuntimeError, match="ckpt_step00002000.pt"):
        _verify_published_checkpoint(
            "orbitorls/thai-sign-ckpt",
            "ckpt_step00002000.pt",
            retries=2,
            delay_sec=0,
        )


def test_write_state_records_latest_and_pending(tmp_path):
    state_path = tmp_path / "publisher_state.json"

    _write_state(
        state_path,
        latest_checkpoint="ckpt_step00002000.pt",
        pending_checkpoint="ckpt_step00002500.pt",
    )

    payload = json.loads(state_path.read_text(encoding="utf-8"))
    assert payload["latest_checkpoint"] == "ckpt_step00002000.pt"
    assert payload["pending_checkpoint"] == "ckpt_step00002500.pt"
    assert isinstance(payload["published_at"], float)


def test_sync_staging_keeps_latest_and_best_checkpoint(tmp_path):
    ckpt_dir = tmp_path / "ckpts"
    staging_dir = tmp_path / "publish"
    ckpt_dir.mkdir()
    staging_dir.mkdir()

    low = ckpt_dir / "ckpt_step00001000.pt"
    best = ckpt_dir / "ckpt_step00001500.pt"
    latest = ckpt_dir / "ckpt_step00002000.pt"
    torch.save({"metrics": {"val_chrf": 9.0}}, low)
    torch.save({"metrics": {"val_chrf": 13.6}}, best)
    torch.save({"metrics": {"val_chrf": 12.5}}, latest)

    _sync_staging(ckpt_dir, staging_dir, latest)

    assert (staging_dir / "ckpt_step00001500.pt").exists()
    assert (staging_dir / "ckpt_step00002000.pt").exists()
    assert not (staging_dir / "ckpt_step00001000.pt").exists()
    assert (staging_dir / "best_checkpoint.txt").read_text(encoding="utf-8") == "ckpt_step00001500.pt"
