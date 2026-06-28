"""Tests for encrypted research video storage."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from tsl.privacy.video_store import VideoStore


def test_encrypt_decrypt_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "config.TSL_DATA_ENCRYPTION_KEY",
        "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=",
    )
    store = VideoStore(tmp_path)
    rel = store.save_encrypted("seg-1", b"fake-webm-bytes")
    assert rel == "videos/seg-1.webm.enc"
    assert store.decrypt("seg-1") == b"fake-webm-bytes"


def test_purge_expired(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "config.TSL_DATA_ENCRYPTION_KEY",
        "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=",
    )
    store = VideoStore(tmp_path)
    store.save_encrypted("old", b"old")
    path = store.encrypted_path("old")
    old_time = datetime.now(timezone.utc) - timedelta(days=400)
    ts = old_time.timestamp()
    path.touch()
    import os

    os.utime(path, (ts, ts))
    removed = store.purge_expired(retention_days=365)
    assert removed == 1
    assert not path.is_file()
