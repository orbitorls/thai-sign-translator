"""Tests for the GET /supported-phrases endpoint."""
from __future__ import annotations

import os
import csv
import pytest
from fastapi.testclient import TestClient

from tsl.api.app import app


def _client():
    return TestClient(app)


def test_supported_phrases_returns_200_when_data_missing(monkeypatch, tmp_path):
    """Endpoint returns 200 with empty list when TSL-51 data dir is absent."""
    import config as cfg
    monkeypatch.setattr(cfg, "DATA_DIR", str(tmp_path))

    resp = _client().get("/supported-phrases")
    assert resp.status_code == 200
    body = resp.json()
    assert body["phrases"] == []
    assert body["total"] == 0
    assert isinstance(body["note"], str)


def test_supported_phrases_returns_phrases_when_manifest_present(monkeypatch, tmp_path):
    """Endpoint parses sentence_metadata.csv and returns sorted, unique phrases."""
    import config as cfg
    monkeypatch.setattr(cfg, "DATA_DIR", str(tmp_path))

    # Create a minimal TSL-51 metadata structure
    meta_dir = tmp_path / "tsl51" / "metadata"
    meta_dir.mkdir(parents=True)
    meta_path = meta_dir / "sentence_metadata.csv"
    rows = [
        {"video_id": "v1", "sentence_id": "s1", "sentence_clean": "สวัสดี", "landmark_path": ""},
        {"video_id": "v2", "sentence_id": "s2", "sentence_clean": "ขอบคุณ", "landmark_path": ""},
        {"video_id": "v3", "sentence_id": "s3", "sentence_clean": "สวัสดี", "landmark_path": ""},  # duplicate
        {"video_id": "v4", "sentence_id": "s4", "sentence_clean": "", "landmark_path": ""},      # empty → skip
    ]
    with meta_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    resp = _client().get("/supported-phrases")
    assert resp.status_code == 200
    body = resp.json()
    # Deduped: "ขอบคุณ", "สวัสดี" (2 unique, empty skipped)
    assert body["total"] == 2
    assert body["phrases"] == sorted(["สวัสดี", "ขอบคุณ"])


def test_supported_phrases_schema(monkeypatch, tmp_path):
    """Response always has 'phrases', 'total', and 'note' keys."""
    import config as cfg
    monkeypatch.setattr(cfg, "DATA_DIR", str(tmp_path))

    body = _client().get("/supported-phrases").json()
    assert "phrases" in body
    assert "total" in body
    assert "note" in body
