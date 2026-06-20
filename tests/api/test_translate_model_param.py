"""Tests for POST /translate with per-request model selection."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

import tsl.api.app as appmod
from tsl.api.app import app


class StubTranslator:
    def __init__(self, sentence: str = "สวัสดี", score: float = 0.9) -> None:
        self.sentence = sentence
        self.score = score
        self.last_features: np.ndarray | None = None

    def translate(self, features: np.ndarray):
        self.last_features = features
        return SimpleNamespace(sentence=self.sentence, score=self.score)


def _raw_frames(T: int = 4) -> list:
    return np.zeros((T, 543, 3), dtype=np.float32).tolist()


def _client():
    return TestClient(app)


# ------------------------------------------------------------------
# Basic routing
# ------------------------------------------------------------------

def test_translate_omitted_model_uses_default():
    """POST /translate with no model field uses the catalog default (v3_poset5)."""
    stub = StubTranslator(sentence="ทดสอบ", score=0.8)
    with patch.object(appmod, "get_translator_for", return_value=stub) as mock_fn:
        resp = _client().post(
            "/translate",
            json={"frames": _raw_frames(), "feature_schema": "raw_mediapipe_543x3"},
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body["sentence"] == "ทดสอบ"
    assert "model" in body   # returned model id present


def test_translate_explicit_model_v2_dispatches_correctly():
    """POST /translate with model='v2_slt' dispatches to that model."""
    stub = StubTranslator(sentence="สองจุด", score=0.7)
    with patch.object(appmod, "get_translator_for", return_value=stub) as mock_fn:
        resp = _client().post(
            "/translate",
            json={
                "frames": _raw_frames(),
                "feature_schema": "raw_mediapipe_543x3",
                "model": "v2_slt",
            },
        )
    assert resp.status_code == 200
    assert resp.json()["model"] == "v2_slt"


def test_translate_unknown_model_returns_400():
    """POST /translate with an unknown model id returns 400."""
    resp = _client().post(
        "/translate",
        json={
            "frames": _raw_frames(),
            "feature_schema": "raw_mediapipe_543x3",
            "model": "nonexistent_model_xyz",
        },
    )
    assert resp.status_code == 400


def test_translate_unavailable_model_returns_503(tmp_path, monkeypatch):
    """POST /translate returns 503 when selected model checkpoint is missing."""
    import tsl.api.model_catalog as catalog_mod
    # Clear translator cache so get_translator_for attempts a fresh load
    appmod._translator_cache.clear()
    # Point v3_poset5 at a non-existent directory so availability returns False
    original_catalog = catalog_mod._CATALOG
    patched = [
        s if s.id != "v3_poset5"
        else s.__class__(
            id=s.id,
            label_th=s.label_th,
            label_en=s.label_en,
            architecture=s.architecture,
            checkpoint_dir=str(tmp_path / "nonexistent"),
            default=s.default,
        )
        for s in original_catalog
    ]
    monkeypatch.setattr(catalog_mod, "_CATALOG", patched)

    resp = _client().post(
        "/translate",
        json={"frames": _raw_frames(), "feature_schema": "raw_mediapipe_543x3", "model": "v3_poset5"},
    )
    assert resp.status_code == 503


def test_translate_empty_frames_returns_empty():
    """POST /translate with empty frames returns sentence='' score=0.0 (no 4xx)."""
    resp = _client().post(
        "/translate",
        json={"frames": [], "feature_schema": "raw_mediapipe_543x3"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["sentence"] == ""
    assert body["score"] == 0.0
    assert "model" in body


def test_translate_bad_shape_returns_400():
    """POST /translate with wrong frame shape returns 400."""
    resp = _client().post(
        "/translate",
        json={"frames": [[1.0, 2.0]], "feature_schema": "raw_mediapipe_543x3"},
    )
    assert resp.status_code == 400


def test_translate_normalized_features_passthrough():
    """POST /translate with feature_schema=selected_312 passes (T,312) features to translator."""
    stub = StubTranslator(sentence="ปกติ", score=0.95)
    frames = np.zeros((3, 312), dtype=np.float32).tolist()
    with patch.object(appmod, "get_translator_for", return_value=stub):
        resp = _client().post(
            "/translate",
            json={"frames": frames, "feature_schema": "selected_312"},
        )
    assert resp.status_code == 200
    assert stub.last_features is not None
    assert stub.last_features.shape == (3, 312)
