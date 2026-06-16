"""Tests for the ``POST /translate-video`` endpoint and v3 checkpoint dispatch."""
from __future__ import annotations

import os
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

import tsl.api.app as appmod
from tsl.api.app import app, get_active_translator


# ---------------------------------------------------------------------------
# Stub translators
# ---------------------------------------------------------------------------

class StubTranslator:
    """Duck-typed translator stub: accepts features, returns a prediction."""

    def __init__(self, sentence: str = "สวัสดี", score: float = 0.9) -> None:
        self.sentence = sentence
        self.score = score
        self.last_features: np.ndarray | None = None

    def translate(self, features: np.ndarray):
        self.last_features = features
        return SimpleNamespace(sentence=self.sentence, score=self.score)


class RaisingTranslator:
    """Translator that raises an exception to exercise error paths."""

    def __init__(self, exc: Exception) -> None:
        self._exc = exc

    def translate(self, features: np.ndarray):
        raise self._exc


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _raw_frames(T: int = 5) -> list:
    """Return T frames of shape (543, 3) as nested lists."""
    arr = np.zeros((T, 543, 3), dtype=np.float32)
    return arr.tolist()


def _norm_frames(T: int = 5) -> list:
    """Return T frames of shape (312,) as nested lists."""
    arr = np.zeros((T, 312), dtype=np.float32)
    return arr.tolist()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_translate_video_raw_frames_returns_200():
    """POST /translate-video with valid (T, 543, 3) frames returns 200 with sentence."""
    stub = StubTranslator(sentence="สวัสดี", score=0.85)
    app.dependency_overrides[get_active_translator] = lambda: stub
    try:
        client = TestClient(app)
        resp = client.post(
            "/translate-video",
            json={"frames": _raw_frames(4), "feature_schema": "raw_mediapipe_543x3"},
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["sentence"] == "สวัสดี"
        assert isinstance(body["score"], float)
        assert 0.0 <= body["score"] <= 1.0
        # Translator should receive (T, 312) normalized features
        assert stub.last_features is not None
        assert stub.last_features.ndim == 2
        assert stub.last_features.shape[1] == 312
    finally:
        app.dependency_overrides.clear()


def test_translate_video_empty_frames_returns_200_empty():
    """POST /translate-video with empty frames list returns sentence='' score=0.0."""
    stub = StubTranslator()
    app.dependency_overrides[get_active_translator] = lambda: stub
    try:
        client = TestClient(app)
        resp = client.post(
            "/translate-video",
            json={"frames": [], "feature_schema": "raw_mediapipe_543x3"},
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body == {"sentence": "", "score": 0.0}
        # Translator should NOT be called for empty input
        assert stub.last_features is None
    finally:
        app.dependency_overrides.clear()


def test_translate_video_wrong_shape_returns_400():
    """POST /translate-video with wrong shape (T, 100) returns 400."""
    stub = StubTranslator()
    app.dependency_overrides[get_active_translator] = lambda: stub
    try:
        client = TestClient(app)
        bad_frames = np.zeros((3, 100), dtype=np.float32).tolist()
        resp = client.post(
            "/translate-video",
            json={"frames": bad_frames, "feature_schema": "raw_mediapipe_543x3"},
        )
        assert resp.status_code == 400, resp.text
        detail = resp.json()["detail"]
        assert "543" in detail or "shape" in detail.lower()
    finally:
        app.dependency_overrides.clear()


def test_translate_video_selected_312_schema():
    """POST /translate-video with selected_312 schema passes features through directly."""
    stub = StubTranslator(sentence="ขอบคุณ", score=0.7)
    app.dependency_overrides[get_active_translator] = lambda: stub
    try:
        client = TestClient(app)
        resp = client.post(
            "/translate-video",
            json={"frames": _norm_frames(6), "feature_schema": "selected_312"},
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["sentence"] == "ขอบคุณ"
        assert stub.last_features is not None
        assert stub.last_features.shape == (6, 312)
    finally:
        app.dependency_overrides.clear()


def test_translate_video_selected_312_wrong_shape_returns_400():
    """POST /translate-video with selected_312 and wrong dim returns 400."""
    stub = StubTranslator()
    app.dependency_overrides[get_active_translator] = lambda: stub
    try:
        client = TestClient(app)
        bad_frames = np.zeros((3, 162), dtype=np.float32).tolist()
        resp = client.post(
            "/translate-video",
            json={"frames": bad_frames, "feature_schema": "selected_312"},
        )
        assert resp.status_code == 400, resp.text
    finally:
        app.dependency_overrides.clear()


def test_translate_video_unknown_schema_returns_400():
    """POST /translate-video with an unknown feature_schema returns 400."""
    stub = StubTranslator()
    app.dependency_overrides[get_active_translator] = lambda: stub
    try:
        client = TestClient(app)
        resp = client.post(
            "/translate-video",
            json={"frames": _raw_frames(2), "feature_schema": "bogus_schema"},
        )
        assert resp.status_code == 400, resp.text
    finally:
        app.dependency_overrides.clear()


def test_translate_video_no_checkpoint_returns_503():
    """POST /translate-video returns 503 when no checkpoint exists."""
    # Remove overrides so the real get_active_translator runs
    app.dependency_overrides.pop(get_active_translator, None)
    saved = appmod._active_translator
    appmod._active_translator = None

    # Patch both checkpoint dirs to non-existent paths
    import config as _config
    orig_v3 = _config.SLT_V3_CHECKPOINT_DIR
    orig_v2 = _config.SLT_CHECKPOINT_DIR
    _config.SLT_V3_CHECKPOINT_DIR = "/nonexistent/pose_t5_v3"
    _config.SLT_CHECKPOINT_DIR = "/nonexistent/slt_v2"

    # Also reset _sentence_runtime to force re-evaluation
    saved_sr = appmod._sentence_runtime
    appmod._sentence_runtime = None

    try:
        client = TestClient(app)
        resp = client.post(
            "/translate-video",
            json={"frames": _raw_frames(2), "feature_schema": "raw_mediapipe_543x3"},
        )
        assert resp.status_code == 503, resp.text
        assert "checkpoint" in resp.json()["detail"].lower()
    finally:
        appmod._active_translator = saved
        appmod._sentence_runtime = saved_sr
        _config.SLT_V3_CHECKPOINT_DIR = orig_v3
        _config.SLT_CHECKPOINT_DIR = orig_v2
        app.dependency_overrides.pop(get_active_translator, None)


def test_translate_video_flat_raw_frames():
    """POST /translate-video with flat (T, 1629) frames is reshaped and accepted."""
    stub = StubTranslator(sentence="ใช่", score=0.6)
    app.dependency_overrides[get_active_translator] = lambda: stub
    try:
        client = TestClient(app)
        flat_frames = np.zeros((3, 543 * 3), dtype=np.float32).tolist()
        resp = client.post(
            "/translate-video",
            json={"frames": flat_frames, "feature_schema": "raw_mediapipe_543x3"},
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["sentence"] == "ใช่"
        # Should normalize to (T, 312)
        assert stub.last_features is not None
        assert stub.last_features.shape == (3, 312)
    finally:
        app.dependency_overrides.clear()


def test_v3_checkpoint_dispatch_loads_pose_t5(tmp_path):
    """get_active_translator() prefers PoseT5Translator when pose_t5_config.json exists."""
    import json
    import config as _config

    # Create a minimal v3 directory with pose_t5_config.json
    v3_dir = tmp_path / "pose_t5_v3"
    v3_dir.mkdir()
    (v3_dir / "pose_t5_config.json").write_text(
        json.dumps({"base_model_name": "google/mt5-small", "d_model": 128}),
        encoding="utf-8",
    )

    orig_v3 = _config.SLT_V3_CHECKPOINT_DIR
    _config.SLT_V3_CHECKPOINT_DIR = str(v3_dir)

    saved = appmod._active_translator
    appmod._active_translator = None

    mock_translator = StubTranslator(sentence="v3_result", score=0.99)

    try:
        with patch(
            "tsl.inference.pose_t5_translator.PoseT5Translator.from_checkpoint_dir",
            return_value=mock_translator,
        ) as mock_load:
            translator = get_active_translator()
            mock_load.assert_called_once_with(str(v3_dir))
            assert translator is mock_translator
    finally:
        appmod._active_translator = saved
        _config.SLT_V3_CHECKPOINT_DIR = orig_v3
        app.dependency_overrides.pop(get_active_translator, None)
