"""Tests for the ``POST /translate-sentence`` endpoint."""
from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient

import tsl.api.app as appmod
from tsl.api.app import app, get_sentence_runtime
from tsl.features.schema import RAW_MEDIAPIPE_543X3, TSL51_162
from tsl.inference.sentence_runtime import FeatureSchemaMismatchError


_DEFAULT_FEATURE_DIM = 162


class StubSentenceRuntime:
    def __init__(self, prediction=None, error: Exception | None = None) -> None:
        self.last_frames = None
        self.last_feature_schema = None
        self.last_max_len = None
        self._prediction = prediction or SimpleNamespace(
            sentence="hello",
            score=0.75,
        )
        self._error = error

    def translate(self, frames, feature_schema: str, max_len: int = 128):
        if self._error is not None:
            raise self._error
        self.last_frames = np.asarray(frames, dtype=np.float32)
        self.last_feature_schema = feature_schema
        self.last_max_len = max_len
        return self._prediction


def test_translate_sentence_returns_expected_shape():
    stub = StubSentenceRuntime()
    app.dependency_overrides[get_sentence_runtime] = lambda: stub
    try:
        client = TestClient(app)
        frame1 = [0.0] * _DEFAULT_FEATURE_DIM
        frame2 = [0.1] * _DEFAULT_FEATURE_DIM
        resp = client.post(
            "/translate-sentence",
            json={"frames": [frame1, frame2], "feature_schema": TSL51_162},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["sentence"] == "hello"
        assert isinstance(body["score"], float)
        assert 0.0 <= body["score"] <= 1.0
        assert stub.last_frames is not None
        assert stub.last_frames.shape == (2, _DEFAULT_FEATURE_DIM)
        assert stub.last_feature_schema == TSL51_162
        assert stub.last_max_len == 128
    finally:
        app.dependency_overrides.clear()


def test_translate_sentence_empty_frames():
    stub = StubSentenceRuntime()
    app.dependency_overrides[get_sentence_runtime] = lambda: stub
    try:
        client = TestClient(app)
        resp = client.post("/translate-sentence", json={"frames": [], "feature_schema": TSL51_162})
        assert resp.status_code == 200
        body = resp.json()
        assert body == {"sentence": "", "score": 0.0}
    finally:
        app.dependency_overrides.clear()


def test_translate_sentence_schema_mismatch_returns_400():
    stub = StubSentenceRuntime(
        error=FeatureSchemaMismatchError(
            requested_schema=RAW_MEDIAPIPE_543X3,
            model_schema=TSL51_162,
            message="raw schema does not match the active model",
        )
    )
    app.dependency_overrides[get_sentence_runtime] = lambda: stub
    try:
        client = TestClient(app)
        raw_frame = [[0.0, 0.0, 0.0] for _ in range(543)]
        resp = client.post(
            "/translate-sentence",
            json={"frames": [raw_frame, raw_frame], "feature_schema": RAW_MEDIAPIPE_543X3},
        )
        assert resp.status_code == 400
        assert "raw schema" in resp.json()["detail"]
    finally:
        app.dependency_overrides.clear()


def test_translate_sentence_missing_checkpoint_returns_503():
    if not hasattr(appmod, "get_sentence_runtime"):
        pytest.skip("get_sentence_runtime not present in app module")

    import config as _config

    ckpt_dir = getattr(_config, "SLT_CHECKPOINT_DIR", None)
    files = {"slt_model.pt", "tokenizer.json", "model_config.json"}
    checkpoint_ready = bool(
        ckpt_dir and os.path.isdir(ckpt_dir) and files.issubset(set(os.listdir(ckpt_dir)))
    )

    if checkpoint_ready:
        app.dependency_overrides.pop(get_sentence_runtime, None)
        saved = appmod._sentence_runtime
        appmod._sentence_runtime = None
        try:
            client = TestClient(app)
            frame = [0.0] * _DEFAULT_FEATURE_DIM
            resp = client.post(
                "/translate-sentence",
                json={"frames": [frame, frame], "feature_schema": TSL51_162},
            )
            assert resp.status_code == 200
            body = resp.json()
            assert isinstance(body["sentence"], str)
            assert isinstance(body["score"], float)
        finally:
            appmod._sentence_runtime = saved
            app.dependency_overrides.pop(get_sentence_runtime, None)
    else:
        app.dependency_overrides.pop(get_sentence_runtime, None)
        saved = appmod._sentence_runtime
        appmod._sentence_runtime = None
        try:
            client = TestClient(app)
            frame = [0.0] * _DEFAULT_FEATURE_DIM
            resp = client.post(
                "/translate-sentence",
                json={"frames": [frame, frame], "feature_schema": TSL51_162},
            )
            assert resp.status_code == 503
            assert "checkpoint" in resp.json()["detail"].lower()
        finally:
            appmod._sentence_runtime = saved
            app.dependency_overrides.pop(get_sentence_runtime, None)


def test_app_imports_without_checkpoint():
    assert app is not None
