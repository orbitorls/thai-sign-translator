"""Tests for the ``POST /translate-sentence`` endpoint."""
from __future__ import annotations

import os

import numpy as np
import pytest
from fastapi.testclient import TestClient

import tsl.api.app as appmod
from tsl.api.app import app, get_sentence_translator
from tsl.inference.sentence_translator import SentencePrediction


_DEFAULT_FEATURE_DIM = 162


class StubTranslator:
    def __init__(self, prediction: SentencePrediction | None = None) -> None:
        self.last_features = None
        self.last_max_len = None
        self._prediction = prediction or SentencePrediction(
            sentence="สวัสดี", token_ids=[1, 5, 6, 2], score=0.75
        )

    def translate(self, features, max_len: int = 128) -> SentencePrediction:
        self.last_features = features
        self.last_max_len = max_len
        return self._prediction


def test_translate_sentence_returns_expected_shape():
    stub = StubTranslator()
    app.dependency_overrides[get_sentence_translator] = lambda: stub
    try:
        client = TestClient(app)
        frame1 = [0.0] * _DEFAULT_FEATURE_DIM
        frame2 = [0.1] * _DEFAULT_FEATURE_DIM
        resp = client.post(
            "/translate-sentence",
            json={"frames": [frame1, frame2]},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["sentence"] == "สวัสดี"
        assert isinstance(body["tokens"], list)
        assert body["tokens"] == [1, 5, 6, 2]
        assert isinstance(body["score"], float)
        assert 0.0 <= body["score"] <= 1.0
        assert stub.last_features is not None
        assert stub.last_features.shape == (2, _DEFAULT_FEATURE_DIM)
        assert stub.last_max_len == 128
    finally:
        app.dependency_overrides.clear()


def test_translate_sentence_empty_frames():
    stub = StubTranslator()
    app.dependency_overrides[get_sentence_translator] = lambda: stub
    try:
        client = TestClient(app)
        resp = client.post("/translate-sentence", json={"frames": []})
        assert resp.status_code == 200
        body = resp.json()
        assert body == {"sentence": "", "tokens": [], "score": 0.0}
    finally:
        app.dependency_overrides.clear()


def test_translate_sentence_wrong_dim_returns_400():
    stub = StubTranslator()
    app.dependency_overrides[get_sentence_translator] = lambda: stub
    try:
        client = TestClient(app)
        bad_frame = [0.0] * 100
        resp = client.post(
            "/translate-sentence",
            json={"frames": [bad_frame, bad_frame], "feature_dim": 162},
        )
        assert resp.status_code == 400
        assert "feature_dim" in resp.json()["detail"]
    finally:
        app.dependency_overrides.clear()


def test_translate_sentence_missing_checkpoint_returns_503():
    if not hasattr(appmod, "get_sentence_translator"):
        pytest.skip("get_sentence_translator not present in app module")

    import config as _config

    ckpt_dir = getattr(_config, "SLT_CHECKPOINT_DIR", None)
    if ckpt_dir and os.path.isdir(ckpt_dir):
        files = {"slt_model.pt", "tokenizer.json", "model_config.json"}
        if files.issubset(set(os.listdir(ckpt_dir))):
            pytest.skip("SLT checkpoint already present; cannot simulate 503")

    app.dependency_overrides.pop(get_sentence_translator, None)
    saved = appmod._translator
    appmod._translator = None
    try:
        client = TestClient(app)
        frame = [0.0] * _DEFAULT_FEATURE_DIM
        resp = client.post("/translate-sentence", json={"frames": [frame, frame]})
        assert resp.status_code == 503
        assert "checkpoint" in resp.json()["detail"].lower()
    finally:
        appmod._translator = saved
        app.dependency_overrides.pop(get_sentence_translator, None)


def test_app_imports_without_checkpoint():
    assert app is not None
