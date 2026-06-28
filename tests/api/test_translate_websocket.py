"""Tests for the realtime WebSocket translation endpoint."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from fastapi.testclient import TestClient

import tsl.api.app as appmod
from tsl.api.app import app


class StubTranslator:
    def __init__(self, sentence: str = "สด", score: float = 0.91) -> None:
        self.sentence = sentence
        self.score = score
        self.last_features: np.ndarray | None = None

    def translate(self, features: np.ndarray):
        self.last_features = features
        return SimpleNamespace(sentence=self.sentence, score=self.score)


def _client() -> TestClient:
    return TestClient(app)


def _ws_payload(**fields) -> dict:
    payload = dict(fields)
    if payload.get("frames"):
        payload.setdefault("service_consent", True)
        payload.setdefault("user_id", "translate-test-user")
    return payload


def _raw_frames(T: int = 4) -> list:
    return np.zeros((T, 543, 3), dtype=np.float32).tolist()


def test_translate_ws_default_model_returns_result():
    stub = StubTranslator(sentence="เรียลไทม์", score=0.82)
    with patch.object(appmod, "get_translator_for", return_value=stub) as mock_fn:
        with _client().websocket_connect("/ws/translate") as ws:
            ws.send_json(_ws_payload(
                request_id="r1",
                frames=_raw_frames(),
                feature_schema="raw_mediapipe_543x3",
            ))
            body = ws.receive_json()

    assert body["type"] == "result"
    assert body["request_id"] == "r1"
    assert body["sentence"] == "เรียลไทม์"
    assert body["score"] == 0.82
    assert body["model"]
    assert body["latency_ms"] >= 0.0
    mock_fn.assert_called_once_with(None)


def test_translate_ws_explicit_model_and_selected_312():
    stub = StubTranslator(sentence="เลือกโมเดล", score=0.73)
    frames = np.zeros((3, 312), dtype=np.float32).tolist()
    with patch.object(appmod, "get_translator_for", return_value=stub) as mock_fn:
        with _client().websocket_connect("/ws/translate") as ws:
            ws.send_json(_ws_payload(
                request_id="r2",
                frames=frames,
                feature_schema="selected_312",
                model="v2_slt",
            ))
            body = ws.receive_json()

    assert body["type"] == "result"
    assert body["request_id"] == "r2"
    assert body["model"] == "v2_slt"
    assert stub.last_features is not None
    assert stub.last_features.shape == (3, 312)
    mock_fn.assert_called_once_with("v2_slt")


def test_translate_ws_bad_shape_returns_error_and_keeps_socket_open():
    stub = StubTranslator(sentence="ต่อได้", score=0.8)
    with patch.object(appmod, "get_translator_for", return_value=stub):
        with _client().websocket_connect("/ws/translate") as ws:
            ws.send_json(_ws_payload(
                request_id="bad",
                frames=[[1.0, 2.0]],
                feature_schema="raw_mediapipe_543x3",
            ))
            bad = ws.receive_json()
            ws.send_json(_ws_payload(
                request_id="good",
                frames=_raw_frames(2),
                feature_schema="raw_mediapipe_543x3",
            ))
            good = ws.receive_json()

    assert bad["type"] == "error"
    assert bad["request_id"] == "bad"
    assert bad["code"] == 400
    assert good["type"] == "result"
    assert good["request_id"] == "good"


def test_translate_ws_unknown_model_returns_error_message():
    with _client().websocket_connect("/ws/translate") as ws:
        ws.send_json(
            {
                "request_id": "missing",
                "frames": [],
                "feature_schema": "raw_mediapipe_543x3",
                "model": "not_a_model",
            }
        )
        body = ws.receive_json()

    assert body["type"] == "error"
    assert body["request_id"] == "missing"
    assert body["code"] == 400


def test_translate_ws_malformed_json_returns_error_message():
    with _client().websocket_connect("/ws/translate") as ws:
        ws.send_text("{")
        body = ws.receive_json()

    assert body["type"] == "error"
    assert body["request_id"] is None
    assert body["code"] == 400


def test_translate_ws_mock_mode_accepts_any_frame_shape():
    with _client().websocket_connect("/ws/translate") as ws:
        ws.send_json(
            {
                "request_id": "mock",
                "frames": [[1.0, 2.0]],
                "feature_schema": "raw_mediapipe_543x3",
                "model": "mock_v1",
                "mock_mode": True,
                "service_consent": True,
                "user_id": "translate-test-user",
            }
        )
        body = ws.receive_json()

    assert body["type"] == "result"
    assert body["request_id"] == "mock"
    assert body["model"] == "mock_v1"
    assert body["sentence"]
    assert body["score"] == 0.99
