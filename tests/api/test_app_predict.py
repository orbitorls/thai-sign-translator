import numpy as np
from fastapi.testclient import TestClient
import tsl.api.app as appmod
from tsl.api.app import app, get_recognizer


class StubRecognizer:
    def __init__(self):
        self.last_seq = None

    def recognize(self, seq_norm):
        self.last_seq = seq_norm
        return {
            "word": "hello",
            "score": 0.9,
            "topk": [("hello", 0.9), ("bye", 0.1)],
        }


def test_predict_returns_predict_response_shape(monkeypatch):
    monkeypatch.setattr(appmod, "normalize_sequence", lambda seq: np.zeros((seq.shape[0], 6), dtype=np.float32))
    stub = StubRecognizer()
    app.dependency_overrides[get_recognizer] = lambda: stub
    try:
        client = TestClient(app)
        frame = [[0.0, 0.0, 0.0] for _ in range(543)]
        resp = client.post("/predict", json={"frames": [frame, frame, frame]})
        assert resp.status_code == 200
        body = resp.json()
        assert body["word"] == "hello"
        assert body["score"] == 0.9
        assert body["topk"] == [
            {"word": "hello", "score": 0.9},
            {"word": "bye", "score": 0.1},
        ]
        assert stub.last_seq is not None
        assert stub.last_seq.shape == (3, 6)
    finally:
        app.dependency_overrides.clear()


def test_app_imports_without_checkpoint():
    assert app is not None
