import numpy as np
from fastapi.testclient import TestClient
import tsl.api.app as appmod
from tsl.api.app import app, get_store


class StubStore:
    def __init__(self):
        self._signs: dict[str, list] = {"existing": []}

    def add_sign(self, name, clips):
        self._signs[name] = clips

    def names(self):
        return list(self._signs.keys())


def test_train_custom_sign_increases_total_signs(monkeypatch):
    monkeypatch.setattr(appmod, "normalize_sequence", lambda seq: np.zeros((seq.shape[0], 6), dtype=np.float32))
    stub = StubStore()
    app.dependency_overrides[get_store] = lambda: stub
    try:
        client = TestClient(app)
        frame = [[0.0, 0.0, 0.0] for _ in range(543)]
        clip = [frame, frame]
        resp = client.post("/train-custom-sign", json={"name": "cat", "clips": [clip, clip]})
        assert resp.status_code == 200
        body = resp.json()
        assert body["name"] == "cat"
        assert body["num_clips"] == 2
        assert body["total_signs"] == 2
        assert stub._signs["cat"][0].shape == (2, 6)
    finally:
        app.dependency_overrides.clear()


def test_signs_lists_names():
    stub = StubStore()
    app.dependency_overrides[get_store] = lambda: stub
    try:
        client = TestClient(app)
        resp = client.get("/signs")
        assert resp.status_code == 200
        assert resp.json() == {"signs": ["existing"]}
    finally:
        app.dependency_overrides.clear()
