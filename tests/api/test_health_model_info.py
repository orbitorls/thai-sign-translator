from __future__ import annotations

from fastapi.testclient import TestClient

from tsl.api.app import app


def _client():
    return TestClient(app)


def test_health_endpoint_returns_model_status(monkeypatch):
    monkeypatch.setattr(
        "tsl.api.app._active_model_spec",
        lambda: type("Spec", (), {"id": "v3_poset5", "architecture": "pose_t5"})(),
    )
    monkeypatch.setattr("tsl.api.app.availability", lambda spec: True)
    monkeypatch.setattr("tsl.api.app.resolve_checkpoint_dir", lambda spec: "checkpoints/demo")
    monkeypatch.setattr(
        "tsl.api.app.validate_model_dir",
        lambda path: type("Meta", (), {"feature_dim": 312})(),
    )

    resp = _client().get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"
    assert resp.json()["feature_dim"] == 312


def test_model_info_returns_generation_config(monkeypatch):
    monkeypatch.setattr(
        "tsl.api.app._active_model_spec",
        lambda: type("Spec", (), {"id": "v3_poset5", "architecture": "pose_t5"})(),
    )
    monkeypatch.setattr("tsl.api.app.availability", lambda spec: True)
    monkeypatch.setattr("tsl.api.app.resolve_checkpoint_dir", lambda spec: "checkpoints/demo")
    monkeypatch.setattr(
        "tsl.api.app.validate_model_dir",
        lambda path: type("Meta", (), {"feature_dim": 312, "decode_config": {"beam_size": 5}})(),
    )

    resp = _client().get("/model-info")
    assert resp.status_code == 200
    assert resp.json()["generation_config"]["beam_size"] == 5
