from __future__ import annotations

from fastapi.testclient import TestClient

from tsl.api.app import app


def _client():
    return TestClient(app)


def test_predict_features_alias_uses_translate(monkeypatch):
    monkeypatch.setattr(
        "tsl.api.app.get_translator_for",
        lambda model=None: type(
            "Translator",
            (),
            {"translate": staticmethod(lambda features: type("Pred", (), {"sentence": "ok", "score": 0.9})())},
        )(),
    )
    resp = _client().post(
        "/predict/features",
        json={
            "frames": [[0.0] * 312, [0.0] * 312],
            "feature_schema": "selected_312",
            "model": None,
            "max_len": 32,
        },
    )
    assert resp.status_code == 200
    assert resp.json()["sentence"] == "ok"


def test_predict_video_alias_uses_translate_video(monkeypatch):
    from tsl.api.app import get_active_translator

    monkeypatch.setattr("tsl.api.app._extract_from_video", lambda path: __import__("numpy").zeros((2, 543, 3), dtype="float32"))
    app.dependency_overrides[get_active_translator] = lambda: type(
        "Translator",
        (),
        {"translate": staticmethod(lambda features: type("Pred", (), {"sentence": "ok", "score": 0.8})())},
    )()
    try:
        resp = _client().post(
            "/predict/video",
            content=b"fake-video",
            headers={"content-type": "video/mp4"},
        )
    finally:
        app.dependency_overrides.pop(get_active_translator, None)

    assert resp.status_code == 200
    assert resp.json()["text"] == "ok"
    assert resp.json()["num_frames"] == 2


def test_predict_video_alias_rejects_non_video_content_type():
    resp = _client().post(
        "/predict/video",
        content=b"not-video",
        headers={"content-type": "application/json"},
    )

    assert resp.status_code == 415
