"""Tests for user feedback API endpoints."""
from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

import tsl.api.app as appmod
from tsl.api.app import app, get_contribution_store, get_consent_store
from tsl.feedback.rate_limit import FeedbackRateLimiter
from tsl.feedback.store import ContributionStore, MAX_FRAMES, MIN_FRAMES
from tsl.privacy.consent_store import ConsentStore
from tsl.serving.cache import get_translator_cache

TEST_USER_ID = "feedback-test-user"


def _client() -> TestClient:
    return TestClient(app)


def _frames(count: int = MIN_FRAMES) -> list:
    return np.zeros((count, 543, 3), dtype=np.float32).tolist()


def _frames_v4(count: int = MIN_FRAMES) -> list:
    arr = np.zeros((count, 543, 4), dtype=np.float32)
    arr[:, :, 3] = 1.0
    return arr.tolist()


def _consent_headers(**extra: str) -> dict[str, str]:
    headers = {
        "X-Feedback-Consent": "true",
        "X-Session-Id": "test-session",
        "X-User-Id": TEST_USER_ID,
    }
    headers.update(extra)
    return headers


@pytest.fixture(autouse=True)
def isolated_feedback_store(tmp_path, monkeypatch):
    store = ContributionStore(tmp_path / "contributions")
    consent_store = ConsentStore(tmp_path / "consent")
    app.dependency_overrides[get_contribution_store] = lambda: store
    app.dependency_overrides[get_consent_store] = lambda: consent_store
    monkeypatch.setattr(appmod, "_feedback_rate_limiter", FeedbackRateLimiter(max_per_hour=20))
    yield store
    app.dependency_overrides.clear()


def test_correction_requires_consent_header():
    resp = _client().post(
        "/feedback/correction",
        headers={"X-User-Id": TEST_USER_ID},
        json={
            "frames": _frames(),
            "predicted_text": "ผิด",
            "corrected_text": "ถูก",
        },
    )
    assert resp.status_code == 403


def test_correction_and_teach_success(isolated_feedback_store):
    client = _client()
    correction = client.post(
        "/feedback/correction",
        headers=_consent_headers(),
        json={
            "frames": _frames(),
            "predicted_text": "ผิด",
            "corrected_text": "ถูกต้อง",
            "model": "pose_t5",
            "score": 0.4,
        },
    )
    assert correction.status_code == 200
    assert correction.json()["kind"] == "correction"

    teach = client.post(
        "/feedback/teach",
        headers=_consent_headers(),
        json={
            "frames": _frames(),
            "label_text": "ประโยคใหม่",
        },
    )
    assert teach.status_code == 200
    assert teach.json()["kind"] == "teach"
    assert isolated_feedback_store.total_count() == 2


def test_feedback_accepts_543x4_and_capture_quality(isolated_feedback_store):
    resp = _client().post(
        "/feedback/teach",
        headers=_consent_headers(),
        json={
            "frames": _frames_v4(),
            "label_text": "ประโยคใหม่",
            "capture_quality": {
                "fps": 180,
                "lighting_ok": True,
                "hand_present": True,
                "warning": None,
                "landmark_quality": 2,
                "feature_schema": "raw_mediapipe_543x4",
                "camera_facing": "environment",
            },
        },
    )
    assert resp.status_code == 200
    row = isolated_feedback_store.iter_meta()[0]
    assert row["capture_quality"]["fps"] == 120.0
    assert row["capture_quality"]["landmark_quality"] == 1.0
    assert row["capture_quality"]["camera_facing"] == "environment"


def test_feedback_stats_endpoint(isolated_feedback_store):
    client = _client()
    client.post(
        "/feedback/teach",
        headers=_consent_headers(),
        json={"frames": _frames(), "label_text": "ทดสอบ"},
    )
    resp = client.get("/feedback/stats")
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["pending_count"] == 1
    assert payload["total_count"] == 1
    assert "model" in payload


def test_rate_limit_blocks_after_threshold(isolated_feedback_store, monkeypatch):
    monkeypatch.setattr(appmod, "_feedback_rate_limiter", FeedbackRateLimiter(max_per_hour=2))
    client = _client()
    for index in range(2):
        resp = client.post(
            "/feedback/teach",
            headers=_consent_headers(),
            json={"frames": _frames(15 + index), "label_text": f"sample {index}"},
        )
        assert resp.status_code == 200
    blocked = client.post(
        "/feedback/teach",
        headers=_consent_headers(),
        json={"frames": _frames(20), "label_text": "blocked"},
    )
    assert blocked.status_code == 429


def test_validation_error_for_short_frames():
    resp = _client().post(
        "/feedback/teach",
        headers=_consent_headers(),
        json={"frames": _frames(MIN_FRAMES - 1), "label_text": "สั้น"},
    )
    assert resp.status_code == 422


def test_validation_rejects_oversized_frames():
    resp = _client().post(
        "/feedback/teach",
        headers=_consent_headers(),
        json={"frames": _frames(MAX_FRAMES + 1), "label_text": "ยาว"},
    )
    assert resp.status_code == 422


def test_duplicate_correction_returns_409(isolated_feedback_store):
    client = _client()
    payload = {
        "frames": _frames(),
        "predicted_text": "ผิด",
        "corrected_text": "ถูก",
    }
    first = client.post("/feedback/correction", headers=_consent_headers(), json=payload)
    assert first.status_code == 200
    dup = client.post("/feedback/correction", headers=_consent_headers(), json=payload)
    assert dup.status_code == 409


def test_rate_limit_not_consumed_on_validation_error(monkeypatch):
    monkeypatch.setattr(appmod, "_feedback_rate_limiter", FeedbackRateLimiter(max_per_hour=1))
    client = _client()
    bad = client.post(
        "/feedback/teach",
        headers=_consent_headers(),
        json={"frames": _frames(MIN_FRAMES - 1), "label_text": "สั้น"},
    )
    assert bad.status_code == 422
    ok = client.post(
        "/feedback/teach",
        headers=_consent_headers(),
        json={"frames": _frames(), "label_text": "พอ"},
    )
    assert ok.status_code == 200


def test_reload_models_requires_admin_token(monkeypatch):
    monkeypatch.delenv("TSL_FEEDBACK_ADMIN_TOKEN", raising=False)
    client = _client()
    assert client.post("/feedback/reload-models").status_code == 503

    monkeypatch.setenv("TSL_FEEDBACK_ADMIN_TOKEN", "secret-token")
    client = _client()
    denied = client.post("/feedback/reload-models")
    assert denied.status_code == 403

    get_translator_cache()["demo"] = object()
    ok = client.post("/feedback/reload-models", headers={"X-Admin-Token": "secret-token"})
    assert ok.status_code == 200
    assert ok.json()["cleared_models"] >= 1
    assert get_translator_cache() == {}
