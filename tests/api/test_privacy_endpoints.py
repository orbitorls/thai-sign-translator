"""Tests for privacy API endpoints."""
from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

import tsl.api.app as appmod
from tsl.api.app import app, get_consent_store, get_contribution_store
from tsl.feedback.rate_limit import FeedbackRateLimiter
from tsl.feedback.store import ContributionStore, MIN_FRAMES
from tsl.privacy.consent_store import ConsentStore
from tsl.privacy.user_hash import compute_user_hash

TEST_USER_ID = "privacy-test-user"
USER_HASH = compute_user_hash(TEST_USER_ID)


def _client() -> TestClient:
    return TestClient(app)


def _user_headers(**extra: str) -> dict[str, str]:
    headers = {"X-User-Id": TEST_USER_ID}
    headers.update(extra)
    return headers


def _frames(count: int = MIN_FRAMES) -> list:
    return np.zeros((count, 543, 3), dtype=np.float32).tolist()


@pytest.fixture(autouse=True)
def isolated_stores(tmp_path, monkeypatch):
    contrib = ContributionStore(tmp_path / "contributions")
    consent = ConsentStore(tmp_path / "consent")
    app.dependency_overrides[get_contribution_store] = lambda: contrib
    app.dependency_overrides[get_consent_store] = lambda: consent
    monkeypatch.setattr(appmod, "_feedback_rate_limiter", FeedbackRateLimiter(max_per_hour=20))
    yield contrib, consent
    app.dependency_overrides.clear()


def test_consent_update_and_status(isolated_stores):
    client = _client()
    resp = client.post(
        "/privacy/consent",
        headers=_user_headers(),
        json={
            "scope": "model_improvement",
            "granted": True,
            "source": "settings_toggle",
        },
    )
    assert resp.status_code == 200
    status = client.get("/privacy/consent/status", headers=_user_headers())
    assert status.status_code == 200
    assert status.json()["scopes"]["model_improvement"] is True


def test_delete_data_removes_contributions(isolated_stores):
    contrib, consent = isolated_stores
    client = _client()
    consent.record_consent(
        user_hash=USER_HASH,
        scope="model_improvement",
        granted=True,
        source="api",
    )
    client.post(
        "/feedback/teach",
        headers={
            **_user_headers(),
            "X-Feedback-Consent": "true",
            "X-Session-Id": "sess",
        },
        json={"frames": _frames(), "label_text": "ทดสอบ"},
    )
    assert contrib.total_count() == 1
    deleted = client.post("/privacy/delete-data", headers=_user_headers())
    assert deleted.status_code == 200
    assert deleted.json()["deleted_samples"] == 1
    assert contrib.total_count() == 0


def test_translate_requires_service_consent(isolated_stores):
    client = _client()
    resp = client.post(
        "/translate",
        headers=_user_headers(),
        json={"frames": _frames()},
    )
    assert resp.status_code == 403


def test_translate_with_service_consent_header(isolated_stores):
    client = _client()
    resp = client.post(
        "/translate",
        headers={**_user_headers(), "X-Service-Consent": "true"},
        json={"frames": []},
    )
    assert resp.status_code == 200
