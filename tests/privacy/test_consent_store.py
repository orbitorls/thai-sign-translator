"""Tests for consent registry."""
from __future__ import annotations

from tsl.privacy.consent_store import ConsentStore


def test_record_and_status(tmp_path):
    store = ConsentStore(tmp_path)
    store.record_consent(
        user_hash="hash-a",
        scope="model_improvement",
        granted=True,
        source="settings_toggle",
    )
    store.record_consent(
        user_hash="hash-a",
        scope="model_improvement",
        granted=False,
        source="withdrawal",
    )
    store.record_consent(
        user_hash="hash-a",
        scope="service",
        granted=True,
        source="consent_modal",
    )
    status = store.current_status("hash-a")
    assert status["service"] is True
    assert status["model_improvement"] is False
    assert status["video_research"] is False


def test_has_scope(tmp_path):
    store = ConsentStore(tmp_path)
    store.record_consent(
        user_hash="hash-b",
        scope="video_research",
        granted=True,
        source="api",
    )
    assert store.has_scope("hash-b", "video_research") is True
    assert store.has_scope("hash-b", "model_improvement") is False


def test_record_withdrawal(tmp_path):
    store = ConsentStore(tmp_path)
    store.record_consent(
        user_hash="hash-c",
        scope="model_improvement",
        granted=True,
        source="api",
    )
    records = store.record_withdrawal(
        user_hash="hash-c",
        scopes=["model_improvement", "video_research"],
    )
    assert len(records) == 2
    assert store.has_scope("hash-c", "model_improvement") is False
