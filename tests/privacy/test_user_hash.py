"""Tests for pseudonymous user hashing."""
from __future__ import annotations

import pytest

from tsl.privacy.user_hash import compute_user_hash


def test_compute_user_hash_deterministic(monkeypatch):
    monkeypatch.setattr("config.TSL_CONSENT_HMAC_KEY", "test-secret")
    first = compute_user_hash("user-123")
    second = compute_user_hash("user-123")
    assert first == second
    assert first != compute_user_hash("user-456")


def test_compute_user_hash_rejects_empty():
    with pytest.raises(ValueError):
        compute_user_hash("   ")
