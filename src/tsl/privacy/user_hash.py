"""Pseudonymous user identity via HMAC."""
from __future__ import annotations

import hashlib
import hmac

import config

_DEV_FALLBACK_KEY = b"dev-only-consent-hmac-key-do-not-use-in-prod"


def _hmac_key() -> bytes:
    key = config.TSL_CONSENT_HMAC_KEY
    if key:
        return key.encode("utf-8")
    return _DEV_FALLBACK_KEY


def compute_user_hash(client_user_id: str) -> str:
    """Return a stable pseudonymous id for a client-generated user id."""
    cleaned = client_user_id.strip()
    if not cleaned:
        raise ValueError("client_user_id must not be empty")
    digest = hmac.new(_hmac_key(), cleaned.encode("utf-8"), hashlib.sha256)
    return digest.hexdigest()
