"""Tests for the GET /models endpoint."""
from __future__ import annotations

from unittest.mock import patch
import pytest
from fastapi.testclient import TestClient

from tsl.api.app import app
import tsl.api.model_catalog as catalog_mod


def _client():
    return TestClient(app)


def test_models_endpoint_returns_200():
    """GET /models always returns 200."""
    resp = _client().get("/models")
    assert resp.status_code == 200


def test_models_endpoint_returns_default_id():
    """Response includes 'default' key matching v3_poset5."""
    resp = _client().get("/models")
    body = resp.json()
    assert body["default"] == "v3_poset5"


def test_models_endpoint_contains_all_three_models():
    """All three catalog models appear in the response."""
    resp = _client().get("/models")
    ids = [m["id"] for m in resp.json()["models"]]
    assert set(ids) == {"v3_poset5", "v2_slt", "combined"}


def test_models_endpoint_has_required_fields():
    """Each model entry has all required fields."""
    resp = _client().get("/models")
    for m in resp.json()["models"]:
        assert "id" in m
        assert "label_th" in m
        assert "label_en" in m
        assert "architecture" in m
        assert "available" in m
        assert "default" in m


def test_models_endpoint_marks_default_model():
    """Exactly one model has default=True and it matches the 'default' key."""
    resp = _client().get("/models")
    body = resp.json()
    defaults = [m for m in body["models"] if m["default"]]
    assert len(defaults) == 1
    assert defaults[0]["id"] == body["default"]


def test_models_endpoint_unavailable_when_checkpoint_missing(tmp_path, monkeypatch):
    """A model with a missing checkpoint_dir is marked available=False."""
    # Point v2_slt at a non-existent directory
    original_catalog = catalog_mod._CATALOG
    patched = [
        s if s.id != "v2_slt"
        else s.__class__(
            id=s.id,
            label_th=s.label_th,
            label_en=s.label_en,
            architecture=s.architecture,
            checkpoint_dir=str(tmp_path / "nonexistent"),
            default=s.default,
        )
        for s in original_catalog
    ]
    monkeypatch.setattr(catalog_mod, "_CATALOG", patched)

    resp = _client().get("/models")
    assert resp.status_code == 200
    v2 = next(m for m in resp.json()["models"] if m["id"] == "v2_slt")
    assert v2["available"] is False
