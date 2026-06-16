"""Tests for tsl.data.unified — the v3-312 enforcing manifest loader."""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from tsl.data.unified import (
    MANIFEST_FILENAME,
    load_features,
    load_manifest,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_npy(root, rel_path: str, shape=(5, 312)) -> str:
    """Write a tiny .npy file and return the relative path."""
    full_path = root / rel_path
    full_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(full_path, np.zeros(shape, dtype=np.float32))
    return rel_path


def _write_manifest(root, rows: list[dict]) -> None:
    pd.DataFrame(rows).to_csv(root / MANIFEST_FILENAME, index=False)


def _valid_row(root, idx: int, split: str = "train", source: str = "test_src") -> dict:
    """Create one valid v3-312 manifest row with a real .npy file."""
    seg_id = f"seg_{idx:03d}"
    rel_path = f"landmarks/{seg_id}.npy"
    _make_npy(root, rel_path)
    return {
        "segment_id": seg_id,
        "npy_path": rel_path,
        "text": f"ข้อความ {idx}",
        "video_id": f"video_{idx}",
        "split": split,
        "source": source,
        "feature_layout_version": "v3-312",
    }


# ---------------------------------------------------------------------------
# load_manifest — happy path
# ---------------------------------------------------------------------------

def test_load_manifest_returns_all_valid_rows(tmp_path):
    rows = [_valid_row(tmp_path, i) for i in range(4)]
    _write_manifest(tmp_path, rows)

    examples = load_manifest(str(tmp_path))

    assert len(examples) == 4
    assert all(ex.source == "test_src" for ex in examples)


def test_load_manifest_metadata_contains_feature_layout_version(tmp_path):
    rows = [_valid_row(tmp_path, 0)]
    _write_manifest(tmp_path, rows)

    examples = load_manifest(str(tmp_path))

    assert examples[0].metadata["feature_layout_version"] == "v3-312"


# ---------------------------------------------------------------------------
# load_manifest — version enforcement
# ---------------------------------------------------------------------------

def test_load_manifest_raises_on_wrong_feature_layout_version(tmp_path):
    row = _valid_row(tmp_path, 0)
    row["feature_layout_version"] = "v2-162"  # wrong version
    _write_manifest(tmp_path, [row])

    with pytest.raises(ValueError, match="v3-312"):
        load_manifest(str(tmp_path))


def test_load_manifest_raises_on_missing_feature_layout_version_column(tmp_path):
    row = _valid_row(tmp_path, 0)
    del row["feature_layout_version"]
    _write_manifest(tmp_path, [row])

    with pytest.raises(ValueError, match="missing columns"):
        load_manifest(str(tmp_path))


def test_load_manifest_accepts_v3_312_with_extra_suffix(tmp_path):
    """Versions like 'v3-312-extra' still start with 'v3-312' and are valid."""
    row = _valid_row(tmp_path, 0)
    row["feature_layout_version"] = "v3-312-extra"
    _write_manifest(tmp_path, [row])

    examples = load_manifest(str(tmp_path))
    assert len(examples) == 1


# ---------------------------------------------------------------------------
# load_manifest — skipping rows
# ---------------------------------------------------------------------------

def test_load_manifest_skips_empty_text(tmp_path):
    rows = [_valid_row(tmp_path, i) for i in range(3)]
    rows[1]["text"] = ""  # should be skipped
    _write_manifest(tmp_path, rows)

    examples = load_manifest(str(tmp_path))

    assert len(examples) == 2


def test_load_manifest_skips_missing_npy_file(tmp_path):
    rows = [_valid_row(tmp_path, i) for i in range(3)]
    rows[2]["npy_path"] = "landmarks/does_not_exist.npy"  # no file on disk
    _write_manifest(tmp_path, rows)

    examples = load_manifest(str(tmp_path))

    assert len(examples) == 2


# ---------------------------------------------------------------------------
# load_manifest — filtering
# ---------------------------------------------------------------------------

def test_load_manifest_filters_by_split(tmp_path):
    rows = [
        _valid_row(tmp_path, 0, split="train"),
        _valid_row(tmp_path, 1, split="train"),
        _valid_row(tmp_path, 2, split="val"),
    ]
    _write_manifest(tmp_path, rows)

    train = load_manifest(str(tmp_path), split="train")
    val = load_manifest(str(tmp_path), split="val")

    assert len(train) == 2
    assert len(val) == 1
    assert all(ex.split == "train" for ex in train)
    assert all(ex.split == "val" for ex in val)


def test_load_manifest_filters_by_source(tmp_path):
    rows = [
        _valid_row(tmp_path, 0, source="src_a"),
        _valid_row(tmp_path, 1, source="src_b"),
        _valid_row(tmp_path, 2, source="src_a"),
    ]
    _write_manifest(tmp_path, rows)

    src_a = load_manifest(str(tmp_path), source="src_a")
    src_b = load_manifest(str(tmp_path), source="src_b")

    assert len(src_a) == 2
    assert len(src_b) == 1


def test_load_manifest_filters_by_split_and_source(tmp_path):
    rows = [
        _valid_row(tmp_path, 0, split="train", source="src_a"),
        _valid_row(tmp_path, 1, split="val", source="src_a"),
        _valid_row(tmp_path, 2, split="train", source="src_b"),
    ]
    _write_manifest(tmp_path, rows)

    result = load_manifest(str(tmp_path), split="train", source="src_a")

    assert len(result) == 1
    assert result[0].example_id == "seg_000"


# ---------------------------------------------------------------------------
# load_manifest — missing manifest file
# ---------------------------------------------------------------------------

def test_load_manifest_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError, match="manifest not found"):
        load_manifest(str(tmp_path))


# ---------------------------------------------------------------------------
# load_features — happy path
# ---------------------------------------------------------------------------

def test_load_features_returns_t_312_float32(tmp_path):
    npy_path = tmp_path / "features.npy"
    np.save(npy_path, np.ones((10, 312), dtype=np.float32))

    arr = load_features(str(npy_path))

    assert arr.shape == (10, 312)
    assert arr.dtype == np.float32


def test_load_features_converts_to_float32(tmp_path):
    npy_path = tmp_path / "features.npy"
    np.save(npy_path, np.ones((7, 312), dtype=np.float64))

    arr = load_features(str(npy_path))

    assert arr.dtype == np.float32


# ---------------------------------------------------------------------------
# load_features — shape validation
# ---------------------------------------------------------------------------

def test_load_features_raises_on_wrong_dim(tmp_path):
    npy_path = tmp_path / "bad_features.npy"
    np.save(npy_path, np.ones((10, 162), dtype=np.float32))  # wrong dim

    with pytest.raises(ValueError, match="312"):
        load_features(str(npy_path))


def test_load_features_raises_on_3d_array(tmp_path):
    npy_path = tmp_path / "bad_features.npy"
    np.save(npy_path, np.ones((10, 312, 1), dtype=np.float32))  # 3D not 2D

    with pytest.raises(ValueError, match="312"):
        load_features(str(npy_path))


def test_load_features_raises_on_1d_array(tmp_path):
    npy_path = tmp_path / "bad_features.npy"
    np.save(npy_path, np.ones((312,), dtype=np.float32))  # 1D

    with pytest.raises(ValueError, match="312"):
        load_features(str(npy_path))
