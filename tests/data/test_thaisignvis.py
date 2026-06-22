"""Tests for tsl.data.thaisignvis loader."""
import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from tsl.data.thaisignvis import (
    FEATURE_DIM,
    MANIFEST_FILENAME,
    load_thaisignvis_features,
    load_thaisignvis_manifest,
)


def _make_manifest(tmp: str, n: int = 6, splits=("train", "val")) -> str:
    lm_dir = os.path.join(tmp, "landmarks")
    os.makedirs(lm_dir, exist_ok=True)
    rows = []
    for i in range(n):
        seg_id = f"seg_{i:03d}"
        npy_rel = os.path.join("landmarks", f"{seg_id}.npy")
        npy_abs = os.path.join(tmp, npy_rel)
        T = 10 + i
        arr = np.random.randn(T, FEATURE_DIM).astype(np.float32)
        np.save(npy_abs, arr)
        split = splits[i % len(splits)]
        rows.append(
            {
                "segment_id": seg_id,
                "npy_path": npy_rel,
                "text": f"ประโยค {i}",
                "video_id": f"vid_{i}",
                "start_ms": i * 1000,
                "end_ms": (i + 1) * 1000,
                "split": split,
            }
        )
    pd.DataFrame(rows).to_csv(os.path.join(tmp, MANIFEST_FILENAME), index=False, encoding="utf-8")
    return tmp


def test_load_manifest_all():
    with tempfile.TemporaryDirectory() as tmp:
        _make_manifest(tmp, n=6)
        examples = load_thaisignvis_manifest(tmp)
        assert len(examples) == 6
        for ex in examples:
            assert ex.source == "thaisignvis"
            assert ex.target_text.startswith("ประโยค")


def test_load_manifest_split_filter():
    with tempfile.TemporaryDirectory() as tmp:
        _make_manifest(tmp, n=6)
        train = load_thaisignvis_manifest(tmp, split="train")
        val = load_thaisignvis_manifest(tmp, split="val")
        assert len(train) + len(val) == 6
        assert all(e.split == "train" for e in train)
        assert all(e.split == "val" for e in val)


def test_load_manifest_preserves_source_and_feature_layout_version():
    with tempfile.TemporaryDirectory() as tmp:
        _make_manifest(tmp, n=1)
        manifest_path = os.path.join(tmp, MANIFEST_FILENAME)
        df = pd.read_csv(manifest_path)
        df["source"] = "thaisignvis"
        df["feature_layout_version"] = "v3-312"
        df.to_csv(manifest_path, index=False, encoding="utf-8")

        examples = load_thaisignvis_manifest(tmp)

        assert examples[0].source == "thaisignvis"
        assert examples[0].metadata["feature_layout_version"] == "v3-312"


def test_load_manifest_missing_npy_skipped():
    with tempfile.TemporaryDirectory() as tmp:
        _make_manifest(tmp, n=4)
        # Delete one npy
        os.remove(os.path.join(tmp, "landmarks", "seg_001.npy"))
        examples = load_thaisignvis_manifest(tmp)
        assert len(examples) == 3


def test_load_manifest_no_manifest_raises():
    with tempfile.TemporaryDirectory() as tmp:
        with pytest.raises(FileNotFoundError, match="manifest not found"):
            load_thaisignvis_manifest(tmp)


def test_load_features_shape():
    with tempfile.TemporaryDirectory() as tmp:
        _make_manifest(tmp, n=2)
        examples = load_thaisignvis_manifest(tmp)
        for ex in examples:
            arr = load_thaisignvis_features(ex.features_path)
            assert arr.ndim == 2
            assert arr.shape[1] == FEATURE_DIM
            assert arr.dtype == np.float32


def test_load_features_wrong_dim_raises():
    with tempfile.TemporaryDirectory() as tmp:
        bad_npy = os.path.join(tmp, "bad.npy")
        np.save(bad_npy, np.zeros((5, 100), dtype=np.float32))
        with pytest.raises(ValueError, match="unexpected shape"):
            load_thaisignvis_features(bad_npy)


def test_manifest_missing_columns_raises():
    with tempfile.TemporaryDirectory() as tmp:
        pd.DataFrame([{"segment_id": "x", "text": "y"}]).to_csv(
            os.path.join(tmp, MANIFEST_FILENAME), index=False
        )
        with pytest.raises(ValueError, match="missing columns"):
            load_thaisignvis_manifest(tmp)
