from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from tsl.data.youtube_sl25 import (
    MANIFEST_FILENAME,
    load_youtube_sl25_features,
    load_youtube_sl25_manifest,
)


def _write_manifest(root, n: int = 4) -> None:
    lm_dir = root / "landmarks"
    lm_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for i in range(n):
        segment_id = f"seg_{i:03d}"
        rel_path = f"landmarks/{segment_id}.npy"
        np.save(root / rel_path, np.full((5 + i, 162), i, dtype=np.float32))
        rows.append(
            {
                "segment_id": segment_id,
                "npy_path": rel_path,
                "text": f"text {i}",
                "video_id": f"video_{i}",
                "start_ms": i * 1000,
                "end_ms": (i + 1) * 1000,
                "split": "val" if i == 0 else "train",
            }
        )

    pd.DataFrame(rows).to_csv(root / MANIFEST_FILENAME, index=False)


def test_load_youtube_sl25_manifest_filters_by_split(tmp_path):
    _write_manifest(tmp_path, n=4)

    train = load_youtube_sl25_manifest(str(tmp_path), split="train")
    val = load_youtube_sl25_manifest(str(tmp_path), split="val")

    assert len(train) == 3
    assert len(val) == 1
    assert all(example.source == "youtube_sl25" for example in train + val)


def test_load_youtube_sl25_features_reads_2d_float32(tmp_path):
    path = tmp_path / "segment.npy"
    np.save(path, np.ones((3, 162), dtype=np.float32))

    arr = load_youtube_sl25_features(str(path))

    assert arr.shape == (3, 162)
    assert arr.dtype == np.float32


def test_load_youtube_sl25_manifest_requires_manifest(tmp_path):
    with pytest.raises(FileNotFoundError, match="manifest not found"):
        load_youtube_sl25_manifest(str(tmp_path))
