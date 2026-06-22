"""YouTube-SL-25 manifest loader for extracted ``.npy`` landmark features.

The current training pipeline uses the Thai subset exported to:

    <data_root>/
        manifest.csv
        landmarks/<segment_id>.npy

The loader stays intentionally generic on feature shape so existing datasets
with 162-dim or 312-dim frame vectors can share the same manifest contract.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from tsl.data.manifest import SignTextExample

__all__ = [
    "MANIFEST_FILENAME",
    "load_youtube_sl25_manifest",
    "load_youtube_sl25_features",
]

MANIFEST_FILENAME = "manifest.csv"

_COL_SEGMENT_ID = "segment_id"
_COL_NPY_PATH = "npy_path"
_COL_TEXT = "text"
_COL_VIDEO_ID = "video_id"
_COL_SPLIT = "split"


def load_youtube_sl25_manifest(
    data_root: str,
    split: str | None = None,
    source: str = "youtube_sl25",
) -> list[SignTextExample]:
    """Load ``manifest.csv`` rows backed by cached ``.npy`` features."""
    manifest_path = os.path.join(data_root, MANIFEST_FILENAME)
    if not os.path.isfile(manifest_path):
        raise FileNotFoundError(f"manifest not found: {manifest_path!r}")

    df = pd.read_csv(manifest_path)
    _check_columns(df, manifest_path)
    if split is not None and _COL_SPLIT in df.columns:
        df = df[df[_COL_SPLIT] == split]

    examples: list[SignTextExample] = []
    for _, row in df.iterrows():
        text = str(row[_COL_TEXT]).strip()
        if not text:
            continue
        npy_path = str(row[_COL_NPY_PATH])
        if not os.path.isabs(npy_path):
            npy_path = os.path.join(data_root, npy_path)
        if not os.path.isfile(npy_path):
            continue
        examples.append(
            SignTextExample(
                example_id=str(row[_COL_SEGMENT_ID]),
                source=source,
                split=str(row.get(_COL_SPLIT, "train")),
                features_path=npy_path,
                target_text=text,
                metadata={"video_id": str(row.get(_COL_VIDEO_ID, ""))},
            )
        )
    return examples


def load_youtube_sl25_features(npy_path: str) -> np.ndarray:
    """Load any frame-major ``(T, D)`` float32 array from ``npy_path``."""
    arr = np.load(npy_path)
    if arr.ndim != 2:
        raise ValueError(f"expected 2-D array in {npy_path!r}, got shape {arr.shape}")
    return arr.astype(np.float32)


def _check_columns(df: pd.DataFrame, path: str) -> None:
    required = {_COL_SEGMENT_ID, _COL_NPY_PATH, _COL_TEXT}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"manifest {path!r} is missing columns: {missing}\n"
            f"Found: {list(df.columns)}"
        )
