"""ThaiSignVis dataset loader for sentence-level Thai sign language translation.

ThaiSignVis (Kaggle, Apache 2.0) ships raw MP4 videos + transcript CSVs.
This module reads the **post-extraction** manifest written by
``scripts/extract_thaisignvis_landmarks.py``, which produces:

    <out_dir>/
        manifest.csv          -- segment_id, npy_path, text, video_id, start_ms, end_ms, split
        landmarks/<seg_id>.npy -- (T, 312) float32 via normalize.normalize_sequence

Column name constants at the top can be updated if the upstream transcript CSV
layout differs from what was assumed when extraction ran.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from tsl.data.manifest import SignTextExample

__all__ = [
    "FEATURE_DIM",
    "MANIFEST_FILENAME",
    "load_thaisignvis_manifest",
    "load_thaisignvis_features",
    "load_npy_manifest",
    "load_npy_features",
]

FEATURE_DIM: int = 312  # 104 selected landmarks × 3 via normalize.normalize_sequence
MANIFEST_FILENAME: str = "manifest.csv"

# Manifest CSV column names written by the extraction script.
_COL_SEG_ID = "segment_id"
_COL_NPY = "npy_path"
_COL_TEXT = "text"
_COL_VIDEO = "video_id"
_COL_SPLIT = "split"


def load_thaisignvis_manifest(
    data_root: str,
    split: str | None = None,
) -> list[SignTextExample]:
    """Load ThaiSignVis segment examples from an extracted data root.

    ``data_root`` must contain ``manifest.csv`` (written by the extraction
    script).  Pass ``split="train"`` / ``"val"`` / ``"test"`` to filter;
    ``None`` returns all rows.

    Rows with empty ``text`` or missing ``.npy`` files are silently skipped.
    """
    manifest_path = os.path.join(data_root, MANIFEST_FILENAME)
    if not os.path.isfile(manifest_path):
        raise FileNotFoundError(
            f"ThaiSignVis manifest not found: {manifest_path!r}\n"
            "Run scripts/extract_thaisignvis_landmarks.py first."
        )

    df = pd.read_csv(manifest_path)
    _check_columns(df, manifest_path)

    if split is not None:
        df = df[df[_COL_SPLIT] == split]

    out: list[SignTextExample] = []
    for _, row in df.iterrows():
        text = str(row[_COL_TEXT]).strip()
        if not text:
            continue
        npy_path = str(row[_COL_NPY])
        if not os.path.isabs(npy_path):
            npy_path = os.path.join(data_root, npy_path)
        if not os.path.isfile(npy_path):
            continue
        out.append(
            SignTextExample(
                example_id=str(row[_COL_SEG_ID]),
                source="thaisignvis",
                split=str(row.get(_COL_SPLIT, "train")),
                features_path=npy_path,
                target_text=text,
                metadata={"video_id": str(row.get(_COL_VIDEO, ""))},
            )
        )
    return out


def load_thaisignvis_features(npy_path: str) -> np.ndarray:
    """Load a (T, 312) float32 landmark array from a cached .npy file."""
    arr = np.load(npy_path)
    if arr.ndim != 2 or arr.shape[1] != FEATURE_DIM:
        raise ValueError(
            f"unexpected shape {arr.shape} in {npy_path!r}; "
            f"expected (T, {FEATURE_DIM})"
        )
    return arr.astype(np.float32)


# ---------------------------------------------------------------------------
# Generic npy manifest loader (any feature dim — used for YouTube-SL-25 etc.)
# ---------------------------------------------------------------------------

def load_npy_manifest(
    data_root: str,
    split: str | None = None,
    source: str = "npy_manifest",
) -> list[SignTextExample]:
    """Load any manifest.csv + .npy dataset regardless of feature dim.

    The manifest must have at minimum: segment_id, npy_path, text.
    An optional ``split`` column is used for filtering when ``split`` is given.
    """
    manifest_path = os.path.join(data_root, MANIFEST_FILENAME)
    if not os.path.isfile(manifest_path):
        raise FileNotFoundError(f"manifest not found: {manifest_path!r}")

    df = pd.read_csv(manifest_path)
    _check_columns(df, manifest_path)

    if split is not None and _COL_SPLIT in df.columns:
        df = df[df[_COL_SPLIT] == split]

    out: list[SignTextExample] = []
    for _, row in df.iterrows():
        text = str(row[_COL_TEXT]).strip()
        if not text:
            continue
        npy_path = str(row[_COL_NPY])
        if not os.path.isabs(npy_path):
            npy_path = os.path.join(data_root, npy_path)
        if not os.path.isfile(npy_path):
            continue
        out.append(
            SignTextExample(
                example_id=str(row[_COL_SEG_ID]),
                source=source,
                split=str(row.get(_COL_SPLIT, "train")),
                features_path=npy_path,
                target_text=text,
                metadata={"video_id": str(row.get(_COL_VIDEO, ""))},
            )
        )
    return out


def load_npy_features(npy_path: str) -> np.ndarray:
    """Load any (T, D) float32 landmark array from a .npy file — no dim check."""
    arr = np.load(npy_path)
    if arr.ndim != 2:
        raise ValueError(f"expected 2-D array in {npy_path!r}, got shape {arr.shape}")
    return arr.astype(np.float32)


def _check_columns(df: pd.DataFrame, path: str) -> None:
    required = {_COL_SEG_ID, _COL_NPY, _COL_TEXT}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"manifest {path!r} is missing columns: {missing}\n"
            f"Found: {list(df.columns)}"
        )
