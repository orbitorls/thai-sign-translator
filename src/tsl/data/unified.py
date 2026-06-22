"""Unified manifest loader for canonical 312-dim and raw holistic features."""
from __future__ import annotations

import os
import re

import numpy as np
import pandas as pd

from tsl.data.manifest import SignTextExample
from tsl.features.normalize import normalize_sequence

__all__ = [
    "MANIFEST_FILENAME",
    "load_manifest",
    "load_features",
]

MANIFEST_FILENAME: str = "manifest.csv"

CANONICAL_FEATURE_DIM: int = 312

_COL_SEGMENT_ID = "segment_id"
_COL_NPY_PATH = "npy_path"
_COL_TEXT = "text"
_COL_VIDEO_ID = "video_id"
_COL_SPLIT = "split"
_COL_SOURCE = "source"
_COL_FEATURE_LAYOUT_VERSION = "feature_layout_version"

_REQUIRED_COLUMNS = {_COL_SEGMENT_ID, _COL_NPY_PATH, _COL_TEXT, _COL_FEATURE_LAYOUT_VERSION}

_FEATURE_LAYOUT_PREFIX = "v3-312"
_RAW_HOLISTIC_LAYOUT = "raw_mediapipe_543x3"


def _infer_source_name(data_root: str) -> str:
    basename = os.path.basename(os.path.normpath(data_root))
    if not basename:
        return "unified"
    lowered = basename.lower().replace("-", "_")
    if "thaisignvis" in lowered:
        return "thaisignvis"
    if "youtube_sl25_thai" in lowered or ("youtube" in lowered and "sl25" in lowered and "thai" in lowered):
        return "youtube_sl25_thai"
    if "youtube_sl25" in lowered:
        return "youtube_sl25"
    if "tsl51" in lowered:
        return "tsl51"
    normalized = re.sub(r"(_v\d+(?:[-_].*)?)$", "", basename)
    normalized = normalized.strip("_-")
    return normalized or basename


def load_manifest(
    data_root: str,
    split: str | None = None,
    source: str | None = None,
) -> list[SignTextExample]:
    """Load a v3-312-enforced manifest from ``data_root/manifest.csv``.

    Parameters
    ----------
    data_root:
        Directory containing ``manifest.csv``.
    split:
        If given, only rows with this ``split`` value are returned.
    source:
        If given, only rows matching this ``source`` column value are returned.
        Rows without a ``source`` column are included when ``source`` is None.

    Returns
    -------
    list[SignTextExample]
        Examples with verified feature layout version.

    Raises
    ------
    FileNotFoundError
        If ``manifest.csv`` does not exist in ``data_root``.
    ValueError
        If any required column is missing, or if any row's
        ``feature_layout_version`` is not a supported layout.
    """
    manifest_path = os.path.join(data_root, MANIFEST_FILENAME)
    if not os.path.isfile(manifest_path):
        raise FileNotFoundError(
            f"Unified manifest not found: {manifest_path!r}\n"
            "Ensure extraction has been run and the manifest is present."
        )

    df = pd.read_csv(manifest_path)
    _check_columns(df, manifest_path)

    # Enforce feature_layout_version for every row before filtering
    for idx, row in df.iterrows():
        version = str(row[_COL_FEATURE_LAYOUT_VERSION])
        if not _is_supported_feature_layout(version):
            raise ValueError(
                f"Row {idx}: feature_layout_version {version!r} is not supported. "
                f"Expected {_FEATURE_LAYOUT_PREFIX!r} prefix or {_RAW_HOLISTIC_LAYOUT!r}."
            )

    if split is not None and _COL_SPLIT in df.columns:
        df = df[df[_COL_SPLIT] == split]

    if source is not None and _COL_SOURCE in df.columns:
        df = df[df[_COL_SOURCE] == source]

    examples: list[SignTextExample] = []
    for _, row in df.iterrows():
        raw_text = row[_COL_TEXT]
        if pd.isna(raw_text):
            continue
        text = str(raw_text).strip()
        if not text:
            continue

        npy_path = str(row[_COL_NPY_PATH]).replace("\\", "/")
        if not os.path.isabs(npy_path):
            npy_path = os.path.join(data_root, npy_path)
        if not os.path.isfile(npy_path):
            continue

        row_split = str(row.get(_COL_SPLIT, "train")) if _COL_SPLIT in row.index else "train"
        row_source = (
            str(row.get(_COL_SOURCE, _infer_source_name(data_root)))
            if _COL_SOURCE in row.index
            else _infer_source_name(data_root)
        )

        examples.append(
            SignTextExample(
                example_id=str(row[_COL_SEGMENT_ID]),
                source=row_source,
                split=row_split,
                features_path=npy_path,
                target_text=text,
                metadata={
                    "video_id": str(row.get(_COL_VIDEO_ID, "")) if _COL_VIDEO_ID in row.index else "",
                    "feature_layout_version": str(row[_COL_FEATURE_LAYOUT_VERSION]),
                },
            )
        )
    return examples


def load_features(npy_path: str) -> np.ndarray:
    """Load or normalize pose features into a ``(T, 312)`` float32 array.

    Parameters
    ----------
    npy_path:
        Path to the ``.npy`` file containing landmark features.

    Returns
    -------
    np.ndarray
        Array of shape ``(T, 312)`` and dtype ``float32``.

    Raises
    ------
    ValueError
        If the loaded array does not match a supported feature shape.
    """
    arr = np.load(npy_path)
    if arr.ndim == 3 and arr.shape[1:] == (543, 3):
        return normalize_sequence(arr)
    if arr.ndim != 2 or arr.shape[1] != CANONICAL_FEATURE_DIM:
        raise ValueError(
            f"unexpected shape {arr.shape} in {npy_path!r}; "
            f"expected (T, {CANONICAL_FEATURE_DIM}) or (T, 543, 3)"
        )
    return arr.astype(np.float32)


def _check_columns(df: pd.DataFrame, path: str) -> None:
    missing = _REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"manifest {path!r} is missing columns: {missing}\n"
            f"Found: {list(df.columns)}"
        )


def _is_supported_feature_layout(version: str) -> bool:
    return version.startswith(_FEATURE_LAYOUT_PREFIX) or version == _RAW_HOLISTIC_LAYOUT
