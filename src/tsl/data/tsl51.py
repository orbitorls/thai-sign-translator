"""TSL-51 sentence-level manifest and landmark loader.

Loads the public ``Namonpas/thai-sign-language-tsl51`` dataset layout:

- ``metadata/sentence_metadata.csv`` with columns
  ``video_id, sentence_id, sentence_clean, landmark_path, video_path``
- ``landmarks/user_sentence/<video_id>.csv`` — one wide-format CSV per
  video where each row is a frame and columns are 162 normalized
  MediaPipe-Holoistic coordinates (6 pose + 6 face + 21x2 hands),
  named with ``_x`` / ``_y`` / ``_z`` suffixes.

This module is intentionally landmarks-only: the raw ``videos/`` are
not touched. Use :func:`load_sentence_manifest` to enumerate examples
and :func:`load_landmark_sequence` to materialise a frame-major
``(T, 162)`` ``float32`` array, with NaN/inf replaced by 0.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from tsl.data.manifest import SignTextExample

__all__ = ["load_sentence_manifest", "load_landmark_sequence", "load_sentence_features"]

# TSL-51 selected MediaPipe Holistic landmark count: 6 pose + 6 face + 21*2 hands.
_LANDMARK_DIM = 162

# Columns we explicitly drop before picking coordinate columns.
_DROP_COLUMNS: frozenset[str] = frozenset({"frame", "t_ms"})


def load_sentence_manifest(data_root: str) -> list[SignTextExample]:
    """Load sentence-level examples from a local TSL-51 dataset root.

    ``data_root`` must contain:
      - ``metadata/sentence_metadata.csv``
      - ``landmarks/user_sentence/<video_id>.csv`` (one per video)

    Rows with empty ``sentence_clean`` or empty ``landmark_path`` are
    silently skipped (the public TSL-51 release has a few such entries).
    All returned examples are tagged ``split="train"`` because the
    public release does not ship train/val/test partitions for
    sentences; downstream code is expected to do its own split.
    """
    if not os.path.isdir(data_root):
        raise FileNotFoundError(f"data_root is not a directory: {data_root!r}")
    meta_path = os.path.join(data_root, "metadata", "sentence_metadata.csv")
    if not os.path.isfile(meta_path):
        raise FileNotFoundError(f"sentence metadata not found: {meta_path!r}")

    df = pd.read_csv(meta_path)
    out: list[SignTextExample] = []
    for _, row in df.iterrows():
        sentence_clean = row.get("sentence_clean")
        landmark_path = row.get("landmark_path")
        if not _is_non_empty_str(sentence_clean):
            continue
        if not _is_non_empty_str(landmark_path):
            continue
        out.append(
            SignTextExample(
                example_id=str(row["video_id"]),
                source="tsl51",
                split="train",
                features_path=os.path.join(data_root, str(landmark_path)),
                target_text=str(sentence_clean),
                metadata={
                    "sentence_id": row.get("sentence_id"),
                    "video_id": row.get("video_id"),
                },
            )
        )
    return out


def load_landmark_sequence(csv_path: str) -> np.ndarray:
    """Read a TSL-51 wide-format landmark CSV into a ``(T, 162)`` array.

    Non-coordinate columns (``frame``, ``t_ms``, and any ``*_vis`` /
    ``*_pres``) are dropped; only columns whose name contains ``_x``,
    ``_y`` or ``_z`` are kept. NaN and +/-inf values are replaced with
    0.0. If the file has no coordinate columns at all, an empty
    ``(0, 162)`` ``float32`` array is returned.
    """
    df = pd.read_csv(csv_path)
    coord_cols = _select_coordinate_columns(df.columns)
    if not coord_cols:
        return np.zeros((0, _LANDMARK_DIM), dtype=np.float32)
    arr = df[coord_cols].to_numpy()
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return arr.astype(np.float32)


def load_sentence_features(example: SignTextExample) -> np.ndarray:
    """Convenience wrapper that reads the landmarks for a manifest example."""
    return load_landmark_sequence(example.features_path)


def _is_non_empty_str(value: object) -> bool:
    if value is None:
        return False
    try:
        if pd.isna(value):
            return False
    except (TypeError, ValueError):
        return False
    if not isinstance(value, str):
        return False
    return value != ""


def _select_coordinate_columns(columns: pd.Index) -> list[str]:
    picked: list[str] = []
    for col in columns:
        name = str(col)
        if name in _DROP_COLUMNS:
            continue
        if any(tag in name for tag in ("_x", "_y", "_z")):
            picked.append(name)
    return picked
