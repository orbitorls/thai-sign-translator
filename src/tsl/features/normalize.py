"""Landmark selection + normalization.

normalize_sequence: (T, 543, 3) -> (T, D) float32, where
  D = len(SELECTED_LANDMARKS) * 3.
normalize_sequence_v4: (T, 543, 4) -> ((T, D), (T, 104)) with visibility weights.

Pipeline (per frame): nose-center (NOSE_IDX) -> scale by inter-shoulder
distance -> select SELECTED_LANDMARKS -> flatten -> NaN/inf -> 0.

SELECTED_LANDMARKS composition (104 landmarks, D = 312):
  left_hand   : 21  (global 468..488)
  right_hand  : 21  (global 522..542)
  pose upper  : 22  (global 489..510, includes NOSE_IDX=489 and shoulders)
  face subset : 40  (brow + lip indices, all within 0..467)
The exact lists below are a fixed, documented contract; do not reorder.
"""
from __future__ import annotations

import numpy as np

from config import NOSE_IDX, LSHOULDER_IDX, RSHOULDER_IDX

CANONICAL_FEATURE_DIM: int = 312  # D = len(SELECTED_LANDMARKS) * 3
FEATURE_LAYOUT_VERSION: str = "v3-312"
FEATURE_LAYOUT_VERSION_V4: str = "v4-312-masked"
SELECTED_LANDMARK_COUNT: int = 104

_FACE_SUBSET: list[int] = [
    70, 63, 105, 66, 107,
    300, 293, 334, 296, 336,
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 0,
    78, 95, 88, 178, 87, 14, 317, 402,
    1, 4, 5, 6, 197,
    33, 133, 362, 263, 168,
]

SELECTED_LANDMARKS: list[int] = (
    list(range(468, 489))
    + list(range(522, 543))
    + list(range(489, 511))
    + _FACE_SUBSET
)

D: int = len(SELECTED_LANDMARKS) * 3


def _normalize_coords(coords: np.ndarray) -> np.ndarray:
    """Normalize xyz landmarks to the canonical 312-dim flat layout."""
    coords = np.asarray(coords, dtype=np.float32)
    T = coords.shape[0]

    nose = coords[:, NOSE_IDX:NOSE_IDX + 1, :]
    centered = coords - nose

    shoulder_vec = centered[:, LSHOULDER_IDX, :] - centered[:, RSHOULDER_IDX, :]
    scale = np.linalg.norm(shoulder_vec, axis=1)
    bad = ~np.isfinite(scale) | (scale < 1e-6)
    scale = np.where(bad, 1.0, scale).astype(np.float32)
    scaled = centered / scale[:, None, None]

    selected = scaled[:, SELECTED_LANDMARKS, :]
    flat = selected.reshape(T, -1)

    flat = np.nan_to_num(flat, nan=0.0, posinf=0.0, neginf=0.0)
    return flat.astype(np.float32)


def normalize_sequence(seq: np.ndarray) -> np.ndarray:
    """v3 path: ``(T, 543, 3)`` -> ``(T, 312)``.

  When the last dimension is 4, xyz channels are normalized and the
  visibility weight is ignored (use :func:`normalize_sequence_v4` instead).
    """
    seq = np.asarray(seq, dtype=np.float32)
    if seq.ndim != 3 or seq.shape[1] != 543 or seq.shape[2] not in (3, 4):
        raise ValueError(
            f"seq must have shape (T, 543, 3) or (T, 543, 4); got {tuple(seq.shape)}"
        )
    coords = seq[:, :, :3] if seq.shape[2] == 4 else seq
    return _normalize_coords(coords)


def normalize_sequence_v4(seq: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """v4 path: ``(T, 543, 4)`` -> ``((T, 312), (T, 104))``."""
    seq = np.asarray(seq, dtype=np.float32)
    if seq.ndim != 3 or seq.shape[1] != 543 or seq.shape[2] != 4:
        raise ValueError(
            f"seq must have shape (T, 543, 4); got {tuple(seq.shape)}"
        )

    weights = seq[:, SELECTED_LANDMARKS, 3].copy()
    weights = np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
    weights = np.clip(weights, 0.0, 1.0).astype(np.float32)
    features = _normalize_coords(seq[:, :, :3])
    return features, weights
