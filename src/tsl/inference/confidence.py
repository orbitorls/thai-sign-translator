"""Small confidence helpers shared by inference entrypoints."""
from __future__ import annotations

import numpy as np

from tsl.features.normalize import SELECTED_LANDMARKS

LOW_LANDMARK_QUALITY_THRESHOLD: float = 0.4
LOW_QUALITY_MIN_RUN: int = 10
COMPOSITE_TOKEN_WEIGHT: float = 0.6
COMPOSITE_LANDMARK_WEIGHT: float = 0.4


def is_low_confidence(score: float, threshold: float = 0.8) -> bool:
    return float(score) < float(threshold)


def per_frame_landmark_quality(weights: np.ndarray) -> np.ndarray:
    """Per-frame quality in ``[0, 1]`` from visibility weights ``(T, 104)``."""
    w = np.asarray(weights, dtype=np.float32)
    if w.ndim != 2 or w.shape[1] != len(SELECTED_LANDMARKS):
        raise ValueError(
            f"weights must have shape (T, {len(SELECTED_LANDMARKS)}); got {tuple(w.shape)}"
        )
    hand_weight = w[:, :42].mean(axis=1)
    pose_weight = w[:, 42:64].mean(axis=1)
    face_weight = w[:, 64:].mean(axis=1)
    return (0.5 * hand_weight + 0.3 * pose_weight + 0.2 * face_weight).astype(np.float32)


def landmark_quality_score(weights: np.ndarray) -> float:
    """Mean landmark visibility quality across the clip."""
    per_frame = per_frame_landmark_quality(weights)
    return float(per_frame.mean())


def composite_confidence(token_score: float, landmark_quality: float) -> float:
    return float(
        COMPOSITE_TOKEN_WEIGHT * float(token_score)
        + COMPOSITE_LANDMARK_WEIGHT * float(landmark_quality)
    )


def extract_landmark_weights(raw: np.ndarray) -> np.ndarray:
    """Build ``(T, 104)`` visibility weights from raw MediaPipe frames.

    Accepts ``(T, 543, 3)`` or ``(T, 543, 4)``. For 3-D frames, landmarks
    near the origin are treated as missing.
    """
    arr = np.asarray(raw, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[1] != 543 or arr.shape[2] not in (3, 4):
        raise ValueError(
            f"raw frames must have shape (T, 543, 3) or (T, 543, 4); got {tuple(arr.shape)}"
        )

    if arr.shape[2] == 4:
        selected = arr[:, SELECTED_LANDMARKS, 3]
        return np.clip(selected, 0.0, 1.0).astype(np.float32)

    selected = arr[:, SELECTED_LANDMARKS, :3]
    norms = np.linalg.norm(selected, axis=-1)
    return (norms > 1e-6).astype(np.float32)


def trim_low_quality_runs(
    features: np.ndarray,
    weights: np.ndarray | None,
    *,
    min_run: int = LOW_QUALITY_MIN_RUN,
    threshold: float = LOW_LANDMARK_QUALITY_THRESHOLD,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Drop contiguous low-quality sub-windows before encoding."""
    if weights is None or features.shape[0] <= min_run:
        return features, weights

    per_frame = per_frame_landmark_quality(weights)
    bad = per_frame < threshold
    drop = np.zeros(len(features), dtype=bool)
    run_start: int | None = None

    for idx, is_bad in enumerate(bad):
        if is_bad:
            if run_start is None:
                run_start = idx
            continue
        if run_start is not None and (idx - run_start) >= min_run:
            drop[run_start:idx] = True
        run_start = None

    if run_start is not None and (len(bad) - run_start) >= min_run:
        drop[run_start:] = True

    if not drop.any():
        return features, weights

    keep = ~drop
    trimmed_weights = weights[keep]
    return features[keep], trimmed_weights


__all__ = [
    "COMPOSITE_LANDMARK_WEIGHT",
    "COMPOSITE_TOKEN_WEIGHT",
    "LOW_LANDMARK_QUALITY_THRESHOLD",
    "LOW_QUALITY_MIN_RUN",
    "composite_confidence",
    "extract_landmark_weights",
    "is_low_confidence",
    "landmark_quality_score",
    "per_frame_landmark_quality",
    "trim_low_quality_runs",
]
