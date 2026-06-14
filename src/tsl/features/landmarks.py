"""MediaPipe Holistic landmark extraction.

Produces the canonical 543-landmark layout (matches Google ISLR):
  face(468)       -> global idx 0..467
  left_hand(21)   -> global idx 468..488
  pose(33)        -> global idx 489..521
  right_hand(21)  -> global idx 522..542
Missing components (e.g. an absent hand) are filled with NaN.
"""
from __future__ import annotations

import numpy as np

from config import FACE, LEFT_HAND, POSE, RIGHT_HAND, N_LANDMARKS

_COMPONENTS = [
    ("face_landmarks", 468, FACE),
    ("left_hand_landmarks", 21, LEFT_HAND),
    ("pose_landmarks", 33, POSE),
    ("right_hand_landmarks", 21, RIGHT_HAND),
]


def _component_to_array(landmark_list, count: int) -> np.ndarray:
    if landmark_list is None:
        return np.full((count, 3), np.nan, dtype=np.float32)
    pts = landmark_list.landmark
    out = np.full((count, 3), np.nan, dtype=np.float32)
    for i, p in enumerate(pts):
        if i >= count:
            break
        out[i, 0] = p.x
        out[i, 1] = p.y
        out[i, 2] = p.z
    return out


def extract_frame_landmarks(holistic, frame_bgr: np.ndarray) -> np.ndarray:
    image_rgb = np.ascontiguousarray(frame_bgr[:, :, ::-1])
    results = holistic.process(image_rgb)
    frame = np.full((N_LANDMARKS, 3), np.nan, dtype=np.float32)
    for attr, count, sl in _COMPONENTS:
        landmark_list = getattr(results, attr, None)
        frame[sl] = _component_to_array(landmark_list, count)
    return frame


def extract_sequence(holistic, frames: list[np.ndarray]) -> np.ndarray:
    if len(frames) == 0:
        return np.empty((0, N_LANDMARKS, 3), dtype=np.float32)
    seq = [extract_frame_landmarks(holistic, f) for f in frames]
    return np.stack(seq, axis=0).astype(np.float32)
