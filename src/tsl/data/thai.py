"""Loader for recorded Thai sign clips."""
from __future__ import annotations

import os

import numpy as np

from tsl.features.normalize import normalize_sequence


def load_thai_clips(root: str) -> dict[str, list[np.ndarray]]:
    out: dict[str, list[np.ndarray]] = {}
    if not os.path.isdir(root):
        return out
    for word in sorted(os.listdir(root)):
        word_dir = os.path.join(root, word)
        if not os.path.isdir(word_dir):
            continue
        clips: list[np.ndarray] = []
        for fname in sorted(os.listdir(word_dir)):
            if not fname.endswith(".npy"):
                continue
            raw = np.load(os.path.join(word_dir, fname))
            norm = normalize_sequence(raw)
            clips.append(np.asarray(norm, dtype=np.float32))
        if clips:
            out[word] = clips
    return out
