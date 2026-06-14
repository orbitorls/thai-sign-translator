"""(Stretch) Micro-pause segmentation of a continuous normalized landmark stream."""
from __future__ import annotations

import numpy as np


def _wrist_velocity(seq_norm, wrist_idx_in_D):
    T = seq_norm.shape[0]
    vel = np.zeros(T, dtype=np.float64)
    if T < 2:
        return vel
    diffs = []
    for w in wrist_idx_in_D:
        coords = seq_norm[:, w:w + 3].astype(np.float64)
        step = np.linalg.norm(np.diff(coords, axis=0), axis=1)
        diffs.append(step)
    mean_step = np.mean(np.stack(diffs, axis=0), axis=0)
    vel[1:] = mean_step
    return vel


def segment_stream(seq_norm, wrist_idx_in_D, v_thresh=0.05, min_pause_frames=4):
    T = int(seq_norm.shape[0])
    if T == 0:
        return []
    if T == 1:
        return [(0, 1)]
    vel = _wrist_velocity(seq_norm, wrist_idx_in_D)
    is_pause_frame = vel < v_thresh
    long_pause = np.zeros(T, dtype=bool)
    i = 0
    while i < T:
        if is_pause_frame[i]:
            j = i
            while j < T and is_pause_frame[j]:
                j += 1
            if (j - i) >= min_pause_frames:
                long_pause[i:j] = True
            i = j
        else:
            i += 1
    spans = []
    i = 0
    while i < T:
        if not long_pause[i]:
            start = i
            while i < T and not long_pause[i]:
                i += 1
            spans.append((int(start), int(i)))
        else:
            i += 1
    return spans
