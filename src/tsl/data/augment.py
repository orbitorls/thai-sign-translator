"""Landmark sequence augmentation for the Thai SLT pipeline.

All functions take and return (T, D) float32 numpy arrays.
No torch dependency — augmentation happens on the numpy side before collation.
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "time_stretch",
    "jitter",
    "frame_dropout",
    "mirror_hands",
    "augment_sequence",
]

# MediaPipe Holistic hand layout (same for both 162-dim and 312-dim):
# left_hand  cols 0..62   (21 pts × 3)
# right_hand cols 63..125 (21 pts × 3)
_LH_START, _LH_END = 0, 63
_RH_START, _RH_END = 63, 126


def time_stretch(seq: np.ndarray, factor: float) -> np.ndarray:
    """Resample T frames to round(T * factor) via linear interpolation.

    factor is clamped to [0.5, 2.0].  factor=1.0 returns the input unchanged.
    """
    factor = float(np.clip(factor, 0.5, 2.0))
    T = seq.shape[0]
    T_new = max(1, round(T * factor))
    if T_new == T:
        return seq
    old_idx = np.linspace(0, T - 1, T_new)
    lo = np.floor(old_idx).astype(np.int32)
    hi = np.minimum(lo + 1, T - 1)
    w = (old_idx - lo).astype(np.float32)[:, None]
    return ((1 - w) * seq[lo] + w * seq[hi]).astype(np.float32)


def jitter(seq: np.ndarray, std: float = 0.02) -> np.ndarray:
    """Add zero-mean Gaussian noise with the given std to every coordinate."""
    return (seq + np.random.randn(*seq.shape).astype(np.float32) * std).astype(np.float32)


def frame_dropout(seq: np.ndarray, rate: float = 0.15) -> np.ndarray:
    """Zero out a random fraction of frames. At least 1 frame is kept."""
    T = seq.shape[0]
    n_drop = min(max(0, round(T * rate)), T - 1)
    if n_drop == 0:
        return seq
    out = seq.copy()
    idx = np.random.choice(T, size=n_drop, replace=False)
    out[idx] = 0.0
    return out


def mirror_hands(seq: np.ndarray, input_dim: int) -> np.ndarray:
    """Swap left-hand and right-hand channels to simulate a mirrored signer.

    Works for input_dim=162 (TSL-51 layout) and input_dim=312 (normalize.py
    layout) — both place left_hand at cols 0..62 and right_hand at cols 63..125.

    Raises ValueError if seq.shape[1] != input_dim or input_dim < 126.
    """
    if seq.shape[1] != input_dim:
        raise ValueError(
            f"seq has D={seq.shape[1]} but input_dim={input_dim}"
        )
    if input_dim < _RH_END:
        raise ValueError(
            f"input_dim={input_dim} is too small to contain two 21-pt hands "
            f"(need at least {_RH_END})"
        )
    out = seq.copy()
    lh = seq[:, _LH_START:_LH_END].copy()
    rh = seq[:, _RH_START:_RH_END].copy()
    out[:, _LH_START:_LH_END] = rh
    out[:, _RH_START:_RH_END] = lh
    return out


def augment_sequence(
    seq: np.ndarray,
    rng: np.random.Generator,
    p_stretch: float = 0.5,
    p_jitter: float = 0.8,
    p_dropout: float = 0.5,
    p_mirror: float = 0.3,
) -> np.ndarray:
    """Apply a random combination of augmentations.

    Each augmentation is applied independently with its probability.
    Uses the provided numpy Generator for reproducibility.

    input_dim is inferred from seq.shape[1]; mirror is skipped when
    input_dim < 126 (e.g. tiny mock arrays in tests).
    """
    out = seq.astype(np.float32, copy=True)

    if rng.random() < p_stretch:
        factor = float(rng.uniform(0.8, 1.2))
        out = time_stretch(out, factor)

    if rng.random() < p_jitter:
        std = float(rng.uniform(0.005, 0.03))
        out = jitter(out, std)

    if rng.random() < p_dropout:
        rate = float(rng.uniform(0.05, 0.2))
        out = frame_dropout(out, rate)

    if rng.random() < p_mirror and out.shape[1] >= _RH_END:
        out = mirror_hands(out, out.shape[1])

    return out
