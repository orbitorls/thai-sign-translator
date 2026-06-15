"""Tests for tsl.data.augment."""
import numpy as np
import pytest

from tsl.data.augment import (
    augment_sequence,
    frame_dropout,
    jitter,
    mirror_hands,
    time_stretch,
)

RNG = np.random.default_rng(0)


def _seq(T=30, D=162):
    return np.random.default_rng(42).random((T, D)).astype(np.float32)


# ── time_stretch ────────────────────────────────────────────────────────────

def test_time_stretch_longer():
    seq = _seq(30)
    out = time_stretch(seq, 1.5)
    assert out.shape == (45, 162)
    assert out.dtype == np.float32


def test_time_stretch_shorter():
    seq = _seq(30)
    out = time_stretch(seq, 0.5)
    assert out.shape == (15, 162)


def test_time_stretch_identity():
    seq = _seq(30)
    out = time_stretch(seq, 1.0)
    assert out.shape == seq.shape
    np.testing.assert_array_equal(out, seq)


def test_time_stretch_clamps():
    seq = _seq(10)
    assert time_stretch(seq, 0.1).shape[0] >= 5   # clamped to 0.5
    assert time_stretch(seq, 5.0).shape[0] <= 20  # clamped to 2.0


def test_time_stretch_preserves_dtype():
    seq = _seq(20)
    assert time_stretch(seq, 1.3).dtype == np.float32


# ── jitter ──────────────────────────────────────────────────────────────────

def test_jitter_changes_values():
    seq = _seq(30)
    out = jitter(seq, std=0.05)
    assert not np.allclose(out, seq)


def test_jitter_preserves_shape_dtype():
    seq = _seq(30)
    out = jitter(seq)
    assert out.shape == seq.shape
    assert out.dtype == np.float32


def test_jitter_zero_std_identity():
    seq = _seq(20)
    out = jitter(seq, std=0.0)
    np.testing.assert_array_almost_equal(out, seq)


# ── frame_dropout ────────────────────────────────────────────────────────────

def test_frame_dropout_zeroes_frames():
    np.random.seed(7)
    seq = np.ones((40, 162), dtype=np.float32)
    out = frame_dropout(seq, rate=0.2)
    zeroed = (out.sum(axis=1) == 0).sum()
    assert zeroed >= 1


def test_frame_dropout_keeps_at_least_one():
    seq = np.ones((5, 162), dtype=np.float32)
    for _ in range(20):
        out = frame_dropout(seq, rate=0.99)
        assert (out.sum(axis=1) != 0).sum() >= 1


def test_frame_dropout_preserves_shape_dtype():
    seq = _seq(20)
    out = frame_dropout(seq)
    assert out.shape == seq.shape
    assert out.dtype == np.float32


# ── mirror_hands ─────────────────────────────────────────────────────────────

def test_mirror_hands_swaps_channels_162():
    seq = _seq(10, 162)
    out = mirror_hands(seq, 162)
    np.testing.assert_array_equal(out[:, 0:63], seq[:, 63:126])
    np.testing.assert_array_equal(out[:, 63:126], seq[:, 0:63])
    # rest unchanged
    np.testing.assert_array_equal(out[:, 126:], seq[:, 126:])


def test_mirror_hands_swaps_channels_312():
    seq = _seq(10, 312)
    out = mirror_hands(seq, 312)
    np.testing.assert_array_equal(out[:, 0:63], seq[:, 63:126])
    np.testing.assert_array_equal(out[:, 63:126], seq[:, 0:63])


def test_mirror_hands_double_mirror_identity():
    seq = _seq(10, 162)
    out = mirror_hands(mirror_hands(seq, 162), 162)
    np.testing.assert_array_almost_equal(out, seq)


def test_mirror_hands_wrong_dim_raises():
    seq = _seq(10, 100)
    with pytest.raises(ValueError):
        mirror_hands(seq, 162)


# ── augment_sequence ──────────────────────────────────────────────────────────

def test_augment_sequence_preserves_D():
    rng = np.random.default_rng(1)
    seq = _seq(30, 162)
    out = augment_sequence(seq, rng)
    assert out.shape[1] == 162


def test_augment_sequence_non_zero_T():
    rng = np.random.default_rng(2)
    seq = _seq(20, 162)
    out = augment_sequence(seq, rng)
    assert out.shape[0] >= 1


def test_augment_sequence_float32():
    rng = np.random.default_rng(3)
    out = augment_sequence(_seq(25, 162), rng)
    assert out.dtype == np.float32


def test_augment_sequence_all_off_returns_copy():
    rng = np.random.default_rng(4)
    seq = _seq(20, 162)
    out = augment_sequence(seq, rng, p_stretch=0, p_jitter=0, p_dropout=0, p_mirror=0)
    np.testing.assert_array_equal(out, seq)
    assert out is not seq  # copy, not same object
