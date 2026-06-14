import numpy as np
import pytest

from config import NOSE_IDX, LSHOULDER_IDX, RSHOULDER_IDX
from tsl.features.normalize import SELECTED_LANDMARKS, normalize_sequence


def test_selected_landmarks_is_deterministic_and_valid():
    assert isinstance(SELECTED_LANDMARKS, list)
    assert all(isinstance(i, int) for i in SELECTED_LANDMARKS)
    assert len(SELECTED_LANDMARKS) == 104
    assert len(set(SELECTED_LANDMARKS)) == 104
    assert min(SELECTED_LANDMARKS) >= 0
    assert max(SELECTED_LANDMARKS) <= 542
    assert NOSE_IDX in SELECTED_LANDMARKS


def test_output_shape_and_dtype():
    D = len(SELECTED_LANDMARKS) * 3
    seq = np.zeros((5, 543, 3), dtype=np.float32)
    seq[:, LSHOULDER_IDX] = [1.0, 0.0, 0.0]
    seq[:, RSHOULDER_IDX] = [-1.0, 0.0, 0.0]
    out = normalize_sequence(seq)
    assert out.shape == (5, D)
    assert out.dtype == np.float32


def test_nose_maps_to_zero():
    seq = np.zeros((3, 543, 3), dtype=np.float32)
    seq[:, NOSE_IDX] = [7.0, -3.0, 2.5]
    seq[:, LSHOULDER_IDX] = [1.0, 0.0, 0.0]
    seq[:, RSHOULDER_IDX] = [-1.0, 0.0, 0.0]
    out = normalize_sequence(seq)
    nose_pos = SELECTED_LANDMARKS.index(NOSE_IDX)
    nose_cols = out[:, nose_pos * 3:(nose_pos + 1) * 3]
    assert np.allclose(nose_cols, 0.0, atol=1e-5)


def test_shoulder_scaling():
    seq = np.zeros((1, 543, 3), dtype=np.float32)
    seq[:, NOSE_IDX] = [0.0, 0.0, 0.0]
    seq[:, LSHOULDER_IDX] = [2.0, 0.0, 0.0]
    seq[:, RSHOULDER_IDX] = [-2.0, 0.0, 0.0]
    out = normalize_sequence(seq)
    ls_pos = SELECTED_LANDMARKS.index(LSHOULDER_IDX)
    ls_cols = out[:, ls_pos * 3:(ls_pos + 1) * 3]
    assert ls_cols[0, 0] == pytest.approx(0.5, abs=1e-5)
    assert ls_cols[0, 1] == pytest.approx(0.0, abs=1e-5)


def test_nan_filled_to_zero():
    seq = np.zeros((2, 543, 3), dtype=np.float32)
    seq[:, NOSE_IDX] = [0.0, 0.0, 0.0]
    seq[:, LSHOULDER_IDX] = [1.0, 0.0, 0.0]
    seq[:, RSHOULDER_IDX] = [-1.0, 0.0, 0.0]
    seq[:, 468:489] = np.nan
    out = normalize_sequence(seq)
    assert np.isfinite(out).all()
    lh_pos = SELECTED_LANDMARKS.index(468)
    lh_cols = out[:, lh_pos * 3:(lh_pos + 1) * 3]
    assert np.allclose(lh_cols, 0.0)


def test_degenerate_shoulders_do_not_produce_inf():
    seq = np.zeros((2, 543, 3), dtype=np.float32)
    seq[:, NOSE_IDX] = [1.0, 1.0, 1.0]
    seq[:, LSHOULDER_IDX] = [0.0, 0.0, 0.0]
    seq[:, RSHOULDER_IDX] = [0.0, 0.0, 0.0]
    out = normalize_sequence(seq)
    assert np.isfinite(out).all()
