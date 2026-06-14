import numpy as np
import pytest

from tsl.features.landmarks import extract_frame_landmarks, extract_sequence


class _FakePoint:
    def __init__(self, x, y, z):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)


class _FakeLandmarkList:
    def __init__(self, n, base):
        self.landmark = [
            _FakePoint(base + i, base + i + 0.1, base + i + 0.2) for i in range(n)
        ]


class _FakeResults:
    def __init__(self, face, left_hand, pose, right_hand):
        self.face_landmarks = face
        self.left_hand_landmarks = left_hand
        self.pose_landmarks = pose
        self.right_hand_landmarks = right_hand


class _FakeHolistic:
    def __init__(self, results):
        self._results = results

    def process(self, image_rgb):
        assert image_rgb.ndim == 3 and image_rgb.shape[2] == 3
        return self._results


def _full_results():
    return _FakeResults(
        face=_FakeLandmarkList(468, base=0.0),
        left_hand=_FakeLandmarkList(21, base=100.0),
        pose=_FakeLandmarkList(33, base=200.0),
        right_hand=_FakeLandmarkList(21, base=300.0),
    )


def test_extract_frame_shape_and_dtype():
    holistic = _FakeHolistic(_full_results())
    frame = np.zeros((4, 5, 3), dtype=np.uint8)
    out = extract_frame_landmarks(holistic, frame)
    assert isinstance(out, np.ndarray)
    assert out.shape == (543, 3)
    assert out.dtype == np.float32


def test_extract_frame_concat_order():
    holistic = _FakeHolistic(_full_results())
    frame = np.zeros((4, 5, 3), dtype=np.uint8)
    out = extract_frame_landmarks(holistic, frame)
    assert out[0, 0] == pytest.approx(0.0)
    assert out[468, 0] == pytest.approx(100.0)
    assert out[489, 0] == pytest.approx(200.0)
    assert out[522, 0] == pytest.approx(300.0)
    assert out[542, 0] == pytest.approx(320.0)


def test_extract_frame_missing_component_is_nan():
    results = _FakeResults(
        face=_FakeLandmarkList(468, base=0.0),
        left_hand=None,
        pose=_FakeLandmarkList(33, base=200.0),
        right_hand=_FakeLandmarkList(21, base=300.0),
    )
    holistic = _FakeHolistic(results)
    frame = np.zeros((4, 5, 3), dtype=np.uint8)
    out = extract_frame_landmarks(holistic, frame)
    assert np.isnan(out[468:489]).all()
    assert np.isfinite(out[0:468]).all()
    assert np.isfinite(out[489:543]).all()


def test_extract_sequence_stacks_frames():
    holistic = _FakeHolistic(_full_results())
    frames = [np.zeros((4, 5, 3), dtype=np.uint8) for _ in range(3)]
    seq = extract_sequence(holistic, frames)
    assert seq.shape == (3, 543, 3)
    assert seq.dtype == np.float32
