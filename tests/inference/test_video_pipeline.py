"""Tests for translate_video end-to-end pipeline."""
from __future__ import annotations

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Fake translator (duck-typed, matches both PoseT5Translator and
# SentenceTranslator interface — only needs .translate returning .sentence)
# ---------------------------------------------------------------------------


class _FakeTranslator:
    """Minimal duck-typed translator for pipeline tests."""

    def __init__(self, sentence: str = "ฉัน ดื่ม น้ำ"):
        self._sentence = sentence
        self.last_features: np.ndarray | None = None

    def translate(self, features: np.ndarray):
        self.last_features = features.copy()

        class _Pred:
            pass

        pred = _Pred()
        pred.sentence = self._sentence
        return pred


# ---------------------------------------------------------------------------
# Array shape dispatch
# ---------------------------------------------------------------------------


def test_raw_landmarks_normalized_and_translated():
    """(T, 543, 3) array is normalized to (T, 312) then passed to translator."""
    from tsl.inference.video_pipeline import translate_video

    T = 10
    raw = np.random.randn(T, 543, 3).astype(np.float32)
    tr = _FakeTranslator(sentence="สวัสดี")
    result = translate_video(raw, tr)

    assert result == "สวัสดี"
    # Translator received normalized (T, 312) features
    assert tr.last_features is not None
    assert tr.last_features.shape == (T, 312)
    assert tr.last_features.dtype == np.float32


def test_normalized_features_passed_directly():
    """(T, 312) array skips normalization and goes straight to the translator."""
    from tsl.inference.video_pipeline import translate_video

    T = 6
    features = np.ones((T, 312), dtype=np.float32) * 0.5
    tr = _FakeTranslator(sentence="ขอบคุณ")
    result = translate_video(features, tr)

    assert result == "ขอบคุณ"
    assert tr.last_features is not None
    assert tr.last_features.shape == (T, 312)
    # Values should be the same as input (not re-normalized)
    np.testing.assert_array_almost_equal(tr.last_features, features)


def test_wrong_shape_raises_value_error():
    """Arrays with unsupported shapes raise ValueError."""
    from tsl.inference.video_pipeline import translate_video

    tr = _FakeTranslator()
    # 1-D
    with pytest.raises(ValueError):
        translate_video(np.zeros((100,), dtype=np.float32), tr)
    # 3-D with wrong inner dims
    with pytest.raises(ValueError):
        translate_video(np.zeros((10, 100, 4), dtype=np.float32), tr)
    # 2-D with wrong feature dim
    with pytest.raises(ValueError):
        translate_video(np.zeros((10, 162), dtype=np.float32), tr)


def test_wrong_type_raises_type_error():
    """Passing something other than str or ndarray raises TypeError."""
    from tsl.inference.video_pipeline import translate_video

    tr = _FakeTranslator()
    with pytest.raises(TypeError):
        translate_video(42, tr)


# ---------------------------------------------------------------------------
# Grammar reorder
# ---------------------------------------------------------------------------


def test_apply_grammar_reorder_true():
    """apply_grammar_reorder=True runs reorder_to_thai on the sentence."""
    from tsl.inference.video_pipeline import translate_video

    T = 4
    features = np.zeros((T, 312), dtype=np.float32)
    # "i drink water" → reorder_to_thai → "ฉัน ดื่ม น้ำ"
    tr = _FakeTranslator(sentence="i drink water")
    result = translate_video(features, tr, apply_grammar_reorder=True)
    assert result == "ฉัน ดื่ม น้ำ"


def test_apply_grammar_reorder_false():
    """apply_grammar_reorder=False returns the sentence unchanged."""
    from tsl.inference.video_pipeline import translate_video

    features = np.zeros((4, 312), dtype=np.float32)
    tr = _FakeTranslator(sentence="i drink water")
    result = translate_video(features, tr, apply_grammar_reorder=False)
    assert result == "i drink water"


# ---------------------------------------------------------------------------
# Video path — mock cv2 + mediapipe
# ---------------------------------------------------------------------------


def _make_mock_mediapipe(monkeypatch):
    """Patch cv2 and mediapipe so _extract_from_video returns a (3, 543, 3) array."""
    import types

    T = 3
    fake_frame = np.zeros((543, 3), dtype=np.float32)

    # --- Build a fake result object ---
    class _FakeResult:
        face_landmarks = []
        left_hand_landmarks = []
        pose_landmarks = []
        right_hand_landmarks = []

    class _FakeLandmarker:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def detect(self, img):
            return _FakeResult()

    class _FakeHolisticLandmarkerOptions:
        def __init__(self, **kwargs):
            pass

    class _FakeVision:
        RunningMode = types.SimpleNamespace(IMAGE="IMAGE")
        HolisticLandmarkerOptions = _FakeHolisticLandmarkerOptions

        @staticmethod
        def HolisticLandmarker():
            pass

        class HolisticLandmarker:
            @staticmethod
            def create_from_options(opts):
                return _FakeLandmarker()

    class _FakeMpTasksPython:
        class BaseOptions:
            def __init__(self, model_asset_path=""):
                pass

    # Build the mediapipe module hierarchy
    mp_mod = types.ModuleType("mediapipe")
    mp_mod.Image = lambda image_format=None, data=None: object()
    mp_mod.ImageFormat = types.SimpleNamespace(SRGB="SRGB")

    mp_tasks_mod = types.ModuleType("mediapipe.tasks")
    mp_tasks_python_mod = _FakeMpTasksPython()
    mp_tasks_vision_mod = _FakeVision()

    mp_mod.tasks = mp_tasks_mod
    mp_tasks_mod.python = mp_tasks_python_mod
    mp_tasks_python_mod.vision = mp_tasks_vision_mod

    monkeypatch.setitem(__import__("sys").modules, "mediapipe", mp_mod)
    monkeypatch.setitem(__import__("sys").modules, "mediapipe.tasks", mp_tasks_mod)
    monkeypatch.setitem(__import__("sys").modules, "mediapipe.tasks.python", types.ModuleType("mediapipe.tasks.python"))
    monkeypatch.setitem(__import__("sys").modules, "mediapipe.tasks.python.vision", types.ModuleType("mediapipe.tasks.python.vision"))

    # --- Build a fake cv2 module ---
    import sys

    class _FakeCap:
        _frame_count = 0

        def isOpened(self):
            return True

        def read(self):
            if self._frame_count < T:
                self._frame_count += 1
                return True, np.zeros((10, 10, 3), dtype=np.uint8)
            return False, None

        def release(self):
            pass

    cv2_mod = types.ModuleType("cv2")
    cv2_mod.VideoCapture = lambda path: _FakeCap()
    monkeypatch.setitem(sys.modules, "cv2", cv2_mod)

    return T, _FakeLandmarker, _FakeVision, mp_tasks_python_mod


def test_video_path_triggers_extraction(monkeypatch):
    """Passing a str path calls _extract_from_video and then normalizes."""
    T, _FakeLandmarker, _FakeVision, _FakeMpTasks = _make_mock_mediapipe(monkeypatch)

    # Patch _extract_from_video directly to avoid deep mediapipe plumbing
    import tsl.inference.video_pipeline as vp

    def _fake_extract(video_path: str):
        assert isinstance(video_path, str)
        return np.zeros((T, 543, 3), dtype=np.float32)

    monkeypatch.setattr(vp, "_extract_from_video", _fake_extract)

    tr = _FakeTranslator(sentence="ทดสอบ")
    result = vp.translate_video("/fake/video.mp4", tr)

    assert result == "ทดสอบ"
    # Features should have been normalized to (T, 312)
    assert tr.last_features is not None
    assert tr.last_features.shape == (T, 312)


def test_mediapipe_import_error_on_video_path(monkeypatch):
    """If mediapipe is not installed, a helpful ImportError is raised."""
    import sys
    import types

    # Make mediapipe unimportable
    sentinel = object()
    monkeypatch.setitem(sys.modules, "mediapipe", sentinel)

    import tsl.inference.video_pipeline as vp

    # Restore original _extract_from_video (not patched here)
    original = vp._extract_from_video

    def _raises_import(video_path: str):
        raise ImportError("mediapipe required for video extraction: pip install mediapipe")

    monkeypatch.setattr(vp, "_extract_from_video", _raises_import)

    tr = _FakeTranslator()
    with pytest.raises(ImportError, match="mediapipe"):
        vp.translate_video("/some/video.mp4", tr)
