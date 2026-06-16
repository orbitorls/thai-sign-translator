"""End-to-end video-to-Thai-text pipeline.

:func:`translate_video` accepts either a video file path or a pre-extracted
numpy landmark array and returns a Thai sentence via a duck-typed translator
(e.g. :class:`tsl.inference.pose_t5_translator.PoseT5Translator` or
:class:`tsl.inference.sentence_translator.SentenceTranslator`).

MediaPipe is **lazy-imported** inside :func:`_extract_from_video` so the
module can be imported without mediapipe installed and tests can mock it.
"""
from __future__ import annotations

import numpy as np

__all__ = ["translate_video"]


def _extract_from_video(video_path: str) -> np.ndarray:
    """Extract raw landmarks ``(T, 543, 3)`` from a video file.

    Uses the MediaPipe Holistic Tasks API.  Requires:
      - ``mediapipe`` package
      - The holistic landmarker ``.task`` model file (path passed via the
        ``HOLISTIC_MODEL_PATH`` attribute of this module, defaulting to
        ``holistic_landmarker.task`` in the current directory)

    Args:
        video_path: Path to the input video file.

    Returns:
        ``(T, 543, 3)`` float32 array of per-frame landmarks (NaN for missing).

    Raises:
        ImportError: If mediapipe is not installed.
        FileNotFoundError: If the video cannot be opened.
    """
    try:
        import mediapipe as mp
        from mediapipe.tasks import python as mp_tasks
        from mediapipe.tasks.python import vision
    except ImportError as e:
        raise ImportError(
            "mediapipe required for video extraction: pip install mediapipe"
        ) from e

    import cv2

    # -----------------------------------------------------------------------
    # Helper: convert Tasks API result to (543, 3) frame
    # -----------------------------------------------------------------------
    def _result_to_frame(result) -> np.ndarray:
        def _lm_to_arr(lm_list, n: int) -> np.ndarray:
            out = np.full((n, 3), np.nan, dtype=np.float32)
            if not lm_list:
                return out
            for i, p in enumerate(lm_list):
                if i >= n:
                    break
                out[i, 0] = p.x
                out[i, 1] = p.y
                out[i, 2] = p.z
            return out

        frame = np.full((543, 3), np.nan, dtype=np.float32)
        frame[0:468]   = _lm_to_arr(result.face_landmarks, 468)
        frame[468:489] = _lm_to_arr(result.left_hand_landmarks, 21)
        frame[489:522] = _lm_to_arr(result.pose_landmarks, 33)
        frame[522:543] = _lm_to_arr(result.right_hand_landmarks, 21)
        return frame

    # -----------------------------------------------------------------------
    # Build landmarker
    # -----------------------------------------------------------------------
    import os
    model_path = getattr(_extract_from_video, "_holistic_model_path", "holistic_landmarker.task")

    options = vision.HolisticLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.IMAGE,
        min_face_detection_confidence=0.5,
        min_pose_detection_confidence=0.5,
        min_hand_landmarks_confidence=0.5,
    )

    # -----------------------------------------------------------------------
    # Read frames
    # -----------------------------------------------------------------------
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path!r}")

    raw_frames: list[np.ndarray] = []
    with vision.HolisticLandmarker.create_from_options(options) as landmarker:
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break
            frame_rgb = np.ascontiguousarray(frame_bgr[:, :, ::-1])
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            result = landmarker.detect(mp_image)
            raw_frames.append(_result_to_frame(result))
    cap.release()

    if not raw_frames:
        return np.empty((0, 543, 3), dtype=np.float32)
    return np.stack(raw_frames, axis=0)  # (T, 543, 3)


def translate_video(
    source,
    translator,
    apply_grammar_reorder: bool = False,
) -> str:
    """End-to-end pipeline: video path or landmark array → Thai text.

    Args:
        source: One of:
            - ``str``: Path to a video file.  MediaPipe is used to extract
              landmarks, which are then normalized and translated.
            - ``np.ndarray`` of shape ``(T, 543, 3)``: Raw holistic landmarks.
              Will be normalized via :func:`tsl.features.normalize.normalize_sequence`.
            - ``np.ndarray`` of shape ``(T, 312)``: Already-normalized features.
              Passed directly to the translator.
        translator: Object with a ``translate(features)`` method that returns
            an object with a ``.sentence`` attribute (e.g.
            :class:`PoseT5Translator` or :class:`SentenceTranslator`).
        apply_grammar_reorder: If ``True``, the output sentence is passed
            through :func:`tsl.grammar.reorder.reorder_to_thai` before
            returning (splits on whitespace, reorders SVO, rejoins).

    Returns:
        Decoded Thai text string.

    Raises:
        ValueError: If *source* is an ndarray with an unsupported shape.
        ImportError: If *source* is a video path and mediapipe is not installed.
    """
    from tsl.features.normalize import normalize_sequence

    # ------------------------------------------------------------------
    # Step 1 → obtain (T, 312) feature array
    # ------------------------------------------------------------------
    if isinstance(source, str):
        raw = _extract_from_video(source)          # (T, 543, 3)
        features = normalize_sequence(raw)         # (T, 312)
    elif isinstance(source, np.ndarray):
        arr = source
        if arr.ndim == 3 and arr.shape[1] == 543 and arr.shape[2] == 3:
            features = normalize_sequence(arr)     # (T, 312)
        elif arr.ndim == 2 and arr.shape[1] == 312:
            features = arr                         # already normalized
        else:
            raise ValueError(
                f"Unsupported array shape {tuple(arr.shape)}. "
                "Expected (T, 543, 3) for raw landmarks or (T, 312) for normalized features."
            )
    else:
        raise TypeError(
            f"source must be a str (video path) or np.ndarray; got {type(source).__name__!r}"
        )

    # ------------------------------------------------------------------
    # Step 2 → translate
    # ------------------------------------------------------------------
    prediction = translator.translate(features)
    sentence: str = prediction.sentence

    # ------------------------------------------------------------------
    # Step 3 → optional grammar reorder
    # ------------------------------------------------------------------
    if apply_grammar_reorder:
        from tsl.grammar.reorder import reorder_to_thai
        sentence = reorder_to_thai(sentence.split())

    return sentence
