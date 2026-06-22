"""Extract MediaPipe Holistic landmarks from ThaiSignVis videos.

Reads the ThaiSignVis directory layout:

    <data_root>/
        process_videos/
            <session>/
                *.mp4
        metadata.csv                      (optional top-level)
        transcript_window_<session>.csv   (one per session, OR inside session dir)

Produces:

    <out_dir>/
        landmarks/<segment_id>.npy        (T, 312) float32 via normalize_sequence
        manifest.csv                      loader-ready manifest

Usage
-----
    python scripts/extract_thaisignvis_landmarks.py \\
        --data-root /path/to/thaisignvis \\
        --out-dir   data/thaisignvis \\
        --limit     50              # optional: process first N segments
        --val-frac  0.1             # fraction held out as val split (default 0.1)
        --fps       0               # 0 = native fps, >0 = resample

The script is idempotent: segments whose .npy already exists are skipped.

Column-name fallbacks
---------------------
ThaiSignVis transcript CSVs are not yet publicly documented.  The script
tries a list of common column aliases for start/end time and text, and
raises a clear error if none match.  Adjust TRANSCRIPT_ALIASES below if
your download uses different names.
"""
from __future__ import annotations

import argparse
import os
import random
import sys
import urllib.request

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Column-name alias tables (tried in order, first match wins)
# ---------------------------------------------------------------------------
_START_ALIASES = ["start_ms", "start", "START_MS", "START", "begin_ms", "begin", "start_time"]
_END_ALIASES   = ["end_ms",   "end",   "END_MS",   "END",   "finish_ms","finish","end_time"]
_TEXT_ALIASES  = ["text", "TEXT", "sentence", "SENTENCE", "transcript", "label", "caption"]
_DURATION_ALIASES = ["duration_ms", "duration", "DURATION_MS", "DURATION"]
_SOURCE_NAME = "thaisignvis"
_HOLISTIC_MODEL_FILENAME = "holistic_landmarker.task"
_HOLISTIC_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/holistic_landmarker/"
    "holistic_landmarker/float16/latest/holistic_landmarker.task"
)

# ---------------------------------------------------------------------------


def _find_col(df_cols: list[str], aliases: list[str], role: str) -> str:
    for a in aliases:
        if a in df_cols:
            return a
    raise ValueError(
        f"Cannot find '{role}' column in transcript CSV.\n"
        f"Tried: {aliases}\nFound: {df_cols}\n"
        "Update the alias list at the top of extract_thaisignvis_landmarks.py."
    )


def _find_optional_col(df_cols: list[str], aliases: list[str]) -> str | None:
    for alias in aliases:
        if alias in df_cols:
            return alias
    return None


def _time_to_ms(value: float, column_name: str) -> float:
    text = column_name.lower()
    if "ms" in text:
        return float(value)
    if abs(float(value)) < 10000:
        return float(value) * 1000.0
    return float(value)


def _iter_transcript_files(data_root: str):
    """Yield ``(session_name, session_dir, transcript_tag, transcript_df)``."""
    found = False
    for root, _, files in os.walk(data_root):
        session = os.path.basename(root)
        for fname in sorted(files):
            if not (fname.startswith("transcript_window") and fname.endswith(".csv")):
                continue
            tag = os.path.splitext(fname)[0]
            df = pd.read_csv(os.path.join(root, fname))
            df["_session"] = session
            yield session, root, tag, df
            found = True

    if not found:
        raise FileNotFoundError(
            "No transcript_window_*.csv files found.\n"
            f"Searched recursively under {data_root!r}.\n"
            "Verify the ThaiSignVis download is complete."
        )


def _find_video(session_dir: str, video_id: str) -> str | None:
    """Try common locations for a session's video file."""
    candidates = []
    if video_id:
        basename = os.path.basename(str(video_id))
        candidates.extend(
            [
                os.path.join(session_dir, basename),
                os.path.join(session_dir, f"{basename}.mp4"),
            ]
        )

    for path in candidates:
        if os.path.isfile(path):
            return path

    mp4_paths = sorted(
        os.path.join(session_dir, fname)
        for fname in os.listdir(session_dir)
        if fname.lower().endswith(".mp4")
    )
    if not mp4_paths:
        return None

    basename = os.path.basename(str(video_id)) if video_id else ""
    if basename:
        for path in mp4_paths:
            name = os.path.basename(path)
            if basename == name or basename in name or name in basename:
                return path

    for preferred in ("process_video_0", "process_video", "video_0"):
        for path in mp4_paths:
            if os.path.basename(path).startswith(preferred):
                return path
    return mp4_paths[0]


def _find_model(out_dir: str, data_root: str) -> str:
    for search in [out_dir, data_root, "."]:
        cand = os.path.join(search, _HOLISTIC_MODEL_FILENAME)
        if os.path.isfile(cand):
            return cand
    dest = os.path.join(out_dir, _HOLISTIC_MODEL_FILENAME)
    print(f"Downloading holistic model -> {dest}")
    urllib.request.urlretrieve(_HOLISTIC_MODEL_URL, dest)
    return dest


def _result_to_frame(result) -> np.ndarray:
    def _lm(landmarks, n: int) -> np.ndarray:
        out = np.full((n, 3), np.nan, dtype=np.float32)
        if not landmarks:
            return out
        for i, point in enumerate(landmarks):
            if i >= n:
                break
            out[i] = [point.x, point.y, point.z]
        return out

    frame = np.full((543, 3), np.nan, dtype=np.float32)
    frame[0:468] = _lm(result.face_landmarks, 468)
    frame[468:489] = _lm(result.left_hand_landmarks, 21)
    frame[489:522] = _lm(result.pose_landmarks, 33)
    frame[522:543] = _lm(result.right_hand_landmarks, 21)
    return frame


def _extract_segment(
    video_path: str,
    start_ms: float,
    end_ms: float,
    target_fps: int,
    mp,
    landmarker,
) -> np.ndarray:
    """Return (T, 543, 3) raw MediaPipe frames for the requested segment."""
    import cv2  # type: ignore

    cap = cv2.VideoCapture(video_path)
    native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    start_frame = max(0, int(start_ms / 1000.0 * native_fps))
    end_frame   = min(total_frames, int(end_ms   / 1000.0 * native_fps))

    if target_fps > 0 and target_fps < native_fps:
        step = max(1, int(native_fps / target_fps))
    else:
        step = 1

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frames_raw: list[np.ndarray] = []
    for fi in range(start_frame, end_frame):
        ret, frame = cap.read()
        if not ret:
            break
        if (fi - start_frame) % step == 0:
            frame_rgb = np.ascontiguousarray(frame[:, :, ::-1])
            result = landmarker.detect(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            )
            frames_raw.append(_result_to_frame(result))
    cap.release()
    if not frames_raw:
        return np.empty((0, 543, 3), dtype=np.float32)
    return np.stack(frames_raw, axis=0)  # (T, 543, 3)


def _resolve_video_id(segment: dict, video_path: str | None) -> str:
    if segment["video_id"]:
        return str(segment["video_id"])
    if video_path:
        return os.path.splitext(os.path.basename(video_path))[0]
    return str(segment["session"])


def _build_segments(data_root: str, limit: int | None) -> list[dict]:
    """Parse all transcript CSVs → list of segment dicts."""
    segments: list[dict] = []
    for session, session_dir, transcript_tag, df in _iter_transcript_files(data_root):
        cols = list(df.columns)
        start_col = _find_col(cols, _START_ALIASES, "start_time")
        end_col = _find_optional_col(cols, _END_ALIASES)
        duration_col = _find_optional_col(cols, _DURATION_ALIASES)
        if end_col is None and duration_col is None:
            raise ValueError(
                f"Cannot find 'end_time' or 'duration' column in transcript CSV.\n"
                f"Tried end columns: {_END_ALIASES}\n"
                f"Tried duration columns: {_DURATION_ALIASES}\n"
                f"Found: {cols}"
            )
        text_col  = _find_col(cols, _TEXT_ALIASES,  "text")

        # video_id: use explicit column or fall back to session name
        vid_col = None
        for vc in ["video_id", "VIDEO_ID", "filename", "video"]:
            if vc in cols:
                vid_col = vc
                break

        for idx, row in df.iterrows():
            text = str(row[text_col]).strip()
            if not text or text.lower() in ("nan", "none", ""):
                continue
            video_id = str(row[vid_col]).strip() if vid_col else ""
            start_ms = _time_to_ms(float(row[start_col]), start_col)
            if end_col is not None:
                end_ms = _time_to_ms(float(row[end_col]), end_col)
            else:
                end_ms = start_ms + _time_to_ms(float(row[duration_col]), duration_col)
            seg_id = f"{session}__{transcript_tag}__{idx:05d}"
            segments.append(
                {
                    "segment_id": seg_id,
                    "session":    session,
                    "session_dir": session_dir,
                    "video_id":   video_id,
                    "start_ms":   start_ms,
                    "end_ms":     end_ms,
                    "text":       text,
                }
            )
        if limit is not None and len(segments) >= limit:
            break

    if limit is not None:
        segments = segments[:limit]
    return segments


def _assign_splits(segments: list[dict], val_frac: float, seed: int) -> None:
    rng = random.Random(seed)
    indices = list(range(len(segments)))
    rng.shuffle(indices)
    n_val = max(1, int(len(segments) * val_frac))
    val_set = set(indices[:n_val])
    for i, seg in enumerate(segments):
        seg["split"] = "val" if i in val_set else "train"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Extract MediaPipe landmarks from ThaiSignVis for SLT training."
    )
    parser.add_argument("--data-root", required=True, help="ThaiSignVis root directory.")
    parser.add_argument("--out-dir",   required=True, help="Output directory for .npy + manifest.")
    parser.add_argument("--limit",     type=int, default=None, help="Cap number of segments (for testing).")
    parser.add_argument("--val-frac",  type=float, default=0.1, help="Fraction held out as val.")
    parser.add_argument("--fps",       type=int, default=0, help="Target FPS (0 = native).")
    parser.add_argument("--seed",      type=int, default=42)
    args = parser.parse_args(argv)

    os.makedirs(args.out_dir, exist_ok=True)
    lm_dir = os.path.join(args.out_dir, "landmarks")
    os.makedirs(lm_dir, exist_ok=True)

    print("Parsing transcript CSVs …")
    segments = _build_segments(args.data_root, args.limit)
    if not segments:
        print("ERROR: no segments found.", file=sys.stderr)
        return 1
    print(f"  {len(segments)} segments from {len({s['session'] for s in segments})} sessions")

    _assign_splits(segments, args.val_frac, args.seed)

    # Import heavy deps late so --help works without them
    try:
        import mediapipe as mp  # type: ignore
        from mediapipe.tasks import python as mp_tasks  # type: ignore
        from mediapipe.tasks.python import vision  # type: ignore
    except ImportError:
        print("ERROR: mediapipe not installed.  Run:  pip install mediapipe opencv-python", file=sys.stderr)
        return 1

    try:
        import cv2  # type: ignore
    except ImportError:
        print("ERROR: opencv-python not installed.  Run:  pip install opencv-python", file=sys.stderr)
        return 1

    # normalize_sequence lives in tsl.features.normalize (312-dim output)
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    src_root = os.path.join(repo_root, "src")
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    if src_root not in sys.path:
        sys.path.insert(0, src_root)
    from tsl.features.normalize import FEATURE_LAYOUT_VERSION, normalize_sequence  # type: ignore

    model_path = _find_model(args.out_dir, args.data_root)
    options = vision.HolisticLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.IMAGE,
        min_face_detection_confidence=0.5,
        min_pose_detection_confidence=0.5,
        min_hand_landmarks_confidence=0.5,
    )

    manifest_rows: list[dict] = []
    skipped = 0
    missing_video = 0

    with vision.HolisticLandmarker.create_from_options(options) as landmarker:
        for i, seg in enumerate(segments):
            npy_rel = os.path.join("landmarks", f"{seg['segment_id']}.npy")
            npy_abs = os.path.join(args.out_dir, npy_rel)
            video_path = _find_video(seg["session_dir"], seg["video_id"])
            resolved_video_id = _resolve_video_id(seg, video_path)

            if os.path.isfile(npy_abs):
                manifest_rows.append(
                    {
                        "segment_id": seg["segment_id"],
                        "npy_path":   npy_rel,
                        "text":       seg["text"],
                        "video_id":   resolved_video_id,
                        "start_ms":   seg["start_ms"],
                        "end_ms":     seg["end_ms"],
                        "split":      seg["split"],
                        "source":     _SOURCE_NAME,
                        "feature_layout_version": FEATURE_LAYOUT_VERSION,
                    }
                )
                continue  # idempotent: skip already-extracted

            if video_path is None:
                missing_video += 1
                continue

            raw = _extract_segment(
                video_path,
                seg["start_ms"],
                seg["end_ms"],
                args.fps,
                mp,
                landmarker,
            )
            if raw.shape[0] == 0:
                skipped += 1
                continue

            feat = normalize_sequence(raw)  # (T, 312)
            np.save(npy_abs, feat)

            manifest_rows.append(
                {
                    "segment_id": seg["segment_id"],
                    "npy_path":   npy_rel,
                    "text":       seg["text"],
                    "video_id":   resolved_video_id,
                    "start_ms":   seg["start_ms"],
                    "end_ms":     seg["end_ms"],
                    "split":      seg["split"],
                    "source":     _SOURCE_NAME,
                    "feature_layout_version": FEATURE_LAYOUT_VERSION,
                }
            )

            if (i + 1) % 50 == 0:
                print(f"  [{i+1}/{len(segments)}] extracted …")

    manifest_path = os.path.join(args.out_dir, "manifest.csv")
    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False, encoding="utf-8")

    print(f"\nDone.")
    print(f"  extracted : {len(manifest_rows)}")
    print(f"  skipped (empty): {skipped}")
    print(f"  missing video  : {missing_video}")
    print(f"  manifest       : {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
