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
import csv
import os
import random
import sys

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Column-name alias tables (tried in order, first match wins)
# ---------------------------------------------------------------------------
_START_ALIASES = ["start_ms", "start", "START_MS", "START", "begin_ms", "begin", "start_time"]
_END_ALIASES   = ["end_ms",   "end",   "END_MS",   "END",   "finish_ms","finish","end_time"]
_TEXT_ALIASES  = ["text", "TEXT", "sentence", "SENTENCE", "transcript", "label", "caption"]

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


def _iter_transcript_files(data_root: str):
    """Yield (session_name, transcript_df) for every transcript CSV found."""
    process_dir = os.path.join(data_root, "process_videos")
    if not os.path.isdir(process_dir):
        raise FileNotFoundError(
            f"process_videos/ not found under {data_root!r}.\n"
            "Is --data-root pointing at the ThaiSignVis root?"
        )

    found = False
    # Pattern 1: <data_root>/transcript_window_<session>.csv
    for fname in sorted(os.listdir(data_root)):
        if fname.startswith("transcript_window") and fname.endswith(".csv"):
            session = fname.replace("transcript_window_", "").replace(".csv", "")
            df = pd.read_csv(os.path.join(data_root, fname))
            df["_session"] = session
            yield session, df
            found = True

    # Pattern 2: <process_videos>/<session>/transcript_window*.csv
    for session in sorted(os.listdir(process_dir)):
        session_dir = os.path.join(process_dir, session)
        if not os.path.isdir(session_dir):
            continue
        for fname in sorted(os.listdir(session_dir)):
            if fname.startswith("transcript_window") and fname.endswith(".csv"):
                df = pd.read_csv(os.path.join(session_dir, fname))
                df["_session"] = session
                yield session, df
                found = True

    if not found:
        raise FileNotFoundError(
            "No transcript_window_*.csv files found.\n"
            f"Searched: {data_root!r} and {process_dir}/<session>/\n"
            "Verify the ThaiSignVis download is complete."
        )


def _find_video(data_root: str, session: str, video_id: str) -> str | None:
    """Try common locations for a session's video file."""
    candidates = [
        os.path.join(data_root, "process_videos", session, f"{video_id}.mp4"),
        os.path.join(data_root, "process_videos", f"{video_id}.mp4"),
        os.path.join(data_root, "process_videos", session, video_id),
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p
    # glob-style fallback: walk session dir
    session_dir = os.path.join(data_root, "process_videos", session)
    if os.path.isdir(session_dir):
        for fname in os.listdir(session_dir):
            if fname.endswith(".mp4") and (
                video_id in fname or fname.startswith(video_id[:8])
            ):
                return os.path.join(session_dir, fname)
    return None


def _extract_segment(
    video_path: str,
    start_ms: float,
    end_ms: float,
    target_fps: int,
    holistic,
) -> np.ndarray:
    """Return (T, 543, 3) raw MediaPipe frames for the requested segment."""
    import cv2  # type: ignore
    from tsl.features.landmarks import extract_frame_landmarks

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
            frames_raw.append(extract_frame_landmarks(holistic, frame))
    cap.release()
    if not frames_raw:
        return np.empty((0, 543, 3), dtype=np.float32)
    return np.stack(frames_raw, axis=0)  # (T, 543, 3)


def _build_segments(data_root: str, limit: int | None) -> list[dict]:
    """Parse all transcript CSVs → list of segment dicts."""
    segments: list[dict] = []
    for session, df in _iter_transcript_files(data_root):
        cols = list(df.columns)
        start_col = _find_col(cols, _START_ALIASES, "start_time")
        end_col   = _find_col(cols, _END_ALIASES,   "end_time")
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
            video_id = str(row[vid_col]) if vid_col else session
            seg_id = f"{session}__{idx:05d}"
            segments.append(
                {
                    "segment_id": seg_id,
                    "session":    session,
                    "video_id":   video_id,
                    "start_ms":   float(row[start_col]),
                    "end_ms":     float(row[end_col]),
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
    except ImportError:
        print("ERROR: mediapipe not installed.  Run:  pip install mediapipe opencv-python", file=sys.stderr)
        return 1

    try:
        import cv2  # type: ignore
    except ImportError:
        print("ERROR: opencv-python not installed.  Run:  pip install opencv-python", file=sys.stderr)
        return 1

    # normalize_sequence lives in tsl.features.normalize (312-dim output)
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
    from tsl.features.normalize import normalize_sequence  # type: ignore

    manifest_rows: list[dict] = []
    skipped = 0
    missing_video = 0

    with mp.solutions.holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as holistic:
        for i, seg in enumerate(segments):
            npy_rel = os.path.join("landmarks", f"{seg['segment_id']}.npy")
            npy_abs = os.path.join(args.out_dir, npy_rel)

            if os.path.isfile(npy_abs):
                manifest_rows.append(
                    {
                        "segment_id": seg["segment_id"],
                        "npy_path":   npy_rel,
                        "text":       seg["text"],
                        "video_id":   seg["video_id"],
                        "start_ms":   seg["start_ms"],
                        "end_ms":     seg["end_ms"],
                        "split":      seg["split"],
                    }
                )
                continue  # idempotent: skip already-extracted

            video_path = _find_video(args.data_root, seg["session"], seg["video_id"])
            if video_path is None:
                missing_video += 1
                continue

            raw = _extract_segment(
                video_path, seg["start_ms"], seg["end_ms"], args.fps, holistic
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
                    "video_id":   seg["video_id"],
                    "start_ms":   seg["start_ms"],
                    "end_ms":     seg["end_ms"],
                    "split":      seg["split"],
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
