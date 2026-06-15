"""Download YouTube-SL-25 Thai sign language subset and extract landmarks.

YouTube-SL-25 (Google Research) contains ~7 hours of Thai Sign Language (tsq)
from 106 videos with auto-aligned captions. The data lives in a public GCS bucket.

This script:
  1. Fetches the Thai video-ID list from GCS
  2. Downloads each video + Thai captions via yt-dlp
  3. Cuts segments from caption timestamps
  4. Extracts MediaPipe Holistic → normalize.py → (T, 312) .npy (162-dim option available)
  5. Writes a manifest.csv compatible with tsl.data.thaisignvis loader

Prerequisites
-------------
    pip install yt-dlp mediapipe opencv-python

Usage
-----
    python scripts/download_youtube_sl25_thai.py \\
        --out-dir data/youtube_sl25_thai \\
        --limit   30          # videos to process (omit for all 106)
        --dim     312         # 312 (default, normalize.py) or 162 (TSL-51 compat)
        --workers 2           # parallel yt-dlp downloads
"""
from __future__ import annotations

import argparse
import os
import random
import subprocess
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# GCS paths (public bucket, no auth needed)
# ---------------------------------------------------------------------------
_GCS_BASE = "https://storage.googleapis.com/gresearch/youtube-sl-25"
_METADATA_URL = f"{_GCS_BASE}/youtube-sl-25-metadata.csv"  # video_id,lang_code CSV

# MediaPipe Tasks holistic landmarker model (mediapipe >= 0.10 Tasks API)
_HOLISTIC_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/holistic_landmarker/"
    "holistic_landmarker/float16/latest/holistic_landmarker.task"
)
_HOLISTIC_MODEL_FILENAME = "holistic_landmarker.task"


def _fetch_video_ids(limit: int | None) -> list[str]:
    """Fetch Thai (tsq) video IDs from the YouTube-SL-25 metadata CSV."""
    try:
        with urllib.request.urlopen(_METADATA_URL, timeout=30) as r:
            data = r.read()
        ids = []
        for line in data.replace(b"\r", b"").split(b"\n"):
            if b",tsq" in line:
                vid = line.split(b",")[0].decode("ascii", errors="replace").strip()
                if vid:
                    ids.append(vid)
        print(f"  fetched {len(ids)} Thai (tsq) video IDs from metadata CSV")
        if limit:
            ids = ids[:limit]
        return ids
    except Exception as e:
        print(f"  WARNING: could not fetch metadata CSV ({e})")

    # Fallback: verified Thai video IDs from YouTube-SL-25 (as of 2025-06)
    print("  Falling back to built-in seed list.")
    seed = [
        "VpXLUNvogJM","wD42Hx2-qKI","wW77GSGjx6o","w9n34ph8Ves","xzpmtJMFA-Q",
        "ZmsMTy2u1sA","X1w_1HbvD1U","-xyTG_ffCJ0","y2UkYonlhA4","BtOCKbs8rmw",
        "JUxRnR4ASF4","nSCOt-iXxT0","9kSsZajhH9I","Q0nTRlutCTI","6RhxHpiMb9s",
        "7wJGk38xftw","sSZfvLt6ups","oR8x0jBuipk","eub4UhTx0tQ","fxygKINuEpY",
        "gT2aONcQdKA","hkJfJkBpW-A","iLxIJH64OFo","jQ1mQJOe7KQ","k7pLbFaH_Ng",
        "l5qHoRXSGHs","mDl4rWBfYhc","nQpO1kZK-Pg","oVp_HkGEYeI","pRx0k4QVy8g",
    ]
    if limit:
        seed = seed[:limit]
    return seed


def _download_video(video_id: str, out_dir: str) -> tuple[str | None, str | None]:
    """Download video + Thai auto-captions via yt-dlp.

    Returns (video_path, caption_path) or (None, None) on failure.
    """
    url = f"https://www.youtube.com/watch?v={video_id}"
    video_path = os.path.join(out_dir, f"{video_id}.mp4")
    # yt-dlp may write .th.vtt (manual) or .th-th.vtt (auto-gen)
    cap_path = None
    for suffix in (f"{video_id}.th.vtt", f"{video_id}.th-th.vtt"):
        p = os.path.join(out_dir, suffix)
        if os.path.isfile(p):
            cap_path = p
            break

    if os.path.isfile(video_path) and cap_path is not None:
        return video_path, cap_path  # already downloaded
    cap_path = None  # reset for fresh download

    cmd = [
        "yt-dlp",
        "--js-runtimes", "node",
        "--remote-components", "ejs:github",
        "--format", "bestvideo[height<=480][ext=mp4]+bestaudio[ext=m4a]/mp4[height<=480]/best[height<=480][ext=mp4]/best[height<=480]",
        "--merge-output-format", "mp4",
        "--write-sub",        # prefer manual subs
        "--write-auto-sub",   # also auto-generated
        "--sub-lang", "th,th-th",
        "--convert-subs", "vtt",
        "--no-playlist",
        "--output", os.path.join(out_dir, f"{video_id}.%(ext)s"),
        "--quiet",
        url,
    ]
    try:
        subprocess.run(cmd, check=True, timeout=300)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        return None, None

    if not os.path.isfile(video_path):
        return None, None
    # Find whichever caption file was saved
    for suffix in (f"{video_id}.th.vtt", f"{video_id}.th-th.vtt"):
        p = os.path.join(out_dir, suffix)
        if os.path.isfile(p):
            return video_path, p
    return None, None


def _parse_vtt(cap_path: str) -> list[dict]:
    """Parse a .vtt caption file into [{start_ms, end_ms, text}, ...].

    Handles WebVTT format produced by yt-dlp --convert-subs vtt.
    """
    segments: list[dict] = []
    try:
        with open(cap_path, encoding="utf-8") as f:
            content = f.read()
    except OSError:
        return segments

    for block in content.split("\n\n"):
        lines = [l.strip() for l in block.strip().splitlines() if l.strip()]
        # Find timestamp line: 00:00:01.000 --> 00:00:03.000
        ts_line = next((l for l in lines if "-->" in l), None)
        if ts_line is None:
            continue
        text_lines = [l for l in lines if "-->" not in l and not l.isdigit() and l != "WEBVTT"]
        text = " ".join(text_lines).strip()
        # Remove VTT tags <c>, </c>, <00:00:01.234>
        import re
        text = re.sub(r"<[^>]+>", "", text).strip()
        if not text:
            continue
        try:
            start_str, end_str = ts_line.split("-->")
            start_ms = _vtt_time_to_ms(start_str.strip().split()[0])
            end_ms   = _vtt_time_to_ms(end_str.strip().split()[0])
        except Exception:
            continue
        if end_ms <= start_ms:
            continue
        segments.append({"start_ms": start_ms, "end_ms": end_ms, "text": text})
    return segments


def _vtt_time_to_ms(t: str) -> float:
    """'00:01:23.456' or '01:23.456' → milliseconds."""
    parts = t.split(":")
    if len(parts) == 3:
        h, m, s = parts
        return (int(h) * 3600 + int(m) * 60 + float(s)) * 1000
    elif len(parts) == 2:
        m, s = parts
        return (int(m) * 60 + float(s)) * 1000
    return float(t) * 1000


def _ensure_holistic_model(cache_dir: str) -> str:
    """Download the holistic landmarker .task file if not already cached."""
    model_path = os.path.join(cache_dir, _HOLISTIC_MODEL_FILENAME)
    if not os.path.isfile(model_path):
        print(f"  Downloading holistic landmarker model (~50 MB) …")
        urllib.request.urlretrieve(_HOLISTIC_MODEL_URL, model_path)
        print(f"  Saved to {model_path}")
    return model_path


def _result_to_frame(result) -> np.ndarray:
    """Convert HolisticLandmarkerResult (Tasks API) to (543, 3) float32.

    Layout: face(468) | left_hand(21) | pose(33) | right_hand(21)
    Missing components are filled with NaN.
    In the Tasks API, each component is a flat list of NormalizedLandmark.
    """
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


def _extract_segment_npy(
    video_path: str,
    start_ms: float,
    end_ms: float,
    landmarker,
    dim: int,
) -> np.ndarray | None:
    """Extract (T, dim) feature array for one caption segment."""
    import cv2
    import mediapipe as mp

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    start_f = max(0, int(start_ms / 1000.0 * fps))
    end_f   = min(total, int(end_ms   / 1000.0 * fps))
    if end_f <= start_f:
        cap.release()
        return None

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
    raw_frames = []
    for _ in range(end_f - start_f):
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = np.ascontiguousarray(frame[:, :, ::-1])
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result = landmarker.detect(mp_image)
        raw_frames.append(_result_to_frame(result))
    cap.release()

    if not raw_frames:
        return None
    seq = np.stack(raw_frames, axis=0)  # (T, 543, 3)

    if dim == 312:
        from tsl.features.normalize import normalize_sequence
        return normalize_sequence(seq)
    else:
        # 162-dim: select same subset as TSL-51
        # left_hand(63) + right_hand(63) + 6 pose upper(18) + 6 face key(18)
        lh = seq[:, 468:489, :].reshape(len(seq), -1)  # (T, 63)
        rh = seq[:, 522:543, :].reshape(len(seq), -1)  # (T, 63)
        pose_abs = [489 + i for i in [0, 11, 12, 13, 14, 15]]
        pose = seq[:, pose_abs, :].reshape(len(seq), -1)   # (T, 18)
        face_idx = [70, 63, 300, 293, 61, 291]
        face = seq[:, face_idx, :].reshape(len(seq), -1)   # (T, 18)
        feat = np.concatenate([lh, rh, pose, face], axis=1)  # (T, 162)
        feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)
        return feat.astype(np.float32)


def _extract_video(args_tuple: tuple) -> tuple[str, list[dict]]:
    """Worker: extract all .npy landmark files for one video and return rows.

    Designed to run in a subprocess (multiprocessing).  Every parameter is
    passed via a single tuple so it is picklable.
    """
    vid_id, vpath, cpath, out_dir, dim, model_path, src_path = args_tuple

    # Import heavy dependencies inside worker so spawn is safe.
    import mediapipe as mp
    from mediapipe.tasks import python as mp_tasks
    from mediapipe.tasks.python import vision
    import sys as _sys
    _sys.path.insert(0, src_path)

    options = vision.HolisticLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.IMAGE,
        min_face_detection_confidence=0.5,
        min_pose_detection_confidence=0.5,
        min_hand_landmarks_confidence=0.5,
    )
    segs = _parse_vtt(cpath)
    rows: list[dict] = []
    with vision.HolisticLandmarker.create_from_options(options) as landmarker:
        for j, seg in enumerate(segs):
            seg_id = f"ytsl25_{vid_id}_{j:04d}"
            npy_rel = os.path.join("landmarks", f"{seg_id}.npy")
            npy_abs = os.path.join(out_dir, npy_rel)
            if os.path.isfile(npy_abs):
                rows.append({**seg, "segment_id": seg_id, "npy_path": npy_rel,
                              "video_id": vid_id})
                continue
            feat = _extract_segment_npy(
                vpath, seg["start_ms"], seg["end_ms"], landmarker, dim
            )
            if feat is None or feat.shape[0] == 0:
                continue
            np.save(npy_abs, feat)
            rows.append({**seg, "segment_id": seg_id, "npy_path": npy_rel,
                          "video_id": vid_id})
    return vid_id, rows


def _assign_splits(segments: list[dict], val_frac: float, seed: int) -> None:
    rng = random.Random(seed)
    idx = list(range(len(segments)))
    rng.shuffle(idx)
    n_val = max(1, int(len(segments) * val_frac))
    val_set = set(idx[:n_val])
    for i, seg in enumerate(segments):
        seg["split"] = "val" if i in val_set else "train"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Download YouTube-SL-25 Thai subset and extract landmarks."
    )
    parser.add_argument("--out-dir",  required=True, help="Output directory.")
    parser.add_argument("--limit",    type=int, default=None, help="Max videos to process.")
    parser.add_argument("--dim",      type=int, default=312, choices=[162, 312],
                        help="Feature dim: 312 (normalize.py, default) or 162 (TSL-51 compat).")
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--workers",  type=int, default=2, help="Parallel yt-dlp downloads.")
    parser.add_argument("--workers-extract", type=int, default=4,
                        help="Parallel processes for landmark extraction (default: 4).")
    parser.add_argument("--seed",     type=int, default=42)
    args = parser.parse_args(argv)

    os.makedirs(args.out_dir, exist_ok=True)
    video_dir = os.path.join(args.out_dir, "videos")
    lm_dir    = os.path.join(args.out_dir, "landmarks")
    os.makedirs(video_dir, exist_ok=True)
    os.makedirs(lm_dir, exist_ok=True)

    # check yt-dlp
    try:
        subprocess.run(["yt-dlp", "--version"], capture_output=True, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        print("ERROR: yt-dlp not installed.  Run:  pip install yt-dlp", file=sys.stderr)
        return 1

    # check mediapipe tasks API (requires mediapipe >= 0.10)
    try:
        from mediapipe.tasks.python import vision as _mp_vision  # noqa: F401
        import cv2  # noqa: F401
    except ImportError as e:
        print(f"ERROR: {e}.  Run:  pip install mediapipe opencv-python", file=sys.stderr)
        return 1

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

    print("Fetching Thai video IDs …")
    video_ids = _fetch_video_ids(args.limit)
    print(f"  {len(video_ids)} videos to process")

    # ---- Phase 1: download videos + captions --------------------------------
    print(f"\nDownloading videos (workers={args.workers}) …")

    def _dl(vid_id):
        return vid_id, *_download_video(vid_id, video_dir)

    downloaded: list[tuple[str, str, str]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_dl, vid): vid for vid in video_ids}
        for i, fut in enumerate(as_completed(futures), 1):
            vid_id, vpath, cpath = fut.result()
            status = "ok" if vpath else "fail"
            print(f"  [{i}/{len(video_ids)}] {vid_id}: {status}")
            if vpath and cpath:
                downloaded.append((vid_id, vpath, cpath))

    print(f"  {len(downloaded)}/{len(video_ids)} downloaded successfully")
    if not downloaded:
        print("ERROR: no videos downloaded.", file=sys.stderr)
        return 1

    # ---- Phase 2: extract landmarks (parallel per video) -------------------
    import multiprocessing as _mp_proc
    n_extract = min(args.workers_extract, len(downloaded))
    print(f"\nExtracting landmarks ({n_extract} parallel workers) …", flush=True)

    model_path = _ensure_holistic_model(args.out_dir)
    src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))

    worker_args = [
        (vid_id, vpath, cpath, args.out_dir, args.dim, model_path, src_path)
        for vid_id, vpath, cpath in downloaded
    ]

    all_rows: list[dict] = []
    ctx = _mp_proc.get_context("fork")
    with ctx.Pool(n_extract) as pool:
        for i, (vid_id, rows) in enumerate(
            pool.imap_unordered(_extract_video, worker_args), 1
        ):
            print(f"  [{i}/{len(downloaded)}] {vid_id}: {len(rows)} seg(s)", flush=True)
            all_rows.extend(rows)

    if not all_rows:
        print("ERROR: no segments extracted.", file=sys.stderr)
        return 1

    _assign_splits(all_rows, args.val_frac, args.seed)

    manifest_path = os.path.join(args.out_dir, "manifest.csv")
    df = pd.DataFrame(all_rows)[
        ["segment_id", "npy_path", "text", "video_id", "start_ms", "end_ms", "split"]
    ]
    df.to_csv(manifest_path, index=False, encoding="utf-8")

    train_n = (df["split"] == "train").sum()
    val_n   = (df["split"] == "val").sum()
    print(f"\nDone.")
    print(f"  segments  : {len(df)}  (train={train_n}, val={val_n})")
    print(f"  feature dim: {args.dim}")
    print(f"  manifest  : {manifest_path}")
    print(f"\nTo train:")
    stage = "thaisignvis" if args.dim == 312 else "tsl51"
    print(f"  PYTHONPATH=src python -m tsl.train.train_slt \\")
    print(f"      --stage {stage} --data-root {args.out_dir} \\")
    print(f"      --epochs 20 --batch-size 8 --model-size base \\")
    print(f"      --out-dir checkpoints/slt_ytsl25")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
