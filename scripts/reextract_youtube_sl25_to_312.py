"""Re-extract YouTube-SL-25 Thai landmarks from local mp4s to 312-dim features.

Groups segments by video_id and opens each mp4 exactly once for all its segments.
Reads the existing manifest.csv (video_id, start_ms, end_ms), finds the source
mp4 in <src-dir>/videos/<video_id>.mp4, extracts MediaPipe Holistic for each
segment → normalize_sequence → (T, 312), saves to <out-dir>/landmarks/<seg_id>.npy.

Idempotent: skips segments whose .npy already exists.

Usage
-----
    python scripts/reextract_youtube_sl25_to_312.py \\
        --src-dir  data/youtube_sl25_thai \\
        --out-dir  data/youtube_sl25_thai_v3 \\
        --workers  4
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

_HOLISTIC_MODEL_FILENAME = "holistic_landmarker.task"
_HOLISTIC_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/holistic_landmarker/"
    "holistic_landmarker/float16/latest/holistic_landmarker.task"
)


def _find_model(src_dir: str) -> str:
    for search in [src_dir, "."]:
        cand = os.path.join(search, _HOLISTIC_MODEL_FILENAME)
        if os.path.isfile(cand):
            return cand
    import urllib.request
    dest = os.path.join(src_dir, _HOLISTIC_MODEL_FILENAME)
    print(f"  Downloading holistic model → {dest}")
    urllib.request.urlretrieve(_HOLISTIC_MODEL_URL, dest)
    return dest


def _result_to_frame(result) -> np.ndarray:
    def _lm(lm_list, n):
        out = np.full((n, 3), np.nan, dtype=np.float32)
        if not lm_list:
            return out
        for i, p in enumerate(lm_list):
            if i >= n:
                break
            out[i] = [p.x, p.y, p.z]
        return out
    frame = np.full((543, 3), np.nan, dtype=np.float32)
    frame[0:468]   = _lm(result.face_landmarks, 468)
    frame[468:489] = _lm(result.left_hand_landmarks, 21)
    frame[489:522] = _lm(result.pose_landmarks, 33)
    frame[522:543] = _lm(result.right_hand_landmarks, 21)
    return frame


_model_init_lock = threading.Lock()


def _process_video(args_tuple: tuple) -> list[dict]:
    """Process ALL segments of one video in a single mp4 open.

    Returns a list of result dicts, one per segment.
    """
    vid_id, video_path, segments, lm_dir, model_path, src_path = args_tuple
    # segments: list of (seg_id, start_ms, end_ms, npy_abs, npy_rel)

    # Ensure src/ and project root are importable (root has config.py)
    root_path = os.path.dirname(src_path)
    for p in (src_path, root_path):
        if p not in sys.path:
            sys.path.insert(0, p)

    results = []

    # Check how many already done
    todo = [(s, start, end, na, nr) for (s, start, end, na, nr) in segments
            if not os.path.isfile(na)]
    done_segs = [(s, nr) for (s, start, end, na, nr) in segments
                 if os.path.isfile(na)]
    for seg_id, npy_rel in done_segs:
        results.append({"seg_id": seg_id, "npy_path": npy_rel, "status": "skipped"})

    if not todo:
        return results

    try:
        import cv2
        import mediapipe as mp
        from mediapipe.tasks import python as mp_tasks
        from mediapipe.tasks.python import vision
        from tsl.features.normalize import normalize_sequence

        if not os.path.isfile(video_path):
            for seg_id, start_ms, end_ms, npy_abs, npy_rel in todo:
                results.append({"seg_id": seg_id, "npy_path": npy_rel,
                                 "status": "failed", "error": "video not found"})
            return results

        options = vision.HolisticLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.IMAGE,
            min_face_detection_confidence=0.5,
            min_pose_detection_confidence=0.5,
            min_hand_landmarks_confidence=0.5,
        )

        # Serialize model loading — TFLite initialization is not thread-safe
        with _model_init_lock:
            landmarker_ctx = vision.HolisticLandmarker.create_from_options(options)

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        with landmarker_ctx as landmarker:
            for seg_id, start_ms, end_ms, npy_abs, npy_rel in todo:
                start_f = max(0, int(start_ms / 1000.0 * fps))
                end_f   = min(total, int(end_ms   / 1000.0 * fps))

                if end_f <= start_f:
                    results.append({"seg_id": seg_id, "npy_path": npy_rel,
                                    "status": "failed", "error": "empty_segment"})
                    continue

                cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
                raw_frames = []
                for _ in range(end_f - start_f):
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame_rgb = np.ascontiguousarray(frame[:, :, ::-1])
                    result = landmarker.detect(
                        mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
                    )
                    raw_frames.append(_result_to_frame(result))

                if not raw_frames:
                    results.append({"seg_id": seg_id, "npy_path": npy_rel,
                                    "status": "failed", "error": "no_frames"})
                    continue

                seq  = np.stack(raw_frames, axis=0)   # (T, 543, 3)
                feat = normalize_sequence(seq)          # (T, 312)
                os.makedirs(os.path.dirname(npy_abs), exist_ok=True)
                np.save(npy_abs, feat)
                results.append({"seg_id": seg_id, "npy_path": npy_rel,
                                 "status": "extracted"})
        cap.release()

    except Exception as e:
        # Mark all remaining todo as failed
        done_ids = {r["seg_id"] for r in results}
        for seg_id, start_ms, end_ms, npy_abs, npy_rel in todo:
            if seg_id not in done_ids:
                results.append({"seg_id": seg_id, "npy_path": npy_rel,
                                 "status": "failed", "error": str(e)})

    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Re-extract YouTube-SL-25 Thai landmarks to 312-dim."
    )
    parser.add_argument("--src-dir",  default="data/youtube_sl25_thai",
                        help="Source directory with manifest.csv and videos/")
    parser.add_argument("--out-dir",  default="data/youtube_sl25_thai_v3",
                        help="Output directory for v3 landmarks + manifest")
    parser.add_argument("--workers",  type=int, default=2,
                        help="Parallel extraction workers (one video per worker)")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--dry-run",  action="store_true")
    args = parser.parse_args(argv)

    src_dir = os.path.abspath(args.src_dir)
    out_dir = os.path.abspath(args.out_dir)
    lm_dir  = os.path.join(out_dir, "landmarks")
    manifest_path = os.path.join(out_dir, "manifest.csv")
    manifest_src  = os.path.join(src_dir, "manifest.csv")

    if not os.path.isfile(manifest_src):
        print(f"ERROR: manifest not found: {manifest_src}", file=sys.stderr)
        return 1

    model_path = args.model_path or _find_model(src_dir)
    src_path   = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))

    df = pd.read_csv(manifest_src)
    print(f"Loaded {len(df)} segments from {manifest_src}")

    videos_dir = os.path.join(src_dir, "videos")
    video_map: dict[str, str] = {}
    for fn in os.listdir(videos_dir):
        if fn.endswith(".mp4"):
            video_map[fn[:-4]] = os.path.join(videos_dir, fn)
    print(f"Found {len(video_map)} source mp4s")

    # Group segments by video_id
    by_video: dict[str, list] = defaultdict(list)
    skipped_no_video = 0
    for _, row in df.iterrows():
        vid_id = str(row["video_id"])
        if vid_id not in video_map:
            skipped_no_video += 1
            continue
        seg_id   = str(row["segment_id"])
        start_ms = float(row["start_ms"])
        end_ms   = float(row["end_ms"])
        npy_rel  = os.path.join("landmarks", f"{seg_id}.npy")
        npy_abs  = os.path.join(out_dir, npy_rel)
        by_video[vid_id].append((seg_id, start_ms, end_ms, npy_abs, npy_rel))

    print(f"Videos to process: {len(by_video)}  (skipped_no_video: {skipped_no_video})")

    if args.dry_run:
        total_segs = sum(len(v) for v in by_video.values())
        already    = sum(1 for segs in by_video.values()
                         for (_, _, _, na, _) in segs if os.path.isfile(na))
        print(f"DRY RUN: {total_segs} segments total, {already} already done, "
              f"{total_segs - already} to extract across {len(by_video)} videos")
        return 0

    os.makedirs(lm_dir, exist_ok=True)

    # Build work list: one item per video
    work_items = [
        (vid_id, video_map[vid_id], segs, lm_dir, model_path, src_path)
        for vid_id, segs in by_video.items()
    ]

    # Pre-warm heavy imports in main thread so threads don't race on module init
    root_path = os.path.dirname(src_path)
    for p in (src_path, root_path):
        if p not in sys.path:
            sys.path.insert(0, p)
    import cv2 as _cv2  # noqa: F401
    import mediapipe as _mp  # noqa: F401
    from mediapipe.tasks import python as _mp_tasks  # noqa: F401
    from mediapipe.tasks.python import vision as _vision  # noqa: F401
    import tsl.features.normalize  # noqa: F401

    n_workers = min(args.workers, len(work_items), os.cpu_count() or 4)
    print(f"Extracting {len(work_items)} videos with {n_workers} threads …")

    all_results: list[dict] = []
    _lock = threading.Lock()
    _counter = [0]

    def _run_and_collect(w):
        rs = _process_video(w)
        with _lock:
            all_results.extend(rs)
            _counter[0] += 1
            i = _counter[0]
            ex = sum(1 for r in all_results if r["status"] == "extracted")
            sk = sum(1 for r in all_results if r["status"] == "skipped")
            fa = sum(1 for r in all_results if r["status"] == "failed")
            if i % max(1, len(work_items) // 20) == 0 or i == len(work_items):
                print(f"  [{i}/{len(work_items)}] extracted={ex} skipped={sk} failed={fa}")

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = [pool.submit(_run_and_collect, w) for w in work_items]
        for f in as_completed(futures):
            f.result()  # re-raise any exception

    from collections import Counter
    counts = Counter(r["status"] for r in all_results)
    print(f"\nResults: {dict(counts)}")

    good_ids = {r["seg_id"]: r["npy_path"] for r in all_results
                if r["status"] in ("extracted", "skipped")}

    out_rows = []
    for _, row in df.iterrows():
        seg_id = str(row["segment_id"])
        if seg_id not in good_ids:
            continue
        text = str(row.get("text", "")).strip()
        if not text:
            continue
        out_rows.append({
            "segment_id":             seg_id,
            "npy_path":               good_ids[seg_id],
            "text":                   text,
            "video_id":               str(row.get("video_id", "")),
            "start_ms":               row.get("start_ms", 0),
            "end_ms":                 row.get("end_ms", 0),
            "split":                  str(row.get("split", "train")),
            "feature_layout_version": "v3-312",
        })

    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "segment_id", "npy_path", "text", "video_id",
            "start_ms", "end_ms", "split", "feature_layout_version",
        ])
        writer.writeheader()
        writer.writerows(out_rows)

    print(f"\nWrote {len(out_rows)} rows → {manifest_path}")
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
