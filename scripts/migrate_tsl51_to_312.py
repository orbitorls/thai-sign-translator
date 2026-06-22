"""Re-extract TSL-51 landmarks from local mp4 videos to 312-dim features.

Reads ``data/tsl51/metadata/sentence_metadata.csv``, extracts MediaPipe Holistic
(543 landmarks) per video, normalizes to (T, 312) via ``normalize_sequence()``,
and saves as ``data/tsl51_v3/landmarks/<video_id>.npy``.

Idempotent/resumable: skips videos whose ``.npy`` already exists.

Degraded fallback
-----------------
If the mp4 is missing, attempts to load the existing 162-dim CSV from
``data/tsl51/landmarks/user_sentence/<video_id>.csv`` and zero-pads to 312
(tag: ``feature_layout_version=v3-312-degraded``).

Usage
-----
    python scripts/migrate_tsl51_to_312.py \\
        --tsl51-root data/tsl51 \\
        --out-dir    data/tsl51_v3 \\
        [--workers 4] \\
        [--model-path path/to/holistic_landmarker.task] \\
        [--dry-run]
"""
from __future__ import annotations

import argparse
import csv
import multiprocessing
import os
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_HOLISTIC_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/holistic_landmarker/"
    "holistic_landmarker/float16/latest/holistic_landmarker.task"
)
_HOLISTIC_MODEL_FILENAME = "holistic_landmarker.task"

_MANIFEST_COLUMNS = [
    "segment_id",
    "npy_path",
    "text",
    "video_id",
    "split",
    "feature_layout_version",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ensure_holistic_model(cache_dir: str, model_path_override: str | None) -> str:
    """Return path to holistic .task model, downloading if necessary."""
    if model_path_override and os.path.isfile(model_path_override):
        return model_path_override

    # Search order: cache_dir, then current directory
    for search_dir in [cache_dir, "."]:
        candidate = os.path.join(search_dir, _HOLISTIC_MODEL_FILENAME)
        if os.path.isfile(candidate):
            return candidate

    # Not found — download to cache_dir
    os.makedirs(cache_dir, exist_ok=True)
    dest = os.path.join(cache_dir, _HOLISTIC_MODEL_FILENAME)
    print(f"  Downloading holistic landmarker model (~50 MB) to {dest} …")
    urllib.request.urlretrieve(_HOLISTIC_MODEL_URL, dest)
    print(f"  Saved to {dest}")
    return dest


def _result_to_frame(result) -> np.ndarray:
    """Convert HolisticLandmarkerResult (Tasks API) to (543, 3) float32.

    Layout: face(468) | left_hand(21) | pose(33) | right_hand(21)
    Missing components filled with NaN.
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


def _extract_video_npy(
    video_path: str,
    landmarker,
) -> np.ndarray | None:
    """Extract (T, 543, 3) from a full video file using MediaPipe Holistic."""
    import cv2
    import mediapipe as mp

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    raw_frames = []
    while True:
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
    return np.stack(raw_frames, axis=0)  # (T, 543, 3)


def _load_degraded_csv(csv_path: str) -> np.ndarray | None:
    """Load existing 162-dim CSV and zero-pad to 312."""
    try:
        arr = pd.read_csv(csv_path, header=None).values.astype(np.float32)
        if arr.ndim != 2 or arr.shape[1] == 0:
            return None
        # arr is (T, 162) — zero-extend to 312
        T, D = arr.shape
        if D < 312:
            padding = np.zeros((T, 312 - D), dtype=np.float32)
            arr = np.concatenate([arr, padding], axis=1)
        elif D > 312:
            arr = arr[:, :312]
        return arr
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Per-video worker (runs in subprocess for parallel extraction)
# ---------------------------------------------------------------------------

def _worker(args_tuple: tuple) -> dict:
    """Extract one video and return a result dict.

    Runs inside a subprocess — all heavy imports happen locally.
    """
    (
        video_id,
        abs_video_path,
        abs_csv_path,
        npy_abs,
        npy_rel,
        src_path,
        model_path,
    ) = args_tuple

    result = {
        "video_id": video_id,
        "status": "failed",          # extracted | skipped | degraded | failed
        "npy_path": npy_rel,
        "feature_layout_version": "v3-312",
    }

    # Already done — skip
    if os.path.isfile(npy_abs):
        result["status"] = "skipped"
        return result

    # Insert src and project root (for config.py) so tsl.features.normalize is importable
    root_path = os.path.dirname(src_path)
    for p in (src_path, root_path):
        if p not in sys.path:
            sys.path.insert(0, p)

    # Try full MediaPipe extraction from mp4
    if abs_video_path and os.path.isfile(abs_video_path):
        try:
            import mediapipe as mp
            from mediapipe.tasks import python as mp_tasks
            from mediapipe.tasks.python import vision

            options = vision.HolisticLandmarkerOptions(
                base_options=mp_tasks.BaseOptions(model_asset_path=model_path),
                running_mode=vision.RunningMode.IMAGE,
                min_face_detection_confidence=0.5,
                min_pose_detection_confidence=0.5,
                min_hand_landmarks_confidence=0.5,
            )
            with vision.HolisticLandmarker.create_from_options(options) as landmarker:
                seq = _extract_video_npy(abs_video_path, landmarker)

            if seq is not None and seq.shape[0] > 0:
                from tsl.features.normalize import normalize_sequence
                feat = normalize_sequence(seq)            # (T, 312)
                os.makedirs(os.path.dirname(npy_abs), exist_ok=True)
                np.save(npy_abs, feat)
                result["status"] = "extracted"
                return result
        except Exception as e:
            result["error"] = str(e)
            # Fall through to degraded path

    # Degraded fallback: existing 162-dim CSV
    if abs_csv_path and os.path.isfile(abs_csv_path):
        feat = _load_degraded_csv(abs_csv_path)
        if feat is not None and feat.shape[0] > 0:
            os.makedirs(os.path.dirname(npy_abs), exist_ok=True)
            np.save(npy_abs, feat)
            result["status"] = "degraded"
            result["feature_layout_version"] = "v3-312-degraded"
            return result

    result["status"] = "failed"
    return result


# ---------------------------------------------------------------------------
# Split assignment (video-level, deterministic)
# ---------------------------------------------------------------------------

def _assign_splits_video_level(
    video_ids: list[str],
    val_frac: float = 0.1,
    seed: int = 42,
) -> dict[str, str]:
    """Return mapping video_id → 'train' or 'val'."""
    import random
    rng = random.Random(seed)
    ids = list(video_ids)
    rng.shuffle(ids)
    n_val = max(1, round(len(ids) * val_frac))
    split_map: dict[str, str] = {}
    for i, vid in enumerate(ids):
        split_map[vid] = "val" if i < n_val else "train"
    return split_map


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Migrate TSL-51 videos to 312-dim landmark .npy files."
    )
    parser.add_argument(
        "--tsl51-root",
        default="data/tsl51",
        help="Root of the TSL-51 dataset (default: data/tsl51).",
    )
    parser.add_argument(
        "--out-dir",
        default="data/tsl51_v3",
        help="Output directory for v3 landmarks + manifest (default: data/tsl51_v3).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel extraction processes (default: 1).",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="Path to holistic_landmarker.task; downloaded if missing.",
    )
    parser.add_argument(
        "--val-frac",
        type=float,
        default=0.1,
        help="Fraction of videos assigned to val split (default: 0.1).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for split assignment (default: 42).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without extracting anything.",
    )
    args = parser.parse_args(argv)

    # Resolve paths
    tsl51_root = os.path.abspath(args.tsl51_root)
    out_dir    = os.path.abspath(args.out_dir)
    lm_dir     = os.path.join(out_dir, "landmarks")
    manifest_path = os.path.join(out_dir, "manifest.csv")

    # Source paths
    metadata_csv = os.path.join(tsl51_root, "metadata", "sentence_metadata.csv")
    if not os.path.isfile(metadata_csv):
        print(f"ERROR: metadata CSV not found: {metadata_csv}", file=sys.stderr)
        return 1

    # src/ must be on path for tsl.* imports in worker
    src_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "src")
    )

    # Read metadata
    df = pd.read_csv(metadata_csv)
    print(f"Loaded metadata: {len(df)} rows from {metadata_csv}")

    # Resolve absolute video + CSV paths
    def _abs_video(row) -> str | None:
        vp = row.get("video_path", "")
        if not vp or pd.isna(vp):
            return None
        if os.path.isabs(vp):
            return vp
        return os.path.join(tsl51_root, vp)

    def _abs_csv(row) -> str | None:
        # Existing 162-dim landmark CSV path
        lp = row.get("landmark_path", "")
        if not lp or pd.isna(lp):
            # Infer: landmarks/user_sentence/<video_id>.csv
            vid = row.get("video_id", "")
            if vid:
                return os.path.join(tsl51_root, "landmarks", "user_sentence", f"{vid}.csv")
            return None
        if os.path.isabs(lp):
            return lp
        return os.path.join(tsl51_root, lp)

    rows_meta = df.to_dict("records")

    # Count what exists
    n_video_found   = sum(1 for r in rows_meta if _abs_video(r) and os.path.isfile(_abs_video(r)))
    n_csv_found     = sum(1 for r in rows_meta if _abs_csv(r) and os.path.isfile(_abs_csv(r)))
    n_npy_exist     = sum(
        1 for r in rows_meta
        if os.path.isfile(os.path.join(lm_dir, f"{r['video_id']}.npy"))
    )

    # Assign splits (video-level, deterministic)
    all_video_ids = [r["video_id"] for r in rows_meta]
    split_map = _assign_splits_video_level(
        all_video_ids, val_frac=args.val_frac, seed=args.seed
    )
    n_train = sum(1 for s in split_map.values() if s == "train")
    n_val   = sum(1 for s in split_map.values() if s == "val")

    print()
    print("=== TSL-51 → 312-dim Migration Summary ===")
    print(f"  Metadata rows    : {len(df)}")
    print(f"  Videos found     : {n_video_found} / {len(df)}")
    print(f"  162-dim CSVs     : {n_csv_found} / {len(df)}")
    print(f"  Already extracted: {n_npy_exist} / {len(df)}")
    print(f"  To extract       : {len(df) - n_npy_exist}")
    print(f"  Split (train/val): {n_train} / {n_val}  (seed={args.seed})")
    print(f"  Output dir       : {out_dir}")
    print(f"  Manifest         : {manifest_path}")
    print()

    if args.dry_run:
        print("[DRY RUN] No files will be written.")
        # Show per-row disposition
        print()
        print(f"{'video_id':<60} {'disposition':<20} {'split'}")
        print("-" * 90)
        for r in rows_meta:
            vid = r["video_id"]
            npy_abs = os.path.join(lm_dir, f"{vid}.npy")
            av = _abs_video(r)
            ac = _abs_csv(r)
            if os.path.isfile(npy_abs):
                disp = "skip (already done)"
            elif av and os.path.isfile(av):
                disp = "extract (mp4)"
            elif ac and os.path.isfile(ac):
                disp = "degraded (162-csv)"
            else:
                disp = "FAIL (no source)"
            split = split_map.get(vid, "train")
            print(f"  {vid[:58]:<58} {disp:<20} {split}")
        return 0

    # ---------------------------------------------------------------------------
    # Real extraction
    # ---------------------------------------------------------------------------
    os.makedirs(lm_dir, exist_ok=True)

    # Ensure model is available
    model_path = _ensure_holistic_model(out_dir, args.model_path)
    print(f"Using model: {model_path}")

    # Build worker arg tuples
    work_items = []
    for r in rows_meta:
        vid = r["video_id"]
        npy_rel = os.path.join("landmarks", f"{vid}.npy")
        npy_abs = os.path.join(out_dir, npy_rel)
        work_items.append((
            vid,
            _abs_video(r),
            _abs_csv(r),
            npy_abs,
            npy_rel,
            src_path,
            model_path,
        ))

    # Run extraction (parallel or serial)
    n_extracted = 0
    n_skipped   = 0
    n_degraded  = 0
    n_failed    = 0
    results_by_vid: dict[str, dict] = {}

    total = len(work_items)
    print(f"Extracting {total} videos (workers={args.workers}) …")

    if args.workers > 1:
        ctx = multiprocessing.get_context("spawn")
        with ctx.Pool(processes=args.workers) as pool:
            for idx, res in enumerate(pool.imap_unordered(_worker, work_items), 1):
                results_by_vid[res["video_id"]] = res
                _tally(res["status"], n_extracted, n_skipped, n_degraded, n_failed)
                n_extracted, n_skipped, n_degraded, n_failed = _update_counts(
                    res["status"], n_extracted, n_skipped, n_degraded, n_failed
                )
                if idx % 10 == 0 or idx == total:
                    print(
                        f"  [{idx}/{total}] extracted={n_extracted} "
                        f"skipped={n_skipped} degraded={n_degraded} failed={n_failed}"
                    )
    else:
        for idx, item in enumerate(work_items, 1):
            res = _worker(item)
            results_by_vid[res["video_id"]] = res
            n_extracted, n_skipped, n_degraded, n_failed = _update_counts(
                res["status"], n_extracted, n_skipped, n_degraded, n_failed
            )
            if idx % 10 == 0 or idx == total:
                print(
                    f"  [{idx}/{total}] extracted={n_extracted} "
                    f"skipped={n_skipped} degraded={n_degraded} failed={n_failed}"
                )

    # ---------------------------------------------------------------------------
    # Write manifest.csv
    # ---------------------------------------------------------------------------
    print(f"\nWriting manifest to {manifest_path} …")
    with open(manifest_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_MANIFEST_COLUMNS)
        writer.writeheader()
        for r in rows_meta:
            vid = r["video_id"]
            res = results_by_vid.get(vid, {})
            status = res.get("status", "failed")
            if status == "failed":
                continue  # omit failed from manifest
            npy_rel = res.get("npy_path", os.path.join("landmarks", f"{vid}.npy"))
            writer.writerow({
                "segment_id":             vid,
                "npy_path":               npy_rel,
                "text":                   r.get("sentence_clean", ""),
                "video_id":               vid,
                "split":                  split_map.get(vid, "train"),
                "feature_layout_version": res.get("feature_layout_version", "v3-312"),
            })

    print()
    print("=== Extraction Complete ===")
    print(f"  Extracted (full 312): {n_extracted}")
    print(f"  Skipped (existed)   : {n_skipped}")
    print(f"  Degraded (162→312)  : {n_degraded}")
    print(f"  Failed              : {n_failed}")
    print(f"  Manifest written to : {manifest_path}")
    return 0


def _tally(status: str, n_extracted, n_skipped, n_degraded, n_failed):
    """No-op placeholder kept for clarity."""
    pass


def _update_counts(
    status: str,
    n_extracted: int,
    n_skipped: int,
    n_degraded: int,
    n_failed: int,
) -> tuple[int, int, int, int]:
    if status == "extracted":
        n_extracted += 1
    elif status == "skipped":
        n_skipped += 1
    elif status == "degraded":
        n_degraded += 1
    else:
        n_failed += 1
    return n_extracted, n_skipped, n_degraded, n_failed


if __name__ == "__main__":
    sys.exit(main())
