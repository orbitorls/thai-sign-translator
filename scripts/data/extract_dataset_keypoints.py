"""Extract keypoints from an arbitrary CSV of video segments.

The input contract is intentionally simple so the Rust wrapper can validate it:

    segment_id,video_path,text,start_ms,end_ms,split[,video_id]

The script writes the same dataset shape used by the existing SLT loaders:

    <out_dir>/
        landmarks/<segment_id>.npy
        manifest.csv

Each `.npy` file is `(T, 312)` float32 produced by the current normalization
contract in `tsl.features.normalize`.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from scripts._bootstrap import ensure_repo_paths

ensure_repo_paths()

from scripts.extract_thaisignvis_landmarks import _extract_segment, _find_model
from tsl.features.normalize import FEATURE_LAYOUT_VERSION, normalize_sequence

_REQUIRED_COLUMNS = ["segment_id", "video_path", "text", "start_ms", "end_ms", "split"]
_SOURCE_NAME = "custom"


def _resolve_path(value: str, base_dir: Path) -> str:
    path = Path(str(value))
    if path.is_absolute():
        return str(path)
    return str((base_dir / path).resolve())


def load_segments(input_csv: str, limit: int | None = None) -> list[dict]:
    csv_path = Path(input_csv)
    df = pd.read_csv(csv_path)
    missing = [name for name in _REQUIRED_COLUMNS if name not in df.columns]
    if missing:
        raise ValueError(f"input CSV is missing columns: {missing}")

    rows: list[dict] = []
    seen_ids: set[str] = set()
    for idx, row in df.iterrows():
        segment_id = _clean_value(row["segment_id"])
        if not segment_id:
            raise ValueError(f"row {idx + 2} has empty segment_id")
        if segment_id in seen_ids:
            raise ValueError(f"duplicate segment_id: {segment_id}")
        seen_ids.add(segment_id)

        text = _clean_value(row["text"])
        if not text:
            raise ValueError(f"row {idx + 2} has empty text")

        split = _clean_value(row["split"]).lower()
        if split not in {"train", "val", "test"}:
            raise ValueError(f"row {idx + 2} has invalid split {split!r}")

        start_ms = float(row["start_ms"])
        end_ms = float(row["end_ms"])
        if start_ms >= end_ms:
            raise ValueError(f"row {idx + 2} has start_ms >= end_ms")

        video_path = _resolve_path(str(row["video_path"]), csv_path.parent)
        if not Path(video_path).is_file():
            raise FileNotFoundError(f"row {idx + 2} video file not found: {video_path}")
        video_id = _clean_value(row["video_id"]) if "video_id" in df.columns else ""
        if not video_id:
            video_id = Path(video_path).stem

        rows.append(
            {
                "segment_id": segment_id,
                "video_path": video_path,
                "video_id": video_id,
                "text": text,
                "start_ms": start_ms,
                "end_ms": end_ms,
                "split": split,
            }
        )
        if limit is not None and len(rows) >= limit:
            break
    return rows


def _clean_value(value: object) -> str:
    text = str(value).strip()
    lowered = text.lower()
    if lowered in {"", "nan", "none"}:
        return ""
    return text


def _make_manifest_row(segment: dict, npy_rel: str, source: str) -> dict:
    return {
        "segment_id": segment["segment_id"],
        "npy_path": npy_rel,
        "text": segment["text"],
        "video_id": segment["video_id"],
        "start_ms": segment["start_ms"],
        "end_ms": segment["end_ms"],
        "split": segment["split"],
        "source": source,
        "feature_layout_version": FEATURE_LAYOUT_VERSION,
    }


def extract_segments(
    *,
    input_csv: str,
    out_dir: str,
    source: str = _SOURCE_NAME,
    fps: int = 0,
    seed: int = 42,
    limit: int | None = None,
    dry_run: bool = False,
) -> dict:
    segments = load_segments(input_csv, limit=limit)
    if not segments:
        raise ValueError("input CSV has no usable rows")

    if dry_run:
        return {"rows": len(segments), "out_dir": str(Path(out_dir)), "dry_run": True}

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    lm_dir = out_path / "landmarks"
    lm_dir.mkdir(parents=True, exist_ok=True)

    try:
        import cv2  # type: ignore
        import mediapipe as mp  # type: ignore
        from mediapipe.tasks import python as mp_tasks  # type: ignore
        from mediapipe.tasks.python import vision  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "mediapipe and opencv-python are required; install them before running extraction"
        ) from exc

    model_path = _find_model(str(out_path), str(Path(input_csv).parent))
    options = vision.HolisticLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.IMAGE,
        min_face_detection_confidence=0.5,
        min_pose_detection_confidence=0.5,
        min_hand_landmarks_confidence=0.5,
    )

    manifest_rows: list[dict] = []
    skipped = 0
    with vision.HolisticLandmarker.create_from_options(options) as landmarker:
        for i, segment in enumerate(segments):
            npy_rel = f"landmarks/{segment['segment_id']}.npy"
            npy_abs = out_path / npy_rel
            if npy_abs.is_file():
                manifest_rows.append(_make_manifest_row(segment, npy_rel, source))
                continue
            if not Path(segment["video_path"]).is_file():
                skipped += 1
                continue

            raw = _extract_segment(
                segment["video_path"],
                segment["start_ms"],
                segment["end_ms"],
                fps,
                mp,
                landmarker,
            )
            if raw.shape[0] == 0:
                skipped += 1
                continue

            feat = normalize_sequence(raw)
            np.save(npy_abs, feat)
            manifest_rows.append(_make_manifest_row(segment, npy_rel, source))
            if (i + 1) % 50 == 0:
                print(f"  [{i + 1}/{len(segments)}] extracted ...")

    manifest_path = out_path / "manifest.csv"
    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False, encoding="utf-8")
    return {
        "rows": len(manifest_rows),
        "skipped": skipped,
        "manifest_path": str(manifest_path),
        "out_dir": str(out_path),
        "dry_run": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Extract keypoints from a segment CSV.")
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--source", default=_SOURCE_NAME)
    parser.add_argument("--fps", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    result = extract_segments(
        input_csv=args.input_csv,
        out_dir=args.out_dir,
        source=args.source,
        fps=args.fps,
        seed=args.seed,
        limit=args.limit,
        dry_run=args.dry_run,
    )
    if args.dry_run:
        print(f"validated {result['rows']} rows from {args.input_csv}")
        return 0

    print(f"wrote {result['rows']} rows -> {result['manifest_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
