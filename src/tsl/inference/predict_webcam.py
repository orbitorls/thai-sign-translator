"""Capture a short webcam clip, then reuse the video inference path."""
from __future__ import annotations

import argparse
import os
import tempfile

from tsl.inference.predict_video import main as predict_video_main


def capture_webcam_clip(
    out_path: str,
    *,
    camera_index: int = 0,
    max_frames: int = 180,
) -> dict[str, int | str]:
    import cv2

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open webcam index {camera_index}")
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    writer = cv2.VideoWriter(
        out_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (width, height),
    )
    frames = 0
    try:
        while frames < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            writer.write(frame)
            frames += 1
    finally:
        cap.release()
        writer.release()
    if frames == 0:
        raise RuntimeError("webcam produced 0 frames")
    return {"video_path": out_path, "num_frames": frames, "fps": fps}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Capture a short webcam clip and run sentence inference.")
    parser.add_argument("--model-dir", default="")
    parser.add_argument("--config", default="")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=180)
    parser.add_argument("--out-video", default="")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if bool(args.model_dir) == bool(args.config):
        raise SystemExit("pass exactly one of --model-dir or --config")

    temp_path = ""
    try:
        if args.out_video:
            video_path = args.out_video
        else:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as handle:
                temp_path = handle.name
            video_path = temp_path
        capture_webcam_clip(
            video_path,
            camera_index=args.camera_index,
            max_frames=args.max_frames,
        )
        common = ["--model-dir", args.model_dir] if args.model_dir else ["--config", args.config]
        return predict_video_main([*common, "--video-path", video_path])
    finally:
        if temp_path:
            try:
                os.unlink(temp_path)
            except OSError:
                pass


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
