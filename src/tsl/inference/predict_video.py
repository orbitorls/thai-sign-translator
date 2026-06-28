"""CLI inference path for PoseT5 video inputs."""
from __future__ import annotations

import argparse
import json
import time

from tsl.features.normalize import normalize_sequence
from tsl.inference.video_pipeline import _extract_from_video
from tsl.models.bundle import ModelBundle


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Predict Thai text from a video file.")
    parser.add_argument("--video-path", required=True)
    parser.add_argument("--model-dir", default="")
    parser.add_argument("--config", default="")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--low-confidence-threshold", type=float, default=0.8)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if bool(args.model_dir) == bool(args.config):
        raise SystemExit("pass exactly one of --model-dir or --config")

    bundle = (
        ModelBundle.from_dir(args.model_dir, device=args.device)
        if args.model_dir
        else ModelBundle.from_config(args.config, device=args.device)
    )
    started = time.perf_counter()
    raw = _extract_from_video(args.video_path)
    if raw.shape[0] == 0:
        raise SystemExit(f"video {args.video_path!r} produced 0 frames")
    features = normalize_sequence(raw)
    payload = bundle.predict(
        features,
        low_confidence_threshold=args.low_confidence_threshold,
    )
    payload["num_frames"] = int(features.shape[0])
    payload["latency_ms"] = round((time.perf_counter() - started) * 1000.0, 3)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
