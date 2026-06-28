"""CLI inference path for PoseT5 feature arrays."""
from __future__ import annotations

import argparse
import json
import time

from tsl.data.unified import load_features
from tsl.models.bundle import ModelBundle


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Predict Thai text from a .npy feature file.")
    parser.add_argument("--feature-path", required=True)
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
    features = load_features(args.feature_path)
    started = time.perf_counter()
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
