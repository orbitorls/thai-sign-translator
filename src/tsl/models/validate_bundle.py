"""CLI for validating PoseT5 runtime bundles."""
from __future__ import annotations

import argparse
import json

from tsl.models.bundle import resolve_model_dir_from_config, validate_model_dir


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate a PoseT5 runtime bundle.")
    parser.add_argument("--model-dir", default="")
    parser.add_argument("--config", default="")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if bool(args.model_dir) == bool(args.config):
        raise SystemExit("pass exactly one of --model-dir or --config")

    model_dir = args.model_dir or str(resolve_model_dir_from_config(args.config))
    metadata = validate_model_dir(model_dir)
    print(json.dumps(metadata.__dict__, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
