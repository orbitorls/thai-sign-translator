"""Export a PoseToTextT5 checkpoint to a runtime-ready directory."""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
from transformers import AutoTokenizer


_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from scripts._bootstrap import ensure_repo_paths

ensure_repo_paths()

from tsl.models.pose_t5 import PoseToTextT5
from tsl.train.checkpointing import find_best_checkpoint


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export a PoseToTextT5 checkpoint to a runtime-ready directory.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--train-dir", required=True, help="Directory containing ckpt_step*.pt files.")
    parser.add_argument("--export-dir", required=True, help="Directory to write the exported model.")
    parser.add_argument(
        "--checkpoint",
        default="best",
        help="'best' to use best_checkpoint.txt / scan by val_chrf, or a checkpoint filename/path.",
    )
    parser.add_argument("--base-model", default="google/mt5-small")
    parser.add_argument("--input-dim", type=int, default=312)
    parser.add_argument("--num-encoder-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--downsample-factor", type=int, default=4)
    parser.add_argument("--report-json", default=None)
    return parser


def _resolve_checkpoint(train_dir: Path, checkpoint_arg: str) -> Path:
    if checkpoint_arg == "best_state":
        best_state = train_dir / "best_model_state.pt"
        if best_state.is_file():
            return best_state
        raise FileNotFoundError(f"best_model_state.pt not found under {train_dir!s}")

    if checkpoint_arg == "best":
        best_state = train_dir / "best_model_state.pt"
        if best_state.is_file():
            return best_state
        best_ref = train_dir / "best_checkpoint.txt"
        if best_ref.is_file():
            name = best_ref.read_text(encoding="utf-8").strip()
            if name:
                candidate = train_dir / name
                if candidate.is_file():
                    return candidate
        best = find_best_checkpoint(train_dir, metric="val_chrf")
        if best:
            return Path(best)
        raise FileNotFoundError(f"no best checkpoint found under {train_dir!s}")

    requested = Path(checkpoint_arg)
    if requested.is_file():
        return requested
    candidate = train_dir / checkpoint_arg
    if candidate.is_file():
        return candidate
    raise FileNotFoundError(f"checkpoint not found: {checkpoint_arg!r}")


def _resolve_local_model_source(train_dir: Path, base_model: str) -> str | None:
    train_dir_config = train_dir / "config.json"
    if train_dir_config.is_file():
        return str(train_dir)
    if os.path.isdir(base_model):
        return base_model
    return None


def _resolve_tokenizer_source(train_dir: Path, base_model: str) -> str:
    for candidate in (train_dir, Path(base_model)):
        if candidate.is_dir() and (candidate / "tokenizer_config.json").is_file():
            return str(candidate)
    return base_model


def _export_checkpoint(args: argparse.Namespace) -> dict:
    train_dir = Path(args.train_dir).resolve()
    export_dir = Path(args.export_dir).resolve()
    checkpoint_path = _resolve_checkpoint(train_dir, args.checkpoint)
    local_model_source = _resolve_local_model_source(train_dir, args.base_model)
    tokenizer_source = _resolve_tokenizer_source(train_dir, args.base_model)

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = PoseToTextT5(
        input_dim=args.input_dim,
        num_encoder_layers=args.num_encoder_layers,
        encoder_dropout=args.dropout,
        downsample_factor=args.downsample_factor,
        base_model_name=args.base_model,
        local_model_path=local_model_source,
    )
    model.load_state_dict(payload["model_state_dict"])
    model.eval()

    export_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(export_dir))
    AutoTokenizer.from_pretrained(tokenizer_source).save_pretrained(str(export_dir))

    report = {
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_name": checkpoint_path.name,
        "checkpoint_type": "best_model_state" if checkpoint_path.name == "best_model_state.pt" else "full_checkpoint",
        "checkpoint_step": payload.get("step"),
        "checkpoint_epoch": payload.get("epoch"),
        "checkpoint_metrics": payload.get("metrics", {}),
        "train_dir": str(train_dir),
        "export_dir": str(export_dir),
        "base_model": args.base_model,
        "local_model_source": local_model_source,
        "tokenizer_source": tokenizer_source,
        "exported_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(export_dir / "runtime_metadata.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    if args.report_json:
        with open(args.report_json, "w", encoding="utf-8") as fh:
            json.dump(report, fh, ensure_ascii=False, indent=2)
            fh.write("\n")
    return report


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    report = _export_checkpoint(args)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
