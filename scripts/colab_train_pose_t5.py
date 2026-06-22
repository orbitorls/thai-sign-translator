"""Google Colab CLI training entrypoint for PoseToTextT5.

This script is intended to be executed remotely via:
    colab exec -s <session> -f scripts/colab_train_pose_t5.py

Because ``colab exec -f`` transmits only this file, the launcher uploads a
lightweight repo zip to ``/content/thai-sign-code.zip`` first. This script
unpacks that archive, mounts Google Drive storage, resolves dataset/checkpoint
paths, and then calls ``tsl.train.train_pose_t5.main``.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path


DEFAULT_DRIVE_ROOT = "/content/drive/MyDrive/thai-sign-translator"
DEFAULT_REPO_ROOT = "/content/thai-sign-translator"
DEFAULT_REPO_ZIP = "/content/thai-sign-code.zip"
DEFAULT_CONFIG_PATH = "/content/thai-sign-colab-config.json"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Remote Colab training entrypoint for PoseToTextT5.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--drive-root", default=DEFAULT_DRIVE_ROOT)
    parser.add_argument("--repo-root", default=DEFAULT_REPO_ROOT)
    parser.add_argument("--repo-zip", default=DEFAULT_REPO_ZIP)
    parser.add_argument("--config-path", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--base-model", default="google/mt5-small")
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--data-roots", default="")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--max-src-len", type=int, default=512)
    parser.add_argument("--downsample-factor", type=int, default=4)
    parser.add_argument("--num-encoder-layers", type=int, default=2)
    parser.add_argument("--amp", default="auto", choices=["auto", "true", "false"])
    parser.add_argument("--resume", default="auto")
    parser.add_argument("--reset-progress-history", action="store_true")
    parser.add_argument("--max-runtime-min", type=int, default=690)
    parser.add_argument("--keep-checkpoints", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--checkpoint-steps", type=int, default=500)
    parser.add_argument("--early-stopping-patience", type=int, default=10)
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.0)
    parser.add_argument(
        "--early-stopping-metric",
        default="val_chrf",
        choices=["val_loss", "val_chrf"],
    )
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--required-sources", default="tsl51,thaisignvis")
    parser.add_argument("--manifest-quality-sources", default="")
    parser.add_argument(
        "--fail-on-manifest-quality",
        default="true",
        choices=["true", "false"],
    )
    parser.add_argument(
        "--allow-noop-resume",
        default="false",
        choices=["true", "false"],
    )
    return parser


def _load_config_overrides(config_path: str) -> dict:
    path = Path(config_path)
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8-sig") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a JSON object: {config_path}")
    return data


def _apply_overrides(args: argparse.Namespace, overrides: dict) -> argparse.Namespace:
    for key, value in overrides.items():
        if hasattr(args, key):
            setattr(args, key, value)
    return args


def _ensure_repo_available(repo_root: str, repo_zip: str) -> Path:
    repo_root_path = Path(repo_root)
    train_script = repo_root_path / "src" / "tsl" / "train" / "train_pose_t5.py"
    if train_script.is_file():
        return repo_root_path
    zip_path = Path(repo_zip)
    if not zip_path.is_file():
        raise FileNotFoundError(
            f"Repo archive not found at {repo_zip}; upload it before running training."
        )
    repo_root_path.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(repo_root_path)
    if not train_script.is_file():
        raise FileNotFoundError(
            f"Repo archive extracted to {repo_root}, but train_pose_t5.py is still missing."
        )
    return repo_root_path


def _setup_pythonpath(repo_root: Path) -> None:
    src_dir = repo_root / "src"
    for path in (str(src_dir), str(repo_root)):
        if path not in sys.path:
            sys.path.insert(0, path)


def _install_dependencies(repo_root: Path) -> None:
    requirements = repo_root / "requirements.txt"
    if requirements.is_file():
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-r", str(requirements)])
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "transformers>=4.40,<5",
            "sentencepiece>=0.2.0",
            "sacrebleu>=2.4.0",
        ]
    )


def _resolve_data_roots(drive_root: str) -> str:
    roots: list[str] = []
    for subdir in (
        "data/mixed_all_train_v6",
        "data/tsl51_v3",
        "data/thaisignvis_v3_probe",
        "data/youtube_sl25_thai_v3",
    ):
        candidate = Path(drive_root) / subdir
        if (candidate / "manifest.csv").is_file():
            roots.append(str(candidate))
    return ",".join(roots)


def _resolve_out_dir(args: argparse.Namespace) -> str:
    if args.out_dir:
        return args.out_dir
    return str(Path(args.drive_root) / "checkpoints" / "pose_t5_mixed_all_v6_colab")


def _print_runtime_banner(args: argparse.Namespace) -> None:
    import torch

    print(f"[colab] drive_root={args.drive_root}")
    print(f"[colab] repo_root={args.repo_root}")
    print(f"[colab] data_roots={args.data_roots}")
    print(f"[colab] out_dir={args.out_dir}")
    print(f"[colab] CUDA={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"[colab] GPU={torch.cuda.get_device_name(0)}")
        print(f"[colab] VRAM_GB={props.total_memory / 1024**3:.1f}")
    else:
        print("[colab] WARNING: CUDA is not available; training will be slow.")


def main(argv: list[str] | None = None) -> dict:
    args = _build_parser().parse_args(argv)
    overrides = _load_config_overrides(args.config_path)
    args = _apply_overrides(args, overrides)
    _ensure_repo_available(args.repo_root, args.repo_zip)
    _setup_pythonpath(Path(args.repo_root))
    _install_dependencies(Path(args.repo_root))

    if not args.data_roots:
        if not Path("/content/drive").exists():
            raise RuntimeError(
                "Google Drive is not mounted and --data-roots was not provided. "
                "Run `colab drivemount` first or pass explicit dataset paths."
            )
        args.data_roots = _resolve_data_roots(args.drive_root)
    if not args.data_roots:
        raise RuntimeError(
            f"No dataset manifests found under {args.drive_root}. Expected "
            "`data/mixed_all_train_v6/manifest.csv` or the component manifests under "
            "`data/tsl51_v3`, `data/thaisignvis_v3_probe`, and `data/youtube_sl25_thai_v3`."
        )

    if not args.out_dir:
        if not Path("/content/drive").exists():
            raise RuntimeError(
                "Google Drive is not mounted and --out-dir was not provided. "
                "Run `colab drivemount` first or pass an explicit checkpoint directory."
            )
        args.out_dir = _resolve_out_dir(args)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    _print_runtime_banner(args)

    from tsl.train.train_pose_t5 import main as train_main

    return train_main(args)


if __name__ == "__main__":
    main()
