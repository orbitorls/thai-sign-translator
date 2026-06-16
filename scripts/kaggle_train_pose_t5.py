"""Kaggle notebook driver for PoseToTextT5 training.

Run from the repo root:
    python scripts/kaggle_train_pose_t5.py

Usage in a Kaggle notebook cell:
    !python scripts/kaggle_train_pose_t5.py

Before running on Kaggle:
----------------------------------------------------------------------
1. Attach your data dataset(s) as Kaggle input datasets.
   Each dataset must contain a ``manifest.csv`` at its root.
   Typical paths after attaching:
       /kaggle/input/tsl51-v3/
       /kaggle/input/yt-sl25-thai/
   Set DATA_ROOTS below to a comma-separated list of these paths.

2. To resume from a prior session's output, attach the previous run's
   output dataset (e.g. "my-ckpt") as an input dataset.
   It will appear at:
       /kaggle/input/my-ckpt/
   The script passes ``--resume auto`` which will automatically find and
   load the latest checkpoint in ``--out-dir``. On resume, copy the
   prior checkpoints into the output dir first:
       !cp -r /kaggle/input/my-ckpt/*.pt /kaggle/working/pose_t5_v3/

3. Quota budgeting: Kaggle grants ~30h GPU/week (T4 × 2 or P100 × 1).
   Each session runs up to 12h; ``--max-runtime-min 690`` terminates
   cleanly 10 minutes before the kill signal and saves a checkpoint.
   With 30h/week you can fit 2–3 sessions per week.
----------------------------------------------------------------------

Effective batch size: --batch-size 4 × --grad-accum 4 = 16 samples/step
Source length cap:    --max-src-len 512 frames → 128 frames after ×4 downsample
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

# ---------------------------------------------------------------------------
# Kaggle-specific paths — edit these to match your attached datasets
# ---------------------------------------------------------------------------

# Comma-separated list of manifest directories.
# Default assumes a single dataset mounted at /kaggle/input/tsl-data/
DATA_ROOTS = os.environ.get(
    "TSL_DATA_ROOTS",
    "/kaggle/input/tsl-data",
)

# Output directory for checkpoints and final model artefacts.
# Kaggle persists /kaggle/working/ between cells in the same session.
OUT_DIR = os.environ.get(
    "TSL_OUT_DIR",
    "/kaggle/working/pose_t5_v3",
)


# ---------------------------------------------------------------------------
# Dependency installation (idempotent — only installs if missing)
# ---------------------------------------------------------------------------

def _ensure_dependencies() -> None:
    """Install extra packages that may not be present on the Kaggle image."""
    try:
        import transformers  # noqa: F401
        import sentencepiece  # noqa: F401
        import sacrebleu  # noqa: F401
    except ImportError:
        print("[kaggle] Installing extra dependencies …")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q",
             "transformers", "sentencepiece", "sacrebleu"],
        )


# ---------------------------------------------------------------------------
# PYTHONPATH setup
# ---------------------------------------------------------------------------

def _setup_pythonpath() -> None:
    """Ensure src/ is on sys.path so ``tsl.*`` imports resolve."""
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src_dir = os.path.join(repo_root, "src")
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    # Also make repo root available for config.py
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)


# ---------------------------------------------------------------------------
# Argument parser (thin wrapper — mirrors train_pose_t5._build_parser)
# ---------------------------------------------------------------------------

def _build_kaggle_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Kaggle driver for PoseToTextT5 training.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Data / output
    p.add_argument(
        "--data-roots",
        type=str,
        default=DATA_ROOTS,
        help=(
            "Comma-separated paths to directories containing manifest.csv. "
            "Override via the TSL_DATA_ROOTS env var or this flag."
        ),
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default=OUT_DIR,
        help="Output directory for checkpoints and final model.",
    )

    # Kaggle-tuned training hyper-parameters
    p.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Per-step batch size. With T4/P100, 4 fits in ~14 GB VRAM.",
    )
    p.add_argument(
        "--grad-accum",
        type=int,
        default=4,
        help="Gradient accumulation steps; effective batch = batch_size × grad_accum.",
    )
    p.add_argument(
        "--max-src-len",
        type=int,
        default=512,
        help=(
            "Maximum source sequence length in frames before downsampling. "
            "After ×4 downsample, caps at ~128 T5 tokens."
        ),
    )
    p.add_argument(
        "--downsample-factor",
        type=int,
        default=4,
        help="Temporal mean-pool factor applied to pose frames.",
    )

    # Resume / runtime
    p.add_argument(
        "--resume",
        type=str,
        default="auto",
        help=(
            "'auto' finds the latest checkpoint in --out-dir. "
            "Pass a .pt path to resume from a specific file."
        ),
    )
    p.add_argument(
        "--max-runtime-min",
        type=int,
        default=690,
        help=(
            "Self-terminate and save a checkpoint after this many minutes. "
            "690 min = 11h 30m, giving 30 min buffer before Kaggle's 12h kill."
        ),
    )

    # AMP
    p.add_argument(
        "--amp",
        type=str,
        default="auto",
        choices=["auto", "true", "false"],
        help="Automatic Mixed Precision: 'auto' enables on CUDA, skips on CPU.",
    )

    # Pass-through args (rarely changed on Kaggle)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--num-encoder-layers", type=int, default=2)
    p.add_argument("--keep-checkpoints", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval-steps", type=int, default=200)
    p.add_argument(
        "--base-model",
        type=str,
        default="google/mt5-small",
        help=(
            "HuggingFace model name or local path for mT5. "
            "On Kaggle offline sessions, attach the mT5 weights as a dataset and "
            "point this to the local path, e.g. /kaggle/input/mt5-small/"
        ),
    )
    return p


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    _ensure_dependencies()
    _setup_pythonpath()

    # Import after PYTHONPATH is configured
    from tsl.train.train_pose_t5 import main as train_main  # noqa: PLC0415

    kaggle_args = _build_kaggle_parser().parse_args()
    print("[kaggle] Training with settings:")
    for k, v in vars(kaggle_args).items():
        print(f"  {k}: {v}")

    metrics = train_main(kaggle_args)
    print("[kaggle] Training finished. Final metrics:")
    print(f"  global_step:    {metrics.get('global_step')}")
    print(f"  stopped_reason: {metrics.get('stopped_reason')}")


if __name__ == "__main__":
    main()
