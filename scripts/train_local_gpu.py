"""Launch PoseToTextT5 training optimized for local RTX 4060 8GB GPU.

Usage:
    python scripts/train_local_gpu.py

After migration completes (preferably via data/mixed_readiness_v3),
this script trains the model with settings aligned to the current managed
Colab regime, but adapted for RTX 4060 8GB:
  - AMP fp16
  - batch_size=4, grad_accum=8 → effective batch=32
  - eval every 100 steps
  - full checkpoint writes are throttled to every 5000 steps on local disk
  - checkpoints in checkpoints/pose_t5_rtx4060_resume_best/
  - seeds a fresh out_dir from kaggle_upload/thai-sign-ckpt when available
  - resumes from the latest full checkpoint and resets stale progress history
"""
import argparse
import os
import shutil
import sys
from pathlib import Path

import torch

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from scripts._bootstrap import ensure_repo_paths

ensure_repo_paths()

from tsl.train.checkpointing import find_best_checkpoint, find_latest_checkpoint  # noqa: E402
from tsl.train.train_pose_t5 import _build_parser, main  # noqa: E402


def _parse_args():
    p = argparse.ArgumentParser(
        description="Local GPU training launcher for PoseToTextT5.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--data-roots",
        default="data/mixed_readiness_v3",
        help="Comma-separated data directories (must have manifest.csv).",
    )
    p.add_argument(
        "--out-dir",
        default="checkpoints/pose_t5_rtx4060_mixed_readiness_v3",
        help="Checkpoint output directory.",
    )
    p.add_argument(
        "--epochs",
        type=int,
        default=300,
        help="Number of training epochs.",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Per-GPU batch size for RTX 4060 8GB.",
    )
    p.add_argument(
        "--grad-accum",
        type=int,
        default=8,
        help="Gradient accumulation steps (effective batch = batch_size × grad_accum).",
    )
    p.add_argument(
        "--lr",
        type=float,
        default=1e-5,
        help="Peak learning rate for AdamW.",
    )
    p.add_argument(
        "--dropout",
        type=float,
        default=0.4,
        help="Dropout applied inside the pose encoder Transformer layers.",
    )
    p.add_argument(
        "--weight-decay",
        type=float,
        default=0.1,
        help="AdamW weight decay coefficient.",
    )
    p.add_argument(
        "--max-src-len",
        type=int,
        default=512,
        help="Cap source sequence length after downsampling.",
    )
    p.add_argument(
        "--downsample-factor",
        type=int,
        default=4,
        help="Temporal mean-pooling factor (reduces 1000-frame clip to 250 tokens).",
    )
    p.add_argument(
        "--eval-steps",
        type=int,
        default=100,
        help="Evaluate every N optimizer steps.",
    )
    p.add_argument(
        "--checkpoint-steps",
        type=int,
        default=5000,
        help="Save checkpoints every N optimizer steps.",
    )
    p.add_argument(
        "--max-runtime-min",
        type=int,
        default=9999,
        help="Max runtime in minutes (9999 = no limit for local training).",
    )
    p.add_argument(
        "--amp",
        default="auto",
        choices=["auto", "true", "false"],
        help="AMP: auto detects CUDA and enables fp16 automatically.",
    )
    p.add_argument(
        "--base-model",
        default="google/mt5-small",
        help="HuggingFace model or local path for the mT5 decoder.",
    )
    p.add_argument("--resume", default="best_state")
    p.add_argument(
        "--preload-train-features",
        default="false",
        choices=["true", "false"],
        help="Pre-load all train features into RAM before training.",
    )
    p.add_argument(
        "--balance-sources",
        default="auto",
        choices=["auto", "true", "false"],
        help="Use inverse-frequency source sampling when multiple data roots are mixed.",
    )
    p.add_argument(
        "--seed-checkpoint-dir",
        default="kaggle_upload/thai-sign-ckpt",
        help="Seed a fresh out_dir from this directory when no local checkpoints exist.",
    )
    p.add_argument(
        "--reset-progress-history",
        action="store_true",
        default=True,
        help="Reset stale progress history when resuming from a seeded checkpoint.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-encoder-layers", type=int, default=2)
    p.add_argument("--keep-checkpoints", type=int, default=3)
    p.add_argument(
        "--required-sources",
        default="tsl51,thaisignvis",
        help="Comma-separated sources that must appear in both train and val splits.",
    )
    p.add_argument(
        "--fail-on-manifest-quality",
        default="true",
        choices=["true", "false"],
        help="Abort before training when readiness-oriented manifest checks fail.",
    )
    p.add_argument(
        "--allow-noop-resume",
        default="false",
        choices=["true", "false"],
        help="Allow resume flows that add zero new optimizer steps.",
    )
    p.add_argument(
        "--early-stopping-patience",
        type=int,
        default=12,
        help="Number of eval windows without tracked-metric improvement before stopping (0 disables).",
    )
    p.add_argument(
        "--early-stopping-min-delta",
        type=float,
        default=0.0,
        help="Minimum tracked-metric improvement required to reset early stopping patience.",
    )
    p.add_argument(
        "--early-stopping-metric",
        default="val_chrf",
        choices=["val_loss", "val_chrf"],
        help="Metric to track for early stopping and best-checkpoint resume.",
    )
    p.add_argument("--num-workers", type=int, default=0)
    return p.parse_args()


def _seed_out_dir_if_needed(
    out_dir: str,
    seed_checkpoint_dir: str,
    *,
    resume_mode: str = "auto",
) -> list[str]:
    out_dir_abs = os.path.abspath(out_dir)
    seed_dir_abs = os.path.abspath(seed_checkpoint_dir)
    os.makedirs(out_dir_abs, exist_ok=True)
    best_state_name = "best_model_state.pt"

    def _checkpoint_files(root: str) -> list[str]:
        return sorted(
            name for name in os.listdir(root)
            if name.startswith("ckpt_step") and name.endswith(".pt")
        )

    def _checkpoint_is_valid(path: str) -> bool:
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
        except Exception:
            return False
        return isinstance(payload, dict) and payload.get("step") is not None

    def _refresh_checkpoint_refs(root: str) -> None:
        latest = find_latest_checkpoint(root)
        best = find_best_checkpoint(root, metric="val_chrf")
        latest_ref = os.path.join(root, "latest_checkpoint.txt")
        best_ref = os.path.join(root, "best_checkpoint.txt")
        if latest:
            with open(latest_ref, "w", encoding="utf-8") as fh:
                fh.write(os.path.basename(latest))
        if best:
            with open(best_ref, "w", encoding="utf-8") as fh:
                fh.write(os.path.basename(best))

    existing_ckpts = _checkpoint_files(out_dir_abs)
    if existing_ckpts or os.path.isfile(os.path.join(out_dir_abs, best_state_name)):
        latest_name = existing_ckpts[-1] if existing_ckpts else ""
        if latest_name:
            latest_ref = os.path.join(out_dir_abs, "latest_checkpoint.txt")
            with open(latest_ref, "w", encoding="utf-8") as fh:
                fh.write(latest_name)
        return []
    if not os.path.isdir(seed_dir_abs):
        return []

    copied: list[str] = []
    names_to_copy: list[str] = []
    best_state_source = os.path.join(seed_dir_abs, best_state_name)
    prefer_best_state_only = (
        resume_mode == "best_state" and os.path.isfile(best_state_source)
    )
    if not prefer_best_state_only:
        for ref_name in ("best_checkpoint.txt", "latest_checkpoint.txt"):
            ref_path = os.path.join(seed_dir_abs, ref_name)
            if not os.path.isfile(ref_path):
                continue
            ref_value = open(ref_path, "r", encoding="utf-8").read().strip()
            if ref_value and ref_value not in names_to_copy:
                names_to_copy.append(ref_value)

        if not names_to_copy:
            names_to_copy = _checkpoint_files(seed_dir_abs)

    for name in names_to_copy:
        source = os.path.join(seed_dir_abs, name)
        target = os.path.join(out_dir_abs, name)
        if os.path.isfile(source) and _checkpoint_is_valid(source):
            shutil.copy2(source, target)
            if _checkpoint_is_valid(target):
                copied.append(name)
            else:
                os.remove(target)

    best_state_target = os.path.join(out_dir_abs, best_state_name)
    if os.path.isfile(best_state_source) and _checkpoint_is_valid(best_state_source):
        shutil.copy2(best_state_source, best_state_target)
        copied.append(best_state_name)

    for name in ("best_checkpoint.txt", "latest_checkpoint.txt"):
        source = os.path.join(seed_dir_abs, name)
        target = os.path.join(out_dir_abs, name)
        if os.path.isfile(source):
            shutil.copy2(source, target)
            copied.append(name)
    if copied:
        _refresh_checkpoint_refs(out_dir_abs)
    return copied


def _recover_or_cleanup_temp_checkpoints(out_dir: str) -> list[str]:
    out_path = Path(out_dir)
    if not out_path.is_dir():
        return []

    actions: list[str] = []
    for tmp_path in sorted(out_path.glob("*.tmp")):
        try:
            payload = torch.load(tmp_path, map_location="cpu", weights_only=False)
        except Exception:
            tmp_path.unlink(missing_ok=True)
            actions.append(f"removed corrupt temp checkpoint {tmp_path.name}")
            continue

        step = payload.get("step") if isinstance(payload, dict) else None
        if step is None:
            tmp_path.unlink(missing_ok=True)
            actions.append(f"removed unusable temp checkpoint {tmp_path.name}")
            continue

        recovered_path = out_path / f"ckpt_step{int(step):08d}.pt"
        if recovered_path.exists():
            tmp_path.unlink(missing_ok=True)
            actions.append(
                f"removed redundant temp checkpoint {tmp_path.name} (step {int(step)})"
            )
            continue

        os.replace(tmp_path, recovered_path)
        actions.append(
            f"recovered temp checkpoint {tmp_path.name} -> {recovered_path.name}"
        )

    return actions


if __name__ == "__main__":
    args = _parse_args()

    # Validate data roots before starting
    import torch
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)} "
              f"({torch.cuda.get_device_properties(0).total_memory // 1024**3}GB VRAM)")
    else:
        print("WARNING: CUDA not available, training on CPU (will be slow)")

    missing = [r for r in args.data_roots.split(",") if not os.path.isdir(r.strip())]
    if missing:
        print(f"\nERROR: These data directories don't exist yet: {missing}")
        print("Wait for migrations to complete:")
        print("  python scripts/migrate_tsl51_to_312.py   (TSL-51)")
        print("  python scripts/reextract_youtube_sl25_to_312.py   (YouTube)")
        sys.exit(1)

    recovered = _recover_or_cleanup_temp_checkpoints(args.out_dir)
    seeded = _seed_out_dir_if_needed(
        args.out_dir,
        args.seed_checkpoint_dir,
        resume_mode=args.resume,
    )

    print(f"\nStarting training:")
    print(f"  data_roots  = {args.data_roots}")
    print(f"  out_dir     = {args.out_dir}")
    print(f"  batch_size  = {args.batch_size} × grad_accum {args.grad_accum} = {args.batch_size * args.grad_accum} effective")
    print(f"  lr          = {args.lr}")
    print(f"  dropout     = {args.dropout}")
    print(f"  weight_decay= {args.weight_decay}")
    print(f"  epochs      = {args.epochs}")
    print(f"  early_stop  = {args.early_stopping_patience} (min_delta={args.early_stopping_min_delta})")
    print(f"  early_metric= {args.early_stopping_metric}")
    print(f"  checkpoint_steps = {args.checkpoint_steps}")
    print(f"  resume      = {args.resume}")
    print(f"  reset_hist  = {args.reset_progress_history}")
    print(f"  amp         = {args.amp}")
    print(f"  preload     = {args.preload_train_features}")
    print(f"  balance_src = {args.balance_sources}")
    print(f"  base_model  = {args.base_model}")
    for action in recovered:
        print(f"  recovery    = {action}")
    if seeded:
        print(f"  seeded_from = {args.seed_checkpoint_dir} ({', '.join(seeded)})")
    print()

    metrics = main(args)
    print("\nTraining complete:", metrics)
