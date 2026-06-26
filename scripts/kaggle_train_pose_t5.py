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
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from scripts._bootstrap import ensure_repo_paths

# ---------------------------------------------------------------------------
# Kaggle-specific paths — edit these to match your attached datasets
# ---------------------------------------------------------------------------

# Comma-separated list of manifest directories.
# Default assumes a single dataset mounted at /kaggle/input/tsl-data/
DATA_ROOTS = os.environ.get(
    "TSL_DATA_ROOTS",
    "/kaggle/input/thai-sign-mixed-all-v6-archived",
)

# Output directory for checkpoints and final model artefacts.
# Kaggle persists /kaggle/working/ between cells in the same session.
OUT_DIR = os.environ.get(
    "TSL_OUT_DIR",
    "/kaggle/working/pose_t5_mixed_all_v6",
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
    ensure_repo_paths()


def _resolve_bool_flag(value) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _parse_csv_list(value: str) -> list[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _stage_archived_dataset_root(data_root: str, *, work_root: str) -> str:
    source_root = Path(data_root).resolve()
    manifest_path = source_root / "manifest.csv"
    features_archive = source_root / "features.zip"
    if not manifest_path.is_file() or not features_archive.is_file():
        return str(source_root)

    stage_root = Path(work_root).resolve() / "staged_inputs" / source_root.name
    _reset_dir(stage_root)
    shutil.copy2(manifest_path, stage_root / "manifest.csv")
    quality_path = source_root / "manifest_quality.json"
    if quality_path.is_file():
        shutil.copy2(quality_path, stage_root / "manifest_quality.json")
    feature_root = stage_root / "features"
    feature_root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(features_archive, "r") as archive:
        archive.extractall(feature_root)
    return str(stage_root)


def _prepare_data_roots(data_roots: str, *, work_root: str = "/kaggle/working") -> str:
    prepared = []
    for data_root in _parse_csv_list(data_roots):
        prepared.append(_stage_archived_dataset_root(data_root, work_root=work_root))
    return ",".join(prepared)


def _write_json(path: str, payload: dict) -> None:
    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _merge_preflight_reports(raw_report: dict, staged_report: dict) -> dict:
    failures = []
    failures.extend(str(item) for item in raw_report.get("failures", []))
    failures.extend(str(item) for item in staged_report.get("failures", []))
    return {
        "passed": bool(raw_report.get("passed", False)) and bool(staged_report.get("passed", False)),
        "raw": raw_report,
        "staged": staged_report,
        "failures": failures,
    }


def _copy_seed_files(src_dir: Path, dst_dir: Path) -> None:
    if not src_dir.exists():
        return
    for item in src_dir.iterdir():
        if item.name == "_smoke" or not item.is_file():
            continue
        shutil.copy2(item, dst_dir / item.name)


def _run_smoke_training(kaggle_args: argparse.Namespace, *, train_main) -> dict | None:
    smoke_steps = int(getattr(kaggle_args, "smoke_steps", 0) or 0)
    if smoke_steps <= 0:
        return None
    out_dir = Path(kaggle_args.out_dir).resolve()
    smoke_dir = out_dir / "_smoke"
    _reset_dir(smoke_dir)
    _copy_seed_files(out_dir, smoke_dir)
    smoke_args = argparse.Namespace(**vars(kaggle_args))
    smoke_args.out_dir = str(smoke_dir)
    smoke_args.max_train_steps = smoke_steps
    smoke_args.require_gpu = True
    # Use a large epoch ceiling so the epoch limit never stops smoke before max_train_steps.
    # Without this, a warm-start seed at epoch=76 with epochs=1 would take 0 steps.
    smoke_args.epochs = 100000
    smoke_args.eval_steps = max(1, smoke_steps)
    smoke_args.checkpoint_steps = max(1, smoke_steps)
    smoke_args.allow_noop_resume = "false"
    metrics = train_main(smoke_args)
    if metrics.get("stopped_reason") != "max_train_steps":
        raise RuntimeError(f"smoke run did not stop at max_train_steps: {metrics.get('stopped_reason')}")
    if int(metrics.get("new_optimizer_steps", 0) or 0) < smoke_steps:
        raise RuntimeError(
            f"smoke run optimizer steps {metrics.get('new_optimizer_steps')} < requested {smoke_steps}"
        )
    if not (smoke_dir / "pose_t5_config.json").is_file():
        raise RuntimeError("smoke run did not write pose_t5_config.json")
    if not ((smoke_dir / "best_model_state.pt").is_file() or any(smoke_dir.glob("ckpt_step*.pt"))):
        raise RuntimeError("smoke run did not write a checkpoint")
    return metrics


def _validate_eval_readiness(eval_report: dict) -> None:
    status = eval_report.get("promotion_status")
    if isinstance(status, dict) and status.get("ready") is False:
        failures = status.get("failures")
        detail = "; ".join(str(item) for item in failures) if isinstance(failures, list) else "not ready"
        raise RuntimeError(f"PoseT5 eval is not promotion-ready: {detail}")


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
    p.add_argument("--device", type=str, default="auto")
    p.add_argument(
        "--require-gpu",
        type=str,
        default="true",
        choices=["true", "false"],
        help="Require CUDA before loading data/model in the trainer.",
    )
    p.add_argument(
        "--preflight-only",
        type=str,
        default="false",
        choices=["true", "false"],
        help="Run raw/staged dataset preflight and exit before smoke/full training.",
    )
    p.add_argument("--smoke-steps", type=int, default=20)
    p.add_argument("--min-val-chrf", type=float, default=80.0)
    p.add_argument("--min-val-exact-match-pct", type=float, default=80.0)

    # Pass-through args (rarely changed on Kaggle)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--dropout", type=float, default=0.4)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--num-encoder-layers", type=int, default=2)
    p.add_argument("--keep-checkpoints", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval-steps", type=int, default=200)
    p.add_argument(
        "--checkpoint-steps",
        type=int,
        default=200,
        help="Save checkpoints every N optimizer steps during eval (0 = save every eval).",
    )
    p.add_argument("--reset-progress-history", action="store_true")
    p.add_argument("--early-stopping-patience", type=int, default=6)
    p.add_argument("--early-stopping-min-delta", type=float, default=0.0)
    p.add_argument(
        "--early-stopping-metric",
        type=str,
        default="val_chrf",
        choices=["val_loss", "val_chrf"],
    )
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument(
        "--preload-train-features",
        type=str,
        default="true",
        choices=["true", "false"],
    )
    p.add_argument(
        "--balance-sources",
        type=str,
        default="auto",
        choices=["auto", "true", "false"],
    )
    p.add_argument("--focus-target-tokens", type=str, default="")
    p.add_argument("--focus-target-max-multiplier", type=float, default=1.0)
    p.add_argument(
        "--split-policy",
        type=str,
        default="auto",
        choices=["auto", "manifest", "video"],
    )
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
    p.add_argument("--required-sources", type=str, default="tsl51,thaisignvis,youtube_sl25_thai")
    p.add_argument(
        "--manifest-quality-sources",
        type=str,
        default="tsl51",
        help="Optional comma-separated sources to enforce in manifest-quality gates. Defaults to all sources in the split.",
    )
    p.add_argument(
        "--fail-on-manifest-quality",
        type=str,
        default="true",
        choices=["true", "false"],
    )
    p.add_argument(
        "--allow-noop-resume",
        type=str,
        default="false",
        choices=["true", "false"],
    )
    p.add_argument(
        "--evaluate-after-train",
        type=str,
        default="true",
        choices=["true", "false"],
        help="Evaluate the runtime export in --out-dir after training finishes.",
    )
    p.add_argument(
        "--eval-data-roots",
        type=str,
        default="",
        help="Optional evaluation data roots. Defaults to --data-roots when omitted.",
    )
    p.add_argument(
        "--eval-sources",
        type=str,
        default="tsl51",
        help="Optional comma-separated sources to keep in the evaluation subset before stratified sampling.",
    )
    p.add_argument(
        "--eval-report-data-roots",
        type=str,
        default="mixed_all_train_v6",
        help="Optional logical dataset ids to store in the eval report.",
    )
    p.add_argument(
        "--eval-split-policy",
        type=str,
        default="auto",
        choices=["auto", "manifest", "video"],
    )
    p.add_argument("--eval-device", type=str, default="cpu")
    p.add_argument("--eval-val-subset-size", type=int, default=50)
    p.add_argument("--expected-manifest-rows", type=int, default=1938)
    p.add_argument("--expected-resolved-examples", type=int, default=1938)
    p.add_argument(
        "--expected-source-counts",
        type=str,
        default="tsl51=252,thaisignvis=60,youtube_sl25_thai=1626",
        help="Expected aggregate source counts after cloud-side manifest extraction/preflight.",
    )
    p.add_argument(
        "--preflight-report-json",
        type=str,
        default="",
        help="Optional output JSON path for the cloud dataset preflight report.",
    )
    p.add_argument(
        "--raw-required-files",
        type=str,
        default="manifest.csv,manifest_quality.json,features.zip",
        help="Comma-separated list of files required in the raw (pre-unzip) dataset root.",
    )
    p.add_argument(
        "--staged-required-files",
        type=str,
        default="manifest.csv,manifest_quality.json",
        help="Comma-separated list of files required after staging (post-unzip) dataset root.",
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
    from scripts.evaluate_pose_t5_export import _evaluate_export  # noqa: PLC0415
    from scripts.verify_pose_t5_cloud_preflight import verify_cloud_preflight  # noqa: PLC0415

    kaggle_args = _build_kaggle_parser().parse_args()
    raw_data_roots = kaggle_args.data_roots
    raw_required = tuple(f.strip() for f in kaggle_args.raw_required_files.split(",") if f.strip())
    staged_required = tuple(f.strip() for f in kaggle_args.staged_required_files.split(",") if f.strip())
    raw_preflight_report = verify_cloud_preflight(
        raw_data_roots,
        expected_manifest_rows=int(getattr(kaggle_args, "expected_manifest_rows", 0) or 0),
        expected_resolved_examples=int(getattr(kaggle_args, "expected_resolved_examples", 0) or 0),
        expected_source_counts=str(getattr(kaggle_args, "expected_source_counts", "")),
        required_files=raw_required,
    )
    kaggle_args.data_roots = _prepare_data_roots(raw_data_roots)
    if getattr(kaggle_args, "eval_data_roots", ""):
        kaggle_args.eval_data_roots = _prepare_data_roots(kaggle_args.eval_data_roots)
    staged_preflight_report = verify_cloud_preflight(
        kaggle_args.data_roots,
        expected_manifest_rows=int(getattr(kaggle_args, "expected_manifest_rows", 0) or 0),
        expected_resolved_examples=int(getattr(kaggle_args, "expected_resolved_examples", 0) or 0),
        expected_source_counts=str(getattr(kaggle_args, "expected_source_counts", "")),
        required_files=staged_required,
    )
    preflight_report = _merge_preflight_reports(raw_preflight_report, staged_preflight_report)
    preflight_report_path = (
        str(Path(kaggle_args.preflight_report_json).resolve())
        if str(getattr(kaggle_args, "preflight_report_json", "")).strip()
        else os.path.join(kaggle_args.out_dir, "cloud_preflight.json")
    )
    _write_json(preflight_report_path, preflight_report)
    if not preflight_report.get("passed", False):
        raise RuntimeError(
            "cloud dataset preflight failed: "
            + "; ".join(str(item) for item in preflight_report.get("failures", []))
        )
    if _resolve_bool_flag(kaggle_args.preflight_only):
        print("[kaggle] Preflight passed; exiting because --preflight-only true.")
        return
    kaggle_args.require_gpu = _resolve_bool_flag(kaggle_args.require_gpu)
    _run_smoke_training(kaggle_args, train_main=train_main)
    print("[kaggle] Training with settings:")
    for k, v in vars(kaggle_args).items():
        print(f"  {k}: {v}")

    metrics = train_main(kaggle_args)
    print("[kaggle] Training finished. Final metrics:")
    print(f"  global_step:    {metrics.get('global_step')}")
    print(f"  stopped_reason: {metrics.get('stopped_reason')}")

    if _resolve_bool_flag(kaggle_args.evaluate_after_train):
        eval_report_json = os.path.join(kaggle_args.out_dir, "verified_eval.json")
        eval_samples_json = os.path.join(kaggle_args.out_dir, "verified_samples.json")
        eval_report = _evaluate_export(
            argparse.Namespace(
                export_dir=kaggle_args.out_dir,
                data_roots=kaggle_args.eval_data_roots or kaggle_args.data_roots,
                device=kaggle_args.eval_device,
                seed=kaggle_args.seed,
                val_subset_size=kaggle_args.eval_val_subset_size,
                split_policy=kaggle_args.eval_split_policy,
                required_sources=kaggle_args.required_sources,
                manifest_quality_sources=getattr(kaggle_args, "manifest_quality_sources", ""),
                report_data_roots=kaggle_args.eval_report_data_roots,
                report_json=eval_report_json,
                samples_json=eval_samples_json,
                eval_sources=kaggle_args.eval_sources,
                max_new_tokens=72,
                beam_size=5,
                no_repeat_ngram_size="3",
                repetition_penalty="1.5",
                length_penalty="0.7",
                min_val_chrf=kaggle_args.min_val_chrf,
                min_val_exact_match_pct=kaggle_args.min_val_exact_match_pct,
            )
        )
        _validate_eval_readiness(eval_report)
        print("[kaggle] Evaluation finished. Verified metrics:")
        print(f"  chrf:           {eval_report.get('chrf')}")
        print(f"  bleu:           {eval_report.get('bleu')}")
        print(f"  exact_match_pct:{eval_report.get('exact_match_pct')}")


if __name__ == "__main__":
    main()
