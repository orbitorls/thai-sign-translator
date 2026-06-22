"""Training entrypoint for PoseToTextT5 (Thai Sign Language Translation).

Multi-stage Kaggle-resumable training script. Loads unified v3-312 manifests,
builds a video-level train/val split, and runs a step-based training loop
with gradient accumulation, optional AMP, periodic evaluation, and clean
self-termination before Kaggle's 12-hour kill signal.

Outputs in ``--out-dir``:
    - ``ckpt_step<N>.pt``       : periodic checkpoints (pruned to --keep-checkpoints)
    - ``pose_encoder.pt``       : final pose encoder weights (via save_pretrained)
    - ``pose_t5_config.json``   : model constructor config (via save_pretrained)
    - ``train_metrics.json``    : per-eval metrics (loss + chrF)
"""
from __future__ import annotations

import argparse
from collections import Counter
import functools
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from transformers import AutoTokenizer

from tsl.data.unified import load_manifest, load_features
from tsl.data.pose_t5_collate import pose_t5_collate, PoseT5Batch
from tsl.inference.pose_t5_translator import PoseT5Translator
from tsl.models.pose_t5 import PoseToTextT5
from tsl.train.runtime import resolve_device
from tsl.train.checkpointing import (
    save_checkpoint,
    load_checkpoint,
    find_latest_checkpoint,
    find_best_checkpoint,
    _torch_save_fast,
)
from tsl.eval.build_splits import split_by_video
from tsl.eval.manifest_quality import analyze_manifest_quality


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train PoseToTextT5 on unified v3-312 sign language data."
    )
    p.add_argument(
        "--data-roots",
        type=str,
        required=True,
        help="Comma-separated paths to directories each containing manifest.csv",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default="checkpoints/pose_t5_v3",
        help="Directory for checkpoints and final model.",
    )
    p.add_argument(
        "--base-model",
        type=str,
        default="google/mt5-small",
        help="HuggingFace model name or local path for mT5.",
    )
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument(
        "--grad-accum",
        type=int,
        default=4,
        help="Gradient accumulation steps; effective batch = batch_size * grad_accum.",
    )
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="Dropout applied inside the pose encoder Transformer layers.",
    )
    p.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
        help="AdamW weight decay coefficient.",
    )
    p.add_argument("--max-src-len", type=int, default=512)
    p.add_argument("--downsample-factor", type=int, default=4)
    p.add_argument("--num-encoder-layers", type=int, default=2)
    p.add_argument(
        "--amp",
        type=str,
        default="auto",
        choices=["auto", "true", "false"],
        help="Enable Automatic Mixed Precision: auto / true / false.",
    )
    p.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Training device: auto, cpu, cuda, or cuda:<index>.",
    )
    p.add_argument(
        "--require-gpu",
        action="store_true",
        help="Fail before data/model loading unless CUDA is selected and available.",
    )
    p.add_argument(
        "--max-train-steps",
        type=int,
        default=0,
        help="Stop after this many new optimizer steps and export a smoke checkpoint (0 disables).",
    )
    p.add_argument(
        "--resume",
        type=str,
        default="auto",
        help="'auto' to find latest checkpoint, 'best' to find the best full checkpoint for the tracked metric, 'best_state' to prefer best_model_state.pt, or a path to a .pt file.",
    )
    p.add_argument(
        "--reset-progress-history",
        action="store_true",
        help="Ignore restored train_metrics history and restart progress/early-stopping state from the resumed checkpoint.",
    )
    p.add_argument(
        "--max-runtime-min",
        type=int,
        default=690,
        help="Self-terminate and save checkpoint after this many minutes.",
    )
    p.add_argument(
        "--keep-checkpoints",
        type=int,
        default=3,
        help="Number of recent checkpoints to keep (plus the best by val_chrf).",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--eval-steps",
        type=int,
        default=50,
        help="Evaluate every this many optimizer steps (default: 50).",
    )
    p.add_argument(
        "--checkpoint-steps",
        type=int,
        default=0,
        help="Save checkpoints every this many optimizer steps during eval (0 = save every eval).",
    )
    p.add_argument(
        "--early-stopping-patience",
        type=int,
        default=0,
        help="Number of evaluation windows without metric improvement before stopping (0 disables).",
    )
    p.add_argument(
        "--early-stopping-min-delta",
        type=float,
        default=0.0,
        help="Minimum metric improvement required to reset early stopping patience.",
    )
    p.add_argument(
        "--early-stopping-metric",
        type=str,
        default="val_loss",
        choices=["val_loss", "val_chrf"],
        help="Validation metric used to track best checkpoint and trigger early stopping.",
    )
    p.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader worker processes (0=main process, safe on Windows).",
    )
    p.add_argument(
        "--preload-train-features",
        type=str,
        default="true",
        choices=["true", "false"],
        help="Whether to pre-load all training feature arrays into RAM before training.",
    )
    p.add_argument(
        "--balance-sources",
        type=str,
        default="auto",
        choices=["auto", "true", "false"],
        help="Use inverse-frequency sampling so each source contributes more evenly to training.",
    )
    p.add_argument(
        "--focus-target-tokens",
        type=str,
        default="",
        help="Comma-separated target tokens to upweight when they remain underfit or frequently confused.",
    )
    p.add_argument(
        "--focus-target-max-multiplier",
        type=float,
        default=1.0,
        help="Maximum extra per-example weight applied from focus target tokens (1.0 disables).",
    )
    p.add_argument(
        "--split-policy",
        type=str,
        default="auto",
        choices=["auto", "manifest", "video"],
        help=(
            "How train/val splits are produced: "
            "'manifest' preserves per-row split labels, "
            "'video' rebuilds a fresh 90/10 split by video_id, "
            "'auto' prefers manifest labels when both train and val are present."
        ),
    )
    p.add_argument(
        "--required-sources",
        type=str,
        default="",
        help="Comma-separated source names that must appear in both train and val manifest-quality checks.",
    )
    p.add_argument(
        "--fail-on-manifest-quality",
        type=str,
        default="false",
        choices=["true", "false"],
        help="Abort before training when manifest-quality gates fail.",
    )
    p.add_argument(
        "--allow-noop-resume",
        type=str,
        default="true",
        choices=["true", "false"],
        help="Allow resume configurations that restore a checkpoint but perform zero new optimizer steps.",
    )
    p.add_argument(
        "--manifest-quality-sources",
        type=str,
        default="",
        help="Optional comma-separated sources to enforce in manifest-quality gates. Defaults to all sources in the split.",
    )
    return p


# ---------------------------------------------------------------------------
# AMP helpers
# ---------------------------------------------------------------------------


def _resolve_amp(amp_flag: str) -> bool:
    """Resolve the --amp flag to a bool."""
    if amp_flag == "true":
        return True
    if amp_flag == "false":
        return False
    # auto: use AMP only if CUDA is available
    return torch.cuda.is_available()


def _resolve_bool_flag(value) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _assert_finite_loss(loss: torch.Tensor, *, global_step: int, epoch: int) -> None:
    value = float(loss.detach())
    if not math.isfinite(value):
        raise RuntimeError(
            f"Non-finite training loss at epoch={epoch}, global_step={global_step}: {value}"
        )


def _should_balance_sources(value) -> bool | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text == "auto":
        return None
    return text in {"1", "true", "yes", "on"}


def _parse_csv_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        result: list[str] = []
        for item in value:
            text = str(item).strip()
            if text:
                result.append(text)
        return result
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _amp_dtype() -> "torch.dtype":
    """bfloat16 if CUDA supports it (Ampere+), else float16."""
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def _build_scaler(use_amp: bool) -> Optional["torch.cuda.amp.GradScaler"]:
    """Return a GradScaler only for fp16 (bfloat16 doesn't need it)."""
    if use_amp and torch.cuda.is_available() and _amp_dtype() == torch.float16:
        grad_scaler = getattr(torch.amp, "GradScaler", None)
        if grad_scaler is not None:
            return grad_scaler("cuda")
        return torch.cuda.amp.GradScaler()
    return None


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------


def _load_all_examples(data_roots: list[str]):
    """Load and concatenate examples from all data roots."""
    examples = []
    for root in data_roots:
        root = root.strip()
        if not root:
            continue
        examples.extend(load_manifest(root))
    return examples


def _has_manifest_train_val_splits(examples) -> bool:
    split_names = {
        str(getattr(ex, "split", "")).strip().lower()
        for ex in examples
        if str(getattr(ex, "split", "")).strip()
    }
    return "train" in split_names and "val" in split_names


def _split_examples_from_manifest(examples) -> dict[str, list]:
    train_examples = []
    val_examples = []
    for ex in examples:
        split_name = str(getattr(ex, "split", "")).strip().lower()
        if split_name == "train":
            train_examples.append(ex)
        elif split_name == "val":
            val_examples.append(ex)
    return {"train": train_examples, "val": val_examples}


def _build_train_val_splits(examples, *, split_policy: str, seed: int) -> tuple[dict[str, list], str]:
    normalized = str(split_policy or "auto").strip().lower()
    if normalized == "manifest":
        splits = _split_examples_from_manifest(examples)
        if not splits["train"] or not splits["val"]:
            raise RuntimeError(
                "split_policy='manifest' requires both train and val rows in the input manifest."
            )
        return splits, "manifest"
    if normalized == "video":
        return (
            split_by_video(
                examples,
                fracs={"train": 0.9, "val": 0.1},
                seed=seed,
            ),
            "video",
        )
    if _has_manifest_train_val_splits(examples):
        splits = _split_examples_from_manifest(examples)
        if splits["train"] and splits["val"]:
            return splits, "manifest"
    return (
        split_by_video(
            examples,
            fracs={"train": 0.9, "val": 0.1},
            seed=seed,
        ),
        "video",
    )


def _write_manifest_quality_report(
    out_dir: Path,
    train_examples,
    val_examples,
    *,
    required_sources: list[str] | None = None,
    gated_sources: list[str] | None = None,
) -> dict:
    report = analyze_manifest_quality(
        train_examples,
        val_examples,
        required_sources=required_sources,
        gated_sources=gated_sources,
    )
    _save_json(report, out_dir / "manifest_quality.json")

    overall = report.get("overall", {})
    print(
        "[data] manifest quality | "
        f"train_examples_per_target={overall.get('train_examples_per_target', 0.0):.4f} | "
        f"target_overlap_ratio={overall.get('target_overlap_ratio', 0.0):.4f} | "
        f"video_overlap_count={overall.get('video_overlap_count', 0)}",
        flush=True,
    )
    if not report.get("passed", False):
        print("[data][warning] manifest quality gate failures detected:", flush=True)
        for failure in report.get("failures", []):
            print(f"[data][warning] - {failure}", flush=True)
    return report


def _finalize_training(
    out_dir: Path,
    model: PoseToTextT5,
    hf_tokenizer,
    *,
    initial_step: int = 0,
    global_step: int,
    stopped_reason: str,
    all_metrics: list[dict],
    tracked_metric: str,
) -> dict:
    final_metrics = {
        "initial_step": initial_step,
        "global_step": global_step,
        "final_step": global_step,
        "new_optimizer_steps": max(0, global_step - initial_step),
        "stopped_reason": stopped_reason,
        "history": all_metrics,
    }
    _save_json(final_metrics, out_dir / "train_metrics.json")
    _restore_best_checkpoint_weights(out_dir, model, tracked_metric)
    model.save_pretrained(str(out_dir))
    if hf_tokenizer is not None:
        hf_tokenizer.save_pretrained(str(out_dir))
    return final_metrics


def _save_best_model_state(
    out_dir: Path,
    model: PoseToTextT5,
    *,
    step: int,
    epoch: int,
    metrics: dict[str, float],
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / "best_model_state.pt"
    tmp_target = out_dir / "best_model_state.tmp"
    payload = {
        "step": step,
        "epoch": epoch,
        "metrics": metrics,
        "model_state_dict": model.state_dict(),
    }
    try:
        _torch_save_fast(payload, tmp_target)
        os.replace(tmp_target, target)
    finally:
        if tmp_target.exists():
            tmp_target.unlink(missing_ok=True)
    return target


def _write_progress_metrics(
    out_dir: Path,
    *,
    initial_step: int,
    global_step: int,
    stopped_reason: str,
    history: list[dict],
) -> None:
    _save_json(
        {
            "initial_step": initial_step,
            "global_step": global_step,
            "final_step": global_step,
            "new_optimizer_steps": max(0, global_step - initial_step),
            "stopped_reason": stopped_reason,
            "history": history,
        },
        out_dir / "train_metrics.json",
    )


def _write_run_status(
    out_dir: Path,
    *,
    phase: str,
    global_step: int,
    epoch: int | None,
    initial_step: int,
    start_epoch: int,
    stopped_reason: str | None = None,
    extra: dict | None = None,
) -> None:
    payload = {
        "phase": phase,
        "global_step": global_step,
        "epoch": epoch,
        "initial_step": initial_step,
        "start_epoch": start_epoch,
        "stopped_reason": stopped_reason,
        "updated_at_unix": time.time(),
    }
    if extra:
        payload.update(extra)
    _save_json(payload, out_dir / "run_status.json")


def _load_progress_metrics(out_dir: Path) -> dict:
    metrics_path = out_dir / "train_metrics.json"
    if not metrics_path.is_file():
        return {}
    try:
        with open(metrics_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _assert_noop_resume_allowed(
    *,
    start_step: int,
    global_step: int,
    allow_noop_resume: bool,
    resume_path: str | None,
) -> None:
    if allow_noop_resume:
        return
    if start_step <= 0:
        return
    if global_step > start_step:
        return
    raise RuntimeError(
        "Resume completed with zero new optimizer steps. "
        f"resume={resume_path!r}, start_step={start_step}, final_step={global_step}. "
        "Use a fresh out_dir, increase epochs, or pass --allow-noop-resume true for export-only flows."
    )


def _trim_history_to_step(history: list[dict], max_step: int) -> list[dict]:
    trimmed: list[dict] = []
    for item in history:
        if not isinstance(item, dict):
            continue
        step = item.get("step")
        if step is None:
            trimmed.append(item)
            continue
        try:
            if int(step) <= max_step:
                trimmed.append(item)
        except (TypeError, ValueError):
            continue
    return trimmed


def _metric_mode(metric_name: str) -> str:
    if metric_name == "val_chrf":
        return "max"
    return "min"


def _find_best_checkpoint_for_metric(checkpoint_dir: Path, metric_name: str) -> Optional[Path]:
    checkpoints = sorted(checkpoint_dir.glob("ckpt_step*.pt"))
    if not checkpoints:
        return None

    best_path: Optional[Path] = None
    best_value = _initial_best_metric(metric_name)
    mode = _metric_mode(metric_name)
    for checkpoint_path in checkpoints:
        try:
            payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        except Exception:
            continue
        metrics = payload.get("metrics", {})
        if metric_name not in metrics:
            continue
        value = float(metrics[metric_name])
        if best_path is None:
            best_path = checkpoint_path
            best_value = value
            continue
        if mode == "max" and value > best_value:
            best_path = checkpoint_path
            best_value = value
        if mode == "min" and value < best_value:
            best_path = checkpoint_path
            best_value = value
    return best_path


def _restore_best_checkpoint_weights(out_dir: Path, model: PoseToTextT5, metric_name: str) -> None:
    best_state_path = out_dir / "best_model_state.pt"
    if best_state_path.is_file():
        payload = torch.load(best_state_path, map_location="cpu", weights_only=False)
        model.load_state_dict(payload["model_state_dict"])
        return
    best_checkpoint = _find_best_checkpoint_for_metric(out_dir, metric_name)
    if best_checkpoint is None:
        return
    payload = torch.load(best_checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model_state_dict"])


def _metric_better(candidate: float, best: float, metric_name: str, min_delta: float) -> bool:
    mode = _metric_mode(metric_name)
    if mode == "max":
        return candidate > (best + min_delta)
    return candidate < (best - min_delta)


def _initial_best_metric(metric_name: str) -> float:
    if _metric_mode(metric_name) == "max":
        return float("-inf")
    return float("inf")


def _count_evals_without_improvement(
    history: list[dict],
    metric_name: str,
    min_delta: float,
) -> int:
    best_metric = _initial_best_metric(metric_name)
    without_improvement = 0
    for item in history:
        if not isinstance(item, dict) or metric_name not in item:
            continue
        metric_value = float(item[metric_name])
        if _metric_better(metric_value, best_metric, metric_name, min_delta):
            best_metric = metric_value
            without_improvement = 0
        else:
            without_improvement += 1
    return without_improvement


class _SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, examples):
        self._examples = examples

    def __len__(self):
        return len(self._examples)

    def __getitem__(self, idx):
        return self._examples[idx]


class _PreloadedDataset(torch.utils.data.Dataset):
    """Pre-loads all feature arrays into RAM at init — eliminates per-batch disk I/O."""

    def __init__(self, examples):
        self._examples = examples
        self._arrays: list[np.ndarray] = []
        for ex in examples:
            self._arrays.append(load_features(ex.features_path))

    def __len__(self):
        return len(self._examples)

    def __getitem__(self, idx):
        return self._examples[idx], self._arrays[idx]


def _tokenize_target_text(text: str) -> list[str]:
    return [token for token in str(text or "").strip().split() if token]


def _build_source_balanced_sampler(
    train_examples,
    balance_mode,
    *,
    focus_target_tokens=None,
    focus_target_max_multiplier: float = 1.0,
) -> tuple[Optional[torch.utils.data.WeightedRandomSampler], dict]:
    requested = _should_balance_sources(balance_mode)
    source_counts = Counter(ex.source for ex in train_examples)
    focus_tokens = _parse_csv_list(focus_target_tokens)
    summary = {
        "enabled": False,
        "source_counts": dict(source_counts),
        "requested": balance_mode,
        "source_balance_enabled": False,
        "focus_target_tokens": focus_tokens,
        "focus_target_max_multiplier": float(focus_target_max_multiplier),
        "focus_balance_enabled": False,
    }
    weights = [1.0 for _ in train_examples]
    source_reason = "single_source"
    if len(source_counts) > 1 and requested is not False:
        weights = [1.0 / source_counts[ex.source] for ex in train_examples]
        summary["source_balance_enabled"] = True
        summary["weights_by_source"] = {
            source: round(1.0 / count, 8) for source, count in sorted(source_counts.items())
        }
        source_reason = "auto" if requested is None else "forced"
    elif requested is False:
        source_reason = "disabled"
    summary["source_reason"] = source_reason

    focus_token_counts = Counter()
    focus_target_max_multiplier = max(float(focus_target_max_multiplier or 1.0), 1.0)
    if focus_tokens and focus_target_max_multiplier > 1.0:
        for ex in train_examples:
            seen = set(_tokenize_target_text(ex.target_text))
            for token in focus_tokens:
                if token in seen:
                    focus_token_counts[token] += 1
        if focus_token_counts:
            max_count = max(focus_token_counts.values())
            focus_token_multipliers = {
                token: round(
                    min(focus_target_max_multiplier, max_count / count),
                    8,
                )
                for token, count in sorted(focus_token_counts.items())
                if count > 0
            }
            boosted_examples = 0
            for idx, ex in enumerate(train_examples):
                matched = [
                    focus_token_multipliers[token]
                    for token in set(_tokenize_target_text(ex.target_text))
                    if token in focus_token_multipliers
                ]
                if matched:
                    weights[idx] *= max(matched)
                    boosted_examples += 1
            summary["focus_balance_enabled"] = True
            summary["focus_token_counts"] = dict(focus_token_counts)
            summary["focus_token_multipliers"] = focus_token_multipliers
            summary["focus_examples"] = boosted_examples
        else:
            summary["focus_reason"] = "missing_tokens"
    elif focus_tokens:
        summary["focus_reason"] = "disabled"
    else:
        summary["focus_reason"] = "not_requested"

    if not summary["source_balance_enabled"] and not summary["focus_balance_enabled"]:
        summary["reason"] = "disabled"
        return None, summary

    sampler = torch.utils.data.WeightedRandomSampler(
        weights=weights,
        num_samples=len(train_examples),
        replacement=True,
    )
    summary["enabled"] = True
    if summary["source_balance_enabled"] and summary["focus_balance_enabled"]:
        summary["reason"] = "source_and_focus"
    elif summary["source_balance_enabled"]:
        summary["reason"] = "source_only"
    else:
        summary["reason"] = "focus_only"
    return sampler, summary


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _compute_val_loss(
    model: PoseToTextT5,
    val_examples,
    hf_tokenizer,
    batch_size: int,
    max_src_len: int,
    device: torch.device,
    use_amp: bool,
) -> float:
    """Compute average cross-entropy loss on the validation set."""
    if not val_examples:
        return float("inf")

    collate_fn = functools.partial(
        pose_t5_collate,
        hf_tokenizer=hf_tokenizer,
        load_features=load_features,
        max_src_len=max_src_len,
    )
    loader = torch.utils.data.DataLoader(
        _SimpleDataset(val_examples),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )

    model.eval()
    total_loss = 0.0
    n_batches = 0
    amp_device = "cuda" if torch.cuda.is_available() else "cpu"
    amp_dtype = _amp_dtype()

    with torch.no_grad():
        for batch in loader:
            src = batch.src.to(device)
            src_lengths = batch.src_lengths.to(device)
            labels = batch.labels.to(device)
            if use_amp:
                with torch.autocast(device_type=amp_device, dtype=amp_dtype):
                    out = model(src, src_lengths, labels=labels)
            else:
                out = model(src, src_lengths, labels=labels)
            total_loss += float(out.loss.detach())
            n_batches += 1

    return total_loss / max(n_batches, 1)


def _compute_val_chrf(
    model: PoseToTextT5,
    val_examples,
    hf_tokenizer,
    max_src_len: int,
    device: torch.device,
    sample_size: int = 50,
) -> float:
    """Compute corpus-level chrF on a sample of validation examples."""
    try:
        import sacrebleu
    except ImportError:
        return 0.0

    if not val_examples:
        return 0.0

    sample = val_examples[:sample_size]
    hypotheses = []
    references = []

    model.eval()
    with torch.no_grad():
        for ex in sample:
            try:
                arr = load_features(ex.features_path)
            except Exception:
                continue
            if arr.shape[0] > max_src_len:
                arr = arr[:max_src_len]
            src = torch.from_numpy(arr).unsqueeze(0).to(device)  # (1, T, 312)
            src_lengths = torch.tensor([arr.shape[0]], dtype=torch.long).to(device)
            token_ids = model.generate(
                src,
                src_lengths,
                max_new_tokens=PoseT5Translator.DEFAULT_MAX_NEW_TOKENS,
                num_beams=PoseT5Translator.DEFAULT_BEAM_SIZE,
                no_repeat_ngram_size=PoseT5Translator.DEFAULT_NO_REPEAT_NGRAM_SIZE,
                repetition_penalty=PoseT5Translator.DEFAULT_REPETITION_PENALTY,
                length_penalty=PoseT5Translator.DEFAULT_LENGTH_PENALTY,
                early_stopping=True,
            )
            # Decode
            text = hf_tokenizer.decode(token_ids[0], skip_special_tokens=True)
            hypotheses.append(text)
            references.append(ex.target_text)

    if not hypotheses:
        return 0.0

    # sacrebleu corpus_chrf: refs is a list of reference lists (one per refset)
    result = sacrebleu.corpus_chrf(hypotheses, [references])
    return float(result.score)


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------


def main(args: argparse.Namespace) -> dict:
    """Run the full training loop.

    Parameters
    ----------
    args:
        Parsed command-line arguments (argparse.Namespace).

    Returns
    -------
    dict
        Final training metrics.
    """
    # ---- Seeding -----------------------------------------------------------
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # ---- Device + AMP ------------------------------------------------------
    device = resolve_device(
        getattr(args, "device", "auto"),
        require_gpu=bool(getattr(args, "require_gpu", False)),
    )
    use_amp = _resolve_amp(args.amp)
    scaler = _build_scaler(use_amp)
    amp_device = "cuda" if device.type == "cuda" else "cpu"
    amp_dtype = _amp_dtype()

    # ---- Output directory --------------------------------------------------
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Data loading ------------------------------------------------------
    data_roots = args.data_roots.split(",")
    required_sources = _parse_csv_list(getattr(args, "required_sources", ""))
    manifest_quality_sources = _parse_csv_list(getattr(args, "manifest_quality_sources", ""))
    fail_on_manifest_quality = _resolve_bool_flag(
        getattr(args, "fail_on_manifest_quality", False)
    )
    allow_noop_resume = _resolve_bool_flag(getattr(args, "allow_noop_resume", True))
    all_examples = _load_all_examples(data_roots)
    if not all_examples:
        raise ValueError(f"No examples loaded from data_roots={data_roots!r}")

    splits, resolved_split_policy = _build_train_val_splits(
        all_examples,
        split_policy=getattr(args, "split_policy", "auto"),
        seed=args.seed,
    )
    train_examples = splits["train"]
    val_examples = splits["val"]

    print(
        f"[data] split_policy={resolved_split_policy} | "
        f"[data] {len(all_examples)} total | "
        f"{len(train_examples)} train | {len(val_examples)} val"
    )
    manifest_quality = _write_manifest_quality_report(
        out_dir,
        train_examples,
        val_examples,
        required_sources=required_sources,
        gated_sources=manifest_quality_sources,
    )
    if fail_on_manifest_quality and not manifest_quality.get("passed", False):
        raise RuntimeError(
            "Manifest-quality gates failed before training: "
            + "; ".join(manifest_quality.get("failures", []))
        )

    # ---- Tokenizer ---------------------------------------------------------
    hf_tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    # ---- Model -------------------------------------------------------------
    local_model_path = args.base_model if os.path.isdir(args.base_model) else None
    model = PoseToTextT5(
        input_dim=312,
        num_encoder_layers=args.num_encoder_layers,
        encoder_dropout=args.dropout,
        downsample_factor=args.downsample_factor,
        base_model_name=args.base_model,
        local_model_path=local_model_path,
    ).to(device)

    # ---- Optimizer + scheduler --------------------------------------------
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    # ---- Resume -----------------------------------------------------------
    start_step = 0
    start_epoch = 0
    all_metrics: list[dict] = []
    resume_metrics: dict[str, float] = {}
    resume_train_state_restored = False

    resume_path: Optional[str] = None
    if args.resume == "auto":
        resume_path = find_latest_checkpoint(out_dir)
    elif args.resume == "best_state":
        best_state = out_dir / "best_model_state.pt"
        resume_path = str(best_state) if best_state.is_file() else find_best_checkpoint(
            out_dir, metric=args.early_stopping_metric
        )
    elif args.resume == "best":
        resume_path = find_best_checkpoint(out_dir, metric=args.early_stopping_metric)
    elif args.resume and args.resume != "none":
        resume_path = args.resume

    if resume_path is not None and os.path.isfile(resume_path):
        print(f"[resume] Loading checkpoint: {resume_path}")
        ckpt_info = load_checkpoint(
            resume_path,
            model,
            optimizer,
            scheduler,
            scaler,
            device=str(device),
            allow_missing_train_state=True,
        )
        start_step = ckpt_info["step"]
        start_epoch = ckpt_info["epoch"]
        resume_metrics = ckpt_info.get("metrics", {})
        resume_train_state_restored = bool(ckpt_info.get("train_state_restored", False))
        print(f"[resume] Resuming from step={start_step}, epoch={start_epoch}")
        if not resume_train_state_restored:
            print(
                "[resume] Restored model weights only; optimizer/scheduler state unavailable.",
                flush=True,
            )
    else:
        print("[resume] No checkpoint found; starting from scratch.")

    # ---- Collate function -------------------------------------------------
    collate_fn = functools.partial(
        pose_t5_collate,
        hf_tokenizer=hf_tokenizer,
        load_features=load_features,
        max_src_len=args.max_src_len,
    )

    def _preloaded_collate(batch):
        """Collate for _PreloadedDataset — batch is list of (example, array)."""
        examples = [item[0] for item in batch]
        arrays = {item[0].features_path: item[1] for item in batch}
        return pose_t5_collate(
            examples,
            hf_tokenizer=hf_tokenizer,
            load_features=lambda path: arrays[path],
            max_src_len=args.max_src_len,
        )

    # ---- Training loop (step-based) ----------------------------------------
    progress_metrics = _load_progress_metrics(out_dir)
    existing_history = progress_metrics.get("history", [])
    all_metrics: list[dict] = existing_history if isinstance(existing_history, list) else []
    if start_step > 0 and bool(getattr(args, "reset_progress_history", False)):
        print(
            f"[resume] Resetting progress history at resumed step {start_step}.",
            flush=True,
        )
        seeded_metrics = {"step": start_step, "epoch": start_epoch}
        for key, value in resume_metrics.items():
            seeded_metrics[key] = float(value)
        all_metrics = [seeded_metrics] if resume_metrics else []
    elif start_step > 0:
        trimmed_history = _trim_history_to_step(all_metrics, start_step)
        trimmed_entries = len(all_metrics) - len(trimmed_history)
        if trimmed_entries > 0:
            print(
                f"[resume] Dropping {trimmed_entries} stale metric entries beyond resumed step {start_step}.",
                flush=True,
            )
        all_metrics = trimmed_history

    global_step = start_step
    initial_step = start_step
    max_train_steps = max(0, int(getattr(args, "max_train_steps", 0) or 0))
    eval_counter = len(all_metrics)
    start_time = time.time()
    max_runtime_sec = args.max_runtime_min * 60
    best_metric = max(
        (
            float(item[args.early_stopping_metric])
            for item in all_metrics
            if isinstance(item, dict) and args.early_stopping_metric in item
        ),
        default=_initial_best_metric(args.early_stopping_metric),
    ) if _metric_mode(args.early_stopping_metric) == "max" else min(
        (
            float(item[args.early_stopping_metric])
            for item in all_metrics
            if isinstance(item, dict) and args.early_stopping_metric in item
        ),
        default=_initial_best_metric(args.early_stopping_metric),
    )
    evals_without_improvement = _count_evals_without_improvement(
        all_metrics,
        args.early_stopping_metric,
        args.early_stopping_min_delta,
    )

    use_preloaded_train_features = _resolve_bool_flag(
        getattr(args, "preload_train_features", True)
    )
    source_sampler, source_sampling_summary = _build_source_balanced_sampler(
        train_examples,
        getattr(args, "balance_sources", "auto"),
        focus_target_tokens=getattr(args, "focus_target_tokens", ""),
        focus_target_max_multiplier=getattr(args, "focus_target_max_multiplier", 1.0),
    )
    _save_json(source_sampling_summary, out_dir / "source_sampling.json")
    if source_sampling_summary.get("enabled"):
        print(
            "[data] weighted sampling enabled | "
            f"reason={source_sampling_summary['reason']} | "
            f"source_counts={source_sampling_summary['source_counts']} | "
            f"focus_examples={source_sampling_summary.get('focus_examples', 0)}",
            flush=True,
        )
    else:
        print(
            "[data] weighted sampling disabled | "
            f"source_counts={source_sampling_summary['source_counts']} | "
            f"reason={source_sampling_summary.get('reason', 'unknown')}",
            flush=True,
        )
    _write_run_status(
        out_dir,
        phase="preparing_data",
        global_step=global_step,
        epoch=start_epoch,
        initial_step=initial_step,
        start_epoch=start_epoch,
        stopped_reason="running",
        extra={
            "train_examples": len(train_examples),
            "val_examples": len(val_examples),
            "split_policy": resolved_split_policy,
        },
    )
    if use_preloaded_train_features:
        print(f"[data] Pre-loading {len(train_examples)} train features into RAM...", flush=True)
        train_dataset = _PreloadedDataset(train_examples)
        print("[data] Pre-load complete.", flush=True)
        train_collate_fn = _preloaded_collate
    else:
        print("[data] Streaming train features from disk per batch.", flush=True)
        train_dataset = _SimpleDataset(train_examples)
        train_collate_fn = collate_fn
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(source_sampler is None),
        sampler=source_sampler,
        collate_fn=train_collate_fn,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    optimizer.zero_grad(set_to_none=True)
    accum_loss = 0.0
    accum_count = 0
    _write_run_status(
        out_dir,
        phase="training",
        global_step=global_step,
        epoch=start_epoch,
        initial_step=initial_step,
        start_epoch=start_epoch,
        stopped_reason="running",
    )

    for epoch in range(start_epoch, args.epochs):
        model.train()

        for batch_idx, batch in enumerate(train_loader):
            # ---- Runtime guard ----------------------------------------
            elapsed = time.time() - start_time
            if elapsed > max_runtime_sec:
                print(
                    f"[timeout] Elapsed {elapsed/60:.1f} min > "
                    f"{args.max_runtime_min} min. Saving and exiting."
                )
                _write_run_status(
                    out_dir,
                    phase="max_runtime",
                    global_step=global_step,
                    epoch=epoch,
                    initial_step=initial_step,
                    start_epoch=start_epoch,
                    stopped_reason="max_runtime",
                )
                save_checkpoint(
                    out_dir,
                    model,
                    optimizer,
                    scheduler,
                    scaler,
                    step=global_step,
                    epoch=epoch,
                    metrics={"val_loss": float("inf")},
                    keep_last_k=args.keep_checkpoints,
                )
                return _finalize_training(
                    out_dir=out_dir,
                    model=model,
                    hf_tokenizer=hf_tokenizer,
                    initial_step=initial_step,
                    global_step=global_step,
                    stopped_reason="max_runtime",
                    all_metrics=all_metrics,
                    tracked_metric=args.early_stopping_metric,
                )

            # ---- Forward pass ------------------------------------------
            src = batch.src.to(device)
            src_lengths = batch.src_lengths.to(device)
            labels = batch.labels.to(device)

            if use_amp:
                with torch.autocast(device_type=amp_device, dtype=amp_dtype):
                    out = model(src, src_lengths, labels=labels)
                _assert_finite_loss(out.loss, global_step=global_step, epoch=epoch)
                loss = out.loss / args.grad_accum
                if scaler is not None:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
            else:
                out = model(src, src_lengths, labels=labels)
                _assert_finite_loss(out.loss, global_step=global_step, epoch=epoch)
                loss = out.loss / args.grad_accum
                loss.backward()

            accum_loss += float(out.loss.detach())
            accum_count += 1

            # ---- Optimizer step every grad_accum batches ---------------
            if accum_count % args.grad_accum == 0:
                if use_amp and scaler is not None:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                if global_step <= 3 or global_step % max(1, args.eval_steps // 4 or 1) == 0:
                    _write_run_status(
                        out_dir,
                        phase="training",
                        global_step=global_step,
                        epoch=epoch,
                        initial_step=initial_step,
                        start_epoch=start_epoch,
                        stopped_reason="running",
                    )

                avg_train_loss = accum_loss / args.grad_accum
                accum_loss = 0.0

                if max_train_steps and (global_step - initial_step) >= max_train_steps:
                    _write_progress_metrics(
                        out_dir,
                        initial_step=initial_step,
                        global_step=global_step,
                        stopped_reason="max_train_steps",
                        history=all_metrics,
                    )
                    _write_run_status(
                        out_dir,
                        phase="max_train_steps",
                        global_step=global_step,
                        epoch=epoch,
                        initial_step=initial_step,
                        start_epoch=start_epoch,
                        stopped_reason="max_train_steps",
                        extra={"train_loss": avg_train_loss},
                    )
                    save_checkpoint(
                        out_dir,
                        model,
                        optimizer,
                        scheduler,
                        scaler,
                        step=global_step,
                        epoch=epoch,
                        metrics={"train_loss": avg_train_loss, "val_loss": float("inf"), "val_chrf": 0.0},
                        keep_last_k=args.keep_checkpoints,
                    )
                    _save_best_model_state(
                        out_dir,
                        model,
                        step=global_step,
                        epoch=epoch,
                        metrics={"train_loss": avg_train_loss, "val_loss": float("inf"), "val_chrf": 0.0},
                    )
                    return _finalize_training(
                        out_dir=out_dir,
                        model=model,
                        hf_tokenizer=hf_tokenizer,
                        initial_step=initial_step,
                        global_step=global_step,
                        stopped_reason="max_train_steps",
                        all_metrics=all_metrics,
                        tracked_metric=args.early_stopping_metric,
                    )

                # ---- Periodic evaluation ---------------------------------
                if global_step % args.eval_steps == 0:
                    val_loss = _compute_val_loss(
                        model,
                        val_examples,
                        hf_tokenizer,
                        args.batch_size,
                        args.max_src_len,
                        device,
                        use_amp,
                    )
                    scheduler.step(val_loss)

                    eval_counter += 1
                    val_chrf = 0.0
                    should_compute_chrf = (
                        args.early_stopping_metric == "val_chrf" or eval_counter % 5 == 0
                    )
                    if should_compute_chrf:
                        val_chrf = _compute_val_chrf(
                            model,
                            val_examples,
                            hf_tokenizer,
                            args.max_src_len,
                            device,
                            sample_size=50,
                        )

                    step_metrics = {
                        "step": global_step,
                        "epoch": epoch,
                        "train_loss": avg_train_loss,
                        "val_loss": val_loss,
                        "val_chrf": val_chrf,
                    }
                    all_metrics.append(step_metrics)
                    _write_progress_metrics(
                        out_dir,
                        initial_step=initial_step,
                        global_step=global_step,
                        stopped_reason="running",
                        history=all_metrics,
                    )
                    _write_run_status(
                        out_dir,
                        phase="evaluated",
                        global_step=global_step,
                        epoch=epoch,
                        initial_step=initial_step,
                        start_epoch=start_epoch,
                        stopped_reason="running",
                        extra={
                            "train_loss": avg_train_loss,
                            "val_loss": val_loss,
                            "val_chrf": val_chrf,
                        },
                    )

                    print(
                        f"[step {global_step}] epoch={epoch} | "
                        f"train_loss={avg_train_loss:.4f} | "
                        f"val_loss={val_loss:.4f} | "
                        f"val_chrf={val_chrf:.2f}",
                        flush=True,
                    )

                    tracked_metric = float(step_metrics[args.early_stopping_metric])
                    improved = _metric_better(
                        tracked_metric,
                        best_metric,
                        args.early_stopping_metric,
                        args.early_stopping_min_delta,
                    )
                    should_save_checkpoint = (
                        args.checkpoint_steps <= 0
                        or global_step % args.checkpoint_steps == 0
                    )
                    if improved:
                        _save_best_model_state(
                            out_dir,
                            model,
                            step=global_step,
                            epoch=epoch,
                            metrics={"val_loss": val_loss, "val_chrf": val_chrf},
                        )
                    if should_save_checkpoint:
                        save_checkpoint(
                            out_dir,
                            model,
                            optimizer,
                            scheduler,
                            scaler,
                            step=global_step,
                            epoch=epoch,
                            metrics={"val_loss": val_loss, "val_chrf": val_chrf},
                            keep_last_k=args.keep_checkpoints,
                        )
                    if improved:
                        best_metric = tracked_metric
                        evals_without_improvement = 0
                    else:
                        evals_without_improvement += 1

                    if (
                        args.early_stopping_patience > 0
                        and evals_without_improvement >= args.early_stopping_patience
                    ):
                        print(
                            "[early-stop] "
                            f"{args.early_stopping_metric} did not improve for "
                            f"{evals_without_improvement} evals. "
                            "Saving and exiting.",
                            flush=True,
                        )
                        _write_run_status(
                            out_dir,
                            phase="early_stopping",
                            global_step=global_step,
                            epoch=epoch,
                            initial_step=initial_step,
                            start_epoch=start_epoch,
                            stopped_reason="early_stopping",
                        )
                        return _finalize_training(
                            out_dir=out_dir,
                            model=model,
                            hf_tokenizer=hf_tokenizer,
                            initial_step=initial_step,
                            global_step=global_step,
                            stopped_reason="early_stopping",
                            all_metrics=all_metrics,
                            tracked_metric=args.early_stopping_metric,
                        )

                    model.train()

    # ---- Final save --------------------------------------------------------
    _assert_noop_resume_allowed(
        start_step=start_step,
        global_step=global_step,
        allow_noop_resume=allow_noop_resume,
        resume_path=resume_path,
    )
    print(f"[done] Training complete. global_step={global_step}")
    _write_run_status(
        out_dir,
        phase="completed",
        global_step=global_step,
        epoch=args.epochs - 1,
        initial_step=initial_step,
        start_epoch=start_epoch,
        stopped_reason="completed",
    )
    return _finalize_training(
        out_dir=out_dir,
        model=model,
        hf_tokenizer=hf_tokenizer,
        initial_step=initial_step,
        global_step=global_step,
        stopped_reason="completed",
        all_metrics=all_metrics,
        tracked_metric=args.early_stopping_metric,
    )


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def _save_json(data: dict, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    parser = _build_parser()
    main(parser.parse_args())
