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
from tsl.models.pose_t5 import PoseToTextT5
from tsl.train.checkpointing import (
    save_checkpoint,
    load_checkpoint,
    find_latest_checkpoint,
)
from tsl.eval.build_splits import split_by_video


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
        "--resume",
        type=str,
        default="auto",
        help="'auto' to find latest checkpoint in --out-dir, or path to a .pt file.",
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
        default=200,
        help="Evaluate every this many optimizer steps.",
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


def _build_scaler(use_amp: bool) -> Optional["torch.cuda.amp.GradScaler"]:
    """Return a GradScaler if AMP on CUDA, else None."""
    if use_amp and torch.cuda.is_available():
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


class _SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, examples):
        self._examples = examples

    def __len__(self):
        return len(self._examples)

    def __getitem__(self, idx):
        return self._examples[idx]


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

    with torch.no_grad():
        for batch in loader:
            src = batch.src.to(device)
            src_lengths = batch.src_lengths.to(device)
            labels = batch.labels.to(device)
            if use_amp:
                with torch.autocast(device_type=amp_device):
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
            token_ids = model.generate(src, src_lengths, max_new_tokens=128)
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = _resolve_amp(args.amp)
    scaler = _build_scaler(use_amp)
    amp_device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- Output directory --------------------------------------------------
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Data loading ------------------------------------------------------
    data_roots = args.data_roots.split(",")
    all_examples = _load_all_examples(data_roots)
    if not all_examples:
        raise ValueError(f"No examples loaded from data_roots={data_roots!r}")

    splits = split_by_video(
        all_examples,
        fracs={"train": 0.9, "val": 0.1},
        seed=args.seed,
    )
    train_examples = splits["train"]
    val_examples = splits["val"]

    print(
        f"[data] {len(all_examples)} total | "
        f"{len(train_examples)} train | {len(val_examples)} val"
    )

    # ---- Tokenizer ---------------------------------------------------------
    hf_tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    # ---- Model -------------------------------------------------------------
    local_model_path = args.base_model if os.path.isdir(args.base_model) else None
    model = PoseToTextT5(
        input_dim=312,
        num_encoder_layers=args.num_encoder_layers,
        encoder_dropout=0.1,
        downsample_factor=args.downsample_factor,
        base_model_name=args.base_model,
        local_model_path=local_model_path,
    ).to(device)

    # ---- Optimizer + scheduler --------------------------------------------
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    # ---- Resume -----------------------------------------------------------
    start_step = 0
    start_epoch = 0
    all_metrics: list[dict] = []

    resume_path: Optional[str] = None
    if args.resume == "auto":
        resume_path = find_latest_checkpoint(out_dir)
    elif args.resume and args.resume != "none":
        resume_path = args.resume

    if resume_path is not None and os.path.isfile(resume_path):
        print(f"[resume] Loading checkpoint: {resume_path}")
        ckpt_info = load_checkpoint(
            resume_path, model, optimizer, scheduler, scaler, device=str(device)
        )
        start_step = ckpt_info["step"]
        start_epoch = ckpt_info["epoch"]
        print(f"[resume] Resuming from step={start_step}, epoch={start_epoch}")
    else:
        print("[resume] No checkpoint found; starting from scratch.")

    # ---- Collate function -------------------------------------------------
    collate_fn = functools.partial(
        pose_t5_collate,
        hf_tokenizer=hf_tokenizer,
        load_features=load_features,
        max_src_len=args.max_src_len,
    )

    # ---- Training loop (step-based) ----------------------------------------
    global_step = start_step
    eval_counter = 0
    start_time = time.time()
    max_runtime_sec = args.max_runtime_min * 60

    train_dataset = _SimpleDataset(train_examples)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
        drop_last=False,
    )

    optimizer.zero_grad(set_to_none=True)
    accum_loss = 0.0
    accum_count = 0

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
                final_metrics = {
                    "global_step": global_step,
                    "stopped_reason": "max_runtime",
                    "history": all_metrics,
                }
                _save_json(final_metrics, out_dir / "train_metrics.json")
                model.save_pretrained(str(out_dir))
                return final_metrics

            # ---- Forward pass ------------------------------------------
            src = batch.src.to(device)
            src_lengths = batch.src_lengths.to(device)
            labels = batch.labels.to(device)

            if use_amp:
                with torch.autocast(device_type=amp_device):
                    out = model(src, src_lengths, labels=labels)
                loss = out.loss / args.grad_accum
                scaler.scale(loss).backward()
            else:
                out = model(src, src_lengths, labels=labels)
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

                avg_train_loss = accum_loss / args.grad_accum
                accum_loss = 0.0

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
                    if eval_counter % 5 == 0:
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

                    print(
                        f"[step {global_step}] epoch={epoch} | "
                        f"train_loss={avg_train_loss:.4f} | "
                        f"val_loss={val_loss:.4f} | "
                        f"val_chrf={val_chrf:.2f}"
                    )

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

                    model.train()

    # ---- Final save --------------------------------------------------------
    print(f"[done] Training complete. global_step={global_step}")
    final_metrics = {
        "global_step": global_step,
        "stopped_reason": "completed",
        "history": all_metrics,
    }
    _save_json(final_metrics, out_dir / "train_metrics.json")
    model.save_pretrained(str(out_dir))
    return final_metrics


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
