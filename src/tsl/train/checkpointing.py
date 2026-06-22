"""Kaggle-resumable step-based checkpointing for Thai Sign Language training.

Provides atomic checkpoint saves (write to temp file then rename) to prevent
corruption from kill signals, and utilities to find the latest or best checkpoint
by metric value. Designed for 12-hour Kaggle sessions where resumability is critical.
"""
from __future__ import annotations

import os
import random
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn


__all__ = ["save_checkpoint", "load_checkpoint", "find_latest_checkpoint", "find_best_checkpoint"]


def _torch_save_fast(payload, path: str | Path) -> None:
    torch.save(payload, path, _use_new_zipfile_serialization=False)


def _ckpt_step(path: Path) -> int:
    """Extract step number from a checkpoint filename like ckpt_step00001234.pt."""
    name = path.stem  # e.g. "ckpt_step00001234"
    return int(name.split("step")[1])


def save_checkpoint(
    checkpoint_dir: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler,
    step: int,
    epoch: int,
    metrics: dict,
    keep_last_k: int = 3,
) -> Path:
    """Save all training state atomically and prune old checkpoints.

    Parameters
    ----------
    checkpoint_dir:
        Directory to write checkpoints into (created if absent).
    model:
        Model whose ``state_dict`` will be saved.
    optimizer:
        Optimizer state to save.
    scheduler:
        LR-scheduler state to save (may be None).
    scaler:
        ``torch.cuda.amp.GradScaler`` instance (or None if not using AMP).
    step:
        Current global training step (used in the filename).
    epoch:
        Current epoch index.
    metrics:
        Arbitrary dict of metrics; ``val_chrf`` is used to track the best ckpt.
    keep_last_k:
        Number of most-recent checkpoints to retain (in addition to the best).

    Returns
    -------
    Path
        Path to the saved checkpoint file.
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    rng_states: dict = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }

    payload = {
        "step": step,
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "rng_states": rng_states,
        "metrics": metrics,
    }

    filename = f"ckpt_step{step:08d}.pt"
    final_path = checkpoint_dir / filename

    # Atomic write: write to temp file in the same directory then rename.
    fd, tmp_path = tempfile.mkstemp(dir=checkpoint_dir, suffix=".tmp")
    try:
        os.close(fd)
        _torch_save_fast(payload, tmp_path)
        os.replace(tmp_path, final_path)
    except Exception:
        # Clean up temp file if anything goes wrong.
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise

    # Prune old checkpoints, retaining keep_last_k most recent + best by val_chrf.
    _prune_checkpoints(checkpoint_dir, keep_last_k=keep_last_k)

    return final_path


def _write_checkpoint_refs(
    checkpoint_dir: Path,
    *,
    latest_path: Optional[Path],
    best_path: Optional[Path],
) -> None:
    latest_ref = checkpoint_dir / "latest_checkpoint.txt"
    best_ref = checkpoint_dir / "best_checkpoint.txt"
    if latest_path is not None:
        latest_ref.write_text(latest_path.name, encoding="utf-8")
    elif latest_ref.exists():
        latest_ref.unlink()
    if best_path is not None:
        best_ref.write_text(best_path.name, encoding="utf-8")
    elif best_ref.exists():
        best_ref.unlink()


def _prune_checkpoints(checkpoint_dir: Path, keep_last_k: int) -> None:
    """Delete old checkpoints, keeping keep_last_k most recent + best by val_chrf."""
    ckpts = sorted(checkpoint_dir.glob("ckpt_step*.pt"), key=_ckpt_step)
    latest_path = ckpts[-1] if ckpts else None

    # Identify the best checkpoint by val_chrf.
    best_path: Optional[Path] = None
    best_val = float("-inf")
    for ckpt in ckpts:
        try:
            data = torch.load(ckpt, map_location="cpu", weights_only=False)
            val = data.get("metrics", {}).get("val_chrf", None)
            if val is not None and val > best_val:
                best_val = val
                best_path = ckpt
        except Exception:
            continue

    if len(ckpts) <= keep_last_k:
        _write_checkpoint_refs(checkpoint_dir, latest_path=latest_path, best_path=best_path)
        return

    # Keep the last keep_last_k checkpoints.
    to_keep = set(ckpts[-keep_last_k:])
    if best_path is not None:
        to_keep.add(best_path)

    for ckpt in ckpts:
        if ckpt not in to_keep:
            try:
                ckpt.unlink()
            except OSError:
                pass
    _write_checkpoint_refs(checkpoint_dir, latest_path=latest_path, best_path=best_path)


def load_checkpoint(
    checkpoint_path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler,
    device: str | torch.device = "cpu",
    *,
    allow_missing_train_state: bool = False,
) -> dict:
    """Load training state from a checkpoint and restore all component states.

    Parameters
    ----------
    checkpoint_path:
        Path to the ``.pt`` checkpoint file to load.
    model:
        Model to restore ``state_dict`` into.
    optimizer:
        Optimizer to restore state into.
    scheduler:
        LR-scheduler to restore state into (may be None).
    scaler:
        ``torch.cuda.amp.GradScaler`` to restore state into (may be None).
    device:
        Device to map tensors to when loading.

    Returns
    -------
    dict
        ``{"step": int, "epoch": int, "metrics": dict}``
    """
    checkpoint_path = Path(checkpoint_path)
    # weights_only=False required to load RNG state objects (numpy arrays, Python tuples, etc.)
    data = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    model.load_state_dict(data["model_state_dict"])
    optimizer_state_dict = data.get("optimizer_state_dict")
    train_state_restored = optimizer_state_dict is not None
    if optimizer_state_dict is not None:
        optimizer.load_state_dict(optimizer_state_dict)
    elif not allow_missing_train_state:
        raise KeyError("optimizer_state_dict")

    if train_state_restored and scheduler is not None and data.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(data["scheduler_state_dict"])

    if train_state_restored and scaler is not None and data.get("scaler_state_dict") is not None:
        scaler.load_state_dict(data["scaler_state_dict"])

    # Restore RNG states for reproducibility.
    rng = data.get("rng_states", {})
    if train_state_restored and "python" in rng:
        random.setstate(rng["python"])
    if train_state_restored and "numpy" in rng:
        np.random.set_state(rng["numpy"])
    if train_state_restored and "torch" in rng:
        torch.set_rng_state(rng["torch"])
    if train_state_restored and "torch_cuda" in rng and torch.cuda.is_available() and rng["torch_cuda"]:
        torch.cuda.set_rng_state_all(rng["torch_cuda"])

    return {
        "step": data["step"],
        "epoch": data["epoch"],
        "metrics": data.get("metrics", {}),
        "train_state_restored": train_state_restored,
    }


def find_latest_checkpoint(checkpoint_dir: str | Path) -> Optional[str]:
    """Return the path to the most recent checkpoint (highest step number).

    Parameters
    ----------
    checkpoint_dir:
        Directory to search.

    Returns
    -------
    str or None
        Absolute path to the latest checkpoint, or ``None`` if none exist.
    """
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return None

    ckpts = sorted(checkpoint_dir.glob("ckpt_step*.pt"), key=_ckpt_step)
    if not ckpts:
        return None

    return str(ckpts[-1])


def find_best_checkpoint(checkpoint_dir: str | Path, metric: str = "val_chrf") -> Optional[str]:
    """Return the path to the checkpoint with the highest value of ``metric``.

    Parameters
    ----------
    checkpoint_dir:
        Directory to search.
    metric:
        Key to look up inside each checkpoint's ``metrics`` dict.

    Returns
    -------
    str or None
        Absolute path to the best checkpoint, or ``None`` if none found.
    """
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return None

    ckpts = list(checkpoint_dir.glob("ckpt_step*.pt"))
    if not ckpts:
        return None

    best_path: Optional[Path] = None
    best_val = float("-inf")
    for ckpt in ckpts:
        try:
            data = torch.load(ckpt, map_location="cpu", weights_only=False)
            val = data.get("metrics", {}).get(metric, None)
            if val is not None and val > best_val:
                best_val = val
                best_path = ckpt
        except Exception:
            continue

    return str(best_path) if best_path is not None else None
