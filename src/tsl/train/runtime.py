from __future__ import annotations

import os

import torch


def is_wsl() -> bool:
    if os.environ.get("WSL_DISTRO_NAME"):
        return True

    try:
        with open("/proc/version", "r", encoding="utf-8") as f:
            version = f.read().lower()
    except OSError:
        return False

    return "microsoft" in version or "wsl" in version


def resolve_device(requested: str | None, require_gpu: bool) -> torch.device:
    normalized = None if requested is None else requested.strip().lower()
    cuda_available = torch.cuda.is_available()

    if require_gpu and not cuda_available:
        raise RuntimeError("CUDA is required but not available.")

    if require_gpu and normalized not in (None, "auto") and not normalized.startswith("cuda"):
        raise RuntimeError("CUDA is required; explicit non-CUDA devices are not allowed.")

    if normalized == "cpu":
        return torch.device("cpu")

    if normalized in (None, "auto"):
        return torch.device("cuda" if cuda_available else "cpu")

    if normalized and normalized.startswith("cuda"):
        if not cuda_available:
            raise RuntimeError("CUDA was requested but is not available.")
        return torch.device(normalized)

    return torch.device(normalized)


__all__ = ["is_wsl", "resolve_device"]
