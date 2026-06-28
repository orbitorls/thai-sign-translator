"""Repo-root oriented path helpers."""
from __future__ import annotations

from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def resolve_repo_path(path: str) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate.resolve()
    return (repo_root() / candidate).resolve()


__all__ = ["repo_root", "resolve_repo_path"]
