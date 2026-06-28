"""Internal schemas for user feedback contributions."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ContributionKind = Literal["correction", "teach"]
ContributionStatus = Literal["pending", "used"]


@dataclass(frozen=True)
class ContributionMeta:
    segment_id: str
    kind: ContributionKind
    text: str
    status: ContributionStatus
    user_hash: str
    consent_version: str
    consent_scope: list[str]
    created_at: str
    original_text: str | None = None
    model: str | None = None
    score: float | None = None
    dedup_hash: str | None = None
    capture_quality: dict | None = None
    train_allowed: bool = True
    delete_requested: bool = False
    video_path: str | None = None
    environment_tag: dict | None = None
