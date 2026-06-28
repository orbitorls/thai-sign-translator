"""Privacy and consent schemas."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

CONSENT_VERSION = "2026-06-28-v1"

ConsentScope = Literal[
    "service",
    "model_improvement",
    "video_research",
    "academic_publication",
]

ALL_CONSENT_SCOPES: tuple[ConsentScope, ...] = (
    "service",
    "model_improvement",
    "video_research",
    "academic_publication",
)

ConsentSource = Literal[
    "consent_modal",
    "settings_toggle",
    "api",
    "withdrawal",
]


@dataclass(frozen=True)
class ConsentRecord:
    user_hash: str
    consent_version: str
    scope: ConsentScope
    granted: bool
    recorded_at: str
    source: ConsentSource
