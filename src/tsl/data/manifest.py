"""Common manifest schema for sentence-level Sign Language Translation.

Every dataset adapter (tsl51, tsl-one-s, nstda-eaf, ...) must export its
records as :class:`SignTextExample` so downstream code only deals with a
single shape. The manifest is intentionally pure Python: no torch, no
numpy, so it can be imported anywhere in the pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass, field

__all__ = ["SignTextExample"]

_VALID_SPLITS: frozenset[str] = frozenset({"train", "val", "test"})


@dataclass(frozen=True)
class SignTextExample:
    """One sentence-level (sign video, Thai text) example."""

    example_id: str
    source: str
    split: str
    features_path: str
    target_text: str
    gloss: str | None = None
    metadata: dict | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.example_id, str) or self.example_id == "":
            raise ValueError("example_id must be a non-empty str")
        if not isinstance(self.source, str) or self.source == "":
            raise ValueError("source must be a non-empty str")
        if self.split not in _VALID_SPLITS:
            raise ValueError(
                f"split must be one of {sorted(_VALID_SPLITS)}, got {self.split!r}"
            )
        if not isinstance(self.features_path, str) or self.features_path == "":
            raise ValueError("features_path must be a non-empty str")
        if not isinstance(self.target_text, str) or self.target_text == "":
            raise ValueError("target_text must be a non-empty str")
