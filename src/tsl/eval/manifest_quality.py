"""Manifest quality checks for sentence-level SLT readiness.

The current sentence-translation path already knows how to produce train
and validation example lists. This module stays compatible with that
contract by accepting the two lists directly instead of re-parsing CSVs
or depending on ``example.split`` values.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass

from tsl.data.manifest import SignTextExample

__all__ = [
    "ManifestQualityThresholds",
    "analyze_manifest_quality",
    "summarize_manifest_quality",
]


@dataclass(frozen=True)
class ManifestQualityThresholds:
    max_video_overlap: int = 0
    min_target_overlap_ratio: float = 0.05
    min_train_examples_per_target: float = 1.25


def summarize_manifest_quality(
    train_examples: list[SignTextExample],
    val_examples: list[SignTextExample],
    required_sources: list[str] | tuple[str, ...] | None = None,
) -> dict:
    """Return manifest statistics overall and per source."""
    source_names = {ex.source for ex in train_examples + val_examples}
    if required_sources:
        source_names.update(str(source).strip() for source in required_sources if str(source).strip())
    source_names = sorted(source_names)
    by_source: dict[str, dict] = {}
    for source in source_names:
        src_train = [ex for ex in train_examples if ex.source == source]
        src_val = [ex for ex in val_examples if ex.source == source]
        by_source[source] = _summarize_source(src_train, src_val)
    return {
        "overall": _summarize_source(train_examples, val_examples),
        "by_source": by_source,
    }


def analyze_manifest_quality(
    train_examples: list[SignTextExample],
    val_examples: list[SignTextExample],
    thresholds: ManifestQualityThresholds | None = None,
    required_sources: list[str] | tuple[str, ...] | None = None,
    gated_sources: list[str] | tuple[str, ...] | None = None,
) -> dict:
    """Run readiness-oriented manifest checks."""
    active_thresholds = thresholds or ManifestQualityThresholds()
    normalized_required_sources = tuple(
        source.strip()
        for source in (required_sources or ())
        if isinstance(source, str) and source.strip()
    )
    normalized_gated_sources = tuple(
        source.strip()
        for source in (gated_sources or ())
        if isinstance(source, str) and source.strip()
    )
    summary = summarize_manifest_quality(
        train_examples,
        val_examples,
        required_sources=normalized_required_sources,
    )

    failures: list[str] = []
    gated_source_set = set(normalized_gated_sources)
    for source, source_report in summary["by_source"].items():
        if gated_source_set and source not in gated_source_set:
            continue
        failures.extend(_check_source(source, source_report, active_thresholds))

    return {
        **summary,
        "passed": len(failures) == 0,
        "failures": failures,
        "required_sources": list(normalized_required_sources),
        "gated_sources": list(normalized_gated_sources),
        "thresholds": asdict(active_thresholds),
    }


def _summarize_source(
    train_examples: list[SignTextExample],
    val_examples: list[SignTextExample],
) -> dict:
    train_targets = Counter(ex.target_text for ex in train_examples)
    val_targets = Counter(ex.target_text for ex in val_examples)

    repeated_train_examples = sum(
        count for count in train_targets.values() if count > 1
    )
    unique_train_targets = len(train_targets)
    unique_val_targets = len(val_targets)
    overlap_targets = set(train_targets) & set(val_targets)
    train_video_ids = _collect_video_ids(train_examples)
    val_video_ids = _collect_video_ids(val_examples)
    overlap_videos = train_video_ids & val_video_ids

    train_examples_per_target = 0.0
    if unique_train_targets > 0:
        train_examples_per_target = len(train_examples) / unique_train_targets

    repeated_train_target_fraction = 0.0
    if train_examples:
        repeated_train_target_fraction = repeated_train_examples / len(train_examples)

    target_overlap_ratio = 0.0
    if unique_val_targets > 0:
        target_overlap_ratio = len(overlap_targets) / unique_val_targets

    return {
        "train_examples": len(train_examples),
        "val_examples": len(val_examples),
        "unique_train_targets": unique_train_targets,
        "unique_val_targets": unique_val_targets,
        "target_overlap_count": len(overlap_targets),
        "target_overlap_ratio": round(target_overlap_ratio, 4),
        "train_examples_per_target": round(train_examples_per_target, 4),
        "repeated_train_target_fraction": round(repeated_train_target_fraction, 4),
        "video_overlap_count": len(overlap_videos),
        "video_overlap_ids": sorted(overlap_videos),
    }


def _check_source(
    source: str,
    source_report: dict,
    thresholds: ManifestQualityThresholds,
) -> list[str]:
    failures: list[str] = []

    if source_report["train_examples"] == 0:
        failures.append(f"{source}: missing train examples for readiness gate.")
        return failures
    if source_report["val_examples"] == 0:
        failures.append(f"{source}: missing val examples for readiness gate.")
        return failures

    if source_report["video_overlap_count"] > thresholds.max_video_overlap:
        failures.append(
            f"{source}: video overlap between train/val is "
            f"{source_report['video_overlap_count']} > {thresholds.max_video_overlap}."
        )

    if source_report["target_overlap_ratio"] < thresholds.min_target_overlap_ratio:
        failures.append(
            f"{source}: target overlap ratio "
            f"{source_report['target_overlap_ratio']:.4f} < "
            f"{thresholds.min_target_overlap_ratio:.4f}."
        )

    if (
        source_report["train_examples_per_target"]
        < thresholds.min_train_examples_per_target
    ):
        failures.append(
            f"{source}: one example per target regime detected "
            f"({source_report['train_examples_per_target']:.4f} < "
            f"{thresholds.min_train_examples_per_target:.4f})."
        )

    return failures


def _collect_video_ids(examples: list[SignTextExample]) -> set[str]:
    video_ids: set[str] = set()
    for ex in examples:
        raw = None
        metadata = getattr(ex, "metadata", None)
        if isinstance(metadata, dict):
            raw = metadata.get("video_id")
        value = str(raw).strip() if raw is not None else ""
        if value:
            video_ids.add(value)
    return video_ids
