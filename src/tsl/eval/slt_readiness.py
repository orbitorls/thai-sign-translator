"""Readiness gates for sentence-level SLT checkpoints."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass

from tsl.data.manifest import SignTextExample
from tsl.eval.slt_metrics import chrf_corpus

__all__ = [
    "SltReadinessThresholds",
    "assess_slt_readiness",
    "compute_source_metrics",
]


@dataclass(frozen=True)
class SltReadinessThresholds:
    min_val_chrf: float = 80.0
    min_val_exact_match_pct: float = 80.0
    max_best_val_loss: float | None = None


def compute_source_metrics(
    examples: list[SignTextExample],
    hypotheses: list[str],
    references: list[str],
) -> dict[str, dict]:
    """Group validation metrics by ``example.source``."""
    if len(examples) != len(hypotheses) or len(examples) != len(references):
        raise ValueError("examples, hypotheses, and references must have matching lengths")

    grouped_hypotheses: dict[str, list[str]] = defaultdict(list)
    grouped_references: dict[str, list[str]] = defaultdict(list)

    for ex, hyp, ref in zip(examples, hypotheses, references):
        grouped_hypotheses[ex.source].append(hyp)
        grouped_references[ex.source].append(ref)

    return {
        source: chrf_corpus(grouped_hypotheses[source], grouped_references[source])
        for source in sorted(grouped_hypotheses)
    }


def assess_slt_readiness(
    overall_metrics: dict,
    source_metrics: dict[str, dict],
    manifest_quality: dict,
    train_metrics: dict | None = None,
    thresholds: SltReadinessThresholds | None = None,
) -> dict:
    """Combine metric thresholds with manifest-quality gates."""
    active_thresholds = thresholds or SltReadinessThresholds()
    train_metrics = train_metrics or {}

    failures: list[str] = []
    warnings: list[str] = []

    if overall_metrics.get("n", 0) <= 0:
        failures.append("No validation examples were evaluated.")

    if not manifest_quality.get("passed", False):
        failures.extend(manifest_quality.get("failures", []))

    val_chrf = float(overall_metrics.get("chrf", 0.0))
    if val_chrf < active_thresholds.min_val_chrf:
        failures.append(
            f"val chrf {val_chrf:.2f} < required {active_thresholds.min_val_chrf:.2f}."
        )

    val_exact = float(overall_metrics.get("exact_match_pct", 0.0))
    if val_exact < active_thresholds.min_val_exact_match_pct:
        failures.append(
            "val exact_match_pct "
            f"{val_exact:.2f} < required {active_thresholds.min_val_exact_match_pct:.2f}."
        )

    max_best_val_loss = active_thresholds.max_best_val_loss
    best_val_loss = train_metrics.get("best_val_loss")
    if max_best_val_loss is not None:
        if best_val_loss is None:
            warnings.append("best_val_loss missing from train_metrics.json; loss gate skipped.")
        elif float(best_val_loss) > max_best_val_loss:
            failures.append(
                f"best_val_loss {float(best_val_loss):.4f} > allowed {max_best_val_loss:.4f}."
            )

    return {
        "ready": len(failures) == 0,
        "failures": failures,
        "warnings": warnings,
        "thresholds": asdict(active_thresholds),
        "overall_metrics": dict(overall_metrics),
        "source_metrics": dict(source_metrics),
        "manifest_quality": dict(manifest_quality),
        "train_metrics": dict(train_metrics),
    }
