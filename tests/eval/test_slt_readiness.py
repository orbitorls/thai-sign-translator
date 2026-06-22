from tsl.data.manifest import SignTextExample
from tsl.eval.slt_readiness import (
    SltReadinessThresholds,
    assess_slt_readiness,
    compute_source_metrics,
)


def _example(
    example_id: str,
    *,
    source: str,
    split: str = "val",
    target_text: str,
) -> SignTextExample:
    return SignTextExample(
        example_id=example_id,
        source=source,
        split=split,
        features_path=f"features/{example_id}.npy",
        target_text=target_text,
    )


def test_compute_source_metrics_returns_per_source_breakdown():
    examples = [
        _example("tsl-1", source="tsl51", target_text="alpha"),
        _example("tsl-2", source="tsl51", target_text="beta"),
        _example("yt-1", source="youtube_sl25", target_text="gamma"),
    ]
    hypotheses = ["alpha", "beta", "zzz"]
    references = ["alpha", "beta", "gamma"]

    metrics = compute_source_metrics(examples, hypotheses, references)

    assert set(metrics) == {"tsl51", "youtube_sl25"}
    assert metrics["tsl51"]["n"] == 2
    assert metrics["tsl51"]["exact_match_pct"] == 100.0
    assert metrics["youtube_sl25"]["n"] == 1
    assert metrics["youtube_sl25"]["exact_match_pct"] == 0.0


def test_assess_slt_readiness_fails_thresholds():
    readiness = assess_slt_readiness(
        overall_metrics={"chrf": 13.05, "exact_match_pct": 0.0, "n": 25},
        source_metrics={"youtube_sl25": {"chrf": 13.05, "exact_match_pct": 0.0, "n": 25}},
        manifest_quality={
            "passed": True,
            "failures": [],
            "by_source": {},
        },
        train_metrics={"best_val_loss": 1.72},
        thresholds=SltReadinessThresholds(
            min_val_chrf=50.0,
            min_val_exact_match_pct=10.0,
            max_best_val_loss=1.0,
        ),
    )

    assert readiness["ready"] is False
    assert any("chrf" in failure.lower() for failure in readiness["failures"])
    assert any("exact" in failure.lower() for failure in readiness["failures"])
    assert any("best_val_loss" in failure for failure in readiness["failures"])


def test_assess_slt_readiness_passes_thresholds():
    readiness = assess_slt_readiness(
        overall_metrics={"chrf": 100.0, "exact_match_pct": 100.0, "n": 25},
        source_metrics={"tsl51": {"chrf": 100.0, "exact_match_pct": 100.0, "n": 25}},
        manifest_quality={
            "passed": True,
            "failures": [],
            "by_source": {},
        },
        train_metrics={"best_val_loss": 0.01},
        thresholds=SltReadinessThresholds(
            min_val_chrf=90.0,
            min_val_exact_match_pct=90.0,
            max_best_val_loss=0.1,
        ),
    )

    assert readiness["ready"] is True
    assert readiness["failures"] == []
