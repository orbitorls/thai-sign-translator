from tsl.data.manifest import SignTextExample
from tsl.eval.manifest_quality import analyze_manifest_quality


def _example(
    example_id: str,
    *,
    split: str,
    source: str = "youtube_sl25",
    target_text: str,
    video_id: str,
) -> SignTextExample:
    return SignTextExample(
        example_id=example_id,
        source=source,
        split=split,
        features_path=f"features/{example_id}.npy",
        target_text=target_text,
        metadata={"video_id": video_id},
    )


def test_manifest_quality_rejects_video_leakage():
    train_examples = [
        _example("train-1", split="train", target_text="alpha", video_id="vid-1"),
        _example("train-2", split="train", target_text="alpha", video_id="vid-2"),
    ]
    val_examples = [
        _example("val-1", split="val", target_text="alpha", video_id="vid-1"),
    ]

    report = analyze_manifest_quality(train_examples, val_examples)

    source_report = report["by_source"]["youtube_sl25"]
    assert report["passed"] is False
    assert source_report["video_overlap_count"] == 1
    assert any("video overlap" in failure.lower() for failure in report["failures"])


def test_manifest_quality_rejects_zero_target_overlap():
    train_examples = [
        _example("train-1", split="train", target_text="alpha", video_id="vid-1"),
        _example("train-2", split="train", target_text="alpha", video_id="vid-2"),
        _example("train-3", split="train", target_text="beta", video_id="vid-3"),
        _example("train-4", split="train", target_text="beta", video_id="vid-4"),
    ]
    val_examples = [
        _example("val-1", split="val", target_text="gamma", video_id="vid-5"),
        _example("val-2", split="val", target_text="delta", video_id="vid-6"),
    ]

    report = analyze_manifest_quality(train_examples, val_examples)

    source_report = report["by_source"]["youtube_sl25"]
    assert report["passed"] is False
    assert source_report["target_overlap_ratio"] == 0.0
    assert any("target overlap" in failure.lower() for failure in report["failures"])


def test_manifest_quality_rejects_one_example_per_target():
    train_examples = [
        _example("train-1", split="train", target_text="alpha", video_id="vid-1"),
        _example("train-2", split="train", target_text="beta", video_id="vid-2"),
        _example("train-3", split="train", target_text="gamma", video_id="vid-3"),
    ]
    val_examples = [
        _example("val-1", split="val", target_text="alpha", video_id="vid-4"),
    ]

    report = analyze_manifest_quality(train_examples, val_examples)

    source_report = report["by_source"]["youtube_sl25"]
    assert report["passed"] is False
    assert source_report["train_examples_per_target"] == 1.0
    assert any("one example per target" in failure.lower() for failure in report["failures"])


def test_manifest_quality_rejects_missing_required_source():
    train_examples = [
        _example("train-1", split="train", source="tsl51", target_text="alpha", video_id="vid-1"),
        _example("train-2", split="train", source="tsl51", target_text="alpha", video_id="vid-2"),
    ]
    val_examples = [
        _example("val-1", split="val", source="tsl51", target_text="alpha", video_id="vid-3"),
    ]

    report = analyze_manifest_quality(
        train_examples,
        val_examples,
        required_sources=["tsl51", "thaisignvis"],
    )

    assert report["passed"] is False
    assert report["required_sources"] == ["tsl51", "thaisignvis"]
    assert "thaisignvis" in report["by_source"]
    assert any("thaisignvis: missing train examples" in failure.lower() for failure in report["failures"])


def test_manifest_quality_can_gate_only_selected_sources():
    train_examples = [
        _example("train-1", split="train", source="tsl51", target_text="alpha", video_id="vid-1"),
        _example("train-2", split="train", source="tsl51", target_text="alpha", video_id="vid-2"),
        _example("train-3", split="train", source="thaisignvis", target_text="beta", video_id="vid-3"),
    ]
    val_examples = [
        _example("val-1", split="val", source="tsl51", target_text="alpha", video_id="vid-4"),
        _example("val-2", split="val", source="thaisignvis", target_text="gamma", video_id="vid-5"),
    ]

    report = analyze_manifest_quality(
        train_examples,
        val_examples,
        required_sources=["tsl51"],
        gated_sources=["tsl51"],
    )

    assert report["passed"] is True
    assert report["gated_sources"] == ["tsl51"]
    assert "thaisignvis" in report["by_source"]
