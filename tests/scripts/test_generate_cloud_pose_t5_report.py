from __future__ import annotations

import json

from scripts.generate_cloud_pose_t5_report import build_html_report


def test_build_html_report_writes_expected_sections(tmp_path):
    artifact_dir = tmp_path / "artifact"
    artifact_dir.mkdir()
    (artifact_dir / "verified_eval.json").write_text(
        json.dumps({"chrf": 12.3, "bleu": 4.5, "exact_match_pct": 6.7, "source_counts": {"tsl51": 25}}),
        encoding="utf-8",
    )
    (artifact_dir / "train_metrics.json").write_text(
        json.dumps({"stopped_reason": "early_stopping"}),
        encoding="utf-8",
    )
    (artifact_dir / "verified_samples.json").write_text(
        json.dumps([{"reference": "a", "hypothesis": "b", "score": 0.9}]),
        encoding="utf-8",
    )
    (artifact_dir / "runtime_metadata.json").write_text(
        json.dumps({"base_model": "google/mt5-small", "checkpoint_step": 200}),
        encoding="utf-8",
    )
    (artifact_dir / "manifest_quality.json").write_text(
        json.dumps({"passed": True}),
        encoding="utf-8",
    )
    output_html = tmp_path / "report.html"

    result = build_html_report(
        artifact_dir=str(artifact_dir),
        output_html=str(output_html),
        dataset_label="mixed_all_train_v6",
        model_label="PoseToTextT5",
    )

    html_text = output_html.read_text(encoding="utf-8")
    assert result["output_html"] == str(output_html.resolve())
    assert "PoseToTextT5" in html_text
    assert "mixed_all_train_v6" in html_text
    assert "Dataset Gate" in html_text
    assert "Predictions" in html_text
