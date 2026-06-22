from __future__ import annotations

import json
from pathlib import Path

from scripts.check_cloud_pose_t5_status import classify_cloud_pose_t5_status


def test_classify_cloud_pose_t5_status_marks_gpu_wait(tmp_path):
    launcher_status = tmp_path / "launcher.status.json"
    launcher_status.write_text(
        json.dumps(
            {
                "phase": "error",
                "message": "Unable to allocate requested GPU.",
                "last_errors": {"T4": "T4: Service Unavailable"},
            }
        ),
        encoding="utf-8",
    )

    report = classify_cloud_pose_t5_status(launcher_status_json=str(launcher_status))

    assert report["classification"] == "gpu_wait"


def test_classify_cloud_pose_t5_status_marks_complete_from_artifacts(tmp_path):
    artifact_dir = tmp_path / "artifact"
    artifact_dir.mkdir()
    (artifact_dir / "verified_eval.json").write_text("{}", encoding="utf-8")
    (artifact_dir / "runtime_metadata.json").write_text("{}", encoding="utf-8")

    report = classify_cloud_pose_t5_status(artifact_dir=str(artifact_dir))

    assert report["classification"] == "complete"


def test_classify_cloud_pose_t5_status_marks_cloud_api_stale(tmp_path):
    orchestrator_status = tmp_path / "orchestrator.status.json"
    orchestrator_status.write_text(
        json.dumps(
            {
                "source_status": "RUNNING",
                "source_visible_stale_suspected": True,
            }
        ),
        encoding="utf-8",
    )

    report = classify_cloud_pose_t5_status(orchestrator_status_json=str(orchestrator_status))

    assert report["classification"] == "cloud_api_stale"
