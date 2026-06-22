from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from scripts.audit_pose_t5_dataset import audit_pose_t5_dataset, main


def _write_pose_t5_manifest(root: Path, rows: list[dict[str, str]]) -> None:
    features_dir = root / "features"
    features_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows = []
    for index, row in enumerate(rows):
        rel_path = f"features/{row['segment_id']}.npy"
        np.save(root / rel_path, np.full((4 + index, 312), index, dtype=np.float32))
        manifest_rows.append(
            {
                "segment_id": row["segment_id"],
                "npy_path": rel_path,
                "text": row["text"],
                "video_id": row["video_id"],
                "split": row["split"],
                "source": row["source"],
                "feature_layout_version": "v3-312",
            }
        )

    with open(root / "manifest.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)


def test_audit_pose_t5_dataset_reports_train_only_sources(tmp_path):
    _write_pose_t5_manifest(
        tmp_path,
        [
            {"segment_id": "tsl-train", "text": "hello", "video_id": "v1", "split": "train", "source": "tsl51"},
            {"segment_id": "tsl-val", "text": "hello", "video_id": "v2", "split": "val", "source": "tsl51"},
            {"segment_id": "yt-train", "text": "unique yt", "video_id": "v3", "split": "train", "source": "youtube_sl25_thai"},
        ],
    )

    report = audit_pose_t5_dataset(
        data_roots=str(tmp_path),
        expected_manifest_rows=3,
        expected_source_counts="tsl51=2,youtube_sl25_thai=1",
        production_gated_sources="tsl51",
    )

    assert report["passed"] is True
    assert report["source_counts"] == {"tsl51": 2, "youtube_sl25_thai": 1}
    assert report["feature_dim_counts"] == {"312": 3}
    assert report["train_only_sources"] == ["youtube_sl25_thai"]
    assert any("youtube_sl25_thai" in warning for warning in report["production_warnings"])


def test_audit_pose_t5_dataset_fails_expected_count_mismatch(tmp_path):
    _write_pose_t5_manifest(
        tmp_path,
        [
            {"segment_id": "seg0", "text": "hello", "video_id": "v1", "split": "train", "source": "tsl51"},
        ],
    )

    report = audit_pose_t5_dataset(
        data_roots=str(tmp_path),
        expected_manifest_rows=2,
        expected_source_counts="tsl51=2",
        production_gated_sources="tsl51",
    )

    assert report["passed"] is False
    assert any("manifest rows 1 != expected 2" in failure for failure in report["failures"])
    assert any("production-gated source tsl51 has no val/test examples" in failure for failure in report["failures"])


def test_audit_pose_t5_dataset_cli_writes_json(tmp_path):
    _write_pose_t5_manifest(
        tmp_path,
        [
            {"segment_id": "seg0", "text": "hello", "video_id": "v1", "split": "train", "source": "tsl51"},
            {"segment_id": "seg1", "text": "hello", "video_id": "v2", "split": "val", "source": "tsl51"},
        ],
    )
    report_path = tmp_path / "audit.json"

    code = main(
        [
            "--data-roots",
            str(tmp_path),
            "--expected-manifest-rows",
            "2",
            "--expected-source-counts",
            "tsl51=2",
            "--production-gated-sources",
            "tsl51",
            "--json-out",
            str(report_path),
        ]
    )

    assert code == 0
    assert json.loads(report_path.read_text(encoding="utf-8"))["passed"] is True
