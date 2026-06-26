from __future__ import annotations

import pandas as pd

from scripts.extract_dataset_keypoints import _make_manifest_row, extract_segments, load_segments


def test_load_segments_reads_required_columns_and_resolves_relative_video_paths(tmp_path):
    csv_path = tmp_path / "segments.csv"
    video_path = tmp_path / "videos" / "sample.mp4"
    video_path.parent.mkdir()
    video_path.write_bytes(b"fake")
    pd.DataFrame(
        [
            {
                "segment_id": "seg_001",
                "video_path": "videos/sample.mp4",
                "text": "สวัสดี",
                "start_ms": 0,
                "end_ms": 1200,
                "split": "train",
            }
        ]
    ).to_csv(csv_path, index=False)

    rows = load_segments(str(csv_path))

    assert rows == [
        {
            "segment_id": "seg_001",
            "video_path": str(video_path.resolve()),
            "video_id": "sample",
            "text": "สวัสดี",
            "start_ms": 0.0,
            "end_ms": 1200.0,
            "split": "train",
        }
    ]


def test_make_manifest_row_uses_existing_contract():
    row = _make_manifest_row(
        {
            "segment_id": "seg_001",
            "video_id": "sample",
            "text": "สวัสดี",
            "start_ms": 0.0,
            "end_ms": 1200.0,
            "split": "train",
        },
        "landmarks/seg_001.npy",
        "custom",
    )

    assert row["feature_layout_version"] == "v3-312"
    assert row["npy_path"] == "landmarks/seg_001.npy"
    assert row["source"] == "custom"


def test_extract_segments_dry_run_does_not_create_output_dir(tmp_path):
    csv_path = tmp_path / "segments.csv"
    video_path = tmp_path / "videos" / "sample.mp4"
    video_path.parent.mkdir()
    video_path.write_bytes(b"fake")
    pd.DataFrame(
        [
            {
                "segment_id": "seg_001",
                "video_path": "videos/sample.mp4",
                "text": "สวัสดี",
                "start_ms": 0,
                "end_ms": 1200,
                "split": "train",
            }
        ]
    ).to_csv(csv_path, index=False)

    out_dir = tmp_path / "out"
    result = extract_segments(
        input_csv=str(csv_path),
        out_dir=str(out_dir),
        dry_run=True,
    )

    assert result["dry_run"] is True
    assert result["rows"] == 1
    assert not out_dir.exists()
