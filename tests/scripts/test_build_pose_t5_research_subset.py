from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from scripts.build_pose_t5_research_subset import _build_parser, _build_subset, main


def _write_manifest(root, source: str, rows: list[tuple[str, str, str]]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    manifest_rows = []
    for idx, (split, text, video_id) in enumerate(rows):
        seg_id = f"{source}_{idx:03d}"
        npy_path = root / "landmarks" / f"{seg_id}.npy"
        npy_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(npy_path, np.ones((4, 312), dtype=np.float32))
        manifest_rows.append(
            {
                "segment_id": seg_id,
                "npy_path": f"landmarks/{seg_id}.npy",
                "text": text,
                "video_id": video_id,
                "split": split,
                "source": source,
                "feature_layout_version": "v3-312",
            }
        )
    pd.DataFrame(manifest_rows).to_csv(root / "manifest.csv", index=False)


def test_build_pose_t5_research_subset_writes_absolute_manifest_and_quality(tmp_path):
    tsl51_root = tmp_path / "tsl51_v3"
    thaisignvis_root = tmp_path / "thaisignvis_v3_probe"
    youtube_root = tmp_path / "youtube_sl25_thai_v3"
    out_dir = tmp_path / "mixed_research_v3"

    _write_manifest(
        tsl51_root,
        "tsl51",
        [("train", "alpha", "v1"), ("train", "alpha", "v2"), ("val", "alpha", "v3")],
    )
    _write_manifest(thaisignvis_root, "thaisignvis", [("train", "beta", "v3"), ("val", "beta", "v4")])
    _write_manifest(
        youtube_root,
        "youtube_sl25_thai",
        [("train", "repeat me", "v5"), ("train", "repeat me", "v6"), ("val", "repeat me", "v7")],
    )

    code = main(
        [
            "--data-roots",
            f"{tsl51_root},{thaisignvis_root},{youtube_root}",
            "--out-dir",
            str(out_dir),
            "--max-thaisignvis-train",
            "1",
            "--max-thaisignvis-val",
            "1",
            "--max-youtube-train",
            "2",
            "--max-youtube-val",
            "1",
        ]
    )

    assert code == 0
    manifest = pd.read_csv(out_dir / "manifest.csv")
    quality = json.loads((out_dir / "manifest_quality.json").read_text(encoding="utf-8"))
    assert manifest["npy_path"].map(lambda value: value.startswith(str(tmp_path))).all()
    assert set(manifest["source"]) == {"tsl51", "thaisignvis", "youtube_sl25_thai"}
    assert quality["overall"]["train_examples"] >= 1
    assert "passed" in quality
    assert quality["required_sources"] == ["tsl51", "thaisignvis"]


def test_build_pose_t5_research_subset_filters_unique_youtube_targets(tmp_path):
    tsl51_root = tmp_path / "tsl51_v3"
    youtube_root = tmp_path / "youtube_sl25_thai_v3"
    out_dir = tmp_path / "mixed_research_v3"

    _write_manifest(
        tsl51_root,
        "tsl51",
        [("train", "alpha", "v1"), ("train", "alpha", "v2"), ("val", "alpha", "v3")],
    )
    _write_manifest(
        youtube_root,
        "youtube_sl25_thai",
        [("train", "u1", "v3"), ("train", "u2", "v4"), ("val", "u3", "v5")],
    )

    code = main(
        [
            "--data-roots",
            f"{tsl51_root},{youtube_root}",
            "--out-dir",
            str(out_dir),
        ]
    )

    assert code == 0
    manifest = pd.read_csv(out_dir / "manifest.csv")
    assert set(manifest["source"]) == {"tsl51"}


def test_build_pose_t5_research_subset_reports_excluded_sources(tmp_path):
    tsl51_root = tmp_path / "tsl51_v3"
    youtube_root = tmp_path / "youtube_sl25_thai_v3"
    out_dir = tmp_path / "mixed_readiness_v4"

    _write_manifest(
        tsl51_root,
        "tsl51",
        [("train", "alpha", "v1"), ("train", "alpha", "v2"), ("val", "alpha", "v3")],
    )
    _write_manifest(
        youtube_root,
        "youtube_sl25_thai",
        [("train", "u1", "v3"), ("train", "u2", "v4"), ("val", "u3", "v5")],
    )

    args = _build_parser().parse_args(
        [
            "--data-roots",
            f"{tsl51_root},{youtube_root}",
            "--out-dir",
            str(out_dir),
        ]
    )
    summary = _build_subset(args)

    assert summary["included_sources"] == ["tsl51"]
    assert summary["excluded_sources"] == ["youtube_sl25_thai"]
    assert "min_youtube_target_count=2" in summary["exclusion_reasons"]["youtube_sl25_thai"]
    assert summary["available_source_counts"]["youtube_sl25_thai"] == 3
    assert summary["val_source_counts"]["tsl51"] == 1


def test_build_pose_t5_research_subset_can_gate_manifest_quality_sources(tmp_path):
    tsl51_root = tmp_path / "tsl51_v3"
    thaisignvis_root = tmp_path / "thaisignvis_v3_probe"
    out_dir = tmp_path / "mixed_train_v5"

    _write_manifest(
        tsl51_root,
        "tsl51",
        [("train", "alpha", "v1"), ("train", "alpha", "v2"), ("val", "alpha", "v3")],
    )
    _write_manifest(
        thaisignvis_root,
        "thaisignvis",
        [("train", "beta", "tv1"), ("val", "gamma", "tv2")],
    )

    args = _build_parser().parse_args(
        [
            "--data-roots",
            f"{tsl51_root},{thaisignvis_root}",
            "--out-dir",
            str(out_dir),
            "--max-thaisignvis-train",
            "1",
            "--max-thaisignvis-val",
            "1",
            "--manifest-quality-sources",
            "tsl51",
        ]
    )
    summary = _build_subset(args)

    assert summary["quality_passed"] is True
    quality = json.loads((out_dir / "manifest_quality.json").read_text(encoding="utf-8"))
    assert quality["gated_sources"] == ["tsl51"]


def test_build_pose_t5_research_subset_can_route_thaisignvis_val_to_train(tmp_path):
    tsl51_root = tmp_path / "tsl51_v3"
    thaisignvis_root = tmp_path / "thaisignvis_v3_probe"
    out_dir = tmp_path / "mixed_research_v3"

    _write_manifest(tsl51_root, "tsl51", [("train", "alpha", "v1"), ("val", "alpha", "v2")])
    _write_manifest(
        thaisignvis_root,
        "thaisignvis",
        [("train", "beta", "tv1"), ("val", "gamma", "tv1")],
    )

    code = main(
        [
            "--data-roots",
            f"{tsl51_root},{thaisignvis_root}",
            "--out-dir",
            str(out_dir),
            "--thaisignvis-train-only",
            "true",
        ]
    )

    assert code == 0
    manifest = pd.read_csv(out_dir / "manifest.csv")
    thaisignvis = manifest.loc[manifest["source"] == "thaisignvis"]
    assert set(thaisignvis["split"]) == {"train"}


def test_build_pose_t5_research_subset_rejects_train_only_readiness_dataset(tmp_path):
    tsl51_root = tmp_path / "tsl51_v3"
    thaisignvis_root = tmp_path / "thaisignvis_v3_probe"
    out_dir = tmp_path / "mixed_research_v3"

    _write_manifest(tsl51_root, "tsl51", [("train", "alpha", "v1"), ("val", "alpha", "v2")])
    _write_manifest(
        thaisignvis_root,
        "thaisignvis",
        [("train", "beta", "tv1"), ("val", "gamma", "tv2")],
    )

    with pytest.raises(ValueError, match="readiness datasets cannot use"):
        main(
            [
                "--data-roots",
                f"{tsl51_root},{thaisignvis_root}",
                "--out-dir",
                str(out_dir),
                "--dataset-role",
                "readiness",
                "--thaisignvis-train-only",
                "true",
            ]
        )
