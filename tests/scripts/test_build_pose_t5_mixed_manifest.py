from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import pytest

from scripts.build_pose_t5_mixed_manifest import build_mixed_manifest, main


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


def test_build_pose_t5_mixed_manifest_keeps_all_examples_and_relabels_train_only(tmp_path):
    tsl51_root = tmp_path / "tsl51_v3"
    thaisignvis_root = tmp_path / "thaisignvis_v3_probe"
    youtube_root = tmp_path / "youtube_sl25_thai_v3"
    out_dir = tmp_path / "mixed_all_train_v6"

    _write_manifest(
        tsl51_root,
        "tsl51",
        [("train", "alpha", "t1"), ("val", "alpha", "t2")],
    )
    _write_manifest(
        thaisignvis_root,
        "thaisignvis",
        [("train", "beta", "v1"), ("val", "gamma", "v2")],
    )
    _write_manifest(
        youtube_root,
        "youtube_sl25_thai",
        [("train", "delta", "y1"), ("val", "epsilon", "y2")],
    )

    rc = main(
        [
            "--data-roots",
            f"{tsl51_root},{thaisignvis_root},{youtube_root}",
            "--out-dir",
            str(out_dir),
        ]
    )

    assert rc == 0
    manifest = pd.read_csv(out_dir / "manifest.csv")
    assert len(manifest) == 6
    assert set(manifest["source"]) == {"tsl51", "thaisignvis", "youtube_sl25_thai"}
    thaisignvis_splits = set(manifest.loc[manifest["source"] == "thaisignvis", "split"])
    youtube_splits = set(manifest.loc[manifest["source"] == "youtube_sl25_thai", "split"])
    assert thaisignvis_splits == {"train"}
    assert youtube_splits == {"train"}
    quality = json.loads((out_dir / "manifest_quality.json").read_text(encoding="utf-8"))
    assert quality["gated_sources"] == ["tsl51"]


def test_build_pose_t5_mixed_manifest_can_report_missing_roots(tmp_path):
    tsl51_root = tmp_path / "tsl51_v3"
    out_dir = tmp_path / "mixed_all_train_v6"
    _write_manifest(
        tsl51_root,
        "tsl51",
        [("train", "alpha", "t1"), ("val", "alpha", "t2")],
    )

    summary = build_mixed_manifest(
        argparse.Namespace(
            data_roots=f"{tsl51_root},{tmp_path / 'missing_root'}",
            out_dir=str(out_dir),
            train_only_sources="",
            required_sources="tsl51",
            manifest_quality_sources="tsl51",
            allow_missing_roots="true",
        )
    )

    assert len(summary["missing_roots"]) == 1
    assert summary["selected_source_counts"] == {"tsl51": 2}


def test_build_pose_t5_mixed_manifest_can_fail_on_missing_roots(tmp_path):
    with pytest.raises(FileNotFoundError, match="dataset root not found"):
        build_mixed_manifest(
            argparse.Namespace(
                data_roots=str(tmp_path / "missing_root"),
                out_dir=str(tmp_path / "out"),
                train_only_sources="",
                required_sources="tsl51",
                manifest_quality_sources="tsl51",
                allow_missing_roots="false",
            )
        )
