from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import re

from scripts.package_portable_manifest_dataset import (
    build_archived_portable_dataset_dir,
    build_portable_bundle,
    build_portable_dataset_dir,
)


def test_build_portable_dataset_dir_rewrites_manifest_paths_and_copies_features(tmp_path):
    source_root = tmp_path / "mixed_dataset"
    source_root.mkdir()
    source_feature = tmp_path / "original.npy"
    np.save(source_feature, np.ones((3, 312), dtype=np.float32))
    pd.DataFrame(
        [
            {
                "segment_id": "seg001",
                "npy_path": str(source_feature),
                "text": "alpha",
                "split": "train",
                "source": "mixed",
                "feature_layout_version": "v3-312",
            }
        ]
    ).to_csv(source_root / "manifest.csv", index=False)

    output_dir = tmp_path / "portable_dir"
    summary = build_portable_dataset_dir(str(source_root), str(output_dir))

    assert summary["rows"] == 1
    feature_files = list((output_dir / "landmarks").glob("*.npy"))
    assert len(feature_files) == 1
    assert re.fullmatch(r"seg_00000_[0-9a-f]{12}\.npy", feature_files[0].name)
    manifest = pd.read_csv(output_dir / "manifest.csv")
    assert manifest.loc[0, "npy_path"] == f"landmarks/{feature_files[0].name}"


def test_build_portable_bundle_rewrites_manifest_paths_and_copies_features(tmp_path):
    source_root = tmp_path / "mixed_dataset"
    source_root.mkdir()
    source_feature = tmp_path / "original.npy"
    np.save(source_feature, np.ones((3, 312), dtype=np.float32))
    pd.DataFrame(
        [
            {
                "segment_id": "seg001",
                "npy_path": str(source_feature),
                "text": "alpha",
                "split": "train",
                "source": "mixed",
                "feature_layout_version": "v3-312",
            }
        ]
    ).to_csv(source_root / "manifest.csv", index=False)

    output_zip = tmp_path / "portable.zip"
    summary = build_portable_bundle(str(source_root), str(output_zip))

    assert summary["rows"] == 1
    with zipfile.ZipFile(output_zip, "r") as archive:
        names = set(archive.namelist())
        assert "manifest.csv" in names
        feature_name = next(name for name in names if name.startswith("landmarks/seg_"))
        assert re.fullmatch(r"landmarks/seg_00000_[0-9a-f]{12}\.npy", feature_name)
        manifest_text = archive.read("manifest.csv").decode("utf-8")
        assert feature_name in manifest_text


def test_build_portable_dataset_dir_flat_layout_keeps_features_at_root(tmp_path):
    source_root = tmp_path / "mixed_dataset"
    source_root.mkdir()
    source_feature = tmp_path / "original.npy"
    np.save(source_feature, np.ones((3, 312), dtype=np.float32))
    pd.DataFrame(
        [
            {
                "segment_id": "seg001",
                "npy_path": str(source_feature),
                "text": "alpha",
                "split": "train",
                "source": "mixed",
                "feature_layout_version": "v3-312",
            }
        ]
    ).to_csv(source_root / "manifest.csv", index=False)

    output_dir = tmp_path / "portable_dir"
    summary = build_portable_dataset_dir(str(source_root), str(output_dir), feature_layout="flat")

    assert summary["feature_layout"] == "flat"
    feature_files = list(output_dir.glob("seg_*.npy"))
    assert len(feature_files) == 1
    assert re.fullmatch(r"seg_00000_[0-9a-f]{12}\.npy", feature_files[0].name)
    assert not (output_dir / "landmarks").exists()
    manifest = pd.read_csv(output_dir / "manifest.csv")
    assert manifest.loc[0, "npy_path"] == feature_files[0].name


def test_build_portable_dataset_dir_can_overwrite_existing_output(tmp_path):
    source_root = tmp_path / "mixed_dataset"
    source_root.mkdir()
    source_feature = tmp_path / "original.npy"
    np.save(source_feature, np.ones((3, 312), dtype=np.float32))
    pd.DataFrame(
        [
            {
                "segment_id": "seg001",
                "npy_path": str(source_feature),
                "text": "alpha",
                "split": "train",
                "source": "mixed",
                "feature_layout_version": "v3-312",
            }
        ]
    ).to_csv(source_root / "manifest.csv", index=False)

    output_dir = tmp_path / "portable_dir"
    output_dir.mkdir()
    (output_dir / "stale.txt").write_text("old", encoding="utf-8")

    summary = build_portable_dataset_dir(str(source_root), str(output_dir), feature_layout="flat")

    assert summary["rows"] == 1
    assert not (output_dir / "stale.txt").exists()
    manifest = pd.read_csv(output_dir / "manifest.csv")
    assert manifest.loc[0, "npy_path"].startswith("seg_")


def test_build_archived_portable_dataset_dir_writes_features_zip_and_manifest(tmp_path):
    source_root = tmp_path / "mixed_dataset"
    source_root.mkdir()
    source_feature = tmp_path / "original.npy"
    np.save(source_feature, np.ones((3, 312), dtype=np.float32))
    pd.DataFrame(
        [
            {
                "segment_id": "seg001",
                "npy_path": str(source_feature),
                "text": "alpha",
                "split": "train",
                "source": "mixed",
                "feature_layout_version": "v3-312",
            }
        ]
    ).to_csv(source_root / "manifest.csv", index=False)

    output_dir = tmp_path / "archived_dir"
    summary = build_archived_portable_dataset_dir(str(source_root), str(output_dir))

    assert summary["feature_layout"] == "archive"
    assert summary["archive_name"] == "features.zip"
    manifest = pd.read_csv(output_dir / "manifest.csv")
    assert manifest.loc[0, "npy_path"].startswith("features/")
    with zipfile.ZipFile(output_dir / "features.zip", "r") as archive:
        assert archive.namelist() == [Path(manifest.loc[0, "npy_path"]).name]
