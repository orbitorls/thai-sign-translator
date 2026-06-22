from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import re
import zipfile

from scripts.prepare_kaggle_pose_t5_assets import main


def test_prepare_kaggle_pose_t5_assets_stages_code_and_portable_dataset(tmp_path, monkeypatch, capsys):
    repo_root = tmp_path / "repo"
    (repo_root / "src" / "tsl").mkdir(parents=True)
    (repo_root / "scripts").mkdir()
    (repo_root / "src" / "tsl" / "__init__.py").write_text("", encoding="utf-8")
    (repo_root / "scripts" / "kaggle_train_pose_t5.py").write_text("print('ok')\n", encoding="utf-8")
    (repo_root / "requirements.txt").write_text("pandas\n", encoding="utf-8")
    (repo_root / "config.py").write_text("VALUE = 1\n", encoding="utf-8")
    (repo_root / "README.md").write_text("# repo\n", encoding="utf-8")

    mixed_root = repo_root / "data" / "mixed_all_train_v6"
    mixed_root.mkdir(parents=True)
    feature_path = mixed_root / "source.npy"
    np.save(feature_path, np.ones((2, 312), dtype=np.float32))
    pd.DataFrame(
        [
            {
                "segment_id": "seg001",
                "npy_path": str(feature_path),
                "text": "hello",
                "split": "train",
                "source": "tsl51",
                "feature_layout_version": "v3-312",
            }
        ]
    ).to_csv(mixed_root / "manifest.csv", index=False)

    monkeypatch.chdir(repo_root)
    rc = main(
        [
            "--repo-root",
            str(repo_root),
            "--staging-root",
            "kaggle_upload",
        ]
    )

    assert rc == 0
    staged_code = repo_root / "kaggle_upload" / "thai-sign-code"
    staged_data = repo_root / "kaggle_upload" / "thai-sign-mixed-all-v6-archived"
    assert (staged_code / "repo_bundle.zip").is_file()
    assert (staged_data / "manifest.csv").is_file()
    manifest = pd.read_csv(staged_data / "manifest.csv")
    assert re.fullmatch(r"seg_00000_[0-9a-f]{12}\.npy", manifest.loc[0, "npy_path"])
    assert (staged_data / manifest.loc[0, "npy_path"]).is_file()
    assert not (staged_data / "landmarks").exists()

    code_metadata = json.loads((staged_code / "dataset-metadata.json").read_text(encoding="utf-8"))
    data_metadata = json.loads((staged_data / "dataset-metadata.json").read_text(encoding="utf-8"))
    assert code_metadata["id"] == "orbitorls/thai-sign-code"
    assert data_metadata["id"] == "orbitorls/thai-sign-mixed-all-v6-archived"

    output = capsys.readouterr().out
    assert '"portable_rows": 1' in output
    assert '"feature_layout": "flat"' in output


def test_prepare_kaggle_pose_t5_assets_can_archive_flat_features(tmp_path, monkeypatch, capsys):
    repo_root = tmp_path / "repo"
    (repo_root / "src" / "tsl").mkdir(parents=True)
    (repo_root / "scripts").mkdir()
    (repo_root / "src" / "tsl" / "__init__.py").write_text("", encoding="utf-8")
    (repo_root / "scripts" / "kaggle_train_pose_t5.py").write_text("print('ok')\n", encoding="utf-8")
    (repo_root / "requirements.txt").write_text("pandas\n", encoding="utf-8")
    (repo_root / "config.py").write_text("VALUE = 1\n", encoding="utf-8")
    (repo_root / "README.md").write_text("# repo\n", encoding="utf-8")

    mixed_root = repo_root / "data" / "mixed_all_train_v6"
    mixed_root.mkdir(parents=True)
    feature_path = mixed_root / "source.npy"
    np.save(feature_path, np.ones((2, 312), dtype=np.float32))
    pd.DataFrame(
        [
            {
                "segment_id": "seg001",
                "npy_path": str(feature_path),
                "text": "hello",
                "split": "train",
                "source": "tsl51",
                "feature_layout_version": "v3-312",
            }
        ]
    ).to_csv(mixed_root / "manifest.csv", index=False)

    monkeypatch.chdir(repo_root)
    rc = main(
        [
            "--repo-root",
            str(repo_root),
            "--staging-root",
            "kaggle_upload",
            "--archive-features",
        ]
    )

    assert rc == 0
    staged_data = repo_root / "kaggle_upload" / "thai-sign-mixed-all-v6-archived"
    assert (staged_data / "features.zip").is_file()
    assert not list(staged_data.glob("seg_*.npy"))
    manifest = pd.read_csv(staged_data / "manifest.csv")
    assert manifest.loc[0, "npy_path"].startswith("features/")
    with zipfile.ZipFile(staged_data / "features.zip", "r") as archive:
        assert archive.namelist() == [Path(manifest.loc[0, "npy_path"]).name]
    output = capsys.readouterr().out
    assert '"features_archive": "features.zip"' in output
    assert '"archived_features": 1' in output


def test_prepare_kaggle_pose_t5_assets_can_rebuild_mixed_manifest_before_staging(tmp_path, monkeypatch, capsys):
    repo_root = tmp_path / "repo"
    (repo_root / "src" / "tsl").mkdir(parents=True)
    (repo_root / "scripts").mkdir()
    (repo_root / "src" / "tsl" / "__init__.py").write_text("", encoding="utf-8")
    (repo_root / "scripts" / "kaggle_train_pose_t5.py").write_text("print('ok')\n", encoding="utf-8")
    (repo_root / "requirements.txt").write_text("pandas\n", encoding="utf-8")
    (repo_root / "config.py").write_text("VALUE = 1\n", encoding="utf-8")
    (repo_root / "README.md").write_text("# repo\n", encoding="utf-8")

    def _write_source_manifest(root: Path, source: str, rows: list[tuple[str, str, str]]) -> None:
        root.mkdir(parents=True, exist_ok=True)
        manifest_rows = []
        for idx, (split, text, video_id) in enumerate(rows):
            seg_id = f"{source}_{idx:03d}"
            feature_path = root / "landmarks" / f"{seg_id}.npy"
            feature_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(feature_path, np.ones((2, 312), dtype=np.float32))
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

    _write_source_manifest(
        repo_root / "data" / "tsl51_v3",
        "tsl51",
        [("train", "alpha", "t1"), ("val", "alpha", "t2")],
    )
    _write_source_manifest(
        repo_root / "data" / "thaisignvis_v3_probe",
        "thaisignvis",
        [("train", "beta", "v1"), ("val", "gamma", "v2")],
    )

    monkeypatch.chdir(repo_root)
    rc = main(
        [
            "--repo-root",
            str(repo_root),
            "--mixed-source-root",
            "data/mixed_all_train_v6",
            "--build-mixed-manifest",
            "true",
            "--mixed-data-roots",
            "data/tsl51_v3,data/thaisignvis_v3_probe",
            "--staging-root",
            "kaggle_upload",
        ]
    )

    assert rc == 0
    rebuilt_root = repo_root / "data" / "mixed_all_train_v6"
    rebuilt_manifest = pd.read_csv(rebuilt_root / "manifest.csv")
    assert len(rebuilt_manifest) == 4
    assert set(rebuilt_manifest.loc[rebuilt_manifest["source"] == "thaisignvis", "split"]) == {"train"}
    output = capsys.readouterr().out
    assert '"mixed_manifest_summary"' in output


def test_kaggle_notebook_targets_mixed_all_v6_assets():
    repo_root = Path(__file__).resolve().parents[2]
    metadata = json.loads(
        (repo_root / "kaggle_upload" / "notebook" / "kernel-metadata.json").read_text(encoding="utf-8")
    )
    notebook = json.loads(
        (repo_root / "kaggle_upload" / "notebook" / "notebook.ipynb").read_text(encoding="utf-8")
    )
    notebook_text = json.dumps(notebook, ensure_ascii=False)

    assert metadata["id"] == "orbitorls/thai-sign-mixed-all-v6-train"
    assert "orbitorls/thai-sign-mixed-all-v6-archived" in metadata["dataset_sources"]
    assert "thai-sign-mixed-all-v6-archived" in notebook_text
    assert "pose_t5_mixed_all_v6" in notebook_text
    assert "data/mixed_all_train_v6" in notebook_text
    assert "'--checkpoint-steps', '200'" in notebook_text
    assert "'--lr', '5e-5'" in notebook_text
    assert "'--early-stopping-patience', '6'" in notebook_text
    assert "thai-sign-mixed-all-v5" not in notebook_text
    assert "pose_t5_mixed_all_v5" not in notebook_text
