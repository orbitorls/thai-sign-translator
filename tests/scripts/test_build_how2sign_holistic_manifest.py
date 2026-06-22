from __future__ import annotations

import json

import numpy as np
import pandas as pd

from scripts.build_how2sign_holistic_manifest import build_manifest, main


def _write_how2sign_holistic_fixture(root):
    dataset_root = root / "how2sign_holistic_features"
    metadata_root = dataset_root / "metadata"
    metadata_root.mkdir(parents=True, exist_ok=True)

    for split in ("train", "val", "test"):
        feature_root = dataset_root / split / "frontal"
        feature_root.mkdir(parents=True, exist_ok=True)
        rows = []
        for idx in range(2):
            sentence_name = f"{split}_sent_{idx}"
            rows.append(
                {
                    "VIDEO_ID": f"{split}_video",
                    "VIDEO_NAME": f"{split}_video_name",
                    "SENTENCE_ID": f"{split}_{idx}",
                    "SENTENCE_NAME": sentence_name,
                    "START_REALIGNED": float(idx),
                    "END_REALIGNED": float(idx + 1),
                    "SENTENCE": f"hello {split} {idx}",
                }
            )
            np.save(feature_root / f"{sentence_name}_holistic.npy", np.zeros((4, 543, 3), dtype=np.float32))
        pd.DataFrame(rows).to_csv(
            metadata_root / f"how2sign_realigned_{split}.csv",
            index=False,
            sep="\t",
        )
    return dataset_root


def test_build_manifest_writes_rows_for_all_splits(tmp_path):
    _write_how2sign_holistic_fixture(tmp_path)
    out_dir = tmp_path / "how2sign_manifest"

    summary = build_manifest(tmp_path, out_dir)

    manifest = pd.read_csv(out_dir / "manifest.csv")
    assert summary["rows"] == 6
    assert summary["split_counts"] == {"train": 2, "val": 2, "test": 2}
    assert set(manifest["split"]) == {"train", "val", "test"}
    assert set(manifest["feature_layout_version"]) == {"raw_mediapipe_543x3"}


def test_main_writes_summary_json(tmp_path, monkeypatch, capsys):
    _write_how2sign_holistic_fixture(tmp_path)
    out_dir = tmp_path / "out"
    monkeypatch.chdir(tmp_path)

    rc = main(["--input-root", str(tmp_path), "--out-dir", str(out_dir)])

    assert rc == 0
    summary = json.loads((out_dir / "build_summary.json").read_text(encoding="utf-8"))
    assert summary["rows"] == 6
    output = capsys.readouterr().out
    assert '"feature_layout_version": "raw_mediapipe_543x3"' in output
