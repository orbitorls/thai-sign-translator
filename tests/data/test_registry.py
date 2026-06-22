from __future__ import annotations

import numpy as np
import pandas as pd

from tsl.data.registry import get_dataset_spec, list_dataset_specs, load_dataset_splits


def _write_tsl51_fixture(root, n: int = 10) -> None:
    meta_dir = root / "metadata"
    lm_dir = root / "landmarks" / "user_sentence"
    meta_dir.mkdir(parents=True, exist_ok=True)
    lm_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for i in range(n):
        video_id = f"video_{i:02d}"
        rel_path = f"landmarks/user_sentence/{video_id}.csv"
        rows.append(
            {
                "video_id": video_id,
                "sentence_id": i,
                "sentence_clean": f"sentence {i}",
                "landmark_path": rel_path,
                "video_path": f"videos/{video_id}.mp4",
            }
        )
        arr = np.full((3, 162), float(i), dtype=np.float32)
        pd.DataFrame(arr).to_csv(root / rel_path, index=False)

    pd.DataFrame(rows).to_csv(meta_dir / "sentence_metadata.csv", index=False)


def test_registry_lists_supported_datasets():
    specs = list_dataset_specs()
    names = {spec.name for spec in specs}
    assert {"tsl51", "how2sign", "thaisignvis", "youtube_sl25"} <= names

    youtube_spec = get_dataset_spec("youtube_sl25")
    assert youtube_spec.source == "youtube_sl25"
    assert youtube_spec.input_dim == 162
    assert youtube_spec.split_policy == "manifest"
    assert youtube_spec.schema_version == "manifest.csv.v1"
    assert youtube_spec.license_name != ""
    assert youtube_spec.provenance != ""


def test_load_dataset_splits_applies_registry_policy(tmp_path):
    _write_tsl51_fixture(tmp_path, n=10)

    splits = load_dataset_splits("tsl51", str(tmp_path), seed=7)

    assert len(splits["train"]) == 9
    assert len(splits["val"]) == 1
    assert splits["test"] == []
    assert all(example.source == "tsl51" for example in splits["train"] + splits["val"])
