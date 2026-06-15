from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tsl.data.how2sign import (
    _H2S_FEATURE_DIM,
    load_how2sign_keypoints,
    load_how2sign_manifest,
)


def _write_synthetic_how2sign(tmp_path, n_per_split: int = 2):
    for split, split_dir in [("train", "train"), ("val", "validation")]:
        csv_dir = tmp_path / "sentence_level" / split_dir / "text/en/raw_text/re_aligned"
        csv_dir.mkdir(parents=True, exist_ok=True)

        kp_dir = tmp_path / "sentence_level" / split_dir / "rgb_front/features/openpose_output_fps_25/json"
        kp_dir.mkdir(parents=True, exist_ok=True)

        rows = []
        for i in range(n_per_split):
            name = f"sent_{split}_{i}"
            rows.append({"SENTENCE_NAME": name, "SENTENCE": f"hello world {i}"})
            arr = np.random.randn(5, 137, 3).astype(np.float32)
            np.save(str(kp_dir / f"{name}.npy"), arr)

        pd.DataFrame(rows, columns=["SENTENCE_NAME", "SENTENCE"]).to_csv(
            csv_dir / f"how2sign_realigned_{split}.csv", index=False
        )


def test_load_how2sign_manifest_returns_examples(tmp_path):
    _write_synthetic_how2sign(tmp_path, n_per_split=2)
    examples = load_how2sign_manifest(str(tmp_path), split="train")
    assert len(examples) == 2
    for ex in examples:
        assert ex.source == "how2sign"
        assert ex.split == "train"
        assert ex.target_text.startswith("hello world")


def test_load_how2sign_keypoints_flattens_3d(tmp_path):
    arr = np.random.randn(7, 137, 3).astype(np.float32)
    path = tmp_path / "test_kp.npy"
    np.save(str(path), arr)
    loaded = load_how2sign_keypoints(str(path))
    assert loaded.shape == (7, _H2S_FEATURE_DIM)


def test_load_how2sign_keypoints_accepts_flat(tmp_path):
    arr = np.random.randn(7, _H2S_FEATURE_DIM).astype(np.float32)
    path = tmp_path / "test_kp_flat.npy"
    np.save(str(path), arr)
    loaded = load_how2sign_keypoints(str(path))
    assert loaded.shape == (7, _H2S_FEATURE_DIM)


def test_load_how2sign_manifest_skips_missing_files(tmp_path):
    csv_dir = tmp_path / "sentence_level" / "train" / "text/en/raw_text/re_aligned"
    csv_dir.mkdir(parents=True, exist_ok=True)
    kp_dir = tmp_path / "sentence_level" / "train" / "rgb_front/features/openpose_output_fps_25/json"
    kp_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        [{"SENTENCE_NAME": "missing", "SENTENCE": "ghost"}],
        columns=["SENTENCE_NAME", "SENTENCE"],
    ).to_csv(csv_dir / "how2sign_realigned_train.csv", index=False)

    examples = load_how2sign_manifest(str(tmp_path), split="train")
    assert len(examples) == 0


def test_load_how2sign_manifest_invalid_split():
    with pytest.raises(ValueError, match="split must be one of"):
        load_how2sign_manifest("/fake", split="invalid")


def test_load_how2sign_keypoints_unexpected_shape(tmp_path):
    arr = np.random.randn(7, 10, 10, 3).astype(np.float32)
    path = tmp_path / "bad_kp.npy"
    np.save(str(path), arr)
    with pytest.raises(ValueError, match="unexpected keypoint shape"):
        load_how2sign_keypoints(str(path))
