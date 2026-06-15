from __future__ import annotations

import json

import numpy as np
import pytest

from tsl.data.tsl_one import (
    _TSL_ONE_FEATURE_DIM,
    load_tsl_one_keypoints,
    load_tsl_one_manifest,
)


def _write_synthetic_tsl_one(tmp_path, split: str = "train", n: int = 2):
    entries = []
    for i in range(n):
        arr = np.random.randn(4, 44, 3).astype(np.float32)
        npy_name = f"word_{i}.npy"
        np.save(str(tmp_path / npy_name), arr)
        entries.append({"id": str(i), "gloss": f"คำ{i}", "keypoints_path": npy_name})
    json_path = tmp_path / f"{split}_words.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(entries, f)


def test_load_tsl_one_manifest_returns_examples(tmp_path):
    _write_synthetic_tsl_one(tmp_path, split="train", n=3)
    examples = load_tsl_one_manifest(str(tmp_path), split="train")
    assert len(examples) == 3
    for ex in examples:
        assert ex.source == "tsl_one"


def test_load_tsl_one_keypoints_flattens(tmp_path):
    arr = np.random.randn(6, 44, 3).astype(np.float32)
    path = tmp_path / "test.npy"
    np.save(str(path), arr)
    loaded = load_tsl_one_keypoints(str(path))
    assert loaded.shape == (6, _TSL_ONE_FEATURE_DIM)


def test_load_tsl_one_manifest_skips_missing_keypoints(tmp_path):
    entries = [{"id": "0", "gloss": "hello", "keypoints_path": "nonexistent.npy"}]
    json_path = tmp_path / "train_words.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(entries, f)
    examples = load_tsl_one_manifest(str(tmp_path), split="train")
    assert len(examples) == 0


def test_load_tsl_one_manifest_handles_alt_json_name(tmp_path):
    entries = [{"id": "0", "gloss": "hello", "keypoints_path": "nonexistent.npy"}]
    json_path = tmp_path / "train.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(entries, f)
    examples = load_tsl_one_manifest(str(tmp_path), split="train")
    assert len(examples) == 0  # no .npy files exist


def test_load_tsl_one_manifest_no_json_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_tsl_one_manifest(str(tmp_path), split="train")
