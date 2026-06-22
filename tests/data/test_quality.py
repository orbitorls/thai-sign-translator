from __future__ import annotations

import math

import numpy as np

from tsl.data.manifest import SignTextExample
from tsl.data.quality import audit_dataset_splits


def _save_npy(path, arr: np.ndarray) -> str:
    np.save(path, arr.astype(np.float32))
    return str(path)


def test_audit_dataset_splits_reports_overlap_and_oov(tmp_path):
    train_a = _save_npy(tmp_path / "train_a.npy", np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
    train_b = _save_npy(tmp_path / "train_b.npy", np.array([[np.nan, 1.0]], dtype=np.float32))
    train_c = _save_npy(tmp_path / "train_c.npy", np.zeros((0, 2), dtype=np.float32))
    val_a = _save_npy(tmp_path / "val_a.npy", np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32))
    val_b = _save_npy(tmp_path / "val_b.npy", np.array([[np.inf, 9.0]], dtype=np.float32))

    splits = {
        "train": [
            SignTextExample(
                example_id="train_1",
                source="youtube_sl25",
                split="train",
                features_path=train_a,
                target_text="hello world",
                metadata={"video_id": "video_1"},
            ),
            SignTextExample(
                example_id="dup_id",
                source="youtube_sl25",
                split="train",
                features_path=train_b,
                target_text="hello world",
                metadata={"video_id": "video_2"},
            ),
            SignTextExample(
                example_id="train_3",
                source="thaisignvis",
                split="train",
                features_path=train_c,
                target_text="see you",
                metadata={"video_id": "shared_video"},
            ),
        ],
        "val": [
            SignTextExample(
                example_id="val_1",
                source="youtube_sl25",
                split="val",
                features_path=val_a,
                target_text="see you",
                metadata={"video_id": "video_4"},
            ),
            SignTextExample(
                example_id="dup_id",
                source="thaisignvis",
                split="val",
                features_path=val_b,
                target_text="hello moon",
                metadata={"video_id": "shared_video"},
            ),
        ],
        "test": [],
    }

    report = audit_dataset_splits(splits, load_features=np.load)

    assert report["source_counts"] == {"youtube_sl25": 3, "thaisignvis": 2}
    assert report["split_counts"] == {"train": 3, "val": 2, "test": 0}
    assert report["split_overlap"]["example_ids"] == ["dup_id"]
    assert report["video_id_leakage"]["count"] == 1
    assert report["video_id_leakage"]["video_ids"] == ["shared_video"]
    assert math.isclose(report["target_uniqueness_ratio"], 0.6)
    assert math.isclose(report["repeated_target_coverage"], 0.8)
    assert report["train_val_target_overlap"]["shared_targets"] == ["see you"]
    assert math.isclose(report["train_only_oov_rate"], 0.25)

    feature_stats = report["feature_stats"]
    assert feature_stats["sequences_scanned"] == 5
    assert feature_stats["empty_sequences"] == 1
    assert feature_stats["nan_sequences"] == 1
    assert feature_stats["inf_sequences"] == 1
    assert feature_stats["nonfinite_values"] == 2
    assert feature_stats["total_frames"] == 6
