from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tsl.data.manifest import SignTextExample
from tsl.data.tsl51 import (
    load_landmark_sequence,
    load_sentence_features,
    load_sentence_manifest,
)

_META_COLUMNS = ["video_id", "sentence_id", "sentence_clean", "landmark_path", "video_path"]


def _write_metadata(root, rows):
    meta_dir = root / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=_META_COLUMNS).to_csv(
        meta_dir / "sentence_metadata.csv", index=False
    )


def _empty_landmarks_dir(root):
    d = root / "landmarks" / "user_sentence"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _landmark_columns():
    cols = ["frame", "t_ms"]
    for i in range(54):
        cols.append(f"lm_{i}_x")
        cols.append(f"lm_{i}_y")
        cols.append(f"lm_{i}_z")
    return cols


def _landmark_row(t, with_nan_col=None):
    row = {"frame": t, "t_ms": t * 33}
    for i in range(54):
        if with_nan_col is not None and i == with_nan_col[0] and t == with_nan_col[1]:
            row[f"lm_{i}_x"] = np.nan
        else:
            row[f"lm_{i}_x"] = float(t * 100 + i)
        row[f"lm_{i}_y"] = float(t * 100 + i) + 0.1
        row[f"lm_{i}_z"] = float(t * 100 + i) + 0.2
    return row


def test_load_sentence_manifest_returns_examples(tmp_path):
    rows = [
        {
            "video_id": "v001",
            "sentence_id": 1,
            "sentence_clean": "สวัสดี",
            "landmark_path": "landmarks/user_sentence/v001.csv",
            "video_path": "videos/v001.mp4",
        },
        {
            "video_id": "v002",
            "sentence_id": 2,
            "sentence_clean": "ขอบคุณ",
            "landmark_path": "landmarks/user_sentence/v002.csv",
            "video_path": "videos/v002.mp4",
        },
    ]
    _write_metadata(tmp_path, rows)
    _empty_landmarks_dir(tmp_path)

    examples = load_sentence_manifest(str(tmp_path))

    assert len(examples) == 2
    assert all(isinstance(e, SignTextExample) for e in examples)
    first = examples[0]
    assert first.example_id == "v001"
    assert first.source == "tsl51"
    assert first.split == "train"
    assert first.target_text == "สวัสดี"
    assert Path(first.features_path) == tmp_path / "landmarks" / "user_sentence" / "v001.csv"
    assert first.metadata == {"sentence_id": 1, "video_id": "v001"}
    assert examples[1].example_id == "v002"
    assert examples[1].target_text == "ขอบคุณ"


def test_load_sentence_manifest_skips_empty_text(tmp_path):
    rows = [
        {
            "video_id": "v001",
            "sentence_id": 1,
            "sentence_clean": "สวัสดี",
            "landmark_path": "landmarks/user_sentence/v001.csv",
            "video_path": "videos/v001.mp4",
        },
        {
            "video_id": "v002",
            "sentence_id": 2,
            "sentence_clean": "",
            "landmark_path": "landmarks/user_sentence/v002.csv",
            "video_path": "videos/v002.mp4",
        },
    ]
    _write_metadata(tmp_path, rows)
    _empty_landmarks_dir(tmp_path)

    examples = load_sentence_manifest(str(tmp_path))

    assert len(examples) == 1
    assert examples[0].example_id == "v001"


def test_load_sentence_manifest_raises_when_metadata_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_sentence_manifest(str(tmp_path))


def test_load_landmark_sequence_shape_and_nan_handling(tmp_path):
    csv = tmp_path / "v001.csv"
    pd.DataFrame(
        [_landmark_row(0), _landmark_row(1, with_nan_col=(0, 1)), _landmark_row(2)],
        columns=_landmark_columns(),
    ).to_csv(csv, index=False)

    seq = load_landmark_sequence(str(csv))

    assert seq.shape == (3, 162)
    assert seq.dtype == np.float32
    assert not np.isnan(seq).any()
    assert not np.isinf(seq).any()
    # Column order is lm_0_x, lm_0_y, lm_0_z, lm_1_x, ... so index 0 = lm_0_x.
    # lm_0_x at t=1 was NaN, must be 0.0; t=0 and t=2 are real values.
    assert seq[1, 0] == 0.0
    assert seq[0, 0] == pytest.approx(0.0)
    assert seq[2, 0] == pytest.approx(200.0)
    # lm_0_y at t=0 is 0*100 + 0 + 0.1.
    assert seq[0, 1] == pytest.approx(0.1, abs=1e-5)
    # lm_1_x at t=1 is 1*100 + 1 = 101 (column 3).
    assert seq[1, 3] == pytest.approx(101.0)


def test_load_landmark_sequence_empty_file(tmp_path):
    csv = tmp_path / "v001.csv"
    pd.DataFrame(columns=["frame", "t_ms"]).to_csv(csv, index=False)

    seq = load_landmark_sequence(str(csv))

    assert seq.shape == (0, 162)
    assert seq.dtype == np.float32


def test_load_sentence_features_via_example(tmp_path):
    rows = [
        {
            "video_id": "v001",
            "sentence_id": 1,
            "sentence_clean": "สวัสดี",
            "landmark_path": "landmarks/user_sentence/v001.csv",
            "video_path": "videos/v001.mp4",
        },
    ]
    _write_metadata(tmp_path, rows)
    lm_dir = _empty_landmarks_dir(tmp_path)
    pd.DataFrame([_landmark_row(0)], columns=_landmark_columns()).to_csv(
        lm_dir / "v001.csv", index=False
    )

    examples = load_sentence_manifest(str(tmp_path))
    seq = load_sentence_features(examples[0])

    assert seq.shape == (1, 162)
    assert seq.dtype == np.float32
    assert seq[0, 0] == pytest.approx(0.0)
    # lm_53_z lives at column 53*3 + 2 = 161; value at t=0 is 53 + 0.2.
    assert seq[0, 53 * 3 + 2] == pytest.approx(53.2, abs=1e-5)
