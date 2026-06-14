import numpy as np
import pandas as pd
import pytest

from tsl.data.islr import load_islr_sequence


def _write_fake_parquet(path, n_frames=3):
    type_counts = {"face": 468, "left_hand": 21, "pose": 33, "right_hand": 21}
    rows = []
    for f in range(n_frames):
        for ltype, count in type_counts.items():
            for li in range(count):
                rows.append({
                    "frame": f,
                    "type": ltype,
                    "landmark_index": li,
                    "x": float(f * 1000 + li),
                    "y": float(f * 1000 + li) + 0.5,
                    "z": float(f * 1000 + li) + 0.25,
                })
    df = pd.DataFrame(rows)
    df.to_parquet(path, engine="pyarrow")


def test_load_islr_sequence_shape_and_order(tmp_path):
    pq = tmp_path / "clip.parquet"
    _write_fake_parquet(pq, n_frames=3)
    seq = load_islr_sequence(str(pq))
    assert seq.shape == (3, 543, 3)
    assert seq.dtype == np.float32
    assert seq[0, 468, 0] == pytest.approx(0.0)
    assert seq[1, 489, 0] == pytest.approx(1000.0)
    assert seq[2, 542, 0] == pytest.approx(2020.0)
    assert seq[0, 0, 0] == pytest.approx(0.0)


def test_load_islr_sequence_missing_landmark_is_nan(tmp_path):
    pq = tmp_path / "clip.parquet"
    type_counts = {"face": 468, "left_hand": 21, "pose": 33, "right_hand": 21}
    rows = []
    for ltype, count in type_counts.items():
        for li in range(count):
            x = np.nan if ltype == "right_hand" else 1.0
            rows.append({
                "frame": 0,
                "type": ltype,
                "landmark_index": li,
                "x": x, "y": x, "z": x,
            })
    pd.DataFrame(rows).to_parquet(str(pq), engine="pyarrow")
    seq = load_islr_sequence(str(pq))
    assert seq.shape == (1, 543, 3)
    assert np.isnan(seq[0, 522:543, :]).all()
    assert not np.isnan(seq[0, 0:522, :]).any()
