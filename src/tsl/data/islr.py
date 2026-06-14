"""Google ISLR (asl-signs) parquet loader.

ISLR stores each clip as a long-format parquet with one row per
(frame, landmark) and columns [frame, type, landmark_index, x, y, z].
We pivot it into the canonical dense array (T, 543, 3) used everywhere
else in the project.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import torch

from tsl.features.normalize import normalize_sequence

_TYPE_OFFSETS = {
    "face": 0,
    "left_hand": 468,
    "pose": 489,
    "right_hand": 522,
}
_N_LANDMARKS = 543


def load_islr_sequence(parquet_path: str) -> np.ndarray:
    df = pd.read_parquet(parquet_path, engine="pyarrow")
    frames = sorted(df["frame"].unique().tolist())
    frame_to_t = {f: t for t, f in enumerate(frames)}
    n_frames = len(frames)
    seq = np.full((n_frames, _N_LANDMARKS, 3), np.nan, dtype=np.float32)
    offsets = df["type"].map(lambda t: _TYPE_OFFSETS[t])
    global_idx = (offsets + df["landmark_index"]).to_numpy()
    t_idx = df["frame"].map(lambda f: frame_to_t[f]).to_numpy()
    seq[t_idx, global_idx, 0] = df["x"].to_numpy(dtype=np.float32)
    seq[t_idx, global_idx, 1] = df["y"].to_numpy(dtype=np.float32)
    seq[t_idx, global_idx, 2] = df["z"].to_numpy(dtype=np.float32)
    return seq


class ISLRDataset(torch.utils.data.Dataset):
    """ISLR dataset reading a train.csv mapping (path, sign).

    Each item is (normalized (T, D) float32 tensor, label int). Labels are
    assigned by sorted unique sign name so ids are stable across runs.
    An optional ``classes`` subset restricts both rows and the label space.
    """

    def __init__(self, parquet_dir: str, csv_path: str, classes: list[str] | None = None):
        self.parquet_dir = parquet_dir
        df = pd.read_csv(csv_path)
        if classes is not None:
            df = df[df["sign"].isin(classes)].reset_index(drop=True)
            label_names = sorted(classes)
        else:
            label_names = sorted(df["sign"].unique().tolist())
        self.label_names: list[str] = label_names
        self._label_to_id = {name: i for i, name in enumerate(label_names)}
        self._paths: list[str] = df["path"].tolist()
        self._labels: list[int] = [self._label_to_id[s] for s in df["sign"]]

    @property
    def num_classes(self) -> int:
        return len(self.label_names)

    def __len__(self) -> int:
        return len(self._paths)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int]:
        abs_path = os.path.join(self.parquet_dir, self._paths[i])
        raw = load_islr_sequence(abs_path)
        norm = normalize_sequence(raw)
        x = torch.from_numpy(norm)
        return x, int(self._labels[i])
