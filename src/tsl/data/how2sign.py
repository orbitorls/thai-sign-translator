from __future__ import annotations

import os

import numpy as np
import pandas as pd

from tsl.data.manifest import SignTextExample

_H2S_FEATURE_DIM = 411  # 137 keypoints * 3 (x, y, confidence)
_H2S_INPUT_DIM = 411
_H2S_SPLITS = ("train", "val", "test")


def _split_to_subdir(split: str) -> str:
    mapping = {"train": "train", "val": "validation", "test": "test"}
    return mapping[split]


def load_how2sign_manifest(
    data_root: str,
    split: str = "train",
    csv_subdir: str = "text/en/raw_text/re_aligned",
    keypoints_subdir: str = "rgb_front/features/openpose_output_fps_25/json",
) -> list[SignTextExample]:
    if split not in _H2S_SPLITS:
        raise ValueError(f"split must be one of {_H2S_SPLITS}, got {split!r}")
    split_subdir = _split_to_subdir(split)

    csv_path = os.path.join(data_root, "sentence_level", split_subdir, csv_subdir)
    csv_candidates = [
        os.path.join(csv_path, f"how2sign_realigned_{split}.csv"),
        os.path.join(csv_path, f"how2sign_{split}.csv"),
    ]
    csv_file = None
    for candidate in csv_candidates:
        if os.path.isfile(candidate):
            csv_file = candidate
            break
    if csv_file is None:
        raise FileNotFoundError(
            f"no CSV found at {csv_path!r} (tried {csv_candidates})"
        )

    kp_dir = os.path.join(data_root, "sentence_level", split_subdir, keypoints_subdir)
    if not os.path.isdir(kp_dir):
        raise FileNotFoundError(
            f"keypoints directory not found: {kp_dir!r}"
        )

    df = pd.read_csv(csv_file)
    out: list[SignTextExample] = []
    for _, row in df.iterrows():
        sentence_name = str(row.get("SENTENCE_NAME", ""))
        sentence_text = str(row.get("SENTENCE", ""))
        if not sentence_name or not sentence_text:
            continue
        kp_path = os.path.join(kp_dir, f"{sentence_name}.npy")
        if not os.path.isfile(kp_path):
            continue
        out.append(
            SignTextExample(
                example_id=sentence_name,
                source="how2sign",
                split=split,
                features_path=kp_path,
                target_text=sentence_text,
                metadata={"dataset": "how2sign", "split": split},
            )
        )
    return out


def load_how2sign_keypoints(npy_path: str) -> np.ndarray:
    arr = np.load(npy_path)
    if arr.ndim == 2 and arr.shape[1] == _H2S_FEATURE_DIM:
        return arr.astype(np.float32)
    if arr.ndim == 3 and arr.shape[1] == 137 and arr.shape[2] == 3:
        T = arr.shape[0]
        return arr.reshape(T, _H2S_FEATURE_DIM).astype(np.float32)
    raise ValueError(
        f"unexpected keypoint shape {arr.shape} in {npy_path!r}; "
        f"expected (T, 137, 3) or (T, {_H2S_FEATURE_DIM})"
    )


__all__ = [
    "_H2S_FEATURE_DIM",
    "_H2S_INPUT_DIM",
    "load_how2sign_manifest",
    "load_how2sign_keypoints",
]
