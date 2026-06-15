from __future__ import annotations

import json
import os

import numpy as np

from tsl.data.manifest import SignTextExample

_TSL_ONE_FEATURE_DIM = 132  # 44 keypoints * 3 (x, y, z) as reported in the TSL-ONE-Pose paper


def load_tsl_one_manifest(
    data_root: str,
    split: str = "train",
) -> list[SignTextExample]:
    json_path = os.path.join(data_root, f"{split}.json")
    if not os.path.isfile(json_path):
        json_path = os.path.join(data_root, f"{split}_words.json")
    if not os.path.isfile(json_path):
        raise FileNotFoundError(
            f"no manifest JSON found in {data_root!r} for split={split!r}"
        )

    with open(json_path, "r", encoding="utf-8") as f:
        entries = json.load(f)

    out: list[SignTextExample] = []
    for entry in entries:
        npy_path = entry.get("keypoints_path") or entry.get("skeleton_path") or entry.get("path")
        if not npy_path:
            continue
        gloss = entry.get("gloss") or entry.get("text") or entry.get("label", "")
        if not gloss:
            continue
        full_path = os.path.join(data_root, npy_path)
        if not os.path.isfile(full_path):
            continue
        out.append(
            SignTextExample(
                example_id=entry.get("id", str(len(out))),
                source="tsl_one",
                split=split,
                features_path=full_path,
                target_text=str(gloss),
                metadata={"dataset": "tsl_one", "split": split},
            )
        )
    return out


def load_tsl_one_keypoints(npy_path: str) -> np.ndarray:
    arr = np.load(npy_path)
    if arr.ndim == 2 and arr.shape[1] == _TSL_ONE_FEATURE_DIM:
        return arr.astype(np.float32)
    if arr.ndim == 3 and arr.shape[2] == 3:
        T, N, _ = arr.shape
        return arr.reshape(T, N * 3).astype(np.float32)
    raise ValueError(
        f"unexpected keypoint shape {arr.shape} in {npy_path!r}; "
        f"expected (T, 44, 3) or (T, {_TSL_ONE_FEATURE_DIM})"
    )


__all__ = [
    "_TSL_ONE_FEATURE_DIM",
    "load_tsl_one_manifest",
    "load_tsl_one_keypoints",
]
