"""Dataset registry for manifest-backed SLT datasets."""
from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Callable

import numpy as np

from tsl.data.how2sign import load_how2sign_keypoints, load_how2sign_manifest
from tsl.data.manifest import SignTextExample
from tsl.data.thaisignvis import load_thaisignvis_features, load_thaisignvis_manifest
from tsl.data.tsl51 import load_landmark_sequence, load_sentence_manifest
from tsl.data.youtube_sl25 import (
    load_youtube_sl25_features,
    load_youtube_sl25_manifest,
)

__all__ = [
    "DatasetSpec",
    "get_dataset_spec",
    "list_dataset_specs",
    "load_dataset_splits",
]

ManifestLoader = Callable[[str, str | None], list[SignTextExample]]
FeatureLoader = Callable[[str], np.ndarray]

_SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    source: str
    input_dim: int
    manifest_loader: ManifestLoader
    feature_loader: FeatureLoader
    split_policy: str
    schema_version: str
    license_name: str
    provenance: str


def _load_tsl51_manifest(data_root: str, split: str | None = None) -> list[SignTextExample]:
    examples = load_sentence_manifest(data_root)
    if split is None or split == "train":
        return examples
    return []


def _load_how2sign_manifest(data_root: str, split: str | None = None) -> list[SignTextExample]:
    if split is None:
        all_examples: list[SignTextExample] = []
        for split_name in _SPLITS:
            try:
                all_examples.extend(load_how2sign_manifest(data_root, split=split_name))
            except FileNotFoundError:
                continue
        return all_examples
    return load_how2sign_manifest(data_root, split=split)


_DATASET_SPECS = {
    "tsl51": DatasetSpec(
        name="tsl51",
        source="tsl51",
        input_dim=162,
        manifest_loader=_load_tsl51_manifest,
        feature_loader=load_landmark_sequence,
        split_policy="random_90_10",
        schema_version="sentence_metadata.csv.v1",
        license_name="CC BY-NC-SA 4.0",
        provenance="Namonpas/thai-sign-language-tsl51 sentence metadata and landmark CSVs",
    ),
    "how2sign": DatasetSpec(
        name="how2sign",
        source="how2sign",
        input_dim=411,
        manifest_loader=_load_how2sign_manifest,
        feature_loader=load_how2sign_keypoints,
        split_policy="official",
        schema_version="how2sign_realigned_csv.v1",
        license_name="How2Sign dataset terms",
        provenance="How2Sign sentence-level English text paired with OpenPose keypoints",
    ),
    "thaisignvis": DatasetSpec(
        name="thaisignvis",
        source="thaisignvis",
        input_dim=312,
        manifest_loader=load_thaisignvis_manifest,
        feature_loader=load_thaisignvis_features,
        split_policy="manifest",
        schema_version="manifest.csv.v1",
        license_name="Apache-2.0",
        provenance="ThaiSignVis extracted landmark manifest from Kaggle source videos",
    ),
    "youtube_sl25": DatasetSpec(
        name="youtube_sl25",
        source="youtube_sl25",
        input_dim=162,
        manifest_loader=load_youtube_sl25_manifest,
        feature_loader=load_youtube_sl25_features,
        split_policy="manifest",
        schema_version="manifest.csv.v1",
        license_name="YouTube-SL-25 research dataset terms",
        provenance="YouTube-SL-25 Thai subset exported to cached landmark npy files",
    ),
}


def list_dataset_specs() -> list[DatasetSpec]:
    return [
        _DATASET_SPECS[name]
        for name in sorted(_DATASET_SPECS)
    ]


def get_dataset_spec(name: str) -> DatasetSpec:
    try:
        return _DATASET_SPECS[name]
    except KeyError as exc:
        known = ", ".join(sorted(_DATASET_SPECS))
        raise KeyError(f"unknown dataset {name!r}; expected one of: {known}") from exc


def load_dataset_splits(
    name: str,
    data_root: str,
    seed: int = 42,
) -> dict[str, list[SignTextExample]]:
    spec = get_dataset_spec(name)
    if spec.split_policy == "random_90_10":
        examples = spec.manifest_loader(data_root, None)
        train_examples, val_examples = _split_examples(examples, seed=seed)
        return {"train": train_examples, "val": val_examples, "test": []}
    if spec.split_policy in {"manifest", "official"}:
        splits: dict[str, list[SignTextExample]] = {}
        for split_name in _SPLITS:
            try:
                splits[split_name] = spec.manifest_loader(data_root, split_name)
            except FileNotFoundError:
                splits[split_name] = []
        return splits
    raise ValueError(f"unsupported split policy: {spec.split_policy!r}")


def _split_examples(
    examples: list[SignTextExample],
    seed: int,
) -> tuple[list[SignTextExample], list[SignTextExample]]:
    rng = random.Random(seed)
    indices = list(range(len(examples)))
    rng.shuffle(indices)
    n_val = len(examples) // 10
    val_set = set(indices[:n_val])
    train_examples = [examples[i] for i in indices if i not in val_set]
    val_examples = [examples[i] for i in indices if i in val_set]
    return train_examples, val_examples
