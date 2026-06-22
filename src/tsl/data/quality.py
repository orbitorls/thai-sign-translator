"""Dataset readiness and manifest-quality checks."""
from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from statistics import mean
from typing import Callable

import numpy as np

from tsl.data.manifest import SignTextExample

__all__ = ["audit_dataset_splits"]

FeatureLoader = Callable[[str], np.ndarray]
_SPLITS = ("train", "val", "test")


def audit_dataset_splits(
    splits: Mapping[str, Sequence[SignTextExample]],
    load_features: FeatureLoader | None = None,
) -> dict:
    normalized = {split: list(splits.get(split, ())) for split in _SPLITS}
    all_examples = [example for split in _SPLITS for example in normalized[split]]

    report = {
        "source_counts": dict(Counter(example.source for example in all_examples)),
        "split_counts": {split: len(normalized[split]) for split in _SPLITS},
        "split_overlap": _compute_split_overlap(normalized),
        "video_id_leakage": _compute_video_id_leakage(normalized),
        "target_uniqueness_ratio": _compute_target_uniqueness_ratio(all_examples),
        "repeated_target_coverage": _compute_repeated_target_coverage(all_examples),
        "train_val_target_overlap": _compute_train_val_target_overlap(
            normalized["train"], normalized["val"]
        ),
        "length_stats": _compute_length_stats(all_examples, load_features),
        "train_only_oov_rate": _compute_train_only_oov_rate(
            normalized["train"], normalized["val"] + normalized["test"]
        ),
    }
    report["feature_stats"] = _compute_feature_stats(all_examples, load_features)
    return report


def _compute_split_overlap(
    splits: Mapping[str, Sequence[SignTextExample]],
) -> dict[str, list[str]]:
    example_to_splits: dict[str, set[str]] = defaultdict(set)
    for split_name, examples in splits.items():
        for example in examples:
            example_to_splits[example.example_id].add(split_name)
    overlaps = sorted(
        example_id
        for example_id, split_names in example_to_splits.items()
        if len(split_names) > 1
    )
    return {"example_ids": overlaps}


def _compute_video_id_leakage(
    splits: Mapping[str, Sequence[SignTextExample]],
) -> dict[str, object]:
    video_to_splits: dict[str, set[str]] = defaultdict(set)
    for split_name, examples in splits.items():
        for example in examples:
            video_id = _video_id_for_example(example)
            if video_id:
                video_to_splits[video_id].add(split_name)
    leaked = sorted(
        video_id
        for video_id, split_names in video_to_splits.items()
        if len(split_names) > 1
    )
    return {"count": len(leaked), "video_ids": leaked}


def _compute_target_uniqueness_ratio(examples: Sequence[SignTextExample]) -> float:
    if not examples:
        return 0.0
    unique_targets = {example.target_text for example in examples}
    return len(unique_targets) / len(examples)


def _compute_repeated_target_coverage(examples: Sequence[SignTextExample]) -> float:
    if not examples:
        return 0.0
    counts = Counter(example.target_text for example in examples)
    covered = sum(1 for example in examples if counts[example.target_text] > 1)
    return covered / len(examples)


def _compute_train_val_target_overlap(
    train_examples: Sequence[SignTextExample],
    val_examples: Sequence[SignTextExample],
) -> dict[str, object]:
    train_targets = {example.target_text for example in train_examples}
    val_targets = {example.target_text for example in val_examples}
    shared = sorted(train_targets & val_targets)
    return {
        "count": len(shared),
        "shared_targets": shared,
        "train_ratio": len(shared) / len(train_targets) if train_targets else 0.0,
        "val_ratio": len(shared) / len(val_targets) if val_targets else 0.0,
    }


def _compute_length_stats(
    examples: Sequence[SignTextExample],
    load_features: FeatureLoader | None,
) -> dict[str, dict[str, float | int | None]]:
    target_char_lengths = [len(example.target_text) for example in examples]
    target_token_lengths = [len(_tokenize_target(example.target_text)) for example in examples]
    frame_lengths: list[int] = []
    if load_features is not None:
        for example in examples:
            frame_lengths.append(int(load_features(example.features_path).shape[0]))
    return {
        "target_chars": _summarize_numbers(target_char_lengths),
        "target_tokens": _summarize_numbers(target_token_lengths),
        "frames": _summarize_numbers(frame_lengths),
    }


def _compute_feature_stats(
    examples: Sequence[SignTextExample],
    load_features: FeatureLoader | None,
) -> dict[str, int]:
    stats = {
        "sequences_scanned": 0,
        "total_frames": 0,
        "empty_sequences": 0,
        "nan_sequences": 0,
        "inf_sequences": 0,
        "nonfinite_values": 0,
    }
    if load_features is None:
        return stats

    for example in examples:
        arr = np.asarray(load_features(example.features_path))
        stats["sequences_scanned"] += 1
        stats["total_frames"] += int(arr.shape[0]) if arr.ndim >= 1 else 0
        if arr.ndim == 0 or arr.shape[0] == 0:
            stats["empty_sequences"] += 1
        has_nan = bool(np.isnan(arr).any())
        has_inf = bool(np.isinf(arr).any())
        if has_nan:
            stats["nan_sequences"] += 1
        if has_inf:
            stats["inf_sequences"] += 1
        stats["nonfinite_values"] += int((~np.isfinite(arr)).sum())
    return stats


def _compute_train_only_oov_rate(
    train_examples: Sequence[SignTextExample],
    eval_examples: Sequence[SignTextExample],
) -> float:
    train_vocab = {
        token
        for example in train_examples
        for token in _tokenize_target(example.target_text)
    }
    eval_tokens = [
        token
        for example in eval_examples
        for token in _tokenize_target(example.target_text)
    ]
    if not eval_tokens:
        return 0.0
    oov_count = sum(1 for token in eval_tokens if token not in train_vocab)
    return oov_count / len(eval_tokens)


def _summarize_numbers(values: Sequence[int]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min": None, "max": None, "mean": None}
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": float(mean(values)),
    }


def _video_id_for_example(example: SignTextExample) -> str | None:
    metadata = example.metadata or {}
    video_id = metadata.get("video_id")
    if video_id is None:
        return None
    return str(video_id)


def _tokenize_target(text: str) -> list[str]:
    stripped = text.strip()
    if not stripped:
        return []
    pieces = [piece for piece in stripped.split() if piece]
    if len(pieces) > 1:
        return pieces
    return list(stripped)
