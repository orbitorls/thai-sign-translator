"""Video-level split builder for Thai Sign Language Translation.

Guarantees that all clips from the same video land in the same split,
preventing train/test leakage from the same signer or sentence appearing
in multiple partitions.
"""
from __future__ import annotations

import csv
import dataclasses
import os
import random
from collections import defaultdict

from tsl.data.manifest import SignTextExample

__all__ = [
    "split_by_video",
    "check_video_leakage",
    "write_splits_to_manifest",
    "read_frozen_test_examples",
]

_CSV_FIELDNAMES = [
    "example_id",
    "video_id",
    "split",
    "source",
    "features_path",
    "target_text",
]


def _get_video_id(example: SignTextExample) -> str:
    """Return the video_id for an example, falling back to example_id."""
    if example.metadata and "video_id" in example.metadata:
        return str(example.metadata["video_id"])
    return example.example_id


def split_by_video(
    examples: list[SignTextExample],
    fracs: dict[str, float],
    seed: int = 42,
) -> dict[str, list[SignTextExample]]:
    """Group examples by video_id and assign video groups to splits.

    Parameters
    ----------
    examples:
        All examples to partition.
    fracs:
        Mapping of split name → fraction (should sum to ~1.0).
        Example: ``{"train": 0.8, "val": 0.1, "test": 0.1}``.
    seed:
        Random seed for deterministic shuffling.

    Returns
    -------
    dict mapping split name → list of SignTextExample.
    Guarantees no video_id appears in more than one split.
    """
    # Group examples by resolved video_id
    groups: dict[str, list[SignTextExample]] = defaultdict(list)
    for ex in examples:
        vid = _get_video_id(ex)
        groups[vid].append(ex)

    video_ids = list(groups.keys())
    rng = random.Random(seed)
    rng.shuffle(video_ids)

    n = len(video_ids)
    result: dict[str, list[SignTextExample]] = {name: [] for name in fracs}

    cumulative = 0.0
    prev_boundary = 0
    split_names = list(fracs.keys())
    for i, name in enumerate(split_names):
        cumulative += fracs[name]
        if i == len(split_names) - 1:
            # Last split gets everything remaining to avoid rounding gaps
            boundary = n
        else:
            boundary = round(cumulative * n)
        for vid in video_ids[prev_boundary:boundary]:
            result[name].extend(groups[vid])
        prev_boundary = boundary

    return result


def _extract_video_ids(examples: list[SignTextExample]) -> set[str]:
    return {_get_video_id(ex) for ex in examples}


def check_video_leakage(
    train: list[SignTextExample],
    val: list[SignTextExample],
    test: list[SignTextExample] | None = None,
) -> None:
    """Assert no video_id appears in more than one split.

    Parameters
    ----------
    train, val, test:
        Lists of examples for each split.

    Raises
    ------
    ValueError
        If any video_id appears in more than one split, listing the offenders.
    """
    train_ids = _extract_video_ids(train)
    val_ids = _extract_video_ids(val)

    leaks: list[str] = []

    train_val = train_ids & val_ids
    if train_val:
        leaks.append(f"train∩val: {sorted(train_val)}")

    if test is not None:
        test_ids = _extract_video_ids(test)
        train_test = train_ids & test_ids
        if train_test:
            leaks.append(f"train∩test: {sorted(train_test)}")
        val_test = val_ids & test_ids
        if val_test:
            leaks.append(f"val∩test: {sorted(val_test)}")

    if leaks:
        raise ValueError("Video leakage detected — " + "; ".join(leaks))


def write_splits_to_manifest(
    examples_by_split: dict[str, list[SignTextExample]],
    output_path: str,
) -> None:
    """Write a CSV manifest of all examples with their split assignments.

    Columns: example_id, video_id, split, source, features_path, target_text.
    Overwrites the file if it already exists.

    Parameters
    ----------
    examples_by_split:
        Mapping of split name → list of examples (as returned by
        :func:`split_by_video`).
    output_path:
        Destination file path.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_CSV_FIELDNAMES)
        writer.writeheader()
        for split_name, examples in examples_by_split.items():
            for ex in examples:
                writer.writerow(
                    {
                        "example_id": ex.example_id,
                        "video_id": _get_video_id(ex),
                        "split": split_name,
                        "source": ex.source,
                        "features_path": ex.features_path,
                        "target_text": ex.target_text,
                    }
                )


def read_frozen_test_examples(
    manifest_path: str,
    data_root: str | None = None,
) -> list[SignTextExample]:
    """Read a previously-written frozen test manifest CSV.

    Parameters
    ----------
    manifest_path:
        Path to the CSV file written by :func:`write_splits_to_manifest`.
    data_root:
        Optional root directory.  When provided, relative ``features_path``
        values are joined with this root.  Absolute paths are left unchanged.

    Returns
    -------
    List of :class:`~tsl.data.manifest.SignTextExample` with ``split="test"``.
    """
    examples: list[SignTextExample] = []
    with open(manifest_path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            features_path = row["features_path"]
            if data_root is not None and not os.path.isabs(features_path):
                features_path = os.path.join(data_root, features_path)
            ex = SignTextExample(
                example_id=row["example_id"],
                source=row["source"],
                split="test",
                features_path=features_path,
                target_text=row["target_text"],
                metadata={"video_id": row["video_id"]},
            )
            examples.append(ex)
    return examples
