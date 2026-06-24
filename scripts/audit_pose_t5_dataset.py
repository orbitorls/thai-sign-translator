"""Audit mixed PoseT5 manifests before one-shot cloud training."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
import os
from pathlib import Path
import sys


_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from scripts._bootstrap import ensure_repo_paths

ensure_repo_paths()

import numpy as np

from tsl.data.quality import audit_dataset_splits
from tsl.data.unified import load_features, load_manifest


def _parse_csv_list(value: str) -> list[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _parse_expected_counts(value: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in _parse_csv_list(value):
        if "=" not in item:
            raise ValueError(f"expected source count item must be name=count: {item!r}")
        name, raw_count = item.split("=", 1)
        counts[name.strip()] = int(raw_count.strip())
    return counts


def _load_all_examples(data_roots: str) -> list:
    examples = []
    for root in _parse_csv_list(data_roots):
        examples.extend(load_manifest(root))
    return examples


def _split_examples(examples: list) -> dict[str, list]:
    splits = {"train": [], "val": [], "test": []}
    for example in examples:
        split = str(getattr(example, "split", "") or "train").strip().lower()
        if split not in splits:
            split = "train"
        splits[split].append(example)
    return splits


def _duplicate_values(examples: list, attr: str) -> list[str]:
    counts = Counter(str(getattr(example, attr, "")).strip() for example in examples)
    return sorted(value for value, count in counts.items() if value and count > 1)


def _target_overlap(train_examples: list, eval_examples: list) -> list[str]:
    train_targets = {str(getattr(example, "target_text", "")).strip() for example in train_examples}
    eval_targets = {str(getattr(example, "target_text", "")).strip() for example in eval_examples}
    return sorted(target for target in train_targets & eval_targets if target)


def _video_leakage(train_examples: list, eval_examples: list) -> list[str]:
    train_videos = {str(getattr(example, "video_id", "")).strip() for example in train_examples}
    eval_videos = {str(getattr(example, "video_id", "")).strip() for example in eval_examples}
    return sorted(video_id for video_id in train_videos & eval_videos if video_id)


def _feature_stats(examples: list) -> tuple[dict, list[str], list[dict]]:
    dim_counts: Counter[str] = Counter()
    failures: list[str] = []
    bad_shapes: list[dict] = []
    empty = nan = inf = 0
    for example in examples:
        try:
            features = load_features(example.features_path)
        except Exception as exc:  # pragma: no cover - message is what matters
            failures.append(f"{example.example_id}: failed to load features: {exc}")
            continue
        shape = tuple(int(part) for part in getattr(features, "shape", ()))
        if features.size == 0:
            empty += 1
        if np.isnan(features).any():
            nan += 1
        if np.isinf(features).any():
            inf += 1
        if len(shape) >= 2:
            dim_counts[str(shape[1])] += 1
        else:
            dim_counts["invalid"] += 1
        if len(shape) != 2 or shape[1] != 312:
            bad_shapes.append(
                {
                    "example_id": example.example_id,
                    "features_path": example.features_path,
                    "shape": list(shape),
                }
            )
    stats = {
        "feature_dim_counts": dict(sorted(dim_counts.items())),
        "empty_feature_count": empty,
        "nan_feature_count": nan,
        "inf_feature_count": inf,
        "bad_feature_shapes": bad_shapes[:25],
        "bad_feature_shape_count": len(bad_shapes),
    }
    return stats, failures, bad_shapes


def audit_pose_t5_dataset(
    *,
    data_roots: str,
    expected_manifest_rows: int = 0,
    expected_source_counts: str = "",
    production_gated_sources: str = "",
) -> dict:
    examples = _load_all_examples(data_roots)
    splits = _split_examples(examples)
    quality = audit_dataset_splits(splits, load_features=load_features)
    source_counts = Counter(str(example.source) for example in examples)
    split_counts = {name: len(rows) for name, rows in splits.items()}
    source_split_counts: dict[str, dict[str, int]] = defaultdict(lambda: {"train": 0, "val": 0, "test": 0})
    for split_name, rows in splits.items():
        for example in rows:
            source_split_counts[str(example.source)][split_name] += 1

    failures: list[str] = []
    warnings: list[str] = []
    if expected_manifest_rows and len(examples) != expected_manifest_rows:
        failures.append(f"manifest rows {len(examples)} != expected {expected_manifest_rows}")
    expected_counts = _parse_expected_counts(expected_source_counts)
    for source, expected in expected_counts.items():
        actual = int(source_counts.get(source, 0))
        if actual != expected:
            failures.append(f"source {source} count {actual} != expected {expected}")

    train_examples = splits["train"]
    eval_examples = splits["val"] + splits["test"]
    train_only_sources = [
        source for source, counts in sorted(source_split_counts.items())
        if counts["train"] > 0 and counts["val"] == 0 and counts["test"] == 0
    ]
    gated_sources = _parse_csv_list(production_gated_sources)
    for source in gated_sources:
        counts = source_split_counts.get(source, {"train": 0, "val": 0, "test": 0})
        if counts["train"] <= 0:
            failures.append(f"production-gated source {source} has no train examples")
        if counts["val"] <= 0 and counts["test"] <= 0:
            failures.append(f"production-gated source {source} has no val/test examples")
    for source in train_only_sources:
        if source not in gated_sources:
            warnings.append(f"source {source} is train-only and is not production-gated")

    feature_report, feature_failures, bad_shapes = _feature_stats(examples)
    failures.extend(feature_failures)
    if bad_shapes:
        failures.append(f"{len(bad_shapes)} feature arrays are not 312-dim PoseT5 features")
    if feature_report["empty_feature_count"]:
        failures.append(f"{feature_report['empty_feature_count']} feature arrays are empty")
    if feature_report["nan_feature_count"] or feature_report["inf_feature_count"]:
        failures.append(
            f"non-finite feature arrays: nan={feature_report['nan_feature_count']} inf={feature_report['inf_feature_count']}"
        )

    repeated_target_ratio = 0.0
    if examples:
        target_counts = Counter(str(getattr(example, "target_text", "")).strip() for example in examples)
        repeated = sum(1 for example in examples if target_counts[str(getattr(example, "target_text", "")).strip()] > 1)
        repeated_target_ratio = repeated / len(examples)

    report = {
        "passed": not failures,
        "data_roots": _parse_csv_list(data_roots),
        "manifest_rows": len(examples),
        "expected_manifest_rows": expected_manifest_rows,
        "source_counts": dict(sorted(source_counts.items())),
        "expected_source_counts": expected_counts,
        "split_counts": split_counts,
        "source_split_counts": {source: dict(counts) for source, counts in sorted(source_split_counts.items())},
        "duplicate_segment_ids": _duplicate_values(examples, "example_id"),
        "duplicate_npy_paths": _duplicate_values(examples, "features_path"),
        "video_leakage": _video_leakage(train_examples, eval_examples),
        "video_leakage_count": len(_video_leakage(train_examples, eval_examples)),
        "target_overlap": _target_overlap(train_examples, eval_examples),
        "target_overlap_count": len(_target_overlap(train_examples, eval_examples)),
        "target_uniqueness_ratio": quality.get("target_uniqueness_ratio", math.nan),
        "repeated_target_ratio": repeated_target_ratio,
        "train_only_sources": train_only_sources,
        "production_gated_sources": gated_sources,
        "production_warnings": warnings,
        "quality": quality,
        "failures": failures,
        **feature_report,
    }
    return report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit mixed PoseT5 dataset readiness before Kaggle training.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-roots", required=True)
    parser.add_argument("--expected-manifest-rows", type=int, default=0)
    parser.add_argument("--expected-source-counts", default="")
    parser.add_argument("--production-gated-sources", default="")
    parser.add_argument("--json-out", default="")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    report = audit_pose_t5_dataset(
        data_roots=args.data_roots,
        expected_manifest_rows=args.expected_manifest_rows,
        expected_source_counts=args.expected_source_counts,
        production_gated_sources=args.production_gated_sources,
    )
    text = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    if args.json_out:
        target = Path(args.json_out)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if report["passed"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
