"""Build a small v3-312 mixed research manifest for PoseT5 experiments."""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import random
import re
import sys
from collections import Counter
from pathlib import Path


_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC_ROOT = os.path.join(_REPO_ROOT, "src")
if _SRC_ROOT not in sys.path:
    sys.path.insert(0, _SRC_ROOT)

import pandas as pd

from tsl.data.unified import load_manifest
from tsl.eval.manifest_quality import analyze_manifest_quality


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a small mixed research manifest for PoseT5 training.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-roots",
        default="data/tsl51_v3,data/thaisignvis_v3_probe,data/youtube_sl25_thai_v3",
    )
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-thaisignvis-train", type=int, default=200)
    parser.add_argument("--max-thaisignvis-val", type=int, default=30)
    parser.add_argument("--max-youtube-train", type=int, default=200)
    parser.add_argument("--max-youtube-val", type=int, default=30)
    parser.add_argument("--min-youtube-target-count", type=int, default=2)
    parser.add_argument(
        "--required-sources",
        default="tsl51,thaisignvis",
        help="Comma-separated sources that must appear in the manifest-quality report.",
    )
    parser.add_argument(
        "--manifest-quality-sources",
        default="",
        help="Optional comma-separated sources to gate in manifest-quality checks. Defaults to all selected sources.",
    )
    parser.add_argument(
        "--dataset-role",
        choices=("research", "readiness"),
        default="research",
        help="Use readiness for manifests that must retain required source validation splits.",
    )
    parser.add_argument(
        "--thaisignvis-train-only",
        choices=("true", "false"),
        default="false",
        help="Route all ThaiSignVis examples into train so val metrics stay on repeated-label sources.",
    )
    parser.add_argument(
        "--train-only-sources",
        default="",
        help="Optional comma-separated sources whose validation examples should be relabeled into train.",
    )
    return parser


def _load_examples(data_roots_arg: str) -> list:
    examples = []
    for root in data_roots_arg.split(","):
        root = root.strip()
        if not root:
            continue
        if not os.path.isdir(root):
            continue
        examples.extend(load_manifest(root))
    return examples


def _parse_csv_list(value: str) -> list[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _normalized_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def _limit_examples(examples: list, limit: int, seed: int) -> list:
    if limit <= 0 or len(examples) <= limit:
        return list(examples)
    pool = list(examples)
    random.Random(seed).shuffle(pool)
    return pool[:limit]


def _youtube_repeated_subset(examples: list, split: str, limit: int, min_target_count: int, seed: int) -> list:
    source_examples = [ex for ex in examples if ex.source == "youtube_sl25_thai" and ex.split == split]
    counts = Counter(
        _normalized_text(ex.target_text)
        for ex in examples
        if ex.source == "youtube_sl25_thai"
    )
    repeated = [ex for ex in source_examples if counts[_normalized_text(ex.target_text)] >= min_target_count]
    return _limit_examples(repeated, limit, seed)


def _build_rows(selected_examples: list) -> list[dict]:
    rows = []
    for ex in selected_examples:
        metadata = ex.metadata if isinstance(ex.metadata, dict) else {}
        rows.append(
            {
                "segment_id": ex.example_id,
                "npy_path": os.path.abspath(ex.features_path),
                "text": ex.target_text,
                "video_id": str(metadata.get("video_id", "")),
                "split": ex.split,
                "source": ex.source,
                "feature_layout_version": str(metadata.get("feature_layout_version", "v3-312")),
            }
        )
    return rows


def _summarize_source_selection(
    all_examples: list,
    selected_examples: list,
    train_examples: list,
    val_examples: list,
    *,
    min_youtube_target_count: int,
) -> dict:
    available_source_counts = Counter(ex.source for ex in all_examples)
    selected_source_counts = Counter(ex.source for ex in selected_examples)
    train_source_counts = Counter(ex.source for ex in train_examples)
    val_source_counts = Counter(ex.source for ex in val_examples)

    included_sources = sorted(source for source, count in selected_source_counts.items() if count > 0)
    excluded_sources = sorted(
        source
        for source, count in available_source_counts.items()
        if count > 0 and selected_source_counts.get(source, 0) == 0
    )

    exclusion_reasons: dict[str, str] = {}
    for source in excluded_sources:
        if source == "youtube_sl25_thai":
            exclusion_reasons[source] = (
                "No repeated-target examples met "
                f"min_youtube_target_count={min_youtube_target_count}."
            )
        else:
            exclusion_reasons[source] = "No examples were selected for this source."

    return {
        "available_source_counts": dict(available_source_counts),
        "selected_source_counts": dict(selected_source_counts),
        "train_source_counts": dict(train_source_counts),
        "val_source_counts": dict(val_source_counts),
        "included_sources": included_sources,
        "excluded_sources": excluded_sources,
        "exclusion_reasons": exclusion_reasons,
    }


def _build_subset(args: argparse.Namespace) -> dict:
    all_examples = _load_examples(args.data_roots)
    thaisignvis_train_only = str(args.thaisignvis_train_only).lower() == "true"
    train_only_sources = set(_parse_csv_list(getattr(args, "train_only_sources", "")))
    if thaisignvis_train_only:
        train_only_sources.add("thaisignvis")
    required_sources = [
        source.strip()
        for source in str(getattr(args, "required_sources", "")).split(",")
        if source.strip()
    ]
    manifest_quality_sources = _parse_csv_list(getattr(args, "manifest_quality_sources", ""))
    dataset_role = str(getattr(args, "dataset_role", "research")).strip().lower()
    if dataset_role == "readiness" and train_only_sources:
        raise ValueError("readiness datasets cannot use train-only source routing.")

    tsl51 = [ex for ex in all_examples if ex.source == "tsl51"]
    thaisignvis = [ex for ex in all_examples if ex.source == "thaisignvis"]
    selected = []
    selected.extend(tsl51)
    thaisignvis_train_pool = [ex for ex in thaisignvis if ex.split == "train"]
    thaisignvis_val_pool = [ex for ex in thaisignvis if ex.split == "val"]
    if "thaisignvis" in train_only_sources:
        moved_val = [dataclasses.replace(ex, split="train") for ex in thaisignvis_val_pool]
        selected.extend(
            _limit_examples(
                thaisignvis_train_pool + moved_val,
                args.max_thaisignvis_train + args.max_thaisignvis_val,
                args.seed,
            )
        )
    else:
        selected.extend(
            _limit_examples(thaisignvis_train_pool, args.max_thaisignvis_train, args.seed)
        )
        selected.extend(
            _limit_examples(thaisignvis_val_pool, args.max_thaisignvis_val, args.seed + 1)
        )
    selected.extend(
        _youtube_repeated_subset(
            all_examples,
            split="train",
            limit=args.max_youtube_train,
            min_target_count=args.min_youtube_target_count,
            seed=args.seed + 2,
        )
    )
    youtube_val_subset = _youtube_repeated_subset(
        all_examples,
        split="val",
        limit=args.max_youtube_val,
        min_target_count=args.min_youtube_target_count,
        seed=args.seed + 3,
    )
    if "youtube_sl25_thai" in train_only_sources:
        youtube_val_subset = [dataclasses.replace(ex, split="train") for ex in youtube_val_subset]
    selected.extend(youtube_val_subset)

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.csv"
    rows = _build_rows(selected)
    pd.DataFrame(rows).to_csv(manifest_path, index=False, encoding="utf-8")

    train_examples = [ex for ex in selected if ex.split == "train"]
    val_examples = [ex for ex in selected if ex.split == "val"]
    if manifest_quality_sources:
        quality = analyze_manifest_quality(
            train_examples,
            val_examples,
            required_sources=required_sources,
            gated_sources=manifest_quality_sources,
        )
    else:
        quality = analyze_manifest_quality(
            train_examples,
            val_examples,
            required_sources=required_sources,
        )
    quality_path = out_dir / "manifest_quality.json"
    quality_path.write_text(json.dumps(quality, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    source_selection = _summarize_source_selection(
        all_examples,
        selected,
        train_examples,
        val_examples,
        min_youtube_target_count=args.min_youtube_target_count,
    )

    summary = {
        "manifest_path": str(manifest_path),
        "manifest_quality_path": str(quality_path),
        "selected_examples": len(selected),
        "source_counts": source_selection["selected_source_counts"],
        "quality_passed": bool(quality.get("passed", False)),
        "required_sources": required_sources,
        "manifest_quality_sources": manifest_quality_sources,
        "dataset_role": dataset_role,
        "research_only": dataset_role != "readiness" or bool(train_only_sources),
        "thaisignvis_train_only": thaisignvis_train_only,
        "train_only_sources": sorted(train_only_sources),
        "included_sources": source_selection["included_sources"],
        "excluded_sources": source_selection["excluded_sources"],
        "exclusion_reasons": source_selection["exclusion_reasons"],
        "available_source_counts": source_selection["available_source_counts"],
        "train_source_counts": source_selection["train_source_counts"],
        "val_source_counts": source_selection["val_source_counts"],
    }
    return summary


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    summary = _build_subset(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
