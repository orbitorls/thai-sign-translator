from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
from collections import Counter
from pathlib import Path


_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from scripts._bootstrap import ensure_repo_paths

ensure_repo_paths()

import pandas as pd

from tsl.data.unified import load_manifest
from tsl.eval.manifest_quality import analyze_manifest_quality


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a mixed PoseT5 manifest from all available finetune datasets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-roots",
        default="data/tsl51_v3,data/thaisignvis_v3_probe,data/youtube_sl25_thai_v3",
        help="Comma-separated manifest roots to merge.",
    )
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--train-only-sources",
        default="thaisignvis,youtube_sl25_thai",
        help="Comma-separated sources whose non-train examples should be relabeled into train.",
    )
    parser.add_argument(
        "--required-sources",
        default="tsl51",
        help="Comma-separated sources that must appear in the manifest-quality report.",
    )
    parser.add_argument(
        "--manifest-quality-sources",
        default="tsl51",
        help="Comma-separated sources to enforce with manifest-quality gates.",
    )
    parser.add_argument(
        "--allow-missing-roots",
        choices=("true", "false"),
        default="true",
        help="Skip missing dataset roots when true; raise when false.",
    )
    return parser


def _parse_csv_list(value: str) -> list[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _resolve_bool_flag(value: str | bool | None) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _load_examples(data_roots_arg: str, *, allow_missing_roots: bool) -> tuple[list, list[str], list[str]]:
    examples = []
    used_roots: list[str] = []
    missing_roots: list[str] = []
    for raw_root in _parse_csv_list(data_roots_arg):
        root_path = Path(raw_root).resolve()
        if not root_path.is_dir():
            missing_roots.append(str(root_path))
            if not allow_missing_roots:
                raise FileNotFoundError(f"dataset root not found: {root_path}")
            continue
        used_roots.append(str(root_path))
        examples.extend(load_manifest(str(root_path)))
    return examples, used_roots, missing_roots


def _relabel_split(example, split: str):
    if getattr(example, "split", None) == split:
        return example
    return dataclasses.replace(example, split=split)


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


def _summarize_selection(
    all_examples: list,
    selected_examples: list,
    *,
    used_roots: list[str],
    missing_roots: list[str],
    train_only_sources: set[str],
) -> dict:
    available_source_counts = Counter(ex.source for ex in all_examples)
    selected_source_counts = Counter(ex.source for ex in selected_examples)
    selected_split_counts = Counter(getattr(ex, "split", "") for ex in selected_examples)
    source_split_counts = Counter(
        (ex.source, getattr(ex, "split", ""))
        for ex in selected_examples
    )
    return {
        "used_roots": used_roots,
        "missing_roots": missing_roots,
        "available_source_counts": dict(available_source_counts),
        "selected_source_counts": dict(selected_source_counts),
        "selected_split_counts": dict(selected_split_counts),
        "selected_source_split_counts": {
            f"{source}|{split}": count
            for (source, split), count in sorted(source_split_counts.items())
        },
        "train_only_sources": sorted(train_only_sources),
    }


def build_mixed_manifest(args: argparse.Namespace) -> dict:
    allow_missing_roots = _resolve_bool_flag(args.allow_missing_roots)
    train_only_sources = set(_parse_csv_list(args.train_only_sources))
    required_sources = _parse_csv_list(args.required_sources)
    manifest_quality_sources = _parse_csv_list(args.manifest_quality_sources)

    all_examples, used_roots, missing_roots = _load_examples(
        args.data_roots,
        allow_missing_roots=allow_missing_roots,
    )

    selected_examples = []
    for ex in all_examples:
        if ex.source in train_only_sources and str(ex.split).strip().lower() != "train":
            selected_examples.append(_relabel_split(ex, "train"))
        else:
            selected_examples.append(ex)

    if not selected_examples:
        raise RuntimeError("no manifest examples were selected from the requested dataset roots")

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.csv"
    pd.DataFrame(_build_rows(selected_examples)).to_csv(manifest_path, index=False, encoding="utf-8")

    train_examples = [ex for ex in selected_examples if ex.split == "train"]
    val_examples = [ex for ex in selected_examples if ex.split == "val"]
    quality = analyze_manifest_quality(
        train_examples,
        val_examples,
        required_sources=required_sources,
        gated_sources=manifest_quality_sources,
    )
    quality_path = out_dir / "manifest_quality.json"
    quality_path.write_text(json.dumps(quality, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    selection = _summarize_selection(
        all_examples,
        selected_examples,
        used_roots=used_roots,
        missing_roots=missing_roots,
        train_only_sources=train_only_sources,
    )
    summary = {
        "manifest_path": str(manifest_path),
        "manifest_quality_path": str(quality_path),
        "selected_examples": len(selected_examples),
        "quality_passed": bool(quality.get("passed", False)),
        "required_sources": required_sources,
        "manifest_quality_sources": manifest_quality_sources,
        **selection,
    }
    summary_path = out_dir / "build_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary["build_summary_path"] = str(summary_path)
    return summary


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    summary = build_mixed_manifest(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
