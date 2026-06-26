"""Evaluate a PoseT5 runtime export on the standard validation subset."""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path


_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from scripts._bootstrap import ensure_repo_paths

ensure_repo_paths()

from tsl.data.unified import load_manifest, load_features
from tsl.eval.build_splits import split_by_video
from tsl.eval.manifest_quality import analyze_manifest_quality
from tsl.eval.slt_readiness import SltReadinessThresholds
from tsl.eval.slt_readiness import assess_slt_readiness
from tsl.eval.slt_readiness import compute_source_metrics
from tsl.eval.slt_metrics import chrf_corpus
from tsl.inference.pose_t5_translator import PoseT5Translator


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a PoseT5 runtime export on the standard validation subset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--export-dir", required=True)
    parser.add_argument("--data-roots", default="data/tsl51_v3,data/youtube_sl25_thai_v3")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-subset-size", type=int, default=50)
    parser.add_argument(
        "--split-policy",
        default="auto",
        choices=["auto", "manifest", "video"],
        help=(
            "How validation examples are produced: "
            "'manifest' preserves per-row split labels, "
            "'video' rebuilds a fresh 90/10 split by video_id, "
            "'auto' prefers manifest labels when both train and val are present."
        ),
    )
    parser.add_argument(
        "--required-sources",
        default="",
        help="Comma-separated source names that must appear in the validation subset.",
    )
    parser.add_argument(
        "--report-data-roots",
        default="",
        help="Optional comma-separated logical dataset ids to write into the report instead of --data-roots paths.",
    )
    parser.add_argument(
        "--manifest-quality-sources",
        default="",
        help="Optional comma-separated sources to enforce in manifest-quality gates. Defaults to all sources in the split.",
    )
    parser.add_argument("--report-json", default=None)
    parser.add_argument("--samples-json", default=None)
    parser.add_argument(
        "--eval-sources",
        default="",
        help="Optional comma-separated sources to keep in the evaluation subset before stratified sampling.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=72)
    parser.add_argument("--beam-size", type=int, default=5)
    parser.add_argument(
        "--no-repeat-ngram-size",
        default="3",
        help="Integer or 'none' to disable no-repeat ngram blocking.",
    )
    parser.add_argument(
        "--repetition-penalty",
        default="1.5",
        help="Float or 'none' to disable repetition penalty.",
    )
    parser.add_argument(
        "--length-penalty",
        default="0.7",
        help="Float or 'none' to disable length penalty.",
    )
    parser.add_argument("--min-val-chrf", type=float, default=80.0)
    parser.add_argument("--min-val-exact-match-pct", type=float, default=80.0)
    return parser


def _load_examples(data_roots_arg: str) -> list:
    examples = []
    for root in data_roots_arg.split(","):
        root = root.strip()
        if not root:
            continue
        examples.extend(load_manifest(root))
    return examples


def _parse_csv_list(value: str) -> list[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _parse_optional_int(value) -> int | None:
    text = str(value).strip().lower()
    if text in {"", "none", "null"}:
        return None
    return int(text)


def _parse_optional_float(value) -> float | None:
    text = str(value).strip().lower()
    if text in {"", "none", "null"}:
        return None
    return float(text)


def _default_report_path(export_dir: Path) -> Path:
    return export_dir.parent / f"{export_dir.name}_eval.json"


def _default_samples_path(export_dir: Path) -> Path:
    return export_dir.parent / f"{export_dir.name}_samples.json"


def _has_manifest_train_val_splits(examples: list) -> bool:
    seen = {
        str(getattr(example, "split", "")).strip().lower()
        for example in examples
        if str(getattr(example, "split", "")).strip()
    }
    return "train" in seen and "val" in seen


def _split_examples_from_manifest(examples: list) -> dict[str, list]:
    splits = {"train": [], "val": []}
    for example in examples:
        split = str(getattr(example, "split", "")).strip().lower()
        if split in splits:
            splits[split].append(example)
    return splits


def _build_train_val_splits(examples: list, split_policy: str, seed: int) -> dict[str, list]:
    if split_policy == "manifest":
        return _split_examples_from_manifest(examples)
    if split_policy == "video":
        return split_by_video(examples, fracs={"train": 0.9, "val": 0.1}, seed=seed)
    if _has_manifest_train_val_splits(examples):
        return _split_examples_from_manifest(examples)
    return split_by_video(examples, fracs={"train": 0.9, "val": 0.1}, seed=seed)


def _select_stratified_val_subset(examples: list, subset_size: int) -> list:
    if subset_size <= 0 or not examples:
        return []

    grouped: dict[str, list] = defaultdict(list)
    for ex in examples:
        grouped[ex.source].append(ex)

    selected: list = []
    offsets = {source: 0 for source in grouped}
    ordered_sources = list(grouped)

    while len(selected) < subset_size:
        progressed = False
        for source in ordered_sources:
            offset = offsets[source]
            bucket = grouped[source]
            if offset >= len(bucket):
                continue
            selected.append(bucket[offset])
            offsets[source] = offset + 1
            progressed = True
            if len(selected) >= subset_size:
                break
        if not progressed:
            break
    return selected


def _example_video_id(example) -> str:
    metadata = getattr(example, "metadata", None) or {}
    if isinstance(metadata, dict):
        video_id = str(metadata.get("video_id", "")).strip()
        if video_id:
            return video_id
    return str(getattr(example, "example_id", "")).strip()


def _augment_required_sources_for_video_eval(
    *,
    all_examples: list,
    val_pool: list,
    required_sources: list[str],
    eval_sources: list[str],
    split_policy: str,
) -> tuple[list, list[dict]]:
    if split_policy != "video":
        return list(val_pool), []

    allowed_sources = set(eval_sources) if eval_sources else None
    augmented = list(val_pool)
    present_sources = {ex.source for ex in augmented}
    diagnostics: list[dict] = []

    for source in required_sources:
        if source in present_sources:
            continue
        if allowed_sources is not None and source not in allowed_sources:
            continue
        source_examples = [ex for ex in all_examples if ex.source == source]
        if not source_examples:
            continue
        unique_video_ids = {_example_video_id(ex) for ex in source_examples}
        if len(unique_video_ids) != 1:
            continue
        augmented.extend(source_examples)
        present_sources.add(source)
        diagnostics.append(
            {
                "source": source,
                "reason": "single_video_source_fallback",
                "video_id_count": len(unique_video_ids),
                "example_count": len(source_examples),
                "video_ids": sorted(unique_video_ids),
            }
        )

    return augmented, diagnostics


def _evaluate_export(args: argparse.Namespace) -> dict:
    export_dir = Path(args.export_dir).resolve()
    report_path = Path(args.report_json).resolve() if args.report_json else _default_report_path(export_dir)
    samples_path = Path(args.samples_json).resolve() if args.samples_json else _default_samples_path(export_dir)
    required_sources = _parse_csv_list(getattr(args, "required_sources", ""))
    report_data_roots = _parse_csv_list(getattr(args, "report_data_roots", ""))
    manifest_quality_sources = _parse_csv_list(getattr(args, "manifest_quality_sources", ""))
    eval_sources = _parse_csv_list(getattr(args, "eval_sources", ""))

    all_examples = _load_examples(args.data_roots)
    splits = _build_train_val_splits(all_examples, args.split_policy, args.seed)
    manifest_quality = analyze_manifest_quality(
        splits["train"],
        splits["val"],
        required_sources=required_sources,
        gated_sources=manifest_quality_sources,
    )
    val_pool = splits["val"]
    if eval_sources:
        allowed_sources = set(eval_sources)
        val_pool = [ex for ex in val_pool if ex.source in allowed_sources]
    val_pool, eval_source_fallbacks = _augment_required_sources_for_video_eval(
        all_examples=all_examples,
        val_pool=val_pool,
        required_sources=required_sources,
        eval_sources=eval_sources,
        split_policy=args.split_policy,
    )
    val_examples = _select_stratified_val_subset(val_pool, args.val_subset_size)
    present_sources = {ex.source for ex in val_examples}
    missing_required_sources = [
        source for source in required_sources
        if source not in present_sources
    ]
    if missing_required_sources:
        raise ValueError(
            "Validation subset is missing required sources: "
            + ", ".join(missing_required_sources)
        )

    translator = PoseT5Translator.from_checkpoint_dir(str(export_dir), device=args.device)
    feature_batch = [load_features(ex.features_path) for ex in val_examples]
    decode_kwargs = {
        "max_new_tokens": int(args.max_new_tokens),
        "beam_size": int(args.beam_size),
        "no_repeat_ngram_size": _parse_optional_int(args.no_repeat_ngram_size),
        "repetition_penalty": _parse_optional_float(args.repetition_penalty),
        "length_penalty": _parse_optional_float(args.length_penalty),
    }
    if hasattr(translator, "translate_batch"):
        predictions = translator.translate_batch(feature_batch, **decode_kwargs)
    else:
        predictions = [translator.translate(features, **decode_kwargs) for features in feature_batch]

    hypotheses = [pred.sentence for pred in predictions]
    references = [ex.target_text for ex in val_examples]
    samples: list[dict] = []
    for ex, pred in zip(val_examples, predictions):
        samples.append(
            {
                "example_id": ex.example_id,
                "source": ex.source,
                "reference": ex.target_text,
                "hypothesis": pred.sentence,
                "score": pred.score,
            }
        )

    metrics = chrf_corpus(hypotheses, references)
    metrics["export_dir"] = str(export_dir)
    metrics["data_roots"] = report_data_roots or [root.strip() for root in args.data_roots.split(",") if root.strip()]
    metrics["seed"] = int(args.seed)
    metrics["split_policy"] = args.split_policy
    metrics["val_subset_size"] = len(val_examples)
    metrics["required_sources"] = required_sources
    metrics["manifest_quality_sources"] = manifest_quality_sources
    metrics["eval_sources"] = eval_sources
    metrics["eval_source_fallbacks"] = eval_source_fallbacks
    metrics["decoding"] = decode_kwargs
    metrics["source_metrics"] = compute_source_metrics(val_examples, hypotheses, references)
    metrics["source_counts"] = {
        source: sum(1 for ex in val_examples if ex.source == source)
        for source in sorted({ex.source for ex in val_examples})
    }
    metrics["manifest_quality"] = manifest_quality
    runtime_metadata_path = export_dir / "runtime_metadata.json"
    if runtime_metadata_path.is_file():
        with open(runtime_metadata_path, "r", encoding="utf-8") as fh:
            metrics["runtime_metadata"] = json.load(fh)
    promotion_status = assess_slt_readiness(
        overall_metrics=metrics,
        source_metrics=metrics["source_metrics"],
        manifest_quality=manifest_quality,
        train_metrics={},
        thresholds=SltReadinessThresholds(
            min_val_chrf=float(getattr(args, "min_val_chrf", 80.0)),
            min_val_exact_match_pct=float(getattr(args, "min_val_exact_match_pct", 80.0)),
        ),
    )
    if missing_required_sources:
        promotion_status["ready"] = False
        promotion_status["failures"].extend(
            f"missing required source metrics for {source}." for source in missing_required_sources
        )
    metrics["promotion_status"] = promotion_status

    report_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    samples_path.write_text(json.dumps(samples, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return metrics


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    metrics = _evaluate_export(args)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
