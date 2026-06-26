"""Search decoding hyperparameters for a PoseT5 runtime export."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from scripts._bootstrap import ensure_repo_paths

ensure_repo_paths()

from scripts.evaluate_pose_t5_export import (
    _augment_required_sources_for_video_eval,
    _build_train_val_splits,
    _load_examples,
    _parse_csv_list,
    _parse_optional_float,
    _parse_optional_int,
    _select_stratified_val_subset,
)
from tsl.data.unified import load_features
from tsl.eval.manifest_quality import analyze_manifest_quality
from tsl.eval.slt_metrics import chrf_corpus
from tsl.inference.pose_t5_translator import PoseT5Translator


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Search decoding hyperparameters for a PoseT5 runtime export.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--export-dir", required=True)
    parser.add_argument("--data-roots", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-subset-size", type=int, default=25)
    parser.add_argument(
        "--split-policy",
        default="auto",
        choices=["auto", "manifest", "video"],
    )
    parser.add_argument("--required-sources", default="")
    parser.add_argument(
        "--manifest-quality-sources",
        default="",
        help="Optional comma-separated sources to enforce in manifest-quality gates. Defaults to all sources in the split.",
    )
    parser.add_argument("--report-data-roots", default="")
    parser.add_argument("--max-new-tokens-grid", default="48,72")
    parser.add_argument("--beam-size-grid", default="1,3,5")
    parser.add_argument("--no-repeat-ngram-grid", default="none,2,3")
    parser.add_argument("--repetition-penalty-grid", default="none,1.1,1.3,1.5")
    parser.add_argument("--length-penalty-grid", default="1.0,0.9,0.7")
    parser.add_argument("--max-trials", type=int, default=32)
    parser.add_argument("--report-json", default=None)
    parser.add_argument("--best-eval-json", default=None)
    parser.add_argument("--best-samples-json", default=None)
    parser.add_argument(
        "--eval-sources",
        default="",
        help="Optional comma-separated sources to keep in the evaluation subset before stratified sampling.",
    )
    return parser


def _default_report_path(export_dir: Path) -> Path:
    return export_dir / "decoding_search.json"


def _default_best_eval_path(export_dir: Path) -> Path:
    return export_dir / "decoding_best_eval.json"


def _default_best_samples_path(export_dir: Path) -> Path:
    return export_dir / "decoding_best_samples.json"


def _trial_configs(args: argparse.Namespace) -> list[dict]:
    max_new_tokens = [int(v) for v in _parse_csv_list(args.max_new_tokens_grid)]
    beam_sizes = [int(v) for v in _parse_csv_list(args.beam_size_grid)]
    no_repeat_values = [_parse_optional_int(v) for v in _parse_csv_list(args.no_repeat_ngram_grid)]
    repetition_values = [_parse_optional_float(v) for v in _parse_csv_list(args.repetition_penalty_grid)]
    length_values = [_parse_optional_float(v) for v in _parse_csv_list(args.length_penalty_grid)]

    configs: list[dict] = [
        {
            "label": "baseline",
            "max_new_tokens": 72,
            "beam_size": 5,
            "no_repeat_ngram_size": 3,
            "repetition_penalty": 1.5,
            "length_penalty": 0.7,
        }
    ]
    seen = {
        (
            cfg["max_new_tokens"],
            cfg["beam_size"],
            cfg["no_repeat_ngram_size"],
            cfg["repetition_penalty"],
            cfg["length_penalty"],
        )
        for cfg in configs
    }
    for beam_size in beam_sizes:
        for no_repeat_ngram_size in no_repeat_values:
            for repetition_penalty in repetition_values:
                for length_penalty in length_values:
                    for max_tokens in max_new_tokens:
                        key = (
                            max_tokens,
                            beam_size,
                            no_repeat_ngram_size,
                            repetition_penalty,
                            length_penalty,
                        )
                        if key in seen:
                            continue
                        seen.add(key)
                        configs.append(
                            {
                                "label": (
                                    f"beam{beam_size}_max{max_tokens}"
                                    f"_nr{no_repeat_ngram_size if no_repeat_ngram_size is not None else 'none'}"
                                    f"_rep{repetition_penalty if repetition_penalty is not None else 'none'}"
                                    f"_len{length_penalty if length_penalty is not None else 'none'}"
                                ),
                                "max_new_tokens": max_tokens,
                                "beam_size": beam_size,
                                "no_repeat_ngram_size": no_repeat_ngram_size,
                                "repetition_penalty": repetition_penalty,
                                "length_penalty": length_penalty,
                            }
                        )
    return configs[: max(1, int(args.max_trials))]


def _sort_key(entry: dict) -> tuple[float, float, float]:
    return (
        float(entry.get("exact_match_pct", 0.0)),
        float(entry.get("chrf", 0.0)),
        float(entry.get("bleu", 0.0)),
    )


def _run_search(args: argparse.Namespace) -> dict:
    export_dir = Path(args.export_dir).resolve()
    report_path = Path(args.report_json).resolve() if args.report_json else _default_report_path(export_dir)
    best_eval_path = Path(args.best_eval_json).resolve() if args.best_eval_json else _default_best_eval_path(export_dir)
    best_samples_path = Path(args.best_samples_json).resolve() if args.best_samples_json else _default_best_samples_path(export_dir)
    required_sources = _parse_csv_list(args.required_sources)
    report_data_roots = _parse_csv_list(args.report_data_roots)
    manifest_quality_sources = _parse_csv_list(args.manifest_quality_sources)
    eval_sources = _parse_csv_list(args.eval_sources)

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
        val_pool = [example for example in val_pool if example.source in allowed_sources]
    val_pool, eval_source_fallbacks = _augment_required_sources_for_video_eval(
        all_examples=all_examples,
        val_pool=val_pool,
        required_sources=required_sources,
        eval_sources=eval_sources,
        split_policy=args.split_policy,
    )
    val_examples = _select_stratified_val_subset(val_pool, args.val_subset_size)
    present_sources = {example.source for example in val_examples}
    missing_required_sources = [source for source in required_sources if source not in present_sources]
    if missing_required_sources:
        raise ValueError(
            "Validation subset is missing required sources: " + ", ".join(missing_required_sources)
        )

    feature_batch = [load_features(example.features_path) for example in val_examples]
    references = [example.target_text for example in val_examples]
    translator = PoseT5Translator.from_checkpoint_dir(str(export_dir), device=args.device)

    trials: list[dict] = []
    best_entry: dict | None = None
    best_samples: list[dict] = []
    for config in _trial_configs(args):
        predictions = translator.translate_batch(
            feature_batch,
            max_new_tokens=config["max_new_tokens"],
            beam_size=config["beam_size"],
            no_repeat_ngram_size=config["no_repeat_ngram_size"],
            repetition_penalty=config["repetition_penalty"],
            length_penalty=config["length_penalty"],
        )
        hypotheses = [prediction.sentence for prediction in predictions]
        metrics = chrf_corpus(hypotheses, references)
        entry = {
            "label": config["label"],
            "max_new_tokens": config["max_new_tokens"],
            "beam_size": config["beam_size"],
            "no_repeat_ngram_size": config["no_repeat_ngram_size"],
            "repetition_penalty": config["repetition_penalty"],
            "length_penalty": config["length_penalty"],
            "chrf": metrics["chrf"],
            "bleu": metrics["bleu"],
            "exact_match": metrics["exact_match"],
            "exact_match_pct": metrics["exact_match_pct"],
        }
        trials.append(entry)
        if best_entry is None or _sort_key(entry) > _sort_key(best_entry):
            best_entry = dict(entry)
            best_samples = [
                {
                    "example_id": example.example_id,
                    "source": example.source,
                    "reference": example.target_text,
                    "hypothesis": prediction.sentence,
                    "score": prediction.score,
                }
                for example, prediction in zip(val_examples, predictions)
            ]

    assert best_entry is not None
    summary = {
        "export_dir": str(export_dir),
        "data_roots": report_data_roots or [root.strip() for root in args.data_roots.split(",") if root.strip()],
        "seed": int(args.seed),
        "split_policy": args.split_policy,
        "val_subset_size": len(val_examples),
        "required_sources": required_sources,
        "manifest_quality_sources": manifest_quality_sources,
        "eval_sources": eval_sources,
        "eval_source_fallbacks": eval_source_fallbacks,
        "manifest_quality": manifest_quality,
        "best": best_entry,
        "trials": trials,
    }
    best_eval = {
        **best_entry,
        "export_dir": str(export_dir),
        "data_roots": summary["data_roots"],
        "seed": int(args.seed),
        "split_policy": args.split_policy,
        "val_subset_size": len(val_examples),
        "required_sources": required_sources,
        "manifest_quality_sources": manifest_quality_sources,
        "eval_sources": eval_sources,
        "eval_source_fallbacks": eval_source_fallbacks,
        "manifest_quality": manifest_quality,
        "source_counts": {
            source: sum(1 for example in val_examples if example.source == source)
            for source in sorted({example.source for example in val_examples})
        },
        "decoding": {
            "max_new_tokens": best_entry["max_new_tokens"],
            "beam_size": best_entry["beam_size"],
            "no_repeat_ngram_size": best_entry["no_repeat_ngram_size"],
            "repetition_penalty": best_entry["repetition_penalty"],
            "length_penalty": best_entry["length_penalty"],
        },
    }

    report_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    best_eval_path.write_text(json.dumps(best_eval, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    best_samples_path.write_text(json.dumps(best_samples, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    summary = _run_search(args)
    print(json.dumps(summary["best"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
