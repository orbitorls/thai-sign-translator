"""Shipping gate for sentence-level SLT checkpoints."""
from __future__ import annotations

import argparse
import json
import os
import sys


_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC_ROOT = os.path.join(_REPO_ROOT, "src")
if _SRC_ROOT not in sys.path:
    sys.path.insert(0, _SRC_ROOT)

from tsl.eval.manifest_quality import ManifestQualityThresholds, analyze_manifest_quality
from tsl.eval.slt_metrics import evaluate_slt
from tsl.eval.slt_readiness import (
    SltReadinessThresholds,
    assess_slt_readiness,
    compute_source_metrics,
)
from tsl.inference.sentence_translator import SentenceTranslator
from tsl.train.train_slt import _load_data


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate SLT checkpoint readiness")
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument(
        "--stage",
        required=True,
        choices=["tsl51", "how2sign", "thaisignvis", "youtube_sl25", "combined", "finetune"],
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--beam-size", type=int, default=4)
    parser.add_argument("--max-len", type=int, default=64)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--min-val-chrf", type=float, default=80.0)
    parser.add_argument("--min-val-exact-match-pct", type=float, default=80.0)
    parser.add_argument("--max-best-val-loss", type=float, default=None)
    parser.add_argument("--min-target-overlap-ratio", type=float, default=0.05)
    parser.add_argument("--min-train-examples-per-target", type=float, default=1.25)
    parser.add_argument("--max-video-overlap", type=int, default=0)
    parser.add_argument("--report-json", default=None)
    return parser.parse_args(argv)


def _load_train_metrics(checkpoint_dir: str) -> dict:
    metrics_path = os.path.join(checkpoint_dir, "train_metrics.json")
    if not os.path.isfile(metrics_path):
        return {}
    with open(metrics_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _build_checkpoint_report(
    *,
    checkpoint_dir: str,
    stage: str,
    data_root: str,
    device: str,
    beam_size: int,
    max_len: int,
    limit: int | None,
    readiness_thresholds: SltReadinessThresholds,
    manifest_thresholds: ManifestQualityThresholds,
) -> dict:
    train_examples, val_examples, load_fn = _load_data(stage, data_root, limit)
    manifest_quality = analyze_manifest_quality(
        train_examples,
        val_examples,
        thresholds=manifest_thresholds,
    )

    overall_metrics = {
        "chrf": 0.0,
        "exact_match": 0,
        "exact_match_pct": 0.0,
        "n": 0,
        "mean_hyp_len": 0.0,
    }
    source_metrics: dict[str, dict] = {}

    if val_examples:
        translator = SentenceTranslator(checkpoint_dir, device=device)
        eval_results = evaluate_slt(
            translator,
            val_examples,
            load_fn,
            beam_size=beam_size,
            max_len=max_len,
            verbose=False,
        )
        source_metrics = compute_source_metrics(
            val_examples,
            eval_results["hypotheses"],
            eval_results["references"],
        )
        overall_metrics = {
            key: value
            for key, value in eval_results.items()
            if key not in {"hypotheses", "references"}
        }

    readiness = assess_slt_readiness(
        overall_metrics=overall_metrics,
        source_metrics=source_metrics,
        manifest_quality=manifest_quality,
        train_metrics=_load_train_metrics(checkpoint_dir),
        thresholds=readiness_thresholds,
    )
    readiness["checkpoint_dir"] = checkpoint_dir
    readiness["stage"] = stage
    readiness["data_root"] = data_root
    return readiness


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    readiness_thresholds = SltReadinessThresholds(
        min_val_chrf=args.min_val_chrf,
        min_val_exact_match_pct=args.min_val_exact_match_pct,
        max_best_val_loss=args.max_best_val_loss,
    )
    manifest_thresholds = ManifestQualityThresholds(
        max_video_overlap=args.max_video_overlap,
        min_target_overlap_ratio=args.min_target_overlap_ratio,
        min_train_examples_per_target=args.min_train_examples_per_target,
    )
    report = _build_checkpoint_report(
        checkpoint_dir=args.checkpoint_dir,
        stage=args.stage,
        data_root=args.data_root,
        device=args.device,
        beam_size=args.beam_size,
        max_len=args.max_len,
        limit=args.limit,
        readiness_thresholds=readiness_thresholds,
        manifest_thresholds=manifest_thresholds,
    )

    payload = json.dumps(report, ensure_ascii=True, indent=2)
    print(payload)
    if args.report_json:
        with open(args.report_json, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.write("\n")

    return 0 if report.get("ready", False) else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
