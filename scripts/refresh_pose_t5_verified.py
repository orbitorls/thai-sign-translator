"""Refresh the verified PoseT5 runtime artifact from the current training lane."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from scripts.evaluate_pose_t5_export import _evaluate_export
from scripts.export_pose_t5_checkpoint import _export_checkpoint
from scripts.promote_pose_t5_export import _promote


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export, evaluate, and promote the current PoseT5 best-state artifact.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--train-dir", default="checkpoints/pose_t5_rtx4060_resume_best")
    parser.add_argument("--candidate-export-dir", default="checkpoints/pose_t5_rtx4060_best_export_auto")
    parser.add_argument("--verified-export-dir", default="checkpoints/pose_t5_rtx4060_best_export_verified")
    parser.add_argument("--candidate-eval-json", default="checkpoints/pose_t5_rtx4060_best_export_auto_eval.json")
    parser.add_argument("--verified-eval-json", default="checkpoints/pose_t5_rtx4060_best_export_verified_eval.json")
    parser.add_argument("--candidate-samples-json", default="checkpoints/pose_t5_rtx4060_best_export_auto_samples.json")
    parser.add_argument("--verified-samples-json", default="checkpoints/pose_t5_rtx4060_best_export_verified_samples.json")
    parser.add_argument("--checkpoint", default="best_state")
    parser.add_argument("--data-roots", default="data/tsl51_v3,data/youtube_sl25_thai_v3")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-subset-size", type=int, default=50)
    parser.add_argument("--split-policy", default="auto", choices=["auto", "manifest", "video"])
    parser.add_argument("--required-sources", default="")
    parser.add_argument("--report-data-roots", default="")
    parser.add_argument("--manifest-quality-sources", default="")
    parser.add_argument("--eval-sources", default="")
    parser.add_argument("--max-new-tokens", type=int, default=72)
    parser.add_argument("--beam-size", type=int, default=5)
    parser.add_argument("--no-repeat-ngram-size", default="3")
    parser.add_argument("--repetition-penalty", default="1.5")
    parser.add_argument("--length-penalty", default="0.7")
    parser.add_argument("--min-val-chrf", type=float, default=80.0)
    parser.add_argument("--min-val-exact-match-pct", type=float, default=80.0)
    parser.add_argument("--base-model", default="google/mt5-small")
    parser.add_argument("--input-dim", type=int, default=312)
    parser.add_argument("--num-encoder-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--downsample-factor", type=int, default=4)
    parser.add_argument("--min-source-examples", type=int, default=5)
    parser.add_argument("--min-source-chrf", type=float, default=20.0)
    parser.add_argument("--min-source-exact-match-pct", type=float, default=5.0)
    parser.add_argument("--force-promote", action="store_true")
    return parser


def _refresh(args: argparse.Namespace) -> dict:
    export_report = _export_checkpoint(
        argparse.Namespace(
            train_dir=args.train_dir,
            export_dir=args.candidate_export_dir,
            checkpoint=args.checkpoint,
            base_model=args.base_model,
            input_dim=args.input_dim,
            num_encoder_layers=args.num_encoder_layers,
            dropout=args.dropout,
            downsample_factor=args.downsample_factor,
            report_json=None,
        )
    )
    eval_report = _evaluate_export(
        argparse.Namespace(
            export_dir=args.candidate_export_dir,
            data_roots=args.data_roots,
            device=args.device,
            seed=args.seed,
            val_subset_size=args.val_subset_size,
            split_policy=args.split_policy,
            required_sources=args.required_sources,
            report_data_roots=args.report_data_roots,
            manifest_quality_sources=args.manifest_quality_sources,
            eval_sources=args.eval_sources,
            max_new_tokens=args.max_new_tokens,
            beam_size=args.beam_size,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
            repetition_penalty=args.repetition_penalty,
            length_penalty=args.length_penalty,
            min_val_chrf=args.min_val_chrf,
            min_val_exact_match_pct=args.min_val_exact_match_pct,
            report_json=args.candidate_eval_json,
            samples_json=args.candidate_samples_json,
        )
    )
    promote_report = _promote(
        argparse.Namespace(
            candidate_export_dir=args.candidate_export_dir,
            candidate_eval_json=args.candidate_eval_json,
            stable_export_dir=args.verified_export_dir,
            stable_eval_json=args.verified_eval_json,
            candidate_samples_json=args.candidate_samples_json,
            stable_samples_json=args.verified_samples_json,
            min_source_examples=args.min_source_examples,
            min_source_chrf=args.min_source_chrf,
            min_source_exact_match_pct=args.min_source_exact_match_pct,
            force=args.force_promote,
        )
    )
    return {
        "export": export_report,
        "evaluation": eval_report,
        "promotion": promote_report,
    }


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    result = _refresh(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
