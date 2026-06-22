from __future__ import annotations

import json

from scripts.refresh_pose_t5_verified import main


def test_refresh_pose_t5_verified_runs_export_eval_promote(monkeypatch):
    calls: list[tuple[str, object]] = []

    def _fake_export(args):
        calls.append(("export", args))
        return {"checkpoint_step": 2100}

    def _fake_evaluate(args):
        calls.append(("evaluate", args))
        assert hasattr(args, "split_policy")
        assert hasattr(args, "required_sources")
        assert hasattr(args, "report_data_roots")
        assert hasattr(args, "manifest_quality_sources")
        assert hasattr(args, "eval_sources")
        assert hasattr(args, "max_new_tokens")
        assert hasattr(args, "beam_size")
        assert hasattr(args, "no_repeat_ngram_size")
        assert hasattr(args, "repetition_penalty")
        assert hasattr(args, "length_penalty")
        assert hasattr(args, "min_val_chrf")
        assert hasattr(args, "min_val_exact_match_pct")
        return {"chrf": 15.0}

    def _fake_promote(args):
        calls.append(("promote", args))
        return {"promoted": True}

    monkeypatch.setattr("scripts.refresh_pose_t5_verified._export_checkpoint", _fake_export)
    monkeypatch.setattr("scripts.refresh_pose_t5_verified._evaluate_export", _fake_evaluate)
    monkeypatch.setattr("scripts.refresh_pose_t5_verified._promote", _fake_promote)

    code = main(
        [
            "--train-dir",
            "train",
            "--candidate-export-dir",
            "candidate",
            "--verified-export-dir",
            "verified",
            "--split-policy",
            "manifest",
            "--required-sources",
            "tsl51",
            "--report-data-roots",
            "data/logical",
            "--manifest-quality-sources",
            "tsl51",
            "--eval-sources",
            "tsl51",
            "--max-new-tokens",
            "48",
            "--beam-size",
            "1",
            "--no-repeat-ngram-size",
            "none",
            "--repetition-penalty",
            "none",
            "--length-penalty",
            "1.0",
            "--min-val-chrf",
            "90",
            "--min-val-exact-match-pct",
            "75",
        ]
    )

    assert code == 0
    assert [name for name, _ in calls] == ["export", "evaluate", "promote"]
    export_args = calls[0][1]
    evaluate_args = calls[1][1]
    promote_args = calls[2][1]
    assert export_args.train_dir == "train"
    assert evaluate_args.export_dir == "candidate"
    assert evaluate_args.split_policy == "manifest"
    assert evaluate_args.required_sources == "tsl51"
    assert evaluate_args.report_data_roots == "data/logical"
    assert evaluate_args.eval_sources == "tsl51"
    assert evaluate_args.min_val_chrf == 90.0
    assert evaluate_args.min_val_exact_match_pct == 75.0
    assert promote_args.stable_export_dir == "verified"
    assert promote_args.min_source_examples == 5
    assert promote_args.min_source_chrf == 20.0
    assert promote_args.min_source_exact_match_pct == 5.0
