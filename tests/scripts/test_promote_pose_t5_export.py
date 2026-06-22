from __future__ import annotations

import json

from scripts.promote_pose_t5_export import main


def _write_json(path, payload) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_promote_when_no_incumbent(tmp_path):
    candidate_export = tmp_path / "candidate"
    candidate_export.mkdir()
    (candidate_export / "model.safetensors").write_text("candidate", encoding="utf-8")
    (candidate_export / "runtime_metadata.json").write_text(
        json.dumps({"export_dir": str(candidate_export)}),
        encoding="utf-8",
    )
    candidate_eval = tmp_path / "candidate_eval.json"
    candidate_samples = tmp_path / "candidate_samples.json"
    _write_json(
        candidate_eval,
        {
            "chrf": 15.0,
            "bleu": 12.0,
            "exact_match_pct": 4.0,
            "source_counts": {"tsl51": 50},
            "export_dir": str(candidate_export),
            "runtime_metadata": {"export_dir": str(candidate_export)},
        },
    )
    _write_json(candidate_samples, [{"id": 1}])

    stable_export = tmp_path / "stable"
    stable_eval = tmp_path / "stable_eval.json"
    stable_samples = tmp_path / "stable_samples.json"

    code = main(
        [
            "--candidate-export-dir",
            str(candidate_export),
            "--candidate-eval-json",
            str(candidate_eval),
            "--stable-export-dir",
            str(stable_export),
            "--stable-eval-json",
            str(stable_eval),
            "--candidate-samples-json",
            str(candidate_samples),
            "--stable-samples-json",
            str(stable_samples),
        ]
    )

    assert code == 0
    assert (stable_export / "model.safetensors").read_text(encoding="utf-8") == "candidate"
    stable_eval_payload = json.loads(stable_eval.read_text(encoding="utf-8"))
    assert stable_eval_payload["chrf"] == 15.0
    assert stable_eval_payload["export_dir"] == str(stable_export.resolve())
    assert stable_eval_payload["runtime_metadata"]["export_dir"] == str(stable_export.resolve())
    assert json.loads((stable_export / "runtime_metadata.json").read_text(encoding="utf-8"))["export_dir"] == str(stable_export.resolve())
    assert json.loads(stable_samples.read_text(encoding="utf-8"))[0]["id"] == 1


def test_does_not_promote_weaker_candidate(tmp_path):
    candidate_export = tmp_path / "candidate"
    candidate_export.mkdir()
    (candidate_export / "model.safetensors").write_text("candidate", encoding="utf-8")
    candidate_eval = tmp_path / "candidate_eval.json"
    _write_json(candidate_eval, {"chrf": 14.0, "bleu": 12.0, "exact_match_pct": 4.0, "source_counts": {"tsl51": 50}})

    stable_export = tmp_path / "stable"
    stable_export.mkdir()
    (stable_export / "model.safetensors").write_text("incumbent", encoding="utf-8")
    stable_eval = tmp_path / "stable_eval.json"
    _write_json(stable_eval, {"chrf": 15.0, "bleu": 10.0, "exact_match_pct": 2.0, "source_counts": {"tsl51": 50}})

    code = main(
        [
            "--candidate-export-dir",
            str(candidate_export),
            "--candidate-eval-json",
            str(candidate_eval),
            "--stable-export-dir",
            str(stable_export),
            "--stable-eval-json",
            str(stable_eval),
        ]
    )

    assert code == 0
    assert (stable_export / "model.safetensors").read_text(encoding="utf-8") == "incumbent"
    assert json.loads(stable_eval.read_text(encoding="utf-8"))["chrf"] == 15.0


def test_rejects_candidate_when_readiness_status_is_false(tmp_path):
    candidate_export = tmp_path / "candidate"
    candidate_export.mkdir()
    (candidate_export / "model.safetensors").write_text("candidate", encoding="utf-8")
    candidate_eval = tmp_path / "candidate_eval.json"
    _write_json(
        candidate_eval,
        {
            "chrf": 20.0,
            "bleu": 20.0,
            "exact_match_pct": 20.0,
            "source_counts": {"tsl51": 50},
            "promotion_status": {"ready": False, "failures": ["below threshold"]},
        },
    )

    stable_export = tmp_path / "stable"
    stable_export.mkdir()
    (stable_export / "model.safetensors").write_text("incumbent", encoding="utf-8")
    stable_eval = tmp_path / "stable_eval.json"
    _write_json(stable_eval, {"chrf": 10.0, "bleu": 10.0, "exact_match_pct": 10.0, "source_counts": {"tsl51": 50}})

    code = main(
        [
            "--candidate-export-dir",
            str(candidate_export),
            "--candidate-eval-json",
            str(candidate_eval),
            "--stable-export-dir",
            str(stable_export),
            "--stable-eval-json",
            str(stable_eval),
        ]
    )

    assert code == 0
    assert (stable_export / "model.safetensors").read_text(encoding="utf-8") == "incumbent"


def test_tie_breaks_on_bleu(tmp_path):
    candidate_export = tmp_path / "candidate"
    candidate_export.mkdir()
    (candidate_export / "model.safetensors").write_text("candidate", encoding="utf-8")
    candidate_eval = tmp_path / "candidate_eval.json"
    _write_json(candidate_eval, {"chrf": 15.0, "bleu": 13.0, "exact_match_pct": 4.0, "source_counts": {"tsl51": 50}})

    stable_export = tmp_path / "stable"
    stable_export.mkdir()
    (stable_export / "model.safetensors").write_text("incumbent", encoding="utf-8")
    stable_eval = tmp_path / "stable_eval.json"
    _write_json(stable_eval, {"chrf": 15.0, "bleu": 12.0, "exact_match_pct": 6.0, "source_counts": {"tsl51": 50}})

    code = main(
        [
            "--candidate-export-dir",
            str(candidate_export),
            "--candidate-eval-json",
            str(candidate_eval),
            "--stable-export-dir",
            str(stable_export),
            "--stable-eval-json",
            str(stable_eval),
        ]
    )

    assert code == 0
    assert (stable_export / "model.safetensors").read_text(encoding="utf-8") == "candidate"


def test_rejects_mixed_candidate_when_source_floor_fails(tmp_path):
    candidate_export = tmp_path / "candidate"
    candidate_export.mkdir()
    (candidate_export / "model.safetensors").write_text("candidate", encoding="utf-8")
    candidate_eval = tmp_path / "candidate_eval.json"
    _write_json(
        candidate_eval,
        {
            "chrf": 30.0,
            "bleu": 20.0,
            "exact_match_pct": 10.0,
            "seed": 42,
            "val_subset_size": 10,
            "data_roots": ["data/a", "data/b"],
            "source_metrics": {
                "tsl51": {"chrf": 70.0, "bleu": 65.0, "exact_match_pct": 40.0, "n": 5},
                "youtube": {"chrf": 10.0, "bleu": 5.0, "exact_match_pct": 0.0, "n": 5},
            },
        },
    )

    stable_export = tmp_path / "stable"
    stable_eval = tmp_path / "stable_eval.json"

    code = main(
        [
            "--candidate-export-dir",
            str(candidate_export),
            "--candidate-eval-json",
            str(candidate_eval),
            "--stable-export-dir",
            str(stable_export),
            "--stable-eval-json",
            str(stable_eval),
        ]
    )

    assert code == 0
    assert not stable_export.exists()


def test_rejects_mixed_candidate_when_source_regresses(tmp_path):
    candidate_export = tmp_path / "candidate"
    candidate_export.mkdir()
    (candidate_export / "model.safetensors").write_text("candidate", encoding="utf-8")
    candidate_eval = tmp_path / "candidate_eval.json"
    _write_json(
        candidate_eval,
        {
            "chrf": 31.0,
            "bleu": 21.0,
            "exact_match_pct": 11.0,
            "seed": 42,
            "val_subset_size": 10,
            "data_roots": ["data/a", "data/b"],
            "source_metrics": {
                "tsl51": {"chrf": 80.0, "bleu": 70.0, "exact_match_pct": 50.0, "n": 5},
                "youtube": {"chrf": 26.0, "bleu": 10.0, "exact_match_pct": 6.0, "n": 5},
            },
        },
    )

    stable_export = tmp_path / "stable"
    stable_export.mkdir()
    (stable_export / "model.safetensors").write_text("incumbent", encoding="utf-8")
    stable_eval = tmp_path / "stable_eval.json"
    _write_json(
        stable_eval,
        {
            "chrf": 30.0,
            "bleu": 19.0,
            "exact_match_pct": 10.0,
            "seed": 42,
            "val_subset_size": 10,
            "data_roots": ["data/a", "data/b"],
            "source_metrics": {
                "tsl51": {"chrf": 79.0, "bleu": 69.0, "exact_match_pct": 50.0, "n": 5},
                "youtube": {"chrf": 27.0, "bleu": 10.0, "exact_match_pct": 6.0, "n": 5},
            },
        },
    )

    code = main(
        [
            "--candidate-export-dir",
            str(candidate_export),
            "--candidate-eval-json",
            str(candidate_eval),
            "--stable-export-dir",
            str(stable_export),
            "--stable-eval-json",
            str(stable_eval),
        ]
    )

    assert code == 0
    assert (stable_export / "model.safetensors").read_text(encoding="utf-8") == "incumbent"


def test_promotes_mixed_candidate_when_all_sources_hold(tmp_path):
    candidate_export = tmp_path / "candidate"
    candidate_export.mkdir()
    (candidate_export / "model.safetensors").write_text("candidate", encoding="utf-8")
    candidate_eval = tmp_path / "candidate_eval.json"
    _write_json(
        candidate_eval,
        {
            "chrf": 31.0,
            "bleu": 21.0,
            "exact_match_pct": 11.0,
            "seed": 42,
            "val_subset_size": 10,
            "data_roots": ["data/a", "data/b"],
            "source_metrics": {
                "tsl51": {"chrf": 80.0, "bleu": 70.0, "exact_match_pct": 50.0, "n": 5},
                "youtube": {"chrf": 28.0, "bleu": 10.0, "exact_match_pct": 6.0, "n": 5},
            },
        },
    )

    stable_export = tmp_path / "stable"
    stable_export.mkdir()
    (stable_export / "model.safetensors").write_text("incumbent", encoding="utf-8")
    stable_eval = tmp_path / "stable_eval.json"
    _write_json(
        stable_eval,
        {
            "chrf": 30.0,
            "bleu": 19.0,
            "exact_match_pct": 10.0,
            "seed": 42,
            "val_subset_size": 10,
            "data_roots": ["data/a", "data/b"],
            "source_metrics": {
                "tsl51": {"chrf": 79.0, "bleu": 69.0, "exact_match_pct": 50.0, "n": 5},
                "youtube": {"chrf": 27.0, "bleu": 10.0, "exact_match_pct": 6.0, "n": 5},
            },
        },
    )

    code = main(
        [
            "--candidate-export-dir",
            str(candidate_export),
            "--candidate-eval-json",
            str(candidate_eval),
            "--stable-export-dir",
            str(stable_export),
            "--stable-eval-json",
            str(stable_eval),
        ]
    )

    assert code == 0
    assert (stable_export / "model.safetensors").read_text(encoding="utf-8") == "candidate"


def test_rejects_promotion_when_mixed_source_evidence_is_missing(tmp_path):
    candidate_export = tmp_path / "candidate"
    candidate_export.mkdir()
    (candidate_export / "model.safetensors").write_text("candidate", encoding="utf-8")
    candidate_eval = tmp_path / "candidate_eval.json"
    _write_json(
        candidate_eval,
        {
            "chrf": 31.0,
            "bleu": 21.0,
            "exact_match_pct": 11.0,
            "data_roots": ["data/a", "data/b"],
            "seed": 42,
            "val_subset_size": 10,
        },
    )

    stable_export = tmp_path / "stable"
    stable_export.mkdir()
    (stable_export / "model.safetensors").write_text("incumbent", encoding="utf-8")
    stable_eval = tmp_path / "stable_eval.json"
    _write_json(
        stable_eval,
        {
            "chrf": 30.0,
            "bleu": 20.0,
            "exact_match_pct": 10.0,
            "data_roots": ["data/a", "data/b"],
            "seed": 42,
            "val_subset_size": 10,
        },
    )

    code = main(
        [
            "--candidate-export-dir",
            str(candidate_export),
            "--candidate-eval-json",
            str(candidate_eval),
            "--stable-export-dir",
            str(stable_export),
            "--stable-eval-json",
            str(stable_eval),
        ]
    )

    assert code == 0
    assert (stable_export / "model.safetensors").read_text(encoding="utf-8") == "incumbent"
