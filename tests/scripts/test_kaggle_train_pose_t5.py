from __future__ import annotations

import json
import io
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd

from argparse import Namespace

from scripts.kaggle_train_pose_t5 import (
    _build_kaggle_parser,
    _merge_preflight_reports,
    _run_smoke_training,
    _validate_eval_readiness,
    _prepare_data_roots,
)
from scripts.verify_pose_t5_cloud_preflight import verify_cloud_preflight


def test_prepare_data_roots_keeps_unarchived_roots(tmp_path):
    root = tmp_path / "dataset"
    root.mkdir()
    (root / "manifest.csv").write_text("segment_id,npy_path,text,feature_layout_version\n", encoding="utf-8")

    result = _prepare_data_roots(str(root), work_root=str(tmp_path / "work"))

    assert result == str(root.resolve())


def test_prepare_data_roots_extracts_archived_features_to_work_root(tmp_path):
    root = tmp_path / "dataset"
    root.mkdir()
    pd.DataFrame(
        [
            {
                "segment_id": "seg001",
                "npy_path": "features/seg001.npy",
                "text": "alpha",
                "split": "train",
                "source": "tsl51",
                "feature_layout_version": "v3-312",
            }
        ]
    ).to_csv(root / "manifest.csv", index=False)
    (root / "manifest_quality.json").write_text(json.dumps({"passed": True}), encoding="utf-8")
    with zipfile.ZipFile(root / "features.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("seg001.npy", b"demo-bytes")

    work_root = tmp_path / "work"
    result = _prepare_data_roots(str(root), work_root=str(work_root))

    staged_root = Path(result)
    assert staged_root == work_root / "staged_inputs" / root.name
    assert (staged_root / "manifest.csv").is_file()
    assert (staged_root / "manifest_quality.json").is_file()
    assert (staged_root / "features" / "seg001.npy").read_bytes() == b"demo-bytes"


def test_verify_cloud_preflight_accepts_expected_counts(tmp_path):
    root = tmp_path / "dataset"
    root.mkdir()
    features_dir = root / "features"
    features_dir.mkdir()
    np.save(features_dir / "seg001.npy", np.ones((3, 312), dtype=np.float32))
    np.save(features_dir / "seg002.npy", np.ones((4, 312), dtype=np.float32))
    pd.DataFrame(
        [
            {
                "segment_id": "seg001",
                "npy_path": "features/seg001.npy",
                "text": "alpha",
                "split": "train",
                "source": "tsl51",
                "feature_layout_version": "v3-312",
            },
            {
                "segment_id": "seg002",
                "npy_path": "features/seg002.npy",
                "text": "beta",
                "split": "val",
                "source": "youtube_sl25_thai",
                "feature_layout_version": "v3-312",
            },
        ]
    ).to_csv(root / "manifest.csv", index=False)
    (root / "manifest_quality.json").write_text(json.dumps({"passed": True}), encoding="utf-8")

    report = verify_cloud_preflight(
        str(root),
        expected_manifest_rows=2,
        expected_resolved_examples=2,
        expected_source_counts="tsl51=1,youtube_sl25_thai=1",
    )

    assert report["passed"] is True
    assert report["aggregate_resolved_examples"] == 2


def test_verify_cloud_preflight_accepts_archived_dataset_roots(tmp_path):
    root = tmp_path / "dataset"
    root.mkdir()
    pd.DataFrame(
        [
            {
                "segment_id": "seg001",
                "npy_path": "features/seg001.npy",
                "text": "alpha",
                "split": "train",
                "source": "tsl51",
                "feature_layout_version": "v3-312",
            }
        ]
    ).to_csv(root / "manifest.csv", index=False)
    (root / "manifest_quality.json").write_text(json.dumps({"passed": True}), encoding="utf-8")
    buffer = io.BytesIO()
    np.save(buffer, np.ones((3, 312), dtype=np.float32))
    with zipfile.ZipFile(root / "features.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("seg001.npy", buffer.getvalue())

    report = verify_cloud_preflight(
        str(root),
        expected_manifest_rows=1,
        expected_resolved_examples=1,
        expected_source_counts="tsl51=1",
        required_files=("manifest.csv", "manifest_quality.json", "features.zip"),
    )

    assert report["passed"] is True
    assert report["aggregate_resolved_examples"] == 1


def test_verify_cloud_preflight_rejects_mismatched_resolved_counts(tmp_path):
    root = tmp_path / "dataset"
    root.mkdir()
    pd.DataFrame(
        [
            {
                "segment_id": "seg001",
                "npy_path": "features/seg001.npy",
                "text": "alpha",
                "split": "train",
                "source": "tsl51",
                "feature_layout_version": "v3-312",
            }
        ]
    ).to_csv(root / "manifest.csv", index=False)
    (root / "manifest_quality.json").write_text(json.dumps({"passed": True}), encoding="utf-8")

    report = verify_cloud_preflight(
        str(root),
        expected_manifest_rows=1,
        expected_resolved_examples=1,
        expected_source_counts="tsl51=1",
    )

    assert report["passed"] is False
    assert any("resolved examples 0 != expected 1" in failure for failure in report["failures"])


def test_kaggle_parser_defaults_to_gpu_smoke_and_readiness_gates():
    args = _build_kaggle_parser().parse_args([])

    assert args.require_gpu == "true"
    assert args.preflight_only == "false"
    assert args.smoke_steps == 20
    assert args.min_val_chrf == 80.0
    assert args.min_val_exact_match_pct == 80.0


def test_merge_preflight_reports_fails_if_raw_or_staged_fails():
    report = _merge_preflight_reports(
        {"passed": True, "failures": []},
        {"passed": False, "failures": ["staged count mismatch"]},
    )

    assert report["passed"] is False
    assert report["failures"] == ["staged count mismatch"]


def test_run_smoke_training_requires_max_step_stop_and_export(tmp_path):
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    (out_dir / "seed.pt").write_text("seed", encoding="utf-8")
    args = Namespace(
        out_dir=str(out_dir),
        smoke_steps=3,
        epochs=100,
        eval_steps=200,
        checkpoint_steps=200,
    )

    def _fake_train(train_args):
        smoke_dir = Path(train_args.out_dir)
        assert smoke_dir.name == "_smoke"
        assert train_args.max_train_steps == 3
        assert train_args.require_gpu is True
        assert (smoke_dir / "seed.pt").read_text(encoding="utf-8") == "seed"
        (smoke_dir / "pose_t5_config.json").write_text("{}", encoding="utf-8")
        (smoke_dir / "best_model_state.pt").write_text("checkpoint", encoding="utf-8")
        return {"stopped_reason": "max_train_steps", "new_optimizer_steps": 3}

    result = _run_smoke_training(args, train_main=_fake_train)

    assert result["stopped_reason"] == "max_train_steps"


def test_validate_eval_readiness_rejects_not_ready_reports():
    try:
        _validate_eval_readiness({"promotion_status": {"ready": False, "failures": ["low chrF"]}})
    except RuntimeError as exc:
        assert "low chrF" in str(exc)
    else:
        raise AssertionError("expected not-ready eval report to fail")
