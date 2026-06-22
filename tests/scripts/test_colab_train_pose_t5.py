from __future__ import annotations

import json
import zipfile
from argparse import Namespace
from pathlib import Path

import pytest

from scripts.colab_train_pose_t5 import (
    _apply_overrides,
    _build_parser,
    _ensure_repo_available,
    _load_config_overrides,
    _resolve_data_roots,
)


def test_parser_defaults_match_round2_training_plan():
    args = _build_parser().parse_args([])

    assert args.lr == pytest.approx(1e-4)
    assert args.dropout == pytest.approx(0.3)
    assert args.weight_decay == pytest.approx(0.05)
    assert args.early_stopping_patience == 10
    assert args.eval_steps == 100
    assert args.checkpoint_steps == 500
    assert args.required_sources == "tsl51,thaisignvis"
    assert args.fail_on_manifest_quality == "true"
    assert args.allow_noop_resume == "false"


def test_load_config_overrides_returns_empty_when_missing(tmp_path):
    assert _load_config_overrides(str(tmp_path / "missing.json")) == {}


def test_load_config_overrides_reads_json_object(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"lr": 5e-5, "dropout": 0.25}), encoding="utf-8")

    assert _load_config_overrides(str(config_path)) == {"lr": 5e-5, "dropout": 0.25}


def test_apply_overrides_only_updates_known_fields():
    args = Namespace(lr=1e-4, dropout=0.3)

    updated = _apply_overrides(args, {"lr": 5e-5, "unknown": 123})

    assert updated.lr == pytest.approx(5e-5)
    assert not hasattr(updated, "unknown")


def test_resolve_data_roots_only_returns_present_manifests(tmp_path):
    mixed = tmp_path / "data" / "mixed_all_train_v6"
    tsl51 = tmp_path / "data" / "tsl51_v3"
    thaisignvis = tmp_path / "data" / "thaisignvis_v3_probe"
    for root in (mixed, tsl51, thaisignvis):
        root.mkdir(parents=True)
        (root / "manifest.csv").write_text("segment_id\n", encoding="utf-8")

    roots = _resolve_data_roots(str(tmp_path))

    assert roots == ",".join([str(mixed), str(tsl51), str(thaisignvis)])


def test_ensure_repo_available_extracts_zip_when_repo_missing(tmp_path):
    repo_zip = tmp_path / "thai-sign-code.zip"
    repo_root = tmp_path / "repo"

    with zipfile.ZipFile(repo_zip, "w") as archive:
        archive.writestr("src/tsl/train/train_pose_t5.py", "print('ok')\n")
        archive.writestr("requirements.txt", "torch>=2.2.0\n")

    resolved = _ensure_repo_available(str(repo_root), str(repo_zip))

    assert resolved == repo_root
    assert (repo_root / "src" / "tsl" / "train" / "train_pose_t5.py").is_file()


def test_parser_accepts_explicit_data_roots_and_out_dir():
    args = _build_parser().parse_args(
        [
            "--data-roots",
            "/content/tsl51_v3,/content/youtube_sl25_thai_v3",
            "--out-dir",
            "/content/checkpoints/pose_t5_v3",
            "--checkpoint-steps",
            "750",
        ]
    )

    assert args.data_roots == "/content/tsl51_v3,/content/youtube_sl25_thai_v3"
    assert args.out_dir == "/content/checkpoints/pose_t5_v3"
    assert args.checkpoint_steps == 750
