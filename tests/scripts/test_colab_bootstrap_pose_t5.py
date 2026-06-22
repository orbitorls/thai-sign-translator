from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from scripts.colab_bootstrap_pose_t5 import (
    _build_train_command,
    _extract_local_data_bundle,
    _ensure_manifest_dataset,
    _materialize_archived_dataset,
    _prune_restored_resume_artifacts,
    _reset_out_dir,
    _restore_resume_checkpoint,
    main,
)


def test_reset_out_dir_removes_prior_training_artifacts(tmp_path):
    out_dir = tmp_path / "ckpts"
    out_dir.mkdir()
    for name in [
        "ckpt_step00002000.pt",
        "train.log",
        "train_metrics.json",
        "publisher_state.json",
        "model.safetensors",
        "pose_t5_config.json",
    ]:
        (out_dir / name).write_text("x", encoding="utf-8")

    _reset_out_dir(out_dir)

    assert list(out_dir.iterdir()) == []


def test_restore_resume_checkpoint_logs_when_no_checkpoint_files_restored(tmp_path, monkeypatch, capsys):
    out_dir = tmp_path / "ckpts"
    out_dir.mkdir()

    monkeypatch.setattr("scripts.colab_bootstrap_pose_t5._ensure_pip_package", lambda *_a, **_k: None)
    monkeypatch.setattr("scripts.colab_bootstrap_pose_t5._download_dataset", lambda *_a, **_k: None)

    _restore_resume_checkpoint(out_dir, "orbitorls/thai-sign-ckpt")

    output = capsys.readouterr().out
    assert "no ckpt_step*.pt files were found" in output


def test_prune_restored_resume_artifacts_keeps_only_checkpoints(tmp_path):
    out_dir = tmp_path / "ckpts"
    out_dir.mkdir()
    for name in [
        "ckpt_step00001000.pt",
        "ckpt_step00001500.pt",
        "train.log",
        "train_metrics.json",
        "launch.json",
        "best_checkpoint.txt",
        "latest_checkpoint.txt",
        "model.safetensors",
    ]:
        (out_dir / name).write_text("x", encoding="utf-8")

    removed = _prune_restored_resume_artifacts(out_dir)

    assert removed == [
        "best_checkpoint.txt",
        "latest_checkpoint.txt",
        "launch.json",
        "model.safetensors",
        "train.log",
        "train_metrics.json",
    ]
    assert sorted(path.name for path in out_dir.iterdir()) == [
        "ckpt_step00001000.pt",
        "ckpt_step00001500.pt",
    ]


def test_main_fails_fast_when_resume_requested_but_no_checkpoint_restored(tmp_path, monkeypatch):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    out_dir = tmp_path / "out"
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "data_datasets": {},
                "out_dir": str(out_dir),
                "resume": "auto",
                "require_resume_checkpoint": True,
                "checkpoint_dataset_slug": "orbitorls/thai-sign-ckpt",
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr("scripts.colab_bootstrap_pose_t5._ensure_repo_available", lambda *_a, **_k: repo_root)
    monkeypatch.setattr("scripts.colab_bootstrap_pose_t5._ensure_kaggle_access_token", lambda *_a, **_k: None)
    monkeypatch.setattr("scripts.colab_bootstrap_pose_t5._restore_resume_checkpoint", lambda *_a, **_k: None)

    with pytest.raises(RuntimeError, match="Resume was requested"):
        main(["--config-path", str(config_path)])


def test_ensure_repo_available_refreshes_existing_repo_from_zip(tmp_path):
    from scripts.colab_bootstrap_pose_t5 import _ensure_repo_available

    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / "scripts").mkdir()
    (repo_root / "scripts" / "colab_train_pose_t5.py").write_text("old\n", encoding="utf-8")
    repo_zip = tmp_path / "thai-sign-code.zip"

    with zipfile.ZipFile(repo_zip, "w") as archive:
        archive.writestr("scripts/colab_train_pose_t5.py", "new\n")

    resolved = _ensure_repo_available(str(repo_root), str(repo_zip))

    assert resolved == repo_root
    assert (repo_root / "scripts" / "colab_train_pose_t5.py").read_text(encoding="utf-8") == "new\n"


def test_extract_local_data_bundle_restores_manifest(tmp_path):
    bundle_zip = tmp_path / "bundle.zip"
    with zipfile.ZipFile(bundle_zip, "w") as archive:
        archive.writestr("manifest.csv", "segment_id,npy_path,text,feature_layout_version\ns1,landmarks/s1.npy,a,v3-312\n")
        archive.writestr("landmarks/s1.npy", b"fake")

    target_dir = tmp_path / "out"
    resolved = _extract_local_data_bundle(str(bundle_zip), target_dir)

    assert resolved == target_dir
    assert (target_dir / "manifest.csv").is_file()
    assert (target_dir / "landmarks" / "s1.npy").is_file()


def test_materialize_archived_dataset_extracts_features_zip(tmp_path):
    target_dir = tmp_path / "dataset"
    target_dir.mkdir()
    (target_dir / "manifest.csv").write_text(
        "segment_id,npy_path,text,feature_layout_version\ns1,features/seg_00000.npy,a,v3-312\n",
        encoding="utf-8",
    )
    bundle_zip = target_dir / "features.zip"
    with zipfile.ZipFile(bundle_zip, "w") as archive:
        archive.writestr("seg_00000.npy", b"fake")

    resolved = _materialize_archived_dataset(target_dir)

    assert resolved == target_dir
    assert (target_dir / "features" / "seg_00000.npy").is_file()


def test_materialize_archived_dataset_flattens_nested_features_dir(tmp_path):
    target_dir = tmp_path / "dataset"
    nested_dir = target_dir / "features" / "features"
    nested_dir.mkdir(parents=True)
    (target_dir / "manifest.csv").write_text(
        "segment_id,npy_path,text,feature_layout_version\ns1,features/seg_00000.npy,a,v3-312\n",
        encoding="utf-8",
    )
    (nested_dir / "seg_00000.npy").write_bytes(b"fake")

    resolved = _materialize_archived_dataset(target_dir)

    assert resolved == target_dir
    assert (target_dir / "features" / "seg_00000.npy").is_file()
    assert not (target_dir / "features" / "features").exists()


def test_ensure_manifest_dataset_refreshes_stale_manifest_layout(tmp_path, monkeypatch):
    target_dir = tmp_path / "dataset"
    target_dir.mkdir()
    (target_dir / "manifest.csv").write_text(
        "segment_id,npy_path,text,feature_layout_version\ns1,features/seg_00000.npy,a,v3-312\n",
        encoding="utf-8",
    )
    (target_dir / "features").mkdir()
    (target_dir / "features" / "seg_00000.npy").write_bytes(b"stale")

    monkeypatch.setattr("scripts.colab_bootstrap_pose_t5._ensure_pip_package", lambda *_a, **_k: None)

    def _fake_download(_slug: str, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "manifest.csv").write_text(
            "segment_id,npy_path,text,feature_layout_version\ns1,features/seg_00000.npy,a,v3-312\n",
            encoding="utf-8",
        )
        with zipfile.ZipFile(out_dir / "features.zip", "w") as archive:
            archive.writestr("seg_00000.npy", b"fresh")

    monkeypatch.setattr("scripts.colab_bootstrap_pose_t5._download_dataset", _fake_download)

    resolved = _ensure_manifest_dataset(target_dir, "orbitorls/demo")

    assert resolved == target_dir
    assert (target_dir / "features" / "seg_00000.npy").is_file()


def test_build_train_command_passes_manifest_quality_guards(tmp_path):
    repo_root = tmp_path / "repo"
    (repo_root / "scripts").mkdir(parents=True)
    (repo_root / "scripts" / "colab_train_pose_t5.py").write_text("print('ok')\n", encoding="utf-8")
    out_dir = tmp_path / "out"

    cmd = _build_train_command(
        repo_root,
        {
            "required_sources": "tsl51,thaisignvis",
            "manifest_quality_sources": "tsl51",
            "fail_on_manifest_quality": "true",
            "allow_noop_resume": "false",
        },
        ["/content/kaggle/mixed_all_train_v6"],
        out_dir,
    )

    assert "--required-sources" in cmd
    assert cmd[cmd.index("--required-sources") + 1] == "tsl51,thaisignvis"
    assert cmd[cmd.index("--manifest-quality-sources") + 1] == "tsl51"
    assert cmd[cmd.index("--fail-on-manifest-quality") + 1] == "true"
    assert cmd[cmd.index("--allow-noop-resume") + 1] == "false"
