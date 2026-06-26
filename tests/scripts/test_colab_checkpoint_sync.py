from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

from scripts.colab_checkpoint_sync import (
    _acquire_sync_lock,
    _copy_dataset_snapshot,
    _download_file,
    _download_text_file_via_exec,
    _release_sync_lock,
    _refresh_dataset_dir_from_kaggle,
    _get_session_status,
    _sanitize_error_text,
    _sanitize_exec_text_output,
    _seed_mirror_from_dataset_dir,
    best_checkpoint_name,
    checkpoint_step,
    _sync_remote_files,
    ensure_dataset_metadata,
    latest_checkpoint_name,
    parse_ls_output,
    parse_status_output,
    publish_latest_checkpoint,
)


def test_parse_ls_output_ignores_directories_and_blank_lines():
    text = """
    subdir/
    ckpt_step00000500.pt

    train.log
    """

    assert parse_ls_output(text) == ["ckpt_step00000500.pt", "train.log"]


def test_parse_status_output_extracts_hardware_and_state():
    text = """
    [thai-sign-train-managed-r5] gpu-t4-s-kkb-usw1b0-3uddyj88dhm0k | Hardware: T4 | Variant: GPU | Status: IDLE
      Last Execution: /mnt/d/New folder/thai-sign-translator/scripts/colab_bootstrap_pose_t5.py at 2026-06-20 01:49:48
    """

    parsed = parse_status_output(text)

    assert parsed["session"] == "thai-sign-train-managed-r5"
    assert parsed["hardware"] == "T4"
    assert parsed["variant"] == "GPU"
    assert parsed["status"] == "IDLE"
    assert parsed["last_execution"] == "/mnt/d/New folder/thai-sign-translator/scripts/colab_bootstrap_pose_t5.py at 2026-06-20 01:49:48"


def test_parse_status_output_treats_not_found_as_missing_session():
    parsed = parse_status_output("[colab] Session 'thai-sign-train-managed-r6' not found.\n")

    assert parsed["raw"] == "[colab] Session 'thai-sign-train-managed-r6' not found."
    assert parsed["session"] is None
    assert parsed["hardware"] is None
    assert parsed["status"] is None


def test_sanitize_error_text_redacts_colab_proxy_token():
    text = (
        "500 Server Error for url: https://example.invalid/api?"
        "authuser=0&colab-runtime-proxy-token=secret.jwt.value&content=1"
    )

    sanitized = _sanitize_error_text(text)

    assert "secret.jwt.value" not in sanitized
    assert "colab-runtime-proxy-token=<redacted>" in sanitized
    assert "content=1" in sanitized


def test_latest_checkpoint_name_picks_highest_step():
    names = [
        "ckpt_step00000100.pt",
        "ckpt_step00000500.pt",
        "ckpt_step00000300.pt",
    ]

    assert latest_checkpoint_name(names) == "ckpt_step00000500.pt"
    assert checkpoint_step("ckpt_step00000500.pt") == 500


def test_ensure_dataset_metadata_writes_expected_file(tmp_path):
    dataset_dir = tmp_path / "thai-sign-ckpt"

    ensure_dataset_metadata(dataset_dir, "orbitorls/thai-sign-ckpt", "Thai Sign Ckpt")

    metadata = json.loads((dataset_dir / "dataset-metadata.json").read_text(encoding="utf-8"))
    assert metadata["id"] == "orbitorls/thai-sign-ckpt"
    assert metadata["title"] == "Thai Sign Ckpt"


def test_publish_latest_checkpoint_keeps_only_latest_in_dataset_dir(tmp_path):
    mirror_dir = tmp_path / "mirror"
    dataset_dir = tmp_path / "dataset"
    mirror_dir.mkdir()

    old_ckpt = mirror_dir / "ckpt_step00000500.pt"
    new_ckpt = mirror_dir / "ckpt_step00001000.pt"
    old_ckpt.write_bytes(b"old")
    new_ckpt.write_bytes(b"new")
    (mirror_dir / "train_metrics.json").write_text("{}", encoding="utf-8")

    published = publish_latest_checkpoint(
        mirror_dir,
        dataset_dir,
        "orbitorls/thai-sign-ckpt",
        "Thai Sign Ckpt",
    )

    assert published == "ckpt_step00001000.pt"
    assert not (dataset_dir / old_ckpt.name).exists()
    assert (dataset_dir / new_ckpt.name).read_bytes() == b"new"
    assert (dataset_dir / "latest_checkpoint.txt").read_text(encoding="utf-8") == new_ckpt.name


def test_best_checkpoint_name_prefers_highest_val_chrf(tmp_path):
    low = tmp_path / "ckpt_step00001000.pt"
    high = tmp_path / "ckpt_step00001500.pt"
    torch.save({"metrics": {"val_chrf": 9.7}}, low)
    torch.save({"metrics": {"val_chrf": 13.6}}, high)

    assert best_checkpoint_name([low, high]) == "ckpt_step00001500.pt"


def test_sync_remote_files_prioritizes_latest_checkpoint_and_keeps_partial_progress(tmp_path, monkeypatch):
    mirror_dir = tmp_path / "mirror"
    mirror_dir.mkdir()
    calls: list[str] = []

    def fake_download(_colab_bin: str, _session_name: str, remote_path: str, local_path: Path) -> None:
        calls.append(Path(remote_path).name)
        if remote_path.endswith("ckpt_step00000500.pt"):
            raise RuntimeError("transient")
        local_path.write_text("ok", encoding="utf-8")

    monkeypatch.setattr("scripts.colab_checkpoint_sync._download_file", fake_download)

    downloaded, failures = _sync_remote_files(
        "/root/.venvs/colabcli/bin/colab",
        "thai-sign-train-managed",
        "/content/checkpoints/pose_t5_v3_colab",
        mirror_dir,
        ["ckpt_step00000500.pt", "ckpt_step00001500.pt", "train.log", "launch.json"],
        download_retries=1,
        retry_delay_sec=0,
    )

    assert calls[:3] == ["launch.json", "train.log", "ckpt_step00001500.pt"]
    assert "ckpt_step00001500.pt" in downloaded
    assert failures == {"ckpt_step00000500.pt": "transient"}


def test_sync_remote_files_downloads_final_export_files(tmp_path, monkeypatch):
    mirror_dir = tmp_path / "mirror"
    mirror_dir.mkdir()
    calls: list[str] = []

    def fake_download(_colab_bin: str, _session_name: str, remote_path: str, local_path: Path) -> None:
        calls.append(Path(remote_path).name)
        local_path.write_text("ok", encoding="utf-8")

    monkeypatch.setattr("scripts.colab_checkpoint_sync._download_file", fake_download)

    downloaded, failures = _sync_remote_files(
        "/root/.venvs/colabcli/bin/colab",
        "thai-sign-train-managed",
        "/content/checkpoints/pose_t5_v3_colab",
        mirror_dir,
        ["model.safetensors", "pose_t5_config.json", "train.log"],
        download_retries=1,
        retry_delay_sec=0,
    )

    assert calls == ["train.log", "model.safetensors", "pose_t5_config.json"]
    assert downloaded == ["train.log", "model.safetensors", "pose_t5_config.json"]
    assert failures == {}


def test_sync_remote_files_reports_progress_after_each_result(tmp_path, monkeypatch):
    mirror_dir = tmp_path / "mirror"
    mirror_dir.mkdir()
    snapshots: list[tuple[list[str], dict[str, str]]] = []

    def fake_download(_colab_bin: str, _session_name: str, remote_path: str, local_path: Path) -> None:
        if remote_path.endswith("ckpt_step00000500.pt"):
            raise RuntimeError("transient")
        local_path.write_text("ok", encoding="utf-8")

    monkeypatch.setattr("scripts.colab_checkpoint_sync._download_file", fake_download)

    _sync_remote_files(
        "/root/.venvs/colabcli/bin/colab",
        "thai-sign-train-managed",
        "/content/checkpoints/pose_t5_v3_colab",
        mirror_dir,
        ["ckpt_step00000500.pt", "train.log"],
        download_retries=1,
        retry_delay_sec=0,
        on_progress=lambda downloaded, failures: snapshots.append((downloaded.copy(), failures.copy())),
    )

    assert snapshots[0] == (["train.log"], {})
    assert snapshots[-1] == (["train.log"], {"ckpt_step00000500.pt": "transient"})


def test_seed_mirror_from_dataset_dir_copies_existing_checkpoints(tmp_path):
    mirror_dir = tmp_path / "mirror"
    dataset_dir = tmp_path / "dataset"
    mirror_dir.mkdir()
    dataset_dir.mkdir()
    (dataset_dir / "ckpt_step00001000.pt").write_bytes(b"checkpoint")
    (dataset_dir / "train.log").write_text("ignore", encoding="utf-8")

    seeded = _seed_mirror_from_dataset_dir(
        mirror_dir,
        dataset_dir,
        ["ckpt_step00001000.pt", "train.log", "ckpt_step00001500.pt"],
    )

    assert seeded == ["ckpt_step00001000.pt"]
    assert (mirror_dir / "ckpt_step00001000.pt").read_bytes() == b"checkpoint"
    assert not (mirror_dir / "train.log").exists()


def test_copy_dataset_snapshot_copies_checkpoints_and_exports(tmp_path):
    snapshot_dir = tmp_path / "snapshot"
    dataset_dir = tmp_path / "dataset"
    snapshot_dir.mkdir()
    dataset_dir.mkdir()
    (snapshot_dir / "ckpt_step00001200.pt").write_bytes(b"checkpoint")
    (snapshot_dir / "model.safetensors").write_bytes(b"weights")
    (snapshot_dir / "notes.txt").write_text("ignore", encoding="utf-8")

    copied = _copy_dataset_snapshot(snapshot_dir, dataset_dir)

    assert copied == ["ckpt_step00001200.pt", "model.safetensors"]
    assert (dataset_dir / "ckpt_step00001200.pt").read_bytes() == b"checkpoint"
    assert (dataset_dir / "model.safetensors").read_bytes() == b"weights"
    assert not (dataset_dir / "notes.txt").exists()


def test_acquire_sync_lock_rejects_live_holder(tmp_path, monkeypatch):
    lock_path = tmp_path / ".sync.lock"
    lock_path.write_text(json.dumps({"pid": 4242}), encoding="utf-8")
    monkeypatch.setattr("scripts.colab_checkpoint_sync._pid_is_running", lambda pid: pid == 4242)

    with pytest.raises(RuntimeError, match="Another sync process is already running"):
        _acquire_sync_lock(lock_path)


def test_acquire_sync_lock_replaces_stale_holder(tmp_path, monkeypatch):
    lock_path = tmp_path / ".sync.lock"
    lock_path.write_text(json.dumps({"pid": 4242}), encoding="utf-8")
    monkeypatch.setattr("scripts.colab_checkpoint_sync._pid_is_running", lambda _pid: False)

    _acquire_sync_lock(lock_path)

    payload = json.loads(lock_path.read_text(encoding="utf-8"))
    assert payload["pid"] > 0
    _release_sync_lock(lock_path)
    assert not lock_path.exists()


def test_refresh_dataset_dir_from_kaggle_downloads_and_copies_snapshot(tmp_path, monkeypatch):
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    seen: dict[str, list[str]] = {}

    def fake_run_local(cmd: list[str], *, capture_output: bool = False):
        seen["cmd"] = cmd
        target_dir = Path(cmd[cmd.index("-p") + 1])
        (target_dir / "ckpt_step00001500.pt").write_bytes(b"checkpoint")
        (target_dir / "train.log").write_text("log", encoding="utf-8")

        class _Result:
            returncode = 0
            stdout = ""
            stderr = ""

        return _Result()

    monkeypatch.setattr("scripts.colab_checkpoint_sync._run_local", fake_run_local)

    copied = _refresh_dataset_dir_from_kaggle(dataset_dir, "orbitorls/thai-sign-ckpt")

    assert seen["cmd"][:6] == [
        sys.executable,
        "-m",
        "kaggle",
        "datasets",
        "download",
        "-d",
    ]
    assert copied == ["ckpt_step00001500.pt", "train.log"]
    assert (dataset_dir / "ckpt_step00001500.pt").read_bytes() == b"checkpoint"
    assert (dataset_dir / "train.log").read_text(encoding="utf-8") == "log"


def test_download_file_converts_local_target_to_wsl_path(tmp_path, monkeypatch):
    target = tmp_path / "train.log"
    seen: dict[str, list[str]] = {}

    monkeypatch.setattr(
        "scripts.colab_checkpoint_sync._to_wsl_path",
        lambda path: "/mnt/c/tmp/train.log",
    )

    def fake_run_colab(_colab_bin: str, args: list[str], *, capture_output: bool = False):
        seen["args"] = args

        class _Result:
            returncode = 0
            stderr = ""
            stdout = ""

        return _Result()

    monkeypatch.setattr("scripts.colab_checkpoint_sync._run_colab", fake_run_colab)

    _download_file(
        "/root/.venvs/colabcli/bin/colab",
        "thai-sign-train-managed",
        "/content/checkpoints/pose_t5_v3_colab/train.log",
        target,
    )

    assert seen["args"] == [
        "download",
        "-s",
        "thai-sign-train-managed",
        "/content/checkpoints/pose_t5_v3_colab/train.log",
        "/mnt/c/tmp/train.log",
    ]


def test_get_session_status_parses_colab_status_output(monkeypatch):
    def fake_run_colab(_colab_bin: str, args: list[str], *, capture_output: bool = False):
        assert args == ["status", "-s", "thai-sign-train-managed-r5"]

        class _Result:
            returncode = 0
            stderr = ""
            stdout = (
                "[thai-sign-train-managed-r5] gpu-a100 | Hardware: A100 | Variant: GPU | Status: RUNNING\n"
                "  Last Execution: /content/run.py at 2026-06-20 02:00:00\n"
            )

        return _Result()

    monkeypatch.setattr("scripts.colab_checkpoint_sync._run_colab", fake_run_colab)

    status = _get_session_status("/root/.venvs/colabcli/bin/colab", "thai-sign-train-managed-r5")

    assert status["hardware"] == "A100"
    assert status["status"] == "RUNNING"


def test_download_text_file_via_exec_writes_local_copy(tmp_path, monkeypatch):
    target = tmp_path / "train.log"

    monkeypatch.setattr("scripts.colab_checkpoint_sync._to_wsl_path", lambda path: "/mnt/c/tmp/fetch.py")

    def fake_run_colab(_colab_bin: str, args: list[str], *, capture_output: bool = False):
        class _Result:
            returncode = 0
            stderr = ""
            stdout = "line1\nline2\n"

        return _Result()

    monkeypatch.setattr("scripts.colab_checkpoint_sync._run_colab", fake_run_colab)

    ok = _download_text_file_via_exec(
        "/root/.venvs/colabcli/bin/colab",
        "thai-sign-train-managed-r8",
        "/content/checkpoints/pose_t5_v3_colab/train.log",
        target,
    )

    assert ok is True
    assert target.read_text(encoding="utf-8") == "line1\nline2\n"


def test_sanitize_exec_text_output_removes_trailing_digit_line():
    text = "line1\nline2\n532\n"

    assert _sanitize_exec_text_output(text) == "line1\nline2\n"


def test_sanitize_exec_text_output_removes_compact_digit_suffix():
    text = '{\n  "a": 1\n}532\n'

    assert _sanitize_exec_text_output(text) == '{\n  "a": 1\n}'


def test_download_file_falls_back_to_exec_for_text_files(tmp_path, monkeypatch):
    target = tmp_path / "train.log"
    calls: list[list[str]] = []

    def fake_run_colab(_colab_bin: str, args: list[str], *, capture_output: bool = False):
        calls.append(args)

        class _Result:
            stderr = ""

        if args[:2] == ["download", "-s"]:
            _Result.returncode = 1
            _Result.stdout = ""
            _Result.stderr = "500"
            return _Result()

        _Result.returncode = 0
        _Result.stdout = "fallback log\n13\n"
        return _Result()

    monkeypatch.setattr("scripts.colab_checkpoint_sync._run_colab", fake_run_colab)
    monkeypatch.setattr("scripts.colab_checkpoint_sync._to_wsl_path", lambda path: "/mnt/c/tmp/train.log")

    _download_file(
        "/root/.venvs/colabcli/bin/colab",
        "thai-sign-train-managed-r8",
        "/content/checkpoints/pose_t5_v3_colab/train.log",
        target,
    )

    assert target.read_text(encoding="utf-8") == "fallback log\n"
    assert any(args[0] == "exec" for args in calls)


def test_list_remote_files_falls_back_to_exec_when_ls_fails(tmp_path, monkeypatch):
    calls: list[list[str]] = []

    def fake_run_colab(_colab_bin: str, args: list[str], *, capture_output: bool = False):
        calls.append(args)

        class _Result:
            stderr = ""

        if args[:3] == ["ls", "-s", "thai-sign-train-managed-r8"]:
            _Result.returncode = 1
            _Result.stdout = ""
            _Result.stderr = "500"
            return _Result()

        _Result.returncode = 0
        _Result.stdout = "ckpt_step00001000.pt\ntrain.log\nsubdir/\n"
        return _Result()

    monkeypatch.setattr("scripts.colab_checkpoint_sync._run_colab", fake_run_colab)
    monkeypatch.setattr("scripts.colab_checkpoint_sync._to_wsl_path", lambda path: "/mnt/c/tmp/fallback.py")

    names = __import__("scripts.colab_checkpoint_sync", fromlist=["_list_remote_files"])._list_remote_files(
        "/root/.venvs/colabcli/bin/colab",
        "thai-sign-train-managed-r8",
        "/content/checkpoints/pose_t5_v3_colab",
    )

    assert names == ["ckpt_step00001000.pt", "train.log"]
    assert calls[1][:4] == ["exec", "-s", "thai-sign-train-managed-r8", "-f"]


def test_publish_latest_checkpoint_copies_final_export_files(tmp_path):
    mirror_dir = tmp_path / "mirror"
    dataset_dir = tmp_path / "dataset"
    mirror_dir.mkdir()

    (mirror_dir / "ckpt_step00001000.pt").write_bytes(b"checkpoint")
    (mirror_dir / "model.safetensors").write_bytes(b"weights")
    (mirror_dir / "pose_t5_config.json").write_text("{}", encoding="utf-8")
    (mirror_dir / "train_metrics.json").write_text("{}", encoding="utf-8")

    publish_latest_checkpoint(
        mirror_dir,
        dataset_dir,
        "orbitorls/thai-sign-ckpt",
        "Thai Sign Ckpt",
    )

    assert (dataset_dir / "model.safetensors").read_bytes() == b"weights"
    assert (dataset_dir / "pose_t5_config.json").read_text(encoding="utf-8") == "{}"
