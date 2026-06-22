from __future__ import annotations

import json
from pathlib import Path

import scripts.check_kaggle_dataset_publish_status as status_module


def test_dataset_dir_snapshot_lists_files(tmp_path):
    root = tmp_path / "dataset_dir"
    root.mkdir()
    (root / "manifest.csv").write_text("segment_id\n", encoding="utf-8")
    (root / "features.zip").write_bytes(b"zip")

    snapshot = status_module._dataset_dir_snapshot(str(root))

    assert snapshot is not None
    assert snapshot["exists"] is True
    names = [item["name"] for item in snapshot["files"]]
    assert names == ["features.zip", "manifest.csv"]


def test_main_writes_status_json(tmp_path, monkeypatch, capsys):
    dataset_dir = tmp_path / "dataset_dir"
    dataset_dir.mkdir()
    (dataset_dir / "manifest.csv").write_text("segment_id\n", encoding="utf-8")

    monkeypatch.setattr(
        status_module,
        "_dataset_visible",
        lambda dataset_ref, env: {
            "visible": False,
            "returncode": 0,
            "stdout_tail": "No datasets found",
            "stderr_tail": "",
        },
    )
    monkeypatch.setattr(
        status_module,
        "_process_snapshot",
        lambda dataset_ref: [{"process_id": 1234, "name": "python.exe", "command_line": "publish"}],
    )

    status_json = tmp_path / "status.json"
    rc = status_module.main(
        [
            "--dataset-ref",
            "orbitorls/thai-sign-mixed-all-v6-archived",
            "--dataset-dir",
            str(dataset_dir),
            "--status-json",
            str(status_json),
        ]
    )

    assert rc == 0
    payload = json.loads(status_json.read_text(encoding="utf-8"))
    assert payload["dataset_ref"] == "orbitorls/thai-sign-mixed-all-v6-archived"
    assert payload["publish_processes"][0]["process_id"] == 1234
    assert payload["dataset_dir"]["exists"] is True
    output = capsys.readouterr().out
    assert '"dataset_visible"' in output
