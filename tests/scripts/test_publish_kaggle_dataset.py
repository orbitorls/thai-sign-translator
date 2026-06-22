from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

from scripts.publish_kaggle_dataset import create_or_version_dataset


def _write_dataset_metadata(dataset_dir: Path, dataset_id: str = "orbitorls/demo") -> None:
    dataset_dir.mkdir(parents=True, exist_ok=True)
    (dataset_dir / "dataset-metadata.json").write_text(
        json.dumps(
            {
                "title": "Demo",
                "id": dataset_id,
                "licenses": [{"name": "CC0-1.0"}],
            }
        ),
        encoding="utf-8",
    )


def test_create_or_version_dataset_prefers_create(tmp_path, monkeypatch):
    dataset_dir = tmp_path / "dataset"
    _write_dataset_metadata(dataset_dir)
    seen: list[list[str]] = []

    def fake_run(cmd: list[str], *, env: dict[str, str]):
        seen.append(cmd)

        class _Result:
            returncode = 0
            stdout = "created"
            stderr = ""

        return _Result()

    monkeypatch.setattr("scripts.publish_kaggle_dataset._run", fake_run)

    result = create_or_version_dataset(dataset_dir, message="demo")

    assert result["action"] == "create"
    assert seen == [[
        sys.executable,
        "-m",
        "kaggle",
        "datasets",
        "create",
        "-p",
        str(dataset_dir.resolve()),
        "--dir-mode",
        "skip",
    ]]


def test_create_or_version_dataset_falls_back_to_version(tmp_path, monkeypatch):
    dataset_dir = tmp_path / "dataset"
    _write_dataset_metadata(dataset_dir)
    seen: list[list[str]] = []

    def fake_run(cmd: list[str], *, env: dict[str, str]):
        seen.append(cmd)

        class _Result:
            stdout = ""
            stderr = ""

        if cmd[4] == "create":
            _Result.returncode = 1
            _Result.stderr = "exists"
            return _Result()

        _Result.returncode = 0
        _Result.stdout = "versioned"
        return _Result()

    monkeypatch.setattr("scripts.publish_kaggle_dataset._run", fake_run)

    result = create_or_version_dataset(dataset_dir, message="demo")

    assert result["action"] == "version"
    assert seen[1][-2:] == ["-m", "demo"]


def test_create_or_version_dataset_falls_back_when_create_prints_error(tmp_path, monkeypatch):
    dataset_dir = tmp_path / "dataset"
    _write_dataset_metadata(dataset_dir)
    seen: list[list[str]] = []

    def fake_run(cmd: list[str], *, env: dict[str, str]):
        seen.append(cmd)

        class _Result:
            stderr = ""

        if cmd[4] == "create":
            _Result.returncode = 0
            _Result.stdout = "Dataset creation error: title already in use"
            return _Result()

        _Result.returncode = 0
        _Result.stdout = "versioned"
        return _Result()

    monkeypatch.setattr("scripts.publish_kaggle_dataset._run", fake_run)

    result = create_or_version_dataset(dataset_dir, message="demo")

    assert result["action"] == "version"
    assert len(seen) == 2


def test_create_or_version_dataset_raises_when_both_fail(tmp_path, monkeypatch):
    dataset_dir = tmp_path / "dataset"
    _write_dataset_metadata(dataset_dir)

    def fake_run(cmd: list[str], *, env: dict[str, str]):
        class _Result:
            returncode = 1
            stdout = ""
            stderr = "denied"

        return _Result()

    monkeypatch.setattr("scripts.publish_kaggle_dataset._run", fake_run)

    with pytest.raises(RuntimeError, match="denied"):
        create_or_version_dataset(dataset_dir, message="demo")
