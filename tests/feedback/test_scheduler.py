"""Tests for feedback retrain background scheduler."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from tsl.feedback import scheduler as feedback_scheduler


def test_scheduler_disabled_without_env(monkeypatch):
    monkeypatch.delenv("TSL_FEEDBACK_SCHEDULER", raising=False)
    assert feedback_scheduler.start_feedback_scheduler() is None


@pytest.mark.parametrize("value", ["1", "true", "yes", "TRUE", "Yes"])
def test_scheduler_enabled_env_values(value: str, monkeypatch):
    monkeypatch.setenv("TSL_FEEDBACK_SCHEDULER", value)
    assert feedback_scheduler.scheduler_enabled() is True


def test_load_schedule_hours_reads_config(tmp_path: Path):
    config_path = tmp_path / "feedback.json"
    config_path.write_text(json.dumps({"schedule_hours": 12}), encoding="utf-8")
    assert feedback_scheduler.load_schedule_hours(config_path) == 12


def test_load_schedule_hours_minimum_is_one(tmp_path: Path):
    config_path = tmp_path / "feedback.json"
    config_path.write_text(json.dumps({"schedule_hours": 0}), encoding="utf-8")
    assert feedback_scheduler.load_schedule_hours(config_path) == 1


def test_run_retrain_job_invokes_script(monkeypatch, tmp_path: Path):
    script = tmp_path / "retrain_from_feedback.py"
    script.write_text("import sys\nsys.exit(0)\n", encoding="utf-8")
    config_path = tmp_path / "feedback.json"
    config_path.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(feedback_scheduler, "_RETRAIN_SCRIPT", script)
    monkeypatch.setattr(feedback_scheduler, "_REPO_ROOT", tmp_path)

    calls: list[list[str]] = []

    def _fake_run(cmd, cwd, check, capture_output=False, text=False):  # noqa: ANN001
        calls.append(cmd)
        return MagicMock(returncode=0, stdout="{}", stderr="")

    monkeypatch.setattr(feedback_scheduler.subprocess, "run", _fake_run)

    code = feedback_scheduler.run_retrain_job(config_path=config_path)
    assert code == 0
    assert calls == [
        [
            feedback_scheduler.sys.executable,
            str(script),
            "--config",
            str(config_path),
        ]
    ]


def test_start_feedback_scheduler_registers_interval_job(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("TSL_FEEDBACK_SCHEDULER", "1")

    config_path = tmp_path / "configs" / "feedback" / "default.json"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(json.dumps({"schedule_hours": 6}), encoding="utf-8")
    monkeypatch.setattr(feedback_scheduler, "_DEFAULT_FEEDBACK_CONFIG", config_path)

    result = feedback_scheduler.start_feedback_scheduler()
    assert result is not None
    jobs = result.get_jobs()
    assert len(jobs) == 1
    job = jobs[0]
    assert job.id == "feedback_retrain"
    assert job.kwargs["config_path"] == config_path
    assert job.trigger.interval.total_seconds() == 6 * 3600

    feedback_scheduler.stop_feedback_scheduler(result)


def test_stop_feedback_scheduler_calls_shutdown():
    scheduler = MagicMock()
    feedback_scheduler.stop_feedback_scheduler(scheduler)
    scheduler.shutdown.assert_called_once_with(wait=False)


def test_stop_feedback_scheduler_noop_for_none():
    feedback_scheduler.stop_feedback_scheduler(None)


def test_run_retrain_job_clears_cache_after_promote(monkeypatch, tmp_path: Path):
    script = tmp_path / "retrain_from_feedback.py"
    script.write_text("print('{}')\n", encoding="utf-8")
    config_path = tmp_path / "feedback.json"
    config_path.write_text(json.dumps({"reload_after_promote": True}), encoding="utf-8")

    monkeypatch.setattr(feedback_scheduler, "_RETRAIN_SCRIPT", script)
    monkeypatch.setattr(feedback_scheduler, "_REPO_ROOT", tmp_path)

    from tsl.serving.cache import clear_translator_cache, get_translator_cache

    get_translator_cache()["demo"] = object()

    payload = {
        "status": "completed",
        "promoted": True,
    }

    def _fake_run(cmd, cwd, check, capture_output=False, text=False):  # noqa: ANN001
        return MagicMock(returncode=0, stdout=json.dumps(payload), stderr="")

    monkeypatch.setattr(feedback_scheduler.subprocess, "run", _fake_run)

    code = feedback_scheduler.run_retrain_job(config_path=config_path)
    assert code == 0
    assert get_translator_cache() == {}
    clear_translator_cache()
