"""Background scheduler for feedback-driven retraining."""
from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_RETRAIN_SCRIPT = _REPO_ROOT / "scripts" / "retrain_from_feedback.py"
_DEFAULT_FEEDBACK_CONFIG = _REPO_ROOT / "configs" / "feedback" / "default.json"
_SCHEDULER_ENABLED = {"1", "true", "yes"}


def feedback_config_path() -> Path:
    return _DEFAULT_FEEDBACK_CONFIG


def load_schedule_hours(config_path: Path | None = None) -> int:
    from tsl.utils.config import load_config

    path = config_path or feedback_config_path()
    schedule_hours = 24
    if path.is_file():
        payload = load_config(str(path))
        schedule_hours = int(payload.get("schedule_hours", schedule_hours))
    return max(schedule_hours, 1)


def load_reload_after_promote(config_path: Path | None = None) -> bool:
    from tsl.utils.config import load_config

    path = config_path or feedback_config_path()
    if not path.is_file():
        return True
    payload = load_config(str(path))
    return bool(payload.get("reload_after_promote", True))


def _parse_retrain_stdout(stdout: str) -> dict | None:
    text = stdout.strip()
    if not text:
        return None
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def run_retrain_job(*, config_path: Path | None = None) -> int:
    if not _RETRAIN_SCRIPT.is_file():
        logger.error("retrain script missing: %s", _RETRAIN_SCRIPT)
        return 1
    config = config_path or feedback_config_path()
    cmd = [sys.executable, str(_RETRAIN_SCRIPT), "--config", str(config)]
    result = subprocess.run(cmd, cwd=str(_REPO_ROOT), check=False, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)

    payload = _parse_retrain_stdout(result.stdout)
    if (
        payload
        and payload.get("status") == "completed"
        and payload.get("promoted") is True
        and load_reload_after_promote(config)
    ):
        from tsl.serving.cache import clear_translator_cache

        cleared = clear_translator_cache()
        logger.info("cleared %s cached translator(s) after feedback promote", cleared)
    return int(result.returncode)


def scheduler_enabled() -> bool:
    return os.environ.get("TSL_FEEDBACK_SCHEDULER", "").strip().lower() in _SCHEDULER_ENABLED


def _purge_expired_videos_job() -> None:
    from tsl.privacy.video_store import VideoStore

    removed = VideoStore().purge_expired()
    if removed:
        logger.info("purged %s expired research video(s)", removed)


def start_feedback_scheduler() -> object | None:
    if not scheduler_enabled():
        return None
    try:
        from apscheduler.schedulers.background import BackgroundScheduler
        from apscheduler.triggers.interval import IntervalTrigger
    except ImportError:
        logger.warning("APScheduler not installed; feedback scheduler disabled")
        return None

    config_path = feedback_config_path()
    schedule_hours = load_schedule_hours(config_path)

    scheduler = BackgroundScheduler()
    scheduler.add_job(
        run_retrain_job,
        kwargs={"config_path": config_path},
        trigger=IntervalTrigger(hours=schedule_hours),
        id="feedback_retrain",
        replace_existing=True,
        max_instances=1,
    )
    scheduler.add_job(
        _purge_expired_videos_job,
        trigger=IntervalTrigger(hours=24),
        id="video_retention_purge",
        replace_existing=True,
        max_instances=1,
    )
    scheduler.start()
    logger.info(
        "feedback retrain scheduler started (every %s hours, config=%s)",
        schedule_hours,
        config_path,
    )
    return scheduler


def stop_feedback_scheduler(scheduler: object | None) -> None:
    if scheduler is None:
        return
    shutdown = getattr(scheduler, "shutdown", None)
    if callable(shutdown):
        shutdown(wait=False)
