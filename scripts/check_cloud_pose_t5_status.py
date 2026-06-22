from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _load_json(path: str) -> dict[str, Any] | None:
    if not str(path).strip():
        return None
    target = Path(path).resolve()
    if not target.is_file():
        return None
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _parse_time(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _recent_artifact_snapshot(artifact_dir: str) -> dict[str, Any]:
    target = Path(artifact_dir).resolve()
    if not target.is_dir():
        return {"exists": False, "path": str(target)}

    interesting = [
        target / "train_metrics.json",
        target / "verified_eval.json",
        target / "best_model_state.pt",
        target / "runtime_metadata.json",
    ]
    checkpoints = sorted(target.glob("ckpt_step*.pt"))
    latest_file = None
    latest_mtime = None
    for path in [*interesting, *checkpoints]:
        if not path.is_file():
            continue
        mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
        if latest_mtime is None or mtime > latest_mtime:
            latest_file = path
            latest_mtime = mtime
    return {
        "exists": True,
        "path": str(target),
        "checkpoint_count": len(checkpoints),
        "latest_checkpoint": checkpoints[-1].name if checkpoints else None,
        "latest_file": latest_file.name if latest_file else None,
        "latest_file_mtime_utc": latest_mtime.isoformat() if latest_mtime else None,
        "verified_eval_present": (target / "verified_eval.json").is_file(),
        "runtime_metadata_present": (target / "runtime_metadata.json").is_file(),
    }


def classify_cloud_pose_t5_status(
    *,
    launcher_status_json: str = "",
    orchestrator_status_json: str = "",
    artifact_dir: str = "",
    stale_minutes: int = 30,
) -> dict[str, Any]:
    launcher = _load_json(launcher_status_json)
    orchestrator = _load_json(orchestrator_status_json)
    artifacts = _recent_artifact_snapshot(artifact_dir) if str(artifact_dir).strip() else None

    classification = "unknown"
    reason = "no usable status inputs"
    recommendations: list[str] = []

    if artifacts and artifacts.get("verified_eval_present") and artifacts.get("runtime_metadata_present"):
        classification = "complete"
        reason = "cloud artifact contains verified_eval.json and runtime_metadata.json"
    elif isinstance(launcher, dict):
        phase = str(launcher.get("phase", "")).strip().lower()
        message = str(launcher.get("message", "")).strip()
        last_errors = json.dumps(launcher.get("last_errors", {}), ensure_ascii=False)
        if phase == "error" and (
            "unable to allocate requested gpu" in message.lower()
            or "service unavailable" in last_errors.lower()
            or "toomanyassignmentserror" in last_errors.lower()
        ):
            classification = "gpu_wait"
            reason = message or last_errors or "GPU backend unavailable"
            recommendations.append("Retry on the next allowed GPU tier instead of waiting indefinitely.")
        elif phase in {"running", "launching"}:
            classification = "running"
            reason = f"launcher phase={phase}"
    if classification == "unknown" and isinstance(orchestrator, dict):
        source_status = str(orchestrator.get("source_status", "")).strip().upper()
        if source_status == "RUNNING" and bool(orchestrator.get("source_visible_stale_suspected")):
            classification = "cloud_api_stale"
            reason = "Kaggle status is running but the visible progress snapshot has been stale past the threshold"
            recommendations.append("Refresh kernel status/logs before assuming the process is dead.")
        elif source_status == "ERROR":
            classification = "process_died"
            reason = str(orchestrator.get("source_failure_message") or "cloud kernel reported ERROR")
        elif source_status == "RUNNING":
            classification = "running"
            reason = "cloud kernel is still running"
    if classification in {"unknown", "running"} and isinstance(artifacts, dict) and artifacts.get("exists"):
        latest_mtime = _parse_time(artifacts.get("latest_file_mtime_utc"))
        if latest_mtime is not None:
            age_seconds = max(0.0, (datetime.now(timezone.utc) - latest_mtime).total_seconds())
            artifacts["latest_file_age_seconds"] = round(age_seconds, 3)
            if classification == "running" and age_seconds >= max(1, int(stale_minutes)) * 60:
                classification = "sync_stale"
                reason = (
                    f"artifact mirror has not changed for {round(age_seconds / 60.0, 1)} minutes while the cloud lane still "
                    "looks active"
                )
                recommendations.append("Check whether cloud output syncing stopped before relaunching training.")
    if classification == "unknown":
        recommendations.append("Inspect the launcher status, kernel status, and latest artifact mtimes together.")

    return {
        "classification": classification,
        "reason": reason,
        "launcher": launcher,
        "orchestrator": orchestrator,
        "artifacts": artifacts,
        "stale_minutes": int(stale_minutes),
        "recommendations": recommendations,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Classify the current cloud PoseT5 lane status from launcher/kernel/artifact snapshots.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--launcher-status-json", default="")
    parser.add_argument("--orchestrator-status-json", default="")
    parser.add_argument("--artifact-dir", default="")
    parser.add_argument("--stale-minutes", type=int, default=30)
    parser.add_argument("--report-json", default="")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    report = classify_cloud_pose_t5_status(
        launcher_status_json=args.launcher_status_json,
        orchestrator_status_json=args.orchestrator_status_json,
        artifact_dir=args.artifact_dir,
        stale_minutes=int(args.stale_minutes),
    )
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    if str(args.report_json).strip():
        target = Path(args.report_json).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
