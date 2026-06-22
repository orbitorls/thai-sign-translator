from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect local Kaggle dataset publish processes and visibility.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset-ref", required=True, help="owner/slug")
    parser.add_argument("--dataset-dir", default="", help="Optional local dataset staging dir")
    parser.add_argument("--temp-dir", default="", help="Optional TEMP/TMP override for kaggle CLI")
    parser.add_argument("--status-json", default="", help="Optional JSON status output path")
    return parser


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _kaggle_env(temp_dir: str = "") -> dict[str, str]:
    env = os.environ.copy()
    if temp_dir:
        temp_path = Path(temp_dir).resolve()
        temp_path.mkdir(parents=True, exist_ok=True)
        env["TEMP"] = str(temp_path)
        env["TMP"] = str(temp_path)
    return env


def _run(cmd: list[str], *, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        check=False,
        text=True,
        capture_output=True,
        env=env,
    )


def _dataset_visible(dataset_ref: str, *, env: dict[str, str]) -> dict[str, Any]:
    slug = dataset_ref.split("/", 1)[-1]
    result = _run([sys.executable, "-m", "kaggle", "datasets", "list", "-s", slug], env=env)
    combined = "\n".join(part for part in [result.stdout, result.stderr] if part).strip()
    return {
        "visible": dataset_ref.lower() in combined.lower(),
        "returncode": result.returncode,
        "stdout_tail": result.stdout[-2000:],
        "stderr_tail": result.stderr[-2000:],
    }


def _process_snapshot(dataset_ref: str) -> list[dict[str, Any]]:
    slug = dataset_ref.split("/", 1)[-1].lower()
    ps_cmd = [
        "powershell",
        "-NoProfile",
        "-Command",
        (
            "Get-CimInstance Win32_Process | "
            "Where-Object { $_.Name -like 'python*' } | "
            "Select-Object ProcessId,Name,CommandLine | ConvertTo-Json -Depth 3"
        ),
    ]
    result = subprocess.run(ps_cmd, check=False, text=True, capture_output=True)
    if result.returncode != 0 or not result.stdout.strip():
        return []
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError:
        return []
    items = payload if isinstance(payload, list) else [payload]
    matches: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        cmdline = str(item.get("CommandLine") or "")
        lowered = cmdline.lower()
        if slug in lowered or "publish_kaggle_dataset.py" in lowered:
            matches.append(
                {
                    "process_id": item.get("ProcessId"),
                    "name": item.get("Name"),
                    "command_line": cmdline,
                }
            )
    return matches


def _dataset_dir_snapshot(dataset_dir: str) -> dict[str, Any] | None:
    if not dataset_dir:
        return None
    root = Path(dataset_dir).resolve()
    if not root.exists():
        return {"exists": False, "path": str(root)}
    files = []
    for path in sorted(root.iterdir(), key=lambda item: item.name):
        try:
            stat = path.stat()
        except OSError:
            continue
        files.append(
            {
                "name": path.name,
                "is_dir": path.is_dir(),
                "size": stat.st_size,
                "mtime_utc": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
            }
        )
    return {"exists": True, "path": str(root), "files": files}


def _write_status_json(path: str, payload: dict[str, Any]) -> None:
    if not path:
        return
    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    env = _kaggle_env(args.temp_dir)
    payload = {
        "observed_at_utc": _now_utc(),
        "dataset_ref": args.dataset_ref,
        "dataset_visible": _dataset_visible(args.dataset_ref, env=env),
        "publish_processes": _process_snapshot(args.dataset_ref),
        "dataset_dir": _dataset_dir_snapshot(args.dataset_dir),
    }
    _write_status_json(args.status_json, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
