"""Config loader that accepts JSON and JSON-shaped .yaml files."""
from __future__ import annotations

import json
from pathlib import Path

from tsl.utils.paths import resolve_repo_path


def load_config(path: str) -> dict:
    target = resolve_repo_path(path)
    text = target.read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml  # type: ignore
        except ImportError as exc:
            raise ValueError(
                f"config {target} is not valid JSON; install PyYAML or keep config JSON-compatible"
            ) from exc
        payload = yaml.safe_load(text)
        if not isinstance(payload, dict):
            raise ValueError(f"config {target} must decode to an object")
        return payload


def write_resolved_config(path: str | Path, payload: dict) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


__all__ = ["load_config", "write_resolved_config"]
