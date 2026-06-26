from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from scripts._bootstrap import ensure_repo_paths

ensure_repo_paths()

try:  # pragma: no cover - runtime portability
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except AttributeError:
    pass

from scripts.maintenance.repo_inventory import _human_bytes, collect_repo_inventory


SAFE_CLEANUP_DIRS = {
    ".pytest_cache",
    ".playwright-mcp",
    "__pycache__",
    "tmp_pytest",
    "frontend/dist",
}

DO_NOT_DELETE_DIRS = {
    "checkpoints",
    "data",
    "kaggle_upload",
}


@dataclass(frozen=True)
class CleanupCandidate:
    path: str
    kind: str
    reason: str
    bytes: int


@dataclass(frozen=True)
class CleanupPlan:
    root: str
    safe_candidates: list[CleanupCandidate]
    protected_artifacts: list[CleanupCandidate]


def _normalized_rel(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _is_under_any(rel_path: str, prefixes: set[str]) -> bool:
    return any(rel_path == prefix or rel_path.startswith(prefix + "/") for prefix in prefixes)


def build_cleanup_plan(root: Path) -> CleanupPlan:
    root = root.resolve()
    safe_candidates: list[CleanupCandidate] = []
    protected_artifacts: list[CleanupCandidate] = []

    for path in root.rglob("*"):
        if not path.exists():
            continue
        rel = _normalized_rel(path, root)
        rel_dir = rel if path.is_dir() else rel.rsplit("/", 1)[0] if "/" in rel else ""
        try:
            size = path.stat().st_size if path.is_file() else 0
        except OSError:
            continue

        if _is_under_any(rel, DO_NOT_DELETE_DIRS):
            if path.is_dir() and any(part in {"checkpoints", "data", "kaggle_upload"} for part in path.parts):
                protected_artifacts.append(
                    CleanupCandidate(
                        path=rel,
                        kind="artifact_dir",
                        reason="Large training/data artifact directory; archive manually if needed.",
                        bytes=0,
                    )
                )
            continue

        if path.is_dir() and rel in SAFE_CLEANUP_DIRS:
            safe_candidates.append(
                CleanupCandidate(
                    path=rel,
                    kind="cache_dir",
                    reason="Generated cache or build output that can be removed safely.",
                    bytes=0,
                )
            )
        elif path.is_file() and path.suffix in {".pyc", ".pyo"}:
            safe_candidates.append(
                CleanupCandidate(
                    path=rel,
                    kind="bytecode",
                    reason="Python bytecode cache.",
                    bytes=size,
                )
            )
        elif path.is_file() and path.name.endswith(".log") and not rel_dir.startswith("checkpoints"):
            safe_candidates.append(
                CleanupCandidate(
                    path=rel,
                    kind="log",
                    reason="Generated log file.",
                    bytes=size,
                )
            )

    safe_candidates = sorted({c.path: c for c in safe_candidates}.values(), key=lambda item: item.path.lower())
    protected_artifacts = sorted({c.path: c for c in protected_artifacts}.values(), key=lambda item: item.path.lower())
    return CleanupPlan(root=str(root), safe_candidates=safe_candidates, protected_artifacts=protected_artifacts)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Print a conservative cleanup plan for a repository.")
    parser.add_argument("--root", default=".", help="Repository root to scan.")
    parser.add_argument("--json", action="store_true", help="Print JSON instead of a human-readable report.")
    parser.add_argument("--inventory", action="store_true", help="Include repo inventory summary in the report.")
    return parser


def _to_jsonable(plan: CleanupPlan) -> dict:
    return {
        "root": plan.root,
        "safe_candidates": [asdict(item) for item in plan.safe_candidates],
        "protected_artifacts": [asdict(item) for item in plan.protected_artifacts],
    }


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    root = Path(args.root)
    plan = build_cleanup_plan(root)

    if args.json:
        payload = _to_jsonable(plan)
        if args.inventory:
            inventory = collect_repo_inventory(root)
            payload["inventory"] = {
                "root": inventory.root,
                "file_count": inventory.file_count,
                "total_bytes": inventory.total_bytes,
                "total_human": _human_bytes(inventory.total_bytes),
            }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    print(f"root: {plan.root}")
    if args.inventory:
        inventory = collect_repo_inventory(root)
        print(f"files: {inventory.file_count}")
        print(f"total: {_human_bytes(inventory.total_bytes)}")
    print("safe cleanup candidates:")
    if not plan.safe_candidates:
        print("  (none)")
    for item in plan.safe_candidates:
        suffix = f" ({_human_bytes(item.bytes)})" if item.bytes else ""
        print(f"  - {item.path}: {item.kind}{suffix} - {item.reason}")
    print("protected artifacts:")
    if not plan.protected_artifacts:
        print("  (none)")
    for item in plan.protected_artifacts:
        print(f"  - {item.path}: {item.kind} - {item.reason}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
