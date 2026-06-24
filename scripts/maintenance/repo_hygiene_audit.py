from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


ROOT_ARTIFACT_PREFIXES = (
    "checkpoints",
    "smoke_",
    "thai-sign-",
)

ROOT_ARTIFACT_SUFFIXES = (
    ".ipynb",
    ".log",
)

ALLOWED_ROOT_ARTIFACT_DIRS = {
    "checkpoints",
    "data",
    "kaggle_upload",
    "tmp",
    "output",
}

SCRIPT_BOOTSTRAP_SNIPPETS = (
    "sys.path.insert(0, _REPO_ROOT)",
    "sys.path.insert(0, _SRC_ROOT)",
    "if _REPO_ROOT not in sys.path",
    "if _SRC_ROOT not in sys.path",
    "os.path.dirname(os.path.dirname(os.path.abspath(__file__)))",
)


@dataclass(frozen=True)
class AuditItem:
    path: str
    kind: str
    reason: str


@dataclass(frozen=True)
class RepoHygieneAudit:
    root: str
    root_artifacts: list[AuditItem]
    legacy_bootstrap_scripts: list[AuditItem]


def _looks_like_root_artifact(path: Path) -> bool:
    name = path.name
    if path.is_dir() and name in ALLOWED_ROOT_ARTIFACT_DIRS:
        return False
    if name.startswith(ROOT_ARTIFACT_PREFIXES):
        return True
    if path.is_file() and name.endswith(ROOT_ARTIFACT_SUFFIXES):
        return True
    return False


def _iter_python_scripts(root: Path) -> Iterable[Path]:
    scripts_dir = root / "scripts"
    if not scripts_dir.is_dir():
        return []
    return scripts_dir.rglob("*.py")


def _uses_legacy_bootstrap(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    if "from scripts._bootstrap import ensure_repo_paths" in text:
        return False
    return any(snippet in text for snippet in SCRIPT_BOOTSTRAP_SNIPPETS)


def audit_repo_hygiene(root: Path) -> RepoHygieneAudit:
    root = root.resolve()
    root_artifacts: list[AuditItem] = []
    for path in root.iterdir():
        if path.name.startswith("."):
            continue
        if not _looks_like_root_artifact(path):
            continue
        kind = "dir" if path.is_dir() else "file"
        root_artifacts.append(
            AuditItem(
                path=path.name,
                kind=f"root_artifact_{kind}",
                reason="Generated artifact is living at repo root instead of an artifact directory.",
            )
        )

    legacy_bootstrap_scripts: list[AuditItem] = []
    for path in _iter_python_scripts(root):
        if path.name in {"_bootstrap.py", "__init__.py"}:
            continue
        if not _uses_legacy_bootstrap(path):
            continue
        legacy_bootstrap_scripts.append(
            AuditItem(
                path=path.relative_to(root).as_posix(),
                kind="legacy_bootstrap",
                reason="Script still inlines repo/src sys.path bootstrap logic.",
            )
        )

    return RepoHygieneAudit(
        root=str(root),
        root_artifacts=sorted(root_artifacts, key=lambda item: item.path.lower()),
        legacy_bootstrap_scripts=sorted(legacy_bootstrap_scripts, key=lambda item: item.path.lower()),
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit repo-root artifact clutter and repeated script bootstrap code.")
    parser.add_argument("--root", default=".", help="Repository root to scan.")
    parser.add_argument("--json", action="store_true", help="Print JSON instead of a human-readable report.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    audit = audit_repo_hygiene(Path(args.root))
    if args.json:
        print(json.dumps(asdict(audit), ensure_ascii=False, indent=2))
        return 0

    print(f"root: {audit.root}")
    print("root artifacts:")
    if not audit.root_artifacts:
        print("  (none)")
    for item in audit.root_artifacts:
        print(f"  - {item.path}: {item.kind} - {item.reason}")
    print("legacy bootstrap scripts:")
    if not audit.legacy_bootstrap_scripts:
        print("  (none)")
    for item in audit.legacy_bootstrap_scripts:
        print(f"  - {item.path}: {item.kind} - {item.reason}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
