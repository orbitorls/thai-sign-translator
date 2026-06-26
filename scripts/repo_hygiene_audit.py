import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts._bootstrap import ensure_repo_paths

ensure_repo_paths()

from scripts.maintenance.repo_hygiene_audit import RepoHygieneAudit, audit_repo_hygiene, main

__all__ = [
    "RepoHygieneAudit",
    "audit_repo_hygiene",
    "main",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
