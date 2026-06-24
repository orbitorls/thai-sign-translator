import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts._bootstrap import ensure_repo_paths

ensure_repo_paths()

from scripts.maintenance.repo_cleanup_plan import (
    DO_NOT_DELETE_DIRS,
    SAFE_CLEANUP_DIRS,
    CleanupCandidate,
    CleanupPlan,
    build_cleanup_plan,
    main,
)

__all__ = [
    "DO_NOT_DELETE_DIRS",
    "SAFE_CLEANUP_DIRS",
    "CleanupCandidate",
    "CleanupPlan",
    "build_cleanup_plan",
    "main",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
