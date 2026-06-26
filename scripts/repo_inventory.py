import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts._bootstrap import ensure_repo_paths

ensure_repo_paths()

from scripts.maintenance.repo_inventory import (
    DEFAULT_EXCLUDED_DIRS,
    FileSummary,
    RepoInventory,
    TopLevelSummary,
    _human_bytes,
    collect_repo_inventory,
    iter_repo_files,
    main,
)

__all__ = [
    "DEFAULT_EXCLUDED_DIRS",
    "FileSummary",
    "RepoInventory",
    "TopLevelSummary",
    "_human_bytes",
    "collect_repo_inventory",
    "iter_repo_files",
    "main",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
