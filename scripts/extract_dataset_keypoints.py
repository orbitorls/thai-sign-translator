import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts._bootstrap import ensure_repo_paths

ensure_repo_paths()

from scripts.data.extract_dataset_keypoints import (
    extract_segments,
    load_segments,
    main,
    _make_manifest_row,
)

__all__ = [
    "extract_segments",
    "load_segments",
    "main",
    "_make_manifest_row",
]


if __name__ == "__main__":
    raise SystemExit(main())
