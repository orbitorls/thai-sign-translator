from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"


def ensure_repo_paths() -> None:
    for path in (REPO_ROOT, SRC_ROOT):
        as_text = str(path)
        if as_text not in sys.path:
            sys.path.insert(0, as_text)
