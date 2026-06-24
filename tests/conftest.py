from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_BASE = _ROOT / "tmp_pytest" / "tmp_path"


@pytest.fixture
def tmp_path() -> Path:
    path = _BASE / f"case_{uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path
