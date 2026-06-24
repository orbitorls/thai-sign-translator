from __future__ import annotations

import json
from pathlib import Path

from scripts.repo_hygiene_audit import audit_repo_hygiene, main


def test_repo_hygiene_audit_flags_root_artifacts_and_legacy_bootstrap(tmp_path: Path, capsys):
    (tmp_path / "scripts").mkdir()
    (tmp_path / "checkpoints").mkdir()
    (tmp_path / "checkpoints_bad.pt").write_bytes(b"x")
    (tmp_path / "colab-run.ipynb").write_text("{}", encoding="utf-8")
    (tmp_path / "scripts" / "legacy.py").write_text(
        "import os\nimport sys\n_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))\n"
        "if _REPO_ROOT not in sys.path:\n    sys.path.insert(0, _REPO_ROOT)\n",
        encoding="utf-8",
    )
    (tmp_path / "scripts" / "clean.py").write_text(
        "from scripts._bootstrap import ensure_repo_paths\nensure_repo_paths()\n",
        encoding="utf-8",
    )

    audit = audit_repo_hygiene(tmp_path)

    root_paths = {item.path for item in audit.root_artifacts}
    legacy_paths = {item.path for item in audit.legacy_bootstrap_scripts}
    assert "checkpoints" not in root_paths
    assert "checkpoints_bad.pt" in root_paths
    assert "colab-run.ipynb" in root_paths
    assert "scripts/legacy.py" in legacy_paths
    assert "scripts/clean.py" not in legacy_paths

    code = main(["--root", str(tmp_path), "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["root_artifacts"]
    assert payload["legacy_bootstrap_scripts"]
