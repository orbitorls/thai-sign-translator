from __future__ import annotations

import zipfile

from scripts.package_colab_bundle import build_bundle


def test_build_bundle_uses_posix_paths_and_skips_pycache(tmp_path):
    repo_root = tmp_path / "repo"
    (repo_root / "src" / "pkg").mkdir(parents=True)
    (repo_root / "scripts").mkdir()
    (repo_root / "scripts" / "__pycache__").mkdir()

    (repo_root / "src" / "pkg" / "mod.py").write_text("print('ok')\n", encoding="utf-8")
    (repo_root / "scripts" / "run.py").write_text("print('run')\n", encoding="utf-8")
    (repo_root / "scripts" / "__pycache__" / "run.pyc").write_bytes(b"pyc")
    (repo_root / "requirements.txt").write_text("torch\n", encoding="utf-8")

    output_path = tmp_path / "bundle.zip"
    build_bundle(repo_root, output_path)

    with zipfile.ZipFile(output_path, "r") as archive:
        names = archive.namelist()

    assert "src/pkg/mod.py" in names
    assert "scripts/run.py" in names
    assert all("\\" not in name for name in names)
    assert all("__pycache__" not in name for name in names)
