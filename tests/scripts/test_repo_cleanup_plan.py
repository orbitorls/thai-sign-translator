from __future__ import annotations

from pathlib import Path

from scripts.repo_cleanup_plan import build_cleanup_plan, main


def test_build_cleanup_plan_marks_safe_generated_dirs_and_protects_artifacts(tmp_path: Path):
    (tmp_path / ".pytest_cache").mkdir()
    (tmp_path / "frontend").mkdir()
    (tmp_path / "frontend" / "dist").mkdir(parents=True)
    (tmp_path / "tmp_pytest").mkdir()
    (tmp_path / "checkpoints").mkdir()
    (tmp_path / "checkpoints" / "model.pt").write_bytes(b"checkpoint")
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "manifest.csv").write_text("x", encoding="utf-8")

    plan = build_cleanup_plan(tmp_path)

    safe_paths = {item.path for item in plan.safe_candidates}
    protected_paths = {item.path for item in plan.protected_artifacts}

    assert ".pytest_cache" in safe_paths
    assert "frontend/dist" in safe_paths
    assert "tmp_pytest" in safe_paths
    assert "checkpoints" in protected_paths
    assert "data" in protected_paths


def test_repo_cleanup_plan_main_json_output(tmp_path: Path, capsys):
    (tmp_path / ".pytest_cache").mkdir()

    code = main(["--root", str(tmp_path), "--json", "--inventory"])

    captured = capsys.readouterr()
    assert code == 0
    assert '"safe_candidates"' in captured.out
    assert '"inventory"' in captured.out