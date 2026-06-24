from __future__ import annotations

import json
from pathlib import Path

from scripts.repo_inventory import collect_repo_inventory, main


def test_collect_repo_inventory_groups_by_top_level(tmp_path: Path):
    (tmp_path / "alpha").mkdir()
    (tmp_path / "beta").mkdir()
    (tmp_path / "alpha" / "a.txt").write_text("abcd", encoding="utf-8")
    (tmp_path / "beta" / "b.txt").write_text("xyz", encoding="utf-8")
    (tmp_path / "root.txt").write_text("root", encoding="utf-8")
    (tmp_path / ".pytest_cache").mkdir()
    (tmp_path / ".pytest_cache" / "ignored.txt").write_text("ignore", encoding="utf-8")

    inventory = collect_repo_inventory(tmp_path)

    assert inventory.file_count == 3
    assert inventory.total_bytes == 11
    assert {row.top_level for row in inventory.top_level} == {"alpha", "beta", "."}
    assert inventory.top_level[0].files == 1
    assert inventory.top_level[0].bytes == 4


def test_repo_inventory_main_json_output(tmp_path: Path, capsys):
    (tmp_path / "only.txt").write_text("hello", encoding="utf-8")

    code = main(["--root", str(tmp_path), "--json"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert code == 0
    assert payload["file_count"] == 1
    assert payload["total_bytes"] == 5
    assert payload["largest_files"][0]["path"].endswith("only.txt")