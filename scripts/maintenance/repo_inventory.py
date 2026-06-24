from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

try:  # pragma: no cover - runtime portability
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except AttributeError:
    pass


DEFAULT_EXCLUDED_DIRS = {
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".playwright-mcp",
    "node_modules",
    ".next",
    "dist",
    "build",
    ".venv",
    "venv",
}


@dataclass(frozen=True)
class TopLevelSummary:
    top_level: str
    files: int
    bytes: int


@dataclass(frozen=True)
class FileSummary:
    path: str
    bytes: int


@dataclass(frozen=True)
class RepoInventory:
    root: str
    file_count: int
    total_bytes: int
    top_level: list[TopLevelSummary]
    largest_files: list[FileSummary]


def _should_skip_path(path: Path) -> bool:
    return any(part in DEFAULT_EXCLUDED_DIRS for part in path.parts)


def iter_repo_files(root: Path) -> Iterable[Path]:
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if _should_skip_path(path):
            continue
        yield path


def collect_repo_inventory(root: Path, *, largest_limit: int = 50) -> RepoInventory:
    root = root.resolve()
    top_level: dict[str, list[int]] = {}
    largest_files: list[FileSummary] = []
    file_count = 0
    total_bytes = 0

    for file_path in iter_repo_files(root):
        try:
            size = file_path.stat().st_size
        except OSError:
            continue
        rel_parts = file_path.relative_to(root).parts
        top = rel_parts[0] if len(rel_parts) > 1 else "."
        bucket = top_level.setdefault(top, [0, 0])
        bucket[0] += 1
        bucket[1] += size
        file_count += 1
        total_bytes += size
        largest_files.append(FileSummary(path=str(file_path), bytes=size))

    top_level_summary = [
        TopLevelSummary(top_level=name, files=stats[0], bytes=stats[1])
        for name, stats in sorted(top_level.items(), key=lambda item: (-item[1][1], item[0].lower()))
    ]
    largest_files = sorted(largest_files, key=lambda item: (-item.bytes, item.path.lower()))[:largest_limit]
    return RepoInventory(
        root=str(root),
        file_count=file_count,
        total_bytes=total_bytes,
        top_level=top_level_summary,
        largest_files=largest_files,
    )


def _human_bytes(num: int) -> str:
    value = float(num)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if value < 1024.0 or unit == "TB":
            if unit == "B":
                return f"{int(value)} {unit}"
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{value:.2f} TB"


def _to_jsonable(inventory: RepoInventory) -> dict:
    data = asdict(inventory)
    data["total_human"] = _human_bytes(inventory.total_bytes)
    for row in data["top_level"]:
        row["human"] = _human_bytes(row["bytes"])
    for row in data["largest_files"]:
        row["human"] = _human_bytes(row["bytes"])
    return data


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize file counts and sizes for a repository.")
    parser.add_argument("--root", default=".", help="Repository root to scan.")
    parser.add_argument("--largest", type=int, default=50, help="How many of the largest files to list.")
    parser.add_argument("--json", action="store_true", help="Print JSON instead of a human-readable report.")
    parser.add_argument("--output", default="", help="Optional output file path for JSON report.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    inventory = collect_repo_inventory(Path(args.root), largest_limit=max(1, int(args.largest)))
    if args.json:
        payload = json.dumps(_to_jsonable(inventory), ensure_ascii=False, indent=2)
        print(payload)
        if str(args.output).strip():
            target = Path(args.output).resolve()
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(payload + "\n", encoding="utf-8")
    else:
        print(f"root: {inventory.root}")
        print(f"files: {inventory.file_count}")
        print(f"total: {_human_bytes(inventory.total_bytes)}")
        print("top-level:")
        for row in inventory.top_level:
            print(f"  - {row.top_level}: {row.files} files, {_human_bytes(row.bytes)}")
        print("largest files:")
        for row in inventory.largest_files:
            print(f"  - {_human_bytes(row.bytes)}  {row.path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
