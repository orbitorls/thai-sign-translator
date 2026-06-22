from __future__ import annotations

import argparse
import zipfile
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Package the repo subset needed for Colab training.")
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--output", required=True)
    return parser


def build_bundle(repo_root: Path, output_path: Path) -> None:
    includes = ["src", "scripts", "requirements.txt", "config.py", "README.md"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for include in includes:
            source = repo_root / include
            if not source.exists():
                continue
            if source.is_file():
                archive.write(source, arcname=source.relative_to(repo_root).as_posix())
                continue
            for path in sorted(source.rglob("*")):
                if not path.is_file():
                    continue
                if "__pycache__" in path.parts:
                    continue
                archive.write(path, arcname=path.relative_to(repo_root).as_posix())


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    build_bundle(Path(args.repo_root), Path(args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
