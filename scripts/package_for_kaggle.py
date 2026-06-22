"""Package code and data into zip files for Kaggle upload.

Creates two archives:
  kaggle_upload/thai-sign-code.zip   — source code + config + scripts
  kaggle_upload/thai-sign-data.zip   — tsl51_v3 + youtube_sl25_thai_v3 manifests + landmarks

Usage:
    python scripts/package_for_kaggle.py
    python scripts/package_for_kaggle.py --data-only     # skip code zip
    python scripts/package_for_kaggle.py --code-only     # skip data zip
"""
from __future__ import annotations

import argparse
import os
import sys
import zipfile
from pathlib import Path


def _zip_code(root: Path, out: Path) -> None:
    includes = [
        "src",
        "config.py",
        "scripts/train_local_gpu.py",
        "scripts/kaggle_train.py",
    ]
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for inc in includes:
            p = root / inc
            if not p.exists():
                print(f"  [skip] {inc} (not found)")
                continue
            if p.is_file():
                zf.write(p, inc)
                print(f"  + {inc}")
            else:
                for f in sorted(p.rglob("*")):
                    if f.is_file() and "__pycache__" not in str(f):
                        arc = str(f.relative_to(root))
                        zf.write(f, arc)
        # Write a minimal requirements snippet
        zf.writestr(
            "kaggle_requirements.txt",
            "transformers>=4.40\nsentencepiece>=0.2.0\nsacrebleu>=2.4.0\n",
        )
    print(f"Code zip: {out}  ({out.stat().st_size / 1024**2:.1f} MB)")


def _zip_data(root: Path, out: Path, data_dirs: list[str]) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for data_dir in data_dirs:
            dp = root / data_dir
            if not dp.exists():
                print(f"  [skip] {data_dir} (not found)")
                continue
            for f in sorted(dp.rglob("*")):
                if f.is_file():
                    arc = str(f.relative_to(root / "data"))
                    zf.write(f, arc)
                    total += 1
    print(f"Data zip: {out}  ({out.stat().st_size / 1024**2:.1f} MB, {total} files)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--code-only", action="store_true")
    parser.add_argument("--data-only", action="store_true")
    parser.add_argument(
        "--data-dirs",
        default="data/tsl51_v3,data/youtube_sl25_thai_v3",
        help="Comma-separated data directories to include",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    out_dir = root / "kaggle_upload"

    if not args.data_only:
        print("\n=== Packaging code ===")
        _zip_code(root, out_dir / "thai-sign-code.zip")

    if not args.code_only:
        print("\n=== Packaging data ===")
        data_dirs = [d.strip() for d in args.data_dirs.split(",") if d.strip()]
        _zip_data(root, out_dir / "thai-sign-data.zip", data_dirs)

    print(f"\nDone. Upload files from: {out_dir}")
    print("\nKaggle upload steps:")
    print("  1. kaggle datasets create -p kaggle_upload/  (first time)")
    print("  OR drag-and-drop on kaggle.com/datasets/new")


if __name__ == "__main__":
    main()
