"""Download ThaiSignVis from Kaggle using the Kaggle API.

Prerequisites
-------------
1. Install Kaggle CLI:
       pip install kaggle

2. Get your API token from https://www.kaggle.com/settings → "API" → "Create New Token"
   It downloads kaggle.json — place it at:
       Windows: C:/Users/<you>/.kaggle/kaggle.json
       Linux/Mac: ~/.kaggle/kaggle.json

3. Run this script:
       python scripts/download_thaisignvis.py --out-dir data/thaisignvis_raw

After download, extract landmarks:
       python scripts/extract_thaisignvis_landmarks.py \\
           --data-root data/thaisignvis_raw \\
           --out-dir   data/thaisignvis \\
           --limit     200              # start small, remove for full dataset

Then train:
       PYTHONPATH=src python -m tsl.train.train_slt \\
           --stage thaisignvis \\
           --data-root data/thaisignvis \\
           --epochs 20 --batch-size 8 --model-size base \\
           --out-dir checkpoints/slt_thaisignvis
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

DATASET_SLUG = "thanawuttimpitak/thaisignvis"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download ThaiSignVis from Kaggle.")
    parser.add_argument(
        "--out-dir", default="data/thaisignvis_raw",
        help="Directory to download and unzip into."
    )
    parser.add_argument(
        "--no-unzip", action="store_true",
        help="Skip automatic unzip (download zip only)."
    )
    args = parser.parse_args(argv)

    os.makedirs(args.out_dir, exist_ok=True)

    # Check kaggle CLI
    try:
        subprocess.run(["kaggle", "--version"], capture_output=True, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        print("ERROR: kaggle CLI not found.  Install with:  pip install kaggle", file=sys.stderr)
        print("Then place your kaggle.json token in ~/.kaggle/kaggle.json", file=sys.stderr)
        return 1

    # Check credentials
    kaggle_json = os.path.expanduser("~/.kaggle/kaggle.json")
    if not os.path.isfile(kaggle_json):
        print(f"ERROR: Kaggle API token not found at {kaggle_json}", file=sys.stderr)
        print("Go to https://www.kaggle.com/settings → API → Create New Token", file=sys.stderr)
        return 1

    cmd = ["kaggle", "datasets", "download", DATASET_SLUG, "-p", args.out_dir]
    if not args.no_unzip:
        cmd.append("--unzip")

    print(f"Downloading {DATASET_SLUG} → {args.out_dir} …")
    print("(~165 GB — this will take a while)")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print("Download failed.", file=sys.stderr)
        return result.returncode

    print(f"\nDownload complete: {args.out_dir}")
    print("\nNext step — extract landmarks:")
    print(f"  python scripts/extract_thaisignvis_landmarks.py \\")
    print(f"      --data-root {args.out_dir} \\")
    print(f"      --out-dir   data/thaisignvis \\")
    print(f"      --limit     200")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
