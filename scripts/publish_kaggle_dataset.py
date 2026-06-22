from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create or version a staged Kaggle dataset directory."
    )
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--message", default="Update dataset")
    parser.add_argument("--dir-mode", default="skip", choices=["skip", "tar"])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--temp-dir",
        default="",
        help="Optional TEMP/TMP override for Kaggle CLI on Windows.",
    )
    return parser


def _run(cmd: list[str], *, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        check=False,
        text=True,
        capture_output=True,
        env=env,
    )


def _looks_like_create_success(result: subprocess.CompletedProcess[str]) -> bool:
    if result.returncode != 0:
        return False
    combined = "\n".join(part for part in [result.stdout, result.stderr] if part).lower()
    if "dataset creation error" in combined:
        return False
    return True


def _kaggle_env(temp_dir: str = "") -> dict[str, str]:
    env = os.environ.copy()
    if temp_dir:
        temp_path = Path(temp_dir).resolve()
        temp_path.mkdir(parents=True, exist_ok=True)
        env["TEMP"] = str(temp_path)
        env["TMP"] = str(temp_path)
    return env


def create_or_version_dataset(
    dataset_dir: str | Path,
    *,
    message: str,
    dir_mode: str = "skip",
    temp_dir: str = "",
    quiet: bool = False,
) -> dict:
    dataset_path = Path(dataset_dir).resolve()
    if not (dataset_path / "dataset-metadata.json").is_file():
        raise FileNotFoundError(f"dataset-metadata.json not found in {dataset_path}")

    env = _kaggle_env(temp_dir)
    shared_args = [
        "-p",
        str(dataset_path),
        "--dir-mode",
        dir_mode,
    ]
    if quiet:
        shared_args.append("-q")

    create_cmd = [sys.executable, "-m", "kaggle", "datasets", "create", *shared_args]
    create_result = _run(create_cmd, env=env)
    if _looks_like_create_success(create_result):
        return {
            "action": "create",
            "dataset_dir": str(dataset_path),
            "stdout": create_result.stdout.strip(),
            "stderr": create_result.stderr.strip(),
        }

    version_cmd = [
        sys.executable,
        "-m",
        "kaggle",
        "datasets",
        "version",
        *shared_args,
        "-m",
        message,
    ]
    version_result = _run(version_cmd, env=env)
    if version_result.returncode != 0:
        error_text = version_result.stderr.strip() or version_result.stdout.strip()
        raise RuntimeError(error_text or "kaggle datasets version failed")

    return {
        "action": "version",
        "dataset_dir": str(dataset_path),
        "stdout": version_result.stdout.strip(),
        "stderr": version_result.stderr.strip(),
    }


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    result = create_or_version_dataset(
        args.dataset_dir,
        message=args.message,
        dir_mode=args.dir_mode,
        temp_dir=args.temp_dir,
        quiet=args.quiet,
    )
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
