"""Validate requirements.txt declares every tech-stack dependency, pinned.

Every non-comment, non-blank line must name a package with a version
specifier (== or >=), and the full stack from the contract must be present.
"""

import os

import config

REQ_PATH = os.path.join(config.ROOT_DIR, "requirements.txt")

EXPECTED_PACKAGES = {
    "torch",
    "mediapipe",
    "fastapi",
    "uvicorn",
    "numpy",
    "pandas",
    "pyarrow",
    "pytest",
    "fastdtw",
    "scikit-learn",
    "matplotlib",
    "pydantic",
    "transformers",
    "sentencepiece",
    "sacrebleu",
}


def _parse_lines():
    with open(REQ_PATH, encoding="utf-8") as f:
        return [
            ln.strip()
            for ln in f
            if ln.strip() and not ln.strip().startswith("#")
        ]


def test_requirements_file_exists():
    assert os.path.isfile(REQ_PATH)


def test_every_line_is_pinned():
    for line in _parse_lines():
        assert "==" in line or ">=" in line, f"unpinned dependency: {line!r}"


def test_all_expected_packages_present():
    names = set()
    for line in _parse_lines():
        for sep in ("==", ">="):
            if sep in line:
                names.add(line.split(sep)[0].strip().lower())
                break
    missing = EXPECTED_PACKAGES - names
    assert not missing, f"missing required packages: {sorted(missing)}"
