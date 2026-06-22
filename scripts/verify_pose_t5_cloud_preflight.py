from __future__ import annotations

import argparse
import json
import sys
import tempfile
import zipfile
from collections import Counter
from pathlib import Path
import shutil

import pandas as pd


_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC_ROOT = _REPO_ROOT / "src"
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from tsl.data.unified import load_manifest


def _parse_csv_list(value: str) -> list[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _parse_expected_source_counts(value: str) -> dict[str, int]:
    result: dict[str, int] = {}
    for item in _parse_csv_list(value):
        name, sep, raw_count = item.partition("=")
        if not sep:
            raise ValueError(f"expected source count entry must be source=count, got {item!r}")
        result[name.strip()] = int(raw_count.strip())
    return result


def _materialize_archived_root(source_root: Path) -> Path:
    stage_root = Path(tempfile.mkdtemp(prefix="pose-t5-preflight-"))
    shutil.copy2(source_root / "manifest.csv", stage_root / "manifest.csv")
    quality_path = source_root / "manifest_quality.json"
    if quality_path.is_file():
        shutil.copy2(quality_path, stage_root / "manifest_quality.json")
    feature_root = stage_root / "features"
    feature_root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(source_root / "features.zip", "r") as archive:
        archive.extractall(feature_root)
    return stage_root


def _summarize_root(data_root: str, *, required_files: tuple[str, ...]) -> dict:
    root = Path(data_root).resolve()
    missing_required_files = [name for name in required_files if not (root / name).is_file()]
    manifest_rows = 0
    manifest_source_counts: dict[str, int] = {}
    if (root / "manifest.csv").is_file():
        manifest = pd.read_csv(root / "manifest.csv")
        manifest_rows = int(len(manifest))
        if "source" in manifest.columns:
            manifest_source_counts = {
                str(source): int(count)
                for source, count in Counter(str(item) for item in manifest["source"]).items()
            }
    resolved_examples = []
    materialized_root: Path | None = None
    if not missing_required_files:
        load_root = root
        if (root / "features.zip").is_file():
            materialized_root = _materialize_archived_root(root)
            load_root = materialized_root
        try:
            resolved_examples = load_manifest(str(load_root))
        finally:
            if materialized_root is not None and materialized_root.exists():
                shutil.rmtree(materialized_root, ignore_errors=True)
    resolved_source_counts = Counter(example.source for example in resolved_examples)
    return {
        "data_root": str(root),
        "missing_required_files": missing_required_files,
        "manifest_rows": manifest_rows,
        "manifest_source_counts": manifest_source_counts,
        "resolved_examples": len(resolved_examples),
        "resolved_source_counts": {
            str(source): int(count) for source, count in sorted(resolved_source_counts.items())
        },
    }


def verify_cloud_preflight(
    data_roots: str,
    *,
    expected_manifest_rows: int = 0,
    expected_resolved_examples: int = 0,
    expected_source_counts: str = "",
    required_files: tuple[str, ...] = ("manifest.csv", "manifest_quality.json"),
) -> dict:
    roots = _parse_csv_list(data_roots)
    expected_counts = _parse_expected_source_counts(expected_source_counts) if expected_source_counts else {}
    root_summaries = [_summarize_root(root, required_files=required_files) for root in roots]

    aggregate_manifest_rows = sum(int(item["manifest_rows"]) for item in root_summaries)
    aggregate_resolved_examples = sum(int(item["resolved_examples"]) for item in root_summaries)
    aggregate_manifest_source_counts = Counter()
    aggregate_resolved_source_counts = Counter()
    failures: list[str] = []

    for item in root_summaries:
        if item["missing_required_files"]:
            failures.append(
                f"{item['data_root']} missing required files: {', '.join(item['missing_required_files'])}"
            )
        aggregate_manifest_source_counts.update(item["manifest_source_counts"])
        aggregate_resolved_source_counts.update(item["resolved_source_counts"])

    if expected_manifest_rows > 0 and aggregate_manifest_rows != expected_manifest_rows:
        failures.append(
            f"manifest rows {aggregate_manifest_rows} != expected {expected_manifest_rows}"
        )
    if expected_resolved_examples > 0 and aggregate_resolved_examples != expected_resolved_examples:
        failures.append(
            f"resolved examples {aggregate_resolved_examples} != expected {expected_resolved_examples}"
        )
    if expected_counts:
        if dict(sorted(aggregate_manifest_source_counts.items())) != dict(sorted(expected_counts.items())):
            failures.append(
                "manifest source counts "
                + json.dumps(dict(sorted(aggregate_manifest_source_counts.items())), ensure_ascii=False)
                + " != expected "
                + json.dumps(dict(sorted(expected_counts.items())), ensure_ascii=False)
            )
        if dict(sorted(aggregate_resolved_source_counts.items())) != dict(sorted(expected_counts.items())):
            failures.append(
                "resolved source counts "
                + json.dumps(dict(sorted(aggregate_resolved_source_counts.items())), ensure_ascii=False)
                + " != expected "
                + json.dumps(dict(sorted(expected_counts.items())), ensure_ascii=False)
            )

    return {
        "passed": not failures,
        "data_roots": roots,
        "required_files": list(required_files),
        "expected_manifest_rows": expected_manifest_rows,
        "expected_resolved_examples": expected_resolved_examples,
        "expected_source_counts": dict(sorted(expected_counts.items())),
        "aggregate_manifest_rows": aggregate_manifest_rows,
        "aggregate_resolved_examples": aggregate_resolved_examples,
        "aggregate_manifest_source_counts": dict(sorted(aggregate_manifest_source_counts.items())),
        "aggregate_resolved_source_counts": dict(sorted(aggregate_resolved_source_counts.items())),
        "roots": root_summaries,
        "failures": failures,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify that cloud-mounted PoseT5 dataset roots resolve exactly as expected.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-roots", required=True)
    parser.add_argument("--expected-manifest-rows", type=int, default=0)
    parser.add_argument("--expected-resolved-examples", type=int, default=0)
    parser.add_argument("--expected-source-counts", default="")
    parser.add_argument("--required-files", default="manifest.csv,manifest_quality.json")
    parser.add_argument("--report-json", default="")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    report = verify_cloud_preflight(
        args.data_roots,
        expected_manifest_rows=int(args.expected_manifest_rows),
        expected_resolved_examples=int(args.expected_resolved_examples),
        expected_source_counts=str(args.expected_source_counts),
        required_files=tuple(_parse_csv_list(args.required_files)),
    )
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    if str(args.report_json).strip():
        target = Path(args.report_json).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
