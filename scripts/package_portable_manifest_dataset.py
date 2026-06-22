from __future__ import annotations

import argparse
import csv
import hashlib
import time
import shutil
import tempfile
import zipfile
from pathlib import Path

import pandas as pd


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Package a manifest-backed dataset into a portable zip bundle."
    )
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--output-zip", required=True)
    parser.add_argument(
        "--feature-layout",
        choices=["landmarks", "flat"],
        default="landmarks",
        help="How copied feature files should be laid out inside the portable dataset.",
    )
    return parser


def _resolve_npy_path(source_root: Path, raw_value: str) -> Path:
    normalized = str(raw_value).replace("\\", "/")
    candidate = Path(normalized)
    if not candidate.is_absolute():
        candidate = source_root / normalized
    return candidate


def _portable_feature_name(segment_id: str, row_index: int) -> str:
    digest = hashlib.sha1(segment_id.encode("utf-8")).hexdigest()[:12]
    return f"seg_{row_index:05d}_{digest}.npy"


def _reset_output_dir(path: Path) -> None:
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
        return
    for child in path.iterdir():
        for attempt in range(20):
            try:
                if child.is_dir():
                    shutil.rmtree(child)
                else:
                    child.unlink()
                break
            except (PermissionError, OSError):
                if attempt == 19:
                    raise
                time.sleep(1.0)


def build_portable_dataset_dir(
    source_root: str,
    output_dir: str,
    *,
    feature_layout: str = "landmarks",
) -> dict:
    source_root_path = Path(source_root).resolve()
    manifest_path = source_root_path / "manifest.csv"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    df = pd.read_csv(manifest_path)
    if "segment_id" not in df.columns or "npy_path" not in df.columns:
        raise ValueError("Manifest must contain segment_id and npy_path columns.")

    output_dir_path = Path(output_dir).resolve()
    _reset_output_dir(output_dir_path)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    if feature_layout not in {"landmarks", "flat"}:
        raise ValueError(f"Unsupported feature layout: {feature_layout}")
    feature_dir = output_dir_path if feature_layout == "flat" else output_dir_path / "landmarks"
    feature_dir.mkdir(parents=True, exist_ok=True)

    portable_rows: list[dict] = []
    copied = 0
    for row_index, (_, row) in enumerate(df.iterrows()):
        source_npy = _resolve_npy_path(source_root_path, row["npy_path"])
        if not source_npy.is_file():
            continue
        segment_id = str(row["segment_id"])
        target_name = _portable_feature_name(segment_id, row_index)
        target_rel = Path(target_name) if feature_layout == "flat" else Path("landmarks") / target_name
        shutil.copy2(source_npy, feature_dir / target_name)
        portable_row = dict(row)
        portable_row["npy_path"] = target_rel.as_posix()
        portable_rows.append(portable_row)
        copied += 1

    if not portable_rows:
        raise RuntimeError("No feature files were copied into the portable dataset bundle.")

    portable_manifest = output_dir_path / "manifest.csv"
    with portable_manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(portable_rows[0].keys()))
        writer.writeheader()
        writer.writerows(portable_rows)

    quality_path = source_root_path / "manifest_quality.json"
    if quality_path.is_file():
        shutil.copy2(quality_path, output_dir_path / "manifest_quality.json")

    return {
        "source_root": str(source_root_path),
        "output_dir": str(output_dir_path),
        "rows": len(portable_rows),
        "copied_features": copied,
        "feature_layout": feature_layout,
    }


def build_archived_portable_dataset_dir(source_root: str, output_dir: str) -> dict:
    source_root_path = Path(source_root).resolve()
    manifest_path = source_root_path / "manifest.csv"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    df = pd.read_csv(manifest_path)
    if "segment_id" not in df.columns or "npy_path" not in df.columns:
        raise ValueError("Manifest must contain segment_id and npy_path columns.")

    output_dir_path = Path(output_dir).resolve()
    _reset_output_dir(output_dir_path)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    archive_path = output_dir_path / "features.zip"
    portable_rows: list[dict] = []
    copied = 0
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for row_index, (_, row) in enumerate(df.iterrows()):
            source_npy = _resolve_npy_path(source_root_path, row["npy_path"])
            if not source_npy.is_file():
                continue
            segment_id = str(row["segment_id"])
            target_name = _portable_feature_name(segment_id, row_index)
            archive.write(source_npy, target_name)
            portable_row = dict(row)
            portable_row["npy_path"] = (Path("features") / target_name).as_posix()
            portable_rows.append(portable_row)
            copied += 1

    if not portable_rows:
        raise RuntimeError("No feature files were copied into the archived portable dataset bundle.")

    portable_manifest = output_dir_path / "manifest.csv"
    with portable_manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(portable_rows[0].keys()))
        writer.writeheader()
        writer.writerows(portable_rows)

    quality_path = source_root_path / "manifest_quality.json"
    if quality_path.is_file():
        shutil.copy2(quality_path, output_dir_path / "manifest_quality.json")

    return {
        "source_root": str(source_root_path),
        "output_dir": str(output_dir_path),
        "rows": len(portable_rows),
        "copied_features": copied,
        "feature_layout": "archive",
        "archive_name": archive_path.name,
        "archived_features": copied,
    }

def build_portable_bundle(source_root: str, output_zip: str, *, feature_layout: str = "landmarks") -> dict:
    output_zip_path = Path(output_zip).resolve()
    output_zip_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="portable-dataset-") as tmp_dir:
        summary = build_portable_dataset_dir(source_root, tmp_dir, feature_layout=feature_layout)
        bundle_root = Path(tmp_dir)
        with zipfile.ZipFile(output_zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(bundle_root.rglob("*")):
                if path.is_file():
                    archive.write(path, path.relative_to(bundle_root).as_posix())

    summary["output_zip"] = str(output_zip_path)
    return summary


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    summary = build_portable_bundle(args.source_root, args.output_zip, feature_layout=args.feature_layout)
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
