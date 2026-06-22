from __future__ import annotations

import argparse
import json
import shutil
import time
import zipfile
from pathlib import Path

import pandas as pd

try:
    from .build_pose_t5_mixed_manifest import build_mixed_manifest
    from .package_portable_manifest_dataset import (
        build_archived_portable_dataset_dir,
        build_portable_dataset_dir,
    )
    from .package_colab_bundle import build_bundle
except ImportError:  # pragma: no cover - direct script execution
    from build_pose_t5_mixed_manifest import build_mixed_manifest
    from package_portable_manifest_dataset import (
        build_archived_portable_dataset_dir,
        build_portable_dataset_dir,
    )
    from package_colab_bundle import build_bundle


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Stage current code and a portable mixed dataset for Kaggle PoseT5 training."
    )
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parent.parent))
    parser.add_argument("--mixed-source-root", default="data/mixed_all_train_v6")
    parser.add_argument(
        "--build-mixed-manifest",
        type=str,
        default="false",
        choices=["true", "false"],
        help="Refresh the mixed-source manifest from the current dataset roots before staging Kaggle assets.",
    )
    parser.add_argument(
        "--mixed-data-roots",
        default="data/tsl51_v3,data/thaisignvis_v3_probe,data/youtube_sl25_thai_v3",
        help="Comma-separated data roots used when --build-mixed-manifest=true.",
    )
    parser.add_argument(
        "--mixed-train-only-sources",
        default="thaisignvis,youtube_sl25_thai",
        help="Comma-separated train-only sources passed to the mixed-manifest builder.",
    )
    parser.add_argument(
        "--mixed-required-sources",
        default="tsl51",
        help="Required sources passed to the mixed-manifest builder.",
    )
    parser.add_argument(
        "--mixed-manifest-quality-sources",
        default="tsl51",
        help="Manifest-quality gated sources passed to the mixed-manifest builder.",
    )
    parser.add_argument(
        "--mixed-allow-missing-roots",
        type=str,
        default="true",
        choices=["true", "false"],
        help="Whether the mixed-manifest builder should skip missing data roots.",
    )
    parser.add_argument("--staging-root", default="kaggle_upload")
    parser.add_argument("--code-dataset-slug", default="thai-sign-code")
    parser.add_argument("--code-dataset-id", default="orbitorls/thai-sign-code")
    parser.add_argument("--code-dataset-title", default="Thai Sign Language Translator - Code")
    parser.add_argument("--data-dataset-slug", default="thai-sign-mixed-all-v6-archived")
    parser.add_argument("--data-dataset-id", default="orbitorls/thai-sign-mixed-all-v6-archived")
    parser.add_argument("--data-dataset-title", default="Thai Sign Language - Mixed All Train v6 Archived")
    parser.add_argument(
        "--feature-layout",
        choices=["landmarks", "flat"],
        default="flat",
        help="Feature layout for the staged Kaggle dataset. Flat avoids Kaggle CLI directory-skip traps.",
    )
    parser.add_argument(
        "--archive-features",
        action="store_true",
        help="Zip staged feature files into a single archive to avoid slow Kaggle CLI per-file uploads.",
    )
    return parser


def _reset_dir(path: Path) -> None:
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
        return
    for child in path.iterdir():
        if child.name == "dataset-metadata.json":
            continue
        if child.is_dir():
            for attempt in range(3):
                try:
                    shutil.rmtree(child)
                    break
                except PermissionError:
                    if attempt == 2:
                        raise
                    time.sleep(0.5)
        else:
            for attempt in range(3):
                try:
                    child.unlink()
                    break
                except PermissionError:
                    if attempt == 2:
                        raise
                    time.sleep(0.5)


def _write_dataset_metadata(path: Path, dataset_id: str, title: str) -> None:
    metadata = {
        "title": title,
        "id": dataset_id,
        "isPrivate": True,
        "licenses": [{"name": "CC0-1.0"}],
    }
    (path / "dataset-metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _resolve_bool_flag(value: str | bool | None) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _stage_code_dataset(repo_root: Path, target_root: Path) -> list[str]:
    bundle_path = target_root / "repo_bundle.zip"
    build_bundle(repo_root, bundle_path)
    return ["repo_bundle.zip"]


def _archive_feature_files(data_root: Path, feature_layout: str) -> dict | None:
    feature_base = data_root if feature_layout == "flat" else data_root / "landmarks"
    if not feature_base.is_dir():
        return None
    feature_files = sorted(path for path in feature_base.rglob("*.npy") if path.is_file())
    if not feature_files:
        return None

    archive_name = "features.zip"
    archive_path = data_root / archive_name
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for feature_path in feature_files:
            relative_feature_path = feature_path.relative_to(feature_base)
            archive.write(feature_path, relative_feature_path.as_posix())

    manifest_path = data_root / "manifest.csv"
    if manifest_path.is_file():
        manifest = pd.read_csv(manifest_path)
        if "npy_path" in manifest.columns:
            manifest["npy_path"] = [
                (Path("features") / Path(str(value).replace("\\", "/")).name).as_posix()
                for value in manifest["npy_path"]
            ]
            manifest.to_csv(manifest_path, index=False, encoding="utf-8")

    for feature_path in feature_files:
        feature_path.unlink()
    if feature_layout != "flat" and feature_base.exists():
        shutil.rmtree(feature_base)

    return {
        "archive_name": archive_name,
        "archived_features": len(feature_files),
    }


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    repo_root = Path(args.repo_root).resolve()
    staging_root = (repo_root / args.staging_root).resolve()
    mixed_source_root = (repo_root / args.mixed_source_root).resolve()

    mixed_manifest_summary = None
    if _resolve_bool_flag(args.build_mixed_manifest):
        mixed_manifest_summary = build_mixed_manifest(
            argparse.Namespace(
                data_roots=",".join(
                    str((repo_root / raw_root.strip()).resolve())
                    for raw_root in str(args.mixed_data_roots).split(",")
                    if raw_root.strip()
                ),
                out_dir=str(mixed_source_root),
                train_only_sources=args.mixed_train_only_sources,
                required_sources=args.mixed_required_sources,
                manifest_quality_sources=args.mixed_manifest_quality_sources,
                allow_missing_roots=args.mixed_allow_missing_roots,
            )
        )

    code_root = staging_root / args.code_dataset_slug
    data_root = staging_root / args.data_dataset_slug

    code_root.mkdir(parents=True, exist_ok=True)
    data_root.mkdir(parents=True, exist_ok=True)
    _reset_dir(code_root)
    _write_dataset_metadata(code_root, args.code_dataset_id, args.code_dataset_title)

    copied = _stage_code_dataset(repo_root, code_root)
    archive_summary = None
    if args.archive_features:
        portable_summary = build_archived_portable_dataset_dir(
            str(mixed_source_root),
            str(data_root),
        )
        archive_summary = {
            "archive_name": portable_summary["archive_name"],
            "archived_features": portable_summary["archived_features"],
        }
    else:
        portable_summary = build_portable_dataset_dir(
            str(mixed_source_root),
            str(data_root),
            feature_layout=args.feature_layout,
        )
    _write_dataset_metadata(data_root, args.data_dataset_id, args.data_dataset_title)

    summary = {
        "code_dataset_dir": str(code_root),
        "code_includes": copied,
        "data_dataset_dir": str(data_root),
        "mixed_source_root": str(mixed_source_root),
        "mixed_manifest_summary": mixed_manifest_summary,
        "portable_rows": portable_summary["rows"],
        "portable_features": portable_summary["copied_features"],
        "feature_layout": portable_summary["feature_layout"],
        "features_archive": archive_summary["archive_name"] if archive_summary else None,
        "archived_features": archive_summary["archived_features"] if archive_summary else 0,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
