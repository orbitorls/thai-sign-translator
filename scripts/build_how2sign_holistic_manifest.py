from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


_SPLIT_PATHS = {
    "train": "train",
    "val": "val",
    "test": "test",
}

_SPLIT_METADATA = {
    "train": "how2sign_realigned_train.csv",
    "val": "how2sign_realigned_val.csv",
    "test": "how2sign_realigned_test.csv",
}

_FEATURE_LAYOUT_VERSION = "raw_mediapipe_543x3"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a manifest.csv for the public How2Sign Holistic Kaggle dataset."
    )
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--dataset-subdir", default="how2sign_holistic_features")
    parser.add_argument("--metadata-subdir", default="metadata")
    parser.add_argument("--view", default="frontal")
    parser.add_argument("--source", default="how2sign")
    return parser


def _resolve_dataset_root(input_root: Path, dataset_subdir: str) -> Path:
    candidate = input_root / dataset_subdir
    if candidate.is_dir():
        return candidate
    return input_root


def _read_metadata(metadata_path: Path) -> pd.DataFrame:
    return pd.read_csv(metadata_path, sep="\t")


def build_manifest(
    input_root: str | Path,
    out_dir: str | Path,
    *,
    dataset_subdir: str = "how2sign_holistic_features",
    metadata_subdir: str = "metadata",
    view: str = "frontal",
    source: str = "how2sign",
) -> dict:
    input_root_path = Path(input_root).resolve()
    out_dir_path = Path(out_dir).resolve()
    dataset_root = _resolve_dataset_root(input_root_path, dataset_subdir)
    metadata_root = dataset_root / metadata_subdir

    rows: list[dict] = []
    missing_feature_rows = 0
    source_counts: dict[str, int] = {}
    split_counts: dict[str, int] = {}
    for split, split_dir in _SPLIT_PATHS.items():
        metadata_path = metadata_root / _SPLIT_METADATA[split]
        if not metadata_path.is_file():
            raise FileNotFoundError(f"How2Sign metadata not found: {metadata_path}")
        features_root = dataset_root / split_dir / view
        if not features_root.is_dir():
            raise FileNotFoundError(f"How2Sign features directory not found: {features_root}")

        df = _read_metadata(metadata_path)
        for _, row in df.iterrows():
            sentence_name = str(row.get("SENTENCE_NAME", "")).strip()
            sentence_text = str(row.get("SENTENCE", "")).strip()
            video_id = str(row.get("VIDEO_ID", "")).strip()
            if not sentence_name or not sentence_text:
                continue
            feature_path = features_root / f"{sentence_name}_holistic.npy"
            if not feature_path.is_file():
                missing_feature_rows += 1
                continue
            rows.append(
                {
                    "segment_id": sentence_name,
                    "npy_path": str(feature_path),
                    "text": sentence_text,
                    "video_id": video_id or sentence_name,
                    "split": split,
                    "source": source,
                    "feature_layout_version": _FEATURE_LAYOUT_VERSION,
                }
            )
            source_counts[source] = source_counts.get(source, 0) + 1
            split_counts[split] = split_counts.get(split, 0) + 1

    if not rows:
        raise RuntimeError(f"No How2Sign rows with resolvable holistic features were found under {dataset_root}")

    out_dir_path.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir_path / "manifest.csv"
    pd.DataFrame(rows).to_csv(manifest_path, index=False, encoding="utf-8")

    summary = {
        "input_root": str(input_root_path),
        "dataset_root": str(dataset_root),
        "out_dir": str(out_dir_path),
        "rows": len(rows),
        "source_counts": source_counts,
        "split_counts": split_counts,
        "missing_feature_rows": missing_feature_rows,
        "feature_layout_version": _FEATURE_LAYOUT_VERSION,
        "view": view,
    }
    (out_dir_path / "build_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    summary = build_manifest(
        args.input_root,
        args.out_dir,
        dataset_subdir=args.dataset_subdir,
        metadata_subdir=args.metadata_subdir,
        view=args.view,
        source=args.source,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
