from __future__ import annotations

import argparse
import json
import os
import sys


_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from scripts._bootstrap import ensure_repo_paths

ensure_repo_paths()

from tsl.data.quality import audit_dataset_splits
from tsl.data.registry import get_dataset_spec, list_dataset_specs, load_dataset_splits


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit dataset manifests and split readiness.")
    parser.add_argument(
        "--dataset",
        required=True,
        choices=[spec.name for spec in list_dataset_specs()],
        help="Dataset name from the data registry.",
    )
    parser.add_argument("--data-root", required=True, help="Path to the dataset root.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for synthetic split policies.")
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the full audit report as JSON.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    spec = get_dataset_spec(args.dataset)
    splits = load_dataset_splits(args.dataset, args.data_root, seed=args.seed)
    report = audit_dataset_splits(splits, load_features=spec.feature_loader)
    payload = {
        "dataset": spec.name,
        "source": spec.source,
        "input_dim": spec.input_dim,
        "split_policy": spec.split_policy,
        "schema_version": spec.schema_version,
        "license_name": spec.license_name,
        "provenance": spec.provenance,
        **report,
    }
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"dataset: {payload['dataset']}")
        print(f"source: {payload['source']}")
        print(f"split_policy: {payload['split_policy']}")
        print(f"split_counts: {payload['split_counts']}")
        print(f"source_counts: {payload['source_counts']}")
        print(f"target_uniqueness_ratio: {payload['target_uniqueness_ratio']:.4f}")
        print(f"repeated_target_coverage: {payload['repeated_target_coverage']:.4f}")
        print(f"train_only_oov_rate: {payload['train_only_oov_rate']:.4f}")
        print(f"feature_stats: {payload['feature_stats']}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
