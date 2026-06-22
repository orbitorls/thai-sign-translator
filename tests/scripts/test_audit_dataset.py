from __future__ import annotations

import json
import os
import runpy
import sys

import numpy as np
import pandas as pd

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from scripts.audit_dataset import main


def _write_manifest(root) -> None:
    lm_dir = root / "landmarks"
    lm_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, split in enumerate(("train", "train", "val")):
        seg_id = f"seg_{i:03d}"
        rel_path = f"landmarks/{seg_id}.npy"
        np.save(root / rel_path, np.full((4 + i, 162), i, dtype=np.float32))
        rows.append(
            {
                "segment_id": seg_id,
                "npy_path": rel_path,
                "text": "shared target" if i < 2 else "held out tokens",
                "video_id": f"video_{i}",
                "start_ms": 0,
                "end_ms": 1000,
                "split": split,
            }
        )
    pd.DataFrame(rows).to_csv(root / "manifest.csv", index=False)


def test_audit_dataset_cli_json_output(tmp_path, capsys):
    _write_manifest(tmp_path)

    ret = main(["--dataset", "youtube_sl25", "--data-root", str(tmp_path), "--json"])

    assert ret == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["dataset"] == "youtube_sl25"
    assert payload["split_counts"] == {"train": 2, "val": 1, "test": 0}
    assert payload["feature_stats"]["sequences_scanned"] == 3


def test_audit_dataset_script_bootstrap_keeps_src_importable():
    script_path = os.path.join(_REPO_ROOT, "scripts", "audit_dataset.py")
    old_path = list(sys.path)
    try:
        sys.path[:] = [path for path in sys.path if path not in {_REPO_ROOT, os.path.join(_REPO_ROOT, "src")}]
        runpy.run_path(script_path, run_name="audit_dataset_bootstrap_test")
        assert os.path.join(_REPO_ROOT, "src") in sys.path
        assert _REPO_ROOT in sys.path
    finally:
        sys.path[:] = old_path
