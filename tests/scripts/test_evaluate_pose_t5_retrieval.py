from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import torch

from scripts.evaluate_pose_t5_retrieval import main


class _FakeModel:
    def encode_pooled(self, src, src_lengths, normalize=True):
        outputs = []
        for row_idx, length in enumerate(src_lengths.tolist()):
            bucket = int(src[row_idx, 0, 0].item())
            if bucket == 0:
                outputs.append(torch.tensor([1.0, 0.0], dtype=torch.float32, device=src.device))
            else:
                outputs.append(torch.tensor([0.0, 1.0], dtype=torch.float32, device=src.device))
        return torch.stack(outputs, dim=0)


class _FakeTranslator:
    def __init__(self):
        self.model = _FakeModel()
        self.device = "cpu"


def test_evaluate_pose_t5_retrieval_writes_report_and_samples(tmp_path, monkeypatch):
    export_dir = tmp_path / "export"
    export_dir.mkdir()
    (export_dir / "runtime_metadata.json").write_text(
        json.dumps({"checkpoint_step": 3075}),
        encoding="utf-8",
    )

    train_examples = [
        SimpleNamespace(example_id="t0", source="tsl51", split="train", target_text="alpha", features_path="f0"),
        SimpleNamespace(example_id="t1", source="thaisignvis", split="train", target_text="beta", features_path="f1"),
    ]
    val_examples = [
        SimpleNamespace(example_id="v0", source="tsl51", split="val", target_text="alpha", features_path="f0"),
        SimpleNamespace(example_id="v1", source="thaisignvis", split="val", target_text="beta", features_path="f1"),
    ]

    monkeypatch.setattr("scripts.evaluate_pose_t5_retrieval.load_manifest", lambda root: train_examples + val_examples)
    monkeypatch.setattr(
        "scripts.evaluate_pose_t5_retrieval.split_by_video",
        lambda all_examples, fracs, seed: {"train": train_examples, "val": val_examples},
    )
    monkeypatch.setattr(
        "scripts.evaluate_pose_t5_retrieval.PoseT5Translator.from_checkpoint_dir",
        lambda checkpoint_dir, device="cpu": _FakeTranslator(),
    )
    monkeypatch.setattr(
        "scripts.evaluate_pose_t5_retrieval.load_features",
        lambda path: np.array([[0.0, 0.0]], dtype=np.float32)
        if path == "f0"
        else np.array([[1.0, 0.0]], dtype=np.float32),
    )

    report_json = tmp_path / "report.json"
    samples_json = tmp_path / "samples.json"
    code = main(
        [
            "--checkpoint-dir",
            str(export_dir),
            "--data-roots",
            "data/a",
            "--report-json",
            str(report_json),
            "--samples-json",
            str(samples_json),
            "--val-subset-size",
            "2",
            "--top-k",
            "1,2",
        ]
    )

    assert code == 0
    report = json.loads(report_json.read_text(encoding="utf-8"))
    assert report["n"] == 2
    assert report["top1_exact"] == 1.0
    assert report["top2_exact"] == 1.0
    assert report["mrr"] == 1.0
    assert report["runtime_metadata"]["checkpoint_step"] == 3075
    assert sorted(report["source_metrics"]) == ["thaisignvis", "tsl51"]

    samples = json.loads(samples_json.read_text(encoding="utf-8"))
    assert samples[0]["top_predictions"][0] == "alpha"
    assert samples[1]["top_predictions"][0] == "beta"
