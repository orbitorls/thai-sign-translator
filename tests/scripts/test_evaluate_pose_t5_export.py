from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.evaluate_pose_t5_export import main


class _FakeTranslator:
    def __init__(self):
        self.last_kwargs = None

    def translate_batch(self, features_batch, **kwargs):
        self.last_kwargs = kwargs
        outputs = []
        for features in features_batch:
            idx = int(features[0, 0])
            sentence = {
                0: "alpha",
                1: "wrong",
                2: "gamma",
            }[idx]
            outputs.append(SimpleNamespace(sentence=sentence, score=0.9))
        return outputs


def test_evaluate_pose_t5_export_writes_report_and_samples(tmp_path, monkeypatch):
    export_dir = tmp_path / "export"
    export_dir.mkdir()
    (export_dir / "runtime_metadata.json").write_text(
        json.dumps({"checkpoint_step": 2100}),
        encoding="utf-8",
    )

    examples = [
        SimpleNamespace(example_id="e0", source="youtube", target_text="alpha", features_path="f0"),
        SimpleNamespace(example_id="e1", source="tsl51", target_text="beta", features_path="f1"),
        SimpleNamespace(example_id="e2", source="youtube", target_text="gamma", features_path="f2"),
    ]

    monkeypatch.setattr("scripts.evaluate_pose_t5_export.load_manifest", lambda root: examples)
    monkeypatch.setattr(
        "scripts.evaluate_pose_t5_export.split_by_video",
        lambda all_examples, fracs, seed: {"train": [], "val": all_examples},
    )
    monkeypatch.setattr(
        "scripts.evaluate_pose_t5_export.PoseT5Translator.from_checkpoint_dir",
        lambda export_dir, device="cpu": _FakeTranslator(),
    )
    monkeypatch.setattr(
        "scripts.evaluate_pose_t5_export.load_features",
        lambda path: np.array(
            [[0, 0]] if path == "f0" else ([[1, 0]] if path == "f1" else [[2, 0]]),
            dtype=np.float32,
        ),
    )

    report_json = tmp_path / "report.json"
    samples_json = tmp_path / "samples.json"
    code = main(
        [
            "--export-dir",
            str(export_dir),
            "--data-roots",
            "data/a",
            "--report-json",
            str(report_json),
            "--samples-json",
            str(samples_json),
            "--val-subset-size",
            "2",
        ]
    )

    assert code == 0
    report = json.loads(report_json.read_text(encoding="utf-8"))
    assert report["n"] == 2
    assert report["exact_match"] == 1
    assert report["seed"] == 42
    assert report["data_roots"] == ["data/a"]
    assert report["split_policy"] == "auto"
    assert report["runtime_metadata"]["checkpoint_step"] == 2100
    assert report["manifest_quality"]["passed"] is False
    assert report["promotion_status"]["ready"] is False
    assert report["source_counts"] == {"tsl51": 1, "youtube": 1}
    assert sorted(report["source_metrics"]) == ["tsl51", "youtube"]
    samples = json.loads(samples_json.read_text(encoding="utf-8"))
    assert samples[0]["hypothesis"] == "alpha"
    assert samples[1]["hypothesis"] == "wrong"


def test_evaluate_pose_t5_export_rejects_missing_required_sources(tmp_path, monkeypatch):
    export_dir = tmp_path / "export"
    export_dir.mkdir()

    examples = [
        SimpleNamespace(example_id="e0", source="tsl51", target_text="alpha", features_path="f0"),
        SimpleNamespace(example_id="e1", source="tsl51", target_text="beta", features_path="f1"),
    ]

    monkeypatch.setattr("scripts.evaluate_pose_t5_export.load_manifest", lambda root: examples)
    monkeypatch.setattr(
        "scripts.evaluate_pose_t5_export.split_by_video",
        lambda all_examples, fracs, seed: {"train": [], "val": all_examples},
    )
    monkeypatch.setattr(
        "scripts.evaluate_pose_t5_export.PoseT5Translator.from_checkpoint_dir",
        lambda export_dir, device="cpu": _FakeTranslator(),
    )
    monkeypatch.setattr(
        "scripts.evaluate_pose_t5_export.load_features",
        lambda path: np.array([[0, 0]], dtype=np.float32),
    )

    with pytest.raises(ValueError, match="missing required sources"):
        main(
            [
                "--export-dir",
                str(export_dir),
                "--data-roots",
                "data/a",
                "--required-sources",
                "tsl51,thaisignvis",
                "--val-subset-size",
                "2",
            ]
        )


def test_evaluate_pose_t5_export_prefers_manifest_splits_and_report_roots(tmp_path, monkeypatch):
    export_dir = tmp_path / "export"
    export_dir.mkdir()

    train_example = SimpleNamespace(
        example_id="train-0",
        source="tsl51",
        target_text="alpha",
        features_path="f0",
        split="train",
    )
    val_example = SimpleNamespace(
        example_id="val-0",
        source="tsl51",
        target_text="beta",
        features_path="f1",
        split="val",
    )
    examples = [train_example, val_example]

    split_calls = {"count": 0}

    def _fail_if_called(all_examples, fracs, seed):
        split_calls["count"] += 1
        return {"train": [], "val": all_examples}

    monkeypatch.setattr("scripts.evaluate_pose_t5_export.load_manifest", lambda root: examples)
    monkeypatch.setattr("scripts.evaluate_pose_t5_export.split_by_video", _fail_if_called)
    monkeypatch.setattr(
        "scripts.evaluate_pose_t5_export.PoseT5Translator.from_checkpoint_dir",
        lambda export_dir, device="cpu": _FakeTranslator(),
    )
    monkeypatch.setattr(
        "scripts.evaluate_pose_t5_export.load_features",
        lambda path: np.array([[0, 0]] if path == "f0" else [[1, 0]], dtype=np.float32),
    )

    report_json = tmp_path / "report.json"
    code = main(
        [
            "--export-dir",
            str(export_dir),
            "--data-roots",
            "/kaggle/input/thai-sign-tsl51-flat-v3",
            "--report-data-roots",
            "data/tsl51_v3",
            "--split-policy",
            "auto",
            "--report-json",
            str(report_json),
            "--val-subset-size",
            "5",
        ]
    )

    assert code == 0
    assert split_calls["count"] == 0
    report = json.loads(report_json.read_text(encoding="utf-8"))
    assert report["data_roots"] == ["data/tsl51_v3"]
    assert report["val_subset_size"] == 1


def test_evaluate_pose_t5_export_forwards_decoding_args(tmp_path, monkeypatch):
    export_dir = tmp_path / "export"
    export_dir.mkdir()
    examples = [
        SimpleNamespace(example_id="e0", source="tsl51", target_text="alpha", features_path="f0", split="train"),
        SimpleNamespace(example_id="e1", source="tsl51", target_text="beta", features_path="f1", split="val"),
    ]
    fake_translator = _FakeTranslator()

    monkeypatch.setattr("scripts.evaluate_pose_t5_export.load_manifest", lambda root: examples)
    monkeypatch.setattr(
        "scripts.evaluate_pose_t5_export.PoseT5Translator.from_checkpoint_dir",
        lambda export_dir, device="cpu": fake_translator,
    )
    monkeypatch.setattr(
        "scripts.evaluate_pose_t5_export.load_features",
        lambda path: np.array([[0, 0]] if path == "f0" else [[1, 0]], dtype=np.float32),
    )

    code = main(
        [
            "--export-dir",
            str(export_dir),
            "--data-roots",
            "data/a",
            "--beam-size",
            "1",
            "--max-new-tokens",
            "48",
            "--no-repeat-ngram-size",
            "none",
            "--repetition-penalty",
            "none",
            "--length-penalty",
            "1.0",
        ]
    )

    assert code == 0
    assert fake_translator.last_kwargs == {
        "max_new_tokens": 48,
        "beam_size": 1,
        "no_repeat_ngram_size": None,
        "repetition_penalty": None,
        "length_penalty": 1.0,
    }


def test_evaluate_pose_t5_export_applies_readiness_thresholds(tmp_path, monkeypatch):
    export_dir = tmp_path / "export"
    export_dir.mkdir()
    examples = [
        SimpleNamespace(example_id="e0", source="tsl51", target_text="alpha", features_path="f0", split="train"),
        SimpleNamespace(example_id="e1", source="tsl51", target_text="alpha", features_path="f1", split="val"),
    ]

    monkeypatch.setattr("scripts.evaluate_pose_t5_export.load_manifest", lambda root: examples)
    monkeypatch.setattr(
        "scripts.evaluate_pose_t5_export.PoseT5Translator.from_checkpoint_dir",
        lambda export_dir, device="cpu": _FakeTranslator(),
    )
    monkeypatch.setattr(
        "scripts.evaluate_pose_t5_export.load_features",
        lambda path: np.array([[0, 0]], dtype=np.float32),
    )

    report_json = tmp_path / "report.json"
    code = main(
        [
            "--export-dir",
            str(export_dir),
            "--data-roots",
            "data/a",
            "--report-json",
            str(report_json),
            "--min-val-chrf",
            "101",
            "--min-val-exact-match-pct",
            "101",
        ]
    )

    assert code == 0
    report = json.loads(report_json.read_text(encoding="utf-8"))
    assert report["promotion_status"]["ready"] is False
    assert any("chrf" in failure.lower() for failure in report["promotion_status"]["failures"])


def test_evaluate_pose_t5_export_can_filter_eval_sources(tmp_path, monkeypatch):
    export_dir = tmp_path / "export"
    export_dir.mkdir()

    train_example = SimpleNamespace(
        example_id="train-0",
        source="tsl51",
        target_text="alpha",
        features_path="f0",
        split="train",
    )
    val_examples = [
        SimpleNamespace(example_id="val-0", source="tsl51", target_text="alpha", features_path="f1", split="val"),
        SimpleNamespace(example_id="val-1", source="thaisignvis", target_text="beta", features_path="f2", split="val"),
    ]
    examples = [train_example, *val_examples]

    monkeypatch.setattr("scripts.evaluate_pose_t5_export.load_manifest", lambda root: examples)
    monkeypatch.setattr(
        "scripts.evaluate_pose_t5_export.PoseT5Translator.from_checkpoint_dir",
        lambda export_dir, device="cpu": _FakeTranslator(),
    )
    monkeypatch.setattr(
        "scripts.evaluate_pose_t5_export.load_features",
        lambda path: np.array([[0, 0]], dtype=np.float32),
    )

    report_json = tmp_path / "report.json"
    code = main(
        [
            "--export-dir",
            str(export_dir),
            "--data-roots",
            "data/a",
            "--report-json",
            str(report_json),
            "--eval-sources",
            "tsl51",
            "--manifest-quality-sources",
            "tsl51",
            "--required-sources",
            "tsl51",
        ]
    )

    assert code == 0
    report = json.loads(report_json.read_text(encoding="utf-8"))
    assert report["source_counts"] == {"tsl51": 1}
    assert report["eval_sources"] == ["tsl51"]
    assert report["manifest_quality"]["gated_sources"] == ["tsl51"]


def test_evaluate_pose_t5_export_falls_back_for_single_video_required_source(tmp_path, monkeypatch):
    export_dir = tmp_path / "export"
    export_dir.mkdir()

    train_examples = [
        SimpleNamespace(
            example_id="train-tsl51",
            source="tsl51",
            target_text="alpha",
            features_path="f0",
            metadata={"video_id": "video-tsl51"},
        ),
        SimpleNamespace(
            example_id="train-yt",
            source="youtube_sl25_thai",
            target_text="beta",
            features_path="f1",
            metadata={"video_id": "video-yt"},
        ),
    ]
    thai_examples = [
        SimpleNamespace(
            example_id="thai-0",
            source="thaisignvis",
            target_text="gamma",
            features_path="f2",
            metadata={"video_id": "thai-only-video"},
        ),
        SimpleNamespace(
            example_id="thai-1",
            source="thaisignvis",
            target_text="gamma",
            features_path="f3",
            metadata={"video_id": "thai-only-video"},
        ),
    ]
    all_examples = [*train_examples, *thai_examples]

    monkeypatch.setattr("scripts.evaluate_pose_t5_export.load_manifest", lambda root: all_examples)
    monkeypatch.setattr(
        "scripts.evaluate_pose_t5_export.split_by_video",
        lambda all_examples, fracs, seed: {"train": all_examples, "val": train_examples},
    )
    monkeypatch.setattr(
        "scripts.evaluate_pose_t5_export.PoseT5Translator.from_checkpoint_dir",
        lambda export_dir, device="cpu": _FakeTranslator(),
    )
    monkeypatch.setattr(
        "scripts.evaluate_pose_t5_export.load_features",
        lambda path: np.array([[0, 0]], dtype=np.float32),
    )

    report_json = tmp_path / "report.json"
    code = main(
        [
            "--export-dir",
            str(export_dir),
            "--data-roots",
            "data/a",
            "--report-json",
            str(report_json),
            "--split-policy",
            "video",
            "--required-sources",
            "tsl51,thaisignvis,youtube_sl25_thai",
            "--eval-sources",
            "tsl51,thaisignvis,youtube_sl25_thai",
            "--val-subset-size",
            "3",
        ]
    )

    assert code == 0
    report = json.loads(report_json.read_text(encoding="utf-8"))
    assert report["source_counts"]["thaisignvis"] >= 1
    assert report["eval_source_fallbacks"] == [
        {
            "source": "thaisignvis",
            "reason": "single_video_source_fallback",
            "video_id_count": 1,
            "example_count": 2,
            "video_ids": ["thai-only-video"],
        }
    ]
