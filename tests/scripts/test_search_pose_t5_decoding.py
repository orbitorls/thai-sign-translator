from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np

from scripts.search_pose_t5_decoding import main


class _FakeTranslator:
    def translate_batch(
        self,
        features_batch,
        *,
        max_new_tokens=72,
        beam_size=5,
        no_repeat_ngram_size=3,
        repetition_penalty=1.5,
        length_penalty=0.7,
    ):
        outputs = []
        for features in features_batch:
            idx = int(features[0, 0])
            if beam_size == 1 and length_penalty == 1.0:
                sentence = {0: "alpha", 1: "beta"}[idx]
            else:
                sentence = {0: "alpha", 1: "wrong"}[idx]
            outputs.append(SimpleNamespace(sentence=sentence, score=0.9))
        return outputs


def test_search_pose_t5_decoding_writes_best_report(tmp_path, monkeypatch):
    export_dir = tmp_path / "export"
    export_dir.mkdir()
    examples = [
        SimpleNamespace(example_id="e0", source="tsl51", target_text="alpha", features_path="f0", split="train"),
        SimpleNamespace(example_id="e1", source="tsl51", target_text="beta", features_path="f1", split="val"),
    ]

    monkeypatch.setattr("scripts.search_pose_t5_decoding._load_examples", lambda roots: examples)
    monkeypatch.setattr(
        "scripts.search_pose_t5_decoding.PoseT5Translator.from_checkpoint_dir",
        lambda export_dir, device="cpu": _FakeTranslator(),
    )
    monkeypatch.setattr(
        "scripts.search_pose_t5_decoding.load_features",
        lambda path: np.array([[0, 0]] if path == "f0" else [[1, 0]], dtype=np.float32),
    )

    report_json = tmp_path / "search.json"
    best_eval_json = tmp_path / "best_eval.json"
    best_samples_json = tmp_path / "best_samples.json"
    code = main(
        [
            "--export-dir",
            str(export_dir),
            "--data-roots",
            "data/a",
            "--report-json",
            str(report_json),
            "--best-eval-json",
            str(best_eval_json),
            "--best-samples-json",
            str(best_samples_json),
            "--val-subset-size",
            "1",
            "--required-sources",
            "tsl51",
            "--beam-size-grid",
            "1,5",
            "--length-penalty-grid",
            "1.0,0.7",
            "--no-repeat-ngram-grid",
            "none",
            "--repetition-penalty-grid",
            "none",
            "--max-new-tokens-grid",
            "48",
            "--max-trials",
            "4",
        ]
    )

    assert code == 0
    best_eval = json.loads(best_eval_json.read_text(encoding="utf-8"))
    assert best_eval["beam_size"] == 1
    assert best_eval["length_penalty"] == 1.0
    assert best_eval["exact_match_pct"] == 100.0
    best_samples = json.loads(best_samples_json.read_text(encoding="utf-8"))
    assert best_samples[0]["hypothesis"] == "beta"


def test_search_pose_t5_decoding_can_filter_eval_sources(tmp_path, monkeypatch):
    export_dir = tmp_path / "export"
    export_dir.mkdir()
    examples = [
        SimpleNamespace(example_id="e0", source="tsl51", target_text="alpha", features_path="f0", split="train"),
        SimpleNamespace(example_id="e1", source="tsl51", target_text="beta", features_path="f1", split="val"),
        SimpleNamespace(example_id="e2", source="thaisignvis", target_text="gamma", features_path="f2", split="val"),
    ]

    monkeypatch.setattr("scripts.search_pose_t5_decoding._load_examples", lambda roots: examples)
    monkeypatch.setattr(
        "scripts.search_pose_t5_decoding.PoseT5Translator.from_checkpoint_dir",
        lambda export_dir, device="cpu": _FakeTranslator(),
    )
    monkeypatch.setattr(
        "scripts.search_pose_t5_decoding.load_features",
        lambda path: np.array([[0, 0]] if path == "f0" else [[1, 0]], dtype=np.float32),
    )

    best_eval_json = tmp_path / "best_eval.json"
    code = main(
        [
            "--export-dir",
            str(export_dir),
            "--data-roots",
            "data/a",
            "--best-eval-json",
            str(best_eval_json),
            "--eval-sources",
            "tsl51",
            "--manifest-quality-sources",
            "tsl51",
            "--required-sources",
            "tsl51",
            "--beam-size-grid",
            "1",
            "--length-penalty-grid",
            "1.0",
            "--no-repeat-ngram-grid",
            "none",
            "--repetition-penalty-grid",
            "none",
            "--max-new-tokens-grid",
            "48",
            "--max-trials",
            "1",
        ]
    )

    assert code == 0
    best_eval = json.loads(best_eval_json.read_text(encoding="utf-8"))
    assert best_eval["source_counts"] == {"tsl51": 1}
    assert best_eval["eval_sources"] == ["tsl51"]


def test_search_pose_t5_decoding_falls_back_for_single_video_required_source(tmp_path, monkeypatch):
    export_dir = tmp_path / "export"
    export_dir.mkdir()
    train_examples = [
        SimpleNamespace(
            example_id="e0",
            source="tsl51",
            target_text="alpha",
            features_path="f0",
            split="train",
            metadata={"video_id": "video-tsl51"},
        ),
        SimpleNamespace(
            example_id="e1",
            source="youtube_sl25_thai",
            target_text="beta",
            features_path="f1",
            split="val",
            metadata={"video_id": "video-yt"},
        ),
    ]
    thai_examples = [
        SimpleNamespace(
            example_id="e2",
            source="thaisignvis",
            target_text="gamma",
            features_path="f2",
            split="train",
            metadata={"video_id": "thai-only-video"},
        )
    ]
    all_examples = [*train_examples, *thai_examples]

    monkeypatch.setattr("scripts.search_pose_t5_decoding._load_examples", lambda roots: all_examples)
    monkeypatch.setattr(
        "scripts.search_pose_t5_decoding._build_train_val_splits",
        lambda examples, split_policy, seed: {"train": all_examples, "val": train_examples},
    )
    monkeypatch.setattr(
        "scripts.search_pose_t5_decoding.PoseT5Translator.from_checkpoint_dir",
        lambda export_dir, device="cpu": _FakeTranslator(),
    )
    monkeypatch.setattr(
        "scripts.search_pose_t5_decoding.load_features",
        lambda path: np.array([[0, 0]], dtype=np.float32),
    )

    best_eval_json = tmp_path / "best_eval.json"
    code = main(
        [
            "--export-dir",
            str(export_dir),
            "--data-roots",
            "data/a",
            "--best-eval-json",
            str(best_eval_json),
            "--split-policy",
            "video",
            "--required-sources",
            "tsl51,thaisignvis,youtube_sl25_thai",
            "--eval-sources",
            "tsl51,thaisignvis,youtube_sl25_thai",
            "--beam-size-grid",
            "1",
            "--length-penalty-grid",
            "1.0",
            "--no-repeat-ngram-grid",
            "none",
            "--repetition-penalty-grid",
            "none",
            "--max-new-tokens-grid",
            "48",
            "--max-trials",
            "1",
            "--val-subset-size",
            "3",
        ]
    )

    assert code == 0
    best_eval = json.loads(best_eval_json.read_text(encoding="utf-8"))
    assert best_eval["source_counts"]["thaisignvis"] >= 1
    assert best_eval["eval_source_fallbacks"] == [
        {
            "source": "thaisignvis",
            "reason": "single_video_source_fallback",
            "video_id_count": 1,
            "example_count": 1,
            "video_ids": ["thai-only-video"],
        }
    ]
