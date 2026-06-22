"""Tests for the sentence-only inference runtime."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tsl.features.schema import RAW_MEDIAPIPE_543X3, SELECTED_312, TSL51_162
from tsl.inference.model_registry import SentenceModelMetadata
from tsl.inference.sentence_runtime import (
    FeatureSchemaMismatchError,
    SentenceRuntime,
)
from tsl.inference.sentence_translator import SentencePrediction


class StubTranslator:
    def __init__(self) -> None:
        self.last_features = None
        self.last_max_len = None

    def translate(self, features, max_len: int = 128) -> SentencePrediction:
        self.last_features = np.asarray(features, dtype=np.float32)
        self.last_max_len = max_len
        return SentencePrediction(sentence="hello", token_ids=[1, 2], score=0.8)


def _metadata(schema_id: str, input_dim: int) -> SentenceModelMetadata:
    return SentenceModelMetadata(
        checkpoint_dir=Path("checkpoints") / "stub",
        checkpoint_name="stub",
        input_dim=input_dim,
        feature_schema_id=schema_id,
        tokenizer_type="word",
        config={"input_dim": input_dim},
        metadata={},
    )


def test_runtime_converts_raw_mediapipe_to_selected_312():
    translator = StubTranslator()
    runtime = SentenceRuntime(_metadata(SELECTED_312, 312), translator)

    frames = np.zeros((2, 543, 3), dtype=np.float32)
    frames[:, 489] = [0.0, 0.0, 0.0]
    frames[:, 500] = [1.0, 0.0, 0.0]
    frames[:, 501] = [-1.0, 0.0, 0.0]

    pred = runtime.translate(frames, feature_schema=RAW_MEDIAPIPE_543X3, max_len=32)

    assert pred.sentence == "hello"
    assert translator.last_features is not None
    assert translator.last_features.shape == (2, 312)
    assert translator.last_max_len == 32


def test_runtime_rejects_raw_input_for_tsl51_model_without_mapping():
    translator = StubTranslator()
    runtime = SentenceRuntime(_metadata(TSL51_162, 162), translator)

    frames = np.zeros((1, 543, 3), dtype=np.float32)

    with pytest.raises(FeatureSchemaMismatchError) as excinfo:
        runtime.translate(frames, feature_schema=RAW_MEDIAPIPE_543X3)

    assert "raw_mediapipe_543x3" in str(excinfo.value)
    assert "tsl51_162" in str(excinfo.value)
