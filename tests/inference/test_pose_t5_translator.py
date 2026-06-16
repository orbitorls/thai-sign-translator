"""Tests for PoseT5Translator inference wrapper.

Uses a tiny MT5 config to avoid network access.
"""
from __future__ import annotations

import json
import os
import tempfile

import numpy as np
import pytest
import torch

from transformers import MT5Config, MT5ForConditionalGeneration, AutoTokenizer


# ---------------------------------------------------------------------------
# Tiny MT5 fixture
# ---------------------------------------------------------------------------

TINY_CONFIG = MT5Config(
    d_model=32,
    num_heads=2,
    num_layers=1,
    d_ff=64,
    d_kv=16,
    vocab_size=250112,
    decoder_start_token_id=0,
    eos_token_id=1,
    pad_token_id=0,
)


def _tiny_mt5_factory(*args, **kwargs) -> MT5ForConditionalGeneration:
    return MT5ForConditionalGeneration(TINY_CONFIG)


@pytest.fixture()
def patch_mt5(monkeypatch):
    monkeypatch.setattr(
        "tsl.models.pose_t5.MT5ForConditionalGeneration.from_pretrained",
        _tiny_mt5_factory,
    )


@pytest.fixture()
def tiny_model(patch_mt5):
    from tsl.models.pose_t5 import PoseToTextT5

    return PoseToTextT5(
        input_dim=312,
        num_encoder_layers=1,
        encoder_dropout=0.0,
        downsample_factor=4,
        base_model_name="google/mt5-small",
    )


class _FakeTokenizer:
    """Minimal tokenizer stub for tests (no network access needed)."""

    pad_token_id = 0
    bos_token_id = 0
    eos_token_id = 1

    def decode(self, ids, skip_special_tokens: bool = True) -> str:
        # Return a deterministic non-empty string for non-empty id lists
        filtered = [i for i in ids if i not in (0, 1)] if skip_special_tokens else ids
        return "ก ข" if filtered else ""


# ---------------------------------------------------------------------------
# Constructor / in-memory path
# ---------------------------------------------------------------------------


def test_predict_returns_correct_types(tiny_model):
    from tsl.inference.pose_t5_translator import PoseT5Translator, PoseT5Prediction

    translator = PoseT5Translator(model=tiny_model, tokenizer=_FakeTokenizer(), device="cpu")
    features = np.random.randn(8, 312).astype(np.float32)
    pred = translator.translate(features, max_new_tokens=5, beam_size=1)

    assert isinstance(pred, PoseT5Prediction)
    assert isinstance(pred.sentence, str)
    assert isinstance(pred.token_ids, list)
    assert all(isinstance(i, int) for i in pred.token_ids)
    assert isinstance(pred.score, float)
    assert 0.0 <= pred.score <= 1.0


def test_predict_returns_prediction_instance(tiny_model):
    from tsl.inference.pose_t5_translator import PoseT5Translator, PoseT5Prediction

    translator = PoseT5Translator(model=tiny_model, tokenizer=_FakeTokenizer(), device="cpu")
    features = np.random.randn(16, 312).astype(np.float32)
    pred = translator.translate(features, max_new_tokens=4, beam_size=2)

    assert isinstance(pred, PoseT5Prediction)
    # token_ids should be a plain list, not a tensor
    assert isinstance(pred.token_ids, list)


def test_translate_empty_features(tiny_model):
    from tsl.inference.pose_t5_translator import PoseT5Translator, PoseT5Prediction

    translator = PoseT5Translator(model=tiny_model, tokenizer=_FakeTokenizer(), device="cpu")
    features = np.zeros((0, 312), dtype=np.float32)
    pred = translator.translate(features)

    assert pred == PoseT5Prediction(sentence="", token_ids=[], score=0.0)


def test_translate_wrong_ndim_raises(tiny_model):
    from tsl.inference.pose_t5_translator import PoseT5Translator

    translator = PoseT5Translator(model=tiny_model, tokenizer=_FakeTokenizer(), device="cpu")
    with pytest.raises(ValueError):
        translator.translate(np.zeros((10, 312, 1), dtype=np.float32))


def test_model_stays_in_eval_mode(tiny_model):
    from tsl.inference.pose_t5_translator import PoseT5Translator

    translator = PoseT5Translator(model=tiny_model, tokenizer=_FakeTokenizer(), device="cpu")
    assert not translator.model.training


# ---------------------------------------------------------------------------
# from_checkpoint_dir
# ---------------------------------------------------------------------------


def test_from_checkpoint_dir_loads_and_translates(monkeypatch):
    """from_checkpoint_dir saves + loads a tiny model and produces a prediction."""
    from tsl.models.pose_t5 import PoseToTextT5
    from tsl.inference.pose_t5_translator import PoseT5Translator, PoseT5Prediction

    monkeypatch.setattr(
        "tsl.models.pose_t5.MT5ForConditionalGeneration.from_pretrained",
        _tiny_mt5_factory,
    )

    model = PoseToTextT5(
        input_dim=312,
        num_encoder_layers=1,
        encoder_dropout=0.0,
        downsample_factor=4,
        base_model_name="google/mt5-small",
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        model.save_pretrained(tmpdir)

        # Patch both from_pretrained calls (model load + tokenizer via AutoTokenizer)
        monkeypatch.setattr(
            "tsl.models.pose_t5.MT5ForConditionalGeneration.from_pretrained",
            _tiny_mt5_factory,
        )

        # Patch AutoTokenizer.from_pretrained at the transformers module level
        # (AutoTokenizer is imported lazily inside from_checkpoint_dir, so we
        # patch the source object directly)
        import transformers
        original_from_pretrained = transformers.AutoTokenizer.from_pretrained
        monkeypatch.setattr(
            transformers.AutoTokenizer,
            "from_pretrained",
            classmethod(lambda cls, *a, **kw: _FakeTokenizer()),
        )

        translator = PoseT5Translator.from_checkpoint_dir(tmpdir, device="cpu")

    assert isinstance(translator, PoseT5Translator)
    features = np.random.randn(8, 312).astype(np.float32)
    pred = translator.translate(features, max_new_tokens=4, beam_size=1)
    assert isinstance(pred, PoseT5Prediction)
