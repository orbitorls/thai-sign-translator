"""Tests for the :class:`SentenceTranslator` inference wrapper."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from tsl.inference.sentence_translator import (
    SentencePrediction,
    SentenceTranslator,
)
from tsl.models.slt import SignToTextTransformer
from tsl.text.tokenizer import CharTokenizer


def _make_model(vocab_size: int = 20) -> SignToTextTransformer:
    torch.manual_seed(0)
    return SignToTextTransformer(
        input_dim=162,
        vocab_size=vocab_size,
        d_model=16,
        nhead=4,
        num_encoder_layers=1,
        num_decoder_layers=1,
        dim_feedforward=32,
        dropout=0.0,
    )


def _make_tokenizer() -> CharTokenizer:
    return CharTokenizer(["ก", "ข", "ค", "ง", "จ"])


# ---------------------------------------------------------------------------
# from_model_and_tokenizer path
# ---------------------------------------------------------------------------


def test_from_model_and_tokenizer_roundtrip():
    tok = _make_tokenizer()
    model = _make_model(vocab_size=tok.vocab_size)
    translator = SentenceTranslator.from_model_and_tokenizer(model, tok)

    features = np.random.randn(5, 162).astype(np.float32)
    pred = translator.translate(features)

    assert isinstance(pred, SentencePrediction)
    assert isinstance(pred.sentence, str)
    assert isinstance(pred.token_ids, list)
    assert all(isinstance(i, int) for i in pred.token_ids)
    assert 0.0 <= pred.score <= 1.0


def test_from_model_and_tokenizer_default_max_len_returns_prediction():
    tok = _make_tokenizer()
    model = _make_model(vocab_size=tok.vocab_size)
    translator = SentenceTranslator.from_model_and_tokenizer(model, tok)

    pred = translator.translate(np.random.randn(8, 162).astype(np.float32))
    assert isinstance(pred, SentencePrediction)
    assert len(pred.token_ids) >= 1
    assert pred.token_ids[0] == tok.bos_id


# ---------------------------------------------------------------------------
# Edge cases on features
# ---------------------------------------------------------------------------


def test_translate_empty_features():
    tok = _make_tokenizer()
    model = _make_model(vocab_size=tok.vocab_size)
    translator = SentenceTranslator.from_model_and_tokenizer(model, tok)

    pred = translator.translate(np.zeros((0, 162), dtype=np.float32))
    assert pred == SentencePrediction(sentence="", token_ids=[], score=0.0)


def test_translate_invalid_shape_raises():
    tok = _make_tokenizer()
    model = _make_model(vocab_size=tok.vocab_size)
    translator = SentenceTranslator.from_model_and_tokenizer(model, tok)

    with pytest.raises(ValueError):
        translator.translate(np.zeros((10,), dtype=np.float32))


# ---------------------------------------------------------------------------
# File I/O path
# ---------------------------------------------------------------------------


def test_translator_raises_missing_files():
    with pytest.raises(FileNotFoundError):
        SentenceTranslator("/nonexistent/path/to/checkpoint")
