"""Tests for the T5-compatible pose collate function."""
from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from tsl.data.manifest import SignTextExample
from tsl.data.pose_t5_collate import PoseT5Batch, pose_t5_collate

_FEAT_DIM = 312
_PAD_ID = 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_example(name: str, text: str) -> SignTextExample:
    return SignTextExample(
        example_id=name,
        source="test",
        split="train",
        features_path=f"/fake/{name}.npy",
        target_text=text,
    )


def _make_features(n_frames: int, value: float = 1.0) -> np.ndarray:
    return np.full((n_frames, _FEAT_DIM), value, dtype=np.float32)


def _make_hf_tokenizer(texts: list[str], pad_id: int = _PAD_ID) -> Any:
    """Return a mock HF tokenizer that produces deterministic padded ids.

    The longest text gets one token per character; shorter texts are
    right-padded with ``pad_id`` so that all rows have the same length.
    """
    max_len = max(len(t) for t in texts)

    def _tokenize(batch_texts, padding=True, return_tensors="pt"):
        ids = []
        local_max = max(len(t) for t in batch_texts)
        for text in batch_texts:
            # Use ord() of each char as a simple deterministic token id,
            # offset by 1 so none collide with pad_id = 0.
            row = [ord(c) % 1000 + 1 for c in text]
            # Pad to local_max with pad_id
            row += [pad_id] * (local_max - len(row))
            ids.append(row)
        return {"input_ids": torch.tensor(ids, dtype=torch.long)}

    tok = MagicMock()
    tok.pad_token_id = pad_id
    tok.side_effect = _tokenize
    return tok


def _build_tokenizer(examples: list[SignTextExample]) -> Any:
    texts = [ex.target_text for ex in examples]
    return _make_hf_tokenizer(texts)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestPoseT5CollateShapes:
    """Correct output shapes for a batch of 3 variable-length examples."""

    def test_basic_shapes(self):
        examples = [
            _make_example("a", "สวัสดี"),   # 6 chars
            _make_example("b", "กิน"),       # 3 chars
            _make_example("c", "ข้าว"),      # 4 chars
        ]
        frame_counts = [4, 7, 5]
        features_map = {ex.features_path: _make_features(n) for ex, n in zip(examples, frame_counts)}
        load_features = features_map.__getitem__

        tokenizer = _build_tokenizer(examples)
        batch = pose_t5_collate(examples, tokenizer, load_features)

        assert isinstance(batch, PoseT5Batch)
        # src shape: (B, T_max, 312)
        assert batch.src.shape == (3, 7, _FEAT_DIM)
        # src_lengths shape: (B,)
        assert batch.src_lengths.shape == (3,)
        assert batch.src_lengths.tolist() == frame_counts
        # src_mask shape: (B, T_max)
        assert batch.src_mask.shape == (3, 7)
        # labels shape: (B, T_tgt) — T_tgt determined by longest text
        assert batch.labels.shape[0] == 3
        assert batch.labels.ndim == 2


class TestPoseT5CollateSrcMask:
    """src_mask is True where real, False where padded."""

    def test_src_mask_values(self):
        examples = [
            _make_example("a", "hello"),
            _make_example("b", "hi"),
            _make_example("c", "bye"),
        ]
        frame_counts = [3, 6, 5]
        features_map = {ex.features_path: _make_features(n) for ex, n in zip(examples, frame_counts)}
        load_features = features_map.__getitem__

        tokenizer = _build_tokenizer(examples)
        batch = pose_t5_collate(examples, tokenizer, load_features)

        t_max = 6
        for i, length in enumerate(frame_counts):
            # Real positions should be True
            assert batch.src_mask[i, :length].all(), f"row {i}: expected True in real range"
            # Padded positions should be False
            if length < t_max:
                assert not batch.src_mask[i, length:].any(), f"row {i}: expected False in padded range"

    def test_src_mask_dtype(self):
        examples = [_make_example("a", "test")]
        load_features = lambda p: _make_features(3)
        tokenizer = _build_tokenizer(examples)
        batch = pose_t5_collate(examples, tokenizer, load_features)
        assert batch.src_mask.dtype == torch.bool


class TestPoseT5CollateLabels:
    """Labels have -100 at padding positions."""

    def test_labels_padding_replaced(self):
        # Use texts of different lengths so tokenizer must pad some rows
        examples = [
            _make_example("a", "กิน"),      # short
            _make_example("b", "สวัสดีครับ"),  # long
        ]
        load_features = lambda p: _make_features(4)

        texts = [ex.target_text for ex in examples]
        max_len = max(len(t) for t in texts)
        tokenizer = _make_hf_tokenizer(texts, pad_id=_PAD_ID)
        batch = pose_t5_collate(examples, tokenizer, load_features)

        # Every label position that was pad_id should now be -100
        for i, text in enumerate(texts):
            real_len = len(text)
            if real_len < max_len:
                padded_portion = batch.labels[i, real_len:]
                assert (padded_portion == -100).all(), (
                    f"row {i}: expected -100 at padding positions, got {padded_portion}"
                )

    def test_labels_real_tokens_preserved(self):
        examples = [_make_example("a", "ab")]
        load_features = lambda p: _make_features(2)
        tokenizer = _make_hf_tokenizer(["ab"], pad_id=_PAD_ID)
        batch = pose_t5_collate(examples, tokenizer, load_features)

        # The real tokens should NOT be -100
        for tok_id in batch.labels[0].tolist():
            assert tok_id != _PAD_ID, "real token should not equal pad_id"
            # -100 is also wrong for real tokens
            assert tok_id != -100, "real token should not be -100"


class TestPoseT5CollateEmptySequence:
    """Empty sequence (T=0) handled without error."""

    def test_empty_sequence_no_crash(self):
        examples = [
            _make_example("good", "สวัสดี"),
            _make_example("empty", "กิน"),
        ]
        features_map = {
            examples[0].features_path: _make_features(5),
            examples[1].features_path: np.zeros((0, _FEAT_DIM), dtype=np.float32),
        }
        load_features = features_map.__getitem__
        tokenizer = _build_tokenizer(examples)
        batch = pose_t5_collate(examples, tokenizer, load_features)

        assert batch.src.shape == (2, 5, _FEAT_DIM)
        assert batch.src_lengths.tolist() == [5, 0]

    def test_empty_sequence_mask_all_false(self):
        examples = [
            _make_example("good", "สวัสดี"),
            _make_example("empty", "กิน"),
        ]
        features_map = {
            examples[0].features_path: _make_features(3),
            examples[1].features_path: np.zeros((0, _FEAT_DIM), dtype=np.float32),
        }
        load_features = features_map.__getitem__
        tokenizer = _build_tokenizer(examples)
        batch = pose_t5_collate(examples, tokenizer, load_features)

        # The empty-sequence row's mask should be all False
        assert not batch.src_mask[1].any(), "empty sequence mask should be all False"

    def test_empty_sequence_src_zeros(self):
        examples = [_make_example("empty", "test")]
        load_features = lambda p: np.zeros((0, _FEAT_DIM), dtype=np.float32)
        tokenizer = _build_tokenizer(examples)
        batch = pose_t5_collate(examples, tokenizer, load_features)

        # src should be all zeros (the promoted synthetic frame)
        assert torch.all(batch.src == 0.0)
        assert batch.src_lengths.tolist() == [0]


class TestPoseT5CollateMaxSrcLen:
    """max_src_len truncates long sequences."""

    def test_truncation(self):
        examples = [_make_example("a", "test")]
        load_features = lambda p: _make_features(100)
        tokenizer = _build_tokenizer(examples)
        batch = pose_t5_collate(examples, tokenizer, load_features, max_src_len=20)

        assert batch.src.shape[1] == 20
        assert batch.src_lengths.tolist() == [20]

    def test_truncation_updates_mask(self):
        examples = [_make_example("a", "test")]
        load_features = lambda p: _make_features(100)
        tokenizer = _build_tokenizer(examples)
        batch = pose_t5_collate(examples, tokenizer, load_features, max_src_len=10)

        # All 10 positions should be real (True) after truncation
        assert batch.src_mask.shape == (1, 10)
        assert batch.src_mask[0].all()

    def test_short_sequence_not_truncated(self):
        examples = [_make_example("a", "test")]
        load_features = lambda p: _make_features(5)
        tokenizer = _build_tokenizer(examples)
        batch = pose_t5_collate(examples, tokenizer, load_features, max_src_len=50)

        # Not truncated, should remain 5 frames
        assert batch.src.shape[1] == 5
        assert batch.src_lengths.tolist() == [5]


class TestPoseT5CollateDtypes:
    """src is float32, labels is long, src_mask is bool."""

    def test_dtype_src(self):
        examples = [_make_example("a", "test")]
        load_features = lambda p: _make_features(4)
        tokenizer = _build_tokenizer(examples)
        batch = pose_t5_collate(examples, tokenizer, load_features)
        assert batch.src.dtype == torch.float32

    def test_dtype_labels(self):
        examples = [_make_example("a", "test")]
        load_features = lambda p: _make_features(4)
        tokenizer = _build_tokenizer(examples)
        batch = pose_t5_collate(examples, tokenizer, load_features)
        assert batch.labels.dtype == torch.long

    def test_dtype_src_mask(self):
        examples = [_make_example("a", "test")]
        load_features = lambda p: _make_features(4)
        tokenizer = _build_tokenizer(examples)
        batch = pose_t5_collate(examples, tokenizer, load_features)
        assert batch.src_mask.dtype == torch.bool

    def test_dtype_src_lengths(self):
        examples = [_make_example("a", "test")]
        load_features = lambda p: _make_features(4)
        tokenizer = _build_tokenizer(examples)
        batch = pose_t5_collate(examples, tokenizer, load_features)
        assert batch.src_lengths.dtype == torch.long
