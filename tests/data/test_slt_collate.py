"""Tests for the SLT batch collate function."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from tsl.data.manifest import SignTextExample
from tsl.data.slt_collate import SltBatch, slt_collate
from tsl.text.tokenizer import CharTokenizer


_FEATURE_DIM = 162


def _make_landmark_csv(path: Path, n_frames: int) -> None:
    cols = ["frame", "t_ms"]
    for i in range(_FEATURE_DIM // 3):
        cols.append(f"lm_{i}_x")
        cols.append(f"lm_{i}_y")
        cols.append(f"lm_{i}_z")
    rows = []
    for t in range(n_frames):
        row = {"frame": t, "t_ms": t * 33}
        for i in range(_FEATURE_DIM // 3):
            row[f"lm_{i}_x"] = float(t * 100 + i)
            row[f"lm_{i}_y"] = float(t * 100 + i) + 0.1
            row[f"lm_{i}_z"] = float(t * 100 + i) + 0.2
        rows.append(row)
    pd.DataFrame(rows, columns=cols).to_csv(path, index=False)


def _make_example(tmp_path: Path, name: str, n_frames: int, text: str) -> SignTextExample:
    p = tmp_path / f"{name}.csv"
    _make_landmark_csv(p, n_frames)
    return SignTextExample(
        example_id=name,
        source="tsl51",
        split="train",
        features_path=str(p),
        target_text=text,
    )


def _load_features(path: str) -> np.ndarray:
    """Test loader that mirrors tsl51.load_landmark_sequence."""
    return pd.read_csv(path).to_numpy()[:, 2:].astype(np.float32)


def _tokenizer() -> CharTokenizer:
    return CharTokenizer(["ฉัน", "กิน", "ข้าว", "สวัสดี", "ครับ"])


def test_slt_collate_basic_shape(tmp_path):
    examples = [
        _make_example(tmp_path, "a", 4, "ฉัน"),
        _make_example(tmp_path, "b", 7, "กิน"),
        _make_example(tmp_path, "c", 5, "ข้าว"),
    ]
    tok = _tokenizer()
    batch = slt_collate(examples, tok, load_features=_load_features)

    assert isinstance(batch, SltBatch)
    assert batch.src.shape[0] == 3
    assert batch.src.shape[1] == 7  # T_src_max
    assert batch.src.shape[2] == _FEATURE_DIM
    assert batch.src.dtype == torch.float32
    assert batch.tgt.shape[0] == 3
    assert batch.tgt.ndim == 2
    assert batch.tgt.dtype == torch.long
    assert batch.src_lengths.ndim == 1
    assert batch.src_lengths.shape[0] == 3
    assert batch.src_lengths.dtype == torch.long
    assert batch.tgt_lengths.ndim == 1
    assert batch.tgt_lengths.shape[0] == 3
    assert batch.tgt_lengths.dtype == torch.long
    assert len(batch.target_texts) == 3
    assert batch.target_texts == ["ฉัน", "กิน", "ข้าว"]


def test_slt_collate_pads_shorter(tmp_path):
    examples = [
        _make_example(tmp_path, "a", 4, "ฉัน"),
        _make_example(tmp_path, "b", 7, "กิน"),
        _make_example(tmp_path, "c", 5, "ข้าว"),
    ]
    tok = _tokenizer()
    batch = slt_collate(examples, tok, load_features=_load_features)

    assert batch.src.shape == (3, 7, _FEATURE_DIM)
    assert batch.src_lengths.tolist() == [4, 7, 5]
    # Padding beyond the real length must be all-zero.
    assert torch.all(batch.src[0, 4:] == 0.0)
    # The real frames of the first example should be the first 4 rows.
    assert not torch.all(batch.src[0, :4] == 0.0)


def test_slt_collate_target_has_bos_eos(tmp_path):
    examples = [
        _make_example(tmp_path, "a", 4, "ฉัน"),
        _make_example(tmp_path, "b", 7, "กิน"),
    ]
    tok = _tokenizer()
    batch = slt_collate(examples, tok, load_features=_load_features)

    # Each row's first token is BOS, last is EOS, and the chars in
    # between decode back to the original text.
    for i, ex in enumerate(examples):
        row = batch.tgt[i].tolist()
        assert row[0] == tok.bos_id
        # The last non-pad position should be EOS.
        eos_positions = [j for j, t in enumerate(row) if t == tok.eos_id]
        assert eos_positions, "expected an EOS in the row"
        # All padding follows the EOS.
        last_eos = eos_positions[0]
        for j in range(last_eos + 1, len(row)):
            assert row[j] == tok.pad_id, f"non-pad token after EOS at row {i}, col {j}"


def test_slt_collate_tgt_lengths_exclude_specials(tmp_path):
    # "ฉัน" is 3 code points. With bos+eos the row has 5 tokens, but
    # tgt_lengths must report the real 3 chars only.
    examples = [_make_example(tmp_path, "a", 4, "ฉัน")]
    tok = _tokenizer()
    batch = slt_collate(examples, tok, load_features=_load_features)

    assert batch.tgt.shape == (1, 5)  # 3 chars + bos + eos
    assert batch.tgt_lengths.tolist() == [3]
    assert int(batch.tgt[0, 0]) == tok.bos_id
    assert int(batch.tgt[0, -1]) == tok.eos_id


def test_slt_collate_empty_features_handled(tmp_path):
    # One normal example and one with no frames at all. The batch
    # should still come out as a regular (B, T_src_max, D) tensor.
    good = _make_example(tmp_path, "good", 3, "ฉัน")
    empty = SignTextExample(
        example_id="empty",
        source="tsl51",
        split="train",
        features_path=str(tmp_path / "empty.csv"),
        target_text="กิน",
    )
    # Write an empty CSV with only the frame / t_ms columns so the
    # loader returns a (0, 0) array. We do NOT infer D from the file
    # itself - the collate should fall back to the other example's dim.
    pd.DataFrame(columns=["frame", "t_ms"]).to_csv(empty.features_path, index=False)

    tok = _tokenizer()
    batch = slt_collate([good, empty], tok, load_features=_load_features)

    assert batch.src.shape == (2, 3, _FEATURE_DIM)
    assert batch.src_lengths.tolist() == [3, 0]
    # The padded "synthetic" frame for the empty row sits at index 0
    # and must be all zeros.
    assert torch.all(batch.src[1] == 0.0)
    # Tokens still build correctly for the empty-features example.
    assert batch.target_texts == ["ฉัน", "กิน"]


def test_slt_collate_empty_batch_raises():
    tok = _tokenizer()
    with pytest.raises(ValueError, match="empty batch"):
        slt_collate([], tok, load_features=_load_features)
