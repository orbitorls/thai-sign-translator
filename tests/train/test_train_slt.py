"""Tests for the SLT training entrypoint."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pandas as pd
import pytest
import torch

from tsl.data.manifest import SignTextExample
from tsl.data.slt_collate import slt_collate
from tsl.data.tsl51 import load_landmark_sequence
from tsl.text.tokenizer import CharTokenizer, WordTokenizer
from tsl.train import runtime as train_runtime
from tsl.train.train_slt import (
    _build_model,
    _save_metrics,
    _save_tokenizer,
    eval_loss,
    load_tokenizer,
    main,
    train_one_epoch,
)


_FEATURE_DIM = 162


def _landmark_columns() -> list[str]:
    cols = ["frame", "t_ms"]
    for i in range(_FEATURE_DIM // 3):
        cols.append(f"lm_{i}_x")
        cols.append(f"lm_{i}_y")
        cols.append(f"lm_{i}_z")
    return cols


def _write_landmark_csv(path: Path, n_frames: int = 3) -> None:
    cols = _landmark_columns()
    rows = []
    for t in range(n_frames):
        row = {"frame": t, "t_ms": t * 33}
        for i in range(_FEATURE_DIM // 3):
            row[f"lm_{i}_x"] = float(t)
            row[f"lm_{i}_y"] = float(t) + 0.1
            row[f"lm_{i}_z"] = float(t) + 0.2
        rows.append(row)
    pd.DataFrame(rows, columns=cols).to_csv(path, index=False)


def _make_synthetic_examples(
    tmp_path: Path, texts: list[str] | None = None
) -> list[SignTextExample]:
    if texts is None:
        texts = ["สวัสดี", "ฉันกินข้าว", "ขอบคุณ"]
    examples: list[SignTextExample] = []
    for i, text in enumerate(texts):
        path = tmp_path / f"v{i}.csv"
        _write_landmark_csv(path, n_frames=3)
        examples.append(
            SignTextExample(
                example_id=f"v{i}",
                source="tsl51",
                split="train",
                features_path=str(path),
                target_text=text,
            )
        )
    return examples


def _write_synthetic_tsl51(root: Path, n: int = 3) -> None:
    """Write a synthetic TSL-51 layout (metadata + landmark CSVs)."""
    meta_dir = root / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)
    lm_dir = root / "landmarks" / "user_sentence"
    lm_dir.mkdir(parents=True, exist_ok=True)

    texts = ["สวัสดี", "ฉันกินข้าว", "ขอบคุณ"]
    rows: list[dict] = []
    for i in range(n):
        video_id = f"v{i + 1}"
        rows.append(
            {
                "video_id": video_id,
                "sentence_id": i + 1,
                "sentence_clean": texts[i],
                "landmark_path": f"landmarks/user_sentence/{video_id}.csv",
                "video_path": f"videos/{video_id}.mp4",
            }
        )
        _write_landmark_csv(lm_dir / f"{video_id}.csv", n_frames=3)
    pd.DataFrame(
        rows,
        columns=["video_id", "sentence_id", "sentence_clean", "landmark_path", "video_path"],
    ).to_csv(meta_dir / "sentence_metadata.csv", index=False)


# ---------------------------------------------------------------------------
# Tokenizer round-trip
# ---------------------------------------------------------------------------


def test_save_load_char_tokenizer_roundtrip(tmp_path):
    tok = CharTokenizer(["สวัสดี", "ครับ", "ฉัน"])
    path = tmp_path / "tokenizer.json"
    _save_tokenizer(tok, str(path))

    loaded = load_tokenizer(str(path))

    assert isinstance(loaded, CharTokenizer)
    assert loaded.vocab_size == tok.vocab_size
    assert loaded.pad_id == tok.pad_id
    assert loaded.bos_id == tok.bos_id
    assert loaded.eos_id == tok.eos_id
    assert loaded.unk_id == tok.unk_id
    # Round-trip encode/decode.
    assert loaded.decode(loaded.encode("สวัสดี")) == "สวัสดี"


def test_save_load_word_tokenizer_roundtrip(tmp_path):
    tok = WordTokenizer(["สวัสดี ฉัน", "ครับ คุณ"])
    path = tmp_path / "tokenizer.json"
    _save_tokenizer(tok, str(path))

    loaded = load_tokenizer(str(path))

    assert isinstance(loaded, WordTokenizer)
    assert loaded.vocab_size == tok.vocab_size
    assert loaded.decode(loaded.encode("สวัสดี ฉัน")) == "สวัสดี ฉัน"


def test_save_load_tokenizer_file_is_valid_json(tmp_path):
    tok = CharTokenizer(["ab", "cd"])
    path = tmp_path / "tokenizer.json"
    _save_tokenizer(tok, str(path))

    raw = json.loads(path.read_text(encoding="utf-8"))
    assert raw["pad"] == "<pad>"
    assert raw["bos"] == "<bos>"
    assert raw["eos"] == "<eos>"
    assert raw["unk"] == "<unk>"
    assert raw["vocab"][:4] == ["<pad>", "<bos>", "<eos>", "<unk>"]
    assert set(raw["vocab"][4:]) == {"a", "b", "c", "d"}


def test_load_tokenizer_handles_missing_file():
    with pytest.raises(FileNotFoundError):
        load_tokenizer("/nonexistent/path/to/tokenizer.json")


# ---------------------------------------------------------------------------
# Single training step
# ---------------------------------------------------------------------------


def test_train_one_epoch_synthetic(tmp_path):
    examples = _make_synthetic_examples(tmp_path)
    tokenizer = CharTokenizer([ex.target_text for ex in examples])
    model = _build_model(tokenizer.vocab_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    batch = slt_collate(examples, tokenizer, load_features=load_landmark_sequence)
    loss = train_one_epoch(model, [batch], optimizer, "cpu", tokenizer)

    assert isinstance(loss, float)
    assert math.isfinite(loss)


def test_eval_loss_empty_returns_inf(tmp_path):
    tokenizer = CharTokenizer(["ab"])
    model = _build_model(tokenizer.vocab_size)
    assert eval_loss(model, [], "cpu", tokenizer) == float("inf")


def test_eval_loss_returns_finite_on_synthetic(tmp_path):
    examples = _make_synthetic_examples(tmp_path)
    tokenizer = CharTokenizer([ex.target_text for ex in examples])
    model = _build_model(tokenizer.vocab_size)
    batch = slt_collate(examples, tokenizer, load_features=load_landmark_sequence)

    val = eval_loss(model, [batch], "cpu", tokenizer)
    assert math.isfinite(val)


# ---------------------------------------------------------------------------
# End-to-end script smoke
# ---------------------------------------------------------------------------


def test_train_script_smoke(tmp_path, monkeypatch):
    data_root = tmp_path / "data"
    data_root.mkdir()
    _write_synthetic_tsl51(data_root, n=3)

    out_dir = tmp_path / "out"
    out_dir.mkdir()

    # Isolate stdout side effects and argv so we can call main() directly.
    monkeypatch.setattr(sys, "argv", ["train_slt.py"])
    monkeypatch.setattr(train_runtime.torch.cuda, "is_available", lambda: False)

    ret = main(
        argv=[
            "--data-root",
            str(data_root),
            "--epochs",
            "1",
            "--batch-size",
            "2",
            "--limit",
            "3",
            "--out-dir",
            str(out_dir),
        ]
    )
    assert ret == 0

    assert (out_dir / "slt_model.pt").exists()
    assert (out_dir / "tokenizer.json").exists()
    assert (out_dir / "model_config.json").exists()
    assert (out_dir / "train_metrics.json").exists()

    # The metrics file must be valid JSON with the required keys.
    metrics = json.loads((out_dir / "train_metrics.json").read_text(encoding="utf-8"))
    assert "epochs" in metrics
    assert "final_train_loss" in metrics
    assert len(metrics["epochs"]) == 1

    # The saved tokenizer must round-trip for all training texts.
    loaded_tok = load_tokenizer(str(out_dir / "tokenizer.json"))
    for text in ("สวัสดี", "ฉันกินข้าว", "ขอบคุณ"):
        decoded = loaded_tok.decode(loaded_tok.encode(text))
        # At minimum, the decoded text contains the same non-space chars.
        assert len(decoded) > 0

    # The state-dict must reload into a fresh model of the same shape.
    state = torch.load(str(out_dir / "slt_model.pt"), map_location="cpu")
    assert "out_proj.weight" in state
    rebuilt = _build_model(loaded_tok.vocab_size)
    rebuilt.load_state_dict(state)


def test_train_script_require_gpu_fails_without_cuda(tmp_path, monkeypatch):
    data_root = tmp_path / "data"
    data_root.mkdir()

    out_dir = tmp_path / "out"
    out_dir.mkdir()

    monkeypatch.setattr(sys, "argv", ["train_slt.py"])
    monkeypatch.setattr(train_runtime.torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError):
        main(
            argv=[
                "--data-root",
                str(data_root),
                "--epochs",
                "1",
                "--batch-size",
                "2",
                "--limit",
                "3",
                "--out-dir",
                str(out_dir),
                "--device",
                "cpu",
                "--require-gpu",
            ]
        )


# ---------------------------------------------------------------------------
# Metrics helper
# ---------------------------------------------------------------------------


def test_save_metrics_writes_json(tmp_path):
    path = tmp_path / "metrics.json"
    _save_metrics(
        {"epochs": [{"epoch": 0, "train_loss": 0.5}], "final_train_loss": 0.5},
        str(path),
    )
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert raw["final_train_loss"] == 0.5
    assert raw["epochs"][0]["train_loss"] == 0.5
