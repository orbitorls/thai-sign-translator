from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import torch
import pandas as pd

from tsl.models.slt import SignToTextTransformer
from tsl.train.config import resolve_config
from tsl.train.train_slt import _load_pretrained_weights, _build_model, _resolve_input_dim


def test_resolve_input_dim_auto():
    assert _resolve_input_dim("how2sign", None) == 411
    assert _resolve_input_dim("tsl51", None) == 162
    assert _resolve_input_dim("finetune", None) == 162


def test_resolve_input_dim_explicit():
    assert _resolve_input_dim("how2sign", 100) == 100
    assert _resolve_input_dim("tsl51", 200) == 200


def test_load_pretrained_weights_loads_compatible_layers(tmp_path):
    cfg = resolve_config("small", input_dim=162)
    config_dict = {f.name: getattr(cfg, f.name) for f in cfg.__dataclass_fields__.values()}

    src_model = SignToTextTransformer(vocab_size=50, **config_dict)
    chkpt_path = tmp_path / "slt_model.pt"
    torch.save(src_model.state_dict(), str(chkpt_path))
    (tmp_path / "model_config.json").write_text(json.dumps({**config_dict, "vocab_size": 50}), encoding="utf-8")

    dst_model = SignToTextTransformer(vocab_size=50, **config_dict)
    dst_model = _load_pretrained_weights(dst_model, str(tmp_path), device="cpu")

    for key in src_model.state_dict():
        assert torch.equal(src_model.state_dict()[key], dst_model.state_dict()[key])


def test_load_pretrained_weights_skips_mismatched_shapes(tmp_path):
    cfg_src = resolve_config("small", input_dim=411)
    cfg_dst = resolve_config("small", input_dim=162)

    src_model = SignToTextTransformer(vocab_size=50, **{f.name: getattr(cfg_src, f.name) for f in cfg_src.__dataclass_fields__.values()})
    chkpt_path = tmp_path / "slt_model.pt"
    torch.save(src_model.state_dict(), str(chkpt_path))
    (tmp_path / "model_config.json").write_text(json.dumps({**cfg_src.__dict__, "vocab_size": 50}), encoding="utf-8")

    dst_model = SignToTextTransformer(vocab_size=80, **{f.name: getattr(cfg_dst, f.name) for f in cfg_dst.__dataclass_fields__.values()})
    dst_model = _load_pretrained_weights(dst_model, str(tmp_path), device="cpu")

    assert dst_model.input_proj.weight.shape[1] == 162
    assert dst_model.out_proj.weight.shape[0] == 80
    assert dst_model.tgt_embed.weight.shape[0] == 80


def test_load_pretrained_weights_missing_file_raises(tmp_path):
    cfg = resolve_config("small", input_dim=162)
    model = SignToTextTransformer(vocab_size=50, **{f.name: getattr(cfg, f.name) for f in cfg.__dataclass_fields__.values()})
    with pytest.raises(FileNotFoundError):
        _load_pretrained_weights(model, str(tmp_path), device="cpu")


def test_load_pretrained_weights_mismatched_config_raises(tmp_path):
    ckpt_dir = tmp_path / "ckpt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    cfg_saved = resolve_config("small", input_dim=162)
    saved_model = SignToTextTransformer(vocab_size=8, **{f.name: getattr(cfg_saved, f.name) for f in cfg_saved.__dataclass_fields__.values()})
    torch.save(saved_model.state_dict(), ckpt_dir / "slt_model.pt")
    (ckpt_dir / "model_config.json").write_text(json.dumps({**cfg_saved.__dict__, "vocab_size": 8}), encoding="utf-8")

    cfg_current = resolve_config("base", input_dim=162)
    current_model = SignToTextTransformer(vocab_size=8, **{f.name: getattr(cfg_current, f.name) for f in cfg_current.__dataclass_fields__.values()})

    with pytest.raises(ValueError, match="architecture does not match"):
        _load_pretrained_weights(current_model, str(ckpt_dir), device="cpu", expected_config=dict(cfg_current.__dict__))


def test_build_model_with_size_variants():
    for size in ("small", "base", "large"):
        model = _build_model(vocab_size=50, model_size=size, input_dim=162)
        assert isinstance(model, SignToTextTransformer)
        total = sum(p.numel() for p in model.parameters())
        assert total > 0


def test_load_data_limit_caps_training_only(tmp_path):
    data_root = tmp_path / "data"
    meta_dir = data_root / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)
    lm_dir = data_root / "landmarks" / "user_sentence"
    lm_dir.mkdir(parents=True, exist_ok=True)

    from tests.train.test_train_slt import _write_landmark_csv

    rows = []
    texts = ["สวัสดี", "ฉันกินข้าว", "ขอบคุณ"]
    for i in range(10):
        video_id = f"v{i + 1}"
        rows.append(
            {
                "video_id": video_id,
                "sentence_id": i + 1,
                "sentence_clean": texts[i % len(texts)],
                "landmark_path": f"landmarks/user_sentence/{video_id}.csv",
                "video_path": f"videos/{video_id}.mp4",
            }
        )
        _write_landmark_csv(lm_dir / f"{video_id}.csv", n_frames=3)

    pd.DataFrame(
        rows,
        columns=["video_id", "sentence_id", "sentence_clean", "landmark_path", "video_path"],
    ).to_csv(meta_dir / "sentence_metadata.csv", index=False)

    from tsl.train.train_slt import _load_data

    train_ex, val_ex, _ = _load_data("tsl51", str(data_root), limit=3)

    assert len(train_ex) == 3
    assert len(val_ex) == 1


def _write_synthetic_how2sign(root: Path, split: str, n: int = 2) -> None:
    split_dir = {"train": "train", "val": "validation", "test": "test"}[split]
    csv_dir = root / "sentence_level" / split_dir / "text/en/raw_text/re_aligned"
    csv_dir.mkdir(parents=True, exist_ok=True)
    kp_dir = root / "sentence_level" / split_dir / "rgb_front/features/openpose_output_fps_25/json"
    kp_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for i in range(n):
        name = f"sent_{split}_{i}"
        rows.append({"SENTENCE_NAME": name, "SENTENCE": f"hello {split} {i}"})
        arr = torch.randn(5, 137, 3).numpy()
        import numpy as np

        np.save(str(kp_dir / f"{name}.npy"), arr.astype("float32"))

    pd.DataFrame(rows, columns=["SENTENCE_NAME", "SENTENCE"]).to_csv(
        csv_dir / f"how2sign_realigned_{split}.csv", index=False
    )


def test_train_main_how2sign_smoke(tmp_path, monkeypatch):
    from tsl.train import train_slt as train_mod

    data_root = tmp_path / "how2sign"
    _write_synthetic_how2sign(data_root, "train", n=2)
    _write_synthetic_how2sign(data_root, "val", n=1)
    monkeypatch.setattr(train_mod, "resolve_device", lambda *args, **kwargs: torch.device("cpu"))

    out_dir = tmp_path / "out_h2s"
    ret = train_mod.main(
        [
            "--stage",
            "how2sign",
            "--data-root",
            str(data_root),
            "--epochs",
            "1",
            "--batch-size",
            "1",
            "--out-dir",
            str(out_dir),
            "--device",
            "auto",
        ]
    )

    assert ret == 0
    assert (out_dir / "slt_model.pt").exists()
    assert (out_dir / "model_config.json").exists()


def _write_npy_manifest(root: Path, n: int = 3, source: str = "youtube_sl25") -> None:
    lm_dir = root / "landmarks"
    lm_dir.mkdir(parents=True, exist_ok=True)
    import numpy as np

    rows = []
    for i in range(n):
        seg_id = f"seg_{i:04d}"
        npy_rel = f"landmarks/{seg_id}.npy"
        arr = np.random.randn(10, 162).astype("float32")
        np.save(str(root / npy_rel), arr)
        split = "val" if i == 0 else "train"
        rows.append({
            "segment_id": seg_id, "npy_path": npy_rel,
            "text": f"ประโยค{i}", "video_id": f"vid_{i}",
            "start_ms": 0, "end_ms": 5000, "split": split,
        })
    pd.DataFrame(rows).to_csv(root / "manifest.csv", index=False)


def test_load_data_youtube_sl25(tmp_path):
    data_root = tmp_path / "ytsl25"
    _write_npy_manifest(data_root, n=5)
    from tsl.train.train_slt import _load_data
    train_ex, val_ex, load_fn = _load_data("youtube_sl25", str(data_root), None)
    assert len(train_ex) > 0
    assert len(val_ex) > 0
    feat = load_fn(train_ex[0].features_path)
    assert feat.shape == (10, 162)


def test_load_data_combined(tmp_path):
    tsl51_root = tmp_path / "tsl51"
    meta_dir = tsl51_root / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)
    lm_dir = tsl51_root / "landmarks" / "user_sentence"
    lm_dir.mkdir(parents=True, exist_ok=True)
    from tests.train.test_train_slt import _write_landmark_csv
    rows_tsl51 = []
    for i in range(4):
        vid = f"v{i}"
        rows_tsl51.append({
            "video_id": vid, "sentence_id": i,
            "sentence_clean": f"คำ{i}", "landmark_path": f"landmarks/user_sentence/{vid}.csv",
            "video_path": f"videos/{vid}.mp4",
        })
        _write_landmark_csv(lm_dir / f"{vid}.csv", n_frames=5)
    pd.DataFrame(rows_tsl51, columns=["video_id","sentence_id","sentence_clean","landmark_path","video_path"]).to_csv(
        meta_dir / "sentence_metadata.csv", index=False)

    ytsl25_root = tmp_path / "ytsl25"
    _write_npy_manifest(ytsl25_root, n=4)

    from tsl.train.train_slt import _load_data
    data_root = f"{tsl51_root},{ytsl25_root}"
    train_ex, val_ex, load_fn = _load_data("combined", data_root, None)
    assert len(train_ex) > 0
    assert len(val_ex) > 0
    sources = {ex.source for ex in train_ex + val_ex}
    assert "tsl51" in sources
    assert "youtube_sl25" in sources
    tsl51_ex = next(ex for ex in train_ex if ex.source == "tsl51")
    yt_ex = next(ex for ex in train_ex if ex.source == "youtube_sl25")
    feat_tsl51 = load_fn(tsl51_ex.features_path)
    feat_yt = load_fn(yt_ex.features_path)
    assert feat_tsl51.ndim == 2
    assert feat_yt.ndim == 2


def test_train_main_finetune_smoke(tmp_path, monkeypatch):
    from tsl.train import train_slt as train_mod

    data_root = tmp_path / "tsl51"
    meta_dir = data_root / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)
    lm_dir = data_root / "landmarks" / "user_sentence"
    lm_dir.mkdir(parents=True, exist_ok=True)

    from tests.train.test_train_slt import _write_landmark_csv

    rows = []
    for i in range(4):
        video_id = f"v{i + 1}"
        rows.append(
            {
                "video_id": video_id,
                "sentence_id": i + 1,
                "sentence_clean": f"คำ{i}",
                "landmark_path": f"landmarks/user_sentence/{video_id}.csv",
                "video_path": f"videos/{video_id}.mp4",
            }
        )
        _write_landmark_csv(lm_dir / f"{video_id}.csv", n_frames=3)

    pd.DataFrame(
        rows,
        columns=["video_id", "sentence_id", "sentence_clean", "landmark_path", "video_path"],
    ).to_csv(meta_dir / "sentence_metadata.csv", index=False)

    ckpt_dir = tmp_path / "pretrained"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    cfg = resolve_config("small", input_dim=162)
    config_dict = dict(cfg.__dict__)
    src_model = SignToTextTransformer(vocab_size=8, **config_dict)
    torch.save(src_model.state_dict(), ckpt_dir / "slt_model.pt")
    (ckpt_dir / "model_config.json").write_text(json.dumps({**config_dict, "vocab_size": 8}), encoding="utf-8")

    monkeypatch.setattr(train_mod, "resolve_device", lambda *args, **kwargs: torch.device("cpu"))
    out_dir = tmp_path / "out_ft"
    ret = train_mod.main(
        [
            "--stage",
            "finetune",
            "--pretrained-checkpoint",
            str(ckpt_dir),
            "--data-root",
            str(data_root),
            "--epochs",
            "1",
            "--batch-size",
            "1",
            "--out-dir",
            str(out_dir),
            "--device",
            "auto",
        ]
    )

    assert ret == 0
    assert (out_dir / "slt_model.pt").exists()
