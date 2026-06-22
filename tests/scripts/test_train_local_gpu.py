from pathlib import Path

import torch

from scripts.train_local_gpu import (
    _recover_or_cleanup_temp_checkpoints,
    _seed_out_dir_if_needed,
)


def test_recover_or_cleanup_temp_checkpoints_promotes_valid_tmp(tmp_path: Path):
    payload = {"step": 123, "epoch": 4, "metrics": {"val_chrf": 15.0}}
    tmp_file = tmp_path / "abc123.tmp"
    torch.save(payload, tmp_file)

    actions = _recover_or_cleanup_temp_checkpoints(str(tmp_path))

    recovered = tmp_path / "ckpt_step00000123.pt"
    assert recovered.is_file()
    assert not tmp_file.exists()
    assert actions == [f"recovered temp checkpoint {tmp_file.name} -> {recovered.name}"]


def test_recover_or_cleanup_temp_checkpoints_removes_corrupt_tmp(tmp_path: Path):
    tmp_file = tmp_path / "broken.tmp"
    tmp_file.write_bytes(b"not a checkpoint")

    actions = _recover_or_cleanup_temp_checkpoints(str(tmp_path))

    assert not tmp_file.exists()
    assert actions == [f"removed corrupt temp checkpoint {tmp_file.name}"]


def test_seed_out_dir_keeps_existing_local_checkpoints_without_revalidating(
    tmp_path: Path, monkeypatch
):
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    (out_dir / "ckpt_step00000100.pt").write_bytes(b"placeholder")
    (out_dir / "best_checkpoint.txt").write_text("ckpt_step00000090.pt", encoding="utf-8")
    seed_dir = tmp_path / "seed"
    seed_dir.mkdir()

    def _boom(*args, **kwargs):
        raise AssertionError("torch.load should not run for existing local checkpoints")

    monkeypatch.setattr("scripts.train_local_gpu.torch.load", _boom)

    copied = _seed_out_dir_if_needed(str(out_dir), str(seed_dir))

    assert copied == []
    assert (out_dir / "latest_checkpoint.txt").read_text(encoding="utf-8") == "ckpt_step00000100.pt"
    assert (out_dir / "best_checkpoint.txt").read_text(encoding="utf-8") == "ckpt_step00000090.pt"


def test_seed_out_dir_copies_best_model_state_for_best_state_resume(tmp_path: Path):
    out_dir = tmp_path / "out"
    seed_dir = tmp_path / "seed"
    out_dir.mkdir()
    seed_dir.mkdir()

    best_state = {
        "step": 2900,
        "epoch": 52,
        "metrics": {"val_chrf": 15.48},
        "model_state_dict": {"dummy": torch.tensor([1.0])},
    }
    latest_ckpt = {
        "step": 2000,
        "epoch": 36,
        "metrics": {"val_chrf": 15.0},
    }
    torch.save(best_state, seed_dir / "best_model_state.pt")
    torch.save(latest_ckpt, seed_dir / "ckpt_step00002000.pt")
    (seed_dir / "latest_checkpoint.txt").write_text("ckpt_step00002000.pt", encoding="utf-8")

    copied = _seed_out_dir_if_needed(str(out_dir), str(seed_dir))

    assert "best_model_state.pt" in copied
    assert "ckpt_step00002000.pt" in copied
    restored = torch.load(out_dir / "best_model_state.pt", map_location="cpu", weights_only=False)
    assert restored["step"] == 2900


def test_seed_out_dir_best_state_mode_skips_full_checkpoints(tmp_path: Path):
    out_dir = tmp_path / "out"
    seed_dir = tmp_path / "seed"
    out_dir.mkdir()
    seed_dir.mkdir()

    best_state = {
        "step": 2900,
        "epoch": 52,
        "metrics": {"val_chrf": 15.48},
        "model_state_dict": {"dummy": torch.tensor([1.0])},
    }
    latest_ckpt = {
        "step": 2000,
        "epoch": 36,
        "metrics": {"val_chrf": 15.0},
    }
    torch.save(best_state, seed_dir / "best_model_state.pt")
    torch.save(latest_ckpt, seed_dir / "ckpt_step00002000.pt")
    (seed_dir / "latest_checkpoint.txt").write_text("ckpt_step00002000.pt", encoding="utf-8")

    copied = _seed_out_dir_if_needed(str(out_dir), str(seed_dir), resume_mode="best_state")

    assert copied == ["best_model_state.pt", "latest_checkpoint.txt"]
    assert (out_dir / "best_model_state.pt").is_file()
    assert not (out_dir / "ckpt_step00002000.pt").exists()
