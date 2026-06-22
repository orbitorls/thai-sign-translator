from __future__ import annotations

import json

import torch

from scripts.export_pose_t5_checkpoint import main


class _FakePoseToTextT5:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.loaded_state = None

    def load_state_dict(self, state):
        self.loaded_state = state

    def eval(self):
        return self

    def save_pretrained(self, output_dir: str) -> None:
        with open(f"{output_dir}/pose_t5_config.json", "w", encoding="utf-8") as fh:
            json.dump(self.kwargs, fh)
        torch.save(self.loaded_state, f"{output_dir}/pose_encoder.pt")


class _FakeTokenizer:
    def save_pretrained(self, output_dir: str) -> None:
        with open(f"{output_dir}/tokenizer_config.json", "w", encoding="utf-8") as fh:
            json.dump({"saved": True}, fh)


def test_export_best_checkpoint_from_reference_file(tmp_path, monkeypatch):
    train_dir = tmp_path / "train"
    export_dir = tmp_path / "export"
    train_dir.mkdir()

    ckpt = train_dir / "ckpt_step00001800.pt"
    torch.save(
        {
            "step": 1800,
            "epoch": 32,
            "metrics": {"val_loss": 3.02, "val_chrf": 14.72},
            "model_state_dict": {"weight": torch.tensor([1.0])},
        },
        ckpt,
    )
    (train_dir / "best_checkpoint.txt").write_text(ckpt.name, encoding="utf-8")

    monkeypatch.setattr("scripts.export_pose_t5_checkpoint.PoseToTextT5", _FakePoseToTextT5)
    monkeypatch.setattr(
        "scripts.export_pose_t5_checkpoint.AutoTokenizer.from_pretrained",
        lambda _name: _FakeTokenizer(),
    )

    code = main(
        [
            "--train-dir",
            str(train_dir),
            "--export-dir",
            str(export_dir),
            "--base-model",
            "google/mt5-small",
        ]
    )

    assert code == 0
    metadata = json.loads((export_dir / "runtime_metadata.json").read_text(encoding="utf-8"))
    assert metadata["checkpoint_name"] == "ckpt_step00001800.pt"
    assert metadata["checkpoint_step"] == 1800
    assert json.loads((export_dir / "pose_t5_config.json").read_text(encoding="utf-8"))["base_model_name"] == "google/mt5-small"
    assert (export_dir / "tokenizer_config.json").is_file()


def test_export_best_checkpoint_falls_back_to_scan(tmp_path, monkeypatch):
    train_dir = tmp_path / "train"
    export_dir = tmp_path / "export"
    train_dir.mkdir()

    low = train_dir / "ckpt_step00001700.pt"
    high = train_dir / "ckpt_step00002000.pt"
    torch.save({"metrics": {"val_chrf": 12.7}, "model_state_dict": {"w": 1}, "step": 1700, "epoch": 31}, low)
    torch.save({"metrics": {"val_chrf": 14.2}, "model_state_dict": {"w": 2}, "step": 2000, "epoch": 35}, high)

    monkeypatch.setattr("scripts.export_pose_t5_checkpoint.PoseToTextT5", _FakePoseToTextT5)
    monkeypatch.setattr(
        "scripts.export_pose_t5_checkpoint.AutoTokenizer.from_pretrained",
        lambda _name: _FakeTokenizer(),
    )

    code = main(["--train-dir", str(train_dir), "--export-dir", str(export_dir)])

    assert code == 0
    metadata = json.loads((export_dir / "runtime_metadata.json").read_text(encoding="utf-8"))
    assert metadata["checkpoint_name"] == "ckpt_step00002000.pt"
    assert metadata["checkpoint_metrics"]["val_chrf"] == 14.2


def test_export_best_prefers_best_model_state(tmp_path, monkeypatch):
    train_dir = tmp_path / "train"
    export_dir = tmp_path / "export"
    train_dir.mkdir()

    torch.save(
        {
            "step": 2200,
            "epoch": 39,
            "metrics": {"val_loss": 3.08, "val_chrf": 14.89},
            "model_state_dict": {"w": 3},
        },
        train_dir / "best_model_state.pt",
    )
    (train_dir / "best_checkpoint.txt").write_text("ckpt_step00001800.pt", encoding="utf-8")

    monkeypatch.setattr("scripts.export_pose_t5_checkpoint.PoseToTextT5", _FakePoseToTextT5)
    monkeypatch.setattr(
        "scripts.export_pose_t5_checkpoint.AutoTokenizer.from_pretrained",
        lambda _name: _FakeTokenizer(),
    )

    code = main(["--train-dir", str(train_dir), "--export-dir", str(export_dir)])

    assert code == 0
    metadata = json.loads((export_dir / "runtime_metadata.json").read_text(encoding="utf-8"))
    assert metadata["checkpoint_name"] == "best_model_state.pt"
    assert metadata["checkpoint_type"] == "best_model_state"
    assert metadata["checkpoint_step"] == 2200
