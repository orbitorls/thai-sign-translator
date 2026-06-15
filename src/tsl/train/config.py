from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ModelSize = Literal["small", "base", "large"]


@dataclass(frozen=True)
class ModelConfig:
    input_dim: int
    d_model: int
    nhead: int
    num_encoder_layers: int
    num_decoder_layers: int
    dim_feedforward: int
    dropout: float
    max_pos_len: int


_PRESETS: dict[ModelSize, dict] = {
    "small": {
        "d_model": 64,
        "nhead": 4,
        "num_encoder_layers": 2,
        "num_decoder_layers": 2,
        "dim_feedforward": 128,
        "dropout": 0.1,
        "max_pos_len": 1024,
    },
    "base": {
        "d_model": 256,
        "nhead": 8,
        "num_encoder_layers": 4,
        "num_decoder_layers": 4,
        "dim_feedforward": 1024,
        "dropout": 0.1,
        "max_pos_len": 2048,
    },
    "large": {
        "d_model": 512,
        "nhead": 8,
        "num_encoder_layers": 6,
        "num_decoder_layers": 6,
        "dim_feedforward": 2048,
        "dropout": 0.1,
        "max_pos_len": 2048,
    },
}


def resolve_config(size: ModelSize, input_dim: int) -> ModelConfig:
    params = _PRESETS[size].copy()
    params["input_dim"] = input_dim
    return ModelConfig(**params)


__all__ = ["ModelSize", "ModelConfig", "resolve_config"]
