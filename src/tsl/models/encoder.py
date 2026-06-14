"""Temporal landmark encoder.

LandmarkEncoder maps a padded sequence of per-frame landmark feature vectors
(B, T, D) to a fixed-size embedding (B, emb_dim) using a Transformer encoder
with masked mean-pooling over the valid (non-padded) frames.
"""
import math

import torch
import torch.nn as nn


class _SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pe = getattr(self, "pe")[:, : x.size(1), :]
        return x + pe


class LandmarkEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        emb_dim: int = 256,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.d_model = d_model
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_enc = _SinusoidalPositionalEncoding(d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.out_proj = nn.Linear(d_model, emb_dim)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        pad_mask = torch.arange(T, device=x.device)[None, :] >= lengths.to(x.device)[:, None]
        h = self.input_proj(x)
        h = self.pos_enc(h)
        h = self.transformer(h, src_key_padding_mask=pad_mask)
        valid = (~pad_mask).unsqueeze(-1).to(h.dtype)
        summed = (h * valid).sum(dim=1)
        counts = valid.sum(dim=1).clamp(min=1.0)
        pooled = summed / counts
        return self.out_proj(pooled)
