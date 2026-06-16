"""PoseToTextT5: pose-encoder front-end + mT5 decoder for Thai SLT.

Architecture:
  - Temporal downsampling (mean-pool every `downsample_factor` frames)
  - Linear projection to T5's d_model
  - Sinusoidal positional encoding
  - nn.TransformerEncoder (pose front-end)
  - MT5ForConditionalGeneration (decoder side only; encoder bypassed via inputs_embeds)
"""
from __future__ import annotations

import json
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import MT5ForConditionalGeneration, MT5Config
from transformers.modeling_outputs import Seq2SeqLMOutput

from tsl.models.encoder import _SinusoidalPositionalEncoding


__all__ = ["PoseToTextT5"]


class PoseToTextT5(nn.Module):
    """Pose-to-text model that combines a temporal pose encoder with mT5.

    The T5 *encoder* is bypassed entirely — we produce ``inputs_embeds`` from
    keypoints and pass them directly to T5, letting T5's cross-attention decoder
    generate Thai text tokens.
    """

    def __init__(
        self,
        input_dim: int = 312,
        num_encoder_layers: int = 2,
        encoder_dropout: float = 0.1,
        downsample_factor: int = 4,
        base_model_name: str = "google/mt5-small",
        local_model_path: str | None = None,
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.num_encoder_layers = num_encoder_layers
        self.encoder_dropout = encoder_dropout
        self.downsample_factor = downsample_factor
        self.base_model_name = base_model_name

        # Load T5 (from disk for Kaggle offline, or download)
        load_path = local_model_path if local_model_path is not None else base_model_name
        self.t5_model: MT5ForConditionalGeneration = MT5ForConditionalGeneration.from_pretrained(load_path)

        # Derive proj_dim from T5 config so it works with any T5 variant
        proj_dim: int = self.t5_model.config.d_model
        self._proj_dim = proj_dim

        # Number of attention heads: use T5 config if available, else derive
        num_heads_t5 = getattr(self.t5_model.config, "num_heads", None)
        if num_heads_t5 is not None and proj_dim % num_heads_t5 == 0:
            nhead = num_heads_t5
        else:
            # Derive: largest power-of-2 divisor of proj_dim that is <= proj_dim // 64
            nhead = max(1, proj_dim // 64)

        self.input_proj = nn.Linear(input_dim, proj_dim)
        self.pos_enc = _SinusoidalPositionalEncoding(proj_dim, max_len=2048)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=proj_dim,
            nhead=nhead,
            dim_feedforward=proj_dim * 4,
            dropout=encoder_dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(enc_layer, num_layers=num_encoder_layers)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _downsample(
        self, src: torch.Tensor, src_lengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Mean-pool every ``downsample_factor`` frames.

        Returns:
            x_ds:       (B, T_ds, D) downsampled keypoints
            ds_lengths: (B,) downsampled frame counts (ceiling division)
        """
        B, T, D = src.shape
        F = self.downsample_factor

        # Pad T to a multiple of F
        pad_len = (F - T % F) % F
        if pad_len > 0:
            src = torch.nn.functional.pad(src, (0, 0, 0, pad_len))

        T_padded = src.size(1)
        x_ds = src.reshape(B, T_padded // F, F, D).mean(dim=2)

        # Ceiling division for valid ds lengths
        ds_lengths = (src_lengths.to(src.device) + F - 1) // F

        return x_ds, ds_lengths

    def _build_pose_embeds(
        self, src: torch.Tensor, src_lengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the pose front-end.

        Returns:
            memory:    (B, T_ds, proj_dim) pose memory for T5
            attn_mask: (B, T_ds) long tensor of 1/0 for T5 attention_mask
        """
        x_ds, ds_lengths = self._downsample(src, src_lengths)
        T_ds = x_ds.size(1)

        # Build padding mask for TransformerEncoder (True = ignore)
        pad_mask = (
            torch.arange(T_ds, device=src.device)[None, :] >= ds_lengths[:, None]
        )

        h = self.input_proj(x_ds)
        h = self.pos_enc(h)
        memory = self.transformer_encoder(h, src_key_padding_mask=pad_mask)

        # T5 attention_mask: 1 for real tokens, 0 for padding
        attn_mask = (~pad_mask).long()

        return memory, attn_mask

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def forward(
        self,
        src: torch.Tensor,
        src_lengths: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> Seq2SeqLMOutput:
        """Run a full forward pass.

        Args:
            src:        (B, T, 312) float32 keypoints
            src_lengths:(B,) long, real frame counts
            labels:     (B, T_tgt) long with -100 for padding positions (or None)

        Returns:
            HF ``Seq2SeqLMOutput`` with ``.loss`` (if labels given) and ``.logits``
        """
        memory, attn_mask = self._build_pose_embeds(src, src_lengths)

        return self.t5_model(
            inputs_embeds=memory,
            attention_mask=attn_mask,
            labels=labels,
        )

    @torch.no_grad()
    def generate(
        self,
        src: torch.Tensor,
        src_lengths: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Encode pose then generate Thai text with T5 beam search.

        Args:
            src:        (B, T, 312) float32 keypoints
            src_lengths:(B,) long, real frame counts
            **kwargs:   forwarded to ``MT5ForConditionalGeneration.generate``

        Returns:
            (B, T_out) long tensor of token ids
        """
        memory, attn_mask = self._build_pose_embeds(src, src_lengths)
        return self.t5_model.generate(
            inputs_embeds=memory,
            attention_mask=attn_mask,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def pose_encoder_state(self) -> dict:
        """State dict of the pose front-end (input_proj + pos_enc + transformer_encoder)."""
        return {
            "input_proj": self.input_proj.state_dict(),
            "pos_enc": self.pos_enc.state_dict(),
            "transformer_encoder": self.transformer_encoder.state_dict(),
        }

    def save_pretrained(self, checkpoint_dir: str) -> None:
        """Save the full model to ``checkpoint_dir``.

        Saves:
          - T5 weights via ``t5_model.save_pretrained``
          - Pose encoder state dict at ``pose_encoder.pt``
          - Config at ``pose_t5_config.json``
        """
        os.makedirs(checkpoint_dir, exist_ok=True)

        self.t5_model.save_pretrained(checkpoint_dir)

        torch.save(
            self.pose_encoder_state(),
            os.path.join(checkpoint_dir, "pose_encoder.pt"),
        )

        config = {
            "input_dim": self.input_dim,
            "num_encoder_layers": self.num_encoder_layers,
            "encoder_dropout": self.encoder_dropout,
            "downsample_factor": self.downsample_factor,
            "base_model_name": self.base_model_name,
        }
        with open(os.path.join(checkpoint_dir, "pose_t5_config.json"), "w", encoding="utf-8") as fh:
            json.dump(config, fh, indent=2)

    @classmethod
    def from_pretrained(cls, checkpoint_dir: str, device: str = "cpu") -> "PoseToTextT5":
        """Load a saved model from ``checkpoint_dir``.

        Reads ``pose_t5_config.json``, loads T5 from disk, loads pose encoder
        weights, moves to ``device``, and sets eval mode.
        """
        config_path = os.path.join(checkpoint_dir, "pose_t5_config.json")
        with open(config_path, "r", encoding="utf-8") as fh:
            config = json.load(fh)

        model = cls(
            input_dim=config["input_dim"],
            num_encoder_layers=config["num_encoder_layers"],
            encoder_dropout=config["encoder_dropout"],
            downsample_factor=config["downsample_factor"],
            base_model_name=config["base_model_name"],
            local_model_path=checkpoint_dir,
        )

        pose_enc_path = os.path.join(checkpoint_dir, "pose_encoder.pt")
        state = torch.load(pose_enc_path, map_location=device, weights_only=True)
        model.input_proj.load_state_dict(state["input_proj"])
        model.pos_enc.load_state_dict(state["pos_enc"])
        model.transformer_encoder.load_state_dict(state["transformer_encoder"])

        model.to(device)
        model.eval()
        return model
