"""PoseT5Translator: inference wrapper for the PoseToTextT5 model.

Loads a :class:`tsl.models.pose_t5.PoseToTextT5` checkpoint produced by
:mod:`tsl.train.train_pose_t5` and decodes a ``(T, 312)`` keypoint feature
array to a Thai sentence using T5 beam search.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from tsl.models.pose_t5 import PoseToTextT5


__all__ = ["PoseT5Prediction", "PoseT5Translator"]


@dataclass(frozen=True)
class PoseT5Prediction:
    """Result of a single :meth:`PoseT5Translator.translate` call."""

    sentence: str
    token_ids: list[int]
    score: float


class PoseT5Translator:
    """Loads a PoseToTextT5 checkpoint and decodes a feature sequence to Thai text.

    The constructor puts the model into ``eval`` mode and moves it to the
    requested device; it never re-enters training mode for the lifetime of
    the instance.
    """

    def __init__(self, model: PoseToTextT5, tokenizer, device: str = "cpu") -> None:
        self.model = model.to(device)
        self.model.eval()
        self.tokenizer = tokenizer
        self.device = device

    @classmethod
    def from_checkpoint_dir(cls, checkpoint_dir: str, device: str = "cpu") -> "PoseT5Translator":
        """Load model and tokenizer from a checkpoint directory.

        Reads ``pose_t5_config.json``, loads T5 weights from the directory,
        and loads the tokenizer via ``AutoTokenizer``.

        The tokenizer is loaded from the checkpoint directory (if tokenizer
        files are present there) or falls back to ``base_model_name`` from the
        config (requires internet or HF cache).

        Args:
            checkpoint_dir: Directory saved by :meth:`PoseToTextT5.save_pretrained`.
            device: PyTorch device string.

        Returns:
            A :class:`PoseT5Translator` ready for inference.
        """
        from transformers import AutoTokenizer

        model = PoseToTextT5.from_pretrained(checkpoint_dir, device=device)

        # Try to load tokenizer from checkpoint dir; fall back to base_model_name
        config_path = os.path.join(checkpoint_dir, "pose_t5_config.json")
        with open(config_path, "r", encoding="utf-8") as fh:
            config = json.load(fh)

        tokenizer_source = checkpoint_dir
        # AutoTokenizer needs at least tokenizer_config.json or tokenizer.json
        tokenizer_config_path = os.path.join(checkpoint_dir, "tokenizer_config.json")
        if not os.path.isfile(tokenizer_config_path):
            tokenizer_source = config.get("base_model_name", "google/mt5-small")

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
        return cls(model=model, tokenizer=tokenizer, device=device)

    @torch.no_grad()
    def translate(
        self,
        features: np.ndarray,
        max_new_tokens: int = 128,
        beam_size: int = 4,
    ) -> PoseT5Prediction:
        """Translate a single ``(T, 312)`` feature array to Thai text.

        Args:
            features:       ``(T, 312)`` float32 keypoint array.
            max_new_tokens: Maximum tokens to generate.
            beam_size:      Number of beams for beam search.

        Returns:
            A :class:`PoseT5Prediction` with the decoded sentence, raw token
            ids, and a mean softmax-probability confidence in ``[0.0, 1.0]``.
        """
        if features.ndim != 2:
            raise ValueError(
                f"features must be 2-D (T, 312); got shape {tuple(features.shape)}"
            )
        T = features.shape[0]
        if T == 0:
            return PoseT5Prediction(sentence="", token_ids=[], score=0.0)

        src = torch.as_tensor(features, dtype=torch.float32, device=self.device).unsqueeze(0)
        src_lengths = torch.tensor([T], dtype=torch.long, device=self.device)

        # Generate with beam search
        generated = self.model.generate(
            src,
            src_lengths,
            max_new_tokens=max_new_tokens,
            num_beams=beam_size,
        )
        ids: list[int] = generated[0].tolist()

        sentence = self.tokenizer.decode(ids, skip_special_tokens=True)
        score = self._score_sequence(src, src_lengths, ids)
        return PoseT5Prediction(sentence=sentence, token_ids=ids, score=score)

    def _score_sequence(
        self,
        src: torch.Tensor,
        src_lengths: torch.Tensor,
        ids: list[int],
    ) -> float:
        """Mean softmax probability of the generated tokens.

        Runs a single forward pass with ``labels=ids_tensor`` and reads off the
        probability that the model assigns to each generated token.  The T5
        forward pass with labels handles the right-shift internally, so
        ``logits[0, i]`` already predicts ``ids[i]``.

        Positions with the pad token id are excluded from the average.
        """
        if len(ids) == 0:
            return 0.0

        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
        ids_tensor = torch.tensor([ids], dtype=torch.long, device=self.device)

        # Forward pass: T5 internally shifts labels → logits[b,i] predicts labels[b,i]
        output = self.model(src, src_lengths, labels=ids_tensor)
        logits = output.logits  # (1, T_out, vocab_size)

        probs = F.softmax(logits, dim=-1)
        gathered = probs.gather(-1, ids_tensor.unsqueeze(-1)).squeeze(-1)  # (1, T_out)

        # Mask out pad positions
        mask = (ids_tensor != pad_id).float()
        denom = float(mask.sum().item())
        if denom <= 0.0:
            return 0.0
        return float((gathered * mask).sum().item() / denom)
