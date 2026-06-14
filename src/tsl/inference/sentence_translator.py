"""Sentence-level sign-to-text inference for the Thai SLT pipeline.

SentenceTranslator loads a :class:`tsl.models.slt.SignToTextTransformer`
from a checkpoint directory produced by :mod:`tsl.train.train_slt` and
runs greedy decoding on a ``(T, D)`` landmark feature array, returning
the decoded Thai sentence and a token-level confidence score.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from tsl.models.slt import SignToTextTransformer
from tsl.text.tokenizer import CharTokenizer
from tsl.train.train_slt import load_tokenizer


__all__ = ["SentencePrediction", "SentenceTranslator"]


@dataclass
class SentencePrediction:
    """Result of a single :meth:`SentenceTranslator.translate` call."""

    sentence: str
    token_ids: list[int]
    score: float


class SentenceTranslator:
    """Loads an SLT checkpoint and decodes a landmark sequence to a Thai sentence.

    The constructor switches the model into ``eval`` mode and moves it to
    the requested device; it never re-enters training mode for the
    lifetime of the instance.
    """

    _CONFIG_FILENAME = "model_config.json"
    _STATE_FILENAME = "slt_model.pt"
    _TOKENIZER_FILENAME = "tokenizer.json"

    def __init__(self, checkpoint_dir: str, device: str = "cpu") -> None:
        config_path = os.path.join(checkpoint_dir, self._CONFIG_FILENAME)
        state_path = os.path.join(checkpoint_dir, self._STATE_FILENAME)
        tokenizer_path = os.path.join(checkpoint_dir, self._TOKENIZER_FILENAME)
        for path, name in (
            (config_path, self._CONFIG_FILENAME),
            (state_path, self._STATE_FILENAME),
            (tokenizer_path, self._TOKENIZER_FILENAME),
        ):
            if not os.path.isfile(path):
                raise FileNotFoundError(
                    f"{name} not found in checkpoint_dir={checkpoint_dir!r} "
                    f"(expected at {path!r})"
                )

        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        self.model = SignToTextTransformer(**config)
        state = torch.load(state_path, map_location=device, weights_only=True)
        self.model.load_state_dict(state)
        self.model.eval()
        self.model.to(device)

        self.tokenizer = load_tokenizer(tokenizer_path)
        self.device = device

    @classmethod
    def from_model_and_tokenizer(
        cls,
        model: SignToTextTransformer,
        tokenizer: CharTokenizer,
        device: str = "cpu",
    ) -> "SentenceTranslator":
        """Build a translator from in-memory objects (skips file I/O)."""
        instance = cls.__new__(cls)
        instance.model = model.to(device)
        instance.model.eval()
        instance.tokenizer = tokenizer
        instance.device = device
        return instance

    @torch.no_grad()
    def translate(self, features: np.ndarray, max_len: int = 128) -> SentencePrediction:
        """Decode ``features`` (``(T, D)`` float32) to a Thai sentence.

        Returns a :class:`SentencePrediction` with the decoded text, the
        raw token ids (including the leading ``<bos>`` and any trailing
        ``<eos>``), and a mean softmax-probability confidence in
        ``[0.0, 1.0]`` (or ``0.0`` if there is nothing to score).
        """
        if features.ndim != 2:
            raise ValueError(
                f"features must be 2-D (T, D); got shape {tuple(features.shape)}"
            )
        T = features.shape[0]
        if T == 0:
            return SentencePrediction(sentence="", token_ids=[], score=0.0)

        src = torch.as_tensor(
            features, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        src_lengths = torch.tensor([T], dtype=torch.long, device=self.device)

        bos_id = self.tokenizer.bos_id
        eos_id = self.tokenizer.eos_id
        pad_id = self.tokenizer.pad_id

        decoded = self.model.greedy_decode(
            src, src_lengths, bos_id, eos_id, max_len
        )
        ids = decoded[0].tolist()
        sentence = self.tokenizer.decode(ids, strip_special=True)
        score = self._score_sequence(src, src_lengths, ids, pad_id)
        return SentencePrediction(sentence=sentence, token_ids=ids, score=score)

    def _score_sequence(
        self,
        src: torch.Tensor,
        src_lengths: torch.Tensor,
        ids: list[int],
        pad_id: int,
    ) -> float:
        """Mean softmax probability of the greedy tokens (excluding ``<pad>``).

        Runs a single teacher-forcing forward pass with
        ``tgt = ids[:-1]`` and reads off the probability of the
        actually-decoded next token at each position. Positions whose
        target is ``pad_id`` are masked out of the average.
        """
        if len(ids) <= 1:
            return 0.0
        ids_tensor = torch.tensor([ids], dtype=torch.long, device=self.device)
        tgt = ids_tensor[:, :-1]
        logits = self.model(src, src_lengths, tgt)
        probs = F.softmax(logits, dim=-1)
        next_ids = ids_tensor[:, 1:]
        gathered = probs.gather(-1, next_ids.unsqueeze(-1)).squeeze(-1)
        mask = (next_ids != pad_id).float()
        denom = float(mask.sum().item())
        if denom <= 0.0:
            return 0.0
        return float((gathered * mask).sum().item() / denom)
