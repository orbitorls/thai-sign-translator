"""Sentence-level sign-to-text inference for the Thai SLT pipeline.

SentenceTranslator loads a :class:`tsl.models.slt.SignToTextTransformer`
from a checkpoint directory produced by :mod:`tsl.train.train_slt` and
runs greedy decoding on a ``(T, D)`` landmark feature array.

Layout expected inside ``checkpoint_dir``:
    - ``slt_model.pt``        : model ``state_dict``
    - ``tokenizer.json``      : serialized :class:`CharTokenizer`
    - ``model_config.json``   : constructor kwargs (incl. ``vocab_size``)

Public surface:
    - :class:`SentencePrediction`: dataclass with ``sentence``, ``token_ids``, ``score``.
    - :class:`SentenceTranslator`: load + :meth:`translate` a single sequence.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass

import numpy as np
import torch

from tsl.models.slt import SignToTextTransformer
from tsl.text.tokenizer import CharTokenizer
from tsl.train.train_slt import load_tokenizer

__all__ = ["SentencePrediction", "SentenceTranslator"]


_MODEL_FILENAME = "slt_model.pt"
_TOKENIZER_FILENAME = "tokenizer.json"
_CONFIG_FILENAME = "model_config.json"


@dataclass
class SentencePrediction:
    """Result of a single greedy decode.

    Attributes:
        sentence: Decoded Thai sentence (special tokens stripped).
        token_ids: The generated token ids (including the leading ``<bos>``
            and the trailing ``<eos>`` when the model emits it).
        score: Mean per-step probability of the chosen tokens, clipped to
            ``[0.0, 1.0]``. ``0.0`` for empty / fully-empty sequences.
    """

    sentence: str
    token_ids: list[int]
    score: float


class SentenceTranslator:
    """Wraps a :class:`SignToTextTransformer` for one-shot inference.

    Loads the model state-dict, tokenizer and config from ``checkpoint_dir``
    on construction. The model is held in eval mode on CPU; the same
    instance can be reused for many calls.
    """

    def __init__(self, checkpoint_dir: str, device: str = "cpu") -> None:
        if not os.path.isdir(checkpoint_dir):
            raise FileNotFoundError(
                f"SLT checkpoint directory not found: {checkpoint_dir!r}"
            )
        model_path = os.path.join(checkpoint_dir, _MODEL_FILENAME)
        tok_path = os.path.join(checkpoint_dir, _TOKENIZER_FILENAME)
        cfg_path = os.path.join(checkpoint_dir, _CONFIG_FILENAME)
        for p in (model_path, tok_path, cfg_path):
            if not os.path.isfile(p):
                raise FileNotFoundError(
                    f"SLT checkpoint file missing: {p!r}"
                )

        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        if "vocab_size" not in cfg:
            raise ValueError(
                f"SLT model config missing 'vocab_size': {cfg_path!r}"
            )

        self.tokenizer: CharTokenizer = load_tokenizer(tok_path)
        self.checkpoint_dir = checkpoint_dir
        self.device = device
        self.model = SignToTextTransformer(**cfg)
        state = torch.load(model_path, map_location="cpu", weights_only=True)
        self.model.load_state_dict(state)
        self.model.eval()
        self.model.to(device)

    def _to_tensor(self, features: np.ndarray) -> torch.Tensor:
        if not isinstance(features, np.ndarray):
            features = np.asarray(features, dtype=np.float32)
        if features.ndim != 2:
            raise ValueError(
                f"features must be 2-D (T, D), got shape {features.shape!r}"
            )
        T = int(features.shape[0])
        src = torch.from_numpy(features.astype(np.float32, copy=False)).unsqueeze(0)
        lengths = torch.tensor([T], dtype=torch.long)
        return src, lengths

    def translate(
        self,
        features: np.ndarray,
        max_len: int = 128,
    ) -> SentencePrediction:
        """Greedy-decode a single ``(T, D)`` feature sequence to a Thai sentence.

        Args:
            features: ``(T, D)`` float32 array of landmark features.
            max_len: Maximum number of decoder steps (inclusive of ``<bos>``).

        Returns:
            A :class:`SentencePrediction` with the decoded sentence, raw
            token ids and a mean-probability score in ``[0, 1]``.

        Raises:
            ValueError: If ``features`` is not a 2-D array or ``max_len < 1``.
        """
        if not isinstance(max_len, int) or max_len < 1:
            raise ValueError(f"max_len must be a positive int, got {max_len!r}")

        src, lengths = self._to_tensor(features)
        T = int(src.size(1))
        if T == 0:
            return SentencePrediction(sentence="", token_ids=[], score=0.0)

        bos_id = self.tokenizer.bos_id
        eos_id = self.tokenizer.eos_id

        with torch.no_grad():
            memory = self.model.encode(src, lengths)
            memory_key_padding_mask = self.model._build_src_pad_mask(src, lengths)
            decoded = torch.full((1, 1), bos_id, dtype=torch.long)
            log_probs: list[float] = []
            finished = torch.zeros(1, dtype=torch.bool)
            for _ in range(max_len - 1):
                T_tgt = decoded.size(1)
                tgt_mask = torch.nn.Transformer.generate_square_subsequent_mask(
                    T_tgt, device=decoded.device, dtype=memory.dtype
                )
                h = self.model.tgt_embed(decoded)
                h = self.model.tgt_pos_enc(h)
                h = self.model.decoder(
                    tgt=h,
                    memory=memory,
                    tgt_mask=tgt_mask,
                    memory_key_padding_mask=memory_key_padding_mask,
                )
                logits = self.model.out_proj(h[:, -1, :])
                probs = torch.softmax(logits, dim=-1)
                next_token = int(probs.argmax(dim=-1).item())
                log_probs.append(float(probs[0, next_token].log().item()))
                next_t = torch.tensor([[next_token]], dtype=torch.long)
                next_t = torch.where(
                    finished, torch.full_like(next_t, eos_id), next_t
                )
                decoded = torch.cat([decoded, next_t], dim=1)
                finished = finished | (next_token == eos_id)
                if finished.item():
                    break

        token_ids = [int(i) for i in decoded.squeeze(0).tolist()]
        sentence = self.tokenizer.decode(token_ids, strip_special=True)
        if log_probs:
            mean_log_prob = sum(log_probs) / len(log_probs)
            score = float(math.exp(mean_log_prob))
        else:
            score = 0.0
        score = max(0.0, min(1.0, score))
        return SentencePrediction(
            sentence=sentence, token_ids=token_ids, score=score
        )
