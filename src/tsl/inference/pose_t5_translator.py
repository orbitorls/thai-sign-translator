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

    DEFAULT_MAX_NEW_TOKENS = 72
    DEFAULT_BEAM_SIZE = 5
    DEFAULT_NO_REPEAT_NGRAM_SIZE: int | None = 3
    DEFAULT_REPETITION_PENALTY: float | None = 1.5
    DEFAULT_LENGTH_PENALTY: float | None = 0.7

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
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        beam_size: int = DEFAULT_BEAM_SIZE,
        no_repeat_ngram_size: int = DEFAULT_NO_REPEAT_NGRAM_SIZE,
        repetition_penalty: float = DEFAULT_REPETITION_PENALTY,
        length_penalty: float = DEFAULT_LENGTH_PENALTY,
    ) -> PoseT5Prediction:
        """Translate a single ``(T, 312)`` feature array to Thai text.

        Args:
            features:       ``(T, 312)`` float32 keypoint array.
            max_new_tokens: Maximum tokens to generate.
            beam_size:      Number of beams for beam search.
            no_repeat_ngram_size: Block repeated n-grams during decoding.
            repetition_penalty: Penalize token reuse > 1.0.
            length_penalty: Bias beam search away from over-long outputs when < 1.0.

        Returns:
            A :class:`PoseT5Prediction` with the decoded sentence, raw token
            ids, and a mean softmax-probability confidence in ``[0.0, 1.0]``.
        """
        return self.translate_batch(
            [features],
            max_new_tokens=max_new_tokens,
            beam_size=beam_size,
            no_repeat_ngram_size=no_repeat_ngram_size,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
        )[0]

    @torch.no_grad()
    def translate_batch(
        self,
        features_batch: list[np.ndarray],
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        beam_size: int = DEFAULT_BEAM_SIZE,
        no_repeat_ngram_size: int | None = DEFAULT_NO_REPEAT_NGRAM_SIZE,
        repetition_penalty: float | None = DEFAULT_REPETITION_PENALTY,
        length_penalty: float | None = DEFAULT_LENGTH_PENALTY,
    ) -> list[PoseT5Prediction]:
        """Translate multiple ``(T, 312)`` feature arrays in one generation call."""
        if not features_batch:
            return []

        results: list[PoseT5Prediction | None] = [None] * len(features_batch)
        active_rows: list[int] = []
        active_features: list[np.ndarray] = []
        active_lengths: list[int] = []

        for row_idx, features in enumerate(features_batch):
            if features.ndim != 2:
                raise ValueError(
                    f"features must be 2-D (T, 312); got shape {tuple(features.shape)}"
                )
            T = int(features.shape[0])
            if T == 0:
                results[row_idx] = PoseT5Prediction(sentence="", token_ids=[], score=0.0)
                continue
            active_rows.append(row_idx)
            active_features.append(features)
            active_lengths.append(T)

        if active_features:
            max_T = max(active_lengths)
            feat_dim = int(active_features[0].shape[1])
            src = torch.zeros(
                (len(active_features), max_T, feat_dim),
                dtype=torch.float32,
                device=self.device,
            )
            for batch_idx, features in enumerate(active_features):
                length = active_lengths[batch_idx]
                src[batch_idx, :length] = torch.as_tensor(
                    features,
                    dtype=torch.float32,
                    device=self.device,
                )
            src_lengths = torch.tensor(active_lengths, dtype=torch.long, device=self.device)

            generate_kwargs = {
                "max_new_tokens": max_new_tokens,
                "num_beams": beam_size,
            }
            if no_repeat_ngram_size is not None:
                generate_kwargs["no_repeat_ngram_size"] = no_repeat_ngram_size
            if repetition_penalty is not None:
                generate_kwargs["repetition_penalty"] = repetition_penalty
            if length_penalty is not None:
                generate_kwargs["length_penalty"] = length_penalty
            if (
                no_repeat_ngram_size is not None
                or repetition_penalty is not None
                or length_penalty is not None
            ):
                generate_kwargs["early_stopping"] = True

            generated = self.model.generate(
                src,
                src_lengths,
                **generate_kwargs,
            )
            token_ids_batch = [generated[idx].tolist() for idx in range(generated.shape[0])]
            scores = self._score_sequences(src, src_lengths, token_ids_batch)

            for row_idx, ids, score in zip(active_rows, token_ids_batch, scores):
                sentence = self.tokenizer.decode(ids, skip_special_tokens=True)
                results[row_idx] = PoseT5Prediction(sentence=sentence, token_ids=ids, score=score)

        return [pred for pred in results if pred is not None]

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
        return self._score_sequences(src, src_lengths, [ids])[0]

    def _score_sequences(
        self,
        src: torch.Tensor,
        src_lengths: torch.Tensor,
        ids_batch: list[list[int]],
    ) -> list[float]:
        if not ids_batch:
            return []

        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
        max_len = max((len(ids) for ids in ids_batch), default=0)
        if max_len == 0:
            return [0.0 for _ in ids_batch]

        ids_tensor = torch.full(
            (len(ids_batch), max_len),
            pad_id,
            dtype=torch.long,
            device=self.device,
        )
        for row_idx, ids in enumerate(ids_batch):
            if ids:
                ids_tensor[row_idx, : len(ids)] = torch.tensor(
                    ids,
                    dtype=torch.long,
                    device=self.device,
                )

        output = self.model(src, src_lengths, labels=ids_tensor)
        logits = output.logits

        probs = F.softmax(logits, dim=-1)
        gathered = probs.gather(-1, ids_tensor.unsqueeze(-1)).squeeze(-1)

        mask = (ids_tensor != pad_id).float()
        denom = mask.sum(dim=-1).clamp(min=1.0)
        scores = (gathered * mask).sum(dim=-1) / denom
        return [float(score) for score in scores.tolist()]
