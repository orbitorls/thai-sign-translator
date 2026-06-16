"""Collate function for T5-based Thai Sign Language Translation.

Turns a list of :class:`SignTextExample` into a single :class:`PoseT5Batch`
of padded torch tensors compatible with HuggingFace T5/mT5 models.

Responsibilities:
    - load each example's feature array via a pluggable loader,
    - cap source to ``max_src_len`` frames by truncating,
    - zero-pad source frames to ``(B, T_max, 312)`` float32,
    - build ``src_lengths`` and ``src_mask`` (bool attention mask),
    - tokenize ``target_text`` with an HF tokenizer and replace pad
      token ids with ``-100`` so cross-entropy ignores them.

Empty (T=0) sequences are handled gracefully: promoted to 1 zero
frame while ``src_length`` remains 0.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch

from tsl.data.manifest import SignTextExample

__all__ = ["PoseT5Batch", "pose_t5_collate"]

_FEAT_DIM = 312


@dataclass
class PoseT5Batch:
    """A padded T5-compatible batch for sign-language translation.

    Attributes:
        src: ``(B, T_src, 312)`` float32 tensor of pose features,
            right-padded with zeros.
        src_lengths: ``(B,)`` long tensor of real frame counts.
        src_mask: ``(B, T_src)`` bool tensor — ``True`` where real,
            ``False`` where padded. Suitable for passing as
            ``attention_mask`` on the encoder side.
        labels: ``(B, T_tgt)`` long tensor of target token ids.
            Padding positions are replaced with ``-100`` so HuggingFace
            cross-entropy loss ignores them.
    """

    src: torch.Tensor
    src_lengths: torch.Tensor
    src_mask: torch.Tensor
    labels: torch.Tensor


def pose_t5_collate(
    examples: list[SignTextExample],
    hf_tokenizer,
    load_features: Callable[[str], np.ndarray],
    max_src_len: int = 512,
) -> PoseT5Batch:
    """Build a padded T5 batch from sentence-level sign examples.

    Args:
        examples: One :class:`SignTextExample` per row. Must be non-empty.
        hf_tokenizer: A HuggingFace tokenizer (e.g. ``MT5Tokenizer``).
            Must expose ``pad_token_id`` and accept the call signature
            ``tokenizer(texts, padding=True, return_tensors="pt")``.
        load_features: Callable that maps a ``features_path`` string to a
            ``(T, 312)`` numpy array.
        max_src_len: Maximum number of source frames to keep (frames
            beyond this are silently truncated). Default: 512.

    Returns:
        A populated :class:`PoseT5Batch`.

    Raises:
        ValueError: If ``examples`` is empty.
    """
    if not examples:
        raise ValueError("empty batch")

    # ------------------------------------------------------------------ #
    # 1. Load and optionally truncate source feature arrays               #
    # ------------------------------------------------------------------ #
    raw_seqs: list[np.ndarray] = []
    raw_lengths: list[int] = []

    for ex in examples:
        arr = np.asarray(load_features(ex.features_path), dtype=np.float32)
        if arr.ndim != 2:
            raise ValueError(
                f"features for {ex.example_id!r} must be 2-D (T, D), "
                f"got shape {arr.shape!r}"
            )
        # Truncate if over the cap
        if arr.shape[0] > max_src_len:
            arr = arr[:max_src_len]
        raw_seqs.append(arr)
        raw_lengths.append(int(arr.shape[0]))

    # ------------------------------------------------------------------ #
    # 2. Handle empty sequences and pad to batch max length               #
    # ------------------------------------------------------------------ #
    # Promote empty arrays to 1 zero frame so tensors stay 3-D.
    padded_seqs: list[np.ndarray] = []
    for arr in raw_seqs:
        if arr.shape[0] == 0:
            padded_seqs.append(np.zeros((1, _FEAT_DIM), dtype=np.float32))
        else:
            padded_seqs.append(arr)

    t_max = max(s.shape[0] for s in padded_seqs)
    b = len(padded_seqs)

    src_np = np.zeros((b, t_max, _FEAT_DIM), dtype=np.float32)
    for i, s in enumerate(padded_seqs):
        src_np[i, : s.shape[0]] = s

    src = torch.from_numpy(src_np)  # (B, T_max, 312) float32

    # ------------------------------------------------------------------ #
    # 3. src_lengths and src_mask                                         #
    # ------------------------------------------------------------------ #
    src_lengths = torch.tensor(raw_lengths, dtype=torch.long)  # (B,)

    # src_mask: True where the frame is real, False where padded.
    # For an empty sequence (length 0) every position in the promoted
    # 1-frame tensor is considered padding, so the mask is all False.
    arange = torch.arange(t_max, dtype=torch.long)  # (T_max,)
    src_mask = arange.unsqueeze(0) < src_lengths.unsqueeze(1)  # (B, T_max) bool

    # ------------------------------------------------------------------ #
    # 4. Tokenise target texts                                            #
    # ------------------------------------------------------------------ #
    texts = [ex.target_text for ex in examples]
    encoding = hf_tokenizer(texts, padding=True, return_tensors="pt")
    labels: torch.Tensor = encoding["input_ids"].long()  # (B, T_tgt)

    # ------------------------------------------------------------------ #
    # 5. Replace padding token id with -100                               #
    # ------------------------------------------------------------------ #
    pad_id: int = hf_tokenizer.pad_token_id
    labels = labels.masked_fill(labels == pad_id, -100)

    return PoseT5Batch(
        src=src,
        src_lengths=src_lengths,
        src_mask=src_mask,
        labels=labels,
    )
