"""Collate function for the Thai Sign Language Translation (SLT) pipeline.

Turns a list of :class:`SignTextExample` (one sentence per item) plus a
fitted :class:`CharTokenizer` into a single :class:`SltBatch` of padded
torch tensors ready to be fed to :class:`SignToTextTransformer`.

Responsibilities:
    - load each example's landmark sequence via a pluggable loader
      (default: :func:`tsl.data.tsl51.load_landmark_sequence`),
    - encode each example's ``target_text`` into char ids with the
      tokenizer and pad to a uniform ``(B, T_tgt_max)`` matrix with
      ``<bos>`` / ``<eos>`` framing,
    - stack the landmark arrays into a zero-padded ``(B, T_src_max, D)``
      float32 tensor,
    - return ``src_lengths`` (in frames) and ``tgt_lengths`` (in real
      characters, specials excluded) so the model can build the
      attention / loss masks it needs.

The empty ``(T=0)`` landmark case is handled gracefully by promoting
the empty example to a single zero frame so the batch shape stays
consistent.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch

from tsl.data.manifest import SignTextExample
from tsl.data.tsl51 import load_landmark_sequence
from tsl.text.tokenizer import CharTokenizer

__all__ = ["SltBatch", "slt_collate"]


@dataclass
class SltBatch:
    """A padded batch of sentence-level sign-translation examples.

    Attributes:
        src: ``(B, T_src_max, D)`` float32 tensor of landmark features,
            right-padded with zeros.
        src_lengths: ``(B,)`` long tensor of real frame counts per item.
        tgt: ``(B, T_tgt_max)`` long tensor of token ids, right-padded
            with :attr:`CharTokenizer.pad_id`. ``<bos>`` is prepended
            and ``<eos>`` is appended to every row (when the collate
            was called with ``add_bos=True`` / ``add_eos=True``).
        tgt_lengths: ``(B,)`` long tensor of real-character counts per
            item. Does **not** include the bos/eos specials; this is
            the length of the original encoded ``target_text``.
        target_texts: The original ``target_text`` for every row,
            kept around for debug printing and offline evaluation.
    """

    src: torch.Tensor
    src_lengths: torch.Tensor
    tgt: torch.Tensor
    tgt_lengths: torch.Tensor
    target_texts: list[str]


def slt_collate(
    examples: list[SignTextExample],
    tokenizer: CharTokenizer,
    load_features: Callable[[str], np.ndarray] | None = None,
    add_bos: bool = True,
    add_eos: bool = True,
) -> SltBatch:
    """Build a padded training batch from sentence-level examples.

    Args:
        examples: One :class:`SignTextExample` per row. Must be
            non-empty.
        tokenizer: A fitted :class:`CharTokenizer` used to encode
            ``example.target_text`` into token ids.
        load_features: Callable that takes a ``features_path`` and
            returns a ``(T, D)`` numpy array. Defaults to
            :func:`tsl.data.tsl51.load_landmark_sequence`, which
            requires a path shaped like the TSL-51 landmark CSV.
        add_bos: If True (default), prepend ``<bos>`` to every target
            row during padding.
        add_eos: If True (default), append ``<eos>`` to every target
            row during padding.

    Returns:
        A populated :class:`SltBatch`.

    Raises:
        ValueError: If ``examples`` is empty.
    """
    if not examples:
        raise ValueError("empty batch")

    if load_features is None:
        load_features = load_landmark_sequence

    raw_seqs: list[np.ndarray] = []
    raw_lengths: list[int] = []
    for ex in examples:
        arr = load_features(ex.features_path)
        arr = np.asarray(arr)
        if arr.ndim != 2:
            raise ValueError(
                f"features for {ex.example_id!r} must be 2-D (T, D), "
                f"got shape {arr.shape!r}"
            )
        if arr.shape[0] == 0:
            # Promote an empty (T=0) sequence to a single zero frame so
            # the batch keeps a consistent shape. src_length stays 0.
            d = arr.shape[1] if arr.ndim == 2 and arr.shape[1] > 0 else 0
            raw_seqs.append(arr)
            raw_lengths.append(0)
            if d == 0:
                # We have no idea what the feature dim is yet; defer to
                # the rest of the batch. The D inference below will
                # raise a clear error if the batch is fully empty.
                continue
        else:
            raw_seqs.append(arr)
            raw_lengths.append(int(arr.shape[0]))

    target_texts = [ex.target_text for ex in examples]
    raw_ids, real_char_lengths = tokenizer.encode_batch(target_texts)
    padded_tgt, _ = tokenizer.pad_to_arrays(
        raw_ids, add_bos=add_bos, add_eos=add_eos
    )

    # ---- source (landmark) tensor -----------------------------------
    inferred_d: int | None = None
    for arr in raw_seqs:
        if arr.shape[0] > 0:
            inferred_d = arr.shape[1]
            break
    if inferred_d is None:
        raise ValueError(
            "cannot infer feature dim: every example in the batch has "
            "an empty (T=0) feature array"
        )

    # Empty rows get one synthetic zero frame so the stack still works.
    # src_lengths records the real frame count (0 for empty rows).
    padded_seqs: list[np.ndarray] = []
    for arr in raw_seqs:
        if arr.shape[0] == 0:
            padded_seqs.append(np.zeros((1, inferred_d), dtype=np.float32))
        else:
            padded_seqs.append(arr.astype(np.float32, copy=False))

    t_src_max = max(s.shape[0] for s in padded_seqs)
    src_np = np.zeros((len(padded_seqs), t_src_max, inferred_d), dtype=np.float32)
    for i, s in enumerate(padded_seqs):
        src_np[i, : s.shape[0]] = s

    src = torch.from_numpy(src_np)
    src_lengths = torch.tensor(raw_lengths, dtype=torch.long)
    tgt = torch.from_numpy(padded_tgt.astype(np.int64, copy=False)).long()
    tgt_lengths = torch.tensor(list(real_char_lengths), dtype=torch.long)

    return SltBatch(
        src=src,
        src_lengths=src_lengths,
        tgt=tgt,
        tgt_lengths=tgt_lengths,
        target_texts=target_texts,
    )
