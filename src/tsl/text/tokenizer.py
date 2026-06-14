"""Char-level Thai tokenizer for the SLT pipeline.

We start with a char-level tokenizer because the TSL-51 dataset is small
and we want a deterministic, dependency-light encoder. Special tokens are
reserved at fixed IDs (``<pad>=0``, ``<bos>=1``, ``<eos>=2``, ``<unk>=3``)
and the rest of the vocab is filled in insertion order from ``fit``.

The module is pure Python at import time; the only heavy dependency
(``numpy``) is used inside :meth:`CharTokenizer.pad_to_arrays` to return
a padded ``(B, L)`` ``int64`` matrix. No ``torch`` is imported here — the
training-time collate function is responsible for wrapping the numpy
arrays into tensors.

Thai-specific note: each Unicode code point is treated as one character.
We do not attempt NFC/NFD normalization or sara-am decomposition; add
later if the data needs it.
"""
from __future__ import annotations

import numpy as np

__all__ = ["SPECIAL_TOKENS", "CharTokenizer"]

SPECIAL_TOKENS: tuple[str, ...] = ("<pad>", "<bos>", "<eos>", "<unk>")

_PAD_ID = 0
_BOS_ID = 1
_EOS_ID = 2
_UNK_ID = 3


class CharTokenizer:
    """A tiny char-level tokenizer with reserved special tokens.

    Vocab layout:
        - indices ``0..len(SPECIAL_TOKENS)-1`` are the special tokens in
          the order declared in :data:`SPECIAL_TOKENS`.
        - subsequent indices are characters added by :meth:`fit`, in the
          order they are first seen.
    """

    def __init__(self, texts: list[str] | None = None) -> None:
        """Build a char vocab. Special tokens are reserved at fixed IDs.

        If ``texts`` is ``None`` the character vocab starts empty; call
        :meth:`fit` later to populate it.
        """
        self._char_to_id: dict[str, int] = {}
        for tok in SPECIAL_TOKENS:
            self._char_to_id[tok] = len(self._char_to_id)
        if texts is not None:
            self.fit(texts)

    @property
    def vocab_size(self) -> int:
        return len(self._char_to_id)

    @property
    def pad_id(self) -> int:
        return _PAD_ID

    @property
    def bos_id(self) -> int:
        return _BOS_ID

    @property
    def eos_id(self) -> int:
        return _EOS_ID

    @property
    def unk_id(self) -> int:
        return _UNK_ID

    def fit(self, texts: list[str]) -> None:
        """Add all unique characters from ``texts`` to the vocab.

        Each Unicode code point counts as one character. Calling ``fit``
        repeatedly is safe: characters already in the vocab are not
        re-added, so vocab indices stay stable.
        """
        for text in texts:
            for ch in text:
                if ch in self._char_to_id:
                    continue
                self._char_to_id[ch] = len(self._char_to_id)

    def encode(
        self,
        text: str,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> list[int]:
        """Convert ``text`` to a list of int ids.

        Out-of-vocabulary characters map to :attr:`unk_id`. Optional
        ``<bos>`` / ``<eos>`` tokens can be prepended/appended.
        """
        ids: list[int] = []
        if add_bos:
            ids.append(self.bos_id)
        for ch in text:
            ids.append(self._char_to_id.get(ch, self.unk_id))
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: list[int], strip_special: bool = True) -> str:
        """Convert ``ids`` back to a string.

        If ``strip_special`` is true, ``<pad>``, ``<bos>`` and ``<unk>``
        tokens are dropped from the output and decoding stops at the
        first ``<eos>``. ``<eos>`` itself is also dropped.
        """
        special_ids = {self.pad_id, self.bos_id, self.eos_id, self.unk_id}
        id_to_char = {idx: ch for ch, idx in self._char_to_id.items()}
        out_chars: list[str] = []
        for i in ids:
            if strip_special and i in special_ids:
                if i == self.eos_id:
                    break
                continue
            out_chars.append(id_to_char.get(i, ""))
        return "".join(out_chars)

    def encode_batch(
        self,
        texts: list[str],
        max_len: int | None = None,
    ) -> tuple[list[list[int]], list[int]]:
        """Encode a batch of texts without adding bos/eos.

        Returns ``(input_ids, lengths)``. If ``max_len`` is given,
        sequences longer than that are truncated to ``max_len``
        characters. The returned ``lengths`` count the real characters
        in each sequence (after truncation) and exclude any bos/eos that
        might be added later by :meth:`pad_to_arrays`.
        """
        input_ids: list[list[int]] = []
        lengths: list[int] = []
        for text in texts:
            chars = list(text)
            if max_len is not None and len(chars) > max_len:
                chars = chars[:max_len]
            ids = [self._char_to_id.get(ch, self.unk_id) for ch in chars]
            input_ids.append(ids)
            lengths.append(len(ids))
        return input_ids, lengths

    def pad_to_arrays(
        self,
        batch_ids: list[list[int]],
        add_bos: bool = True,
        add_eos: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Pad a batch to a ``(B, L)`` ``int64`` array, optionally with bos/eos.

        The padding value is :attr:`pad_id`. The returned ``lengths``
        array counts only the real (non-special, non-pad) characters per
        row, i.e. the original encoded length before bos/eos were added.
        If ``batch_ids`` is empty, an empty ``(0, 0)`` array is returned
        for the padded ids and an empty ``(0,)`` array for the lengths.
        """
        if not batch_ids:
            return np.zeros((0, 0), dtype=np.int64), np.zeros((0,), dtype=np.int64)

        real_lengths = [len(ids) for ids in batch_ids]
        widths = [real + (1 if add_bos else 0) + (1 if add_eos else 0) for real in real_lengths]
        max_width = max(widths) if widths else 0

        bsz = len(batch_ids)
        padded = np.full((bsz, max_width), self.pad_id, dtype=np.int64)

        bos = self.bos_id if add_bos else None
        eos = self.eos_id if add_eos else None

        for i, ids in enumerate(batch_ids):
            col = 0
            if bos is not None:
                padded[i, col] = bos
                col += 1
            if ids:
                padded[i, col : col + len(ids)] = ids
                col += len(ids)
            if eos is not None:
                padded[i, col] = eos

        return padded, np.asarray(real_lengths, dtype=np.int64)
