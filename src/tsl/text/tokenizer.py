"""Thai tokenizers for the SLT pipeline.

Two tokenizer variants share the same API:

* :class:`CharTokenizer` — one token per Unicode code point.
* :class:`WordTokenizer` — one token per whitespace-delimited word.

Special tokens are reserved at fixed IDs in both variants:
    <pad>=0, <bos>=1, <eos>=2, <unk>=3

The module is pure Python at import time; numpy is used only inside
``pad_to_arrays``.  No torch is imported here.
"""
from __future__ import annotations

import numpy as np

__all__ = ["SPECIAL_TOKENS", "CharTokenizer", "WordTokenizer"]

SPECIAL_TOKENS: tuple[str, ...] = ("<pad>", "<bos>", "<eos>", "<unk>")

_PAD_ID = 0
_BOS_ID = 1
_EOS_ID = 2
_UNK_ID = 3


# ---------------------------------------------------------------------------
# Shared mixin — both tokenizers inherit from this to avoid duplication
# ---------------------------------------------------------------------------

class _BaseTokenizer:
    """Properties and helpers shared between CharTokenizer and WordTokenizer."""

    # Sub-classes must assign self._tok_to_id: dict[str, int]

    @property
    def vocab_size(self) -> int:
        return len(self._tok_to_id)

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

    def _tokenize(self, text: str) -> list[str]:
        raise NotImplementedError

    def _join(self, tokens: list[str]) -> str:
        raise NotImplementedError

    def fit(self, texts: list[str]) -> None:
        for text in texts:
            for tok in self._tokenize(text):
                if tok not in self._tok_to_id:
                    self._tok_to_id[tok] = len(self._tok_to_id)

    def encode(
        self,
        text: str,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> list[int]:
        ids: list[int] = []
        if add_bos:
            ids.append(self.bos_id)
        for tok in self._tokenize(text):
            ids.append(self._tok_to_id.get(tok, self.unk_id))
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: list[int], strip_special: bool = True) -> str:
        special_ids = {self.pad_id, self.bos_id, self.eos_id, self.unk_id}
        id_to_tok = {idx: tok for tok, idx in self._tok_to_id.items()}
        out: list[str] = []
        for i in ids:
            if strip_special and i in special_ids:
                if i == self.eos_id:
                    break
                continue
            out.append(id_to_tok.get(i, ""))
        return self._join(out)

    def encode_batch(
        self,
        texts: list[str],
        max_len: int | None = None,
    ) -> tuple[list[list[int]], list[int]]:
        input_ids: list[list[int]] = []
        lengths: list[int] = []
        for text in texts:
            toks = self._tokenize(text)
            if max_len is not None and len(toks) > max_len:
                toks = toks[:max_len]
            ids = [self._tok_to_id.get(t, self.unk_id) for t in toks]
            input_ids.append(ids)
            lengths.append(len(ids))
        return input_ids, lengths

    def pad_to_arrays(
        self,
        batch_ids: list[list[int]],
        add_bos: bool = True,
        add_eos: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        if not batch_ids:
            return np.zeros((0, 0), dtype=np.int64), np.zeros((0,), dtype=np.int64)

        real_lengths = [len(ids) for ids in batch_ids]
        widths = [r + (1 if add_bos else 0) + (1 if add_eos else 0) for r in real_lengths]
        max_width = max(widths) if widths else 0

        bsz = len(batch_ids)
        padded = np.full((bsz, max_width), self.pad_id, dtype=np.int64)

        for i, ids in enumerate(batch_ids):
            col = 0
            if add_bos:
                padded[i, col] = self.bos_id
                col += 1
            if ids:
                padded[i, col : col + len(ids)] = ids
                col += len(ids)
            if add_eos:
                padded[i, col] = self.eos_id

        return padded, np.asarray(real_lengths, dtype=np.int64)


# ---------------------------------------------------------------------------
# CharTokenizer
# ---------------------------------------------------------------------------

class CharTokenizer(_BaseTokenizer):
    """Char-level tokenizer. Each Unicode code point is one token.

    Thai-specific note: we do not attempt NFC/NFD normalization or
    sara-am decomposition; add later if the data needs it.
    """

    def __init__(self, texts: list[str] | None = None) -> None:
        self._tok_to_id: dict[str, int] = {}
        for tok in SPECIAL_TOKENS:
            self._tok_to_id[tok] = len(self._tok_to_id)
        if texts is not None:
            self.fit(texts)

    # legacy alias used by external code that reads _char_to_id directly
    @property
    def _char_to_id(self) -> dict[str, int]:
        return self._tok_to_id

    @_char_to_id.setter
    def _char_to_id(self, value: dict[str, int]) -> None:
        self._tok_to_id = value

    def _tokenize(self, text: str) -> list[str]:
        return list(text)

    def _join(self, tokens: list[str]) -> str:
        return "".join(tokens)


# ---------------------------------------------------------------------------
# WordTokenizer
# ---------------------------------------------------------------------------

class WordTokenizer(_BaseTokenizer):
    """Word-level tokenizer. Splits on whitespace; decodes with spaces.

    Ideal for small Thai datasets where every word in the target vocabulary
    is known at training time (e.g. TSL-51 with only 30 unique words).
    """

    def __init__(self, texts: list[str] | None = None) -> None:
        self._tok_to_id: dict[str, int] = {}
        for tok in SPECIAL_TOKENS:
            self._tok_to_id[tok] = len(self._tok_to_id)
        if texts is not None:
            self.fit(texts)

    def _tokenize(self, text: str) -> list[str]:
        return text.split()

    def _join(self, tokens: list[str]) -> str:
        return " ".join(tokens)
