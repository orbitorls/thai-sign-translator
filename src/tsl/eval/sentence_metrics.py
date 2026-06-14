"""Sentence-level evaluation metrics: exact match, CER, TER.

Operates on plain ``str``; tokenization is a simple whitespace split and
happens inside :func:`token_error_rate` so the metrics layer stays
decoupled from any Thai-aware tokenizer.
"""
from __future__ import annotations

__all__ = [
    "edit_distance",
    "exact_match",
    "char_error_rate",
    "token_error_rate",
    "evaluate_sentences",
]


def edit_distance(ref: list, hyp: list) -> int:
    """Levenshtein distance using dynamic programming.

    Works for any list (str-as-list-of-chars or list of tokens). Uses two
    rolling rows for O(min(len(ref), len(hyp))) extra space.
    """
    if len(ref) == 0:
        return len(hyp)
    if len(hyp) == 0:
        return len(ref)

    prev = list(range(len(hyp) + 1))
    for i in range(1, len(ref) + 1):
        cur = [i] + [0] * len(hyp)
        for j in range(1, len(hyp) + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            cur[j] = min(
                prev[j] + 1,
                cur[j - 1] + 1,
                prev[j - 1] + cost,
            )
        prev = cur
    return prev[len(hyp)]


def exact_match(references: list[str], hypotheses: list[str]) -> float:
    """Fraction of hypotheses exactly equal to their reference. 0.0 if inputs empty."""
    if len(references) == 0:
        return 0.0
    if len(references) != len(hypotheses):
        raise ValueError(f"length mismatch: {len(references)} refs vs {len(hypotheses)} hyps")
    correct = sum(1 for r, h in zip(references, hypotheses) if r == h)
    return correct / len(references)


def char_error_rate(references: list[str], hypotheses: list[str]) -> float:
    """Macro-averaged CER: average per-pair CER, or 0.0 if no examples."""
    if len(references) == 0:
        return 0.0
    if len(references) != len(hypotheses):
        raise ValueError(f"length mismatch: {len(references)} refs vs {len(hypotheses)} hyps")
    total = 0.0
    for r, h in zip(references, hypotheses):
        if len(r) == 0:
            total += 0.0 if len(h) == 0 else 1.0
        else:
            total += edit_distance(list(r), list(h)) / len(r)
    return total / len(references)


def token_error_rate(references: list[str], hypotheses: list[str]) -> float:
    """Macro-averaged TER using whitespace split. 0.0 if no examples."""
    if len(references) == 0:
        return 0.0
    if len(references) != len(hypotheses):
        raise ValueError(f"length mismatch: {len(references)} refs vs {len(hypotheses)} hyps")
    total = 0.0
    for r, h in zip(references, hypotheses):
        r_tokens = r.split()
        h_tokens = h.split()
        if len(r_tokens) == 0:
            total += 0.0 if len(h_tokens) == 0 else 1.0
        else:
            total += edit_distance(r_tokens, h_tokens) / len(r_tokens)
    return total / len(references)


def evaluate_sentences(references: list[str], hypotheses: list[str]) -> dict:
    """Return ``{"exact_match", "cer", "ter", "n"}``. All zeros when ``n == 0``."""
    n = len(references)
    if n == 0:
        return {"exact_match": 0.0, "cer": 0.0, "ter": 0.0, "n": 0}
    return {
        "exact_match": exact_match(references, hypotheses),
        "cer": char_error_rate(references, hypotheses),
        "ter": token_error_rate(references, hypotheses),
        "n": n,
    }
