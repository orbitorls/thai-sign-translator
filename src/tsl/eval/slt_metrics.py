"""Sentence-level evaluation metrics for Thai SLT.

chrF (character F-score) is the primary metric because:
- It operates at character level → matches our CharTokenizer
- Works well for Thai (no word-boundary spaces needed)
- Standard in sign language translation research

Usage
-----
    from tsl.eval.slt_metrics import chrf_score, evaluate_slt

    score = chrf_score(hypotheses=["สวัสดี"], references=["สวัสดีครับ"])
    results = evaluate_slt(translator, examples, load_fn)
"""
from __future__ import annotations

import math

import numpy as np

__all__ = ["chrf_score", "chrf_corpus", "evaluate_slt"]


# ---------------------------------------------------------------------------
# chrF implementation (no sacrebleu dependency)
# ---------------------------------------------------------------------------

def _char_ngrams(text: str, n: int) -> dict[str, int]:
    counts: dict[str, int] = {}
    for i in range(len(text) - n + 1):
        ng = text[i : i + n]
        counts[ng] = counts.get(ng, 0) + 1
    return counts


def _f_score(precision: float, recall: float, beta: float = 2.0) -> float:
    b2 = beta * beta
    denom = b2 * precision + recall
    if denom < 1e-12:
        return 0.0
    return (1 + b2) * precision * recall / denom


def chrf_score(
    hypotheses: list[str],
    references: list[str],
    max_n: int = 6,
    beta: float = 2.0,
) -> float:
    """Corpus-level chrF score in [0, 1].

    beta=2 weights recall twice as heavily as precision (standard for MT).
    Returns 0.0 for empty inputs.
    """
    if not hypotheses:
        return 0.0

    total_p = total_r = 0.0
    count = 0
    for hyp, ref in zip(hypotheses, references):
        p_sum = r_sum = 0.0
        for n in range(1, max_n + 1):
            hyp_ng = _char_ngrams(hyp, n)
            ref_ng = _char_ngrams(ref, n)
            if not ref_ng:
                continue
            match = sum(min(hyp_ng.get(k, 0), v) for k, v in ref_ng.items())
            hyp_total = sum(hyp_ng.values())
            ref_total = sum(ref_ng.values())
            p_sum += match / hyp_total if hyp_total else 0.0
            r_sum += match / ref_total if ref_total else 0.0
        total_p += p_sum / max_n
        total_r += r_sum / max_n
        count += 1

    if count == 0:
        return 0.0
    avg_p = total_p / count
    avg_r = total_r / count
    return _f_score(avg_p, avg_r, beta)


def chrf_corpus(
    hypotheses: list[str],
    references: list[str],
) -> dict:
    """Return chrF + per-sentence breakdown."""
    score = chrf_score(hypotheses, references)
    exact = sum(h == r for h, r in zip(hypotheses, references))
    lengths = [len(h) for h in hypotheses]
    return {
        "chrf": round(score * 100, 2),
        "exact_match": exact,
        "exact_match_pct": round(exact / max(len(hypotheses), 1) * 100, 1),
        "n": len(hypotheses),
        "mean_hyp_len": round(sum(lengths) / max(len(lengths), 1), 1),
    }


# ---------------------------------------------------------------------------
# High-level evaluation runner
# ---------------------------------------------------------------------------

def evaluate_slt(
    translator,
    examples,
    load_fn,
    beam_size: int = 4,
    max_len: int = 64,
    verbose: bool = False,
) -> dict:
    """Decode every example and compute chrF.

    Parameters
    ----------
    translator : SentenceTranslator
    examples   : list[SignTextExample]
    load_fn    : callable(path) -> (T, D) ndarray
    beam_size  : beam width (1 = greedy)
    verbose    : print each hypothesis vs reference

    Returns
    -------
    dict with keys: chrf, exact_match, exact_match_pct, n, mean_hyp_len,
                    hypotheses, references
    """
    hypotheses: list[str] = []
    references: list[str] = []

    for ex in examples:
        features = load_fn(ex.features_path)
        pred = translator.translate(features, max_len=max_len, beam_size=beam_size)
        hypotheses.append(pred.sentence)
        references.append(ex.target_text)
        if verbose:
            match = "✓" if pred.sentence == ex.target_text else "✗"
            print(f"  {match} hyp: {pred.sentence!r}  ref: {ex.target_text!r}  score={pred.score:.3f}")

    metrics = chrf_corpus(hypotheses, references)
    metrics["hypotheses"] = hypotheses
    metrics["references"] = references
    return metrics
