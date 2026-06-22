"""Tests for sacrebleu-based metrics in tsl.eval.slt_metrics."""
import pytest
from tsl.eval.slt_metrics import (
    chrf_score,
    bleu_score,
    chrf_corpus,
    _chrf_score_legacy,
    evaluate_slt_with_sources,
)


# ---------------------------------------------------------------------------
# chrf_score
# ---------------------------------------------------------------------------

def test_chrf_score_returns_float_in_range():
    score = chrf_score(["สวัสดี"], ["สวัสดีครับ"])
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


def test_chrf_score_perfect_match():
    score = chrf_score(["สวัสดี"], ["สวัสดี"])
    assert abs(score - 1.0) < 1e-6


def test_chrf_score_higher_for_closer_strings():
    close_score = chrf_score(["สวัสดีครับ"], ["สวัสดี"])
    far_score = chrf_score(["กขค"], ["สวัสดี"])
    assert close_score > far_score


def test_chrf_score_does_not_treat_mixed_pairs_as_perfect():
    score = chrf_score(["abc", "def"], ["abc", "xyz"])
    assert score < 1.0


def test_chrf_score_empty_input_returns_zero():
    assert chrf_score([], []) == 0.0


def test_chrf_score_empty_hypothesis():
    score = chrf_score([""], ["สวัสดี"])
    assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# bleu_score
# ---------------------------------------------------------------------------

def test_bleu_score_returns_float_in_range():
    score = bleu_score(["สวัสดี"], ["สวัสดีครับ"])
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


def test_bleu_score_perfect_match():
    score = bleu_score(["สวัสดี"], ["สวัสดี"])
    assert abs(score - 1.0) < 1e-6


def test_bleu_score_does_not_treat_mixed_pairs_as_perfect():
    score = bleu_score(["abc", "def"], ["abc", "xyz"])
    assert score < 1.0


def test_bleu_score_empty_input_returns_zero():
    assert bleu_score([], []) == 0.0


# ---------------------------------------------------------------------------
# chrf_corpus
# ---------------------------------------------------------------------------

def test_chrf_corpus_has_required_keys():
    result = chrf_corpus(["สวัสดี"], ["สวัสดี"])
    assert "chrf" in result
    assert "bleu" in result
    assert "exact_match" in result
    assert "exact_match_pct" in result
    assert "n" in result
    assert "mean_hyp_len" in result


def test_chrf_corpus_perfect_match_values():
    result = chrf_corpus(["สวัสดี"], ["สวัสดี"])
    assert result["chrf"] == 100.0
    assert result["exact_match"] == 1
    assert result["exact_match_pct"] == 100.0
    assert result["n"] == 1


def test_chrf_corpus_empty_input_returns_zeros():
    result = chrf_corpus([], [])
    assert result["chrf"] == 0.0
    assert result["bleu"] == 0.0
    assert result["exact_match"] == 0
    assert result["n"] == 0


def test_chrf_corpus_multiple_examples():
    hyps = ["สวัสดี", "ขอบคุณ", "สบายดี"]
    refs = ["สวัสดี", "ขอบคุณครับ", "ไม่สบาย"]
    result = chrf_corpus(hyps, refs)
    assert result["n"] == 3
    assert result["exact_match"] == 1  # only "สวัสดี" matches exactly
    assert 0.0 <= result["chrf"] <= 100.0


# ---------------------------------------------------------------------------
# evaluate_slt_with_sources
# ---------------------------------------------------------------------------

class _FakeTranslation:
    def __init__(self, sentence):
        self.sentence = sentence
        self.score = 0.0


class _FakeTranslator:
    def __init__(self, responses):
        self._responses = iter(responses)

    def translate(self, features, max_len=64, beam_size=4):
        return _FakeTranslation(next(self._responses))


class _FakeExample:
    def __init__(self, target_text, source, features_path="dummy"):
        self.target_text = target_text
        self.source = source
        self.features_path = features_path


def _fake_load_fn(path):
    return None  # features not used by FakeTranslator


def test_evaluate_slt_with_sources_returns_overall_and_per_source():
    examples = [
        _FakeExample("สวัสดี", "tsl51"),
        _FakeExample("ขอบคุณ", "tsl51"),
        _FakeExample("สบายดี", "youtube_sl25"),
    ]
    translator = _FakeTranslator(["สวัสดี", "ขอบคุณครับ", "สบายดี"])
    result = evaluate_slt_with_sources(translator, examples, _fake_load_fn)

    assert "overall" in result
    assert "per_source" in result


def test_evaluate_slt_with_sources_per_source_breakdown():
    examples = [
        _FakeExample("สวัสดี", "tsl51"),
        _FakeExample("ขอบคุณ", "tsl51"),
        _FakeExample("สบายดี", "youtube_sl25"),
    ]
    translator = _FakeTranslator(["สวัสดี", "ขอบคุณครับ", "สบายดี"])
    result = evaluate_slt_with_sources(translator, examples, _fake_load_fn)

    assert "tsl51" in result["per_source"]
    assert "youtube_sl25" in result["per_source"]
    assert result["per_source"]["tsl51"]["n"] == 2
    assert result["per_source"]["youtube_sl25"]["n"] == 1


def test_evaluate_slt_with_sources_overall_metrics():
    examples = [
        _FakeExample("สวัสดี", "tsl51"),
        _FakeExample("สบายดี", "youtube_sl25"),
    ]
    # Both hypotheses match perfectly
    translator = _FakeTranslator(["สวัสดี", "สบายดี"])
    result = evaluate_slt_with_sources(translator, examples, _fake_load_fn)

    overall = result["overall"]
    assert overall["n"] == 2
    assert overall["exact_match"] == 2
    assert "chrf" in overall
    assert "bleu" in overall


# ---------------------------------------------------------------------------
# Legacy backward compatibility
# ---------------------------------------------------------------------------

def test_legacy_chrf_score_still_works():
    score = _chrf_score_legacy(["สวัสดี"], ["สวัสดี"])
    assert abs(score - 1.0) < 1e-6


def test_legacy_chrf_score_returns_float_in_range():
    score = _chrf_score_legacy(["สวัสดีครับ"], ["สวัสดี"])
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


def test_legacy_chrf_score_empty():
    assert _chrf_score_legacy([], []) == 0.0
