"""Tests for tsl.eval.slt_metrics."""
from tsl.eval.slt_metrics import chrf_score, chrf_corpus


def test_chrf_perfect():
    score = chrf_score(["สวัสดี"], ["สวัสดี"])
    assert abs(score - 1.0) < 1e-6


def test_chrf_empty_hypothesis():
    score = chrf_score([""], ["สวัสดี"])
    assert score == 0.0


def test_chrf_no_overlap():
    score = chrf_score(["กขค"], ["ไก่จิก"])
    assert score >= 0.0
    assert score < 0.5


def test_chrf_partial():
    score = chrf_score(["สวัสดีครับ"], ["สวัสดี"])
    assert 0.0 < score < 1.0


def test_chrf_corpus_keys():
    result = chrf_corpus(["สวัสดี"], ["สวัสดี"])
    assert "chrf" in result
    assert "exact_match" in result
    assert "exact_match_pct" in result
    assert result["chrf"] == 100.0
    assert result["exact_match"] == 1


def test_chrf_empty_list():
    assert chrf_score([], []) == 0.0
