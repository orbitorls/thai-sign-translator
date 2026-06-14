import pytest

from tsl.eval.sentence_metrics import (
    char_error_rate,
    edit_distance,
    evaluate_sentences,
    exact_match,
    token_error_rate,
)


def test_edit_distance_basic():
    assert edit_distance(list(""), list("a")) == 1
    assert edit_distance(list("abc"), list("abd")) == 1
    assert edit_distance(list("kitten"), list("sitting")) == 3


def test_edit_distance_empty():
    assert edit_distance([], []) == 0


def test_exact_match_perfect():
    assert exact_match(["a", "b"], ["a", "b"]) == pytest.approx(1.0)


def test_exact_match_partial():
    assert exact_match(["a", "b"], ["a", "c"]) == pytest.approx(0.5)


def test_char_error_rate_perfect():
    assert char_error_rate(["abc", "def"], ["abc", "def"]) == pytest.approx(0.0)


def test_char_error_rate_one_off():
    assert char_error_rate(["abc"], ["abd"]) == pytest.approx(1 / 3)


def test_char_error_rate_empty_ref():
    assert char_error_rate([""], ["a"]) == pytest.approx(1.0)
    assert char_error_rate([""], [""]) == pytest.approx(0.0)


def test_token_error_rate_basic():
    assert token_error_rate(["hello world"], ["hello there"]) == pytest.approx(1 / 2)


def test_token_error_rate_thai():
    assert token_error_rate(["ฉัน กิน ข้าว"], ["ฉัน กิน ข้าว"]) == pytest.approx(0.0)
    assert token_error_rate(["ฉัน กิน ข้าว"], ["ฉัน กิน ข้าวมาก"]) == pytest.approx(1 / 3)


def test_evaluate_sentences_returns_dict():
    out = evaluate_sentences(["hello world"], ["hello there"])
    assert set(out.keys()) == {"exact_match", "cer", "ter", "n"}
    assert out["n"] == 1


def test_evaluate_sentences_empty_inputs():
    out = evaluate_sentences([], [])
    assert out == {"exact_match": 0.0, "cer": 0.0, "ter": 0.0, "n": 0}
