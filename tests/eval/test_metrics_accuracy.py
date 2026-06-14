import pytest
from tsl.eval.metrics import accuracy, topk_accuracy


def test_accuracy_handworked():
    y_true = [0, 1, 2, 3]
    y_pred = [0, 1, 2, 0]
    assert accuracy(y_true, y_pred) == pytest.approx(0.75)


def test_accuracy_all_correct():
    assert accuracy([1, 2, 3], [1, 2, 3]) == pytest.approx(1.0)


def test_accuracy_empty_is_zero():
    assert accuracy([], []) == 0.0


def test_topk_accuracy_handworked():
    y_true = [0, 1, 2, 3]
    topk_preds = [[0, 5], [9, 1], [7, 8], [3, 0]]
    assert topk_accuracy(y_true, topk_preds, k=2) == pytest.approx(0.75)


def test_topk_accuracy_k1_matches_accuracy():
    y_true = [0, 1, 2]
    topk_preds = [[0, 9], [9, 1], [2, 9]]
    assert topk_accuracy(y_true, topk_preds, k=1) == pytest.approx(2 / 3)
