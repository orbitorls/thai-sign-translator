import numpy as np
from tsl.baseline.dtw import DTWBaseline


def test_dtw_classifies_separable_pair():
    rng = np.random.default_rng(0)
    D = 6
    low_clips = [np.zeros((4, D), dtype=np.float32) + rng.normal(0, 0.01, (4, D)) for _ in range(2)]
    high_clips = [np.full((4, D), 5.0, dtype=np.float32) + rng.normal(0, 0.01, (4, D)) for _ in range(2)]
    base = DTWBaseline()
    base.add_sign("low", low_clips)
    base.add_sign("high", high_clips)
    query_low = np.zeros((3, D), dtype=np.float32)
    query_high = np.full((3, D), 5.0, dtype=np.float32)
    word_low, score_low = base.predict(query_low)
    word_high, _ = base.predict(query_high)
    assert word_low == "low"
    assert word_high == "high"
    assert isinstance(score_low, float)


def test_dtw_predict_returns_known_name():
    D = 4
    base = DTWBaseline()
    base.add_sign("only", [np.ones((2, D), dtype=np.float32)])
    word, score = base.predict(np.ones((5, D), dtype=np.float32))
    assert word == "only"
    assert score >= 0.0
