import time
from tsl.eval.metrics import kshot_curve, measure_latency


def test_kshot_curve_keyed_by_shot():
    def fake_eval(shot: int) -> float:
        return shot / 10.0
    result = kshot_curve(fake_eval, shots=[1, 3, 5])
    assert set(result.keys()) == {1, 3, 5}
    assert result[1] == 0.1
    assert result[3] == 0.3
    assert result[5] == 0.5


def test_measure_latency_keys_and_ordering():
    def quick(x):
        return x + 1
    stats = measure_latency(quick, 41, repeats=5)
    assert set(stats.keys()) == {"mean_ms", "p95_ms"}
    assert stats["mean_ms"] >= 0.0
    assert stats["p95_ms"] >= stats["mean_ms"]


def test_measure_latency_reflects_sleep():
    def slow(_):
        time.sleep(0.01)
    stats = measure_latency(slow, None, repeats=3)
    assert stats["mean_ms"] >= 8.0
