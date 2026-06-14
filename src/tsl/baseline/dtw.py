"""(STRETCH) DTW nearest-neighbor baseline over normalized landmark sequences."""
from __future__ import annotations

import numpy as np
from scipy.spatial.distance import euclidean

try:  # pragma: no cover - optional stretch dependency
    from fastdtw import fastdtw
except Exception:  # pragma: no cover
    fastdtw = None


def _dtw_fallback(a: np.ndarray, b: np.ndarray) -> float:
    n, m = len(a), len(b)
    dp = np.full((n + 1, m + 1), np.inf, dtype=np.float64)
    dp[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = float(euclidean(a[i - 1], b[j - 1]))
            dp[i, j] = cost + min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1])
    return float(dp[n, m])


class DTWBaseline:
    def __init__(self) -> None:
        self._clips: dict[str, list[np.ndarray]] = {}

    def add_sign(self, name: str, clips: list[np.ndarray]) -> None:
        store = self._clips.setdefault(name, [])
        for clip in clips:
            store.append(np.asarray(clip, dtype=np.float32))

    def predict(self, seq: np.ndarray) -> tuple[str, float]:
        if not self._clips:
            raise ValueError("DTWBaseline has no registered signs")
        query = np.asarray(seq, dtype=np.float32)
        best_name, best_dist = None, float("inf")
        for name, examples in self._clips.items():
            for example in examples:
                if fastdtw is not None:
                    dist, _path = fastdtw(query, example, dist=euclidean)
                else:
                    dist = _dtw_fallback(query, example)
                if dist < best_dist:
                    best_dist = float(dist)
                    best_name = name
        score = 1.0 / (1.0 + best_dist)
        assert best_name is not None
        return best_name, float(score)
