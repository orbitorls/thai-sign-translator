"""Simple in-memory rate limiter for feedback submissions."""
from __future__ import annotations

import time
from collections import defaultdict, deque


class FeedbackRateLimiter:
    """Allow at most *max_per_hour* submissions per key within a rolling hour."""

    def __init__(self, max_per_hour: int = 20) -> None:
        self._max_per_hour = max_per_hour
        self._events: dict[str, deque[float]] = defaultdict(deque)

    def _prune(self, key: str, now: float) -> deque[float]:
        window_start = now - 3600.0
        events = self._events[key]
        while events and events[0] < window_start:
            events.popleft()
        return events

    def check(self, key: str) -> bool:
        now = time.time()
        events = self._prune(key, now)
        return len(events) < self._max_per_hour

    def record(self, key: str) -> None:
        now = time.time()
        events = self._prune(key, now)
        events.append(now)

    def check_both(self, session_id: str, ip_key: str) -> bool:
        return self.check(session_id) and self.check(ip_key)

    def record_both(self, session_id: str, ip_key: str) -> None:
        self.record(session_id)
        self.record(ip_key)

    def reset(self) -> None:
        self._events.clear()

    # Backward-compatible alias used by older tests.
    def allow(self, session_id: str) -> bool:
        if not self.check(session_id):
            return False
        self.record(session_id)
        return True
