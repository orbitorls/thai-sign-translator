"""Unit tests for feedback rate limiter."""
from __future__ import annotations

from tsl.feedback.rate_limit import FeedbackRateLimiter


def test_check_does_not_record():
    limiter = FeedbackRateLimiter(max_per_hour=2)
    assert limiter.check("sess-a") is True
    assert limiter.check("sess-a") is True


def test_record_enforces_limit():
    limiter = FeedbackRateLimiter(max_per_hour=2)
    limiter.record("sess-b")
    limiter.record("sess-b")
    assert limiter.check("sess-b") is False


def test_check_both_requires_both_keys_under_limit():
    limiter = FeedbackRateLimiter(max_per_hour=1)
    limiter.record("ip:1.2.3.4")
    assert limiter.check_both("sess-c", "ip:1.2.3.4") is False
    assert limiter.check_both("sess-d", "ip:9.9.9.9") is True
