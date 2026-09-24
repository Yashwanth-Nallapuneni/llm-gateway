from __future__ import annotations

import random

import pytest

from llm_gateway.retry import RetryPolicy
from llm_gateway.types import ProviderError, RateLimitError


def policy(**kw) -> RetryPolicy:
    kw.setdefault("jitter", False)
    return RetryPolicy(**kw)


@pytest.mark.parametrize("status", [408, 429, 500, 502, 503, 504])
def test_retryable_statuses(status):
    assert policy().should_retry(ProviderError("x", status=status), 0) is True


@pytest.mark.parametrize("status", [400, 401, 403, 404, 409, 422])
def test_non_retryable_statuses(status):
    assert policy().should_retry(ProviderError("x", status=status), 0) is False


@pytest.mark.parametrize(
    "exc", [TimeoutError(), ConnectionResetError(), OSError("reset")]
)
def test_transport_failures_are_retryable(exc):
    assert policy().should_retry(exc, 0) is True


def test_programming_errors_are_not_retryable():
    assert policy().should_retry(ValueError("bad arg"), 0) is False


def test_attempt_budget_is_exhausted_independently_of_status():
    p = policy(max_attempts=3)
    exc = RateLimitError()
    assert p.should_retry(exc, 0) is True
    assert p.should_retry(exc, 1) is True
    assert p.should_retry(exc, 2) is False  # 3 attempts used


def test_exponential_growth():
    p = policy(base_delay=0.5, max_delay=1000)
    assert [p.delay_for(i) for i in range(5)] == [0.5, 1.0, 2.0, 4.0, 8.0]


def test_delay_is_capped():
    p = policy(base_delay=0.5, max_delay=10)
    assert p.delay_for(20) == 10
    assert all(p.delay_for(i) <= 10 for i in range(50))


def test_jitter_produces_varied_delays():
    p = RetryPolicy(base_delay=1.0, jitter=True, rng=random.Random(7))
    samples = [p.delay_for(3) for _ in range(100)]
    assert len(set(samples)) > 90  # essentially all distinct
    assert all(0 <= s <= 8.0 for s in samples)
    # Full jitter samples uniformly over [0, delay]: mean should land near half.
    assert 3.0 < sum(samples) / len(samples) < 5.0


def test_jitter_stays_within_the_cap():
    p = RetryPolicy(base_delay=1.0, max_delay=5.0, jitter=True)
    assert all(p.delay_for(i) <= 5.0 for i in range(30))


def test_retry_after_overrides_a_shorter_computed_delay():
    p = policy(base_delay=0.5)
    # computed would be 0.5s; the provider said 30.
    assert p.delay_for(0, retry_after=30.0) == 30.0


def test_retry_after_does_not_shorten_a_longer_computed_delay():
    p = policy(base_delay=1.0, max_delay=1000)
    computed = p.delay_for(6)  # 64s
    assert p.delay_for(6, retry_after=5.0) == computed


def test_retry_after_is_read_off_the_exception():
    p = policy()
    exc = RateLimitError(retry_after=12.0)
    assert p.retry_after_from(exc) == 12.0
    assert p.retry_after_from(ValueError()) is None


def test_unknown_status_defaults_to_its_class():
    p = policy()
    assert p.should_retry(ProviderError("x", status=507), 0) is True  # unknown 5xx
    assert p.should_retry(ProviderError("x", status=418), 0) is False  # unknown 4xx


def test_status_none_is_treated_as_transport_failure():
    assert policy().should_retry(ProviderError("connection dropped"), 0) is True
