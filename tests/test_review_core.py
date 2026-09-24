"""Targeted regression tests written during a correctness review of
rate_limit.py, retry.py, batching.py, queue.py, breaker.py and routing.py.

Offline, fast, deterministic -- fake clocks only, no network, no real sleeps
beyond what the fake clock/event loop needs to settle.
"""

from __future__ import annotations

import pytest

from llm_gateway import LLMGateway
from llm_gateway.breaker import CircuitBreaker, State
from llm_gateway.providers.base import Provider
from llm_gateway.providers.mock import MockClient, MockProvider
from llm_gateway.rate_limit import ProviderLimiter
from llm_gateway.retry import RetryPolicy
from llm_gateway.routing import ProviderRouter
from llm_gateway.types import LLMRequest, ProviderCapabilities

# ---------------------------------------------------------------------------
# rate_limit.py
# ---------------------------------------------------------------------------


async def test_provider_limiter_acquire_deducts_full_token_cost(clock) -> None:
    """A request within capacity must deduct its full cost, not a clamped one."""
    limiter = ProviderLimiter(
        rpm_limit=600, tpm_limit=1000, clock=clock, sleep=clock.sleep
    )
    await limiter.acquire(n_requests=1, n_tokens=900)
    assert limiter.tokens.available == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# retry.py
# ---------------------------------------------------------------------------


def test_delay_for_does_not_overflow_on_a_huge_attempt_count() -> None:
    """delay_for(attempt) must saturate at max_delay, never crash.

    A caller can configure a large max_attempts (nothing stops them), and
    the retry loop passes the running `attempt` straight through. Before
    the fix, `2.0 ** attempt` raised OverflowError once attempt reached the
    low thousands -- long before the min(raw, max_delay) clamp got a
    chance to bound it -- turning "give up gracefully after many retries"
    into a crash instead.
    """
    policy = RetryPolicy(
        max_attempts=10_000, base_delay=0.5, max_delay=60.0, jitter=False
    )
    assert policy.delay_for(5000) == pytest.approx(60.0)
    assert policy.delay_for(2000) == pytest.approx(60.0)


def test_delay_for_still_grows_normally_for_small_attempts() -> None:
    policy = RetryPolicy(max_attempts=10, base_delay=1.0, max_delay=60.0, jitter=False)
    assert policy.delay_for(0) == pytest.approx(1.0)
    assert policy.delay_for(2) == pytest.approx(4.0)
    assert policy.delay_for(10) == pytest.approx(60.0)  # capped, not 1024


# ---------------------------------------------------------------------------
# breaker.py / routing.py: half-open probe budget
# ---------------------------------------------------------------------------


def test_half_open_allows_exactly_one_probe(clock) -> None:
    cb = CircuitBreaker(failure_threshold=1, recovery_timeout=30.0, clock=clock)
    cb.record_failure()
    state_after_trip = cb.state
    assert state_after_trip is State.OPEN
    clock.advance(30.0)
    state_after_recovery = cb.state
    assert state_after_recovery is State.HALF_OPEN
    assert cb.allows_request() is True
    # A second concurrent caller must not also get through.
    assert cb.allows_request() is False


async def test_unused_half_open_candidate_keeps_its_probe(clock) -> None:
    # Two providers are both half-open. The router reserves a probe on each
    # while ranking, but the first one serves the request, so the second
    # was never tried and must get its probe back.
    def make(name: str) -> Provider:
        return Provider(
            name,
            MockClient(name),
            ProviderCapabilities(),
            rpm_limit=600,
            tpm_limit=150_000,
            failure_threshold=1,
            recovery_timeout=30.0,
            clock=clock,
        )

    a, b = make("a"), make("b")
    a.breaker.record_failure()
    b.breaker.record_failure()
    clock.advance(30.0)
    assert a.breaker.state is State.HALF_OPEN
    assert b.breaker.state is State.HALF_OPEN

    gw = LLMGateway(providers=[a, b])
    async with gw:
        resp = await gw.submit(LLMRequest("hi"))

    served, unused = (a, b) if resp.provider == "a" else (b, a)
    assert served.breaker.state is State.CLOSED
    assert unused.breaker.allows_request() is True


# ---------------------------------------------------------------------------
# routing.py: sanity check that ranking still behaves under ties
# ---------------------------------------------------------------------------


def test_select_all_empty_when_everything_filtered() -> None:
    router = ProviderRouter()
    p = MockProvider("only", max_context_tokens=10)
    req = LLMRequest("x" * 1000, max_tokens=900)  # far exceeds context
    assert router.select_all(req, [p]) == []
