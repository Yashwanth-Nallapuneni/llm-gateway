"""Adaptive (AIMD) rate limiting: opt-in slow-down-on-429, creep-back-up-on-success.

All offline and deterministic -- a fake clock stands in for time.monotonic(),
and no test relies on asyncio.sleep actually waiting.
"""

from __future__ import annotations

import asyncio

from llm_gateway import LLMGateway, LLMRequest
from llm_gateway.providers.mock import MockClient, MockProvider
from llm_gateway.rate_limit import ProviderLimiter
from llm_gateway.retry import RetryPolicy


def fast_retry(**kw) -> RetryPolicy:
    kw.setdefault("base_delay", 0.001)
    kw.setdefault("max_delay", 0.01)
    return RetryPolicy(**kw)


def test_rate_halves_on_429():
    limiter = ProviderLimiter(rpm_limit=600, tpm_limit=150_000, adaptive=True)
    limiter.on_throttled()
    assert limiter.requests.refill_rate == 5.0  # 600/60 == 10, halved == 5


def test_rate_does_not_go_below_floor_after_repeated_429s():
    limiter = ProviderLimiter(rpm_limit=600, tpm_limit=150_000, adaptive=True)
    for _ in range(20):
        limiter.on_throttled()
    # Floor is 10% of the configured rate (10 tokens/sec configured -> 1.0).
    assert limiter.requests.refill_rate == 1.0


def test_rate_grows_back_by_fixed_step_on_success():
    limiter = ProviderLimiter(rpm_limit=600, tpm_limit=150_000, adaptive=True)
    limiter.on_throttled()  # 10 -> 5
    limiter.on_success()  # +1% of 10 == +0.1
    assert limiter.requests.refill_rate == 5.1


def test_rate_capped_at_configured_rate():
    limiter = ProviderLimiter(rpm_limit=600, tpm_limit=150_000, adaptive=True)
    for _ in range(1000):
        limiter.on_success()
    assert limiter.requests.refill_rate == 10.0


def test_adaptive_disabled_by_default_leaves_rate_untouched():
    limiter = ProviderLimiter(rpm_limit=600, tpm_limit=150_000)
    assert limiter.adaptive is False
    limiter.on_throttled()
    limiter.on_throttled()
    limiter.on_success()
    assert limiter.requests.refill_rate == 10.0


async def test_gateway_end_to_end_slows_down_after_429s():
    # First two calls hit a simulated 429 (with no Retry-After, so the
    # retry policy's own backoff is used, not a provider-dictated delay);
    # everything after that succeeds. Adaptive mode is on, so those two
    # 429s should have halved the request bucket's refill rate twice by
    # the time the run finishes.
    sequence: list[int | None] = [429, 429, None, None, None]
    client = MockClient("flaky", fail_sequence=sequence)
    provider = MockProvider("flaky", client=client, rpm_limit=600)

    gw = LLMGateway(providers=[provider], retry=fast_retry(), adaptive=True)
    async with gw:
        resp = await gw.submit(LLMRequest("hello"))

    assert resp.provider == "flaky"
    # 10 -> 5 (first 429) -> 2.5 (second 429) -> 2.6 (the eventual success
    # adds back one step of 1% of the configured rate, 0.1).
    assert provider.limiter.requests.refill_rate == 2.6


async def test_one_batch_level_429_halves_rate_once():
    # A single 429 returned for a whole batch of requests is one rejection
    # from the provider, so it must halve the rate once, not once per request.
    client = MockClient("p", fail_sequence=[429])
    provider = MockProvider("p", client=client, rpm_limit=600)
    gw = LLMGateway(providers=[provider], adaptive=True)
    before = provider.limiter.requests.refill_rate
    await asyncio.gather(*(gw.submit(LLMRequest(prompt=f"q{i}")) for i in range(5)))
    after = provider.limiter.requests.refill_rate
    # Halved once, then grown back a little by up to 5 successes.
    assert before * 0.5 <= after <= before * 0.5 + 5 * before * 0.01 + 1e-9
    await gw.aclose()
