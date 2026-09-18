from __future__ import annotations

import asyncio

import pytest

from llm_gateway.rate_limit import (
    ProviderLimiter,
    TokenBucket,
    _parse_duration,
    _parse_float,
)


async def test_burst_up_to_capacity_is_immediate(clock):
    bucket = TokenBucket(10, 2, clock=clock, sleep=clock.sleep)
    for _ in range(10):
        await bucket.acquire()
    assert clock.now == 1000.0  # no virtual time passed
    assert bucket.available == pytest.approx(0.0)


async def test_one_over_capacity_blocks_for_one_refill_interval(clock):
    bucket = TokenBucket(10, 2, clock=clock, sleep=clock.sleep)
    for _ in range(10):
        await bucket.acquire()

    await bucket.acquire()
    # refill_rate 2/s -> one token takes 0.5s
    assert clock.now - 1000.0 == pytest.approx(0.5, abs=1e-6)


async def test_idle_bucket_does_not_accrue_past_capacity(clock):
    bucket = TokenBucket(5, 1, clock=clock, sleep=clock.sleep)
    await bucket.acquire(5)
    clock.advance(3600)  # an idle hour
    assert bucket.available == pytest.approx(5.0)
    # The clamp is what stops the next burst blowing through the limit.
    assert bucket.try_acquire(5) is True
    assert bucket.try_acquire(1) is False


async def test_sustained_load_never_exceeds_rate_times_time_plus_capacity(clock):
    capacity, rate, window = 10.0, 4.0, 5.0
    bucket = TokenBucket(capacity, rate, clock=clock, sleep=clock.sleep)

    issued = 0
    start = clock.now
    while clock.now - start < window:
        await bucket.acquire()
        issued += 1

    elapsed = clock.now - start
    assert issued <= rate * elapsed + capacity + 1


async def test_concurrent_callers_never_over_issue():
    # Real time here on purpose: the point is genuine interleaving at awaits.
    bucket = TokenBucket(10, 200)
    issued = 0

    async def worker():
        nonlocal issued
        await bucket.acquire()
        issued += 1

    start = asyncio.get_running_loop().time()
    await asyncio.gather(*(worker() for _ in range(50)))
    elapsed = asyncio.get_running_loop().time() - start

    assert issued == 50
    assert 200 * elapsed + 10 + 1 >= 50


async def test_lock_is_not_held_across_sleep():
    """If the lock were held across the sleep, waiters would serialize.

    With capacity 1 and rate 100/s, ten waiters take ~0.09s when they share
    the refill and ~0.09s * 10 if each waits behind the previous one's sleep.
    """
    bucket = TokenBucket(1, 100)
    await bucket.acquire()

    start = asyncio.get_running_loop().time()
    await asyncio.gather(*(bucket.acquire() for _ in range(10)))
    elapsed = asyncio.get_running_loop().time() - start

    assert elapsed < 0.5


async def test_try_acquire_is_all_or_nothing(clock):
    bucket = TokenBucket(3, 1, clock=clock, sleep=clock.sleep)
    assert bucket.try_acquire(2) is True
    assert bucket.try_acquire(2) is False
    # The failed attempt must not have deducted anything.
    assert bucket.available == pytest.approx(1.0)
    assert bucket.try_acquire(1) is True


async def test_request_larger_than_capacity_raises(clock):
    bucket = TokenBucket(5, 1, clock=clock, sleep=clock.sleep)
    with pytest.raises(ValueError):
        await bucket.acquire(6)
    assert bucket.try_acquire(6) is False


async def test_rate_limiter_never_touches_wall_clock(no_wall_clock, clock):
    bucket = TokenBucket(2, 1, clock=clock, sleep=clock.sleep)
    await bucket.acquire()
    await bucket.acquire()
    await bucket.acquire()  # forces the refill path
    assert bucket.available >= 0


def test_invalid_construction():
    with pytest.raises(ValueError):
        TokenBucket(0, 1)
    with pytest.raises(ValueError):
        TokenBucket(1, 0)


async def test_two_dimensions_are_independent(clock):
    limiter = ProviderLimiter(60, 6000, clock=clock, sleep=clock.sleep)
    # Tiny prompts: RPM binds first.
    for _ in range(60):
        await limiter.acquire(1, 1)
    assert limiter.requests.available == pytest.approx(0.0)
    assert limiter.tokens.available > 5000

    limiter2 = ProviderLimiter(60, 600, clock=clock, sleep=clock.sleep)
    # Huge prompts: TPM binds first.
    for _ in range(6):
        await limiter2.acquire(1, 100)
    assert limiter2.requests.available > 50
    assert limiter2.tokens.available == pytest.approx(0.0)


async def test_headroom_reports_the_binding_dimension(clock):
    limiter = ProviderLimiter(60, 6000, clock=clock, sleep=clock.sleep)
    await limiter.acquire(30, 0)
    assert limiter.headroom == pytest.approx(0.5)


# ----------------------------------------------------------------------
# TokenBucket.sync
# ----------------------------------------------------------------------


async def test_sync_lowers_local_count_to_server_value(clock):
    bucket = TokenBucket(10, 1, clock=clock, sleep=clock.sleep)
    assert bucket.available == pytest.approx(10.0)
    bucket.sync(remaining=3)
    assert bucket.available == pytest.approx(3.0)


async def test_sync_never_raises_local_count_above_local_value(clock):
    bucket = TokenBucket(10, 1, clock=clock, sleep=clock.sleep)
    await bucket.acquire(6)  # local: 4 remaining
    assert bucket.available == pytest.approx(4.0)
    # Server claims more room than we locally believe -- must be ignored,
    # per the TRAP in TokenBucket.sync: trusting a larger number would hand
    # back tokens we've already spent.
    bucket.sync(remaining=9)
    assert bucket.available == pytest.approx(4.0)


async def test_sync_clamps_to_zero_on_negative_remaining(clock):
    bucket = TokenBucket(10, 1, clock=clock, sleep=clock.sleep)
    bucket.sync(remaining=-5)
    assert bucket.available == pytest.approx(0.0)


async def test_sync_clamps_to_capacity_even_if_server_says_more(clock):
    bucket = TokenBucket(10, 1, clock=clock, sleep=clock.sleep)
    # A misconfigured capacity, or a server number that exceeds it, must
    # never leave _tokens > capacity -- that would break `headroom`.
    bucket.sync(remaining=999)
    assert bucket.available == pytest.approx(10.0)


async def test_sync_accepts_and_ignores_reset_in_for_accounting(clock):
    bucket = TokenBucket(10, 1, clock=clock, sleep=clock.sleep)
    bucket.sync(remaining=2, reset_in=45.0)
    assert bucket.available == pytest.approx(2.0)
    # reset_in must not fabricate elapsed time / snap the refill forward.
    clock.advance(1.0)
    assert bucket.available == pytest.approx(3.0)  # plain 1s * 1 tok/s refill


async def test_sync_refills_before_comparing(clock):
    bucket = TokenBucket(10, 2, clock=clock, sleep=clock.sleep)
    await bucket.acquire(10)  # empty
    clock.advance(2.0)  # local should now be back to 4 tokens
    # Server saw a stale remaining=1 from before our local refill caught up;
    # min() against the *refreshed* local value (4) keeps the higher, more
    # current local number rather than a comparison against the pre-refill
    # value of 0.
    bucket.sync(remaining=1)
    assert bucket.available == pytest.approx(1.0)


# ----------------------------------------------------------------------
# duration / number parsing
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("7.66s", 7.66),
        ("2m59.56s", 179.56),
        ("1h2m3s", 3723.0),
        ("120ms", 0.12),
        ("0s", 0.0),
        ("42", 42.0),
        ("3.5", 3.5),
        ("1h", 3600.0),
        ("5m", 300.0),
    ],
)
def test_parse_duration_formats(raw, expected):
    assert _parse_duration(raw) == pytest.approx(expected)


@pytest.mark.parametrize("raw", ["", "not-a-duration", "sm", "h", None, "--"])
def test_parse_duration_rejects_junk(raw):
    if raw is None:
        assert _parse_duration(raw) is None  # type: ignore[arg-type]
    else:
        assert _parse_duration(raw) is None


@pytest.mark.parametrize("raw", ["42", "-3", "0", "7.5"])
def test_parse_float_accepts_plain_numbers(raw):
    assert _parse_float(raw) == pytest.approx(float(raw))


@pytest.mark.parametrize("raw", ["", "abc", None, "7.66s"])
def test_parse_float_rejects_non_numbers(raw):
    if raw is None:
        assert _parse_float(raw) is None  # type: ignore[arg-type]
    else:
        assert _parse_float(raw) is None


# ----------------------------------------------------------------------
# ProviderLimiter.sync_from_headers
# ----------------------------------------------------------------------


async def test_sync_from_headers_openai_style(clock):
    limiter = ProviderLimiter(60, 6000, clock=clock, sleep=clock.sleep)
    limiter.sync_from_headers(
        {
            "x-ratelimit-remaining-requests": "10",
            "x-ratelimit-remaining-tokens": "500",
            "x-ratelimit-reset-requests": "5s",
            "x-ratelimit-reset-tokens": "1m2s",
        }
    )
    assert limiter.requests.available == pytest.approx(10.0)
    assert limiter.tokens.available == pytest.approx(500.0)


async def test_sync_from_headers_case_insensitive(clock):
    limiter = ProviderLimiter(60, 6000, clock=clock, sleep=clock.sleep)
    limiter.sync_from_headers(
        {
            "X-RateLimit-Remaining-Requests": "7",
            "X-RATELIMIT-REMAINING-TOKENS": "1234",
        }
    )
    assert limiter.requests.available == pytest.approx(7.0)
    assert limiter.tokens.available == pytest.approx(1234.0)


async def test_sync_from_headers_generic_fallback(clock):
    limiter = ProviderLimiter(60, 6000, clock=clock, sleep=clock.sleep)
    # A provider that doesn't distinguish requests vs tokens: both buckets
    # get corrected from the same unqualified header, per spec.
    limiter.sync_from_headers({"x-ratelimit-remaining": "3", "x-ratelimit-reset": "10s"})
    assert limiter.requests.available == pytest.approx(3.0)
    assert limiter.tokens.available == pytest.approx(3.0)


async def test_sync_from_headers_prefers_specific_over_generic(clock):
    limiter = ProviderLimiter(60, 6000, clock=clock, sleep=clock.sleep)
    limiter.sync_from_headers(
        {
            "x-ratelimit-remaining-requests": "9",
            "x-ratelimit-remaining": "1",
        }
    )
    assert limiter.requests.available == pytest.approx(9.0)


async def test_sync_from_headers_missing_headers_are_ignored(clock):
    limiter = ProviderLimiter(60, 6000, clock=clock, sleep=clock.sleep)
    limiter.sync_from_headers({"content-type": "application/json"})
    assert limiter.requests.available == pytest.approx(60.0)
    assert limiter.tokens.available == pytest.approx(6000.0)


async def test_sync_from_headers_junk_values_are_ignored(clock):
    limiter = ProviderLimiter(60, 6000, clock=clock, sleep=clock.sleep)
    limiter.sync_from_headers(
        {
            "x-ratelimit-remaining-requests": "not-a-number",
            "x-ratelimit-remaining-tokens": "",
            "x-ratelimit-reset-requests": "garbage-duration",
        }
    )
    assert limiter.requests.available == pytest.approx(60.0)
    assert limiter.tokens.available == pytest.approx(6000.0)


async def test_sync_from_headers_bad_reset_does_not_block_remaining_sync(clock):
    limiter = ProviderLimiter(60, 6000, clock=clock, sleep=clock.sleep)
    limiter.sync_from_headers(
        {
            "x-ratelimit-remaining-requests": "12",
            "x-ratelimit-reset-requests": "not-a-duration",
        }
    )
    # The remaining value is still valid and must still apply even though
    # the paired reset value could not be parsed.
    assert limiter.requests.available == pytest.approx(12.0)


async def test_sync_from_headers_never_raises_on_hostile_input(clock):
    limiter = ProviderLimiter(60, 6000, clock=clock, sleep=clock.sleep)

    class ExplodingMapping:
        def items(self):
            raise RuntimeError("boom")

    # None of these should raise.
    limiter.sync_from_headers({})
    limiter.sync_from_headers({"x-ratelimit-remaining-requests": None})  # type: ignore[dict-item]
    limiter.sync_from_headers({123: "5"})  # type: ignore[dict-item]
    limiter.sync_from_headers(ExplodingMapping())  # type: ignore[arg-type]
    limiter.sync_from_headers(
        {"x-ratelimit-remaining-requests": object()}  # type: ignore[dict-item]
    )
