from __future__ import annotations

import asyncio

import pytest

from llm_gateway.rate_limit import ProviderLimiter, TokenBucket


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
    assert 50 <= 200 * elapsed + 10 + 1


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
