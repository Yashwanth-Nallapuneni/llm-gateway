"""Shared fixtures. The important one is the fake clock.

Rate-limiter tests must be able to express "sixty seconds pass" without the
suite taking sixty seconds. Every component that reads time takes an
injectable `clock` and `sleep` for exactly this reason.
"""

from __future__ import annotations

import asyncio

import pytest


class FakeClock:
    """Virtual time driven by the coroutines that sleep on it.

    `sleep()` does not wait; it registers a deadline and then lets the
    earliest-deadline sleeper pull virtual time forward to its own deadline.
    That keeps multiple concurrent sleepers in the right relative order while
    the whole test still runs in microseconds.
    """

    def __init__(self, start: float = 1000.0) -> None:
        self.now = start
        self._deadlines: list[float] = []
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds

    async def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        if seconds <= 0:
            await asyncio.sleep(0)
            return
        deadline = self.now + seconds
        self._deadlines.append(deadline)
        try:
            while True:
                await asyncio.sleep(0)
                if self.now >= deadline:
                    return
                if deadline <= min(self._deadlines):
                    self.now = deadline
                    return
        finally:
            self._deadlines.remove(deadline)


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def no_wall_clock(monkeypatch):
    """Fail the test if anything calls time.time().

    Wall clock in a rate limiter is a real bug (NTP steps it backwards), so
    the suite asserts against it rather than trusting a code review.
    """
    import time

    def boom(*_a, **_kw):
        raise AssertionError("time.time() must not be used in the rate limiter")

    monkeypatch.setattr(time, "time", boom)
    return boom
