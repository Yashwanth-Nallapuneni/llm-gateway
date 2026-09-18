"""Token-bucket rate limiting.

Read `rate_limit.EXPLAIN.md` before this file.

Every non-obvious decision below carries a WHY / ALT / TRAP / ASK tag. The
tags are the point of this module: the code is ~60 lines and the reasoning
behind it is what an interviewer actually probes.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable


class TokenBucket:
    """Capacity `C` tokens, refilled at `refill_rate` tokens/sec.

    Allows a burst of up to `C` while bounding the long-run average to
    `refill_rate`. O(1) memory and O(1) time per acquisition.
    """

    def __init__(
        self,
        capacity: float,
        refill_rate: float,
        *,
        clock: Callable[[], float] | None = None,
        sleep: Callable[[float], Awaitable[None]] | None = None,
    ) -> None:
        """capacity: burst allowance. refill_rate: sustained tokens/sec.

        `clock` and `sleep` exist only so tests can drive a fake clock; the
        defaults are the real thing.
        """
        if capacity <= 0:
            raise ValueError("capacity must be > 0")
        if refill_rate <= 0:
            raise ValueError("refill_rate must be > 0")

        self.capacity = float(capacity)
        self.refill_rate = float(refill_rate)

        # WHY: monotonic() is immune to system clock adjustments -- NTP sync,
        #      DST, an operator running `date -s`. It only ever moves forward.
        # ALT: time.time() is wall-clock. If NTP steps the clock backward
        #      mid-run, `elapsed` goes negative, the bucket refills by a
        #      negative amount, and the limiter silently locks up until wall
        #      time catches back up to where it was.
        # ASK: Why monotonic() instead of time.time() in a rate limiter?
        self._clock = clock or time.monotonic
        self._sleep = sleep or asyncio.sleep

        # WHY: start full. A process that has just booted has consumed none of
        #      the provider's budget, so it is entitled to the full burst.
        # ALT: start empty and make callers wait C/r seconds for no reason.
        self._tokens = self.capacity
        self._last_refill = self._clock()

        # WHY: the read-modify-write of (_tokens, _last_refill) is not atomic.
        #      Between reading _tokens and writing it back there is an await
        #      point in acquire(), so two coroutines can both observe "enough
        #      tokens" and both deduct, issuing 2x the allowed traffic.
        # ALT: no lock at all "because asyncio is single-threaded" -- true but
        #      irrelevant: single-threaded does not mean uninterrupted, it
        #      means interrupted only at awaits, and this method has one.
        # ASK: Does single-threaded asyncio need locks? When?
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _refill(self) -> None:
        """Bring the bucket up to date. Caller must hold the lock."""
        now = self._clock()
        elapsed = now - self._last_refill

        # WHY: lazy refill -- compute what *would* have accumulated since the
        #      last call, rather than a background task ticking every N ms.
        # ALT: a background asyncio task adding tokens on a timer. Costs one
        #      task per bucket, drifts under event-loop congestion, and needs
        #      explicit shutdown handling or it leaks on every gateway close.
        # ASK: How do you refill a token bucket without a background timer?
        self._tokens = min(self.capacity, self._tokens + elapsed * self.refill_rate)

        # TRAP: the min() clamp above is load-bearing. Without it an idle
        #       bucket accrues unbounded tokens, and the first burst after a
        #       quiet hour blows straight through the provider's limit -- the
        #       exact 429 storm the limiter exists to prevent.
        # ASK: What breaks if you drop the min(capacity, ...) clamp?

        self._last_refill = now

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    async def acquire(self, tokens: float = 1.0) -> None:
        """Block until `tokens` available, then deduct.

        Must not busy-wait. Must be correct under concurrent callers.
        """
        # TRAP: a request larger than the bucket can never be satisfied -- the
        #       clamp in _refill() caps _tokens at capacity, so the loop below
        #       would sleep forever. Fail loudly at the boundary instead of
        #       hanging a coroutine with no traceback.
        # ASK: What happens if someone asks for more tokens than capacity?
        if tokens > self.capacity:
            raise ValueError(
                f"requested {tokens} tokens but bucket capacity is {self.capacity}"
            )

        # WHY: a loop, not a single sleep-then-take. Between computing the wait
        #      and waking up, another coroutine can consume the very tokens we
        #      were waiting for. Re-check after every sleep.
        # TRAP: sleeping for `deficit / refill_rate` and then deducting without
        #       re-checking is the classic over-issue bug: N waiters all wake
        #       up "entitled" to the same tokens and all deduct.
        # ASK: Why does the sleep sit inside a re-check loop?
        while True:
            async with self._lock:
                self._refill()
                if self._tokens >= tokens:
                    self._tokens -= tokens
                    return
                deficit = tokens - self._tokens
                # WHY: sleep exactly as long as the missing tokens take to
                #      refill. Precise, and no CPU burned in the meantime.
                # ALT: poll on a fixed 10ms tick -- busy-waits, adds up to 10ms
                #      of latency per acquisition, and scales badly with waiters.
                wait = deficit / self.refill_rate

            # TRAP: this await is deliberately OUTSIDE the `async with`. Holding
            #       the lock across a sleep serializes every waiter behind the
            #       first one: waiter #2 cannot even *check* the bucket until
            #       waiter #1 has slept its full duration, so throughput
            #       collapses to one acquisition per sleep instead of one per
            #       refill interval. This release/reacquire boundary is the
            #       subtlest bug in the module.
            # ASK: Why must the lock not be held across `await asyncio.sleep()`?
            await self._sleep(wait)

    def try_acquire(self, tokens: float = 1.0) -> bool:
        """Non-blocking. Deduct and return True, or return False untouched."""
        # WHY: no lock and no await. This function contains no await point, so
        #      the event loop cannot interleave another coroutine in the middle
        #      of it -- the read-modify-write is already atomic with respect to
        #      other coroutines on this loop.
        # ALT: making it `async` and taking the lock. That would make it
        #      awaitable, which is exactly wrong for a routing hot path that
        #      wants to *peek* at headroom without yielding control.
        # TRAP: this reasoning only holds for a single event loop. Sharing one
        #       bucket across threads needs a threading.Lock instead.
        # ASK: Why does try_acquire not need the lock but acquire does?
        if tokens > self.capacity:
            return False
        self._refill()
        if self._tokens >= tokens:
            self._tokens -= tokens
            return True
        return False

    @property
    def available(self) -> float:
        """Current token count, refreshed. Used by the router for headroom."""
        self._refill()
        return self._tokens

    @property
    def headroom(self) -> float:
        """Available tokens as a fraction of capacity, in [0, 1].

        WHY: the router compares buckets of different sizes (600 rpm against
             150k tpm). Raw token counts are not comparable across dimensions;
             a normalized fraction is.
        """
        return self.available / self.capacity

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"TokenBucket(capacity={self.capacity}, rate={self.refill_rate}, "
            f"available={self._tokens:.2f})"
        )


class ProviderLimiter:
    """The two independent buckets a provider needs.

    WHY: request-rate and token-rate are independent limits. A workload of many
         tiny prompts hits RPM first; a workload of few huge prompts hits TPM
         first. One bucket cannot express both, so both are checked and both
         are deducted before dispatch.
    ASK: Why two buckets per provider instead of one?
    """

    def __init__(
        self,
        rpm_limit: float,
        tpm_limit: float,
        *,
        clock: Callable[[], float] | None = None,
        sleep: Callable[[float], Awaitable[None]] | None = None,
    ) -> None:
        # WHY: capacity == the per-minute limit, refill == limit/60 per second.
        #      That reproduces the provider's own model: a full minute's budget
        #      may be spent as a burst, but the sustained rate is the limit.
        self.requests = TokenBucket(rpm_limit, rpm_limit / 60.0, clock=clock, sleep=sleep)
        self.tokens = TokenBucket(tpm_limit, tpm_limit / 60.0, clock=clock, sleep=sleep)

    async def acquire(self, n_requests: float = 1.0, n_tokens: float = 0.0) -> None:
        # TRAP: acquire the request bucket first, then the token bucket. Doing
        #       it in a consistent order everywhere is what keeps two buckets
        #       from deadlocking against each other under contention.
        await self.requests.acquire(n_requests)
        if n_tokens > 0:
            await self.tokens.acquire(min(n_tokens, self.tokens.capacity))

    @property
    def headroom(self) -> float:
        """The binding constraint: a provider is only as free as its tightest
        dimension, so take the minimum rather than an average."""
        return min(self.requests.headroom, self.tokens.headroom)
