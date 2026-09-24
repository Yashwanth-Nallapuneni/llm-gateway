"""Token-bucket rate limiting."""

from __future__ import annotations

import asyncio
import re
import time
from collections.abc import Awaitable, Callable, Mapping


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
        # Kept aside so adaptive mode (throttle()/recover() below) always
        # has a fixed floor and ceiling to measure against, even after
        # refill_rate itself has drifted up or down.
        self._configured_rate = self.refill_rate

        # monotonic() only ever moves forward, unlike time.time(), which can
        # jump backward on an NTP sync -- that would make elapsed time go
        # negative and lock the limiter up.
        self._clock = clock or time.monotonic
        self._sleep = sleep or asyncio.sleep

        # Start full: a freshly started process hasn't used any of the
        # provider's budget yet, so it's entitled to the full burst.
        self._tokens = self.capacity
        self._last_refill = self._clock()

        # acquire() awaits while holding tokens in hand, so two coroutines
        # could both see "enough tokens" and both deduct without this lock,
        # issuing twice the allowed traffic.
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _refill(self) -> None:
        """Bring the bucket up to date. Caller must hold the lock."""
        now = self._clock()
        elapsed = now - self._last_refill

        # Computes what would have accumulated since the last call instead
        # of running a background timer task, which would need its own
        # shutdown handling and could drift under event-loop congestion.
        # The min() clamp matters: without it, an idle bucket would accrue
        # unbounded tokens and the next burst would blow through the
        # provider's real limit.
        self._tokens = min(self.capacity, self._tokens + elapsed * self.refill_rate)

        self._last_refill = now

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    async def acquire(self, tokens: float = 1.0) -> None:
        """Block until `tokens` available, then deduct.

        Must not busy-wait. Must be correct under concurrent callers.
        """
        # A request bigger than the bucket's capacity can never be
        # satisfied, since _refill() caps _tokens at capacity -- fail
        # loudly here instead of hanging the caller forever.
        if tokens > self.capacity:
            raise ValueError(
                f"requested {tokens} tokens but bucket capacity is {self.capacity}"
            )

        # Loops and re-checks after every sleep, rather than sleeping once
        # and taking the tokens on trust -- another coroutine could grab the
        # same tokens while this one was asleep.
        while True:
            async with self._lock:
                self._refill()
                if self._tokens >= tokens:
                    self._tokens -= tokens
                    return
                deficit = tokens - self._tokens
                # Sleep exactly as long as the missing tokens take to
                # refill, rather than polling on a fixed tick.
                wait = deficit / self.refill_rate

            # This sleep happens outside the lock on purpose. Holding the
            # lock while asleep would stop every other waiter from even
            # checking the bucket until this one wakes up, serializing
            # everyone behind a single sleep instead of letting them overlap.
            await self._sleep(wait)

    def try_acquire(self, tokens: float = 1.0) -> bool:
        """Non-blocking. Deduct and return True, or return False untouched."""
        # No lock and no await here, since this method never yields control,
        # so the event loop can't interleave another coroutine in the
        # middle of it. This only holds for a single event loop; sharing
        # one bucket across real OS threads would need a threading.Lock.
        if tokens > self.capacity:
            return False
        self._refill()
        if self._tokens >= tokens:
            self._tokens -= tokens
            return True
        return False

    def sync(self, remaining: float, reset_in: float | None = None) -> None:
        """Reconcile the local count with a provider's `x-ratelimit-remaining`.

        `remaining` is the provider's own count of what is left in its
        window, as of the instant it built the response we just received.
        `reset_in`, if given, is how many seconds the provider says are left
        until that window resets; see below for why it is accepted but not
        used to adjust refill timing.

        The local bucket is only ever a model, seeded from a configured
        limit and updated by guessing at elapsed time -- it can drift from
        what the provider actually enforces. This method folds the
        provider's own count back in to correct that drift.
        """
        # Bring local accounting up to date before comparing against the
        # server's number, so the comparison isn't against stale state.
        self._refill()

        # Take the minimum, never the maximum. The server's number is
        # already a little stale by the time we see it, since we may have
        # spent more tokens after it was generated but before its response
        # arrived. Trusting a larger server number would hand back tokens
        # already spent locally. Taking the minimum means sync() can only
        # make the bucket more conservative, never less, which is what
        # makes it safe to call on every response.
        self._tokens = min(self._tokens, remaining)

        # Clamp to [0, capacity] defensively, in case of a provider bug or
        # two processes sharing a key with different configured limits.
        # Callers like the router's headroom calculation assume tokens
        # always fall in this range.
        self._tokens = max(0.0, min(self.capacity, self._tokens))

        # `reset_in` is accepted but deliberately not folded into `_tokens`
        # or `_last_refill`. This bucket models a continuous refill, while
        # the provider's `reset_in` describes a discrete window that jumps
        # back to full capacity at one instant -- the two models disagree
        # about the shape of the curve in between, so there's no clean way
        # to combine them without guessing at the provider's own algorithm.
        # It's kept as a parameter anyway because `sync_from_headers`
        # already has to parse it out of the response, and dropping it
        # silently here would look like the parsing was the bug.

    def throttle(self) -> None:
        """AIMD backoff: halve the refill rate after a rejection (a 429).

        AIMD (additive increase / multiplicative decrease) is the same idea
        TCP congestion control uses: back off fast when you get pushback,
        then creep back up slowly. Cutting in half means a provider that is
        genuinely overloaded gets a big, quick relief; a floor of 10% of
        the configured rate stops repeated 429s from ever driving the rate
        to (or toward) zero and stalling the provider forever.
        """
        # Credit the tokens earned so far at the old rate before changing it.
        self._refill()
        self.refill_rate = max(self.refill_rate * 0.5, self._configured_rate * 0.1)

    def recover(self) -> None:
        """AIMD recovery: nudge the refill rate back up after each success.

        The increase is additive (a small fixed step, 1% of the configured
        rate) rather than multiplicative, so recovery is deliberately much
        slower than the backoff in throttle() -- that asymmetry is the
        point of AIMD. Never grows past the originally configured rate.
        """
        self._refill()
        self.refill_rate = min(
            self.refill_rate + self._configured_rate * 0.01, self._configured_rate
        )

    @property
    def available(self) -> float:
        """Current token count, refreshed. Used by the router for headroom."""
        self._refill()
        return self._tokens

    @property
    def headroom(self) -> float:
        """Available tokens as a fraction of capacity, in [0, 1].

        The router compares buckets of different sizes (600 rpm against
        150k tpm). Raw token counts are not comparable across dimensions;
        a normalized fraction is.
        """
        return self.available / self.capacity

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"TokenBucket(capacity={self.capacity}, rate={self.refill_rate}, "
            f"available={self._tokens:.2f})"
        )


# ----------------------------------------------------------------------
# header parsing for sync_from_headers
# ----------------------------------------------------------------------

# Groq and some other gateways send reset windows as Go-style duration
# strings ("7.66s", "2m59.56s", "1h2m3s", "120ms") instead of a bare number
# of seconds like OpenAI does. This regex matches any ordered combination of
# hours/minutes/seconds/milliseconds, all optional. The `(?!s)` after the
# minutes group stops "120ms" from being misread, since "m" and "ms" share
# a prefix.
_DURATION_RE = re.compile(
    r"^(?:(?P<h>\d+(?:\.\d+)?)h)?"
    r"(?:(?P<m>\d+(?:\.\d+)?)m(?!s))?"
    r"(?:(?P<s>\d+(?:\.\d+)?)s)?"
    r"(?:(?P<ms>\d+(?:\.\d+)?)ms)?$"
)


def _parse_float(value: str) -> float | None:
    """Best-effort float parse. Returns None instead of raising.

    A header value comes from the network and should never be trusted to
    parse cleanly; a raised exception here must not take down a response
    that otherwise succeeded.
    """
    try:
        return float(value.strip())
    except (ValueError, AttributeError):
        return None


def _parse_duration(value: str) -> float | None:
    """Parse a plain number of seconds OR a Go-style duration string.

    Tries a bare float first, since a plain numeric value like "120" is the
    common case (OpenAI-compatible APIs); this also avoids a numeric string
    being misread by the duration regex as "no unit, so zero".
    """
    try:
        stripped = value.strip()
    except AttributeError:
        return None
    try:
        return float(stripped)
    except ValueError:
        pass

    match = _DURATION_RE.fullmatch(stripped)
    # Every group in the regex is optional, so it also matches an empty
    # string. Require at least one unit to have matched, or a blank/garbage
    # header would silently parse as "0 seconds".
    if not match or not any(match.groups()):
        return None

    hours = float(match.group("h") or 0.0)
    minutes = float(match.group("m") or 0.0)
    seconds = float(match.group("s") or 0.0)
    millis = float(match.group("ms") or 0.0)
    return hours * 3600.0 + minutes * 60.0 + seconds + millis / 1000.0


class ProviderLimiter:
    """The two independent buckets a provider needs.

    Request-rate and token-rate are independent limits. A workload of many
    tiny prompts hits RPM first; a workload of few huge prompts hits TPM
    first. One bucket cannot express both, so both are checked and both
    are deducted before dispatch.
    """

    def __init__(
        self,
        rpm_limit: float,
        tpm_limit: float,
        *,
        clock: Callable[[], float] | None = None,
        sleep: Callable[[float], Awaitable[None]] | None = None,
        adaptive: bool = False,
    ) -> None:
        # capacity == the per-minute limit, refill == limit/60 per second.
        # That reproduces the provider's own model: a full minute's budget
        # may be spent as a burst, but the sustained rate is the limit.
        self.requests = TokenBucket(rpm_limit, rpm_limit / 60.0, clock=clock, sleep=sleep)
        self.tokens = TokenBucket(tpm_limit, tpm_limit / 60.0, clock=clock, sleep=sleep)
        # Off by default so existing behaviour doesn't change. When on, the
        # request bucket's rate self-adjusts (see on_throttled/on_success
        # below) for providers that send no rate-limit headers to sync
        # from. This never conflicts with sync_from_headers, since that
        # corrects the token count while adaptive mode only adjusts rate.
        self.adaptive = adaptive

    async def acquire(self, n_requests: float = 1.0, n_tokens: float = 0.0) -> None:
        # Always acquiring the request bucket first, then the token bucket,
        # keeps the two buckets from deadlocking against each other.
        await self.requests.acquire(n_requests)
        if n_tokens > 0:
            # A batch can ask for more tokens than the bucket even holds.
            # Rather than fail it, wait for a full bucket and take it all.
            await self.tokens.acquire(min(n_tokens, self.tokens.capacity))

    def on_throttled(self) -> None:
        """Call this when the provider returns a 429. No-op unless adaptive."""
        if self.adaptive:
            self.requests.throttle()

    def on_success(self) -> None:
        """Call this after a successful call. No-op unless adaptive."""
        if self.adaptive:
            self.requests.recover()

    @property
    def headroom(self) -> float:
        """The binding constraint: a provider is only as free as its tightest
        dimension, so take the minimum rather than an average."""
        return min(self.requests.headroom, self.tokens.headroom)

    def sync_from_headers(self, headers: Mapping[str, str]) -> None:
        """Correct both buckets from a provider's `x-ratelimit-*` response headers.

        Recognises the common OpenAI-compatible spellings
        (`x-ratelimit-remaining-requests`, `x-ratelimit-remaining-tokens`,
        `x-ratelimit-reset-requests`, `x-ratelimit-reset-tokens`) plus the
        unqualified `x-ratelimit-remaining` / `x-ratelimit-reset` some
        gateways send when they don't distinguish the two dimensions.

        Header text comes straight from the network, in whatever casing or
        format the remote server likes, so this method's job is to turn
        that into either a valid call to `TokenBucket.sync` or nothing.
        """
        # Header names are case-insensitive per HTTP, but a plain dict (as
        # tests may pass) doesn't guarantee that the way an httpx/aiohttp
        # headers object does, so build a lowercased copy once up front.
        # The whole method must never raise, since it runs on the response
        # path of every call; a malformed headers argument should just mean
        # nothing gets synced, not a crash.
        try:
            lowered = {str(k).lower(): v for k, v in headers.items()}
        except Exception:
            return

        def first(*names: str) -> str | None:
            for name in names:
                value = lowered.get(name)
                if value is not None:
                    return value
            return None

        # Prefer the dimension-specific header, falling back to the
        # unqualified one some gateways send when they don't separate
        # RPM/TPM.
        req_remaining = first("x-ratelimit-remaining-requests", "x-ratelimit-remaining")
        req_reset = first("x-ratelimit-reset-requests", "x-ratelimit-reset")
        tok_remaining = first("x-ratelimit-remaining-tokens", "x-ratelimit-remaining")
        tok_reset = first("x-ratelimit-reset-tokens", "x-ratelimit-reset")

        try:
            if req_remaining is not None:
                remaining = _parse_float(req_remaining)
                if remaining is not None:
                    reset_in = (
                        _parse_duration(req_reset) if req_reset is not None else None
                    )
                    self.requests.sync(remaining, reset_in)

            if tok_remaining is not None:
                remaining = _parse_float(tok_remaining)
                if remaining is not None:
                    reset_in = (
                        _parse_duration(tok_reset) if tok_reset is not None else None
                    )
                    self.tokens.sync(remaining, reset_in)
        except Exception:
            # Any unparseable or unexpected value (including one that isn't
            # even a string) is ignored rather than allowed to crash the
            # response handler.
            return
