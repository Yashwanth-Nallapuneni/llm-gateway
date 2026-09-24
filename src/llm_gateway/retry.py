"""Retry classification and backoff."""

from __future__ import annotations

import asyncio
import random
from collections.abc import Iterable
from typing import Protocol

from .types import ProviderError


class _RandomLike(Protocol):
    """Structural type for the `rng or random` trick below.

    Both `random.Random` instances and the `random` module itself expose a
    module/instance-level `uniform(a, b) -> float`, which is all this file
    needs -- this Protocol lets mypy see through the union without pinning
    the injected RNG to a concrete class.
    """

    def uniform(self, a: float, b: float) -> float: ...


# Retryable means the same request sent again could plausibly succeed: the
# request itself was fine, the far end just failed to serve it this time.
RETRYABLE_STATUSES: frozenset[int] = frozenset({408, 425, 429, 500, 502, 503, 504})

# These mean the request itself is wrong (bad body, bad key, bad schema), so
# retrying it is pointless and just adds traffic against a guaranteed failure.
NON_RETRYABLE_STATUSES: frozenset[int] = frozenset({400, 401, 403, 404, 405, 409, 422})

# Transport-level failures with no HTTP status at all.
RETRYABLE_EXCEPTIONS: tuple[type[BaseException], ...] = (
    asyncio.TimeoutError,
    TimeoutError,
    ConnectionError,
    ConnectionResetError,
    OSError,
)


class RetryPolicy:
    """Exponential backoff with full jitter, and a Retry-After override."""

    def __init__(
        self,
        max_attempts: int = 5,
        base_delay: float = 0.5,
        max_delay: float = 60.0,
        jitter: bool = True,
        *,
        rng: random.Random | None = None,
    ) -> None:
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.jitter = jitter
        # Injectable only so tests can seed it; production uses the global RNG.
        self._rng: _RandomLike = rng or random

    def should_retry(self, exc: Exception, attempt: int) -> bool:
        """Retryable: 429, 500, 502, 503, 504, timeouts, conn errors.

        Not retryable: 400, 401, 403, 404, 422 -- identical failure on retry.
        """
        # Check the attempt budget separately from whether the error is
        # retryable, so the two questions never get conflated in the logs.
        # `attempt` is 0-indexed, so attempt 4 of max_attempts=5 is the last.
        if attempt >= self.max_attempts - 1:
            return False

        if isinstance(exc, ProviderError):
            status = exc.status
            # An unknown status falls back to the class's usual behavior
            # (5xx retryable, 4xx not), since providers invent new codes and
            # a strict allowlist would silently stop retrying any new one.
            if status is None:
                return True
            if status in NON_RETRYABLE_STATUSES:
                return False
            if status in RETRYABLE_STATUSES:
                return True
            return status >= 500

        # A single isinstance against the tuple, rather than separate checks,
        # avoids overlap bugs since ConnectionError is an OSError subclass.
        return isinstance(exc, RETRYABLE_EXCEPTIONS)

    def delay_for(self, attempt: int, retry_after: float | None = None) -> float:
        """Seconds to wait before attempt N (0-indexed)."""
        # Doubling the delay each attempt (rather than a fixed wait) backs
        # off fast enough for an overloaded service to actually drain.
        # `2.0 ** attempt` uses a float base so the type stays float instead
        # of the `Any` that int's power operator produces. The exponent is
        # capped at 64 so it can't overflow for callers with a very large
        # max_attempts; any attempt past that already produces a delay far
        # beyond max_delay anyway.
        raw = self.base_delay * (2.0 ** min(attempt, 64))

        # Cap before applying jitter, not after, so jitter samples from a
        # sane range instead of one that could reach hours.
        delay = min(raw, self.max_delay)

        if self.jitter:
            # Without jitter, many clients that get throttled at the same
            # instant compute the same delay and retry in lockstep, which
            # reproduces the overload that caused the failure (a "thundering
            # herd"). Full jitter (sampling uniformly between 0 and delay)
            # is used here because it is stateless and simple to test; it
            # trades some variance in wait time for that simplicity.
            delay = self._rng.uniform(0, delay)

        if retry_after is not None:
            # A provider's Retry-After header is authoritative, so take the
            # larger of it and our own backoff (max, never min) -- returning
            # too early just earns another 429.
            delay = max(float(retry_after), delay)

        return delay

    def retry_after_from(self, exc: Exception) -> float | None:
        """Pull a Retry-After value off an exception, if it carries one."""
        return getattr(exc, "retry_after", None)

    def delays(self, n: int) -> Iterable[float]:  # pragma: no cover - convenience
        return (self.delay_for(i) for i in range(n))
