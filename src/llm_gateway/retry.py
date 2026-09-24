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


# Retryable means "the same request, sent again, could plausibly succeed".
# These are server-side or transport-side conditions: the request itself
# was fine, the far end was momentarily unable to serve it.
RETRYABLE_STATUSES: frozenset[int] = frozenset({408, 425, 429, 500, 502, 503, 504})

# These are contract violations instead. A 400 means the request body is
# wrong; a 401 means the key is wrong; a 422 means the schema is wrong.
# Nothing about waiting and sending the identical bytes again changes any
# of that. Retrying them is worse than useless -- it multiplies your
# traffic against a guaranteed failure and can trip the provider's abuse
# limits, turning a clean 400 into a 429 on every other request you send.
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
        # The attempt budget is checked first and separately. Whether an
        # error *is* retryable and whether we still have budget *to* retry
        # are different questions; collapsing them makes the logs lie
        # ("gave up: non-retryable" when really you ran out of attempts).
        # `attempt` is 0-indexed, so attempt 4 of max_attempts=5 is the last.
        if attempt >= self.max_attempts - 1:
            return False

        if isinstance(exc, ProviderError):
            status = exc.status
            # An unknown 5xx is treated as retryable and an unknown 4xx as
            # not. Default to the class's behaviour rather than refusing
            # to decide -- providers invent status codes constantly. An
            # explicit allowlist only would silently give up on any 5xx
            # the provider adds after you shipped.
            if status is None:
                return True
            if status in NON_RETRYABLE_STATUSES:
                return False
            if status in RETRYABLE_STATUSES:
                return True
            return status >= 500

        # Order matters here. ConnectionError is a subclass of OSError, and
        # asyncio.TimeoutError is TimeoutError on 3.11+ -- a single
        # isinstance against the tuple handles both without the
        # overlapping-except-clause bug you get writing them separately.
        return isinstance(exc, RETRYABLE_EXCEPTIONS)

    def delay_for(self, attempt: int, retry_after: float | None = None) -> float:
        """Seconds to wait before attempt N (0-indexed)."""
        # Exponential, not linear. Under a linear backoff a persistent
        # overload gets hammered at a nearly constant rate; doubling backs
        # the fleet off fast enough for the far end to actually drain. A
        # fixed delay would be simpler, but with N clients it converges on
        # a constant-rate DDoS against an already-struggling service.
        # `2.0 ** attempt`, not `2 ** attempt`: typeshed types int.__pow__ as
        # returning `Any` (it must cover negative exponents, which escape int),
        # which would otherwise leak Any through `raw` and `delay` below. The
        # float base gives the identical value with a real `float` type.
        raw = self.base_delay * (2.0**attempt)

        # The cap must be applied BEFORE jitter, not after. 2**attempt
        # overflows into minutes-then-hours by attempt 12; capping first
        # bounds the worst case, and it also means jitter samples from a
        # sane range instead of from [0, 4 hours].
        delay = min(raw, self.max_delay)

        if self.jitter:
            # Jitter is not optional. Without it, 500 clients that received
            # 429 at the same instant compute the same delay and retry at
            # the same instant. The retry storm reproduces exactly the
            # overload that caused the failure -- the thundering herd.
            # Equal jitter, delay/2 + uniform(0, delay/2), guarantees a
            # minimum wait and has lower variance, but leaves half the
            # delay synchronized across clients. Decorrelated jitter
            # (sleep = uniform(base, prev*3)) spreads best but needs the
            # previous delay as state, which makes delay_for() impure and
            # much harder to test. Full jitter is chosen here: stateless,
            # pure, and it minimizes collision probability. The price is
            # high variance -- an unlucky client may retry almost
            # immediately -- which is acceptable because the cap and the
            # attempt budget bound the damage.
            delay = self._rng.uniform(0, delay)

        if retry_after is not None:
            # Retry-After dominates the computation -- use max(), never
            # min(). The header is the provider telling you exactly when
            # your quota resets; your backoff is a guess. Taking the
            # smaller value means returning while still throttled, which
            # earns another 429 and, on many providers, extends the
            # penalty window. Returning retry_after verbatim instead would
            # ignore your own escalation on repeated failures; max() keeps
            # whichever is more conservative.
            delay = max(float(retry_after), delay)

        return delay

    def retry_after_from(self, exc: Exception) -> float | None:
        """Pull a Retry-After value off an exception, if it carries one."""
        return getattr(exc, "retry_after", None)

    def delays(self, n: int) -> Iterable[float]:  # pragma: no cover - convenience
        return (self.delay_for(i) for i in range(n))
