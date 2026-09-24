"""Per-provider circuit breaker.

Normal comments -- the state machine is small and the interesting part is
*why it exists*, which is documented on the class.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from enum import StrEnum

from .types import CircuitOpenError


class State(StrEnum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """Fails fast for a provider that is comprehensively down.

    Without a breaker, a provider that is hard-down still absorbs the full
    retry budget of every single request: five attempts with exponential
    backoff each, per request, all guaranteed to fail. That adds tens of
    seconds of latency and holds concurrency slots open for work that cannot
    succeed. The breaker converts a slow, expensive failure into an instant
    one, which is what lets the router move traffic elsewhere in time to
    matter.
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        *,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self._clock = clock or time.monotonic

        self._state = State.CLOSED
        # Consecutive, not cumulative: one success resets the count. A provider
        # with a 1% error rate should never trip; five failures in a row is a
        # qualitatively different signal from five failures out of a thousand.
        self._consecutive_failures = 0
        self._opened_at = 0.0
        # Guards the half-open probe so exactly one caller gets through.
        self._probe_in_flight = False

    @property
    def state(self) -> State:
        # Transition OPEN -> HALF_OPEN lazily on read, for the same reason the
        # token bucket refills lazily: no background timer to own or shut down.
        if self._state is State.OPEN and self._clock() - self._opened_at >= (
            self.recovery_timeout
        ):
            self._state = State.HALF_OPEN
            self._probe_in_flight = False
        return self._state

    def allows_request(self) -> bool:
        state = self.state
        if state is State.CLOSED:
            return True
        if state is State.OPEN:
            return False
        # HALF_OPEN: exactly one trial call. If the provider is still down,
        # letting a hundred probes through would recreate the pile-up the
        # breaker just protected us from.
        if self._probe_in_flight:
            return False
        self._probe_in_flight = True
        return True

    def check(self) -> None:
        """Raise instead of returning False. Used on the dispatch path."""
        if not self.allows_request():
            raise CircuitOpenError(f"circuit is {self.state.value}", status=None)

    def release_probe(self) -> None:
        """Give back a half-open probe that was reserved but never used.

        The router reserves the probe while ranking candidates; if the
        request was served by a different provider, this one never got its
        trial call and must be allowed to try again later.
        """
        if self._state is State.HALF_OPEN:
            self._probe_in_flight = False

    def record_success(self) -> None:
        self._state = State.CLOSED
        self._consecutive_failures = 0
        self._probe_in_flight = False

    def record_failure(self) -> None:
        # A failed half-open probe re-opens immediately and restarts the timer,
        # regardless of the failure count -- the probe *was* the test.
        if self._state is State.HALF_OPEN:
            self._trip()
            return
        # An already-open breaker ignores further failures. In-flight calls
        # can land after the trip, and letting them re-trip would restart the
        # recovery timer each time -- the breaker would never reach HALF_OPEN
        # while any straggler was still failing.
        if self._state is State.OPEN:
            return
        self._consecutive_failures += 1
        if self._consecutive_failures >= self.failure_threshold:
            self._trip()

    def _trip(self) -> None:
        self._state = State.OPEN
        self._opened_at = self._clock()
        self._probe_in_flight = False

    def __repr__(self) -> str:  # pragma: no cover
        return f"CircuitBreaker({self.state.value}, fails={self._consecutive_failures})"
