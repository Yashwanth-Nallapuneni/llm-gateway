"""A spending ceiling for a run: reserve before dispatch, settle after.

The cost of an LLM call isn't known until the response comes back, since
output token counts depend on generation. If admission checked only settled
spend, a burst of concurrent requests could all see budget headroom against
the same stale number and all get admitted, together spending far past the
limit before any of them settles.

The fix mirrors what a token bucket does for capacity: before dispatch,
reserve the worst-case cost (from `max_tokens`, which the caller controls),
subtracting it from the budget right away. A concurrent request a moment
later then sees a smaller remaining balance and can be rejected before it
ever reaches a provider. Once the real response arrives, the reservation is
replaced with the actual cost. Treating reserved-but-unsettled money as
already spent is what makes the ceiling hold under concurrency.

Thread-safety: this module assumes a single asyncio event loop and uses no
lock. Every method runs synchronously with no `await` inside it, so no other
coroutine on the same loop can interleave a read-modify-write. It is not
safe to share across OS threads or multiple event loops.
"""

from __future__ import annotations

import itertools
from types import TracebackType

from .types import GatewayError

_seq_counter = itertools.count()


class BudgetExceeded(GatewayError):
    """Raised by `BudgetLedger.reserve` when a reservation would push
    committed spend past the configured limit.
    """

    def __init__(self, *, limit: float, committed: float, requested: float) -> None:
        self.limit = limit
        self.committed = committed
        self.requested = requested
        super().__init__(
            f"budget exceeded: limit is ${limit:.6f}, ${committed:.6f} already "
            f"committed, requested ${requested:.6f} more "
            f"(would total ${committed + requested:.6f})"
        )


class TokenBudgetExceeded(GatewayError):
    """Raised by `BudgetLedger.reserve` when a reservation would push
    committed token usage past the configured `max_tokens_total`.
    """

    def __init__(self, *, limit: int, committed: int, requested: int) -> None:
        self.limit = limit
        self.committed = committed
        self.requested = requested
        super().__init__(
            f"token budget exceeded: limit is {limit} tokens, {committed} already "
            f"committed, requested {requested} more (would total {committed + requested})"
        )


class Reservation:
    """A single hold against a `BudgetLedger`, returned by `reserve()`.

    Exactly one of `settle()` or `release()` should be called on a given
    reservation, exactly once. Calling either twice, or calling both,
    raises, since that would double-count or double-free against the
    ledger.

    Also usable as a context manager: entering does nothing, since the
    reservation is already made by the time `reserve()` returns it. Exiting
    releases the hold if the block raised, and otherwise expects the caller
    to have already called `settle()` with the real cost:

        with ledger.reserve(estimate) as r:
            response = await do_the_call()
            r.settle(actual_cost(response))
    """

    __slots__ = ("_closed", "_ledger", "id", "tokens", "usd")

    def __init__(
        self, ledger: BudgetLedger, reservation_id: int, usd: float, tokens: int
    ) -> None:
        self._ledger = ledger
        self.id = reservation_id
        self.usd = usd
        self.tokens = tokens
        self._closed = False

    def settle(self, actual_usd: float, actual_tokens: int | None = None) -> None:
        """Replace this reservation with the real cost of the call.

        `actual_usd` may be larger than the reservation's `usd` -- a
        provider can bill more than `max_tokens` implied. That's allowed
        and recorded honestly: `spent` grows by the true amount, and
        `committed` stays correct because the reservation is removed in the
        same step the actual cost is added.

        `actual_tokens` defaults to this reservation's estimated tokens when
        not given, since not every caller tracks real token counts.
        """
        if self._closed:
            raise RuntimeError("reservation already settled or released")
        # Validate before changing anything, so a bad value leaves the
        # reservation open and still releasable.
        if actual_usd < 0:
            raise ValueError(f"actual_usd must be >= 0, got {actual_usd}")
        if actual_tokens is not None and actual_tokens < 0:
            raise ValueError(f"actual_tokens must be >= 0, got {actual_tokens}")
        self._closed = True
        self._ledger._settle(self, actual_usd, actual_tokens)

    def release(self) -> None:
        """Give back the full reservation, unspent -- for a call that failed
        and cost nothing (a connection error before the provider did any
        work, a request rejected before it was ever sent, etc).

        A retry after `release()` must call `reserve()` again to get a new
        reservation.
        """
        if self._closed:
            raise RuntimeError("reservation already settled or released")
        self._closed = True
        self._ledger._release(self)

    def __enter__(self) -> Reservation:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        # Nothing to do if the caller already settled or released inside
        # the block. If the block raised without settling, release the
        # hold so a failure doesn't permanently consume budget. If it
        # exited cleanly without settling, that's a caller bug -- silently
        # releasing would let a successful, unbilled call vanish from the
        # ledger, so raise instead.
        if self._closed:
            return
        if exc_type is not None:
            self.release()
            return
        raise RuntimeError(
            "Reservation exited its `with` block without settle() or release() "
            "being called -- the caller must settle with the real cost (or "
            "release explicitly) before leaving the block"
        )


class BudgetLedger:
    """Tracks spend against a fixed USD ceiling (and, optionally, a total
    token ceiling) for the lifetime of one run.

    Retry policy: a retry should reuse the same reservation rather than
    releasing and re-reserving on every attempt. One `Reservation` should
    be held across all attempts for a given request, with `settle()` called
    once on the attempt that succeeds, or `release()` once after retries
    are exhausted. Re-reserving each attempt would briefly free up money
    between attempts for another request to grab, making committed spend
    oscillate instead of staying pinned to the worst case for the whole
    request.
    """

    def __init__(self, limit_usd: float, *, max_tokens_total: int | None = None) -> None:
        if limit_usd <= 0:
            raise ValueError(f"limit_usd must be > 0, got {limit_usd}")
        if max_tokens_total is not None and max_tokens_total <= 0:
            raise ValueError(f"max_tokens_total must be > 0, got {max_tokens_total}")

        self.limit_usd = limit_usd
        self.max_tokens_total = max_tokens_total

        self._spent = 0.0
        self._reserved = 0.0
        self._tokens_spent = 0
        self._tokens_reserved = 0

        # Outstanding reservations, keyed by id, so settle()/release() can
        # look up what they are closing out.
        self._outstanding: dict[int, Reservation] = {}

    # ------------------------------------------------------------------
    # reservation lifecycle
    # ------------------------------------------------------------------

    def reserve(self, estimated_usd: float, estimated_tokens: int = 0) -> Reservation:
        """Reserve `estimated_usd` (and, if tracked, `estimated_tokens`)
        against the ledger.

        Raises `BudgetExceeded` if this would push `committed` past
        `limit_usd`, or `TokenBudgetExceeded` if it would push committed
        tokens past `max_tokens_total`. Neither error mutates the ledger.

        The dollar check runs first, so a request that fails both reports
        the dollar ceiling -- the primary limit this module exists for,
        with the token ceiling as a secondary, optional guard.
        """
        if estimated_usd < 0:
            raise ValueError(f"estimated_usd must be >= 0, got {estimated_usd}")
        if estimated_tokens < 0:
            raise ValueError(f"estimated_tokens must be >= 0, got {estimated_tokens}")

        committed = self.committed
        if committed + estimated_usd > self.limit_usd:
            raise BudgetExceeded(
                limit=self.limit_usd, committed=committed, requested=estimated_usd
            )

        if self.max_tokens_total is not None:
            committed_tokens = self.committed_tokens
            if committed_tokens + estimated_tokens > self.max_tokens_total:
                raise TokenBudgetExceeded(
                    limit=self.max_tokens_total,
                    committed=committed_tokens,
                    requested=estimated_tokens,
                )

        self._reserved += estimated_usd
        self._tokens_reserved += estimated_tokens
        reservation_id = next(_seq_counter)
        reservation = Reservation(self, reservation_id, estimated_usd, estimated_tokens)
        self._outstanding[reservation_id] = reservation
        return reservation

    def _settle(
        self, reservation: Reservation, actual_usd: float, actual_tokens: int | None
    ) -> None:
        if actual_usd < 0:
            raise ValueError(f"actual_usd must be >= 0, got {actual_usd}")
        self._outstanding.pop(reservation.id, None)
        # Remove the reservation's estimate first, then add the real cost,
        # so `_reserved` shrinks only by what this reservation added and
        # can't go negative even if the actual cost is larger. Clamped to
        # 0.0 anyway as a defense against float rounding.
        self._reserved = max(0.0, self._reserved - reservation.usd)
        self._spent += actual_usd

        tokens = reservation.tokens if actual_tokens is None else actual_tokens
        if tokens < 0:
            raise ValueError(f"actual_tokens must be >= 0, got {tokens}")
        self._tokens_reserved = max(0, self._tokens_reserved - reservation.tokens)
        self._tokens_spent += tokens

    def _release(self, reservation: Reservation) -> None:
        self._outstanding.pop(reservation.id, None)
        self._reserved = max(0.0, self._reserved - reservation.usd)
        self._tokens_reserved = max(0, self._tokens_reserved - reservation.tokens)

    # ------------------------------------------------------------------
    # accounting
    # ------------------------------------------------------------------

    @property
    def spent(self) -> float:
        """Settled spend only -- real cost of calls that have completed."""
        return self._spent

    @property
    def committed(self) -> float:
        """Settled spend plus outstanding (unsettled) reservations.

        This is the number `reserve()` checks against the limit, and it is
        the whole point of the module: it counts money that *might* still
        be released, but treats it as spent for admission so that N
        concurrent reservations can never collectively exceed the limit.
        """
        return self._spent + self._reserved

    @property
    def remaining(self) -> float:
        """Headroom against the limit, treating outstanding reservations as
        spent. Never negative -- a reservation that would drive this below
        zero is rejected by `reserve()` before it is ever made.
        """
        return max(0.0, self.limit_usd - self.committed)

    @property
    def tokens_spent(self) -> int:
        return self._tokens_spent

    @property
    def committed_tokens(self) -> int:
        return self._tokens_spent + self._tokens_reserved

    @property
    def remaining_tokens(self) -> int | None:
        """`None` when no token ceiling is configured."""
        if self.max_tokens_total is None:
            return None
        return max(0, self.max_tokens_total - self.committed_tokens)

    @property
    def outstanding_count(self) -> int:
        """Number of reservations made but not yet settled or released.
        Mostly useful for tests and debugging -- a run that has finished
        cleanly should end with this at 0.
        """
        return len(self._outstanding)
