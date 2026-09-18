"""A spending ceiling for a run: reserve before dispatch, settle after.

The problem this module solves: the cost of an LLM call is not known until
the response comes back, because output token counts are only known after
generation. If admission were based on settled spend alone, a burst of N
concurrent requests could all check "is there budget left?" against the same
stale number, all see headroom, and all be admitted -- collectively spending
N times over the limit before any of them settles.

The fix is the same reserve/settle shape a token bucket uses for capacity:
before a request is dispatched, reserve its worst-case cost (computed from
`max_tokens`, which the caller controls, so the ceiling is knowable up
front). That reservation is subtracted from the budget immediately, so a
concurrent request arriving a moment later sees a smaller remaining balance
and can be rejected before it ever reaches a provider. Once the real
response arrives, the reservation is replaced with the actual cost. Money
that is reserved but not yet settled must count as spent for admission
purposes -- that is the entire mechanism that makes the ceiling hold under
concurrency.

Thread-safety: this module assumes a single asyncio event loop and does not
use a lock. Every method here runs synchronously with no `await` inside it,
so there is no point where another coroutine on the same loop can interleave
a read-modify-write. This is not safe to share across OS threads or multiple
event loops -- callers doing that would need to add their own locking.
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
    reservation, exactly once. Calling either twice, or calling both, raises
    -- that would double-count or double-free against the ledger.

    Also usable as a context manager: entering does nothing (the reservation
    is already made when `reserve()` returns it), and exiting calls
    `release()` if the block raised, or does nothing if it exited cleanly --
    the caller is expected to have called `settle()` themselves with the
    real cost before falling out of the `with` block. This mirrors the
    common call-site shape "reserve, make the call, settle with the actual
    cost; if anything raises, the reservation is released instead":

        with ledger.reserve(estimate) as r:
            response = await do_the_call()
            r.settle(actual_cost(response))
    """

    __slots__ = ("_closed", "_ledger", "id", "tokens", "usd")

    def __init__(self, ledger: BudgetLedger, reservation_id: int, usd: float, tokens: int) -> None:
        self._ledger = ledger
        self.id = reservation_id
        self.usd = usd
        self.tokens = tokens
        self._closed = False

    def settle(self, actual_usd: float, actual_tokens: int | None = None) -> None:
        """Replace this reservation with the real cost of the call.

        `actual_usd` may be larger than the reservation's `usd` -- a
        provider can bill more than `max_tokens` implied (e.g. billing
        quirks, or a response that used every token of `max_tokens` at a
        higher per-token rate than estimated). That is allowed and recorded
        honestly: the ledger's `spent` grows by the true amount, not the
        estimate, and `committed` never goes negative because the
        reservation is removed in the same step that the actual is added.

        `actual_tokens` defaults to this reservation's estimated tokens when
        not given, since not every caller tracks real token counts.
        """
        if self._closed:
            raise RuntimeError("reservation already settled or released")
        self._closed = True
        self._ledger._settle(self, actual_usd, actual_tokens)

    def release(self) -> None:
        """Give back the full reservation, unspent -- for a call that failed
        and cost nothing (a connection error before the provider did any
        work, a request rejected before it was ever sent, etc).

        A retry after `release()` must call `reserve()` again to get a new
        reservation; see the module-level note on retries in
        `BudgetLedger.reserve`.
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
        # If the caller settled (or released) inside the block, there is
        # nothing left to do. If the block raised without settling, release
        # the hold so a failure does not permanently consume budget. If the
        # block exited cleanly *without* settling, that is a caller bug --
        # silently releasing would let a successful, unbilled call vanish
        # from the ledger, so raise instead of guessing.
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

    Retry policy: a retry re-uses the *same* reservation rather than
    releasing and re-reserving. Concretely, the retry loop should hold one
    `Reservation` across all attempts for a given logical request and only
    call `settle()` once, on the attempt that finally succeeds (or
    `release()` once, after the last attempt has exhausted retries). This
    is deliberate: re-reserving on every attempt would let a request that
    keeps failing transiently race its own retries against concurrent
    admission, each attempt briefly free money for another request to grab
    and then take it back -- committed would oscillate instead of staying
    pinned to the worst case for the whole logical request. Holding one
    reservation for the whole retry sequence keeps the worst-case hold
    constant from the first attempt to the last, which is the simpler and
    safer invariant for a ceiling meant to bound a run, not a single HTTP
    call.
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

        # Outstanding reservations, by id, so settle()/release() can look
        # up what they are closing out. A dict rather than a set because
        # Reservation objects are mutable across their lifetime only in
        # the sense of being opened/closed once; keying by id makes double
        # settle/release detectable even if a caller somehow constructed
        # two Reservation objects with the same id (they can't, in normal
        # use -- ids only come from `_seq_counter` inside `reserve()`).
        self._outstanding: dict[int, Reservation] = {}

    # ------------------------------------------------------------------
    # reservation lifecycle
    # ------------------------------------------------------------------

    def reserve(self, estimated_usd: float, estimated_tokens: int = 0) -> Reservation:
        """Reserve `estimated_usd` (and, if tracked, `estimated_tokens`)
        against the ledger.

        Raises `BudgetExceeded` if committing this reservation would push
        `committed` past `limit_usd`, or `TokenBudgetExceeded` if it would
        push committed tokens past `max_tokens_total`. Neither error
        mutates the ledger -- a rejected reservation holds nothing.

        The dollar check runs before the token check so a request that
        fails both reports the dollar ceiling, since that is the ceiling
        this module exists for; the token ceiling is the secondary,
        optional guard.
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

    def _settle(self, reservation: Reservation, actual_usd: float, actual_tokens: int | None) -> None:
        if actual_usd < 0:
            raise ValueError(f"actual_usd must be >= 0, got {actual_usd}")
        self._outstanding.pop(reservation.id, None)
        # Remove the reservation's estimate first, then add the real cost.
        # Doing it in this order (rather than e.g. `+= actual - estimated`)
        # keeps the arithmetic obviously non-negative-preserving: `_reserved`
        # only ever shrinks by exactly what was added for this reservation,
        # so it cannot be driven negative by an `actual` larger than the
        # estimate. Clamped to 0.0 anyway as a defense against float
        # rounding leaving a -1e-16 residue.
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
