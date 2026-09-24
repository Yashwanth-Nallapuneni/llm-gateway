"""The public front door: LLMGateway.

Wires queue -> batcher -> router -> limiter -> retry -> provider, and hands
results back through the futures held by each queue entry.
"""

from __future__ import annotations

import asyncio
import dataclasses
import time
from collections.abc import Callable
from types import TracebackType

from .batching import Batcher
from .budget import BudgetExceeded, BudgetLedger, Reservation, TokenBudgetExceeded
from .metrics import MetricsSink
from .providers.base import Provider
from .queue import RequestQueue
from .retry import RetryPolicy
from .routing import ProviderRouter
from .store import RunStore, idempotency_key
from .types import (
    GatewayError,
    LLMRequest,
    LLMResponse,
    NoEligibleProviderError,
    ProviderError,
    QueuedRequest,
    RequestTimeout,
)


class LLMGateway:
    def __init__(
        self,
        providers: list[Provider],
        batcher: Batcher | None = None,
        retry: RetryPolicy | None = None,
        router: ProviderRouter | None = None,
        metrics: MetricsSink | None = None,
        *,
        clock: Callable[[], float] | None = None,
        store: RunStore | None = None,
        budget: BudgetLedger | None = None,
        adaptive: bool = False,
    ) -> None:
        if not providers:
            raise ValueError("at least one provider is required")
        self.providers = providers
        if adaptive:
            # Opt-in, off by default. Flips every provider's request bucket
            # into AIMD mode (see rate_limit.ProviderLimiter): rate halves
            # on a 429, creeps back up on success. Meant for providers that
            # send no rate-limit headers for sync_from_headers to use.
            for p in providers:
                p.limiter.adaptive = True
        self.batcher = batcher or Batcher()
        self.retry = retry or RetryPolicy()
        self.router = router or ProviderRouter()
        self.metrics = metrics or MetricsSink()
        # Optional. `None` (the default) means every code path below that
        # mentions `self.store` is simply skipped, so behaviour with no
        # store is byte-for-byte what it was before this attribute existed.
        self.store = store
        # Same shape as `store`: `None` (the default) means every code path
        # below that mentions `self.budget` is skipped, so behaviour with no
        # budget is byte-for-byte what it was before this attribute existed.
        self.budget = budget
        self._providers_by_name = {p.name: p for p in providers}
        # Run-summary counters for the CLI ("N served from store, M freshly
        # called"). Meaningless (and left at zero) when `store` is None.
        self.served_from_store = 0
        self.freshly_called = 0
        # Drives enqueued_at, blocked-time and end-to-end latency below, the
        # same way TokenBucket/ProviderLimiter/CircuitBreaker/Provider take an
        # injectable clock so tests can control time deterministically. The
        # batcher still uses real time.monotonic() for its own wait-deadline
        # math (see batching.py), so a fake clock here should stay close to
        # real monotonic time, or the batcher should be given a generous
        # max_wait_ms, or its max_wait timeout math will misbehave.
        self._clock = clock or time.monotonic

        self.queue = RequestQueue()
        self._in_flight = 0
        self._dispatcher: asyncio.Task[None] | None = None
        self._workers: set[asyncio.Task[None]] = set()
        # Every request whose caller is still waiting, so aclose() can answer
        # all of them no matter where each one is (queue, batcher, dispatch).
        self._waiting: dict[int, QueuedRequest] = {}
        # Set if the dispatcher loop itself dies. Without this, a crash in the
        # dispatcher leaves every caller awaiting a future nobody will ever
        # resolve -- the program hangs instead of reporting the error.
        self._fatal: BaseException | None = None

        # Let the batcher see outstanding dispatches, so "queue is empty" is
        # not mistaken for "the work is done". See batching.py.
        self.batcher._in_flight = lambda: self._in_flight

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------

    def _ensure_started(self) -> None:
        # Started lazily on first submit so constructing a gateway outside a
        # running loop (module import, tests, config code) does not blow up.
        if self._dispatcher is None or self._dispatcher.done():
            self._dispatcher = asyncio.create_task(self._run())

    async def aclose(self) -> None:
        for task in (self._dispatcher, *self._workers):
            if task is not None:
                task.cancel()
        pending = [t for t in (self._dispatcher, *self._workers) if t is not None]
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        self._dispatcher = None
        self._workers.clear()
        # Answer every caller still waiting, wherever its request was (queue,
        # batcher, or mid-dispatch); otherwise their submit() waits forever.
        self._drain_with_error(GatewayError("gateway closed before the request finished"))
        for entry in list(self._waiting.values()):
            if not entry.future.done():
                entry.future.set_exception(
                    GatewayError("gateway closed before the request finished")
                )

    async def __aenter__(self) -> LLMGateway:
        self._ensure_started()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        await self.aclose()

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def _read_fatal(self) -> BaseException | None:
        # A method call, not a bare attribute read: `_fatal` can be set by the
        # dispatcher task concurrently, between the guard below and the
        # re-check further down, and mypy would otherwise narrow a repeated
        # `self._fatal` read to `None` after the first guard and treat the
        # second check as unreachable.
        return self._fatal

    async def submit(self, request: LLMRequest) -> LLMResponse:
        coro = (
            self._submit_with_store(self.store, request)
            if self.store is not None
            else self._submit_uncached(request)
        )
        if request.timeout_s is None:
            return await coro
        # Run the work as its own task and only stop *waiting* on timeout.
        # shield() keeps the task running, so the store still records the
        # final outcome instead of leaving the row stuck "in_flight".
        task = asyncio.ensure_future(coro)
        # Nobody awaits the task after a timeout; read its exception so
        # asyncio does not warn that it was never retrieved.
        task.add_done_callback(lambda t: t.exception() if not t.cancelled() else None)
        try:
            return await asyncio.wait_for(asyncio.shield(task), request.timeout_s)
        except TimeoutError:
            raise RequestTimeout(
                f"request timed out after {request.timeout_s}s"
            ) from None

    async def _submit_with_store(
        self, store: RunStore, request: LLMRequest
    ) -> LLMResponse:
        """Store-backed path: check first, reserve, call, then durably
        record the outcome. See store.py for why `reserve`/`complete`/
        `fail` are ordered the way they are -- this is just the caller of
        that contract.
        """
        key = idempotency_key(request)
        cached = await store.get_response(key)
        if cached is not None:
            self.served_from_store += 1
            return cached

        # `reserve` can itself return a cached response: another attempt
        # (in this process or a previous one, resumed from the same file)
        # finished this exact key between the `get_response` above and now.
        # That is a benign race, not an error -- use the answer, skip the
        # call.
        raced = await store.reserve(request, key)
        if raced is not None:
            self.served_from_store += 1
            return raced

        try:
            response = await self._submit_uncached(request)
        except Exception as exc:
            await store.fail(key, str(exc))
            raise
        else:
            self.freshly_called += 1
            provider = self._providers_by_name.get(response.provider)
            cost = (
                provider.estimated_cost(response.input_tokens, response.output_tokens)
                if provider is not None
                else 0.0
            )
            await store.complete(key, response, cost=cost)
            return response

    async def _submit_uncached(self, request: LLMRequest) -> LLMResponse:
        fatal_on_entry = self._read_fatal()
        if fatal_on_entry is not None:
            raise fatal_on_entry
        self._ensure_started()
        loop = asyncio.get_running_loop()
        entry = QueuedRequest(
            request=request, future=loop.create_future(), enqueued_at=self._clock()
        )
        self.metrics.submitted += 1
        self._waiting[id(entry)] = entry
        entry.future.add_done_callback(lambda _: self._waiting.pop(id(entry), None))
        self.queue.put(entry)
        # Re-check: the dispatcher can have crashed and drained the queue
        # between the check above and this put, which would leave this entry
        # sitting in a queue nobody is reading.
        fatal = self._read_fatal()
        if fatal is not None and not entry.future.done():
            entry.future.set_exception(fatal)
        return await entry.future

    async def submit_many(self, requests: list[LLMRequest]) -> list[LLMResponse]:
        """Submit everything at once and await all results, in input order."""
        return await asyncio.gather(*(self.submit(r) for r in requests))

    # ------------------------------------------------------------------
    # dispatcher
    # ------------------------------------------------------------------

    async def _run(self) -> None:
        try:
            await self._loop()
        except asyncio.CancelledError:
            raise
        except BaseException as exc:
            self._fatal = exc
            self._drain_with_error(exc)
            raise

    def _drain_with_error(self, exc: BaseException) -> None:
        while True:
            entry = self.queue.pop_nowait()
            if entry is None:
                return
            if not entry.future.done():
                entry.future.set_exception(exc)

    async def _loop(self) -> None:
        while True:
            batch = await self.batcher.collect(self.queue)
            if not batch:
                continue
            # Dispatch concurrently: the batcher must be free to assemble the
            # next batch while this one is in flight, otherwise throughput is
            # capped at one batch per round trip.
            self._in_flight += len(batch)
            task = asyncio.create_task(self._dispatch(batch))
            self._workers.add(task)
            task.add_done_callback(self._workers.discard)

    def _composite(self, batch: list[QueuedRequest]) -> LLMRequest:
        """A synthetic request describing the batch's hardest requirements.

        A batch goes to one provider, so it can only go somewhere that
        satisfies the union of what its members need. Taking the max context
        rather than the sum because providers size batch items independently.
        """
        # Start from the member with the largest context footprint, so the
        # router's context-window check sees a real prompt and a real
        # max_tokens. The old version built an empty prompt and stuffed the
        # total token estimate into `max_tokens`, which only worked by accident
        # and would have reported a nonsense max_tokens to anything else that
        # read it (a cost estimate, a log line).
        largest = max(batch, key=lambda q: q.request.estimated_total_tokens())
        return dataclasses.replace(
            largest.request,
            needs_logprobs=any(q.request.needs_logprobs for q in batch),
            needs_strict_json=any(q.request.needs_strict_json for q in batch),
        )

    async def _dispatch(self, batch: list[QueuedRequest]) -> None:
        try:
            composite = self._composite(batch)
            candidates = self.router.select_all(composite, self.providers)

            if not candidates:
                try:
                    self.router.select(composite, self.providers)
                except NoEligibleProviderError as exc:
                    self.metrics.rejected += len(batch)
                    self._fail_batch(batch, exc)
                    return

            remaining = batch
            reservations: dict[int, Reservation] = {}
            if self.budget is not None:
                remaining, reservations = self._reserve_for_batch(candidates, batch)

            last_exc: Exception = ProviderError("no attempt was made")
            for i, provider in enumerate(candidates):
                if not remaining:
                    # The rest were never tried; hand back any half-open
                    # probe the router reserved for them while ranking.
                    for unused in candidates[i:]:
                        unused.breaker.release_probe()
                    break
                # `_call_with_retry` resolves every entry it can -- partially,
                # for a non-batching provider whose members fail
                # independently -- and hands back only the ones it could
                # not. Only those move on to the next candidate; a
                # true-batch provider either resolves the whole set or none
                # of it, so this degenerates to the old all-or-nothing
                # failover for that case.
                remaining, provider_exc = await self._call_with_retry(
                    provider, remaining, reservations
                )
                if provider_exc is not None:
                    last_exc = provider_exc

            if remaining:
                # Every provider that could take these entries has now
                # either failed or exhausted its retries against them --
                # this is the one point in a logical request's life where
                # its failure is final, so any reservation still open for
                # it is released here rather than carried further.
                for entry in remaining:
                    reservation = reservations.pop(id(entry), None)
                    if reservation is not None:
                        reservation.release()
                self._fail_batch(remaining, last_exc)
        except Exception as exc:
            # An unexpected bug anywhere above (not one of the already-handled
            # provider/budget/routing failure paths) would otherwise propagate
            # out of this task with nobody awaiting it -- every future for
            # this batch would then hang forever instead of surfacing the
            # error to the caller who is actually waiting on it.
            self._fail_batch(batch, exc)
        finally:
            self._in_flight -= len(batch)

    def _reserve_for_batch(
        self, candidates: list[Provider], batch: list[QueuedRequest]
    ) -> tuple[list[QueuedRequest], dict[int, Reservation]]:
        """Reserve budget for every entry in `batch` before it is dispatched.

        One `Reservation` is made per request, up front, sized at the
        worst-case cost among every eligible candidate for this batch
        (`candidates`, already router-filtered and ordered best-first), not
        just the one that ends up serving it. That single reservation is
        then carried across every retry and failover attempt for the
        request, settling or releasing exactly once at the end, rather than
        released and re-reserved on each failover. Re-reserving on failover
        would match the actual serving provider's price more closely, but it
        would briefly free the money for a concurrent request to grab
        between attempts, violating the invariant `budget.py` documents that
        a request's worst-case reservation stays pinned for its whole
        lifetime. The tradeoff is sometimes holding more budget than the
        request ends up costing until it settles.

        A request whose worst-case reservation does not fit the remaining
        budget fails immediately with `BudgetExceeded`/`TokenBudgetExceeded`
        -- it is not added to the returned batch, so it never enters the
        retry/failover loop; running out of money is not a provider fault.
        """
        assert self.budget is not None
        admitted: list[QueuedRequest] = []
        reservations: dict[int, Reservation] = {}
        for entry in batch:
            request = entry.request
            worst_usd = max(
                provider.estimated_cost(
                    request.estimated_input_tokens(), request.max_tokens
                )
                for provider in candidates
            )
            try:
                reservation = self.budget.reserve(
                    worst_usd, request.estimated_total_tokens()
                )
            except (BudgetExceeded, TokenBudgetExceeded) as exc:
                self.metrics.rejected += 1
                if not entry.future.done():
                    entry.future.set_exception(exc)
                continue
            reservations[id(entry)] = reservation
            admitted.append(entry)
        return admitted, reservations

    def _fail_batch(self, batch: list[QueuedRequest], exc: BaseException) -> None:
        for entry in batch:
            if not entry.future.done():
                entry.future.set_exception(exc)

    async def _settle(
        self, provider: Provider, requests: list[LLMRequest]
    ) -> list[LLMResponse | Exception]:
        """One dispatch round against `provider`, one outcome per request.

        A lone request bypasses the batch machinery entirely and calls
        `provider.complete()` directly, same as before this module grew
        batch awareness. Anything larger goes through
        `Provider.complete_batch_settled()`, which is itself either the
        all-or-nothing call a true-batch provider makes it, or N
        independent calls for a fan-out one -- this method does not need
        to know which; it only needs the per-request outcomes either way.
        """
        if len(requests) == 1:
            try:
                return [await provider.complete(requests[0])]
            except Exception as solo_exc:
                return [solo_exc]

        results = await provider.complete_batch_settled(requests)
        if len(results) != len(requests):
            # A provider that answers a batch of N with fewer than N
            # responses would otherwise leave the unmatched callers awaiting
            # futures nobody resolves. Treat it as one failure shared by
            # every request in the round -- the same way a true-batch
            # provider failing outright is one failure shared by all of
            # them. status=None makes it retryable.
            short_batch_exc: Exception = ProviderError(
                f"{provider.name} returned {len(results)} responses "
                f"for {len(requests)} requests",
                provider=provider.name,
            )
            return [short_batch_exc] * len(requests)
        return results

    async def _call_with_retry(
        self,
        provider: Provider,
        batch: list[QueuedRequest],
        reservations: dict[int, Reservation],
    ) -> tuple[list[QueuedRequest], Exception | None]:
        """Drive retries for `batch` against `provider`.

        Returns `(leftover, last_exc)`. `leftover` is the subset of `batch`
        this provider could not resolve inside its retry budget -- empty
        when everything succeeded -- for the caller to hand to the next
        candidate provider. `last_exc` is the most recent failure seen, kept
        so the caller has something to report if every provider is
        eventually exhausted. Every entry NOT in `leftover` has already had
        its future resolved with a successful response by the time this
        returns -- callers must not resolve them again.

        `attempt` is shared by every entry still `pending` in a round: they
        were dispatched together, so they back off together too (one
        retry-delay computation per round, matching the granularity a
        provider's Retry-After header speaks at), even though a fan-out
        provider can resolve different entries on different rounds.
        `response.attempts` is still correct per entry since it is stamped
        from the round that actually resolved that entry.
        """
        pending = list(batch)
        leftover: list[QueuedRequest] = []
        last_exc: Exception | None = None
        attempt = 0

        while pending:
            requests = [q.request for q in pending]
            n_tokens = sum(r.estimated_total_tokens() for r in requests)
            try:
                # Only on retries. The router already consumed this
                # provider's admission (and, in HALF_OPEN, its single probe
                # permit) when it selected the provider; checking again
                # here would spend a second permit that does not exist,
                # reject our own probe, and leave the breaker stuck
                # HALF_OPEN with an in-flight probe that never resolves --
                # a provider that has recovered would never be used again.
                if attempt > 0:
                    provider.breaker.check()

                blocked_start = self._clock()
                await provider.limiter.acquire(len(requests), n_tokens)
                self.metrics.record_blocked(provider.name, self._clock() - blocked_start)
                self.metrics.record_attempt(provider.name)

                # The concurrency slot is acquired inside provider.complete()
                # / provider.complete_batch_settled(), around each
                # individual network call -- not here, and not across the
                # rate-limiter wait above or the retry sleep below.
                # Acquiring it per call (rather than once per batch, here)
                # is what lets a non-batching provider's fallback dispatch
                # many requests concurrently while max_concurrency still
                # bounds actual open sockets instead of batches; holding it
                # across the retry sleep would let a few failing requests
                # occupy every slot doing nothing, starving healthy traffic
                # behind them.
                results = await self._settle(provider, requests)
            except Exception as exc:
                # The breaker check or the limiter itself raised, before any
                # request was even attempted this round -- there are no
                # per-request outcomes to look at, so every pending entry
                # shares this one cause.
                last_exc = exc
                provider.breaker.record_failure()
                self.metrics.record_failure(provider.name, getattr(exc, "status", None))
                leftover.extend(pending)
                return leftover, last_exc

            self.metrics.record_batch(provider.name, len(requests))

            now = self._clock()
            retryable: list[QueuedRequest] = []
            # A real batch endpoint fails as a single HTTP call, so
            # complete_batch_settled() hands every member the *same*
            # exception object (see base.py). Deduplicating by identity here
            # makes the breaker and failure counter see that as the one
            # attempt it actually was, instead of N -- otherwise one failed
            # 8-request batch call could trip a breaker sized for 5
            # consecutive failures by itself. A fan-out provider never
            # shares an exception object between entries, so each of its
            # failures is still counted on its own.
            seen_failures: set[int] = set()
            for entry, result in zip(pending, results, strict=True):
                if isinstance(result, Exception):
                    last_exc = result
                    status = getattr(result, "status", None)
                    if id(result) not in seen_failures:
                        seen_failures.add(id(result))
                        provider.breaker.record_failure()
                        # One 429 for a whole batch is one rejection, so the
                        # rate is halved once, not once per request in it.
                        # No-op unless the gateway was built with adaptive=True.
                        if status == 429:
                            provider.limiter.on_throttled()
                    self.metrics.record_failure(provider.name, status)
                    if self.retry.should_retry(result, attempt):
                        retryable.append(entry)
                    else:
                        leftover.append(entry)
                else:
                    provider.breaker.record_success()
                    provider.limiter.on_success()
                    response = result
                    response.attempts = attempt + 1
                    # End-to-end: from the moment the caller submitted,
                    # through queueing, batching, rate-limiter waits and
                    # retries. This is the latency the caller actually
                    # experiences; provider call time alone hides exactly
                    # the delays this library introduces.
                    response.latency_s = now - entry.enqueued_at
                    actual_cost = provider.estimated_cost(
                        response.input_tokens, response.output_tokens
                    )
                    self.metrics.record_success(
                        provider.name, response.latency_s, actual_cost
                    )
                    reservation = reservations.pop(id(entry), None)
                    if reservation is not None:
                        # Settle with what the call actually cost, not the
                        # worst-case estimate reserve() held -- this is what
                        # frees an over-reservation's slack back to the
                        # ledger for the next request.
                        reservation.settle(
                            actual_cost,
                            response.input_tokens + response.output_tokens,
                        )
                    if not entry.future.done():
                        entry.future.set_result(response)

            if not retryable:
                return leftover, last_exc

            # retryable is non-empty only because at least one iteration of
            # the loop above hit the isinstance(result, Exception) branch,
            # which always sets last_exc first -- so this is never None here.
            assert last_exc is not None
            self.metrics.record_retry(provider.name)
            delay = self.retry.delay_for(attempt, self.retry.retry_after_from(last_exc))
            attempt += 1
            await asyncio.sleep(delay)
            pending = retryable

        return leftover, last_exc
