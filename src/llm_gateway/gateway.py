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
from .metrics import MetricsSink
from .providers.base import Provider
from .queue import RequestQueue
from .retry import RetryPolicy
from .routing import ProviderRouter
from .types import (
    LLMRequest,
    LLMResponse,
    NoEligibleProviderError,
    ProviderError,
    QueuedRequest,
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
    ) -> None:
        if not providers:
            raise ValueError("at least one provider is required")
        self.providers = providers
        self.batcher = batcher or Batcher()
        self.retry = retry or RetryPolicy()
        self.router = router or ProviderRouter()
        self.metrics = metrics or MetricsSink()
        # Drives enqueued_at, blocked-time and end-to-end latency below, the
        # same way TokenBucket/ProviderLimiter/CircuitBreaker/Provider take an
        # injectable clock so tests can control time deterministically.
        #
        # Constraint: the batcher (batching.py) intentionally keeps using
        # real time.monotonic() for its own wait-deadline math, since its
        # correctness depends on interacting with real asyncio timeouts (see
        # batching.py). Batcher.collect() reads QueuedRequest.enqueued_at
        # (set from self._clock() in submit(), below) and compares it against
        # real time.monotonic() to decide whether a batch's max_wait has
        # elapsed. If a fake clock here is far from real monotonic time, that
        # comparison is meaningless. A test that injects a fake clock should
        # either keep it offset-compatible with real time.monotonic() (e.g.
        # a fixed value near "now", or an offset from it) or configure the
        # batcher with a max_wait_ms large enough that timing out on the
        # bogus elapsed time never happens.
        self._clock = clock or time.monotonic

        self.queue = RequestQueue()
        self._in_flight = 0
        self._dispatcher: asyncio.Task[None] | None = None
        self._workers: set[asyncio.Task[None]] = set()
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
        # Indirection, not a shortcut for `self._fatal`: mypy narrows a plain
        # attribute read to `None` once it has seen a guard like `if
        # self._fatal is not None: raise ...` earlier in the function, and
        # keeps treating it as `None` for the rest of the function body. That
        # is correct for code that never mutates the attribute again -- but
        # `_fatal` is shared state the dispatcher task can set concurrently,
        # out from under `submit()`, between the guard below and the re-check
        # further down. Routing every read through a method call (whose
        # result mypy cannot narrow the way it narrows a bare attribute)
        # keeps both checks live instead of one being "optimized" away by the
        # type checker as unreachable.
        return self._fatal

    async def submit(self, request: LLMRequest) -> LLMResponse:
        fatal_on_entry = self._read_fatal()
        if fatal_on_entry is not None:
            raise fatal_on_entry
        self._ensure_started()
        loop = asyncio.get_running_loop()
        entry = QueuedRequest(
            request=request, future=loop.create_future(), enqueued_at=self._clock()
        )
        self.metrics.submitted += 1
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
            last_exc: Exception = ProviderError("no attempt was made")
            for provider in candidates:
                if not remaining:
                    break
                # `_call_with_retry` resolves every entry it can -- partially,
                # for a non-batching provider whose members fail
                # independently -- and hands back only the ones it could
                # not. Only those move on to the next candidate; a
                # true-batch provider either resolves the whole set or none
                # of it, so this degenerates to the old all-or-nothing
                # failover for that case.
                remaining, provider_exc = await self._call_with_retry(provider, remaining)
                if provider_exc is not None:
                    last_exc = provider_exc

            if remaining:
                self._fail_batch(remaining, last_exc)
        finally:
            self._in_flight -= len(batch)

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
        self, provider: Provider, batch: list[QueuedRequest]
    ) -> tuple[list[QueuedRequest], Exception | None]:
        """Drive retries for `batch` against `provider`.

        Returns `(leftover, last_exc)`. `leftover` is the subset of `batch`
        this provider could not resolve inside its retry budget -- empty
        when everything succeeded -- for the caller to hand to the next
        candidate provider. `last_exc` is the most recent failure seen,
        kept so the caller has something to report if every provider is
        eventually exhausted. Every entry NOT in `leftover` has already had
        its future resolved with a successful response by the time this
        returns -- callers must not resolve them again.

        `attempt` is shared by every entry still `pending` in a given
        round: they were dispatched together, so they back off together
        too, even though a fan-out provider can resolve different entries
        on different rounds (one member retries while its siblings are
        already done). That keeps one retry-delay computation per round
        instead of per request, which matches the granularity a provider's
        Retry-After header speaks at, and it means `response.attempts`
        still ends up correct per entry: it is stamped from the round that
        actually resolved that entry, not from whatever round the batch
        started at.
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
                self.metrics.record_blocked(
                    provider.name, self._clock() - blocked_start
                )
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
                self.metrics.record_failure(
                    provider.name, getattr(exc, "status", None)
                )
                leftover.extend(pending)
                return leftover, last_exc

            self.metrics.record_batch(provider.name, len(requests))

            now = self._clock()
            retryable: list[QueuedRequest] = []
            # A real batch endpoint fails as a single HTTP call:
            # complete_batch_settled() hands every member of the batch back
            # the *same* exception object in that case (see base.py).
            # Deduplicating by identity here is what makes the breaker (and
            # the failure counter) see that as the one attempt it actually
            # was, rather than N. Without this, one failed 8-request batch
            # call could push a breaker sized for 5 consecutive failures
            # straight to open on its own -- exactly the over-eager tripping
            # this file is trying to avoid, just from the opposite
            # direction. A fan-out (non-batching) provider never produces
            # two entries sharing an exception object -- each is its own
            # call and raises its own exception -- so nothing collides here
            # and every failure is still counted on its own; that is also
            # what makes a single bad request out of sixteen move the
            # breaker's consecutive-failure count by exactly one, nowhere
            # near enough to trip a threshold of five by itself.
            seen_failures: set[int] = set()
            for entry, result in zip(pending, results, strict=True):
                if isinstance(result, Exception):
                    last_exc = result
                    if id(result) not in seen_failures:
                        seen_failures.add(id(result))
                        provider.breaker.record_failure()
                    self.metrics.record_failure(
                        provider.name, getattr(result, "status", None)
                    )
                    if self.retry.should_retry(result, attempt):
                        retryable.append(entry)
                    else:
                        leftover.append(entry)
                else:
                    provider.breaker.record_success()
                    response = result
                    response.attempts = attempt + 1
                    # End-to-end: from the moment the caller submitted,
                    # through queueing, batching, rate-limiter waits and
                    # retries. This is the latency the caller actually
                    # experiences; provider call time alone hides exactly
                    # the delays this library introduces.
                    response.latency_s = now - entry.enqueued_at
                    self.metrics.record_success(
                        provider.name,
                        response.latency_s,
                        provider.estimated_cost(
                            response.input_tokens, response.output_tokens
                        ),
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
            delay = self.retry.delay_for(
                attempt, self.retry.retry_after_from(last_exc)
            )
            attempt += 1
            await asyncio.sleep(delay)
            pending = retryable

        return leftover, last_exc
