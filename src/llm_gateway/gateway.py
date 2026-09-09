"""The public front door: LLMGateway.

Wires queue -> batcher -> router -> limiter -> retry -> provider, and hands
results back through the futures held by each queue entry.
"""

from __future__ import annotations

import asyncio
import time

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
    ) -> None:
        if not providers:
            raise ValueError("at least one provider is required")
        self.providers = providers
        self.batcher = batcher or Batcher()
        self.retry = retry or RetryPolicy()
        self.router = router or ProviderRouter()
        self.metrics = metrics or MetricsSink()

        self.queue = RequestQueue()
        self._in_flight = 0
        self._dispatcher: asyncio.Task | None = None
        self._workers: set[asyncio.Task] = set()
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

    async def __aenter__(self) -> "LLMGateway":
        self._ensure_started()
        return self

    async def __aexit__(self, *exc) -> None:
        await self.aclose()

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    async def submit(self, request: LLMRequest) -> LLMResponse:
        if self._fatal is not None:
            raise self._fatal
        self._ensure_started()
        loop = asyncio.get_running_loop()
        entry = QueuedRequest(
            request=request, future=loop.create_future(), enqueued_at=time.monotonic()
        )
        self.metrics.submitted += 1
        self.queue.put(entry)
        # Re-check: the dispatcher can have crashed and drained the queue
        # between the check above and this put, which would leave this entry
        # sitting in a queue nobody is reading.
        if self._fatal is not None and not entry.future.done():
            entry.future.set_exception(self._fatal)
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
        except BaseException as exc:  # noqa: BLE001 - last line of defence
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
        return LLMRequest(
            prompt="",
            max_tokens=max(q.request.max_tokens for q in batch),
            needs_logprobs=any(q.request.needs_logprobs for q in batch),
            needs_strict_json=any(q.request.needs_strict_json for q in batch),
        )

    async def _dispatch(self, batch: list[QueuedRequest]) -> None:
        try:
            composite = self._composite(batch)
            # Context sizing is per-item, so check the largest member too.
            composite.max_tokens = max(
                q.request.estimated_total_tokens() for q in batch
            )
            candidates = self.router.select_all(composite, self.providers)

            if not candidates:
                try:
                    self.router.select(composite, self.providers)
                except NoEligibleProviderError as exc:
                    self.metrics.rejected += len(batch)
                    self._fail_batch(batch, exc)
                    return

            last_exc: Exception = ProviderError("no attempt was made")
            for provider in candidates:
                try:
                    responses = await self._call_with_retry(provider, batch)
                except Exception as exc:  # noqa: BLE001 - failover boundary
                    last_exc = exc
                    # Fall through to the next provider. This is the failover:
                    # the retry budget is spent per provider, and exhausting it
                    # is the signal to move on rather than to give up.
                    continue

                for entry, response in zip(batch, responses):
                    if not entry.future.done():
                        response.latency_s = time.monotonic() - entry.enqueued_at
                        entry.future.set_result(response)
                return

            self._fail_batch(batch, last_exc)
        finally:
            self._in_flight -= len(batch)

    def _fail_batch(self, batch: list[QueuedRequest], exc: BaseException) -> None:
        for entry in batch:
            if not entry.future.done():
                entry.future.set_exception(exc)

    async def _call_with_retry(
        self, provider: Provider, batch: list[QueuedRequest]
    ) -> list[LLMResponse]:
        requests = [q.request for q in batch]
        n_tokens = sum(r.estimated_total_tokens() for r in requests)

        attempt = 0
        while True:
            # Only on retries. The router already consumed this provider's
            # admission (and, in HALF_OPEN, its single probe permit) when it
            # selected the provider; checking again here would spend a second
            # permit that does not exist, reject our own probe, and leave the
            # breaker stuck HALF_OPEN with an in-flight probe that never
            # resolves -- a provider that has recovered would never be used
            # again.
            if attempt > 0:
                provider.breaker.check()

            blocked_start = time.monotonic()
            await provider.limiter.acquire(len(requests), n_tokens)
            self.metrics.record_blocked(
                provider.name, time.monotonic() - blocked_start
            )

            self.metrics.record_attempt(provider.name)
            started = time.monotonic()
            try:
                if len(requests) == 1:
                    responses = [await provider.complete(requests[0])]
                else:
                    responses = await provider.complete_batch(requests)
            except Exception as exc:  # noqa: BLE001
                provider.breaker.record_failure()
                self.metrics.record_failure(
                    provider.name, getattr(exc, "status", None)
                )
                if not self.retry.should_retry(exc, attempt):
                    raise
                self.metrics.record_retry(provider.name)
                delay = self.retry.delay_for(
                    attempt, self.retry.retry_after_from(exc)
                )
                attempt += 1
                await asyncio.sleep(delay)
                continue

            elapsed = time.monotonic() - started
            provider.breaker.record_success()
            self.metrics.record_batch(provider.name, len(requests))
            for response in responses:
                response.attempts = attempt + 1
                self.metrics.record_success(
                    provider.name,
                    elapsed,
                    provider.estimated_cost(
                        response.input_tokens, response.output_tokens
                    ),
                )
            return responses
