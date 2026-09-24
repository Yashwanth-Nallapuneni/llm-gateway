"""Provider handle: client + capabilities + its own limiter and breaker."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import cast

from ..breaker import CircuitBreaker
from ..rate_limit import ProviderLimiter
from ..types import AsyncLLMClient, LLMRequest, LLMResponse, ProviderCapabilities


class Provider:
    """Everything the gateway needs to know about one upstream.

    Rate limiter and breaker are per-provider and owned here rather than by
    the gateway, because they are properties of the upstream, not of the
    caller. Two gateways talking to the same provider would want to share
    these; two providers behind one gateway must never share them.
    """

    def __init__(
        self,
        name: str,
        client: AsyncLLMClient,
        capabilities: ProviderCapabilities,
        rpm_limit: float = 600,
        tpm_limit: float = 150_000,
        priority: int = 0,
        max_concurrency: int = 64,
        *,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        clock: Callable[[], float] | None = None,
        sleep: Callable[[float], Awaitable[None]] | None = None,
    ) -> None:
        self.name = name
        self.client = client
        self.capabilities = capabilities
        self.rpm_limit = rpm_limit
        self.tpm_limit = tpm_limit
        # Operator preference, used only as the last tiebreak in routing.
        self.priority = priority

        if max_concurrency < 1:
            raise ValueError("max_concurrency must be >= 1")
        self.max_concurrency = max_concurrency
        # RPM/TPM cap how much is sent per minute; this caps how many calls
        # are open at once. Some hosts only limit concurrency, and without
        # this a large sweep would open unbounded sockets.
        self.concurrency = asyncio.Semaphore(max_concurrency)

        self.limiter = ProviderLimiter(rpm_limit, tpm_limit, clock=clock, sleep=sleep)
        self.breaker = CircuitBreaker(
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout,
            clock=clock,
        )

    @property
    def supports_batching(self) -> bool:
        return self.capabilities.supports_batching

    def estimated_cost(self, input_tokens: int, output_tokens: int) -> float:
        c = self.capabilities
        return (
            input_tokens / 1000.0 * c.cost_per_1k_input
            + output_tokens / 1000.0 * c.cost_per_1k_output
        )

    def blended_cost_per_1k(self) -> float:
        """One number for ranking.

        Weighted toward output because output tokens are both more expensive
        and, for generation workloads, more numerous than the prompt.
        """
        c = self.capabilities
        return 0.3 * c.cost_per_1k_input + 0.7 * c.cost_per_1k_output

    async def complete(self, request: LLMRequest) -> LLMResponse:
        # Acquired here, around the single call, rather than by the caller
        # around a whole batch, so the fallback below can dispatch many
        # requests at once while max_concurrency still bounds live sockets.
        async with self.concurrency:
            return await self.client.complete(request)

    async def complete_batch_settled(
        self, requests: list[LLMRequest]
    ) -> list[LLMResponse | Exception]:
        """Per-request outcomes: one entry per request, and it never raises.

        The gateway calls this directly, not `complete_batch()`, because it
        needs to know which requests in a batch failed so it can retry or
        fail over only those, instead of treating one bad request as reason
        to kill every sibling in the batch.

        A real batch endpoint is one HTTP call, so it succeeds or fails as a
        unit: on failure every entry gets back the same exception object
        (not equal-but-distinct copies), which gateway.py relies on to avoid
        counting one atomic failure as many.
        """
        if self.supports_batching and hasattr(self.client, "complete_batch"):
            # One HTTP call carries the whole batch, so it costs one slot.
            async with self.concurrency:
                try:
                    responses = await self.client.complete_batch(requests)
                except Exception as exc:
                    return [exc] * len(requests)
            return list(responses)

        # Fallback for providers with no synchronous multi-prompt endpoint
        # (Groq, OpenRouter): the batch was grouped only for rate-limit
        # accounting, but each member is really its own HTTP call, so fire
        # them all at once instead of one after another. Each `complete()`
        # call acquires its own concurrency slot independently, so a batch
        # of 16 against max_concurrency=2 just serializes into 8 waves of 2
        # with no deadlock risk. `return_exceptions=True` matters because
        # these calls are independent: one raising must not cancel or block
        # the others, and `gather()` still waits for all of them either way.
        tasks = [asyncio.create_task(self.complete(r)) for r in requests]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return cast("list[LLMResponse | Exception]", results)

    async def complete_batch(self, requests: list[LLMRequest]) -> list[LLMResponse]:
        """All-or-nothing view for external callers who don't need per-request
        granularity: built on `complete_batch_settled()`, and raises the
        first failure if anything failed, after every sibling has finished.
        """
        results = await self.complete_batch_settled(requests)
        for result in results:
            if isinstance(result, Exception):
                raise result
        return cast("list[LLMResponse]", results)

    def __repr__(self) -> str:  # pragma: no cover
        return f"Provider({self.name})"
