"""Provider handle: client + capabilities + its own limiter and breaker."""

from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Callable

from ..breaker import CircuitBreaker
from ..rate_limit import ProviderLimiter
from ..types import LLMRequest, LLMResponse, ProviderCapabilities


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
        client: Any,
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
        # Rate and concurrency are different limits. RPM/TPM bound how much you
        # send per minute; this bounds how many calls are open at once. Some
        # hosts (DeepInfra: concurrent requests per model) limit only the
        # latter, and without a cap a large sweep opens unbounded sockets.
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
        return await self.client.complete(request)

    async def complete_batch(self, requests: list[LLMRequest]) -> list[LLMResponse]:
        if self.supports_batching and hasattr(self.client, "complete_batch"):
            return await self.client.complete_batch(requests)
        # Fallback: sequential. Correct, just slower -- keeps the dispatch path
        # uniform so the gateway never branches on capability at call time.
        return [await self.client.complete(r) for r in requests]

    def __repr__(self) -> str:  # pragma: no cover
        return f"Provider({self.name})"
