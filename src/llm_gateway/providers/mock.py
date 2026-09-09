"""A fake provider with configurable latency, failures and rate limits.

The entire test suite runs against this. No network calls, ever.
"""

from __future__ import annotations

import asyncio
import time
from typing import Callable

from ..types import (
    LLMRequest,
    LLMResponse,
    ProviderCapabilities,
    ProviderError,
    RateLimitError,
)
from .base import Provider


class MockClient:
    """Deterministic stand-in for a provider SDK.

    Failure injection is a list of statuses consumed in order, so a test can
    say "fail 503, fail 503, then succeed" and get exactly that.
    """

    def __init__(
        self,
        name: str = "mock",
        latency: float = 0.0,
        fail_sequence: list[int | None] | None = None,
        fail_status: int | None = None,
        fail_first_n: int = 0,
        retry_after: float | None = None,
        rpm_limit: int | None = None,
        supports_logprobs: bool = False,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self.name = name
        self.latency = latency
        self.fail_sequence = list(fail_sequence) if fail_sequence else None
        self.fail_status = fail_status
        self.fail_first_n = fail_first_n
        self._fail_first_configured = fail_first_n > 0
        self.retry_after = retry_after
        self.rpm_limit = rpm_limit
        self.supports_logprobs = supports_logprobs
        self._clock = clock or time.monotonic

        self.calls = 0
        self.batch_calls = 0
        self.batch_sizes: list[int] = []
        # Timestamps of accepted calls, for simulating the provider's own
        # rate limit: it 429s when too many land inside a 60s window.
        self._window: list[float] = []

    # -- failure policy ------------------------------------------------

    def _next_failure(self) -> int | None:
        if self.fail_sequence is not None:
            if not self.fail_sequence:
                return None
            return self.fail_sequence.pop(0)
        if self._fail_first_configured:
            # "fail the first N calls, then behave" -- once the budget is
            # spent this client is healthy, regardless of fail_status.
            if self.fail_first_n > 0:
                self.fail_first_n -= 1
                return self.fail_status or 503
            return None
        if self.fail_status is not None:
            return self.fail_status
        return None

    def _enforce_rate_limit(self) -> None:
        if self.rpm_limit is None:
            return
        now = self._clock()
        self._window = [t for t in self._window if now - t < 60.0]
        if len(self._window) >= self.rpm_limit:
            raise RateLimitError(
                f"{self.name}: simulated 429", retry_after=1.0, provider=self.name
            )
        self._window.append(now)

    # -- API -----------------------------------------------------------

    async def complete(self, request: LLMRequest) -> LLMResponse:
        self.calls += 1
        if self.latency:
            await asyncio.sleep(self.latency)

        self._enforce_rate_limit()

        status = self._next_failure()
        if status is not None:
            if status == 429:
                raise RateLimitError(
                    f"{self.name}: injected 429",
                    retry_after=self.retry_after,
                    provider=self.name,
                )
            raise ProviderError(
                f"{self.name}: injected {status}",
                status=status,
                retry_after=self.retry_after,
                provider=self.name,
            )

        return LLMResponse(
            text=f"[{self.name}] {request.prompt[:40]}",
            provider=self.name,
            input_tokens=request.estimated_input_tokens(),
            output_tokens=min(request.max_tokens, 32),
            logprobs=[-0.1, -0.2] if self.supports_logprobs else None,
        )

    async def complete_batch(self, requests: list[LLMRequest]) -> list[LLMResponse]:
        self.batch_calls += 1
        self.batch_sizes.append(len(requests))
        if self.latency:
            await asyncio.sleep(self.latency)
        self._enforce_rate_limit()

        status = self._next_failure()
        if status is not None:
            # A batch fails as a unit -- that is how provider batch endpoints
            # behave, and it is what makes batching a latency/blast-radius
            # tradeoff rather than a free win.
            if status == 429:
                raise RateLimitError(
                    f"{self.name}: injected 429 (batch)",
                    retry_after=self.retry_after,
                    provider=self.name,
                )
            raise ProviderError(
                f"{self.name}: injected {status} (batch)",
                status=status,
                retry_after=self.retry_after,
                provider=self.name,
            )

        self.calls += len(requests)
        return [
            LLMResponse(
                text=f"[{self.name}] {r.prompt[:40]}",
                provider=self.name,
                input_tokens=r.estimated_input_tokens(),
                output_tokens=min(r.max_tokens, 32),
                logprobs=[-0.1, -0.2] if self.supports_logprobs else None,
            )
            for r in requests
        ]


def MockProvider(
    name: str = "mock",
    *,
    supports_logprobs: bool = False,
    supports_strict_json: bool = False,
    supports_batching: bool = True,
    max_context_tokens: int = 32_000,
    cost_per_1k_input: float = 0.05,
    cost_per_1k_output: float = 0.10,
    rpm_limit: float = 600,
    tpm_limit: float = 150_000,
    priority: int = 0,
    failure_threshold: int = 5,
    recovery_timeout: float = 30.0,
    client: MockClient | None = None,
    **client_kwargs,
) -> Provider:
    """Convenience constructor: a Provider wired to a MockClient.

    Breaker settings are named explicitly rather than swept into
    **client_kwargs: they configure the Provider, not the client, and when
    an explicit `client=` is passed **client_kwargs is ignored entirely --
    so a swept-up `failure_threshold` would be silently dropped.
    """
    return Provider(
        name=name,
        client=client
        or MockClient(name=name, supports_logprobs=supports_logprobs, **client_kwargs),
        capabilities=ProviderCapabilities(
            supports_logprobs=supports_logprobs,
            supports_strict_json=supports_strict_json,
            supports_batching=supports_batching,
            max_context_tokens=max_context_tokens,
            cost_per_1k_input=cost_per_1k_input,
            cost_per_1k_output=cost_per_1k_output,
        ),
        rpm_limit=rpm_limit,
        tpm_limit=tpm_limit,
        priority=priority,
        failure_threshold=failure_threshold,
        recovery_timeout=recovery_timeout,
    )
