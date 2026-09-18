"""Shared dataclasses and exceptions.

Deliberately dependency-free: everything here is stdlib so the four core
modules can be read (and rebuilt) without chasing imports.
"""

from __future__ import annotations

import asyncio
import itertools
from dataclasses import dataclass, field
from typing import Any, Protocol

# --------------------------------------------------------------------------
# Requests and responses
# --------------------------------------------------------------------------


@dataclass(slots=True)
class LLMRequest:
    """A single unit of work handed to the gateway."""

    prompt: str
    max_tokens: int = 256
    model: str | None = None
    needs_logprobs: bool = False
    needs_strict_json: bool = False
    priority: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def estimated_input_tokens(self) -> int:
        """Cheap heuristic: ~4 characters per token.

        A real tokenizer would be more accurate, but the batcher only needs
        this to decide *when to stop adding requests to a batch*, and it is
        called on every enqueue. Being off by 15% costs a slightly smaller
        batch; importing tiktoken would cost a heavyweight dependency.
        """
        return max(1, len(self.prompt) // 4)

    def estimated_total_tokens(self) -> int:
        return self.estimated_input_tokens() + self.max_tokens


@dataclass(slots=True)
class LLMResponse:
    text: str
    provider: str
    input_tokens: int = 0
    output_tokens: int = 0
    logprobs: list[float] | None = None
    attempts: int = 1
    latency_s: float = 0.0


@dataclass(slots=True)
class ProviderCapabilities:
    supports_logprobs: bool = False
    supports_strict_json: bool = False
    supports_batching: bool = False
    max_context_tokens: int = 8_192
    cost_per_1k_input: float = 0.0
    cost_per_1k_output: float = 0.0


# --------------------------------------------------------------------------
# Queue entry
# --------------------------------------------------------------------------

_seq_counter = itertools.count()


@dataclass(slots=True)
class QueuedRequest:
    """A request plus the plumbing needed to return its result to the caller."""

    request: LLMRequest
    future: asyncio.Future[LLMResponse]
    enqueued_at: float
    # Monotonic tiebreaker. See queue.py for why this field is load-bearing.
    seq: int = field(default_factory=lambda: next(_seq_counter))

    @property
    def priority(self) -> int:
        return self.request.priority


# --------------------------------------------------------------------------
# Errors
# --------------------------------------------------------------------------


class GatewayError(Exception):
    """Base for everything this library raises."""


class ProviderError(GatewayError):
    """A failure that came back from (or on the way to) a provider.

    `status` carries the HTTP status when there is one. `retry_after` carries
    the parsed Retry-After header when the provider sent one -- RetryPolicy
    treats it as authoritative.
    """

    def __init__(
        self,
        message: str,
        *,
        status: int | None = None,
        retry_after: float | None = None,
        provider: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.retry_after = retry_after
        self.provider = provider


class RateLimitError(ProviderError):
    """HTTP 429."""

    def __init__(self, message: str = "rate limited", **kw: Any) -> None:
        kw.setdefault("status", 429)
        super().__init__(message, **kw)


class NoEligibleProviderError(GatewayError):
    """No provider survived routing.

    Carries `reasons`: provider name -> the constraint that eliminated it.
    The whole point is that this message names *why*, per provider.
    """

    def __init__(self, reasons: dict[str, str]) -> None:
        self.reasons = reasons
        detail = "; ".join(f"{name}: {why}" for name, why in reasons.items())
        super().__init__(f"no eligible provider ({detail or 'no providers configured'})")


class CircuitOpenError(ProviderError):
    """The breaker rejected the call without touching the provider."""


# --------------------------------------------------------------------------
# Provider protocol
# --------------------------------------------------------------------------


class AsyncLLMClient(Protocol):
    """Minimal surface the gateway needs from a provider SDK."""

    async def complete(self, request: LLMRequest) -> LLMResponse: ...

    async def complete_batch(self, requests: list[LLMRequest]) -> list[LLMResponse]: ...
