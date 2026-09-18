"""llm-gateway: rate-limited, batching, failover-capable LLM client layer."""

from typing import Any

from .batching import Batcher
from .breaker import CircuitBreaker
from .breaker import State as BreakerState
from .budget import BudgetExceeded, BudgetLedger, Reservation, TokenBudgetExceeded
from .gateway import LLMGateway
from .metrics import MetricsSink
from .providers.base import Provider
from .providers.mock import MockClient, MockProvider
from .queue import RequestQueue
from .rate_limit import ProviderLimiter, TokenBucket
from .retry import RetryPolicy
from .routing import ProviderRouter
from .store import RunStore, idempotency_key
from .types import (
    CircuitOpenError,
    GatewayError,
    LLMRequest,
    LLMResponse,
    NoEligibleProviderError,
    ProviderCapabilities,
    ProviderError,
    QueuedRequest,
    RateLimitError,
)

__version__ = "0.1.0"

__all__ = [
    "Batcher",
    "BreakerState",
    "BudgetExceeded",
    "BudgetLedger",
    "CircuitBreaker",
    "CircuitOpenError",
    "GatewayError",
    "GroqClient",
    "LLMGateway",
    "LLMRequest",
    "LLMResponse",
    "MetricsSink",
    "MockClient",
    "MockProvider",
    "NoEligibleProviderError",
    "OpenAICompatibleClient",
    "OpenRouterClient",
    "Provider",
    "ProviderCapabilities",
    "ProviderError",
    "ProviderLimiter",
    "ProviderRouter",
    "QueuedRequest",
    "RateLimitError",
    "RequestQueue",
    "Reservation",
    "RetryPolicy",
    "RunStore",
    "TokenBucket",
    "TokenBudgetExceeded",
    "groq_provider",
    "idempotency_key",
    "openrouter_provider",
    "parse_retry_after",
]

# groq_provider / openrouter_provider / OpenAICompatibleClient / GroqClient /
# OpenRouterClient / parse_retry_after all live behind the optional `[http]`
# extra (they import httpx). Re-exporting them lazily here, the same way
# providers/__init__.py does, keeps `import llm_gateway` working with httpx
# absent -- only actually touching one of these names imports httpx and can
# raise the friendly "pip install aiollm-gateway[http]" error.
_LAZY_ATTRS = {
    "OpenAICompatibleClient",
    "parse_retry_after",
    "GroqClient",
    "groq_provider",
    "OpenRouterClient",
    "openrouter_provider",
}


def __getattr__(name: str) -> Any:
    if name not in _LAZY_ATTRS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from . import providers

    return getattr(providers, name)


def __dir__() -> list[str]:
    return sorted(__all__)
