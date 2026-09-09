"""llm-gateway: rate-limited, batching, failover-capable LLM client layer."""

from .batching import Batcher
from .breaker import CircuitBreaker, State as BreakerState
from .gateway import LLMGateway
from .metrics import MetricsSink
from .providers.base import Provider
from .providers.mock import MockClient, MockProvider
from .queue import RequestQueue
from .rate_limit import ProviderLimiter, TokenBucket
from .retry import RetryPolicy
from .routing import ProviderRouter
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
    "CircuitBreaker",
    "CircuitOpenError",
    "GatewayError",
    "LLMGateway",
    "LLMRequest",
    "LLMResponse",
    "MetricsSink",
    "MockClient",
    "MockProvider",
    "NoEligibleProviderError",
    "Provider",
    "ProviderCapabilities",
    "ProviderError",
    "ProviderLimiter",
    "ProviderRouter",
    "QueuedRequest",
    "RateLimitError",
    "RequestQueue",
    "RetryPolicy",
    "TokenBucket",
]
