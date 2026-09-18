"""Factory for a Groq-backed Provider.

Groq serves an OpenAI-compatible chat-completions endpoint, so this is a
thin configuration layer over OpenAICompatibleClient rather than a new
transport implementation.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from ..types import AsyncLLMClient, ProviderCapabilities
from .base import Provider
from .http import OpenAICompatibleClient

GROQ_BASE_URL = "https://api.groq.com/openai/v1"


class GroqClient(OpenAICompatibleClient):
    """OpenAI-compatible client pinned at Groq's endpoint.

    Groq does not support the `logprobs` / `top_logprobs` parameters at all
    (unlike OpenAI, which accepts them but may ignore them). Capabilities on
    the returned Provider set `supports_logprobs=False`, which is what keeps
    the router from ever handing this provider a request that needs them --
    so logprobs_params() here is dead code in normal operation, kept only so
    a caller who constructs GroqClient directly (bypassing the router) gets
    a real request body instead of a silent no-op.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs.setdefault("provider_name", "groq")
        super().__init__(*args, **kwargs)


def _wire_header_sync(
    provider: Provider,
    client: object,
    user_callback: Callable[[Mapping[str, str]], None] | None,
) -> None:
    """Point the client's on_headers hook at this provider's rate limiter.

    Without this, the limiter runs purely on the numbers configured above,
    which are a guess: the real limit depends on your plan, and on whatever
    else is sharing the API key right now. The provider reports the truth on
    every response, so feed it back into the buckets.

    Any callback the caller supplied still runs -- it is chained, not replaced,
    so passing `on_headers=` remains a way to observe headers rather than a way
    to accidentally disable the sync.
    """
    sync = provider.limiter.sync_from_headers

    if user_callback is None:
        combined: Callable[[Mapping[str, str]], None] = sync
    else:

        def combined(headers: Mapping[str, str]) -> None:
            sync(headers)
            user_callback(headers)

    # Only wire it up if this really is one of our HTTP clients. A caller who
    # injected their own client object keeps whatever behaviour it already has.
    if hasattr(client, "on_headers"):
        client.on_headers = combined


def groq_provider(
    api_key: str,
    *,
    model: str = "llama-3.3-70b-versatile",
    name: str = "groq",
    default_headers: Mapping[str, str] | None = None,
    timeout: float = 60.0,
    client: AsyncLLMClient | None = None,
    on_headers: Callable[[Mapping[str, str]], None] | None = None,
    # Groq's free tier (as of writing) is roughly 30 requests/minute and
    # 6,000 tokens/minute for the popular 70B models; paid tiers raise both
    # substantially. These are deliberately conservative defaults -- a
    # caller on a paid plan should override them, not the other way around.
    rpm_limit: float = 30,
    tpm_limit: float = 6_000,
    priority: int = 0,
    max_concurrency: int = 16,
    max_context_tokens: int = 128_000,
    cost_per_1k_input: float = 0.0,
    cost_per_1k_output: float = 0.0,
    **provider_kwargs: Any,
) -> Provider:
    """Build a fully configured Groq Provider.

    `cost_per_1k_*` default to 0.0 because Groq's pricing varies by model
    (and the free tier is, well, free) -- pass real numbers if cost-aware
    routing matters for your setup.
    """
    groq_client = client or GroqClient(
        base_url=GROQ_BASE_URL,
        api_key=api_key,
        model=model,
        default_headers=default_headers,
        timeout=timeout,
        on_headers=on_headers,
    )
    capabilities = ProviderCapabilities(
        supports_logprobs=False,
        supports_strict_json=True,
        # Groq has no synchronous multi-prompt batch endpoint (its batch API,
        # like OpenAI's, is an asynchronous 24h file-based job) -- so the
        # gateway's Batcher falls back to concurrent dispatch of individual
        # requests rather than a single grouped call. See Provider.complete_batch.
        supports_batching=False,
        max_context_tokens=max_context_tokens,
        cost_per_1k_input=cost_per_1k_input,
        cost_per_1k_output=cost_per_1k_output,
    )
    provider = Provider(
        name=name,
        client=groq_client,
        capabilities=capabilities,
        rpm_limit=rpm_limit,
        tpm_limit=tpm_limit,
        priority=priority,
        max_concurrency=max_concurrency,
        **provider_kwargs,
    )
    _wire_header_sync(provider, groq_client, on_headers)
    return provider
