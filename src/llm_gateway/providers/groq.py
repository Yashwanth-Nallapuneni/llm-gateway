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

    Groq does not support `logprobs`/`top_logprobs` at all, unlike OpenAI
    which accepts but may ignore them. The Provider this builds sets
    `supports_logprobs=False` so the router never sends a request that needs
    them; `logprobs_params()` stays here only for a caller who constructs
    GroqClient directly, bypassing the router.

    Reasoning-model trap: some Groq models (e.g. `openai/gpt-oss-20b`) spend
    part of the token budget on internal reasoning before emitting any
    `content`. With a small `max_tokens` the whole budget can go to
    reasoning, returning a normal 200 with `content: ""` and
    `finish_reason: "length"`. Check `LLMResponse.was_truncated` rather than
    treating empty text alone as failure.
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

    Without this, the limiter runs on the configured numbers alone, which are
    a guess: the real limit depends on your plan and whoever else shares the
    API key. The provider reports the truth on every response, so feed it
    back into the buckets. Any caller-supplied callback is chained, not
    replaced, so `on_headers=` stays a way to observe rather than disable it.
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
    # WARNING: this default will go stale. Groq deprecates models regularly,
    # so check what is actually live before trusting it:
    #   curl -H "Authorization: Bearer $GROQ_API_KEY" \
    #       https://api.groq.com/openai/v1/models
    # `allam-2-7b` was verified live on 2026-09-18: small, cheap, not a
    # reasoning model (see the GroqClient docstring), and it returns
    # non-empty content even at max_tokens=16.
    # `tests/test_live_groq.py::test_default_model_is_live` catches drift.
    model: str = "allam-2-7b",
    name: str = "groq",
    default_headers: Mapping[str, str] | None = None,
    timeout: float = 60.0,
    client: AsyncLLMClient | None = None,
    on_headers: Callable[[Mapping[str, str]], None] | None = None,
    # Groq's per-model rate limits vary and aren't published as one stable
    # number, so these are conservative approximations, not guarantees.
    # Observed live (2026-09-18) for the default `allam-2-7b`: ~7000 rpm and
    # ~6000 tpm on the free tier; tpm is the tighter limit, and larger/popular
    # models report much lower rpm (around 30/min), so rpm stays conservative
    # here too. A caller who knows their model's real limits should override
    # both -- `on_headers`/`_wire_header_sync` also syncs the live
    # `x-ratelimit-*` headers into the limiter automatically.
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
