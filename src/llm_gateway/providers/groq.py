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

    Reasoning-model trap: some Groq-hosted models (e.g. `openai/gpt-oss-20b`,
    `openai/gpt-oss-120b`, `openai/gpt-oss-safeguard-20b`) spend part of the
    completion's token budget on an internal `reasoning` field before they
    ever emit `content`. At a small `max_tokens` the entire budget can go to
    reasoning, and the API still returns a normal 200 response -- just with
    `content: ""` and `finish_reason: "length"`. `LLMResponse` carries
    `finish_reason` and `was_truncated`, so check those rather than treating
    empty text alone as a failure. If you get empty completions from a Groq
    model, raise `max_tokens` and check whether it is a reasoning model
    before assuming the library itself is broken.
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
    # WARNING -- this default WILL rot. Groq deprecates and removes models
    # regularly (this project shipped with `llama-3.3-70b-versatile` as the
    # default, which no longer exists on Groq at all -- it does not appear
    # in a live `GET /openai/v1/models` response any more). Before trusting
    # this default, check what is actually live:
    #
    #   curl -H "Authorization: Bearer $GROQ_API_KEY" \
    #       https://api.groq.com/openai/v1/models
    #
    # `allam-2-7b` was verified live on 2026-09-18: it appeared in that
    # models listing, is small (7B) and cheap, is NOT a reasoning model, and
    # returned real, non-empty `content` at `max_tokens=16` (many of the
    # other available chat models on Groq are reasoning models -- e.g.
    # `openai/gpt-oss-20b` -- and burn the entire small token budget on
    # internal reasoning, returning empty content instead; see the
    # `GroqClient` docstring above). It will not stay current forever --
    # `tests/test_live_groq.py::test_default_model_is_live` exists precisely
    # to catch the day it goes stale.
    model: str = "allam-2-7b",
    name: str = "groq",
    default_headers: Mapping[str, str] | None = None,
    timeout: float = 60.0,
    client: AsyncLLMClient | None = None,
    on_headers: Callable[[Mapping[str, str]], None] | None = None,
    # Groq's per-model rate limits vary a lot and are not published as a
    # single stable number -- these two defaults are necessarily an
    # approximation, not a guarantee for whatever model you actually pass.
    # Observed live (2026-09-18) for the current default, `allam-2-7b`:
    # ~7000 requests/min and ~6000 tokens/min on the free tier. rpm is set
    # well below that observed value on purpose: tpm is the tighter
    # constraint at 6,000/min regardless (a handful of requests can exhaust
    # it long before 7000 requests would), and larger/popular models on
    # Groq (e.g. the 70B-class ones) have historically reported much lower
    # rpm than this -- around 30/min. Rather than pick one model's numbers
    # and silently mislead callers using a different model, this stays
    # conservative on rpm; a caller who knows their model's real limits
    # (via `GET /openai/v1/models` or the `x-ratelimit-*` response headers,
    # which `on_headers`/`_wire_header_sync` sync into the limiter live
    # anyway) should override both, not rely on these as ground truth.
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
