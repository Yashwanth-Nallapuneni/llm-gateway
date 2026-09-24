"""Factory for an OpenRouter-backed Provider.

OpenRouter is an aggregator, not a model host: a single OpenRouter "model"
id (e.g. "meta-llama/llama-3.3-70b-instruct") can be served by several
different upstream inference providers behind the scenes, chosen per
request. That matters for two things this module handles specially:

1. `require_parameters`: without it, OpenRouter may silently route a
   request to an upstream that ignores a parameter it doesn't support (e.g.
   logprobs, or strict JSON mode) instead of erroring -- you get back a
   well-formed 200 response that is just missing the field you asked for.
   Setting `provider.require_parameters: true` tells OpenRouter to only
   route to upstreams that actually honor every parameter in the request,
   which turns that silent gap into a normal routing failure instead.
2. No synchronous batch endpoint exists at all (see supports_batching below).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from ..types import AsyncLLMClient, LLMRequest, ProviderCapabilities, ProviderError
from .base import Provider
from .http import OpenAICompatibleClient

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class OpenRouterClient(OpenAICompatibleClient):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs.setdefault("provider_name", "openrouter")
        super().__init__(*args, **kwargs)

    def extra_body_params(self, request: LLMRequest) -> dict[str, Any]:
        # Only force parameter-honoring routing when this particular request
        # actually needs a parameter honored -- a plain request is free to
        # go to any upstream, which keeps the widest routing pool (and
        # lowest latency/cost) for the common case.
        if request.needs_logprobs or request.needs_strict_json:
            return {"provider": {"require_parameters": True}}
        return {}

    def _raise_for_status(self, response: Any) -> None:
        # 402 = out of credit. Map it explicitly (rather than letting it
        # fall through to the generic 4xx path) so the message is clear;
        # it lands in NON_RETRYABLE_STATUSES via RetryPolicy's `status >= 500`
        # fallback being False for 402, so it is already not retried, but we
        # still want to surface the specific "add credit" message.
        #
        # No _notify_headers call here: `complete()` in http.py already calls
        # it once for every response, success or error, before reaching
        # `_raise_for_status`. Calling it again here would invoke on_headers
        # (and the limiter's sync_from_headers) twice for the same response.
        if response.status_code == 402:
            message = self._error_message(response)
            raise ProviderError(
                f"OpenRouter: out of credit (402): {message}",
                status=402,
                provider=self.provider_name,
            )
        super()._raise_for_status(response)


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


def openrouter_provider(
    api_key: str,
    *,
    model: str,
    name: str = "openrouter",
    default_headers: Mapping[str, str] | None = None,
    timeout: float = 60.0,
    client: AsyncLLMClient | None = None,
    on_headers: Callable[[Mapping[str, str]], None] | None = None,
    rpm_limit: float = 60,
    tpm_limit: float = 100_000,
    priority: int = 0,
    max_concurrency: int = 32,
    supports_logprobs: bool = True,
    supports_strict_json: bool = True,
    max_context_tokens: int = 128_000,
    cost_per_1k_input: float = 0.0,
    cost_per_1k_output: float = 0.0,
    **provider_kwargs: Any,
) -> Provider:
    """Build a fully configured OpenRouter Provider.

    `model` has no default: OpenRouter hosts hundreds of models under one
    endpoint, so picking one for the caller would be arbitrary.
    `supports_logprobs`/`supports_strict_json` default to True because the
    router-level `require_parameters` guard (see OpenRouterClient above)
    is what actually keeps a request honest -- but set them to False if you
    know the specific model you're pinning never supports these, to save a
    failed round-trip.
    """
    openrouter_client = client or OpenRouterClient(
        base_url=OPENROUTER_BASE_URL,
        api_key=api_key,
        model=model,
        default_headers=default_headers,
        timeout=timeout,
        on_headers=on_headers,
    )
    capabilities = ProviderCapabilities(
        supports_logprobs=supports_logprobs,
        supports_strict_json=supports_strict_json,
        # OpenRouter has no batch endpoint whatsoever (it is a routing layer
        # over other providers' synchronous APIs) -- the gateway's Batcher
        # falls back to concurrent dispatch of individual requests. See
        # Provider.complete_batch.
        supports_batching=False,
        max_context_tokens=max_context_tokens,
        cost_per_1k_input=cost_per_1k_input,
        cost_per_1k_output=cost_per_1k_output,
    )
    provider = Provider(
        name=name,
        client=openrouter_client,
        capabilities=capabilities,
        rpm_limit=rpm_limit,
        tpm_limit=tpm_limit,
        priority=priority,
        max_concurrency=max_concurrency,
        **provider_kwargs,
    )
    _wire_header_sync(provider, openrouter_client, on_headers)
    return provider
