"""Factory for an OpenRouter-backed Provider.

OpenRouter is an aggregator, not a model host: one OpenRouter "model" id can
be served by several different upstream inference providers, chosen per
request. That is why `require_parameters` matters: without it, OpenRouter
may silently route to an upstream that ignores a parameter it doesn't
support (e.g. logprobs or strict JSON), returning a well-formed 200 that is
just missing the field you asked for. Setting `provider.require_parameters:
true` turns that silent gap into a normal routing failure instead. There is
also no synchronous batch endpoint at all (see supports_batching below).
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
        # 402 = out of credit. Mapped explicitly, rather than falling through
        # to the generic 4xx path, so the message is clear -- RetryPolicy
        # already treats it as non-retryable (status < 500), this just adds
        # the specific "add credit" wording.
        #
        # No _notify_headers call here: http.py's complete() already calls it
        # once per response before reaching _raise_for_status; calling it
        # again would double-invoke on_headers for the same response.
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
