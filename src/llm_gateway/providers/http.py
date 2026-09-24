"""A real HTTP client for OpenAI-compatible chat-completion APIs.

This module is the only place in the package that imports httpx, and it is
imported lazily by consumers (see providers/__init__.py and the top-level
package __init__.py) so that `import llm_gateway` keeps working when the
optional `[http]` extra is not installed.

`OpenAICompatibleClient` implements the `AsyncLLMClient` protocol (just
`complete`, here) against the "POST /chat/completions" shape that OpenAI,
Groq, OpenRouter, and most self-hosted inference servers all speak. Provider
specific quirks (logprobs shape, strict-json shape, extra body fields) are
factored out into small overridable hooks rather than branched on here, so
that groq.py / openrouter.py can each subclass and adjust only what differs.
"""

from __future__ import annotations

import contextlib
import re
import time
from collections.abc import Callable, Mapping
from email.utils import parsedate_to_datetime
from typing import Any

try:
    import httpx
except ImportError as exc:  # pragma: no cover - exercised via a dedicated test
    raise ImportError(
        "llm_gateway.providers.http requires httpx. "
        "Install it with: pip install aiollm-gateway[http]"
    ) from exc

from ..types import LLMRequest, LLMResponse, ProviderError, RateLimitError

# --------------------------------------------------------------------------
# Retry-After parsing
# --------------------------------------------------------------------------

# Matches Groq-style durations such as "7.66s", "2m59.56s", "1h2m3s".
_DURATION_RE = re.compile(
    r"^\s*(?:(?P<h>\d+(?:\.\d+)?)h)?"
    r"(?:(?P<m>\d+(?:\.\d+)?)m)?"
    r"(?:(?P<s>\d+(?:\.\d+)?)s)?\s*$"
)


def parse_retry_after(value: str) -> float | None:
    """Parse a Retry-After header value into seconds.

    Handles three formats seen in the wild:
      * plain integer (or float) seconds, per RFC 9110: "30"
      * Groq-style compound durations: "7.66s", "2m59.56s", "1h"
      * an HTTP-date: "Wed, 21 Oct 2015 07:28:00 GMT"

    Returns None for anything that doesn't parse as one of these -- callers
    treat "no Retry-After" and "unparseable Retry-After" the same way, so
    this never raises.
    """
    text = value.strip()
    if not text:
        return None

    # Plain number of seconds.
    try:
        return float(text)
    except ValueError:
        pass

    # Groq-style duration, e.g. "2m59.56s". Requires at least one of
    # h/m/s to avoid matching the empty string (the regex is fully optional).
    match = _DURATION_RE.match(text)
    if match and any(match.group(g) for g in ("h", "m", "s")):
        hours = float(match.group("h") or 0.0)
        minutes = float(match.group("m") or 0.0)
        seconds = float(match.group("s") or 0.0)
        return hours * 3600.0 + minutes * 60.0 + seconds

    # HTTP-date.
    try:
        dt = parsedate_to_datetime(text)
    except (TypeError, ValueError):
        return None
    if dt.tzinfo is None:
        # RFC 9110 dates are always GMT; parsedate_to_datetime can return a
        # naive datetime for legacy formats. Treat naive as UTC.
        from datetime import UTC

        dt = dt.replace(tzinfo=UTC)
    delta = dt.timestamp() - time.time()
    return max(0.0, delta)


# --------------------------------------------------------------------------
# Client
# --------------------------------------------------------------------------


class OpenAICompatibleClient:
    """Talks to a POST {base_url}/chat/completions endpoint.

    Only `complete` is implemented (no `complete_batch`): none of the
    OpenAI-compatible chat endpoints this targets offer a synchronous
    multi-prompt batch call, so batching is left to the gateway's own
    concurrent-dispatch fallback (see providers/base.py:Provider.complete_batch).
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        *,
        default_headers: Mapping[str, str] | None = None,
        timeout: float = 60.0,
        client: httpx.AsyncClient | None = None,
        on_headers: Callable[[Mapping[str, str]], None] | None = None,
        provider_name: str = "openai-compatible",
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.on_headers = on_headers
        # Name stamped onto LLMResponse.provider -- distinct from `model`
        # (the response also echoes the model the API actually used, but
        # LLMResponse.provider identifies the upstream, matching how
        # MockClient stamps its own `name`). Adapters pass "groq"/"openrouter".
        self.provider_name = provider_name

        # If the caller injected a client (tests do this with MockTransport),
        # we must never close it -- it isn't ours. Only a client we build
        # ourselves gets closed in aclose().
        self._owns_client = client is None
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            **(default_headers or {}),
        }
        self._client = client or httpx.AsyncClient(
            base_url=self.base_url, headers=headers, timeout=timeout
        )

    # -- overridable per-adapter hooks ----------------------------------

    def logprobs_params(self) -> dict[str, Any]:
        """Body fields to request logprobs. Default: OpenAI's shape.

        Overridden per adapter because not every OpenAI-compatible provider
        expects `top_logprobs` to be an int the same way -- this hook exists
        so an adapter can change the shape without touching the shared POST
        logic.
        """
        return {"logprobs": True, "top_logprobs": 1}

    def strict_json_params(self) -> dict[str, Any]:
        """Body fields to request strict JSON output. Default: OpenAI's shape."""
        return {
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "response",
                    "strict": True,
                    "schema": {"type": "object", "additionalProperties": True},
                },
            }
        }

    def extra_body_params(self, request: LLMRequest) -> dict[str, Any]:
        """Extra body fields an adapter wants on every request. Default: none."""
        return {}

    # -- request/response plumbing ---------------------------------------

    def _build_payload(self, request: LLMRequest) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": request.model or self.model,
            "messages": [{"role": "user", "content": request.prompt}],
            "max_tokens": request.max_tokens,
        }
        if request.needs_logprobs:
            payload.update(self.logprobs_params())
        if request.needs_strict_json:
            payload.update(self.strict_json_params())
        payload.update(self.extra_body_params(request))
        return payload

    def _notify_headers(self, response: httpx.Response) -> None:
        if self.on_headers is None:
            return
        # The on_headers contract is fire-and-forget: a broken consumer must
        # never take down the actual HTTP call.
        with contextlib.suppress(Exception):
            self.on_headers(response.headers)

    def _error_message(self, response: httpx.Response) -> str:
        try:
            body = response.json()
        except ValueError:
            return response.text[:500] if response.text else response.reason_phrase
        if isinstance(body, dict):
            err = body.get("error")
            if isinstance(err, dict) and "message" in err:
                return str(err["message"])
            if isinstance(err, str):
                return err
            if "message" in body:
                return str(body["message"])
        return response.text[:500] if response.text else response.reason_phrase

    def _raise_for_status(self, response: httpx.Response) -> None:
        if response.is_success:
            return
        message = self._error_message(response)
        retry_after_header = response.headers.get("retry-after")
        retry_after = (
            parse_retry_after(retry_after_header) if retry_after_header else None
        )
        status = response.status_code
        if status == 429:
            raise RateLimitError(
                f"rate limited: {message}",
                retry_after=retry_after,
                provider=self.provider_name,
            )
        raise ProviderError(
            f"HTTP {status}: {message}",
            status=status,
            retry_after=retry_after,
            provider=self.provider_name,
        )

    def _parse_response(self, request: LLMRequest, body: dict[str, Any]) -> LLMResponse:
        choices = body.get("choices") or []
        text = ""
        logprobs: list[float] | None = None
        finish_reason: str | None = None
        if choices:
            choice = choices[0]
            message = choice.get("message") or {}
            text = message.get("content") or ""
            finish_reason = choice.get("finish_reason")
            lp = choice.get("logprobs")
            if lp and isinstance(lp, dict):
                content = lp.get("content")
                if content:
                    logprobs = [
                        item["logprob"]
                        for item in content
                        if isinstance(item, dict) and "logprob" in item
                    ]

        usage = body.get("usage") or {}
        input_tokens = int(usage.get("prompt_tokens") or 0)
        output_tokens = int(usage.get("completion_tokens") or 0)

        return LLMResponse(
            text=text,
            provider=self.provider_name,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            logprobs=logprobs,
            finish_reason=finish_reason,
        )

    async def complete(self, request: LLMRequest) -> LLMResponse:
        payload = self._build_payload(request)
        start = time.monotonic()
        try:
            response = await self._client.post("/chat/completions", json=payload)
        except httpx.TimeoutException as exc:
            raise ProviderError(
                f"request timed out: {exc}", status=None, provider=self.provider_name
            ) from exc
        except httpx.TransportError as exc:
            # Covers ConnectError and other transport-level failures. No
            # HTTP status exists here at all, so status=None -- RetryPolicy
            # treats that as retryable, which is right: the request never
            # reached the far end.
            raise ProviderError(
                f"transport error: {exc}", status=None, provider=self.provider_name
            ) from exc

        self._notify_headers(response)
        self._raise_for_status(response)

        body = response.json()
        result = self._parse_response(request, body)
        result.latency_s = time.monotonic() - start
        return result

    async def complete_batch(self, requests: list[LLMRequest]) -> list[LLMResponse]:
        """Sequential fallback so this class structurally satisfies AsyncLLMClient.

        None of the OpenAI-compatible chat endpoints this client targets have
        a synchronous multi-prompt batch call (see the module docstring), so
        there is no real batch request to make -- this exists only so the
        type satisfies AsyncLLMClient's Protocol; Provider.complete_batch
        already prefers concurrent dispatch via the gateway's own batching
        fallback when `supports_batching` is False, which is how groq.py and
        openrouter.py configure their capabilities.
        """
        return [await self.complete(r) for r in requests]

    async def list_models(self) -> list[str]:
        """Return the model IDs this provider currently offers, sorted.

        Hits the OpenAI-compatible GET {base_url}/models endpoint, which
        Groq and OpenRouter both expose as {"data": [{"id": ...}, ...]}.
        """
        try:
            response = await self._client.get("/models")
        except httpx.TimeoutException as exc:
            raise ProviderError(
                f"request timed out: {exc}", status=None, provider=self.provider_name
            ) from exc
        except httpx.TransportError as exc:
            raise ProviderError(
                f"transport error: {exc}", status=None, provider=self.provider_name
            ) from exc
        self._raise_for_status(response)
        body = response.json()
        return sorted(item["id"] for item in body.get("data") or [])

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def __aenter__(self) -> OpenAICompatibleClient:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()
