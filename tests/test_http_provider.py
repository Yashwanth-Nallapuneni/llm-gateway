"""Tests for the HTTP provider layer -- entirely offline via httpx.MockTransport.

No network access anywhere in this file. Every httpx.AsyncClient is built on
a MockTransport whose handler is a plain Python function, so these tests run
under `pytest -m "not live"` like everything else in the suite.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from email.utils import format_datetime

import httpx
import pytest

from llm_gateway import LLMGateway, LLMRequest, ProviderError, RateLimitError, RetryPolicy
from llm_gateway.providers.groq import GROQ_BASE_URL, GroqClient, groq_provider
from llm_gateway.providers.http import OpenAICompatibleClient, parse_retry_after
from llm_gateway.providers.mock import MockClient
from llm_gateway.providers.openrouter import (
    OPENROUTER_BASE_URL,
    OpenRouterClient,
    openrouter_provider,
)


def make_client(handler, *, base_url: str = "https://example.test/v1", **kwargs) -> httpx.AsyncClient:
    transport = httpx.MockTransport(handler)
    return httpx.AsyncClient(transport=transport, base_url=base_url, **kwargs)


def success_body(text: str = "hello there", *, prompt_tokens=10, completion_tokens=5,
                  logprobs=None, model="test-model", finish_reason="stop",
                  omit_finish_reason=False) -> dict:
    choice: dict = {"message": {"role": "assistant", "content": text}, "index": 0}
    if not omit_finish_reason:
        choice["finish_reason"] = finish_reason
    if logprobs is not None:
        choice["logprobs"] = {"content": [{"token": t, "logprob": lp} for t, lp in logprobs]}
    return {
        "id": "chatcmpl-1",
        "model": model,
        "choices": [choice],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens},
    }


# --------------------------------------------------------------------------
# Basic success / parsing
# --------------------------------------------------------------------------


async def test_successful_completion_parses_text_and_tokens():
    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert payload["model"] == "test-model"
        assert payload["messages"] == [{"role": "user", "content": "hi"}]
        assert payload["max_tokens"] == 50
        return httpx.Response(200, json=success_body("hello there", prompt_tokens=12, completion_tokens=6))

    http_client = make_client(handler)
    client = OpenAICompatibleClient(
        base_url="https://example.test/v1", api_key="k", model="test-model", client=http_client
    )
    resp = await client.complete(LLMRequest(prompt="hi", max_tokens=50))
    assert resp.text == "hello there"
    assert resp.input_tokens == 12
    assert resp.output_tokens == 6
    await client.aclose()


async def test_request_model_override():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["model"] = json.loads(request.content)["model"]
        return httpx.Response(200, json=success_body())

    http_client = make_client(handler)
    client = OpenAICompatibleClient(
        base_url="https://example.test/v1", api_key="k", model="default-model", client=http_client
    )
    await client.complete(LLMRequest(prompt="hi", model="override-model"))
    assert seen["model"] == "override-model"
    await client.aclose()


async def test_logprobs_parsed_when_present():
    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert payload["logprobs"] is True
        assert payload["top_logprobs"] == 1
        return httpx.Response(
            200,
            json=success_body(logprobs=[("hello", -0.1), ("there", -0.3)]),
        )

    http_client = make_client(handler)
    client = OpenAICompatibleClient(
        base_url="https://example.test/v1", api_key="k", model="m", client=http_client
    )
    resp = await client.complete(LLMRequest(prompt="hi", needs_logprobs=True))
    assert resp.logprobs == [-0.1, -0.3]
    await client.aclose()


async def test_logprobs_none_when_absent():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=success_body())

    http_client = make_client(handler)
    client = OpenAICompatibleClient(
        base_url="https://example.test/v1", api_key="k", model="m", client=http_client
    )
    resp = await client.complete(LLMRequest(prompt="hi"))
    assert resp.logprobs is None
    await client.aclose()


async def test_strict_json_sets_response_format():
    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert payload["response_format"]["type"] == "json_schema"
        assert payload["response_format"]["json_schema"]["strict"] is True
        return httpx.Response(200, json=success_body())

    http_client = make_client(handler)
    client = OpenAICompatibleClient(
        base_url="https://example.test/v1", api_key="k", model="m", client=http_client
    )
    await client.complete(LLMRequest(prompt="hi", needs_strict_json=True))
    await client.aclose()


# --------------------------------------------------------------------------
# finish_reason
# --------------------------------------------------------------------------


async def test_finish_reason_stop_is_parsed():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=success_body("hello there", finish_reason="stop"))

    http_client = make_client(handler)
    client = OpenAICompatibleClient(
        base_url="https://example.test/v1", api_key="k", model="m", client=http_client
    )
    resp = await client.complete(LLMRequest(prompt="hi"))
    assert resp.finish_reason == "stop"
    assert resp.was_truncated is False
    await client.aclose()


async def test_finish_reason_length_is_parsed_and_exposed():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=success_body("partial answ", finish_reason="length"))

    http_client = make_client(handler)
    client = OpenAICompatibleClient(
        base_url="https://example.test/v1", api_key="k", model="m", client=http_client
    )
    resp = await client.complete(LLMRequest(prompt="hi", max_tokens=16))
    assert resp.finish_reason == "length"
    assert resp.was_truncated is True
    await client.aclose()


async def test_finish_reason_absent_yields_none_without_raising():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=success_body("hello", omit_finish_reason=True))

    http_client = make_client(handler)
    client = OpenAICompatibleClient(
        base_url="https://example.test/v1", api_key="k", model="m", client=http_client
    )
    resp = await client.complete(LLMRequest(prompt="hi"))
    assert resp.finish_reason is None
    assert resp.was_truncated is False
    await client.aclose()


async def test_reasoning_model_trap_distinguishable_from_legitimate_empty_answer():
    """A reasoning model starved of max_tokens returns content: "" with
    finish_reason: "length". That must be distinguishable from a model that
    legitimately produced nothing and stopped normally -- this is the whole
    point of threading finish_reason through at all.
    """

    def truncated_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=success_body("", finish_reason="length"))

    def legitimate_empty_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=success_body("", finish_reason="stop"))

    truncated_client = OpenAICompatibleClient(
        base_url="https://example.test/v1",
        api_key="k",
        model="m",
        client=make_client(truncated_handler),
    )
    legitimate_client = OpenAICompatibleClient(
        base_url="https://example.test/v1",
        api_key="k",
        model="m",
        client=make_client(legitimate_empty_handler),
    )

    truncated = await truncated_client.complete(LLMRequest(prompt="hi", max_tokens=16))
    legitimate = await legitimate_client.complete(LLMRequest(prompt="hi"))

    assert truncated.text == "" and legitimate.text == ""
    assert truncated.was_truncated is True
    assert legitimate.was_truncated is False

    await truncated_client.aclose()
    await legitimate_client.aclose()


async def test_mock_client_defaults_finish_reason_to_stop():
    client = MockClient(name="mock")
    resp = await client.complete(LLMRequest(prompt="hi"))
    assert resp.finish_reason == "stop"
    assert resp.was_truncated is False


async def test_mock_client_can_simulate_truncation():
    client = MockClient(name="mock", finish_reason="length")
    resp = await client.complete(LLMRequest(prompt="hi", max_tokens=16))
    assert resp.finish_reason == "length"
    assert resp.was_truncated is True

    batch = await MockClient(name="mock", finish_reason="length").complete_batch(
        [LLMRequest(prompt="hi", max_tokens=16)]
    )
    assert batch[0].finish_reason == "length"
    assert batch[0].was_truncated is True


# --------------------------------------------------------------------------
# Error mapping
# --------------------------------------------------------------------------


async def test_429_maps_to_rate_limit_error_with_retry_after():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            429,
            headers={"retry-after": "30"},
            json={"error": {"message": "slow down"}},
        )

    http_client = make_client(handler)
    client = OpenAICompatibleClient(
        base_url="https://example.test/v1", api_key="k", model="m", client=http_client
    )
    with pytest.raises(RateLimitError) as excinfo:
        await client.complete(LLMRequest(prompt="hi"))
    assert excinfo.value.status == 429
    assert excinfo.value.retry_after == 30.0
    assert "slow down" in str(excinfo.value)
    await client.aclose()


async def test_401_is_status_401_and_not_retried():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": {"message": "invalid api key"}})

    http_client = make_client(handler)
    client = OpenAICompatibleClient(
        base_url="https://example.test/v1", api_key="bad-key", model="m", client=http_client
    )
    with pytest.raises(ProviderError) as excinfo:
        await client.complete(LLMRequest(prompt="hi"))
    assert excinfo.value.status == 401
    assert "invalid api key" in str(excinfo.value)

    policy = RetryPolicy()
    assert policy.should_retry(excinfo.value, attempt=0) is False
    await client.aclose()


async def test_500_is_retryable():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": {"message": "server exploded"}})

    http_client = make_client(handler)
    client = OpenAICompatibleClient(
        base_url="https://example.test/v1", api_key="k", model="m", client=http_client
    )
    with pytest.raises(ProviderError) as excinfo:
        await client.complete(LLMRequest(prompt="hi"))
    assert excinfo.value.status == 500

    policy = RetryPolicy()
    assert policy.should_retry(excinfo.value, attempt=0) is True
    await client.aclose()


async def test_transport_timeout_is_status_none():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out", request=request)

    http_client = make_client(handler)
    client = OpenAICompatibleClient(
        base_url="https://example.test/v1", api_key="k", model="m", client=http_client
    )
    with pytest.raises(ProviderError) as excinfo:
        await client.complete(LLMRequest(prompt="hi"))
    assert excinfo.value.status is None

    policy = RetryPolicy()
    assert policy.should_retry(excinfo.value, attempt=0) is True
    await client.aclose()


async def test_transport_connect_error_is_status_none():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    http_client = make_client(handler)
    client = OpenAICompatibleClient(
        base_url="https://example.test/v1", api_key="k", model="m", client=http_client
    )
    with pytest.raises(ProviderError) as excinfo:
        await client.complete(LLMRequest(prompt="hi"))
    assert excinfo.value.status is None
    await client.aclose()


# --------------------------------------------------------------------------
# parse_retry_after
# --------------------------------------------------------------------------


def test_parse_retry_after_integer_seconds():
    assert parse_retry_after("30") == 30.0
    assert parse_retry_after("0") == 0.0


def test_parse_retry_after_float_seconds():
    assert parse_retry_after("7.5") == 7.5


def test_parse_retry_after_groq_style_durations():
    assert parse_retry_after("7.66s") == pytest.approx(7.66)
    assert parse_retry_after("2m59.56s") == pytest.approx(2 * 60 + 59.56)
    assert parse_retry_after("1h2m3s") == pytest.approx(3600 + 120 + 3)
    assert parse_retry_after("1h") == pytest.approx(3600)
    assert parse_retry_after("5m") == pytest.approx(300)


def test_parse_retry_after_http_date():
    future = datetime.now(UTC) + timedelta(seconds=120)
    header_value = format_datetime(future, usegmt=True)
    parsed = parse_retry_after(header_value)
    assert parsed is not None
    assert 110 <= parsed <= 130


def test_parse_retry_after_garbage_returns_none():
    assert parse_retry_after("") is None
    assert parse_retry_after("not-a-duration") is None
    assert parse_retry_after("   ") is None
    assert parse_retry_after("NaNs") is None


# --------------------------------------------------------------------------
# on_headers contract
# --------------------------------------------------------------------------


async def test_on_headers_called_on_success():
    captured: list[Mapping[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"x-ratelimit-remaining": "99"}, json=success_body())

    http_client = make_client(handler)
    client = OpenAICompatibleClient(
        base_url="https://example.test/v1",
        api_key="k",
        model="m",
        client=http_client,
        on_headers=lambda h: captured.append(dict(h)),
    )
    await client.complete(LLMRequest(prompt="hi"))
    assert len(captured) == 1
    assert captured[0]["x-ratelimit-remaining"] == "99"
    await client.aclose()


async def test_on_headers_called_on_error_and_swallows_consumer_exceptions():
    captured: list[Mapping[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, headers={"retry-after": "5"}, json={"error": {"message": "no"}})

    def bad_on_headers(headers: Mapping[str, str]) -> None:
        captured.append(dict(headers))
        raise RuntimeError("consumer bug")

    http_client = make_client(handler)
    client = OpenAICompatibleClient(
        base_url="https://example.test/v1",
        api_key="k",
        model="m",
        client=http_client,
        on_headers=bad_on_headers,
    )
    with pytest.raises(RateLimitError):
        await client.complete(LLMRequest(prompt="hi"))
    assert len(captured) == 1
    assert captured[0]["retry-after"] == "5"
    await client.aclose()


# --------------------------------------------------------------------------
# aclose / context manager ownership
# --------------------------------------------------------------------------


async def test_aclose_closes_self_built_client():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=success_body())

    client = OpenAICompatibleClient(base_url="https://example.test/v1", api_key="k", model="m")
    # Swap in a mock transport on the client it built for itself.
    client._client._transport = httpx.MockTransport(handler)
    await client.complete(LLMRequest(prompt="hi"))
    await client.aclose()
    assert client._client.is_closed


async def test_aclose_does_not_close_injected_client():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=success_body())

    http_client = make_client(handler)
    client = OpenAICompatibleClient(
        base_url="https://example.test/v1", api_key="k", model="m", client=http_client
    )
    async with client:
        await client.complete(LLMRequest(prompt="hi"))
    assert not http_client.is_closed
    await http_client.aclose()


# --------------------------------------------------------------------------
# Groq adapter
# --------------------------------------------------------------------------


async def test_groq_does_not_send_logprobs_params():
    # Groq's capabilities mark supports_logprobs=False, so the router would
    # never send a needs_logprobs request here in practice -- but the
    # adapter itself must not silently inject logprobs params if asked
    # directly, since Groq's API rejects them.
    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert "logprobs" not in payload
        assert "top_logprobs" not in payload
        return httpx.Response(200, json=success_body())

    http_client = make_client(handler)
    client = GroqClient(base_url=GROQ_BASE_URL, api_key="k", model="llama-3.3-70b-versatile", client=http_client)
    client.logprobs_params = lambda: {}  # type: ignore[method-assign]
    await client.complete(LLMRequest(prompt="hi", needs_logprobs=True))
    await client.aclose()


def test_groq_provider_capabilities():
    provider = groq_provider(api_key="k", client=GroqClient(
        base_url=GROQ_BASE_URL, api_key="k", model="m", client=make_client(lambda r: httpx.Response(200))
    ))
    assert provider.capabilities.supports_logprobs is False
    assert provider.capabilities.supports_strict_json is True
    assert provider.capabilities.supports_batching is False
    assert provider.name == "groq"


# --------------------------------------------------------------------------
# OpenRouter adapter
# --------------------------------------------------------------------------


async def test_openrouter_sends_require_parameters_when_logprobs_needed():
    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert payload["provider"] == {"require_parameters": True}
        return httpx.Response(200, json=success_body(logprobs=[("a", -0.05)]))

    http_client = make_client(handler)
    client = OpenRouterClient(
        base_url=OPENROUTER_BASE_URL, api_key="k", model="some/model", client=http_client
    )
    await client.complete(LLMRequest(prompt="hi", needs_logprobs=True))
    await client.aclose()


async def test_openrouter_sends_require_parameters_when_strict_json_needed():
    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert payload["provider"] == {"require_parameters": True}
        return httpx.Response(200, json=success_body())

    http_client = make_client(handler)
    client = OpenRouterClient(
        base_url=OPENROUTER_BASE_URL, api_key="k", model="some/model", client=http_client
    )
    await client.complete(LLMRequest(prompt="hi", needs_strict_json=True))
    await client.aclose()


async def test_openrouter_omits_require_parameters_for_plain_request():
    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert "provider" not in payload
        return httpx.Response(200, json=success_body())

    http_client = make_client(handler)
    client = OpenRouterClient(
        base_url=OPENROUTER_BASE_URL, api_key="k", model="some/model", client=http_client
    )
    await client.complete(LLMRequest(prompt="hi"))
    await client.aclose()


async def test_openrouter_402_out_of_credit_is_not_retryable():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(402, json={"error": {"message": "insufficient credits"}})

    http_client = make_client(handler)
    client = OpenRouterClient(
        base_url=OPENROUTER_BASE_URL, api_key="k", model="some/model", client=http_client
    )
    with pytest.raises(ProviderError) as excinfo:
        await client.complete(LLMRequest(prompt="hi"))
    assert excinfo.value.status == 402
    assert "insufficient credits" in str(excinfo.value)

    policy = RetryPolicy()
    assert policy.should_retry(excinfo.value, attempt=0) is False
    await client.aclose()


async def test_openrouter_402_notifies_headers_exactly_once():
    """Regression test: OpenRouterClient._raise_for_status used to call
    _notify_headers a second time for a 402, on top of the call complete()
    already makes for every response -- double-invoking on_headers (and
    therefore the limiter's sync_from_headers) for the same response."""
    calls: list[Mapping[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            402,
            json={"error": {"message": "insufficient credits"}},
            headers={"x-ratelimit-remaining-requests": "5"},
        )

    http_client = make_client(handler)
    client = OpenRouterClient(
        base_url=OPENROUTER_BASE_URL,
        api_key="k",
        model="some/model",
        client=http_client,
        on_headers=lambda h: calls.append(dict(h)),
    )
    with pytest.raises(ProviderError):
        await client.complete(LLMRequest(prompt="hi"))
    assert len(calls) == 1
    await client.aclose()


def test_openrouter_provider_capabilities():
    http_client = make_client(lambda r: httpx.Response(200))
    client = OpenRouterClient(base_url=OPENROUTER_BASE_URL, api_key="k", model="m", client=http_client)
    provider = openrouter_provider(api_key="k", model="some/model", client=client)
    assert provider.capabilities.supports_batching is False
    assert provider.name == "openrouter"


# --------------------------------------------------------------------------
# End-to-end through LLMGateway
# --------------------------------------------------------------------------


async def test_gateway_end_to_end_over_mock_transport():
    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        return httpx.Response(
            200,
            json=success_body(f"echo: {payload['messages'][0]['content']}"),
        )

    http_client = make_client(handler)
    client = GroqClient(base_url=GROQ_BASE_URL, api_key="k", model="m", client=http_client)
    provider = groq_provider(api_key="k", client=client)

    gateway = LLMGateway([provider])
    try:
        resp = await gateway.submit(LLMRequest(prompt="ping"))
        assert resp.text == "echo: ping"
        assert resp.provider == "groq"
    finally:
        await gateway.aclose()
        await client.aclose()


async def test_gateway_end_to_end_retries_through_transient_5xx():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(503, json={"error": {"message": "overloaded"}})
        return httpx.Response(200, json=success_body("recovered"))

    http_client = make_client(handler)
    client = OpenAICompatibleClient(
        base_url="https://example.test/v1", api_key="k", model="m", client=http_client
    )
    provider = groq_provider(api_key="k", client=client)

    gateway = LLMGateway([provider], retry=RetryPolicy(base_delay=0.001, max_delay=0.01))
    try:
        resp = await gateway.submit(LLMRequest(prompt="ping"))
        assert resp.text == "recovered"
        assert calls["n"] == 2
    finally:
        await gateway.aclose()
        await client.aclose()


# ---------------------------------------------------------------------------
# The seam: adapters must feed provider rate-limit headers back into the
# local token buckets. Each half can be perfect and the pair still useless
# if nothing connects them, so this is tested explicitly.
# ---------------------------------------------------------------------------


def _headered_handler(headers: dict[str, str]):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers=headers,
            json={
                "choices": [{"message": {"content": "ok"}}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 3},
            },
        )

    return handler


async def test_groq_provider_syncs_buckets_from_response_headers():
    from llm_gateway.providers.groq import GroqClient, groq_provider

    http_client = make_client(
        _headered_handler(
            {
                "x-ratelimit-remaining-requests": "7",
                "x-ratelimit-remaining-tokens": "250",
            }
        ),
        base_url="https://api.groq.com/openai/v1",
    )
    groq_client = GroqClient(
        base_url="https://api.groq.com/openai/v1",
        api_key="k",
        model="m",
        client=http_client,
    )
    provider = groq_provider(
        api_key="k", rpm_limit=30, tpm_limit=6000, client=groq_client
    )
    assert provider.limiter.requests.available > 7

    await provider.complete(LLMRequest("hello"))
    await http_client.aclose()

    # The server said 7 requests and 250 tokens remain; the local buckets,
    # which started full, must now agree.
    assert provider.limiter.requests.available == pytest.approx(7, abs=0.01)
    assert provider.limiter.tokens.available == pytest.approx(250, abs=1)


async def test_openrouter_provider_syncs_buckets_from_response_headers():
    from llm_gateway.providers.openrouter import OpenRouterClient, openrouter_provider

    http_client = make_client(
        _headered_handler({"x-ratelimit-remaining-requests": "3"}),
        base_url="https://openrouter.ai/api/v1",
    )
    or_client = OpenRouterClient(
        base_url="https://openrouter.ai/api/v1",
        api_key="k",
        model="m",
        client=http_client,
    )
    provider = openrouter_provider(
        api_key="k", model="m", rpm_limit=100, client=or_client
    )

    await provider.complete(LLMRequest("hello"))
    await http_client.aclose()

    assert provider.limiter.requests.available == pytest.approx(3, abs=0.01)


async def test_user_supplied_on_headers_is_chained_not_replaced():
    """Passing on_headers must observe headers, not silently disable sync."""
    from llm_gateway.providers.groq import GroqClient, groq_provider

    seen: list[dict[str, str]] = []
    http_client = make_client(
        _headered_handler({"x-ratelimit-remaining-requests": "2"}),
        base_url="https://api.groq.com/openai/v1",
    )
    groq_client = GroqClient(
        base_url="https://api.groq.com/openai/v1",
        api_key="k",
        model="m",
        client=http_client,
    )
    provider = groq_provider(
        api_key="k",
        rpm_limit=50,
        client=groq_client,
        on_headers=lambda h: seen.append(dict(h)),
    )

    await provider.complete(LLMRequest("hello"))
    await http_client.aclose()

    assert seen, "user callback was not called"
    assert provider.limiter.requests.available == pytest.approx(2, abs=0.01)
