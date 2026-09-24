"""Targeted tests for error paths and edge cases not covered elsewhere.

Each test here exists to close a specific coverage gap found by running
`pytest --cov=llm_gateway --cov-report=term-missing`: negative-value guards,
provider-factory callback chaining, HTTP transport failures on a secondary
code path, and queue/rate-limit edge cases. Entirely offline and
deterministic.
"""

from __future__ import annotations

import httpx
import pytest

from llm_gateway import LLMGateway, LLMRequest, ProviderError, RateLimitError
from llm_gateway.budget import BudgetLedger
from llm_gateway.providers.http import OpenAICompatibleClient
from llm_gateway.providers.mock import MockClient
from llm_gateway.providers.openrouter import OpenRouterClient, openrouter_provider
from llm_gateway.queue import RequestQueue


def make_client(handler, *, base_url: str = "https://example.test/v1", **kwargs):
    transport = httpx.MockTransport(handler)
    return httpx.AsyncClient(transport=transport, base_url=base_url, **kwargs)


# --------------------------------------------------------------------------
# budget.py -- negative-value guards on settle
# --------------------------------------------------------------------------


def test_settle_rejects_negative_actual_usd():
    ledger = BudgetLedger(5.0)
    r = ledger.reserve(1.0)
    with pytest.raises(ValueError, match="actual_usd must be >= 0"):
        r.settle(-1.0)
    # The reservation is still open -- the rejected settle must not have
    # mutated the ledger.
    assert ledger.outstanding_count == 1
    r.release()


def test_settle_rejects_negative_actual_tokens():
    ledger = BudgetLedger(5.0, max_tokens_total=1000)
    r = ledger.reserve(1.0, estimated_tokens=100)
    with pytest.raises(ValueError, match="actual_tokens must be >= 0"):
        r.settle(1.0, actual_tokens=-5)
    # Nothing changed: the reservation is still open and can be released,
    # which frees both the dollar and the token holds.
    assert ledger.outstanding_count == 1
    r.release()
    assert ledger.committed_tokens == 0
    assert ledger.spent == 0


def test_remaining_tokens_never_goes_negative_when_overcommitted():
    # remaining_tokens clamps to 0 even if committed_tokens could exceed
    # the ceiling through settle() overrides larger than the estimate.
    ledger = BudgetLedger(100.0, max_tokens_total=100)
    r = ledger.reserve(1.0, estimated_tokens=50)
    r.settle(1.0, actual_tokens=500)
    assert ledger.remaining_tokens == 0


# --------------------------------------------------------------------------
# gateway.py -- construction guard
# --------------------------------------------------------------------------


def test_gateway_requires_at_least_one_provider():
    with pytest.raises(ValueError, match="at least one provider is required"):
        LLMGateway(providers=[])


# --------------------------------------------------------------------------
# providers/openrouter.py -- user on_headers callback is chained
# --------------------------------------------------------------------------


async def test_openrouter_user_supplied_on_headers_is_chained():
    seen: list[dict[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"x-ratelimit-remaining-requests": "3"},
            json={
                "choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            },
        )

    http_client = make_client(handler, base_url="https://openrouter.ai/api/v1")
    or_client = OpenRouterClient(
        base_url="https://openrouter.ai/api/v1",
        api_key="k",
        model="m",
        client=http_client,
    )
    provider = openrouter_provider(
        api_key="k",
        model="m",
        rpm_limit=100,
        client=or_client,
        on_headers=lambda h: seen.append(dict(h)),
    )

    await provider.complete(LLMRequest("hello"))
    await http_client.aclose()

    assert seen, "user-supplied on_headers callback was not invoked"
    assert provider.limiter.requests.available == pytest.approx(3, abs=0.01)


# --------------------------------------------------------------------------
# providers/http.py -- list_models transport failures
# --------------------------------------------------------------------------


async def test_list_models_timeout_maps_to_provider_error():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out", request=request)

    http_client = make_client(handler)
    client = OpenAICompatibleClient(
        base_url="https://example.test/v1", api_key="k", model="m", client=http_client
    )
    with pytest.raises(ProviderError) as excinfo:
        await client.list_models()
    assert excinfo.value.status is None
    assert "timed out" in str(excinfo.value)
    await client.aclose()


async def test_list_models_transport_error_maps_to_provider_error():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    http_client = make_client(handler)
    client = OpenAICompatibleClient(
        base_url="https://example.test/v1", api_key="k", model="m", client=http_client
    )
    with pytest.raises(ProviderError) as excinfo:
        await client.list_models()
    assert excinfo.value.status is None
    assert "transport error" in str(excinfo.value)
    await client.aclose()


# --------------------------------------------------------------------------
# providers/http.py -- _error_message fallback branches
# --------------------------------------------------------------------------


async def test_error_message_falls_back_to_plain_string_error_field():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": "bad request string"})

    http_client = make_client(handler)
    client = OpenAICompatibleClient(
        base_url="https://example.test/v1", api_key="k", model="m", client=http_client
    )
    with pytest.raises(ProviderError) as excinfo:
        await client.complete(LLMRequest(prompt="hi"))
    assert "bad request string" in str(excinfo.value)
    await client.aclose()


async def test_error_message_falls_back_to_top_level_message_field():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"message": "top level message"})

    http_client = make_client(handler)
    client = OpenAICompatibleClient(
        base_url="https://example.test/v1", api_key="k", model="m", client=http_client
    )
    with pytest.raises(ProviderError) as excinfo:
        await client.complete(LLMRequest(prompt="hi"))
    assert "top level message" in str(excinfo.value)
    await client.aclose()


async def test_error_message_falls_back_to_raw_text_when_not_json():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, content=b"internal server error, not json")

    http_client = make_client(handler)
    client = OpenAICompatibleClient(
        base_url="https://example.test/v1", api_key="k", model="m", client=http_client
    )
    with pytest.raises(ProviderError) as excinfo:
        await client.complete(LLMRequest(prompt="hi"))
    assert "internal server error" in str(excinfo.value)
    await client.aclose()


# --------------------------------------------------------------------------
# providers/http.py -- complete_batch sequential fallback
# --------------------------------------------------------------------------


async def test_complete_batch_sequential_fallback_preserves_order():
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = httpx.Request.read(request)
        import json as _json

        payload = _json.loads(body)
        prompt = payload["messages"][-1]["content"]
        calls.append(prompt)
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": prompt}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            },
        )

    http_client = make_client(handler)
    client = OpenAICompatibleClient(
        base_url="https://example.test/v1", api_key="k", model="m", client=http_client
    )
    requests = [LLMRequest(prompt=f"p{i}") for i in range(3)]
    responses = await client.complete_batch(requests)
    assert [r.text for r in responses] == ["p0", "p1", "p2"]
    assert calls == ["p0", "p1", "p2"]
    await client.aclose()


# --------------------------------------------------------------------------
# providers/mock.py -- injected rate limit
# --------------------------------------------------------------------------


async def test_mock_client_enforces_its_own_rpm_limit():
    client = MockClient(name="capped", rpm_limit=1)
    await client.complete(LLMRequest(prompt="first"))
    with pytest.raises(RateLimitError):
        await client.complete(LLMRequest(prompt="second"))


# --------------------------------------------------------------------------
# queue.py -- pop() edge cases
# --------------------------------------------------------------------------


async def test_pop_returns_none_immediately_when_deadline_already_passed():
    # timeout=0 means the deadline has already elapsed by the time the
    # empty-queue re-check runs, so pop() must return None without ever
    # reaching asyncio.wait_for.
    q = RequestQueue()
    assert await q.pop(timeout=0) is None


async def test_pop_catches_item_that_arrives_between_check_and_clear():
    import asyncio

    q = RequestQueue()

    real_clear = q._arrival.clear
    seen: list[bool] = []

    def racing_clear() -> None:
        # Simulate a producer landing an item in the gap between the first
        # pop_nowait() and the Event.clear() call inside pop().
        if not seen:
            seen.append(True)
            loop = asyncio.get_running_loop()
            from llm_gateway.types import LLMRequest as _Req
            from llm_gateway.types import QueuedRequest as _Q

            q.put(
                _Q(
                    request=_Req(prompt="raced-in"),
                    future=loop.create_future(),
                    enqueued_at=0.0,
                )
            )
        real_clear()

    q._arrival.clear = racing_clear  # type: ignore[method-assign]
    result = await q.pop(timeout=1.0)
    assert result is not None
    assert result.request.prompt == "raced-in"
