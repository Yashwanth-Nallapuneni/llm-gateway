"""Regression tests for gateway lifecycle bugs found in review.

Both bugs share the same shape: a request's future is left unresolved
forever because nothing on the failure path calls set_exception/set_result
on it. Before the corresponding fix, each test below hangs (and would only
be caught by pytest-timeout / CI wall-clock, not by an assertion) -- so each
test wraps the awaited call in `asyncio.wait_for` with a short deadline and
asserts it resolves, not just that it eventually raises the right thing.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from llm_gateway import GatewayError, LLMGateway, LLMRequest
from llm_gateway.batching import Batcher
from llm_gateway.providers.mock import MockProvider


def fast_gateway(**kw: Any) -> LLMGateway:
    return LLMGateway(providers=[MockProvider("a")], **kw)


async def test_aclose_resolves_requests_still_sitting_in_the_queue() -> None:
    """A request enqueued but not yet picked up by the dispatcher must not
    hang forever when the gateway is closed.

    Use a batcher with a long max_wait_ms so the first submitted request is
    guaranteed to still be waiting in the queue (not yet dispatched) when
    aclose() runs immediately after submit() starts.
    """
    gw = fast_gateway(batcher=Batcher(max_wait_ms=5_000))
    task = asyncio.ensure_future(gw.submit(LLMRequest("hello")))
    # Let the event loop start the submit coroutine (enqueue the entry and
    # start the dispatcher) without letting the batcher's long wait elapse.
    await asyncio.sleep(0)
    await gw.aclose()

    with pytest.raises(GatewayError):
        await asyncio.wait_for(task, timeout=1.0)


async def test_dispatch_bug_fails_batch_instead_of_hanging(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unexpected exception inside the dispatcher's per-batch handling
    (anything not already one of the modeled failure paths) must resolve
    the batch's futures with that exception, not leave them pending.
    """
    gw = fast_gateway()

    def boom(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("boom: unexpected routing bug")

    monkeypatch.setattr(gw.router, "select_all", boom)

    async with gw:
        with pytest.raises(RuntimeError, match="boom"):
            await asyncio.wait_for(gw.submit(LLMRequest("hello")), timeout=1.0)
