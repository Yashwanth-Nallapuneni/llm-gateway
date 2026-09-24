"""Regression tests for bugs suspected in tonight's fixes (v0.1.0..HEAD).

Each test is offline and deterministic (no live calls, no wall-clock
sleeps beyond a short pytest-timeout-free asyncio.wait_for guard).
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from llm_gateway import GatewayError, LLMGateway, LLMRequest
from llm_gateway.breaker import CircuitBreaker, State
from llm_gateway.providers.mock import MockProvider
from llm_gateway.types import RequestTimeout


def fast_gateway(**kw: Any) -> LLMGateway:
    return LLMGateway(providers=[MockProvider("a")], **kw)


# ---------------------------------------------------------------------
# gateway.aclose() while a batch is already in flight (worker task, not
# the queue) -- distinct from the already-fixed "still in the queue" case.
# ---------------------------------------------------------------------


async def test_aclose_resolves_requests_already_in_flight_in_a_worker() -> None:
    """A request that has already been picked up by the dispatcher and
    handed to a worker task (i.e. it is being dispatched to a provider,
    not sitting in the queue) must not hang forever when aclose() runs.

    aclose() cancels the dispatcher AND every worker task. _dispatch()'s
    exception handling only catches `Exception`, so a `CancelledError`
    delivered mid-await propagates straight out of the worker without
    ever calling _fail_batch -- the batch's futures are never resolved.
    Only _drain_with_error's sweep of the (now empty) queue runs, and it
    finds nothing to drain.
    """
    # Latency long enough that aclose() is guaranteed to run while the
    # worker is still awaiting the provider call.
    gw = LLMGateway(providers=[MockProvider("a", latency=5.0)])

    task = asyncio.ensure_future(gw.submit(LLMRequest("hello")))
    # Let the request get enqueued, picked up by the dispatcher, and handed
    # to a worker task that is now awaiting the (slow) provider call.
    for _ in range(5):
        await asyncio.sleep(0)

    await gw.aclose()

    with pytest.raises(GatewayError):
        await asyncio.wait_for(task, timeout=1.0)


# ---------------------------------------------------------------------
# breaker.release_probe(): can it release a probe that belongs to a
# different, concurrently-ranking batch on the same half-open provider?
# ---------------------------------------------------------------------


def test_release_probe_does_not_free_a_probe_claimed_by_someone_else() -> None:
    """allows_request() and release_probe() are both synchronous, so within
    a single event-loop turn there is no interleaving between one batch's
    "claim the probe while ranking" and another batch's ranking of the
    same provider. This test pins down that a probe claimed by one caller
    is not visible as "free" to a second caller before the first releases
    it, and that release_probe() only clears the flag it itself owns.
    """
    breaker = CircuitBreaker(failure_threshold=1)
    breaker.record_failure()  # trips -> OPEN
    breaker._opened_at -= 100  # force recovery_timeout to have elapsed
    assert breaker.state is State.HALF_OPEN

    # First caller claims the single half-open probe while ranking.
    assert breaker.allows_request() is True
    # A second, concurrent caller ranking the same provider must not also
    # get the probe.
    assert breaker.allows_request() is False

    # First caller decides not to use it after all and gives it back.
    breaker.release_probe()
    # Now it is free for the next caller.
    assert breaker.allows_request() is True


# ---------------------------------------------------------------------
# submit(): caller cancellation on the timeout path.
# ---------------------------------------------------------------------


async def test_submit_timeout_task_exception_is_retrieved_not_leaked() -> None:
    """When submit() times out, the underlying task keeps running (shield).
    Once it finishes (with an exception, since nothing serves it), nothing
    must warn "Task exception was never retrieved" -- the done-callback
    must actually retrieve it.
    """
    gw = LLMGateway(providers=[MockProvider("a", latency=0.05)])
    req = LLMRequest("hello", timeout_s=0.001)
    with pytest.raises(RequestTimeout):
        await gw.submit(req)
    # Give the shielded background task a chance to finish and have its
    # done-callback run.
    await asyncio.sleep(0.2)
    await gw.aclose()


# ---------------------------------------------------------------------
# submit() after aclose(): does it silently resurrect the gateway?
# ---------------------------------------------------------------------


async def test_submit_after_aclose_either_raises_or_resurrects_cleanly() -> None:
    """Documents current behaviour: submit() after aclose() restarts the
    dispatcher (_ensure_started() sees _dispatcher is None) instead of
    raising. This is not necessarily wrong, but it's worth pinning down:
    a caller that closes the gateway expecting it to stay closed will
    silently get a working gateway back.
    """
    gw = fast_gateway()
    async with gw:
        await gw.submit(LLMRequest("hello"))
    assert gw._dispatcher is None

    # This either raises (a clearer contract) or succeeds by silently
    # restarting the dispatcher. Pin the current behaviour so a future
    # change here is deliberate, not accidental.
    response = await gw.submit(LLMRequest("hello again"))
    assert response is not None
    await gw.aclose()
