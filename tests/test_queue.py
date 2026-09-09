from __future__ import annotations

import asyncio
import time

import pytest

from llm_gateway.queue import RequestQueue
from llm_gateway.types import LLMRequest, QueuedRequest


def entry(prompt: str, priority: int = 0) -> QueuedRequest:
    loop = asyncio.get_running_loop()
    return QueuedRequest(
        request=LLMRequest(prompt=prompt, priority=priority),
        future=loop.create_future(),
        enqueued_at=time.monotonic(),
    )


async def test_priority_ordering_with_fifo_tiebreak():
    q = RequestQueue()
    for name, pri in [("a", 0), ("b", 5), ("c", 0), ("d", 5), ("e", 9)]:
        q.put(entry(name, pri))
    order = [(await q.pop()).request.prompt for _ in range(5)]
    assert order == ["e", "b", "d", "a", "c"]


async def test_equal_priorities_never_compare_payloads():
    """Without the seq tiebreak this raises TypeError inside heapq."""
    q = RequestQueue()
    for i in range(50):
        q.put(entry(f"r{i}", priority=1))
    assert len(q) == 50
    assert (await q.pop()).request.prompt == "r0"


async def test_pop_times_out_on_an_empty_queue():
    q = RequestQueue()
    assert await q.pop(timeout=0.02) is None


async def test_pop_wakes_on_arrival():
    q = RequestQueue()

    async def later():
        await asyncio.sleep(0.01)
        q.put(entry("x"))

    asyncio.create_task(later())
    got = await q.pop(timeout=1.0)
    assert got is not None and got.request.prompt == "x"
