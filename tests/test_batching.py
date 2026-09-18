"""Batcher tests run on real time with millisecond windows.

The fake clock is deliberately not used here: the batcher's correctness
depends on the interaction between asyncio timeouts and arrivals, and
faking that away would test a different program. Windows are kept in the
20-60ms range so the whole file still runs in well under a second.
"""

from __future__ import annotations

import asyncio
import time

from llm_gateway.batching import Batcher
from llm_gateway.queue import RequestQueue
from llm_gateway.types import LLMRequest, QueuedRequest


def entry(prompt: str = "hi", priority: int = 0, max_tokens: int = 10) -> QueuedRequest:
    loop = asyncio.get_running_loop()
    return QueuedRequest(
        request=LLMRequest(prompt=prompt, priority=priority, max_tokens=max_tokens),
        future=loop.create_future(),
        enqueued_at=time.monotonic(),
    )


async def test_flush_on_size():
    q = RequestQueue()
    b = Batcher(max_batch_size=4, max_wait_ms=5_000, in_flight=lambda: 1)
    for _ in range(10):
        q.put(entry())
    batch = await b.collect(q)
    assert len(batch) == 4


async def test_flush_on_token_budget():
    q = RequestQueue()
    # Each request is ~250 prompt tokens + 100 max_tokens = 350.
    b = Batcher(
        max_batch_size=100,
        max_wait_ms=5_000,
        max_batch_tokens=1000,
        in_flight=lambda: 1,
    )
    for _ in range(20):
        q.put(entry(prompt="x" * 1000, max_tokens=100))
    batch = await b.collect(q)
    assert 1 < len(batch) <= 4
    assert sum(e.request.estimated_total_tokens() for e in batch) >= 1000


async def test_flush_on_oldest_request_timeout():
    q = RequestQueue()
    b = Batcher(max_batch_size=100, max_wait_ms=40, in_flight=lambda: 1)
    q.put(entry())
    start = time.monotonic()
    batch = await b.collect(q)
    elapsed = time.monotonic() - start
    assert len(batch) == 1
    assert 0.02 < elapsed < 0.25


async def test_single_request_into_empty_queue_dispatches_immediately():
    q = RequestQueue()
    b = Batcher(max_batch_size=16, max_wait_ms=5_000)  # in_flight defaults to 0
    q.put(entry())
    start = time.monotonic()
    batch = await b.collect(q)
    elapsed = time.monotonic() - start
    assert len(batch) == 1
    assert elapsed < 0.05, "adaptive dispatch should not wait out max_wait_ms"


async def test_does_not_flush_early_while_work_is_in_flight():
    """Empty-queue alone must not trigger the adaptive flush."""
    q = RequestQueue()
    b = Batcher(max_batch_size=8, max_wait_ms=60, in_flight=lambda: 3)
    q.put(entry())

    async def trickle():
        for _ in range(3):
            await asyncio.sleep(0.008)
            q.put(entry())

    task = asyncio.create_task(trickle())
    batch = await b.collect(q)
    await task
    assert len(batch) == 4


async def test_slow_trickle_cannot_starve_the_oldest_request():
    """The deadline is anchored to the oldest request, so a steady trickle
    of arrivals must not push the flush out indefinitely."""
    q = RequestQueue()
    b = Batcher(max_batch_size=1000, max_wait_ms=50, in_flight=lambda: 1)
    oldest = entry("oldest")
    q.put(oldest)

    stop = False

    async def trickle():
        while not stop:
            await asyncio.sleep(0.005)
            q.put(entry())

    task = asyncio.create_task(trickle())
    start = time.monotonic()
    batch = await b.collect(q)
    elapsed = time.monotonic() - start
    stop = True
    task.cancel()

    assert batch[0] is oldest
    assert elapsed < 0.3, "timer must not reset on every arrival"


async def test_collect_blocks_until_the_first_request_arrives():
    q = RequestQueue()
    b = Batcher(max_batch_size=4, max_wait_ms=20)

    async def later():
        await asyncio.sleep(0.03)
        q.put(entry("late"))

    task = asyncio.create_task(later())
    batch = await asyncio.wait_for(b.collect(q), 1.0)
    await task
    assert len(batch) == 1
    assert batch[0].request.prompt == "late"


async def test_deadline_already_passed_flushes_without_another_wait():
    """A request that sat in the queue past max_wait_ms goes out immediately."""
    q = RequestQueue()
    b = Batcher(max_batch_size=100, max_wait_ms=10, in_flight=lambda: 1)
    stale = entry("stale")
    stale.enqueued_at = time.monotonic() - 5.0  # enqueued long ago
    q.put(stale)
    q.put(entry("also-waiting"))

    start = time.monotonic()
    batch = await b.collect(q)
    assert time.monotonic() - start < 0.02
    assert batch[0] is stale


async def test_batch_does_not_mix_capability_requirements():
    """One logprobs request must not drag fifteen others onto a strict
    provider -- that silently destroys the point of having a provider mix."""
    q = RequestQueue()
    b = Batcher(max_batch_size=16, max_wait_ms=30, in_flight=lambda: 1)

    plain_a = entry("plain-a")
    q.put(plain_a)
    special = entry("special")
    special.request.needs_logprobs = True
    q.put(special)
    q.put(entry("plain-b"))

    batch = await b.collect(q)
    prompts = [e.request.prompt for e in batch]
    assert prompts == ["plain-a", "plain-b"]
    # The mismatched request is requeued, not dropped.
    assert len(q) == 1
    assert (await q.pop()).request.prompt == "special"


async def test_highest_priority_request_sets_the_batch_capability():
    """The head of the queue decides the batch's requirements; everything
    incompatible is set aside and requeued with its priority intact."""
    q = RequestQueue()
    b = Batcher(max_batch_size=8, max_wait_ms=20, in_flight=lambda: 1)
    urgent = entry("urgent", priority=10)
    urgent.request.needs_strict_json = True
    q.put(entry("plain-1"))
    q.put(urgent)
    q.put(entry("plain-2"))

    batch = await b.collect(q)
    assert [e.request.prompt for e in batch] == ["urgent"]
    # The deferred pair went back, still ahead of nothing and behind nobody.
    assert len(q) == 2
    assert [(await q.pop()).request.prompt for _ in range(2)] == ["plain-1", "plain-2"]
