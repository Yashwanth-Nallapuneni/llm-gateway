"""Priority request queue."""

from __future__ import annotations

import asyncio
import heapq
import time

from .types import QueuedRequest


class RequestQueue:
    """Max-heap by priority, FIFO within a priority."""

    def __init__(self) -> None:
        self._heap: list[tuple[int, int, QueuedRequest]] = []
        # Signals "the heap went from empty to non-empty". Waiters clear it
        # before sleeping so they never miss an arrival that happened while
        # they were between checks.
        self._arrival = asyncio.Event()

    def put(self, item: QueuedRequest) -> None:
        # Sort key is (-priority, seq). Negating priority turns the min-heap
        # into a max-heap, so higher priority comes out first. seq breaks
        # ties in arrival order and also stops heapq from ever comparing two
        # QueuedRequest objects directly, which would raise a confusing
        # TypeError on a priority tie.
        heapq.heappush(self._heap, (-item.priority, item.seq, item))
        self._arrival.set()

    def pop_nowait(self) -> QueuedRequest | None:
        if not self._heap:
            return None
        return heapq.heappop(self._heap)[2]

    async def pop(
        self,
        timeout: float | None = None,  # noqa: ASYNC109
    ) -> QueuedRequest | None:
        """Pop the highest-priority item, waiting up to `timeout` seconds.

        Returns None if the timeout expired with the queue still empty.
        The `timeout` parameter is kept (instead of asking callers to wrap
        this in `asyncio.timeout()`) because the deadline must survive
        several loop iterations here, and callers like `batching.py` rely
        on a plain None return for "nothing arrived in time".
        """
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            item = self.pop_nowait()
            if item is not None:
                return item

            self._arrival.clear()
            # Check again after clearing: an item could have arrived in the
            # gap between the check above and this clear(), and clearing
            # would otherwise throw away the only wakeup signal for it.
            item = self.pop_nowait()
            if item is not None:
                return item

            if deadline is None:
                await self._arrival.wait()
                continue

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            try:
                await asyncio.wait_for(self._arrival.wait(), remaining)
            except TimeoutError:
                return None

    def empty(self) -> bool:
        return not self._heap

    def __len__(self) -> int:
        return len(self._heap)
