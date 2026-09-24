"""Priority request queue.

Normal comments here -- this module is plumbing, not one of the four
concepts the project exists to teach.
"""

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
        # The key is (-priority, seq).
        #
        # -priority: heapq is a min-heap, so negating turns it into a max-heap
        # and higher-priority requests come out first.
        #
        # seq: a monotonic counter, and it is doing two jobs. It makes equal
        # priorities come out in arrival order (FIFO), and -- more subtly -- it
        # guarantees the tuple comparison never reaches the third element. Two
        # requests with the same priority would otherwise make heapq compare
        # QueuedRequest objects, which raises TypeError deep inside heapq with
        # a traceback that points at the standard library instead of at you.
        # That failure only shows up under load, when a priority tie finally
        # happens, which is what makes it confusing in the wild.
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

        ASYNC109 wants callers to wrap the call in `asyncio.timeout()`
        instead of passing a `timeout` here. That does not fit this method:
        the deadline has to survive several iterations of the retry loop
        below (re-checking `pop_nowait()` after each partial wait), and on
        expiry this returns None rather than raising -- callers such as
        `batching.py` depend on that to mean "no item arrived in time", not
        "something failed". Switching to `asyncio.timeout()` would turn a
        normal, expected outcome into a caught `TimeoutError` at every call
        site for no behavioural benefit.
        """
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            item = self.pop_nowait()
            if item is not None:
                return item

            self._arrival.clear()
            # Re-check after clearing: an item could have arrived between the
            # pop_nowait() above and the clear(), and clearing would then have
            # discarded the only wakeup we were going to get.
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
