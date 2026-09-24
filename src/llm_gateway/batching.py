"""Request batching."""

from __future__ import annotations

import time
from collections.abc import Callable

from .queue import RequestQueue
from .types import LLMRequest, QueuedRequest


def _capability_key(req: LLMRequest) -> tuple[bool, bool]:
    """What a request demands of a provider.

    A batch goes to exactly one provider, so every request in a batch must
    need the same capabilities. Grouping by this key stops one request that
    needs logprobs from forcing an entire batch onto a pricier provider that
    the other fifteen requests didn't need.
    """
    return (req.needs_logprobs, req.needs_strict_json)


class Batcher:
    """Coalesce queued requests into batches on size, tokens, or time."""

    def __init__(
        self,
        max_batch_size: int = 16,
        max_wait_ms: float = 50.0,
        max_batch_tokens: int = 8000,
        *,
        in_flight: Callable[[], int] | None = None,
    ) -> None:
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms
        self.max_batch_tokens = max_batch_tokens
        # Lets the batcher ask the gateway "is anything already dispatched?".
        # Defaults to "nothing in flight" so the Batcher is usable standalone.
        self._in_flight = in_flight or (lambda: 0)

    async def collect(self, queue: RequestQueue) -> list[QueuedRequest]:
        """Return the next batch to dispatch."""
        # Wait indefinitely for the first request rather than polling on an
        # empty queue; the wait clock only starts once there's work to time.
        first = await queue.pop()
        if first is None:  # pragma: no cover - only on cancellation paths
            return []

        batch = [first]
        # An estimate, not an exact count, based on prompt size plus
        # max_tokens -- counting the prompt alone would overflow as soon as
        # a batch of short prompts asks for long completions.
        total_tokens = first.request.estimated_total_tokens()
        key = _capability_key(first.request)
        # Requests popped off the queue that don't fit this batch. They go
        # back onto the queue before collect() returns, keeping their
        # original seq so ordering is preserved.
        deferred: list[QueuedRequest] = []

        # Anchored to the oldest request's enqueue time, not to "now" or to
        # each new arrival. Re-anchoring on every arrival would mean a
        # steady trickle of requests keeps resetting the timer and the
        # oldest request never gets flushed. Note enqueued_at is in seconds
        # and max_wait_ms is in milliseconds -- mixing the two silently
        # makes the wait 1000x too long.
        deadline = first.enqueued_at + (self.max_wait_ms / 1000.0)

        while True:
            # Three independent limits, any one of which can end the batch:
            # size (the provider's per-call limit), tokens (the context
            # window and TPM budget), and time (latency). A longer
            # max_wait_ms fills bigger, more efficient batches at the cost
            # of worse latency per request -- there's no one right value,
            # it depends on whether the caller wants fast replies or high
            # throughput.
            if len(batch) >= self.max_batch_size:
                break
            if total_tokens >= self.max_batch_tokens:
                break

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break

            # If the queue is empty and nothing is already in flight, no
            # more requests can show up on their own, so waiting out the
            # rest of max_wait_ms would just add latency. The in-flight
            # check matters too: during a burst of submissions the queue
            # drains and refills constantly, so "queue empty" alone would
            # wrongly flush a batch that was about to grow.
            if queue.empty() and self._in_flight() == 0:
                break

            nxt = await queue.pop(timeout=remaining)
            if nxt is None:
                # Timed out with nothing new: the deadline condition fired.
                break

            if _capability_key(nxt.request) != key:
                # Set aside and requeue after the loop rather than dropping
                # it or dispatching it here.
                deferred.append(nxt)
                continue

            batch.append(nxt)
            total_tokens += nxt.request.estimated_total_tokens()

        # Requeued after the loop rather than inside it, so the collector
        # doesn't immediately pop the same incompatible request right back
        # and spin on it until the deadline expires.
        for entry in deferred:
            queue.put(entry)

        return batch
