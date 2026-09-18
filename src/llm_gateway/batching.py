"""Request batching.

Read `batching.EXPLAIN.md` before this file.
"""

from __future__ import annotations

import time
from collections.abc import Callable

from .queue import RequestQueue
from .types import LLMRequest, QueuedRequest


def _capability_key(req: LLMRequest) -> tuple[bool, bool]:
    """What a request demands of a provider.

    WHY: a batch is dispatched to exactly ONE provider, so it can only go
         somewhere that satisfies the union of what its members need. Mixing a
         logprobs request into a batch of fifteen that do not need logprobs
         forces all sixteen onto the strict provider -- which, measured on the
         500-prompt example, sent 99% of a workload to a provider costing 10x
         the alternative because 10% of it needed one extra field.
    ALT: batch indiscriminately and let the router take the union. Simpler, and
         it silently destroys the cost benefit of having several providers.
    ASK: What happens to your provider mix if one request in sixteen needs a
         capability the cheap provider lacks?
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
        # WHY: block indefinitely for the FIRST request. There is no batch to
        #      time out yet, and spinning on an empty queue burns CPU to
        #      discover nothing. The clock starts when work exists.
        # ALT: `pop(timeout=max_wait_ms)` in a loop, returning empty batches on
        #      an idle queue. The dispatcher then has to filter out empty
        #      batches forever, and an idle gateway wakes up twenty times a
        #      second to do nothing.
        # ASK: Why does the first pop have no timeout when every later one does?
        first = await queue.pop()
        if first is None:  # pragma: no cover - only on cancellation paths
            return []

        batch = [first]
        # WHY: estimated, not exact. This is recomputed on every enqueue, so a
        #      real tokenizer would put a model forward pass on the hot path to
        #      save a few percent of batch utilisation.
        # TRAP: count prompt AND max_tokens. Budgeting on the prompt alone
        #       overflows the moment a batch of short prompts asks for long
        #       completions -- which is the normal shape of a generation
        #       workload.
        # ASK: What do you count toward the token budget, and how precisely?
        total_tokens = first.request.estimated_total_tokens()
        key = _capability_key(first.request)
        # Requests pulled off the queue that do not belong in THIS batch. They
        # go back before collect() returns, keeping their original seq so the
        # queue's priority and FIFO ordering survive the round trip.
        deferred: list[QueuedRequest] = []

        # TRAP: the deadline is anchored to the OLDEST request in the batch --
        #       `first.enqueued_at`, not "now", and never re-anchored as new
        #       requests arrive. Timing from the newest arrival means a steady
        #       trickle resets the timer on every arrival, and the oldest
        #       request waits forever while the batch never fills. That is the
        #       single most common batching bug there is, and it is invisible
        #       under load testing with bursty traffic -- it only shows up as
        #       a p99 cliff when real traffic arrives as a trickle.
        # ALT: anchoring to enqueue time rather than to "when collect() started"
        #      also charges the batch for time the request spent waiting in the
        #      queue behind other batches, which is what the caller actually
        #      experiences as latency.
        # ASK: Should the batch timer start from the oldest or the newest
        #      request, and what breaks with the other choice?
        # TRAP: enqueued_at is monotonic seconds and max_wait_ms is
        #       milliseconds. Mixing the units here yields a deadline 1000x too
        #       far out, and the symptom is not a crash -- it is a batcher that
        #       looks like it works and quietly holds every request for a
        #       minute.
        deadline = first.enqueued_at + (self.max_wait_ms / 1000.0)

        while True:
            # WHY: these three flush conditions are ORed, and each bounds a
            #      different resource -- batch size bounds the provider's
            #      per-call limit, the token budget bounds the payload against
            #      the context window and the TPM bucket, and the deadline
            #      bounds latency. Any one of them alone leaves a hole: size
            #      alone lets 16 enormous prompts overflow the context; tokens
            #      alone lets 4000 one-word prompts through in a single call;
            #      time alone gives you unbounded batches under a burst.
            #
            #      The whole component is one tradeoff: batching trades latency
            #      for throughput. A longer max_wait_ms fills bigger batches --
            #      fewer round trips, better rate-limit utilisation, higher
            #      throughput -- and every single request pays the wait, so p99
            #      gets worse. There is no universally correct setting. An
            #      interactive chat path wants max_wait_ms near zero; an
            #      offline eval sweep over 50k prompts wants it as large as the
            #      provider's batch limit allows.
            # ASK: What does increasing max_wait_ms buy you, and what does it cost?
            if len(batch) >= self.max_batch_size:
                break
            if total_tokens >= self.max_batch_tokens:
                break

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break

            # WHY: adaptive dispatch. If the queue is empty AND nothing is in
            #      flight, no further request can arrive as a result of work we
            #      are already doing, so waiting out max_wait_ms is pure added
            #      latency for a batch that will never grow. Dispatch now.
            # TRAP: the in-flight check is the half everyone forgets. "Queue is
            #       empty" alone is wrong during a submit_many sweep -- the
            #       queue empties constantly while 400 responses are still
            #       outstanding, and the next arrivals are microseconds away.
            #       Flushing on empty-alone would degrade a batched sweep into
            #       one-request-per-call, which is the exact behaviour the
            #       batcher was written to eliminate.
            # ASK: What happens to a single request arriving into an empty
            #      queue, and how do you fix it without breaking throughput?
            if queue.empty() and self._in_flight() == 0:
                break

            nxt = await queue.pop(timeout=remaining)
            if nxt is None:
                # Timed out with nothing new: the deadline condition fired.
                break

            if _capability_key(nxt.request) != key:
                # TRAP: do not drop it and do not dispatch it here. Set it
                #       aside and requeue below -- a request quietly discarded
                #       at this point leaves its caller awaiting a future that
                #       nobody will ever resolve.
                deferred.append(nxt)
                continue

            batch.append(nxt)
            total_tokens += nxt.request.estimated_total_tokens()

        # WHY: requeued after the loop, not inside it. Putting one back mid-loop
        #      makes the next pop hand it straight back -- the collector spins
        #      on the same incompatible request until the deadline expires,
        #      burning the whole window on one item it was never going to take.
        # ASK: Why can't you requeue a mismatched request as soon as you see it?
        for entry in deferred:
            queue.put(entry)

        return batch
