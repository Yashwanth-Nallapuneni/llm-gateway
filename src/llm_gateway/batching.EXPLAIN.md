# Batching: design notes

## The problem

You have 500 independent prompts. Sent one at a time, each pays the full cost of
a round trip: TLS, HTTP, the provider's own queueing and, critically, one
whole request out of your RPM budget. Five hundred small prompts against a 600
RPM limit takes most of a minute, and almost none of that minute is spent on
inference.

The provider will take sixteen of them in one call. That is sixteen prompts for
one round trip and one RPM token. The batcher's job is to notice that requests
are arriving close together and group them, without knowing anything about the
workload in advance.

## The tradeoff, which is the whole component

**Batching trades latency for throughput.** There is no version of this where
you get both.

Wait longer before dispatching and batches fill up: fewer round trips, better
rate-limit utilisation, higher throughput. And every single request in the batch
waited that much longer than it needed to, so p99 latency gets worse by
approximately the wait.

There is no universally correct setting. An interactive chat path wants
`max_wait_ms` near zero, since a user is watching. An offline eval sweep over 50,000
prompts wants it as large as the provider's batch limit allows, because nobody
is watching and the only thing that matters is when the whole sweep finishes.
The right value is a property of the workload, not of the library, which is why
it is a constructor argument and not a constant.

## The algorithm

Block until the first request arrives; there is nothing to time out yet.
Anchor a deadline to that request. Then keep pulling requests until any one of
three conditions holds:

1. the batch has `max_batch_size` requests,
2. the cumulative token estimate reaches `max_batch_tokens`,
3. the oldest request in the batch has waited `max_wait_ms`.

Each bounds a different resource, and any one alone leaves a hole: size alone
lets sixteen enormous prompts overflow the context window, tokens alone lets
four thousand one-word prompts into a single call, and time alone gives you
unbounded batches under a burst.

Worked example: `max_batch_size=4`, `max_wait_ms=50`.

```
t=0ms    request A arrives. batch=[A], deadline = 50ms.
t=10ms   B arrives.         batch=[A,B]
t=15ms   C arrives.         batch=[A,B,C]
t=50ms   deadline hit.      dispatch [A,B,C]   <- condition 3
```

Busier, same settings: A, B, C and D all arrive at t=0, the batch reaches four
immediately, and condition 1 dispatches without any wait at all.

## The oldest-request trap

Condition 3 times from the **oldest** request in the batch, and the deadline is
never re-anchored as new requests arrive.

Time from the newest arrival instead and a steady trickle resets the timer on
every single arrival. Request A arrives at t=0; B at t=40 resets the deadline to
t=90; C at t=80 resets it to t=130. A never ships. It starves for as long as the
trickle continues, which in a real service is forever.

What makes this bug so common is that it is invisible in testing. Load tests
send bursts, and under a burst the size condition fires first, so the timer
never matters. It only appears in production, as an unexplained p99 cliff, when
real traffic arrives as a trickle rather than a burst.

## Adaptive dispatch

One request arrives into an empty queue. Waiting out `max_wait_ms` for a batch
that will never fill is pure added latency. So: if the queue is empty *and*
nothing is in flight, dispatch now.

The in-flight half is the part people forget. "Queue is empty" alone is wrong
during a bulk sweep: the queue empties constantly while 400 responses are still
outstanding, and the next arrivals are microseconds away. Flushing on
empty-alone degrades a batched sweep into one request per call, which is exactly
the behaviour the batcher exists to eliminate. The two conditions together mean
"no work is pending and no work can arrive as a consequence of work we are
already doing", which is the only safe moment to stop waiting.
