# Rate limiting — read this before `rate_limit.py`

## The problem

A provider will accept 600 requests per minute and 150,000 tokens per minute.
Go over either and you get `429`. Your program does not know how much of that
budget it has spent, because "spent" is a function of time: a request you sent
59 seconds ago almost doesn't count any more, and one you sent 61 seconds ago
doesn't count at all.

So you need a thing that answers one question — *may I send this right now, and
if not, how long until I may?* — cheaply enough to ask before every single call,
and correctly enough that fifty coroutines asking simultaneously don't all get
told yes.

## The algorithm

A bucket holds up to `C` tokens and gains `r` tokens per second. Sending costs
tokens. If the bucket has enough, take them and go. If not, sleep until it
does.

The trick is that nothing refills the bucket on a timer. Instead, every time
anyone touches it, you compute how many tokens *would have* arrived since the
last touch and add those, clamped at `C`:

```
tokens = min(C, tokens + (now - last_touch) * r)
```

Worked example — capacity 10, refill 2/sec, bucket starts full at t=0.

Five requests arrive at t=0. Bucket has 10, each costs 1. All five go
immediately; bucket is at 5. Five more arrive at t=0 (same instant). Bucket
goes to 0. All ten went out in a burst — that is the point of `C`.

An eleventh arrives at t=0. Refill adds `0 * 2 = 0`. Deficit is 1 token, so it
sleeps `1 / 2 = 0.5s`. At t=0.5 it wakes, refill adds `0.5 * 2 = 1`, takes it,
bucket back to 0.

Now nothing happens until t=60. Refill would add `59.5 * 2 = 119` tokens. The
clamp cuts that to 10. This is the whole reason the clamp exists: without it
the bucket would hold 119 tokens and the next burst would send 119 requests
into a limit of 10.

Long-run behaviour: over any window of length `T`, at most `r*T + C` requests
get out. Bursty when idle, exactly `r` when saturated.

## Alternatives, and how each one fails

**Fixed window counter.** Count requests, reset the counter every 60s. Fails at
the boundary: 100 requests at 11:59:59 and 100 at 12:00:01 is 200 requests in
two seconds, and both windows report "within limit". You get 429s from a
limiter that believes it is working.

**Sliding window log.** Keep a timestamp per request, evict anything older than
the window, count what's left. Exactly correct. Costs O(n) memory in the number
of requests per window and O(n) work to evict — at 150k tokens/min that is a
list you do not want.

**Leaky bucket.** Requests drain at a constant rate, full stop. Correct, but
strictly stricter than the provider: it forbids bursts the provider would
happily have accepted, so you leave throughput on the table for no benefit.

**Token bucket.** O(1) memory, O(1) per acquisition, allows bursts to `C`,
bounds the average to `r`. And — the practical argument — it is what providers
themselves implement, so modelling their limiter with the same shape means your
estimate of "am I allowed to send" tracks theirs instead of drifting from it.

## Three questions an interviewer will ask

**Why `time.monotonic()` and not `time.time()`?**
Wall clock can move backwards — NTP correction, DST, an operator setting the
date. `elapsed` then goes negative, the bucket refills by a negative amount, and
the limiter locks up silently until wall time catches back up. Monotonic only
moves forward, and it is unaffected by any of that.

**Why does the sleep sit inside a loop instead of just sleeping once?**
Because between computing your wait and waking up, another coroutine can consume
exactly the tokens you were waiting for. Sleep-then-deduct over-issues whenever
there is more than one waiter: they all wake "entitled" to the same tokens and
all take them. Re-check after every sleep.

**Why must the lock not be held across the `await`?**
Holding it serializes every waiter behind the first: waiter #2 cannot even check
the bucket until waiter #1 has slept its full duration. Throughput collapses
from "one acquisition per refill interval" to "one per sleep", which under
contention is arbitrarily worse. The lock protects the read-modify-write of the
token count; it must be released across the wait and reacquired after.
