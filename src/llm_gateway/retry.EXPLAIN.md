# Retries and backoff: design notes

## The problem

Some failures are worth repeating and some are not, and the cost of confusing
them is asymmetric. Retrying a `400` wastes traffic on a guaranteed failure.
*Not* retrying a `503` throws away a request that would have succeeded a second
later. And retrying a `429` badly is worse than not retrying at all, because
the thing you are retrying into is an overload you are contributing to.

Two decisions, then: **should I retry**, and **when**.

## Should I retry

The rule is one question: *would sending these identical bytes again plausibly
produce a different answer?*

- `400`, `401`, `403`, `404`, `422`: no. The request is malformed, the key is
  wrong, the route doesn't exist, the schema doesn't validate. None of that is
  time-dependent. Retrying multiplies your traffic against a certain failure.
- `429`, `500`, `502`, `503`, `504`, timeouts, connection resets: yes. The
  request was fine; the far end was momentarily unable to serve it.

Unknown codes default to their class: unknown `5xx` retryable, unknown `4xx`
not. Providers invent status codes, and defaulting is better than refusing to
decide.

The attempt budget is checked separately from the classification. They are
different questions, and collapsing them makes your logs lie. You end up
reporting "gave up: non-retryable error" when what actually happened is that
you ran out of attempts.

## When: the backoff

Delay grows exponentially: `base_delay * 2**attempt`, capped at `max_delay`.

Worked example: `base_delay=0.5`, `max_delay=60`, jitter off:

```
attempt 0 -> 0.5s
attempt 1 -> 1.0s
attempt 2 -> 2.0s
attempt 3 -> 4.0s
attempt 4 -> 8.0s
...
attempt 12 -> 2048s, capped to 60s
```

Cap before jittering, not after. `2**attempt` reaches hours by attempt 12, and
capping first both bounds the worst case and keeps the jitter sampling from a
sane range.

Then jitter. Full jitter replaces the delay with `uniform(0, delay)`, so
attempt 3 above becomes a uniform draw from `[0, 4.0]`.

## Why jitter is not optional

Five hundred clients hit a provider that briefly falls over. All five hundred
receive `429` within the same few milliseconds. Without jitter, all five hundred
compute `0.5s` and retry at the same instant. That spike is the same shape as
the one that caused the outage, so it fails again, and now all five hundred
compute `1.0s`. The retry schedule has synchronized the fleet into a metronome
hammering a service that is trying to recover. This is the thundering herd, and
backoff without jitter causes it rather than preventing it.

**Full jitter** (`uniform(0, delay)`) is what this module uses. It is
stateless, pure (so `delay_for` is trivially testable), and it minimizes the
probability that two clients pick the same moment. The cost is high variance:
an unlucky client may retry almost immediately. That is acceptable here because
the cap and the attempt budget bound the total damage.

**Equal jitter** (`delay/2 + uniform(0, delay/2)`) guarantees a minimum wait
and has lower variance, but leaves half the delay synchronized across the fleet.

**Decorrelated jitter** (`min(max_delay, uniform(base, prev * 3))`) spreads
best of the three, but needs the previous delay as state, which makes the
function impure and much harder to reason about and test.

## `Retry-After` dominates

When the response carries `Retry-After`, use `max(retry_after, computed)`.
Never `min`.

The header is the provider telling you exactly when your quota resets. Your
backoff is a guess. Taking the smaller value means returning while still
throttled: you earn another `429`, and on several providers a retry inside the
penalty window extends it. Taking the larger keeps whichever party is more
conservative, which is the correct behaviour in both directions: if the header
says 30s and you computed 2s, you wait 30; if you have escalated to 64s after
six failures and the header says 5s, you keep your 64.
