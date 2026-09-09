# Learning path

This repository is a teaching artifact. The code works, but the reason it
exists is so you can read it, delete the core of it, and write it again from
memory against a green test suite.

## Reading order

Four modules matter. For each one, in this order:

1. **`X.EXPLAIN.md`** — the problem in plain language, the algorithm in prose
   with a worked numeric example, the alternatives and how each one fails, and
   the questions you will be asked about it.
2. **`X.py`** — the annotated source. Every non-obvious decision carries a tag:
   `WHY:` why this exists, `ALT:` what else was possible and why it is worse,
   `TRAP:` the specific bug this guards against, `ASK:` an interview question
   this code answers.
3. **`tests/test_X.py`** — last, deliberately. The tests are the specification
   you will rebuild against, and reading them first turns recall into
   recognition.

The four, in dependency order:

| # | Module | Explainer |
|---|--------|-----------|
| 1 | `src/llm_gateway/rate_limit.py` | `rate_limit.EXPLAIN.md` |
| 2 | `src/llm_gateway/retry.py` | `retry.EXPLAIN.md` |
| 3 | `src/llm_gateway/batching.py` | `batching.EXPLAIN.md` |
| 4 | `src/llm_gateway/routing.py` | `routing.EXPLAIN.md` |

Everything else — `gateway.py`, `queue.py`, `breaker.py`, `metrics.py`,
`providers/` — is normally commented plumbing. Read it once to see how the
four fit together, then leave it alone.

## The rebuild loop

```bash
python scripts/strip.py          # blank the four functions
pytest                           # confirm exactly those areas go red
# implement from scratch against the failing tests
pytest                           # green means you have it
diff src/llm_gateway/rate_limit.py .reference/rate_limit.py
```

The diff is the point. Where your version differs from the reference, one of
you is wrong — and working out which is where the learning actually happens.
Sometimes it will be the reference.

`python scripts/strip.py --check` reports the current state.
`python scripts/strip.py --restore` puts the originals back.

Do one module per sitting. Stripping all four at once means the integration
tests fail for four reasons simultaneously and tell you nothing.

## Self-check questions

Collected from every `ASK:` tag. Answer all of them without looking and you can
defend this project in an interview.

### Rate limiting (`rate_limit.py`)

- Why `monotonic()` instead of `time.time()` in a rate limiter?
- How do you refill a token bucket without a background timer?
- What breaks if you drop the `min(capacity, ...)` clamp?
- Does single-threaded asyncio need locks? When?
- Why does the sleep sit inside a re-check loop?
- Why must the lock not be held across `await asyncio.sleep()`?
- Why does `try_acquire` not need the lock but `acquire` does?
- What happens if someone asks for more tokens than capacity?
- Why two buckets per provider instead of one?
- Why is fixed-window counting wrong at the boundary?
- Why not a sliding window log, given that it is exactly correct?

### Retries (`retry.py`)

- What makes a status code retryable?
- Why is 400 not retryable but 429 is?
- Why exponential rather than linear backoff?
- Why cap the delay, and why cap *before* jittering?
- Full, equal, or decorrelated jitter — which and why?
- Why does backoff without jitter cause the failure it is meant to prevent?
- If the provider sends `Retry-After: 30` and your backoff says 2s, what do you
  wait — and what if the header says 60s and you computed 64s?
- Why is the attempt budget checked separately from the classification?

### Batching (`batching.py`)

- Should the batch timer start from the oldest or the newest request, and what
  breaks with the other choice?
- What happens to a single request arriving into an empty queue, and how do you
  fix it without breaking throughput?
- What does increasing `max_wait_ms` buy you, and what does it cost?
- Why are there three flush conditions rather than one?
- What happens to your provider mix if one request in sixteen needs a
  capability the cheap provider lacks?
- Why does "the queue is empty" alone not mean "stop waiting"?
- Why does the first pop have no timeout when every later one does?
- What do you count toward the token budget, and how precisely?
- Why can't you requeue a mismatched request as soon as you see it?

### Routing (`routing.py`)

- Why is capability a filter rather than a term in the score?
- Why rank on rate-limit headroom before cost?
- Why does a down provider look *attractive* to a headroom-based ranker, and
  how do you stop that?
- What do you compare against `max_context_tokens` — prompt size, or prompt
  plus `max_tokens`?
- What is the worst thing a router can do when nothing matches?
- Why quantize the headroom score instead of comparing raw floats?
- How does the gateway fail over without re-selecting the provider that just
  failed?

### Circuit breaking (`breaker.py`)

- What does a circuit breaker buy you that retries alone do not?
- Why consecutive failures rather than a failure rate?
- Why exactly one probe in `HALF_OPEN`?
- Why must failures arriving after the trip not restart the recovery timer?

## What you can claim, and when

Only after the strip-and-rebuild loop has been completed at least once per core
module. Every noun in the resume bullet should map to something you can write
from an empty file. If one doesn't yet, cut it from the bullet rather than from
the code.
