# Benchmark: what does llm-gateway buy you over a naive async loop?

`bench.py` answers one question as honestly as a simulated harness can:
against a server with a real, enforced rate limit and occasional transient
failures, what's the actual difference between

- **naive**: `asyncio.gather` over every prompt through a bounded
  semaphore, with a *properly implemented* retry — exponential backoff,
  full jitter, `Retry-After` respected — on 429/5xx. This is not a
  strawman: the only thing it's missing on purpose is rate limiting,
  because that's the variable under test. It reuses `llm_gateway.RetryPolicy`
  for the backoff/jitter/classification logic itself (not for the rate
  limiting, batching, or failover the gateway adds), so both arms get the
  same quality of retry decision.
- **gateway**: `llm_gateway.LLMGateway`, configured with a token bucket set
  to the server's real limits, request batching, and the same retry
  policy.

Both arms are pointed at their own instance of the same simulated server
(`SimClient`, a small subclass of `llm_gateway.providers.mock.MockClient` —
see "Threats to validity" below for exactly what "same" means here).

## Run it

```
python3 benchmarks/bench.py --n-prompts 200 --runs 5 --seed 42 --markdown
```

That one command reproduces the run below end to end: it builds the
workload, runs both arms `--runs` times, and prints a stdout table plus
(with `--markdown`) a table ready to paste elsewhere. Add `--json out.json`
to also dump every raw per-run number, so a skeptical reader can recompute
the medians and percentiles themselves instead of taking the summary on
faith.

Every tunable — prompt count, server RPM, injected failure rate, latency
distribution, retry budget, batch shape, pricing — is a CLI flag; run
`python3 benchmarks/bench.py --help` for the full list. Increase `--runs`
for tighter percentile estimates; the default is 10.

## Hypothesis (stated before you scroll to the numbers)

- **naive** should rack up a real count of 429s as soon as its unthrottled
  burst exceeds the server's configured RPM. Because the server enforces a
  genuine 60-second sliding window but naive's retry budget is finite, some
  requests can exhaust their retries *before* the window frees capacity —
  meaning naive is not guaranteed 100% eventual success the way a
  patient-enough retry loop would be.
- **gateway** should show 429s approaching zero, because the token bucket
  paces requests under the limit *before* they're sent rather than
  discovering the limit by exceeding it. It should also send meaningfully
  fewer total requests to the server. The cost should be added latency —
  a caller occasionally waits in the gateway's queue for capacity that
  naive would have burned as a rejected-then-retried call instead.

`bench.py` prints this same paragraph before it prints results, so nobody
can accuse the harness of being tuned to match an after-the-fact story.

## Measured result

Run: `python3 benchmarks/bench.py --n-prompts 200 --runs 5 --seed 42 --markdown`

```
timestamp (UTC):  2026-09-18T06:03:12.822547+00:00
git commit:       9dd26c9
python:           3.13.2 (CPython)
platform:         macOS-26.6.1-arm64-arm-64bit-Mach-O
processor:        arm
cpu_count:        8
parameters:
  base_delay = 0.5, batch_size = 16, batch_tokens = 8000, batch_wait_ms = 25.0
  concurrency = 40, cost_per_1k_input = 0.02, cost_per_1k_output = 0.04
  gateway_tpm = 10000000.0, max_attempts = 6, max_delay = 8.0, max_tokens = 256
  n_prompts = 200, runs = 5, seed = 42, server_fail_rate = 0.03
  server_latency_mean_ms = 30.0, server_latency_sigma = 0.5, server_rpm = 190.0
```

_median [p5, p95] across 5 runs, 200 prompts/run_

**Configuration A — 200 prompts against a 190/min server (over the limit).**
This is the regime the library is built for.

| metric | naive | gateway |
|---|---|---|
| wall-clock (s) | 14.313 [13.609, 14.641] | **6.126** [5.816, 9.899] |
| requests sent | 275.0 [271.0, 286.0] | **206.0** [205.0, 218.0] |
| 429s received | 85.0 [81.0, 96.0] | **16.0** [15.0, 28.0] |
| 5xx received | 5.0 [4.0, 7.0] | 5.0 [4.0, 7.0] |
| successes | 185.0 [183.0, 186.0] | 185.0 [183.0, 186.0] |
| failures | 15.0 [14.0, 17.0] | 15.0 [14.0, 17.0] |
| success rate | 92.5% [91.5%, 93.0%] | 92.5% [91.5%, 93.0%] |
| latency p50 (s) | **0.082** [0.078, 0.092] | 0.106 [0.090, 0.116] |
| latency p95 (s) | 0.146 [0.140, 0.154] | **0.141** [0.130, 0.169] |
| latency p99 (s) | **0.157** [0.151, 0.160] | 0.518 [0.419, 4.482] |
| est. cost (USD) | 0.3563 [0.3531, 0.3574] | 0.3579 [0.3544, 0.3603] |

**Configuration B — 150 prompts against the same server (inside the limit).**
Run because a benchmark that only reports its favourable regime is marketing.

| metric | naive | gateway |
|---|---|---|
| wall-clock (s) | 0.567 | 0.508 |
| requests sent | 155.0 | 155.0 |
| 429s received | 0.0 | 0.0 |
| success rate | 100.0% | 100.0% |
| latency p50 (s) | 0.069 | 0.086 |

Reproduce with `python3 benchmarks/bench.py --n-prompts 150 --runs 5 --seed 42`.

### Reading this honestly

- **The gateway's real win is 429s and wasted work**: 85 rate-limit rejections
  drop to 16, and 275 requests sent drop to 206 for the same 200 prompts.
  Those 69 extra requests are work the naive loop performed and threw away.
  Wall-clock follows from that: 6.1s vs 14.3s, because time spent backing off
  from self-inflicted 429s dwarfs the time spent pacing to avoid them.

- **Success rate is identical, not better.** Both arms complete 185/200. The
  15 failures are structural, not flakiness: 200 prompts against a 190/minute
  allowance means ~10 requests cannot be served inside the window, and a
  6-attempt retry budget capped at 8s cannot outlast a 60s window. Neither
  client can conjure capacity that does not exist. An earlier version of this
  file claimed 100% for the gateway; that number came from a misconfiguration
  described below, and was wrong.

- **The gateway is worse at the tail.** p50 is slightly worse (0.106s vs
  0.082s) because batching deliberately waits up to `max_wait_ms`, and p99 is
  markedly worse (0.518s vs 0.157s) because the requests that lose the
  rate-limit lottery wait for bucket refill instead of being rejected quickly.
  If your workload is interactive, that tradeoff is the wrong one and you
  should set `max_wait_ms` near zero.

- **Cost is a wash** (0.3563 vs 0.3579). Both arms ultimately bill for the same
  successful completions. Anyone claiming a client-side gateway saves money on
  identical work should be asked to show the arithmetic.

- **Configuration B is the honest counterweight: below the provider's limit,
  this library buys you nothing.** Identical requests, identical success,
  identical 429 count (zero), and slightly worse p50 from batching wait. If
  you are not near a rate limit, do not add this dependency.

### Two corrections made to this benchmark

Both were found by reviewing results that looked too good, and both changed the
conclusions materially. They are recorded here rather than quietly fixed.

1. **A batch endpoint that does not exist.** The first run configured the
   simulated server with `supports_batching=True`, so a 16-prompt batch counted
   as ONE request, producing a headline "14 requests sent vs 275". Neither Groq
   nor OpenRouter — the two providers this library ships adapters for — has a
   synchronous multi-prompt endpoint, so that saving is uncollectable in
   production. The benchmark now defaults to no batch endpoint; `--batch-endpoint`
   models the other case for providers that genuinely offer one.

2. **Blast radius, a real defect this exposed.** With the fake batch endpoint
   removed, the gateway's success rate fell to **76%, well below the naive
   loop's 92.5%**. The cause was that a batch was retried and failed as a unit,
   so one transient 503 doomed its 15 healthy siblings. That is correct for a
   true batch endpoint and wrong for N independent HTTP calls that merely share
   rate-limit accounting. Failures are now isolated per request, which restored
   parity. The benchmark earned its keep by catching this.

## Threats to validity

Read this before trusting any number above.

- **This is a simulated server, not a real provider.** `SimClient` enforces
  a sliding-window RPM limit and injects latency/failures the same *shape*
  a real API uses, but it has none of a real provider's queueing behavior,
  regional routing, per-key vs per-org limits, or the myriad undocumented
  quirks (soft throttling before the hard limit, warm-up penalties, etc.)
  that real gateways exhibit. A result here is a statement about this
  simulation, not a guarantee about any specific real provider.
- **Latency is synthetic.** It's drawn from a seeded log-normal
  distribution chosen to *look* plausible (mean 30ms, σ=0.5), not measured
  from any real API. Real latency distributions are typically fatter-tailed
  and can correlate with load in ways a fixed, load-independent distribution
  does not capture.
- **The naive baseline is one particular implementation.** It uses a
  bounded semaphore plus `RetryPolicy`'s backoff. A different naive
  implementation — a lower concurrency cap, a smarter (but still
  rate-limit-unaware) circuit breaker, client-side request coalescing,
  adaptive concurrency — could look meaningfully better or worse than what
  is measured here. This is not "the" naive baseline, just an honest one.
- **Results depend entirely on the configured limits.** The specific gap
  shown above (85 vs 0 429s, a long gateway p95 tail) is a direct
  consequence of `--server-rpm 190` against `--n-prompts 200` — a workload
  chosen to sit close to the server's burst allowance so both the
  "gateway avoids 429s" and "gateway pays a queueing-latency tax when
  genuinely over capacity" behaviors show up in one run. Move `--server-rpm`
  well above `--n-prompts` and naive stops erroring at all (nothing to
  measure); move it well below and *both* arms take proportionally longer
  because the workload's true minimum completion time grows to
  `(N - burst_capacity) / (rpm / 60)` seconds regardless of client
  implementation — no library can make a server accept more traffic than
  its own configured limit permits. Different limits, meaningfully
  different story.
- **The naive and gateway arms do not literally share one server
  instance.** Each arm gets its own `SimClient`, seeded identically per
  run (same latency/failure RNG seed, same retry-jitter RNG seed) so the
  *draw sequence available* to each arm is equivalent. But because the two
  arms issue calls in different order and at different concurrency (batched
  vs unbatched, paced vs unpaced), they do not consume that identical
  sequence in the same order — so a given prompt is not guaranteed to see
  the exact same simulated latency/failure in both arms, only a
  statistically equivalent source of them. This is the standard tradeoff
  for comparing two systems with different concurrency structure; a
  perfectly shared server would require serializing both arms through one
  stateful object, which would itself distort the comparison by forcing
  artificial serialization between them.
- **Failover is configured but not exercised.** The gateway is built with
  `LLMGateway`'s normal multi-provider machinery, but this benchmark gives
  it exactly one provider (matching naive's single endpoint, for a fair
  1:1 comparison). The circuit breaker and failover-to-a-second-provider
  path exist in the library and are exercised by the test suite, but this
  benchmark says nothing about their effect — that would need a
  multi-provider run, which is a different (and separately useful)
  benchmark.
- **A mock cannot reproduce real-world queueing or regional effects.**
  No connection pooling limits, no TCP/TLS handshake cost, no DNS, no
  geographic latency variance, no provider-side autoscaling lag, no
  multi-tenant noisy-neighbor effects. Everything here happens in one
  process against in-memory `asyncio.sleep()` calls.
- **"Reproducible" means the deterministic counts, not wall-clock.** With
  a fixed `--seed`, the *count*-based metrics (requests sent, 429s, 5xxs,
  successes, failures, cost) are bit-for-bit identical across repeated
  invocations — `tests/test_benchmark.py` asserts this directly. Wall-clock
  time and per-call latency are not: they depend on real OS scheduling,
  asyncio event-loop timing, and machine load, and will vary run to run
  and machine to machine even with the same seed. Report wall-clock as a
  distribution (which `--runs` does), never as a single number, for exactly
  this reason.

## Metric definitions

- **wall-clock**: time from dispatching the first prompt to
  `asyncio.gather` returning for that arm, whether or not every prompt
  succeeded.
- **requests sent**: count of actual calls made to the simulated server
  (one per `complete()`/`complete_batch()` invocation, including retries
  and failed attempts), *not* count of prompts. A batch call counts once
  regardless of how many prompts it carries — this is why gateway's number
  is so much smaller than naive's: both fewer retries *and* batching
  reduce it.
- **429s / 5xx received**: count of *call-level* rejection events (one per
  rejected call, so one rejected batch call counts once even though it
  fails every prompt in that batch).
- **latency p50/p95/p99**: end-to-end, per prompt, from submission to
  final success — includes queueing, retries, and (for gateway) time spent
  waiting in the batcher and the rate limiter. Computed only over prompts
  that succeeded.
- **success rate**: successes / total prompts submitted, per run.
- **est. cost**: `input_tokens/1000 * cost_per_1k_input + output_tokens/1000
  * cost_per_1k_output`, summed over successful prompts, using the mock's
  token accounting (`~len(prompt)//4` input tokens, `min(max_tokens, 32)`
  output tokens) and the `--cost-per-1k-*` prices (defaults: $0.02 in /
  $0.04 out per 1k tokens — placeholder numbers, not any real provider's
  pricing).
