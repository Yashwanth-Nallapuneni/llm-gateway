# llm-gateway

[![PyPI](https://img.shields.io/pypi/v/aiollm-gateway)](https://pypi.org/project/aiollm-gateway/)
[![Python](https://img.shields.io/pypi/pyversions/aiollm-gateway)](https://pypi.org/project/aiollm-gateway/)
[![CI](https://github.com/Yashwanth-Nallapuneni/llm-gateway/actions/workflows/ci.yml/badge.svg)](https://github.com/Yashwanth-Nallapuneni/llm-gateway/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/Yashwanth-Nallapuneni/llm-gateway/blob/main/LICENSE)

**Send 500 prompts to an LLM API without writing a single `sleep()`.**

An async Python layer that sits between your code and a provider's API. It
paces requests under rate limits, retries what is worth retrying, groups
prompts to cut round trips, and moves traffic off a provider that starts
failing.

Python 3.11+, asyncio, **zero runtime dependencies**. Not a proxy, not a
server, no caching or prompt management.

```bash
pip install aiollm-gateway
```

![Terminal recording of examples/failover_demo.py: the router sends traffic to the cheap primary provider, the primary starts failing and the circuit breaker moves traffic to backup, then the primary recovers and traffic moves back, ending with the batching/latency/cost metrics report](https://raw.githubusercontent.com/Yashwanth-Nallapuneni/llm-gateway/main/docs/failover-demo.svg)
<sub>`examples/failover_demo.py`, real captured output: primary fails, the breaker trips traffic to backup, then recovers.</sub>

## Quickstart

This runs as-is, with no API key and no network. `MockProvider` ships with the library
precisely so you can see the machinery work before wiring up a real provider.

```python
import asyncio
from llm_gateway import Batcher, LLMGateway, LLMRequest, MockProvider


async def main():
    gateway = LLMGateway(
        providers=[
            MockProvider("cheap", rpm_limit=600, cost_per_1k_output=0.02),
            MockProvider("backup", rpm_limit=1200, cost_per_1k_output=0.50),
        ],
        batcher=Batcher(max_batch_size=16, max_wait_ms=50),
    )
    async with gateway:
        responses = await gateway.submit_many(
            [LLMRequest(prompt=f"Summarise document {i}") for i in range(500)]
        )
    print(f"{len(responses)} responses")
    print(gateway.metrics.report())


asyncio.run(main())
```

500 requests go out as roughly 50 batched dispatches, paced under both the
request and token limits, with failures retried and routed around.

## Command line

No Python required. `llm-gateway run` reads a file of prompts and writes one
JSON result per line, in order:

```bash
echo '{"prompt": "Summarise document 1"}
{"prompt": "Summarise document 2"}' | llm-gateway run - --provider mock
```

Input is a `.jsonl` file (one `{"prompt": "..."}` object per line, with
optional `id`/`max_tokens`/`priority`/`needs_logprobs`/`needs_strict_json`/
`model`), a `.txt` file (one prompt per line), or `-` for stdin. `--provider
{mock,groq,openrouter}` selects the backend; for `groq`/`openrouter`, the API
key comes from `GROQ_API_KEY`/`OPENROUTER_API_KEY` (or `--api-key`) -- it is
never printed. `--dry-run` estimates prompt count, tokens and cost without
making any calls, and any paid provider requires a confirmed cost estimate
(or `--yes`) before it sends a single request. A failed prompt is written as
an `{"error": ...}` line rather than aborting the run; `gateway.metrics.report()`
prints to stderr unless `--no-metrics`. `--budget USD` caps total spend for the
run: once committed spend would cross it, further prompts fail fast with a
budget-exceeded error instead of being sent to a provider -- results already
obtained are still written out in full, and the run exits non-zero with a
count of how many prompts were skipped. Each prompt holds its worst-case
cost (priced at `--max-tokens` output tokens) until the real cost is known,
so with a tight budget, set `--max-tokens` close to what you need or
prompts that would have fit get refused. A progress line is written to
stderr as the run goes, unless `--quiet`; the metrics report after the run
is turned off separately with `--no-metrics`. See `llm-gateway run --help` for
the full option list.

`--provider` also accepts a comma-separated priority list, e.g. `--provider
groq,openrouter`, which runs several providers under one gateway using the
library's own capability-filtered routing and failover. With more than one
provider, `--model` takes `name=value` pairs instead of a single value,
e.g. `--model groq=allam-2-7b,openrouter=meta-llama/llama-3.1-8b-instruct`.

`--timeout SECONDS` sets a per-prompt timeout covering queueing and
retries, not just the network call; a prompt that exceeds it fails with a
timeout error instead of hanging the run (`LLMRequest(timeout_s=...)` from
Python). `--metrics-json PATH` writes a machine-readable snapshot of the
run's metrics (`MetricsSink.to_dict()`) alongside the usual human-readable
report.

`llm-gateway models --provider groq` (or `openrouter`) prints the model IDs
that provider currently offers, one per line and sorted, straight from its
`GET /models` endpoint (`--contains TEXT` filters by substring) -- useful
since Groq in particular retires models often and a hardcoded default can
go stale.

### Adaptive rate limiting

`--adaptive` (or `LLMGateway(..., adaptive=True)` from Python) is for
providers that send no rate-limit headers to sync against, such as
OpenRouter: on a 429 it halves the local request rate, then grows it back
slowly on success, instead of trusting a fixed configured limit for the
life of the run. It is opt-in and does nothing for a provider like Groq
that already syncs its buckets from real response headers. See
`examples/timeouts_and_adaptive.py`.

## Resuming a crashed sweep

`--store PATH` records every prompt's outcome in a SQLite file as the run
goes, keyed by a hash of the fields that determine the answer (model,
prompt, max_tokens, capability flags -- not priority or metadata). Rerun
the same command with `--store` pointing at that file and `--resume`, and
prompts already completed are served straight from the file instead of
calling the provider again; only what never finished gets sent out:

```bash
llm-gateway run 20k_prompts.jsonl --store sweep.db --output out.jsonl
# ...killed at prompt 14,000...
llm-gateway run 20k_prompts.jsonl --store sweep.db --resume --output out.jsonl
```

`--resume` is required whenever `--store` points at a file that already has
rows in it, so a stale or mistyped path fails loudly instead of silently
merging into an unrelated run. The run summary reports how many prompts
were served from the store versus freshly called; the metrics table
above that line counts only the fresh calls. Replay returns each
prompt's first answer, not a new sample -- correct for reproducing an eval,
not for resampling at temperature > 0.

Use `--store` for any run you might stop with Ctrl-C: `--output` is only
written when a run finishes, but the store keeps every completed prompt as
it goes. The hash does not include the provider name, so resuming with a
different `--provider` but the same model reuses the stored answers.

## Troubleshooting

- **401 from a provider:** the API key is missing or wrong. The CLI reads
  `GROQ_API_KEY` or `OPENROUTER_API_KEY`; auth errors are not retried.
- **Model not found (Groq 404, OpenRouter 400):** providers retire models
  often. `llm-gateway models --provider groq` lists the ones that exist
  right now.
- **Empty `text` with `finish_reason: "length"`:** a reasoning model spent
  all of `max_tokens` thinking before writing an answer. Raise
  `--max-tokens` or pick a non-reasoning model; `was_truncated` is set on
  the response so this is not mistaken for a real empty answer.
- **OpenRouter 404 "No endpoints found":** you asked for a capability such
  as logprobs that no provider behind that model supports. This is
  deliberate; the request fails instead of silently dropping the feature.
- **Many 429s on a provider without rate-limit headers (OpenRouter):** set
  `--rpm` lower, or add `--adaptive` so the rate backs off on its own.

## Does it actually help?

Measured against a simulated server enforcing 190 requests/minute, 200 prompts,
median of 5 runs ([full method and caveats](https://github.com/Yashwanth-Nallapuneni/llm-gateway/blob/main/benchmarks/README.md)):

| | naive async loop | llm-gateway |
|---|---|---|
| 429s received | 85 | **16** |
| requests sent | 275 | **206** |
| wall-clock | 14.3s | **6.1s** |
| success rate | 92.5% | 92.5% |
| latency p99 | **0.157s** | 0.518s |

The win is avoided rate-limit rejections and the wasted work behind them. The
cost is tail latency: batching waits, and throttled requests queue instead of
failing fast. **Below the provider's limit this library buys you nothing:**
the same benchmark at 150 prompts is a dead heat. If you are not near a rate
limit, do not add it.

## Status

The core is complete, tested and typed, and adapters for Groq and OpenRouter
are implemented and unit-tested against a fake HTTP server. Both adapters
have also been run against their real APIs. OpenRouter was validated live
once, against real completions, logprobs, and both a bad-model and a
bad-key error path; it also confirmed OpenRouter sends no rate-limit
headers, which is why the adaptive rate limiter above exists. See
`docs/LIVE_TESTING.md` for the full breakdown and cost (about $0.0045). The
Groq adapter has now run against the real Groq API three times
(`benchmarks/bench.py --provider groq`; see
[benchmarks/README.md](https://github.com/Yashwanth-Nallapuneni/llm-gateway/blob/main/benchmarks/README.md#live-results-real-groq-api) for
all three runs in full): real HTTP calls, real 429/`Retry-After` handling,
real token usage, no simulation involved, every time. The first two attempts
were methodology failures, kept on record rather than deleted. Attempt 1 had
the gateway arm's rate-limit ceiling set from a misread header (~200x too
permissive on the dimension that actually binds), so both arms got hit hard
by real 429s and the comparison tested nothing. Attempt 2 fixed the ceiling
but ran both arms back-to-back with no cooldown, so the naive arm's burst
drained the shared account's token bucket and the gateway arm absorbed
leftover 429s that weren't its own doing. Neither attempt found a defect in
the library itself: every 429 was correctly detected and retried both
times.

A third attempt (90s cooldown between arms, arm order alternated across
runs, 2 runs of 40 prompts each) fixed both earlier problems but left one
gap: the cooldown only applied *between arms within a run*, not between one
run's last arm and the next run's first arm, so account rate-limit state
could carry across run boundaries. Run 2's gateway arm absorbed 15 real
429s from run 1's leftover state as a result, and its naive arm's "17
failures" were actually the harness's own `--max-live-calls` budget running
out mid-arm, not real rejections -- an earlier version of this section
quoted that as a 57.5% success figure; it wasn't a real result, and the
mistake has been corrected here (never cite it).

The current, headline result is **attempt 4**, which fixes that gap: a 90s
cooldown before *every* arm, including across run boundaries, 3 runs of 40
prompts per arm, arm order alternated, $0 on the free tier. Across the 3
runs, the gateway succeeded on **120/120 prompts (100%, 0 rate-limit
rejections)**, at roughly 31s per run; naive succeeded on 107/120
(**97.5%, 85.0%, 85.0%** per run) with 87 real 429s total, at roughly
7-10s per run. The gateway is slower and that is the honest tradeoff shown
here: it paces itself under the account's real rate ceiling instead of
bursting and eating rejections the way the naive loop does, so it trades
wall-clock time for zero dropped prompts. This does **not** reproduce the
simulated benchmark's wall-clock win above, and it does not need to --
the two measure different things. Read
[benchmarks/README.md](https://github.com/Yashwanth-Nallapuneni/llm-gateway/blob/main/benchmarks/README.md#live-results-real-groq-api) for
the full breakdown of all four attempts, including the two earlier
methodology failures and the retracted figure, before citing any of these
numbers. Broader real-endpoint coverage (more providers, more models) is
the next milestone. See [ROADMAP.md](https://github.com/Yashwanth-Nallapuneni/llm-gateway/blob/main/ROADMAP.md).

## The problem

Driving LLM APIs at volume, four things go wrong and they compound:

1. **Rate limits.** Providers cap requests-per-minute and tokens-per-minute.
   Exceed either and you get `429`. Naive retry-on-429 makes it worse.
2. **Transient failures.** `502`, `503`, resets, gateway timeouts. Recoverable,
   but only with correct backoff.
3. **Wasted round trips.** Hundreds of small independent prompts sent one at a
   time pay full network and queueing latency each, when the provider would
   have taken sixteen of them together.
4. **Capability fragmentation.** One provider returns logprobs but not strict
   JSON schema; another does schema but not logprobs. Choosing by hand, per
   request, is error-prone in a way you find out about hours later.

This is the exact set of failures hit running batch evaluations across
DeepInfra, Novita and OpenRouter. The library formalizes what was previously
ad-hoc `time.sleep()` calls and manual provider switching.

## Usage

```python
from llm_gateway import (
    Batcher, LLMGateway, LLMRequest, Provider, ProviderCapabilities, RetryPolicy,
)

gateway = LLMGateway(
    providers=[
        Provider(
            name="provider_a",
            client=SomeAsyncClient(...),
            capabilities=ProviderCapabilities(
                supports_logprobs=True,
                supports_strict_json=False,
                supports_batching=True,
                max_context_tokens=32_000,
                cost_per_1k_input=0.05,
                cost_per_1k_output=0.10,
            ),
            rpm_limit=600,
            tpm_limit=150_000,
        ),
        Provider(name="provider_b", ...),
    ],
    batcher=Batcher(max_batch_size=16, max_wait_ms=50),
    retry=RetryPolicy(max_attempts=5),
)

async with gateway:
    resp = await gateway.submit(
        LLMRequest(prompt="...", max_tokens=256, needs_logprobs=True)
    )
    responses = await gateway.submit_many([LLMRequest(...) for _ in range(500)])

print(gateway.metrics.report())
```

`submit_many` is the headline: 500 requests in, correctly rate-limited,
batched, retried and failover-routed, without the caller writing a single
`sleep`.

Your client needs one method, `async complete(request) -> LLMResponse`, plus
`complete_batch(requests)` if it supports batching. `providers/mock.py` is a
complete worked example.

## Architecture

For a step-by-step walk through one request and a map of every file, see
[docs/ARCHITECTURE.md](https://github.com/Yashwanth-Nallapuneni/llm-gateway/blob/main/docs/ARCHITECTURE.md).

```
   caller ──submit()──▶ LLMGateway
                            │
                        RequestQueue        priority heap + FIFO tiebreak
                            │
                         Batcher            coalesce on size OR tokens OR time
                            │
                      ProviderRouter        capability filter → health → rank
                        │        │
                  RateLimiter  RateLimiter  two token buckets per provider
                        │        │
                  ProviderA    ProviderB    + RetryPolicy + CircuitBreaker
                        └────┬───┘
                        MetricsSink
```

Data flows down; results and failures propagate back up through the
`asyncio.Future` held by each queue entry.

## Running it

```bash
git clone https://github.com/Yashwanth-Nallapuneni/llm-gateway
cd llm-gateway
pip install -e ".[dev]"
pytest
python examples/bulk_eval.py
python examples/failover_demo.py
```

The default test run and every example use `MockProvider`, so they need no
API key and make no network calls. The live tests against Groq and
OpenRouter are opt-in (`pytest -m live`; see
[docs/LIVE_TESTING.md](https://github.com/Yashwanth-Nallapuneni/llm-gateway/blob/main/docs/LIVE_TESTING.md)). All examples are listed in
[examples/README.md](https://github.com/Yashwanth-Nallapuneni/llm-gateway/blob/main/examples/README.md); how to contribute is in
[CONTRIBUTING.md](https://github.com/Yashwanth-Nallapuneni/llm-gateway/blob/main/CONTRIBUTING.md), and what changed in each release is in
[CHANGELOG.md](https://github.com/Yashwanth-Nallapuneni/llm-gateway/blob/main/CHANGELOG.md).

## Measured output

`python examples/bulk_eval.py` runs 500 prompts across two mock providers, one cheap and
rate-limited, one expensive with headroom. Ten percent of the workload requires
logprobs, which only the expensive provider supports.

```text
==============================================================================
LLM GATEWAY METRICS
==============================================================================
submitted=500  completed=500  rejected=0

provider          ok  fail  retry      p50      p95      p99   blocked      cost
--------------------------------------------------------------------------------
cheap_co         144     0      0    0.021    0.021    0.021      0.00    0.3263
premium_co       356     0      0    0.031    0.031    0.031      0.00   10.1492

batch size distribution
    1 | ######################## 18
    2 | # 1
   16 | ######################################## 30
  mean batch size: 10.20 over 49 dispatches
==============================================================================

wall clock: 0.06s for 500 requests (8597 req/s)
logprob-requiring requests routed correctly: 50/50
```

The batch-size histogram is the thing to look at: 500 requests left as roughly
50 dispatches. The split between providers is the router at work: the cheap
provider takes the bulk until its bucket drains, at which point headroom beats
cost and traffic shifts rather than blocks.

`python examples/failover_demo.py` shows a provider failing mid-run:

```text
phase 1: both healthy, router prefers the cheaper provider
  {'primary': 20}

phase 2: primary starts returning 503
  {'backup': 20}
  primary breaker: open

phase 3: primary recovers, breaker half-opens and closes
  {'primary': 20}
  primary breaker: closed
```

## Scope

Client-side gateway only. A run-wide spending ceiling (`BudgetLedger`,
`--budget`) is supported, but routing itself is not cost-aware beyond the
existing headroom-then-cost tiebreak. Adaptive rate limiting (see above) is
opt-in and additive to the header-synced limiter, not a replacement for it.
Not implemented, on purpose: streaming (in fundamental tension with
batching) and a persistent queue that survives a crash.
