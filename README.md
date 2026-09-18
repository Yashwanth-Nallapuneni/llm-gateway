# llm-gateway

**Send 500 prompts to an LLM API without writing a single `sleep()`.**

An async Python layer that sits between your code and a provider's API. It
paces requests under rate limits, retries what is worth retrying, groups
prompts to cut round trips, and moves traffic off a provider that starts
failing.

[![CI](https://github.com/Yashwanth-Nallapuneni/llm-gateway/actions/workflows/ci.yml/badge.svg)](https://github.com/Yashwanth-Nallapuneni/llm-gateway/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/Yashwanth-Nallapuneni/llm-gateway/branch/main/graph/badge.svg)](https://codecov.io/gh/Yashwanth-Nallapuneni/llm-gateway)
[![PyPI](https://img.shields.io/pypi/v/aiollm-gateway.svg)](https://pypi.org/project/aiollm-gateway/)
[![Python](https://img.shields.io/pypi/pyversions/aiollm-gateway.svg)](https://pypi.org/project/aiollm-gateway/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Python 3.11+, asyncio, **zero runtime dependencies**. Not a proxy, not a
server, no caching or prompt management.

```bash
pip install aiollm-gateway
```

## Quickstart

This runs as-is — no API key, no network. `MockProvider` ships with the library
precisely so you can see the machinery work before wiring up a real provider.

```python
import asyncio
from llm_gateway import Batcher, LLMGateway, LLMRequest, MockProvider

async def main():
    gateway = LLMGateway(
        providers=[MockProvider("cheap", rpm_limit=600, cost_per_1k_output=0.02),
                   MockProvider("backup", rpm_limit=1200, cost_per_1k_output=0.50)],
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

## Status

Honest current state: the core is complete, tested and typed, but every
number below comes from the built-in mock provider. Adapters for real
providers (Groq, OpenRouter) and a measured benchmark against them are the
next milestones — see [ROADMAP.md](ROADMAP.md).

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

Everything in the suite and both examples run against `MockProvider`. There are
no network calls anywhere in this repository.

## Measured output

`python examples/bulk_eval.py` — 500 prompts, two mock providers, one cheap and
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
50 dispatches. The split between providers is the router at work — the cheap
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

## This repo is a teaching artifact

The four core modules are annotated to be read, not just run. Every non-obvious
decision carries a `WHY:` / `ALT:` / `TRAP:` / `ASK:` tag, and each has a
companion `EXPLAIN.md` written to be read *before* the code.

`scripts/strip.py` blanks the bodies of exactly four functions —
`TokenBucket.acquire`, `TokenBucket.try_acquire`, `RetryPolicy.delay_for`,
`Batcher.collect`, `ProviderRouter.select` — leaving signatures, docstrings and
all tests untouched, after copying the originals to `.reference/`. Rebuild them
against the failing suite, then diff.

See **[LEARNING_PATH.md](LEARNING_PATH.md)** for the reading order, the rebuild
loop, and the full list of self-check questions.

## Scope

Client-side gateway only. Not implemented, on purpose: streaming (in
fundamental tension with batching), cost-aware routing with a budget ceiling,
a persistent queue that survives a crash, and adaptive rate limiting that
infers the real limit from observed `429`s rather than trusting configuration.
