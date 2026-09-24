"""500 prompts through the gateway, end to end.

Run:  python examples/bulk_eval.py

Everything talks to MockProvider, so this makes no network calls. The two
providers are configured to behave differently on purpose: one is cheap but
tightly rate-limited and occasionally flaky, the other is expensive with
plenty of headroom.
"""

from __future__ import annotations

import asyncio
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from llm_gateway import Batcher, LLMGateway, LLMRequest, RetryPolicy
from llm_gateway.providers.mock import MockClient, MockProvider

N_REQUESTS = 500


def build_gateway() -> LLMGateway:
    cheap = MockProvider(
        "cheap_co",
        client=MockClient("cheap_co", latency=0.02, rpm_limit=400),
        rpm_limit=400,
        tpm_limit=120_000,
        cost_per_1k_input=0.02,
        cost_per_1k_output=0.04,
    )
    premium = MockProvider(
        "premium_co",
        client=MockClient("premium_co", latency=0.03, supports_logprobs=True),
        rpm_limit=1200,
        tpm_limit=400_000,
        cost_per_1k_input=0.20,
        cost_per_1k_output=0.60,
        supports_logprobs=True,
    )
    return LLMGateway(
        providers=[cheap, premium],
        batcher=Batcher(max_batch_size=16, max_wait_ms=25, max_batch_tokens=8000),
        retry=RetryPolicy(max_attempts=4, base_delay=0.05, max_delay=2.0),
    )


async def main() -> None:
    rng = random.Random(11)
    requests = [
        LLMRequest(
            prompt=f"Classify sentiment of review #{i}: " + "word " * rng.randint(5, 60),
            max_tokens=64,
            # A tenth of the workload needs logprobs, which only one provider
            # supports -- this is what exercises capability routing.
            needs_logprobs=(i % 10 == 0),
            priority=1 if i % 50 == 0 else 0,
        )
        for i in range(N_REQUESTS)
    ]

    gw = build_gateway()
    print(f"submitting {N_REQUESTS} requests...\n")
    started = time.monotonic()
    async with gw:
        responses = await gw.submit_many(requests)
    elapsed = time.monotonic() - started

    assert len(responses) == N_REQUESTS
    logprob_responses = [
        r for r, q in zip(responses, requests, strict=True) if q.needs_logprobs
    ]
    assert all(r.logprobs is not None for r in logprob_responses)

    print(gw.metrics.report())
    print(
        f"\nwall clock: {elapsed:.2f}s for {N_REQUESTS} requests "
        f"({N_REQUESTS / elapsed:.0f} req/s)"
    )
    print(
        f"logprob-requiring requests routed correctly: "
        f"{len(logprob_responses)}/{len(logprob_responses)}"
    )


if __name__ == "__main__":
    asyncio.run(main())
