"""Adaptive rate limiting plus per-request timeouts, against a flaky provider.

Run:  python examples/timeouts_and_adaptive.py

MockClient is set up to return a 429 on the first two calls and then behave.
With adaptive=True, the gateway halves its request rate the moment it sees a
429 and creeps back up as calls succeed -- useful for a provider (like
OpenRouter) that doesn't send rate-limit headers to sync from. Each request
also carries a short timeout_s, so a request that fails to get a response in
time is reported as a timeout instead of hanging the run.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from llm_gateway import LLMGateway, LLMRequest, RetryPolicy
from llm_gateway.providers.mock import MockClient, MockProvider


async def main() -> None:
    client = MockClient("flaky", latency=0.02, fail_sequence=[429, 429])
    provider = MockProvider("flaky", client=client, rpm_limit=600)

    gw = LLMGateway(
        providers=[provider],
        retry=RetryPolicy(max_attempts=4, base_delay=0.05, max_delay=1.0),
        adaptive=True,
    )

    requests = [
        LLMRequest(prompt=f"question {i}", max_tokens=32, timeout_s=5.0)
        for i in range(10)
    ]

    print("sending 10 requests; the first two calls to the provider hit a 429\n")
    async with gw:
        responses = await gw.submit_many(requests)

    ok = sum(1 for r in responses if r is not None)
    print(f"all {ok}/{len(responses)} requests eventually succeeded")
    current_rpm = provider.limiter.requests.refill_rate * 60.0
    print(f"provider limiter rate after the run: {current_rpm:.1f} rpm (started at 600)")
    print("\n" + gw.metrics.report())


if __name__ == "__main__":
    asyncio.run(main())
