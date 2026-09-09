"""Watch traffic move off a provider that fails mid-run.

Run:  python examples/failover_demo.py

The primary is cheap, so the router prefers it. Partway through the run it
starts returning 503 on every call. The retry policy absorbs the first few,
the circuit breaker trips, and the router stops offering it work at all.
"""

from __future__ import annotations

import asyncio
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from llm_gateway import Batcher, LLMGateway, LLMRequest, RetryPolicy  # noqa: E402
from llm_gateway.providers.mock import MockClient, MockProvider  # noqa: E402


async def main() -> None:
    primary_client = MockClient("primary", latency=0.01)
    primary = MockProvider(
        "primary",
        client=primary_client,
        cost_per_1k_input=0.01,
        cost_per_1k_output=0.02,
        failure_threshold=3,
        recovery_timeout=0.5,
    )
    backup = MockProvider(
        "backup",
        client=MockClient("backup", latency=0.01),
        cost_per_1k_input=0.50,
        cost_per_1k_output=1.00,
    )

    gw = LLMGateway(
        providers=[primary, backup],
        batcher=Batcher(max_batch_size=4, max_wait_ms=10),
        retry=RetryPolicy(max_attempts=3, base_delay=0.02, max_delay=0.2),
    )

    used: Counter = Counter()

    async with gw:
        print("phase 1: both healthy, router prefers the cheaper provider")
        for i in range(20):
            used[(await gw.submit(LLMRequest(f"p{i}"))).provider] += 1
        print(f"  {dict(used)}\n")

        print("phase 2: primary starts returning 503")
        primary_client.fail_status = 503
        phase2: Counter = Counter()
        for i in range(20):
            phase2[(await gw.submit(LLMRequest(f"q{i}"))).provider] += 1
        print(f"  {dict(phase2)}")
        print(f"  primary breaker: {primary.breaker.state.value}\n")

        print("phase 3: primary recovers, breaker half-opens and closes")
        primary_client.fail_status = None
        await asyncio.sleep(0.6)  # let the recovery timeout elapse
        phase3: Counter = Counter()
        for i in range(20):
            phase3[(await gw.submit(LLMRequest(f"r{i}"))).provider] += 1
        print(f"  {dict(phase3)}")
        print(f"  primary breaker: {primary.breaker.state.value}\n")

    print(gw.metrics.report())


if __name__ == "__main__":
    asyncio.run(main())
