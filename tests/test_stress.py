"""Stress test: many concurrent requests against flaky mock providers.

Offline and deterministic (fixed seeds). Checks that a large, mixed batch of
submits never hangs, that every result is either a response or one of the
library's documented exceptions, and that nothing is left dangling in the
budget ledger or the run store once things settle.
"""

from __future__ import annotations

import asyncio
import random
from pathlib import Path

import pytest

from llm_gateway import (
    BudgetExceeded,
    BudgetLedger,
    GatewayError,
    LLMGateway,
    LLMRequest,
    LLMResponse,
    ProviderError,
    RequestTimeout,
    RetryPolicy,
    RunStore,
)
from llm_gateway.providers.base import Provider
from llm_gateway.providers.mock import MockClient, MockProvider

N_REQUESTS = 300
OVERALL_TIMEOUT = 30.0


def fast_retry() -> RetryPolicy:
    return RetryPolicy(max_attempts=3, base_delay=0.001, max_delay=0.01)


def make_providers(rng: random.Random) -> list[Provider]:
    providers: list[Provider] = []
    for i, name in enumerate(["alpha", "beta", "gamma"]):
        # A mixed bag of failures: some 429s, some 500s, a couple of
        # non-retryable 400s, interspersed with successes (None).
        pool = [429, 500, 500, 400, None, None, None, 429, None]
        fail_sequence = [rng.choice(pool) for _ in range(200)]
        client = MockClient(
            name=name,
            latency=0.001 + 0.001 * i,
            fail_sequence=fail_sequence,
            retry_after=0.001,
        )
        providers.append(MockProvider(name, client=client, priority=i))
    return providers


@pytest.mark.parametrize("seed", [1, 2, 3])
async def test_concurrent_load_never_hangs_and_settles_cleanly(
    tmp_path: Path, seed: int
) -> None:
    rng = random.Random(seed)
    providers = make_providers(rng)
    budget = BudgetLedger(10_000.0)
    store = RunStore(tmp_path / f"run-{seed}.db")

    gw = LLMGateway(
        providers=providers,
        retry=fast_retry(),
        budget=budget,
        store=store,
        adaptive=True,
    )

    requests = []
    for i in range(N_REQUESTS):
        priority = rng.choice([0, 1, 2, 5])
        # A slice of requests get an aggressive timeout to exercise the
        # timeout path alongside plain failures.
        timeout_s = 0.005 if rng.random() < 0.1 else None
        requests.append(LLMRequest(f"prompt-{i}", priority=priority, timeout_s=timeout_s))

    async def submit_one(req: LLMRequest) -> LLMResponse | GatewayError:
        try:
            return await gw.submit(req)
        except (ProviderError, RequestTimeout, GatewayError, BudgetExceeded) as exc:
            return exc

    async with gw:
        results = await asyncio.wait_for(
            asyncio.gather(*(submit_one(r) for r in requests)),
            timeout=OVERALL_TIMEOUT,
        )

        assert len(results) == N_REQUESTS
        for r in results:
            assert isinstance(
                r,
                (
                    LLMResponse,
                    ProviderError,
                    RequestTimeout,
                    GatewayError,
                    BudgetExceeded,
                ),
            )

        # Give any background bookkeeping a moment to settle, then confirm
        # nothing was left outstanding.
        for _ in range(20):
            counts = await store.counts()
            if budget.outstanding_count == 0 and counts.get("in_flight", 0) == 0:
                break
            await asyncio.sleep(0.05)

        assert budget.outstanding_count == 0
        counts = await store.counts()
        assert counts.get("in_flight", 0) == 0

    await store.aclose()
