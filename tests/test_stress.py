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


@pytest.mark.parametrize("seed", [1, 2])
async def test_closing_mid_run_resolves_every_caller(tmp_path: Path, seed: int) -> None:
    # Close the gateway while requests are queued and in flight. Every
    # caller must still get an answer or an error; none may wait forever,
    # and the store must not be left with rows marked in_flight.
    rng = random.Random(seed)
    store = RunStore(tmp_path / f"close-{seed}.db")
    gw = LLMGateway(providers=make_providers(rng), retry=fast_retry(), store=store)

    async def submit_one(i: int) -> object:
        try:
            return await gw.submit(LLMRequest(f"prompt-{i}"))
        except (ProviderError, GatewayError) as exc:
            return exc

    tasks = [asyncio.ensure_future(submit_one(i)) for i in range(N_REQUESTS)]
    await asyncio.sleep(0.02)
    await gw.aclose()
    results = await asyncio.wait_for(asyncio.gather(*tasks), timeout=OVERALL_TIMEOUT)

    assert len(results) == N_REQUESTS
    assert all(isinstance(r, (LLMResponse, ProviderError, GatewayError)) for r in results)
    counts = await store.counts()
    assert counts.get("in_flight", 0) == 0
    await store.aclose()


@pytest.mark.parametrize("seed", [1, 2])
async def test_budget_runs_out_and_capabilities_are_mixed(
    tmp_path: Path, seed: int
) -> None:
    # A small budget that runs out partway, and some requests that need
    # logprobs, which only one provider offers. Every caller must still get
    # an answer or an error, and no budget hold may be left behind.
    rng = random.Random(seed)
    plain = make_providers(rng)
    with_logprobs = MockProvider(
        "logprob",
        supports_logprobs=True,
        client=MockClient(name="logprob", supports_logprobs=True, latency=0.001),
    )
    budget = BudgetLedger(0.05)
    store = RunStore(tmp_path / f"budget-{seed}.db")
    gw = LLMGateway(
        providers=[*plain, with_logprobs], retry=fast_retry(), budget=budget, store=store
    )

    async def submit_one(i: int) -> object:
        req = LLMRequest(f"prompt-{i}", needs_logprobs=rng.random() < 0.3)
        try:
            return await gw.submit(req)
        except (ProviderError, GatewayError, BudgetExceeded) as exc:
            return exc

    async with gw:
        results = await asyncio.wait_for(
            asyncio.gather(*(submit_one(i) for i in range(N_REQUESTS))),
            timeout=OVERALL_TIMEOUT,
        )
        for _ in range(20):
            if budget.outstanding_count == 0:
                break
            await asyncio.sleep(0.05)

    assert len(results) == N_REQUESTS
    assert any(isinstance(r, BudgetExceeded) for r in results)
    assert any(isinstance(r, LLMResponse) for r in results)
    assert budget.outstanding_count == 0
    counts = await store.counts()
    assert counts.get("in_flight", 0) == 0
    await store.aclose()


async def test_every_provider_down_fails_fast_without_hanging() -> None:
    # Both providers fail every call, so their breakers open. Every caller
    # must get an error rather than wait, and quickly.
    providers = [
        MockProvider(
            name,
            client=MockClient(name=name, latency=0.001, fail_sequence=[500] * 10_000),
            failure_threshold=2,
        )
        for name in ("down1", "down2")
    ]
    gw = LLMGateway(providers=providers, retry=fast_retry())

    async def submit_one(i: int) -> object:
        try:
            return await gw.submit(LLMRequest(f"prompt-{i}"))
        except (ProviderError, GatewayError) as exc:
            return exc

    async with gw:
        results = await asyncio.wait_for(
            asyncio.gather(*(submit_one(i) for i in range(N_REQUESTS))), timeout=10.0
        )
    assert len(results) == N_REQUESTS
    assert not any(isinstance(r, LLMResponse) for r in results)


async def test_providers_recover_under_load() -> None:
    # Each provider fails its first calls and trips its breaker. While both
    # are open, callers get a fast error (that is the point of a breaker).
    # Once the failures stop, the breakers must close again rather than get
    # stuck half-open, and a second wave of traffic must fully succeed.
    providers = [
        MockProvider(
            name,
            client=MockClient(name=name, latency=0.002, fail_sequence=[500] * 6),
            failure_threshold=2,
            recovery_timeout=0.05,
        )
        for name in ("p1", "p2")
    ]
    gw = LLMGateway(providers=providers, retry=fast_retry())

    async def submit_one(i: int) -> object:
        try:
            return await gw.submit(LLMRequest(f"prompt-{i}"))
        except (ProviderError, GatewayError) as exc:
            return exc

    async with gw:
        # First wave: keep sending until both providers have recovered.
        for _ in range(100):
            await asyncio.wait_for(submit_one(0), timeout=5.0)
            if all(p.breaker.state.value == "closed" for p in providers):
                break
            await asyncio.sleep(0.01)
        assert all(p.breaker.state.value == "closed" for p in providers)

        second_wave = await asyncio.wait_for(
            asyncio.gather(*(submit_one(i) for i in range(N_REQUESTS))), timeout=10.0
        )
    assert all(isinstance(r, LLMResponse) for r in second_wave)
