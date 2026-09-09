from __future__ import annotations

import asyncio

import pytest

from llm_gateway import (
    Batcher,
    LLMGateway,
    LLMRequest,
    NoEligibleProviderError,
    RetryPolicy,
)
from llm_gateway.providers.mock import MockClient, MockProvider


def fast_retry(**kw) -> RetryPolicy:
    kw.setdefault("base_delay", 0.001)
    kw.setdefault("max_delay", 0.01)
    return RetryPolicy(**kw)


async def test_single_request_round_trip():
    gw = LLMGateway(providers=[MockProvider("a")], retry=fast_retry())
    async with gw:
        resp = await gw.submit(LLMRequest("hello"))
    assert resp.provider == "a"
    assert "hello" in resp.text


async def test_submit_many_returns_results_in_input_order():
    gw = LLMGateway(
        providers=[MockProvider("a")],
        batcher=Batcher(max_batch_size=16, max_wait_ms=10),
        retry=fast_retry(),
    )
    async with gw:
        reqs = [LLMRequest(f"prompt-{i}") for i in range(60)]
        responses = await gw.submit_many(reqs)
    assert len(responses) == 60
    assert [r.text.split()[-1] for r in responses] == [f"prompt-{i}" for i in range(60)]


async def test_batching_actually_batches():
    client = MockClient("a")
    gw = LLMGateway(
        providers=[MockProvider("a", client=client)],
        batcher=Batcher(max_batch_size=16, max_wait_ms=15),
        retry=fast_retry(),
    )
    async with gw:
        await gw.submit_many([LLMRequest(f"p{i}") for i in range(100)])

    hist = gw.metrics.batch_histogram()
    assert sum(hist.values()) < 100, "should be far fewer dispatches than requests"
    assert max(hist) > 1


async def test_transient_failures_are_retried():
    client = MockClient("flaky", fail_first_n=2, fail_status=503)
    gw = LLMGateway(providers=[MockProvider("flaky", client=client)], retry=fast_retry())
    async with gw:
        resp = await gw.submit(LLMRequest("hi"))
    assert resp.provider == "flaky"
    assert resp.attempts == 3


async def test_non_retryable_failure_propagates_without_retrying():
    client = MockClient("strict", fail_status=400)
    gw = LLMGateway(providers=[MockProvider("strict", client=client)], retry=fast_retry())
    async with gw:
        with pytest.raises(Exception) as excinfo:
            await gw.submit(LLMRequest("hi"))
    assert getattr(excinfo.value, "status", None) == 400
    assert client.calls == 1


async def test_failover_when_primary_starts_returning_503():
    primary_client = MockClient("primary")
    primary = MockProvider(
        "primary", client=primary_client, cost_per_1k_output=0.01, failure_threshold=3
    )
    backup = MockProvider("backup", cost_per_1k_output=5.00)

    gw = LLMGateway(
        providers=[primary, backup],
        batcher=Batcher(max_batch_size=1, max_wait_ms=1),
        retry=fast_retry(max_attempts=2),
    )
    async with gw:
        first = await gw.submit(LLMRequest("before"))
        assert first.provider == "primary"

        # Primary goes hard down mid-run.
        primary_client.fail_status = 503
        results = [await gw.submit(LLMRequest(f"during-{i}")) for i in range(8)]

    assert all(r.provider == "backup" for r in results)
    # Once the breaker opens, the router stops even trying the primary.
    assert primary.breaker.state.value == "open"


async def test_rate_limiting_prevents_provider_side_429s():
    # The mock enforces its own 60 rpm; the gateway is configured to match.
    client = MockClient("limited", rpm_limit=60)
    gw = LLMGateway(
        providers=[MockProvider("limited", client=client, rpm_limit=60, tpm_limit=10**7)],
        batcher=Batcher(max_batch_size=8, max_wait_ms=5),
        retry=fast_retry(),
    )
    async with gw:
        await gw.submit_many([LLMRequest(f"p{i}") for i in range(50)])
    assert gw.metrics._p("limited").failures_by_class["429"] == 0


async def test_uncompletable_request_fails_with_a_named_reason():
    gw = LLMGateway(providers=[MockProvider("plain")], retry=fast_retry())
    async with gw:
        with pytest.raises(NoEligibleProviderError) as excinfo:
            await gw.submit(LLMRequest("hi", needs_logprobs=True))
    assert "logprobs" in str(excinfo.value)


async def test_capability_routing_end_to_end():
    plain = MockProvider("plain")
    fancy = MockProvider("fancy", supports_logprobs=True)
    gw = LLMGateway(
        providers=[plain, fancy],
        batcher=Batcher(max_batch_size=4, max_wait_ms=5),
        retry=fast_retry(),
    )
    async with gw:
        resp = await gw.submit(LLMRequest("hi", needs_logprobs=True))
    assert resp.provider == "fancy"
    assert resp.logprobs is not None


async def test_metrics_report_renders():
    gw = LLMGateway(
        providers=[MockProvider("a")],
        batcher=Batcher(max_batch_size=8, max_wait_ms=5),
        retry=fast_retry(),
    )
    async with gw:
        await gw.submit_many([LLMRequest(f"p{i}") for i in range(20)])
    report = gw.metrics.report()
    assert "batch size distribution" in report
    assert "p99" in report
    assert gw.metrics.completed == 20


async def test_high_priority_request_jumps_the_queue():
    """A burst arrives faster than it can be dispatched; priority decides
    who is in the first batch out the door."""
    client = MockClient("a", latency=0.01)
    dispatched: list[list[str]] = []
    original = client.complete_batch

    async def spy(requests):
        dispatched.append([r.prompt for r in requests])
        return await original(requests)

    client.complete_batch = spy

    gw = LLMGateway(
        providers=[MockProvider("a", client=client)],
        batcher=Batcher(max_batch_size=4, max_wait_ms=5),
        retry=fast_retry(),
    )
    async with gw:
        reqs = [LLMRequest(f"low-{i}") for i in range(40)]
        reqs.append(LLMRequest("URGENT", priority=100))
        await gw.submit_many(reqs)

    assert dispatched, "expected at least one batched dispatch"
    assert "URGENT" in dispatched[0], f"urgent request landed in {dispatched[:2]}"


async def test_breaker_recovers_and_traffic_returns_to_the_primary():
    """Regression: the router and the dispatch path used to consume the
    half-open probe twice, wedging a recovered provider open forever."""
    primary_client = MockClient("primary", latency=0.001)
    primary = MockProvider(
        "primary",
        client=primary_client,
        cost_per_1k_output=0.01,
        failure_threshold=3,
        recovery_timeout=0.05,
    )
    backup = MockProvider("backup", cost_per_1k_output=5.0)
    gw = LLMGateway(
        providers=[primary, backup],
        batcher=Batcher(max_batch_size=1, max_wait_ms=1),
        retry=fast_retry(max_attempts=2),
    )
    async with gw:
        primary_client.fail_status = 503
        for i in range(6):
            await gw.submit(LLMRequest(f"down-{i}"))
        assert primary.breaker.state.value == "open"

        primary_client.fail_status = None
        await asyncio.sleep(0.08)  # let the recovery timeout elapse

        resp = await gw.submit(LLMRequest("after-recovery"))

    assert resp.provider == "primary"
    assert primary.breaker.state.value == "closed"


async def test_batches_do_not_mix_capability_requirements_end_to_end():
    # Large limits on both so headroom stays level and cost decides; this
    # test is about capability grouping, not about the headroom ranker.
    plain = MockProvider("plain", cost_per_1k_output=0.01, rpm_limit=100_000, tpm_limit=10**7)
    fancy = MockProvider(
        "fancy", supports_logprobs=True, cost_per_1k_output=5.0, rpm_limit=100_000, tpm_limit=10**7
    )
    gw = LLMGateway(
        providers=[plain, fancy],
        batcher=Batcher(max_batch_size=16, max_wait_ms=10),
        retry=fast_retry(),
    )
    reqs = [LLMRequest(f"p{i}", needs_logprobs=(i % 10 == 0)) for i in range(100)]
    async with gw:
        responses = await gw.submit_many(reqs)

    for req, resp in zip(reqs, responses):
        if req.needs_logprobs:
            assert resp.provider == "fancy"
    # The cheap provider must still get the bulk of the work.
    assert sum(r.provider == "plain" for r in responses) == 90
