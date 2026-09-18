from __future__ import annotations

import asyncio
import time

import pytest

from llm_gateway import (
    Batcher,
    LLMGateway,
    LLMRequest,
    LLMResponse,
    NoEligibleProviderError,
    ProviderError,
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

    for req, resp in zip(reqs, responses, strict=True):
        if req.needs_logprobs:
            assert resp.provider == "fancy"
    # The cheap provider must still get the bulk of the work.
    assert sum(r.provider == "plain" for r in responses) == 90


# --------------------------------------------------------------------------
# Regression tests for the four gaps fixed after the first build
# --------------------------------------------------------------------------


async def test_concurrency_cap_bounds_simultaneous_calls():
    client = MockClient("capped", latency=0.02)
    provider = MockProvider("capped", client=client, max_concurrency=2)
    gw = LLMGateway(
        providers=[provider],
        batcher=Batcher(max_batch_size=1, max_wait_ms=1),
        retry=fast_retry(),
    )
    async with gw:
        await gw.submit_many([LLMRequest(f"p{i}") for i in range(20)])
    assert client.peak_in_flight == 2


async def test_non_batching_fallback_dispatches_concurrently():
    # No client.complete_batch -- forces Provider.complete_batch() onto the
    # fallback path used by real non-batching providers (Groq, OpenRouter).
    client = MockClient("seq", latency=0.2)
    provider = MockProvider(
        "seq", client=client, supports_batching=False, max_concurrency=16
    )
    start = time.monotonic()
    responses = await provider.complete_batch([LLMRequest(f"p{i}") for i in range(16)])
    elapsed = time.monotonic() - start

    assert len(responses) == 16
    assert client.peak_in_flight > 1
    # Sequential would take ~16 * 0.2s = 3.2s; concurrent stays near one
    # latency. Generous margin to keep this stable under load.
    assert elapsed < 1.0


async def test_max_concurrency_bounds_non_batching_fallback():
    client = MockClient("seq", latency=0.05)
    provider = MockProvider(
        "seq", client=client, supports_batching=False, max_concurrency=2
    )
    responses = await asyncio.wait_for(
        provider.complete_batch([LLMRequest(f"p{i}") for i in range(16)]),
        timeout=5.0,
    )
    assert len(responses) == 16
    assert client.peak_in_flight == 2


async def test_non_batching_fallback_preserves_order():
    client = MockClient("seq", latency=0.01)
    provider = MockProvider("seq", client=client, supports_batching=False)
    requests = [LLMRequest(f"prompt-{i}") for i in range(10)]
    responses = await provider.complete_batch(requests)
    for i, resp in enumerate(responses):
        assert f"prompt-{i}" in resp.text


async def test_non_batching_fallback_failure_propagates_without_orphans():
    client = MockClient("flaky", latency=0.05)
    # Every 4th call injected via fail_sequence-like behavior: simplest is
    # fail_status set, but that would fail every call. Use fail_first_n=0 and
    # instead flip a flag after enough calls have started by making the 5th
    # request itself fail through fail_sequence.
    fail_after = [None] * 4 + [503] + [None] * 11
    client.fail_sequence = fail_after
    provider = MockProvider(
        "flaky", client=client, supports_batching=False, max_concurrency=16
    )

    with pytest.raises(ProviderError):
        await asyncio.wait_for(
            provider.complete_batch([LLMRequest(f"p{i}") for i in range(16)]),
            timeout=5.0,
        )

    # Give any leaked task a chance to run, then confirm nothing is still in
    # flight. `in_flight` is decremented in a `finally` around every call, so
    # a task that was left running (never cancelled, never awaited) after the
    # gather() raised would show up here as still in flight.
    await asyncio.sleep(0.2)
    assert client.in_flight == 0


async def test_short_batch_response_fails_over_instead_of_hanging():
    short = MockClient("short")
    short.drop_responses = 1
    broken = MockProvider("short", client=short, cost_per_1k_output=0.01)
    backup = MockProvider("backup", cost_per_1k_output=5.0)
    gw = LLMGateway(
        providers=[broken, backup],
        batcher=Batcher(max_batch_size=4, max_wait_ms=20),
        retry=fast_retry(max_attempts=2),
    )
    async with gw:
        responses = await asyncio.wait_for(
            gw.submit_many([LLMRequest(f"p{i}") for i in range(4)]), timeout=2.0
        )
    assert len(responses) == 4
    assert all(r.provider == "backup" for r in responses)
    assert gw.metrics._p("short").failures_by_class["transport"] >= 1


async def test_reported_latency_is_end_to_end_not_call_time():
    # The provider answers instantly; all the delay is the batcher's wait.
    gw = LLMGateway(
        providers=[MockProvider("a")],
        batcher=Batcher(max_batch_size=100, max_wait_ms=60),
        retry=fast_retry(),
    )
    gw.batcher._in_flight = lambda: 1  # force the batcher to wait out the window
    async with gw:
        resp = await gw.submit(LLMRequest("hi"))
    assert resp.latency_s >= 0.05
    assert min(gw.metrics._p("a").latencies) >= 0.05


def test_composite_request_uses_a_real_member_and_ors_capabilities():
    loop = asyncio.new_event_loop()
    try:
        gw = LLMGateway(providers=[MockProvider("a")])
        from llm_gateway.types import QueuedRequest

        small = LLMRequest("x" * 40, max_tokens=10, needs_logprobs=True)
        big = LLMRequest("y" * 4000, max_tokens=500)
        batch = [
            QueuedRequest(request=r, future=loop.create_future(), enqueued_at=0.0)
            for r in (small, big)
        ]
        composite = gw._composite(batch)
    finally:
        loop.close()

    assert composite.max_tokens == 500  # a real max_tokens, not a token total
    assert composite.estimated_total_tokens() == big.estimated_total_tokens()
    assert composite.needs_logprobs is True
    assert big.needs_logprobs is False  # members are not mutated


def test_max_concurrency_must_be_positive():
    with pytest.raises(ValueError):
        MockProvider("bad", max_concurrency=0)


async def test_injected_clock_drives_latency_and_blocked_metrics():
    # A deterministic fake clock, offset-compatible with real
    # time.monotonic() (it starts near "now") so the batcher's own
    # real-time deadline math (batching.py) does not spuriously expire --
    # see the constraint documented on LLMGateway._clock in gateway.py.
    # Each call advances by a fixed step, so every duration the gateway
    # derives from the clock is exactly predictable.
    base = time.monotonic()
    step = 0.25
    calls = {"n": 0}

    def fake_clock() -> float:
        value = base + calls["n"] * step
        calls["n"] += 1
        return value

    gw = LLMGateway(providers=[MockProvider("a")], retry=fast_retry(), clock=fake_clock)
    async with gw:
        resp = await gw.submit(LLMRequest("hello"))

    # 4 clock reads for a single successful, non-retried request:
    # enqueued_at, blocked_start, the post-acquire blocked-time read, and the
    # final "now" used for end-to-end latency. That is exactly 3 steps.
    assert resp.latency_s == pytest.approx(3 * step)
    assert gw.metrics._p("a").blocked_seconds == pytest.approx(step)
    assert calls["n"] == 4


# --------------------------------------------------------------------------
# Per-request blast radius on non-batching providers (Groq, OpenRouter):
# one bad request in a batch must not take its healthy siblings down with
# it, which is the defect the benchmark caught.
# --------------------------------------------------------------------------


async def test_one_transient_failure_among_sixteen_only_retries_itself():
    # Exactly one 503 seeded into a 16-request dispatch to a non-batching
    # provider, plus exactly one more slot for the retry that should
    # follow. If the old all-or-nothing batch retry were still in effect,
    # the whole batch of 16 would be retried and this client would see 32
    # calls instead of 17.
    sequence: list[int | None] = [None] * 16
    sequence[7] = 503
    sequence.append(None)  # the lone retry of the one request that failed
    client = MockClient("fanout", latency=0.01, fail_sequence=sequence)
    provider = MockProvider(
        "fanout", client=client, supports_batching=False, max_concurrency=16
    )
    gw = LLMGateway(
        providers=[provider],
        batcher=Batcher(max_batch_size=16, max_wait_ms=50),
        retry=fast_retry(),
    )
    async with gw:
        responses = await gw.submit_many([LLMRequest(f"p{i}") for i in range(16)])

    assert len(responses) == 16
    assert all(r.provider == "fanout" for r in responses)
    assert client.calls == 17


async def test_one_permanent_failure_among_fifteen_healthy_is_isolated():
    class PoisonClient:
        """One prompt always 503s; everything else succeeds immediately."""

        def __init__(self) -> None:
            self.calls = 0

        async def complete(self, request: LLMRequest) -> LLMResponse:
            self.calls += 1
            if request.prompt == "poison":
                raise ProviderError("permanently broken", status=503, provider="p")
            return LLMResponse(text=f"ok:{request.prompt}", provider="p")

    provider = MockProvider(
        "p", client=PoisonClient(), supports_batching=False, max_concurrency=16
    )
    gw = LLMGateway(
        providers=[provider],
        batcher=Batcher(max_batch_size=16, max_wait_ms=50),
        retry=fast_retry(max_attempts=3),
    )
    reqs = [LLMRequest(f"p{i}") for i in range(15)] + [LLMRequest("poison")]
    async with gw:
        results = await asyncio.gather(
            *(gw.submit(r) for r in reqs), return_exceptions=True
        )

    healthy, poisoned = results[:15], results[15]
    assert all(isinstance(r, LLMResponse) for r in healthy)
    assert isinstance(poisoned, ProviderError)


async def test_true_batch_provider_still_fails_as_a_unit():
    # supports_batching=True: one call really is one HTTP round trip, so a
    # failure still has to take the whole group down together -- unlike the
    # fan-out path above, there is no such thing as "3 of 4 succeeded".
    client = MockClient("atomic", fail_status=503)
    provider = MockProvider("atomic", client=client)  # supports_batching=True (default)
    backup = MockProvider("backup")
    gw = LLMGateway(
        providers=[provider, backup],
        batcher=Batcher(max_batch_size=4, max_wait_ms=20),
        retry=fast_retry(max_attempts=1),
    )
    async with gw:
        responses = await gw.submit_many([LLMRequest(f"p{i}") for i in range(4)])
    assert all(r.provider == "backup" for r in responses)


async def test_complete_batch_settled_shares_one_exception_for_a_true_batch_failure():
    # The internal per-request view still reports a real batch endpoint's
    # failure as one event, not four -- it hands every entry back the same
    # exception object rather than four separately-raised equal ones. The
    # gateway's breaker/metrics accounting for the true-batch path depends
    # on being able to tell "one call failed" from "four calls failed" by
    # checking object identity (see the comment in gateway.py).
    client = MockClient("atomic", fail_status=503)
    provider = MockProvider("atomic", client=client)
    results = await provider.complete_batch_settled([LLMRequest(f"p{i}") for i in range(4)])
    assert len(results) == 4
    assert all(isinstance(r, ProviderError) for r in results)
    first = results[0]
    assert all(r is first for r in results)


async def test_partial_failure_leaves_no_orphaned_tasks_or_futures():
    fail_at = {3, 9}
    sequence: list[int | None] = [503 if i in fail_at else None for i in range(16)]
    sequence.extend([None, None])  # both retries succeed
    client = MockClient("fanout", latency=0.01, fail_sequence=sequence)
    provider = MockProvider(
        "fanout", client=client, supports_batching=False, max_concurrency=16
    )
    gw = LLMGateway(
        providers=[provider],
        batcher=Batcher(max_batch_size=16, max_wait_ms=50),
        retry=fast_retry(),
    )
    async with gw:
        responses = await gw.submit_many([LLMRequest(f"p{i}") for i in range(16)])
        # Every future this dispatch owned is resolved (submit_many would
        # not have returned otherwise) and the worker task that resolved
        # them has actually finished, not just handed back its result --
        # i.e. nothing is left running in the background.
        await asyncio.sleep(0.05)
        assert all(worker.done() for worker in gw._workers)

    assert len(responses) == 16
    assert all(r.provider == "fanout" for r in responses)
    assert client.in_flight == 0


async def test_whole_provider_outage_still_fails_over_every_request_in_the_batch():
    # Existing failover behaviour, generalized past a batch size of 1: when
    # every request in a multi-request dispatch fails against a
    # non-batching provider, all of them -- not just the one blocking the
    # retry budget -- move to the backup together.
    client = MockClient("dead", fail_status=503)
    dead = MockProvider("dead", client=client, supports_batching=False, max_concurrency=16)
    backup = MockProvider("backup")
    gw = LLMGateway(
        providers=[dead, backup],
        batcher=Batcher(max_batch_size=16, max_wait_ms=50),
        retry=fast_retry(max_attempts=1),
    )
    async with gw:
        responses = await gw.submit_many([LLMRequest(f"p{i}") for i in range(16)])
    assert all(r.provider == "backup" for r in responses)
