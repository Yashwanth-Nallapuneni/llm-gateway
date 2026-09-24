"""Budget wiring: BudgetLedger actually enforcing a ceiling on LLMGateway.

Everything here runs offline against MockProvider/MockClient, same as
test_integration.py. `fast_retry()` mirrors the helper in that file so a
retry loop (where exercised) does not slow the suite down with real sleeps.
"""

from __future__ import annotations

import asyncio

import pytest

from llm_gateway import (
    Batcher,
    BudgetExceeded,
    BudgetLedger,
    LLMGateway,
    LLMRequest,
    RetryPolicy,
    RunStore,
)
from llm_gateway.providers.mock import MockClient, MockProvider


def fast_retry(**kw) -> RetryPolicy:
    kw.setdefault("base_delay", 0.001)
    kw.setdefault("max_delay", 0.01)
    return RetryPolicy(**kw)


# --------------------------------------------------------------------------
# A generous budget changes nothing observable except the ledger filling up
# --------------------------------------------------------------------------


async def test_generous_budget_completes_normally_and_ledger_matches_cost():
    budget = BudgetLedger(1000.0)
    provider = MockProvider("a")
    gw = LLMGateway(providers=[provider], retry=fast_retry(), budget=budget)

    async with gw:
        responses = await gw.submit_many(
            [LLMRequest(f"p{i}", max_tokens=16) for i in range(10)]
        )

    assert len(responses) == 10
    assert budget.spent > 0
    assert budget.outstanding_count == 0
    # The ledger settles with exactly the cost the metrics sink recorded for
    # every successful call -- both are computed the same way, from the same
    # actual input/output token counts, just recorded into two different
    # places.
    metrics_cost = sum(m.cost for m in gw.metrics._providers.values())
    assert budget.spent == pytest.approx(metrics_cost, rel=1e-9)


# --------------------------------------------------------------------------
# A tight budget stops the run partway through
# --------------------------------------------------------------------------


async def test_tight_budget_stops_the_run_without_exceeding_the_limit():
    provider = MockProvider("a", cost_per_1k_input=0.0, cost_per_1k_output=1.0)
    # A single request's worst-case reservation, at max_tokens=100:
    # 100/1000 * 1.0 == 0.10.
    price = provider.estimated_cost(0, 100)
    n = 10
    # Room for three reservations, not four -- some requests must succeed,
    # the rest must be turned away.
    budget = BudgetLedger(price * 3.5)
    gw = LLMGateway(
        providers=[provider],
        batcher=Batcher(max_batch_size=1, max_wait_ms=1),
        retry=fast_retry(),
        budget=budget,
    )

    async with gw:
        results = await asyncio.gather(
            *(gw.submit(LLMRequest(f"p{i}", max_tokens=100)) for i in range(n)),
            return_exceptions=True,
        )

    successes = [r for r in results if not isinstance(r, BaseException)]
    failures = [r for r in results if isinstance(r, BaseException)]
    assert len(successes) + len(failures) == n
    assert successes, "at least one request should have gone through"
    assert failures, "at least one request should have been turned away"
    assert all(isinstance(exc, BudgetExceeded) for exc in failures)
    # The whole point of reserve-before-dispatch: committed spend never once
    # crosses the ceiling, even though these requests were all admitted
    # concurrently.
    assert budget.committed <= budget.limit_usd + 1e-9


# --------------------------------------------------------------------------
# BudgetExceeded is terminal: no retry, no failover
# --------------------------------------------------------------------------


async def test_budget_exceeded_is_not_retried_or_failed_over():
    primary_client = MockClient("primary")
    backup_client = MockClient("backup")
    primary = MockProvider("primary", client=primary_client, cost_per_1k_output=1.0)
    backup = MockProvider("backup", client=backup_client, cost_per_1k_output=1.0)

    # Smaller than either provider's worst-case reservation for this
    # request, so `reserve()` raises for both candidates before a call is
    # ever attempted.
    budget = BudgetLedger(1e-6)
    gw = LLMGateway(
        providers=[primary, backup],
        retry=fast_retry(max_attempts=5),
        budget=budget,
    )

    async with gw:
        with pytest.raises(BudgetExceeded):
            await gw.submit(LLMRequest("hello", max_tokens=100))

    assert primary_client.calls == 0
    assert backup_client.calls == 0
    assert gw.metrics._p("primary").attempted == 0
    assert gw.metrics._p("backup").attempted == 0
    assert gw.metrics.rejected == 1
    assert budget.outstanding_count == 0


# --------------------------------------------------------------------------
# Cache hits from RunStore cost nothing
# --------------------------------------------------------------------------


async def test_cache_hit_consumes_no_budget(tmp_path):
    store = RunStore(str(tmp_path / "run.db"))
    budget = BudgetLedger(10.0)
    gw = LLMGateway(
        providers=[MockProvider("a")],
        retry=fast_retry(),
        store=store,
        budget=budget,
    )

    try:
        async with gw:
            request = LLMRequest("same prompt every time", max_tokens=16)
            first = await gw.submit(request)
            spent_after_first = budget.spent
            assert spent_after_first > 0

            second = await gw.submit(request)
    finally:
        await store.aclose()

    assert second.text == first.text
    assert gw.served_from_store == 1
    assert gw.freshly_called == 1
    # The second call was served entirely from the store -- no reservation,
    # no settlement, nothing added to the ledger.
    assert budget.spent == pytest.approx(spent_after_first)
    assert budget.outstanding_count == 0


# --------------------------------------------------------------------------
# budget=None changes nothing
# --------------------------------------------------------------------------


async def test_budget_none_behaves_exactly_as_before():
    gw = LLMGateway(providers=[MockProvider("a")], retry=fast_retry())
    assert gw.budget is None

    async with gw:
        responses = await gw.submit_many([LLMRequest(f"p{i}") for i in range(5)])

    assert len(responses) == 5
    assert all(r.provider == "a" for r in responses)


# --------------------------------------------------------------------------
# Settling with actual usage frees the reservation's unused slack
# --------------------------------------------------------------------------


async def test_settle_with_actual_usage_frees_reservation_slack():
    # MockClient always caps output at 32 tokens regardless of max_tokens,
    # so asking for a much larger max_tokens reserves far more than the
    # call actually costs.
    provider = MockProvider("a", cost_per_1k_input=0.0, cost_per_1k_output=1.0)
    worst_case = provider.estimated_cost(0, 1000)  # == 1.0
    actual = provider.estimated_cost(0, 32)  # == 0.032, what the call really costs

    # Room for one worst-case reservation plus a bit of slack -- not two.
    # Two full worst-case reservations held at once would not fit; the
    # second request only succeeds because the first settled down to its
    # real, much smaller cost first.
    limit = worst_case + actual + 0.001
    assert limit < 2 * worst_case

    budget = BudgetLedger(limit)
    gw = LLMGateway(
        providers=[provider],
        batcher=Batcher(max_batch_size=1, max_wait_ms=1),
        retry=fast_retry(),
        budget=budget,
    )

    async with gw:
        # Sequential on purpose: the second reservation must be made after
        # the first has already settled, not concurrently with it.
        await gw.submit(LLMRequest("first", max_tokens=1000))
        committed_after_settle = budget.committed
        assert committed_after_settle == pytest.approx(actual, rel=1e-9)
        assert committed_after_settle < worst_case

        # This would raise BudgetExceeded if the first reservation's full
        # worst-case amount were still held instead of having been settled
        # down to its actual cost.
        await gw.submit(LLMRequest("second", max_tokens=1000))

    assert budget.spent == pytest.approx(2 * actual, rel=1e-9)
    assert budget.outstanding_count == 0
