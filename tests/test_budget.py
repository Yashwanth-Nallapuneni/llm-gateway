from __future__ import annotations

import asyncio

import pytest

from llm_gateway.budget import (
    BudgetExceeded,
    BudgetLedger,
    TokenBudgetExceeded,
)


def test_construction_rejects_zero_or_negative_limit():
    with pytest.raises(ValueError):
        BudgetLedger(0)
    with pytest.raises(ValueError):
        BudgetLedger(-1.0)


def test_construction_rejects_zero_or_negative_token_limit():
    with pytest.raises(ValueError):
        BudgetLedger(5.0, max_tokens_total=0)
    with pytest.raises(ValueError):
        BudgetLedger(5.0, max_tokens_total=-10)


def test_reserve_blocks_once_limit_would_be_exceeded():
    ledger = BudgetLedger(5.0)
    ledger.reserve(3.0)
    ledger.reserve(2.0)
    assert ledger.committed == pytest.approx(5.0)

    with pytest.raises(BudgetExceeded):
        ledger.reserve(0.01)


def test_reserve_exactly_at_limit_succeeds():
    ledger = BudgetLedger(5.0)
    r = ledger.reserve(5.0)
    assert ledger.remaining == pytest.approx(0.0)
    r.release()


def test_settling_below_reservation_frees_the_difference():
    ledger = BudgetLedger(5.0)
    r = ledger.reserve(3.0)
    r.settle(1.0)
    assert ledger.spent == pytest.approx(1.0)
    assert ledger.committed == pytest.approx(1.0)
    # The 2.0 difference between the 3.0 reservation and the 1.0 actual
    # cost is available again for a later request.
    r2 = ledger.reserve(4.0)
    assert ledger.committed == pytest.approx(5.0)
    r2.settle(4.0)


def test_settling_above_reservation_is_recorded_honestly():
    ledger = BudgetLedger(5.0)
    r = ledger.reserve(1.0)
    r.settle(2.0)  # provider billed more than max_tokens implied
    assert ledger.spent == pytest.approx(2.0)
    assert ledger.committed == pytest.approx(2.0)
    assert ledger.remaining == pytest.approx(3.0)


def test_settling_above_reservation_can_still_exceed_effective_limit():
    # settle() is allowed to push actual spend past the configured limit --
    # the ceiling only prevents *new* reservations from being admitted, it
    # cannot retroactively cap a bill a provider already sent.
    ledger = BudgetLedger(5.0)
    r = ledger.reserve(1.0)
    r.settle(6.0)
    assert ledger.spent == pytest.approx(6.0)
    assert ledger.remaining == pytest.approx(0.0)
    with pytest.raises(BudgetExceeded):
        ledger.reserve(0.01)


def test_release_restores_the_full_amount():
    ledger = BudgetLedger(5.0)
    r = ledger.reserve(5.0)
    with pytest.raises(BudgetExceeded):
        ledger.reserve(0.01)
    r.release()
    assert ledger.committed == pytest.approx(0.0)
    assert ledger.remaining == pytest.approx(5.0)
    # budget is fully usable again
    r2 = ledger.reserve(5.0)
    r2.release()


def test_double_settle_raises():
    ledger = BudgetLedger(5.0)
    r = ledger.reserve(1.0)
    r.settle(1.0)
    with pytest.raises(RuntimeError):
        r.settle(1.0)


def test_double_release_raises():
    ledger = BudgetLedger(5.0)
    r = ledger.reserve(1.0)
    r.release()
    with pytest.raises(RuntimeError):
        r.release()


def test_settle_after_release_raises():
    ledger = BudgetLedger(5.0)
    r = ledger.reserve(1.0)
    r.release()
    with pytest.raises(RuntimeError):
        r.settle(1.0)


def test_context_manager_releases_on_exception():
    ledger = BudgetLedger(5.0)
    with pytest.raises(ValueError), ledger.reserve(2.0) as r:
        assert r.usd == pytest.approx(2.0)
        raise ValueError("boom")
    assert ledger.committed == pytest.approx(0.0)


def test_context_manager_settled_inside_block_is_left_alone():
    ledger = BudgetLedger(5.0)
    with ledger.reserve(2.0) as r:
        r.settle(1.5)
    assert ledger.spent == pytest.approx(1.5)
    assert ledger.committed == pytest.approx(1.5)


def test_context_manager_without_settle_or_release_raises():
    ledger = BudgetLedger(5.0)
    with pytest.raises(RuntimeError), ledger.reserve(2.0):
        pass


def test_exception_message_names_limit_committed_requested():
    ledger = BudgetLedger(5.0)
    ledger.reserve(4.0)
    with pytest.raises(BudgetExceeded) as excinfo:
        ledger.reserve(2.0)
    err = excinfo.value
    assert err.limit == pytest.approx(5.0)
    assert err.committed == pytest.approx(4.0)
    assert err.requested == pytest.approx(2.0)
    message = str(err)
    assert "5" in message
    assert "4" in message
    assert "2" in message


def test_outstanding_count_tracks_open_reservations():
    ledger = BudgetLedger(5.0)
    assert ledger.outstanding_count == 0
    r1 = ledger.reserve(1.0)
    r2 = ledger.reserve(1.0)
    assert ledger.outstanding_count == 2
    r1.settle(1.0)
    assert ledger.outstanding_count == 1
    r2.release()
    assert ledger.outstanding_count == 0


# --------------------------------------------------------------------------
# token ceiling
# --------------------------------------------------------------------------


def test_token_ceiling_blocks_once_exceeded():
    ledger = BudgetLedger(100.0, max_tokens_total=1000)
    ledger.reserve(1.0, estimated_tokens=600)
    ledger.reserve(1.0, estimated_tokens=400)
    assert ledger.committed_tokens == 1000
    with pytest.raises(TokenBudgetExceeded):
        ledger.reserve(1.0, estimated_tokens=1)


def test_token_ceiling_dollar_check_runs_first():
    # A reservation that fails both ceilings should report the dollar one,
    # since that is the ceiling this module primarily exists for.
    ledger = BudgetLedger(1.0, max_tokens_total=10)
    with pytest.raises(BudgetExceeded):
        ledger.reserve(2.0, estimated_tokens=100)


def test_no_token_ceiling_by_default():
    ledger = BudgetLedger(100.0)
    assert ledger.remaining_tokens is None
    r = ledger.reserve(1.0, estimated_tokens=10_000_000)
    r.settle(1.0)


def test_settle_can_override_actual_tokens():
    ledger = BudgetLedger(100.0, max_tokens_total=1000)
    r = ledger.reserve(1.0, estimated_tokens=500)
    r.settle(1.0, actual_tokens=800)
    assert ledger.tokens_spent == 800
    assert ledger.committed_tokens == 800


# --------------------------------------------------------------------------
# concurrency: the whole point of reserve/settle
# --------------------------------------------------------------------------


async def test_concurrent_reservations_never_collectively_exceed_limit():
    limit = 10.0
    cost = 1.0
    ledger = BudgetLedger(limit)

    successes: list[object] = []
    failures: list[BudgetExceeded] = []

    async def attempt() -> None:
        # Yield to the loop before reserving so all 50 coroutines are truly
        # racing to reserve rather than running strictly in submission order.
        await asyncio.sleep(0)
        try:
            r = ledger.reserve(cost)
            successes.append(r)
        except BudgetExceeded as exc:
            failures.append(exc)

    await asyncio.gather(*(attempt() for _ in range(50)))

    # Exactly limit/cost reservations should have been admitted -- not more.
    assert len(successes) == 10
    assert len(failures) == 40
    assert ledger.committed == pytest.approx(10.0)
    assert ledger.remaining == pytest.approx(0.0)

    for r in successes:
        r.settle(cost)  # type: ignore[attr-defined]
    assert ledger.spent == pytest.approx(10.0)
    assert ledger.committed == pytest.approx(10.0)


async def test_concurrent_reserve_settle_release_mix_stays_within_limit():
    limit = 5.0
    ledger = BudgetLedger(limit)
    max_committed_observed = 0.0

    async def worker(i: int) -> None:
        nonlocal max_committed_observed
        await asyncio.sleep(0)
        try:
            r = ledger.reserve(0.3)
        except BudgetExceeded:
            return
        max_committed_observed = max(max_committed_observed, ledger.committed)
        await asyncio.sleep(0)
        if i % 3 == 0:
            r.release()
        else:
            r.settle(0.3 if i % 2 == 0 else 0.1)

    await asyncio.gather(*(worker(i) for i in range(100)))

    assert max_committed_observed <= limit + 1e-9
    assert ledger.remaining >= -1e-9
    assert ledger.outstanding_count == 0


# --------------------------------------------------------------------------
# float drift
# --------------------------------------------------------------------------


def test_many_small_settlements_do_not_drift_remaining_negative():
    ledger = BudgetLedger(1.0)
    n = 100_000
    per = 1.0 / n
    for _ in range(n):
        r = ledger.reserve(per)
        r.settle(per)
    assert ledger.remaining >= -1e-9
    assert ledger.spent <= 1.0 + 1e-6


def test_many_reserve_release_cycles_do_not_drift_committed_negative():
    ledger = BudgetLedger(1.0)
    for _ in range(50_000):
        r = ledger.reserve(0.3333333333)
        r.release()
    assert ledger.committed >= -1e-9
    assert ledger.committed == pytest.approx(0.0, abs=1e-6)
    # Budget is still fully usable -- no drift ate into it.
    r = ledger.reserve(1.0)
    r.release()
