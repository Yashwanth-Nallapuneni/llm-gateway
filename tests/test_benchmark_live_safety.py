"""Offline tests for the live-benchmark safety fixes in benchmarks/bench.py:
cross-run cooldown, the worst-case --max-live-calls ceiling, and excluding
truncated arms from the reported summary. No network access -- these never
touch GROQ_API_KEY or api.groq.com; see tests/test_live_openrouter.py for
the actual live-call tests.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

BENCH_DIR = Path(__file__).resolve().parent.parent / "benchmarks"
sys.path.insert(0, str(BENCH_DIR))

import bench  # noqa: E402


def make_run(*, invalid: bool = False, successes: int = 5) -> bench.RunResult:
    return bench.RunResult(
        label="naive",
        wall_s=1.0,
        requests_sent=successes,
        status_429=0,
        status_5xx=0,
        successes=successes,
        failures=0,
        latencies_s=[0.1] * successes,
        cost_usd=0.0,
        invalid=invalid,
        invalid_reason="hit --max-live-calls mid-arm" if invalid else "",
    )


def test_is_budget_exceeded_direct() -> None:
    assert bench._is_budget_exceeded(bench.LiveCallBudgetExceeded("no budget left"))
    assert not bench._is_budget_exceeded(ValueError("something else"))


def test_is_budget_exceeded_walks_cause_and_context() -> None:
    original = bench.LiveCallBudgetExceeded("no budget left")
    try:
        try:
            raise original
        except bench.LiveCallBudgetExceeded as exc:
            raise RuntimeError("wrapped by a retry policy") from exc
    except RuntimeError as wrapped:
        assert bench._is_budget_exceeded(wrapped)

    # a wrapped, unrelated exception must not be misreported as a budget cutoff
    try:
        try:
            raise ValueError("boom")
        except ValueError as exc:
            raise RuntimeError("wrapped") from exc
    except RuntimeError as wrapped_other:
        assert not bench._is_budget_exceeded(wrapped_other)


def test_is_budget_exceeded_does_not_loop_on_self_referential_chain() -> None:
    exc = ValueError("cycle")
    exc.__cause__ = exc  # pathological, but must not hang
    assert not bench._is_budget_exceeded(exc)


def test_valid_runs_passes_through_when_nothing_invalid(capsys: pytest.CaptureFixture[str]) -> None:
    runs = [make_run(), make_run()]
    ok = bench.valid_runs(runs, "naive")
    assert ok == runs
    assert "WARNING" not in capsys.readouterr().out


def test_valid_runs_drops_truncated_arms_and_warns(capsys: pytest.CaptureFixture[str]) -> None:
    runs = [make_run(), make_run(invalid=True)]
    ok = bench.valid_runs(runs, "naive")
    assert ok == [runs[0]]
    assert "WARNING" in capsys.readouterr().out


def test_valid_runs_raises_when_every_run_is_invalid() -> None:
    runs = [make_run(invalid=True), make_run(invalid=True)]
    with pytest.raises(SystemExit):
        bench.valid_runs(runs, "gateway")


def test_summarize_of_truncated_runs_never_reaches_reporting() -> None:
    """The seam the retracted 57.5% figure fell through: a truncated run's
    numbers must be excluded before summarize() ever sees them, not merely
    flagged after the fact."""
    runs = [make_run(successes=5), make_run(invalid=True, successes=0)]
    ok = bench.valid_runs(runs, "naive")
    summary = bench.summarize(ok, n_prompts=5)
    assert summary["success_rate"]["median"] == 1.0


@pytest.mark.asyncio
async def test_cooldown_fires_at_every_arm_boundary_including_across_runs(monkeypatch: pytest.MonkeyPatch) -> None:
    """The actual confound fix: with --arm-cooldown set, a sleep must happen
    before every arm -- 2 per run (start-of-run boundary + mid-run boundary)
    -- not just once between the two arms inside each run."""
    sleeps: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    monkeypatch.setattr(bench.asyncio, "sleep", fake_sleep)

    args = bench.argparse.Namespace(
        n_prompts=3,
        runs=3,
        seed=1,
        concurrency=4,
        server_rpm=10_000.0,
        server_latency_mean_ms=0.1,
        server_latency_sigma=0.3,
        server_fail_rate=0.0,
        gateway_tpm=10_000_000.0,
        batch_size=16,
        batch_wait_ms=1.0,
        batch_tokens=8000,
        max_attempts=2,
        base_delay=0.001,
        max_delay=0.01,
        max_tokens=16,
        cost_per_1k_input=0.02,
        cost_per_1k_output=0.04,
        arm_cooldown=5.0,
        arm_order="alternate",
    )
    await bench.run_all_sim(args)

    # SimClient also calls asyncio.sleep, for simulated per-call latency
    # (sub-millisecond here), so filter to the cooldown's own 5.0s sleeps.
    cooldowns = [s for s in sleeps if s == 5.0]

    # one cooldown before each arm: 2 arms * 3 runs = 6 boundaries total,
    # including the one before run 0's first arm and the ones between runs.
    assert len(cooldowns) == 6


def test_live_refuses_to_start_below_worst_case_ceiling(monkeypatch: pytest.MonkeyPatch) -> None:
    """A ceiling sized to the best case (no retries) is exactly what let an
    arm get truncated mid-run in the retracted attempt. run_all_live must
    refuse to start rather than risk it -- and must refuse before touching
    the network, so this needs no real key."""
    monkeypatch.setenv("GROQ_API_KEY", "not-a-real-key")

    args = bench.argparse.Namespace(
        n_prompts=40,
        runs=3,
        max_attempts=4,
        # best case (no retries) = 40*3*2 = 240, worst case = 240*4 = 960
        max_live_calls=300,
    )
    with pytest.raises(SystemExit) as exc_info:
        import asyncio

        asyncio.run(bench.run_all_live(args))
    assert "worst case" in str(exc_info.value)


def test_live_missing_api_key_refuses_to_start(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    args = bench.argparse.Namespace(n_prompts=1, runs=1, max_attempts=1, max_live_calls=100)
    with pytest.raises(SystemExit) as exc_info:
        import asyncio

        asyncio.run(bench.run_all_live(args))
    assert "GROQ_API_KEY" in str(exc_info.value)
