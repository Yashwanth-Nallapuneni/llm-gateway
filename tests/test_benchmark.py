"""Smoke test for benchmarks/bench.py.

Not a benchmark itself -- just: does a tiny run complete quickly and produce
the result structure the reporting code expects? benchmarks/ is not part of
the installed package, so it's imported the same way bench.py imports
llm_gateway: by inserting its directory onto sys.path.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest

BENCH_DIR = Path(__file__).resolve().parent.parent / "benchmarks"
sys.path.insert(0, str(BENCH_DIR))

import bench  # noqa: E402


def fast_args(**overrides: object) -> bench.argparse.Namespace:
    defaults = dict(
        n_prompts=6,
        runs=1,
        seed=123,
        concurrency=8,
        server_rpm=10_000.0,
        server_latency_mean_ms=0.5,
        server_latency_sigma=0.3,
        server_fail_rate=0.0,
        gateway_tpm=10_000_000.0,
        batch_size=16,
        batch_wait_ms=5.0,
        batch_tokens=8000,
        max_attempts=3,
        base_delay=0.001,
        max_delay=0.01,
        max_tokens=32,
        cost_per_1k_input=0.02,
        cost_per_1k_output=0.04,
        json_path=None,
        markdown=False,
    )
    defaults.update(overrides)
    return bench.argparse.Namespace(**defaults)


@pytest.mark.asyncio
async def test_run_all_produces_expected_structure() -> None:
    args = fast_args()
    result = await bench.run_all(args)

    assert result["n_prompts"] == 6
    assert len(result["naive"]) == 1
    assert len(result["gateway"]) == 1

    for arm in ("naive", "gateway"):
        run = result[arm][0]
        assert isinstance(run, bench.RunResult)
        assert run.successes + run.failures == 6
        assert run.wall_s >= 0.0
        assert len(run.latencies_s) == run.successes

    # naive never batches, so each success took at least one call; gateway
    # may batch several prompts into one call, so it can (and here, with a
    # generous batch size, should) send fewer calls than prompts.
    naive_run = result["naive"][0]
    gateway_run = result["gateway"][0]
    assert naive_run.requests_sent >= naive_run.successes
    assert gateway_run.requests_sent >= 1
    assert gateway_run.requests_sent <= gateway_run.successes

    naive_summary = bench.summarize(result["naive"], result["n_prompts"])
    gateway_summary = bench.summarize(result["gateway"], result["n_prompts"])

    for summary in (naive_summary, gateway_summary):
        for key, _, _ in bench.METRIC_ORDER:
            assert key in summary
            for stat in ("median", "p5", "p95"):
                assert stat in summary[key]

    # With no injected failures and a server well above the request rate,
    # both arms should complete every prompt successfully.
    assert naive_summary["success_rate"]["median"] == 1.0
    assert gateway_summary["success_rate"]["median"] == 1.0

    # Rendering must not blow up on a real result.
    table = bench.render_table(naive_summary, gateway_summary, 6, 1)
    assert "wall-clock" in table
    md = bench.render_markdown(naive_summary, gateway_summary, 6, 1)
    assert "| metric | naive | gateway |" in md


def test_percentile_matches_metrics_module_semantics() -> None:
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert bench._percentile([], 50) == 0.0
    assert bench._percentile(values, 50) == 3.0
    assert bench._percentile(values, 100) == 5.0


def test_make_prompts_is_deterministic() -> None:
    a = bench.make_prompts(10, seed=7, max_tokens=64)
    b = bench.make_prompts(10, seed=7, max_tokens=64)
    assert [p.prompt for p in a] == [p.prompt for p in b]


@pytest.mark.asyncio
async def test_same_seed_reproduces_counts_not_just_prompts() -> None:
    """The documented reproducibility guarantee: deterministic counts
    (requests sent, 429s, successes/failures, cost) must match exactly
    across two runs with the same seed. Wall-clock and per-call latency
    depend on real scheduling and are explicitly excluded -- see
    benchmarks/README.md's "Threats to validity"."""
    args = fast_args(n_prompts=8, runs=1, seed=55)
    r1 = await bench.run_all(args)
    r2 = await bench.run_all(args)

    for arm in ("naive", "gateway"):
        run1, run2 = r1[arm][0], r2[arm][0]
        assert run1.requests_sent == run2.requests_sent
        assert run1.status_429 == run2.status_429
        assert run1.status_5xx == run2.status_5xx
        assert run1.successes == run2.successes
        assert run1.failures == run2.failures
        assert run1.cost_usd == run2.cost_usd


def test_sim_client_latency_is_seeded_and_reproducible() -> None:
    rng_a = random.Random(1)
    rng_b = random.Random(1)
    client_a = bench.SimClient(
        "a",
        rpm_limit=1000,
        latency_rng=rng_a,
        latency_mean_s=0.01,
        latency_sigma=0.4,
        fail_rng=random.Random(2),
        transient_fail_rate=0.0,
    )
    client_b = bench.SimClient(
        "b",
        rpm_limit=1000,
        latency_rng=rng_b,
        latency_mean_s=0.01,
        latency_sigma=0.4,
        fail_rng=random.Random(2),
        transient_fail_rate=0.0,
    )
    samples_a = [client_a._sample_latency() for _ in range(20)]
    samples_b = [client_b._sample_latency() for _ in range(20)]
    assert samples_a == samples_b
