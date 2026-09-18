"""What does llm-gateway actually buy you over a naive async loop?

Runs two clients against equivalent instances of a simulated rate-limited,
occasionally-flaky server and compares them:

  naive   -- asyncio.gather over all prompts through a bounded semaphore,
             with a proper retry-with-exponential-backoff-and-jitter on
             429/5xx. No rate limiting: that's the variable under test, not
             a strawman -- the retry logic is the same RetryPolicy the
             gateway itself uses.
  gateway -- llm_gateway.LLMGateway, configured with a token bucket set to
             the server's real limits, batching, and the same retry policy.

One command reproduces everything:

    python3 benchmarks/bench.py --n-prompts 200 --runs 5 --seed 42 --markdown

See benchmarks/README.md for methodology, the stated hypothesis, and
"Threats to validity" -- read that before trusting any number this prints.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import math
import os
import platform
import random
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Not installed as a package in this checkout -- match examples/bulk_eval.py.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from llm_gateway import Batcher, LLMGateway, LLMRequest, RetryPolicy
from llm_gateway.providers.mock import MockClient, MockProvider
from llm_gateway.types import LLMResponse, ProviderError

REPO_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# The simulated server
# ---------------------------------------------------------------------------


@dataclass
class ServerCallMetrics:
    """Counted at the call boundary, so naive (unbatched) and gateway
    (batched) calls are compared on the same footing: one entry per call
    actually made to the server, regardless of how many prompts it carries."""

    calls: int = 0
    status_429: int = 0
    status_5xx: int = 0
    other_fail: int = 0


class SimClient(MockClient):
    """A MockClient extended with seeded log-normal latency and a seeded
    rate of transient 503s.

    Subclasses MockClient rather than editing it: RPM-window enforcement
    (429 + Retry-After) and "a batch fails as a unit" semantics are
    inherited unchanged from mock.py. This only adds two things mock.py
    does not have -- per-call latency sampled from a distribution instead
    of a fixed float, and probabilistic 503 injection -- both driven by
    dedicated seeded `random.Random` instances so a run is exactly
    reproducible.
    """

    def __init__(
        self,
        name: str,
        *,
        rpm_limit: int,
        latency_rng: random.Random,
        latency_mean_s: float,
        latency_sigma: float,
        fail_rng: random.Random,
        transient_fail_rate: float,
    ) -> None:
        # latency=0.0: disable MockClient's own fixed-latency sleep, since
        # we sample our own latency below instead.
        super().__init__(name=name, latency=0.0, rpm_limit=rpm_limit)
        self._latency_rng = latency_rng
        self._fail_rng = fail_rng
        self._transient_fail_rate = transient_fail_rate
        # lognormvariate(mu, sigma) has mean exp(mu + sigma^2/2); solve mu
        # so `latency_mean_s` is the distribution's actual mean, not its
        # median -- a log-normal's median is noticeably below its mean, and
        # a caller who passes "mean latency 30ms" wants 30ms on average.
        self._mu = math.log(latency_mean_s) - (latency_sigma**2) / 2.0
        self._sigma = latency_sigma
        self.server_metrics = ServerCallMetrics()

    def _sample_latency(self) -> float:
        return max(0.0, self._latency_rng.lognormvariate(self._mu, self._sigma))

    def _next_failure(self) -> int | None:
        # MockClient's own failure policy (fail_sequence / fail_status /
        # fail_first_n) takes precedence if configured; we don't use those
        # here, but preserving the chain keeps this class safe to reuse
        # with them. Beyond that, inject a transient 503 at the configured
        # rate -- a fresh draw per call, not "fail every Nth call", so runs
        # look like real background flakiness rather than a fixed pattern.
        base = super()._next_failure()
        if base is not None:
            return base
        if self._fail_rng.random() < self._transient_fail_rate:
            return 503
        return None

    async def complete(self, request: LLMRequest) -> LLMResponse:
        self.server_metrics.calls += 1
        await asyncio.sleep(self._sample_latency())
        try:
            return await super().complete(request)
        except ProviderError as exc:
            self._tally(exc)
            raise

    async def complete_batch(self, requests: list[LLMRequest]) -> list[LLMResponse]:
        self.server_metrics.calls += 1
        await asyncio.sleep(self._sample_latency())
        try:
            return await super().complete_batch(requests)
        except ProviderError as exc:
            self._tally(exc)
            raise

    def _tally(self, exc: ProviderError) -> None:
        m = self.server_metrics
        if exc.status == 429:
            m.status_429 += 1
        elif exc.status is not None and exc.status >= 500:
            m.status_5xx += 1
        else:
            m.other_fail += 1


# ---------------------------------------------------------------------------
# Workload
# ---------------------------------------------------------------------------


def make_prompts(n: int, seed: int, max_tokens: int) -> list[LLMRequest]:
    rng = random.Random(seed)
    return [
        LLMRequest(
            prompt=f"Summarize item #{i}: " + "word " * rng.randint(5, 40),
            max_tokens=max_tokens,
        )
        for i in range(n)
    ]


def _cost(resp: LLMResponse, cost_per_1k_input: float, cost_per_1k_output: float) -> float:
    return (
        resp.input_tokens / 1000.0 * cost_per_1k_input
        + resp.output_tokens / 1000.0 * cost_per_1k_output
    )


# ---------------------------------------------------------------------------
# Per-run result
# ---------------------------------------------------------------------------


@dataclass
class RunResult:
    label: str
    wall_s: float
    requests_sent: int
    status_429: int
    status_5xx: int
    successes: int
    failures: int
    latencies_s: list[float] = field(default_factory=list)
    cost_usd: float = 0.0


# ---------------------------------------------------------------------------
# naive arm
# ---------------------------------------------------------------------------


async def run_naive(
    prompts: list[LLMRequest],
    server: SimClient,
    *,
    concurrency: int,
    retry: RetryPolicy,
    cost_per_1k_input: float,
    cost_per_1k_output: float,
) -> RunResult:
    sem = asyncio.Semaphore(concurrency)
    latencies: list[float] = []
    failures = 0
    cost = 0.0
    clock = time.monotonic

    async def one(req: LLMRequest) -> None:
        nonlocal failures, cost
        start = clock()
        attempt = 0
        while True:
            try:
                # The semaphore is held only around the network call, not
                # across the backoff sleep below -- the same discipline
                # gateway.py uses, so a pile of retrying requests can't
                # starve healthy ones out of every concurrency slot.
                async with sem:
                    resp = await server.complete(req)
            except Exception as exc:
                if not retry.should_retry(exc, attempt):
                    failures += 1
                    return
                delay = retry.delay_for(attempt, retry.retry_after_from(exc))
                attempt += 1
                await asyncio.sleep(delay)
                continue
            latencies.append(clock() - start)
            cost += _cost(resp, cost_per_1k_input, cost_per_1k_output)
            return

    t0 = clock()
    await asyncio.gather(*(one(r) for r in prompts))
    wall = clock() - t0

    m = server.server_metrics
    return RunResult(
        label="naive",
        wall_s=wall,
        requests_sent=m.calls,
        status_429=m.status_429,
        status_5xx=m.status_5xx,
        successes=len(latencies),
        failures=failures,
        latencies_s=latencies,
        cost_usd=cost,
    )


# ---------------------------------------------------------------------------
# gateway arm
# ---------------------------------------------------------------------------


async def run_gateway(
    prompts: list[LLMRequest],
    server: SimClient,
    *,
    rpm_limit: float,
    tpm_limit: float,
    batch_size: int,
    batch_wait_ms: float,
    batch_tokens: int,
    retry: RetryPolicy,
    cost_per_1k_input: float,
    cost_per_1k_output: float,
    batch_endpoint: bool = False,
) -> RunResult:
    # Default False on purpose. Neither Groq nor OpenRouter -- the two
    # providers this library actually ships adapters for -- has a synchronous
    # multi-prompt endpoint, so counting a 16-prompt batch as ONE request
    # would credit the gateway with a saving no real deployment can collect.
    # With it False the gateway still groups requests for rate-limit
    # accounting and dispatches them concurrently, which is what really
    # happens against a chat API. --batch-endpoint models the other case:
    # a provider that genuinely accepts many prompts per call.
    provider = MockProvider(
        "sim",
        client=server,
        supports_batching=batch_endpoint,
        rpm_limit=rpm_limit,
        tpm_limit=tpm_limit,
        cost_per_1k_input=cost_per_1k_input,
        cost_per_1k_output=cost_per_1k_output,
        max_concurrency=64,
    )
    gw = LLMGateway(
        providers=[provider],
        batcher=Batcher(
            max_batch_size=batch_size,
            max_wait_ms=batch_wait_ms,
            max_batch_tokens=batch_tokens,
        ),
        retry=retry,
    )

    clock = time.monotonic
    t0 = clock()
    async with gw:
        results = await asyncio.gather(
            *(gw.submit(r) for r in prompts), return_exceptions=True
        )
    wall = clock() - t0

    latencies = [r.latency_s for r in results if isinstance(r, LLMResponse)]
    failures = sum(1 for r in results if not isinstance(r, LLMResponse))
    cost = sum(
        _cost(r, cost_per_1k_input, cost_per_1k_output)
        for r in results
        if isinstance(r, LLMResponse)
    )

    m = server.server_metrics
    return RunResult(
        label="gateway",
        wall_s=wall,
        requests_sent=m.calls,
        status_429=m.status_429,
        status_5xx=m.status_5xx,
        successes=len(latencies),
        failures=failures,
        latencies_s=latencies,
        cost_usd=cost,
    )


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------


def _percentile(values: list[float], p: float) -> float:
    """Nearest-rank percentile. Same method as llm_gateway.metrics -- never
    interpolates a value nothing actually produced. Reimplemented locally
    (rather than importing the private `_percentile` from metrics.py) since
    it's a five-line function and importing a leading-underscore name across
    modules is worse than duplicating it."""
    if not values:
        return 0.0
    ordered = sorted(values)
    k = max(0, min(len(ordered) - 1, round(p / 100.0 * len(ordered) + 0.5) - 1))
    return ordered[k]


def run_scalars(r: RunResult, n_prompts: int) -> dict[str, float]:
    return {
        "wall_s": r.wall_s,
        "requests_sent": float(r.requests_sent),
        "status_429": float(r.status_429),
        "status_5xx": float(r.status_5xx),
        "successes": float(r.successes),
        "failures": float(r.failures),
        "success_rate": r.successes / n_prompts if n_prompts else 0.0,
        "p50_s": _percentile(r.latencies_s, 50),
        "p95_s": _percentile(r.latencies_s, 95),
        "p99_s": _percentile(r.latencies_s, 99),
        "cost_usd": r.cost_usd,
    }


METRIC_ORDER: list[tuple[str, str, str]] = [
    ("wall_s", "wall-clock (s)", "{:.3f}"),
    ("requests_sent", "requests sent", "{:.1f}"),
    ("status_429", "429s received", "{:.1f}"),
    ("status_5xx", "5xx received", "{:.1f}"),
    ("successes", "successes", "{:.1f}"),
    ("failures", "failures", "{:.1f}"),
    ("success_rate", "success rate", "{:.1%}"),
    ("p50_s", "latency p50 (s)", "{:.3f}"),
    ("p95_s", "latency p95 (s)", "{:.3f}"),
    ("p99_s", "latency p99 (s)", "{:.3f}"),
    ("cost_usd", "est. cost (USD)", "{:.4f}"),
]


def summarize(runs: list[RunResult], n_prompts: int) -> dict[str, dict[str, float]]:
    """Median and p5-p95 of each metric, across runs -- never a single run."""
    per_run = [run_scalars(r, n_prompts) for r in runs]
    out: dict[str, dict[str, float]] = {}
    for key, _, _ in METRIC_ORDER:
        values = [row[key] for row in per_run]
        out[key] = {
            "median": _percentile(values, 50),
            "p5": _percentile(values, 5),
            "p95": _percentile(values, 95),
        }
    return out


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------


async def run_all(args: argparse.Namespace) -> dict[str, Any]:
    prompts = make_prompts(args.n_prompts, args.seed, args.max_tokens)
    master = random.Random(args.seed)
    # One seed per run, drawn once from the master RNG -- reused for BOTH
    # arms so run i in the naive arm and run i in the gateway arm face
    # identically-seeded server latency/failure streams and identically
    # seeded retry jitter. See "Threats to validity" in the README for what
    # this reproducibility guarantee does and does not buy you.
    run_seeds = [master.randrange(2**31) for _ in range(args.runs)]

    naive_runs: list[RunResult] = []
    gateway_runs: list[RunResult] = []

    for rs in run_seeds:
        naive_server = SimClient(
            "sim-naive",
            rpm_limit=int(args.server_rpm),
            latency_rng=random.Random(rs),
            latency_mean_s=args.server_latency_mean_ms / 1000.0,
            latency_sigma=args.server_latency_sigma,
            fail_rng=random.Random(rs + 1),
            transient_fail_rate=args.server_fail_rate,
        )
        naive_retry = RetryPolicy(
            max_attempts=args.max_attempts,
            base_delay=args.base_delay,
            max_delay=args.max_delay,
            rng=random.Random(rs + 2),
        )
        naive_runs.append(
            await run_naive(
                prompts,
                naive_server,
                concurrency=args.concurrency,
                retry=naive_retry,
                cost_per_1k_input=args.cost_per_1k_input,
                cost_per_1k_output=args.cost_per_1k_output,
            )
        )

        gateway_server = SimClient(
            "sim-gateway",
            rpm_limit=int(args.server_rpm),
            latency_rng=random.Random(rs),
            latency_mean_s=args.server_latency_mean_ms / 1000.0,
            latency_sigma=args.server_latency_sigma,
            fail_rng=random.Random(rs + 1),
            transient_fail_rate=args.server_fail_rate,
        )
        gateway_retry = RetryPolicy(
            max_attempts=args.max_attempts,
            base_delay=args.base_delay,
            max_delay=args.max_delay,
            rng=random.Random(rs + 2),
        )
        gateway_runs.append(
            await run_gateway(
                prompts,
                gateway_server,
                rpm_limit=args.server_rpm,
                tpm_limit=args.gateway_tpm,
                batch_size=args.batch_size,
                batch_endpoint=getattr(args, "batch_endpoint", False),
                batch_wait_ms=args.batch_wait_ms,
                batch_tokens=args.batch_tokens,
                retry=gateway_retry,
                cost_per_1k_input=args.cost_per_1k_input,
                cost_per_1k_output=args.cost_per_1k_output,
            )
        )

    return {
        "n_prompts": len(prompts),
        "naive": naive_runs,
        "gateway": gateway_runs,
    }


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------

EXPECTATION = """\
Hypothesis, stated before the numbers below (so it's falsifiable, not
reverse-engineered from the outcome):
  - naive should show a meaningful count of 429s once load exceeds the
    server's configured RPM, and -- because its retry budget is finite
    while the server's RPM window is a real 60s sliding window -- some
    requests may exhaust their retries and fail outright rather than
    eventually succeed.
  - gateway should show 429s approaching zero (the token bucket paces
    requests under the limit before they're sent) and fewer total requests
    sent to the server, at the cost of a bit more added queueing latency.
If the numbers below don't show that pattern, that's a real result, not a
bug in this harness -- it will be reported as such, not tuned away.
"""


def build_header(args: argparse.Namespace) -> str:
    lines = ["=" * 78, "LLM-GATEWAY BENCHMARK: naive asyncio loop vs LLMGateway", "=" * 78]
    lines.append(f"timestamp (UTC):  {datetime.now(UTC).isoformat()}")
    lines.append(f"git commit:       {git_commit()}")
    lines.append(f"python:           {sys.version.split()[0]} ({platform.python_implementation()})")
    lines.append(f"platform:         {platform.platform()}")
    lines.append(f"processor:        {platform.processor() or 'unknown'}")
    lines.append(f"cpu_count:        {os.cpu_count()}")
    lines.append("-" * 78)
    lines.append("parameters:")
    for key, val in sorted(vars(args).items()):
        lines.append(f"  {key} = {val}")
    lines.append("=" * 78)
    return "\n".join(lines)


def git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return "unknown"


def _cell(summary: dict[str, dict[str, float]], key: str, fmt: str) -> str:
    s = summary[key]
    med = fmt.format(s["median"])
    p5 = fmt.format(s["p5"])
    p95 = fmt.format(s["p95"])
    return f"{med} [{p5}, {p95}]"


def render_table(
    naive: dict[str, dict[str, float]],
    gateway: dict[str, dict[str, float]],
    n_prompts: int,
    n_runs: int,
) -> str:
    lines = [f"results (median [p5, p95] across {n_runs} runs, {n_prompts} prompts/run)"]
    label_w = max(len(label) for _, label, _ in METRIC_ORDER)
    header = f"  {'metric':<{label_w}}  {'naive':<28}  {'gateway':<28}"
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))
    for key, label, fmt in METRIC_ORDER:
        n_cell = _cell(naive, key, fmt)
        g_cell = _cell(gateway, key, fmt)
        lines.append(f"  {label:<{label_w}}  {n_cell:<28}  {g_cell:<28}")
    return "\n".join(lines)


def render_markdown(
    naive: dict[str, dict[str, float]],
    gateway: dict[str, dict[str, float]],
    n_prompts: int,
    n_runs: int,
) -> str:
    lines = [
        f"_median [p5, p95] across {n_runs} runs, {n_prompts} prompts/run_",
        "",
        "| metric | naive | gateway |",
        "|---|---|---|",
    ]
    for key, label, fmt in METRIC_ORDER:
        n_cell = _cell(naive, key, fmt)
        g_cell = _cell(gateway, key, fmt)
        lines.append(f"| {label} | {n_cell} | {g_cell} |")
    return "\n".join(lines)


def dump_json(
    path: str,
    args: argparse.Namespace,
    result: dict[str, Any],
    naive_summary: dict[str, dict[str, float]],
    gateway_summary: dict[str, dict[str, float]],
) -> None:
    payload = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "git_commit": git_commit(),
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "params": vars(args),
        "n_prompts": result["n_prompts"],
        "naive": {
            "summary": naive_summary,
            "runs": [dataclasses.asdict(r) for r in result["naive"]],
        },
        "gateway": {
            "summary": gateway_summary,
            "runs": [dataclasses.asdict(r) for r in result["gateway"]],
        },
    }
    Path(path).write_text(json.dumps(payload, indent=2))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Benchmark llm-gateway against a naive rate-limit-free async loop."
    )
    p.add_argument("--n-prompts", type=int, default=200)
    p.add_argument("--runs", type=int, default=10, help="repetitions; report median + p5-p95")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--concurrency", type=int, default=40, help="naive: bounded semaphore size")
    p.add_argument("--server-rpm", type=float, default=190.0, help="server's real (enforced) RPM limit")
    p.add_argument("--server-latency-mean-ms", type=float, default=30.0)
    p.add_argument("--server-latency-sigma", type=float, default=0.5, help="log-normal shape parameter")
    p.add_argument("--server-fail-rate", type=float, default=0.03, help="probability of an injected transient 503 per call")
    p.add_argument("--gateway-tpm", type=float, default=10_000_000.0, help="high on purpose: TPM is not the constraint under test")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument(
        "--batch-endpoint",
        action="store_true",
        help="model a provider with a real synchronous multi-prompt endpoint "
        "(neither Groq nor OpenRouter has one; off by default)",
    )
    p.add_argument("--batch-wait-ms", type=float, default=25.0)
    p.add_argument("--batch-tokens", type=int, default=8000)
    p.add_argument("--max-attempts", type=int, default=6, help="shared by both arms' RetryPolicy")
    p.add_argument("--base-delay", type=float, default=0.5)
    p.add_argument("--max-delay", type=float, default=8.0)
    p.add_argument("--max-tokens", type=int, default=256, help="LLMRequest.max_tokens for every prompt")
    p.add_argument("--cost-per-1k-input", type=float, default=0.02)
    p.add_argument("--cost-per-1k-output", type=float, default=0.04)
    p.add_argument("--json", dest="json_path", default=None, help="dump raw per-run numbers to this file")
    p.add_argument("--markdown", action="store_true", help="also print a markdown table")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    print(build_header(args))
    print()
    print(EXPECTATION)

    result = asyncio.run(run_all(args))
    naive_summary = summarize(result["naive"], result["n_prompts"])
    gateway_summary = summarize(result["gateway"], result["n_prompts"])

    print(render_table(naive_summary, gateway_summary, result["n_prompts"], args.runs))

    if args.markdown:
        print()
        print(render_markdown(naive_summary, gateway_summary, result["n_prompts"], args.runs))

    if args.json_path:
        dump_json(args.json_path, args, result, naive_summary, gateway_summary)
        print(f"\nraw per-run numbers written to {args.json_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
