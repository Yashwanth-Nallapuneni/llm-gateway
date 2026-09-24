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

Live mode (--provider groq) runs the SAME two arms -- same code paths,
same metric collection -- against the real Groq API instead of the
simulated server. It exists to answer a different question than the
simulated run: not "does the gateway behave the way the token-bucket math
says it should" (the simulated run already answers that, exactly, because
it controls the server), but "does this actually work end to end against a
real provider's real HTTP behaviour." It is not a bigger or more trustworthy
version of the simulated benchmark -- it is a much smaller, much noisier
sanity check, deliberately kept tiny by a hard-coded call budget:

    GROQ_API_KEY=$(cat ~/.groq_key) python3 benchmarks/bench.py \\
        --provider groq --model allam-2-7b --n-prompts 50 --runs 3 --markdown

Every live call is capped at --max-tokens 16 by convention (kept small on
purpose to stay well inside free-tier token budgets) and the whole run
refuses to start above --max-live-calls (default 400) planned calls. See
benchmarks/README.md's "Live results" section for what a 50-prompt run can
and cannot tell you about rate limiting.
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
# Live provider mode (real Groq API)
# ---------------------------------------------------------------------------


class LiveCallBudgetExceeded(RuntimeError):
    """Raised instead of making another live HTTP call once the hard
    per-invocation ceiling (--max-live-calls) is reached.

    Deliberately not a ProviderError: RetryPolicy.should_retry() only
    retries ProviderError/known transport exceptions, so this is never
    retried -- it marks the in-flight prompt as failed and stops spending
    real quota, rather than a retry storm burning through the ceiling
    while "retrying into the wall."
    """


def _is_budget_exceeded(exc: BaseException) -> bool:
    """True if `exc` is, or was caused/raised-from, a LiveCallBudgetExceeded
    -- checked by walking __cause__/__context__ since RetryPolicy and the
    gateway's own error handling may wrap the original exception rather than
    re-raising it bare."""
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        if isinstance(current, LiveCallBudgetExceeded):
            return True
        seen.add(id(current))
        current = current.__cause__ or current.__context__
    return False


class LiveCallBudget:
    """Shared across every arm and every run of one live invocation.

    Counts real HTTP calls about to be made, not prompts -- so it also
    catches retries, which a purely static preflight check (planned =
    n_prompts * runs * 2) cannot.
    """

    def __init__(self, max_calls: int) -> None:
        self.max_calls = max_calls
        self.used = 0

    def take(self) -> None:
        # No lock needed: asyncio is single-threaded/cooperative and this
        # method never awaits, so no other task can interleave between the
        # check and the increment.
        if self.used >= self.max_calls:
            raise LiveCallBudgetExceeded(
                f"live call budget exceeded ({self.max_calls} calls total this "
                "invocation) -- stopping rather than continuing to spend real "
                "provider quota. See --max-live-calls."
            )
        self.used += 1


class LiveCallCounter:
    """Wraps a real AsyncLLMClient (GroqClient, here) with the same
    call-level counters SimClient provides for the simulated server
    (a ServerCallMetrics: calls / 429s / 5xx), so run_naive() and
    run_gateway() work against a live provider completely unmodified --
    both only ever touch `.complete()` and `.server_metrics`.

    Also where the live call budget is actually enforced: `take()` runs
    before the network call, not after, so a call that would exceed the
    ceiling never goes out.
    """

    def __init__(self, client: Any, budget: LiveCallBudget) -> None:
        self._client = client
        self._budget = budget
        self.server_metrics = ServerCallMetrics()

    async def complete(self, request: LLMRequest) -> LLMResponse:
        self._budget.take()
        self.server_metrics.calls += 1
        try:
            return await self._client.complete(request)
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

    async def aclose(self) -> None:
        aclose = getattr(self._client, "aclose", None)
        if aclose is not None:
            await aclose()


def build_live_groq_client(model: str, api_key: str) -> Any:
    """Lazy import: the `[http]` extra (httpx) is only required for live
    mode, not for the simulated benchmark this file otherwise runs entirely
    offline. Importing groq.py at module scope would make `import bench`
    fail without httpx installed even for a purely simulated run."""
    from llm_gateway.providers.groq import GROQ_BASE_URL, GroqClient

    return GroqClient(base_url=GROQ_BASE_URL, api_key=api_key, model=model)


def build_live_groq_provider(
    model: str,
    api_key: str,
    client: Any,
    *,
    rpm_limit: float,
    tpm_limit: float,
    max_concurrency: int,
) -> Any:
    from llm_gateway.providers.groq import groq_provider

    # cost_per_1k_* left at their groq_provider default (0.0): Groq's free
    # tier bills nothing, and this run's own metrics report tokens instead
    # of a dollar figure precisely because a real dollar cost isn't in play.
    return groq_provider(
        api_key,
        model=model,
        client=client,
        rpm_limit=rpm_limit,
        tpm_limit=tpm_limit,
        max_concurrency=max_concurrency,
    )


# ---------------------------------------------------------------------------
# Workload
# ---------------------------------------------------------------------------


def make_prompts(
    n: int,
    seed: int,
    max_tokens: int,
    *,
    words_min: int = 5,
    words_max: int = 40,
) -> list[LLMRequest]:
    rng = random.Random(seed)
    return [
        LLMRequest(
            prompt=f"Summarize item #{i}: " + "word " * rng.randint(words_min, words_max),
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
    # Populated in both modes, but the metric that matters in live mode:
    # Groq's free tier is $0, so cost_usd is always 0 there and tokens are
    # the only real measure of "how much work this arm actually did."
    input_tokens: int = 0
    output_tokens: int = 0
    # Set when this arm was cut off mid-run by --max-live-calls
    # (LiveCallBudgetExceeded) rather than completing on its own. An invalid
    # run's successes/failures/429s are not a measurement of the client under
    # test -- they're an artifact of the harness refusing to place calls it
    # had no budget left for -- so callers must exclude it from any reported
    # summary rather than averaging it in. See the retracted "57.5%" figure
    # in benchmarks/README.md for what happens when this isn't done.
    invalid: bool = False
    invalid_reason: str = ""


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
    input_tokens = 0
    output_tokens = 0
    clock = time.monotonic
    budget_exceeded = False

    async def one(req: LLMRequest) -> None:
        nonlocal failures, cost, input_tokens, output_tokens, budget_exceeded
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
                if _is_budget_exceeded(exc):
                    # Not a real rejection: the harness itself refused to
                    # place this call. Flag the whole arm invalid rather
                    # than letting it masquerade as a failed prompt -- see
                    # RunResult.invalid.
                    budget_exceeded = True
                if not retry.should_retry(exc, attempt):
                    failures += 1
                    return
                delay = retry.delay_for(attempt, retry.retry_after_from(exc))
                attempt += 1
                await asyncio.sleep(delay)
                continue
            latencies.append(clock() - start)
            cost += _cost(resp, cost_per_1k_input, cost_per_1k_output)
            input_tokens += resp.input_tokens
            output_tokens += resp.output_tokens
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
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        invalid=budget_exceeded,
        invalid_reason=(
            "hit --max-live-calls mid-arm (LiveCallBudgetExceeded) -- not a "
            "real measurement of this arm's success/failure/429 rate"
        )
        if budget_exceeded
        else "",
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
    provider: Any | None = None,
) -> RunResult:
    # `provider` lets a caller hand in an already-built llm_gateway.Provider
    # instead of the simulated MockProvider below -- this is how live mode
    # (run_all, --provider groq) reuses this exact function against a real
    # Provider wrapping the real Groq API, with server.server_metrics still
    # the source of requests_sent/429/5xx (see LiveCallCounter).
    if provider is None:
        # Default False on purpose. Neither Groq nor OpenRouter -- the two
        # providers this library actually ships adapters for -- has a
        # synchronous multi-prompt endpoint, so counting a 16-prompt batch as
        # ONE request would credit the gateway with a saving no real
        # deployment can collect. With it False the gateway still groups
        # requests for rate-limit accounting and dispatches them
        # concurrently, which is what really happens against a chat API.
        # --batch-endpoint models the other case: a provider that genuinely
        # accepts many prompts per call.
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
    input_tokens = sum(r.input_tokens for r in results if isinstance(r, LLMResponse))
    output_tokens = sum(r.output_tokens for r in results if isinstance(r, LLMResponse))
    # Same reasoning as run_naive: a LiveCallBudgetExceeded anywhere in this
    # arm's results means the harness cut it off mid-run, not that the
    # gateway actually failed those prompts.
    budget_exceeded = any(_is_budget_exceeded(r) for r in results if isinstance(r, BaseException))

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
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        invalid=budget_exceeded,
        invalid_reason=(
            "hit --max-live-calls mid-arm (LiveCallBudgetExceeded) -- not a "
            "real measurement of this arm's success/failure/429 rate"
        )
        if budget_exceeded
        else "",
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
        "input_tokens": float(r.input_tokens),
        "output_tokens": float(r.output_tokens),
        "total_tokens": float(r.input_tokens + r.output_tokens),
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
    ("total_tokens", "total tokens (in+out)", "{:.0f}"),
]


def valid_runs(runs: list[RunResult], label: str) -> list[RunResult]:
    """Drop any run cut off mid-arm by --max-live-calls (RunResult.invalid)
    before it reaches summarize(). The refuse-to-start check in
    run_all_live sizes --max-live-calls so this should never trigger, but
    summarize() must not silently average a harness artifact into a real
    result if it somehow does -- that is exactly how the retracted '57.5%'
    figure happened."""
    ok = [r for r in runs if not r.invalid]
    dropped = len(runs) - len(ok)
    if dropped:
        print(
            f"WARNING: dropping {dropped}/{len(runs)} {label} run(s), cut off "
            "mid-arm by --max-live-calls -- not a real result, see RunResult.invalid",
            flush=True,
        )
    if not ok:
        raise SystemExit(
            f"every {label} run was cut off mid-arm by --max-live-calls; nothing "
            "valid to report. Raise --max-live-calls and rerun."
        )
    return ok


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


def arm_order_for_run(run_index: int, arm_order: str) -> tuple[str, str]:
    """Which arm goes first in this run (0-indexed).

    Exists because, against a real provider, the two arms share one
    account's rate-limit state: whichever arm runs first gets a clean
    bucket, and whichever runs second inherits however much of that
    bucket the first arm just spent. Running every run in the same order
    bakes that bias into every number the same way; "alternate" makes any
    residual bias visible in the per-run spread instead (see
    --arm-cooldown for the other half of the fix -- actually letting the
    bucket refill between arms).
    """
    if arm_order == "naive-first":
        return ("naive", "gateway")
    if arm_order == "gateway-first":
        return ("gateway", "naive")
    # "alternate": run 0 naive-first, run 1 gateway-first, run 2 naive-first, ...
    return ("naive", "gateway") if run_index % 2 == 0 else ("gateway", "naive")


async def _cooldown(seconds: float, *, from_label: str, to_label: str) -> None:
    """Sleep between two arm executions, or before the very first one
    (`from_label="start"`).

    Applied at EVERY arm boundary -- including the boundary between one
    run's last arm and the next run's first arm, not just between the two
    arms inside a single run. A cooldown that only fires within a run lets a
    shared account's rate-limit state carry from one run into the next
    (documented in benchmarks/README.md, "Live results," as the residual
    confound behind run 2's gateway arm picking up 15 real 429s it should
    not have)."""
    if seconds <= 0:
        return
    if from_label == "start":
        print(
            f"waiting out a {seconds:.0f}s cooldown before the first arm, so the "
            "account starts this invocation with a full token bucket (not hung)...",
            flush=True,
        )
    else:
        print(
            f"  cooling down {seconds:.0f}s between {from_label} and {to_label} arms "
            "(letting the account's rate-limit bucket refill -- not hung)...",
            flush=True,
        )
    await asyncio.sleep(seconds)


async def run_all(args: argparse.Namespace) -> dict[str, Any]:
    if getattr(args, "provider", "sim") == "groq":
        return await run_all_live(args)
    return await run_all_sim(args)


async def run_all_sim(args: argparse.Namespace) -> dict[str, Any]:
    prompts = make_prompts(
        args.n_prompts,
        args.seed,
        args.max_tokens,
        words_min=getattr(args, "prompt_words_min", 5),
        words_max=getattr(args, "prompt_words_max", 40),
    )
    master = random.Random(args.seed)
    # One seed per run, drawn once from the master RNG -- reused for BOTH
    # arms so run i in the naive arm and run i in the gateway arm face
    # identically-seeded server latency/failure streams and identically
    # seeded retry jitter. See "Threats to validity" in the README for what
    # this reproducibility guarantee does and does not buy you.
    run_seeds = [master.randrange(2**31) for _ in range(args.runs)]

    naive_runs: list[RunResult] = []
    gateway_runs: list[RunResult] = []
    cooldown_s = getattr(args, "arm_cooldown", 0.0) or 0.0
    arm_order = getattr(args, "arm_order", "alternate")
    prev_label = "start"

    for run_index, rs in enumerate(run_seeds):

        async def do_naive(rs: int = rs) -> RunResult:
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
            return await run_naive(
                prompts,
                naive_server,
                concurrency=args.concurrency,
                retry=naive_retry,
                cost_per_1k_input=args.cost_per_1k_input,
                cost_per_1k_output=args.cost_per_1k_output,
            )

        async def do_gateway(rs: int = rs) -> RunResult:
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
            return await run_gateway(
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

        first, second = arm_order_for_run(run_index, arm_order)
        runners = {"naive": (do_naive, naive_runs), "gateway": (do_gateway, gateway_runs)}
        first_fn, first_list = runners[first]
        second_fn, second_list = runners[second]
        # Cooldown before EVERY arm, including the first arm of run 0
        # (from_label="start", a no-op unless --arm-cooldown is set) and the
        # first arm of every run after the first -- not just between the two
        # arms inside one run.
        await _cooldown(cooldown_s, from_label=prev_label, to_label=first)
        first_list.append(await first_fn())
        await _cooldown(cooldown_s, from_label=first, to_label=second)
        second_list.append(await second_fn())
        prev_label = second

    return {
        "n_prompts": len(prompts),
        "naive": naive_runs,
        "gateway": gateway_runs,
    }


async def run_all_live(args: argparse.Namespace) -> dict[str, Any]:
    """Same two arms, same metric collection as run_all_sim -- against the
    real Groq API instead of SimClient. See the module docstring and
    benchmarks/README.md's "Live results" section for what this can and
    cannot show at the scale this is run at.
    """
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise SystemExit(
            "GROQ_API_KEY is not set. Export it from the key file first, e.g.:\n"
            "  GROQ_API_KEY=$(cat ~/.groq_key) python3 benchmarks/bench.py --provider groq ..."
        )

    max_live_calls = args.max_live_calls
    # Worst case, not expected case: every prompt in every arm of every run
    # exhausts its full retry budget (--max-attempts calls each) before
    # succeeding or giving up. This is deliberately pessimistic -- a
    # ceiling sized to the *expected* call count is exactly what let
    # --max-live-calls truncate an arm mid-run in an earlier attempt (see
    # benchmarks/README.md's retracted "57.5%" figure): the harness cut the
    # arm off, and the harness's own refusal was then misread as a real
    # rate-limit failure. Refusing to start unless the ceiling comfortably
    # covers the worst case makes that class of truncation impossible,
    # rather than merely making it visible after the fact.
    worst_case_calls = args.n_prompts * args.runs * 2 * args.max_attempts
    planned_calls = args.n_prompts * args.runs * 2  # 2 arms, no retries, for the message below
    if worst_case_calls > max_live_calls:
        raise SystemExit(
            f"refusing to run live: worst case is --n-prompts {args.n_prompts} * "
            f"--runs {args.runs} * 2 arms * --max-attempts {args.max_attempts} = "
            f"{worst_case_calls} calls (every prompt exhausting its retry budget), "
            f"above --max-live-calls {max_live_calls}. The best case is "
            f"{planned_calls} calls (no retries at all), but sizing the ceiling to "
            "the best case is what let a real arm get truncated mid-run in an "
            "earlier attempt -- see benchmarks/README.md's retracted '57.5%' "
            "figure. Lower --n-prompts/--runs/--max-attempts or raise "
            "--max-live-calls deliberately so no arm can hit the ceiling before "
            "it finishes on its own."
        )

    prompts = make_prompts(
        args.n_prompts,
        args.seed,
        args.max_tokens,
        words_min=getattr(args, "prompt_words_min", 5),
        words_max=getattr(args, "prompt_words_max", 40),
    )
    budget = LiveCallBudget(max_live_calls)

    naive_runs: list[RunResult] = []
    gateway_runs: list[RunResult] = []
    live_clients: list[Any] = []
    cooldown_s = getattr(args, "arm_cooldown", 90.0) or 0.0
    arm_order = getattr(args, "arm_order", "alternate")
    prev_label = "start"

    try:
        for run_index in range(args.runs):

            async def do_naive() -> RunResult:
                # Naive arm: bare GroqClient, wrapped only for call counting
                # -- no rate limiting, same concurrency/retry discipline as
                # the simulated naive arm.
                naive_client = LiveCallCounter(
                    build_live_groq_client(args.model, api_key), budget
                )
                live_clients.append(naive_client)
                naive_retry = RetryPolicy(
                    max_attempts=args.max_attempts,
                    base_delay=args.base_delay,
                    max_delay=args.max_delay,
                )
                return await run_naive(
                    prompts,
                    naive_client,
                    concurrency=args.concurrency,
                    retry=naive_retry,
                    cost_per_1k_input=0.0,
                    cost_per_1k_output=0.0,
                )

            async def do_gateway() -> RunResult:
                # Gateway arm: real Provider wrapping the same kind of
                # counted client, paced by a token bucket set at/slightly
                # under Groq's real free-tier limits (--live-rpm-limit /
                # --live-tpm-limit).
                gateway_client = LiveCallCounter(
                    build_live_groq_client(args.model, api_key), budget
                )
                live_clients.append(gateway_client)
                gateway_provider = build_live_groq_provider(
                    args.model,
                    api_key,
                    gateway_client,
                    rpm_limit=args.live_rpm_limit,
                    tpm_limit=args.live_tpm_limit,
                    max_concurrency=args.concurrency,
                )
                gateway_retry = RetryPolicy(
                    max_attempts=args.max_attempts,
                    base_delay=args.base_delay,
                    max_delay=args.max_delay,
                )
                return await run_gateway(
                    prompts,
                    gateway_client,
                    rpm_limit=args.live_rpm_limit,
                    tpm_limit=args.live_tpm_limit,
                    batch_size=args.batch_size,
                    batch_wait_ms=args.batch_wait_ms,
                    batch_tokens=args.batch_tokens,
                    retry=gateway_retry,
                    cost_per_1k_input=0.0,
                    cost_per_1k_output=0.0,
                    provider=gateway_provider,
                )

            first, second = arm_order_for_run(run_index, arm_order)
            runners = {"naive": (do_naive, naive_runs), "gateway": (do_gateway, gateway_runs)}
            first_fn, first_list = runners[first]
            second_fn, second_list = runners[second]
            print(f"run {run_index + 1}/{args.runs}: {first} arm first this time", flush=True)
            # Cooldown before EVERY arm, including this run's first arm --
            # whether that's the very first arm of the whole invocation
            # (prev_label="start") or the arm right after the previous run's
            # last arm. This is the actual fix for the documented confound:
            # previously the cooldown only ran between the two arms inside
            # one run, so a run's last arm and the next run's first arm ran
            # back-to-back and could share leftover rate-limit state.
            await _cooldown(cooldown_s, from_label=prev_label, to_label=first)
            first_list.append(await first_fn())
            await _cooldown(cooldown_s, from_label=first, to_label=second)
            second_list.append(await second_fn())
            prev_label = second
    finally:
        for c in live_clients:
            await c.aclose()

    return {
        "n_prompts": len(prompts),
        "naive": naive_runs,
        "gateway": gateway_runs,
        "live_calls_used": budget.used,
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

LIVE_EXPECTATION = """\
LIVE MODE -- read this before the numbers below.

This is a small, real-network sanity check, not a bigger version of the
simulated benchmark. For this account/model, measured response headers put
the binding constraint on TOKENS PER MINUTE (x-ratelimit-limit-tokens=6000,
fast-refilling), not requests (x-ratelimit-limit-requests=7000 over a
~7-minute window, i.e. generous). --live-tpm-limit (default 5500, just under
6000) is what should actually pace the gateway arm; --live-rpm-limit
(default 900) is intentionally generous because requests are not the
constraint under test here. Two opposite failure modes are both real
possibilities and neither is a bug in this harness:
  - The workload may not generate enough tokens/min to approach 6000 (e.g.
    too few prompts, or prompts too short). If status_429 is 0 for BOTH
    arms below, that is not "the gateway ties naive" -- it is "this run
    never got close enough to the token ceiling to exercise the thing
    under test," and the simulated benchmark above remains the only
    evidence for the rate-limiting claim.
  - --live-tpm-limit may still be misconfigured relative to the account's
    actual current limit (limits can change, or differ by model). A first
    attempt at this benchmark set the gateway ceiling from a misread
    header and left the token bucket effectively unpaced -- see
    benchmarks/README.md's "Live results" for what actually happened and
    how it was corrected. If BOTH arms show heavy 429s and low success,
    check whether --live-tpm-limit was really under the account's real
    limit before concluding anything about the library.
Read the actual --live-rpm-limit/--live-tpm-limit used (in the parameters
block above) against the 429 counts below before drawing any conclusion.
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
    p.add_argument(
        "--prompt-words-min",
        type=int,
        default=5,
        help="min words in the random filler each generated prompt carries "
        "(prompt is 'Summarize item #i: ' + this many-to-'--prompt-words-max' "
        "repetitions of 'word '). Default reproduces the original simulated "
        "benchmark's short prompts; raise both bounds for a live run where "
        "token throughput (not request count) needs to bind.",
    )
    p.add_argument(
        "--prompt-words-max",
        type=int,
        default=40,
        help="max words in the random filler each generated prompt carries; see --prompt-words-min",
    )
    p.add_argument("--cost-per-1k-input", type=float, default=0.02)
    p.add_argument("--cost-per-1k-output", type=float, default=0.04)
    p.add_argument("--json", dest="json_path", default=None, help="dump raw per-run numbers to this file")
    p.add_argument("--markdown", action="store_true", help="also print a markdown table")
    p.add_argument(
        "--arm-cooldown",
        type=float,
        default=None,
        help="seconds to sleep before EVERY arm -- before the first arm of "
        "the whole invocation, between the two arms inside a run, and "
        "between one run's last arm and the next run's first arm -- so a "
        "shared account's rate-limit bucket can refill before the next arm "
        "inherits whatever the previous one used. Applying this only "
        "within a run (not across runs too) was a documented confound in "
        "an earlier live attempt -- see benchmarks/README.md's 'Live "
        "results'. Default: 0 for --provider sim (the simulated server "
        "gives each arm its own independent SimClient, so there's no "
        "shared state to let refill), 90 for --provider groq (all arms "
        "share one real account/token bucket otherwise). Pass explicitly "
        "to override either default.",
    )
    p.add_argument(
        "--arm-order",
        choices=["alternate", "naive-first", "gateway-first"],
        default="alternate",
        help="which arm goes first within each run. 'alternate' (default): "
        "run 1 naive-first, run 2 gateway-first, run 3 naive-first, ... so "
        "any residual ordering bias (e.g. an imperfect --arm-cooldown) shows "
        "up as spread across runs instead of being baked into every run the "
        "same way. 'naive-first'/'gateway-first' pin the order for all runs.",
    )

    live = p.add_argument_group(
        "live mode", "run against the real Groq API instead of the simulated server"
    )
    live.add_argument(
        "--provider",
        choices=["sim", "groq"],
        default="sim",
        help="'sim' (default): the simulated server above. 'groq': the real "
        "Groq API -- reads GROQ_API_KEY from the environment, never from a "
        "flag, so the key is never captured in argv, the printed header, or "
        "the --json dump.",
    )
    live.add_argument(
        "--model",
        default="allam-2-7b",
        help="model id passed to Groq (live mode only). Default is a small, "
        "verified-working model -- do not rely on the library's own default, "
        "which may not match a given account's access.",
    )
    live.add_argument(
        "--max-live-calls",
        type=int,
        default=400,
        help="hard ceiling on real HTTP calls for the whole invocation (both "
        "arms, all runs, retries included). The run refuses to start if "
        "n_prompts * runs * 2 already exceeds this, and stops mid-run if "
        "retries would push it over. Exists so a typo can't fire tens of "
        "thousands of requests at a real provider.",
    )
    live.add_argument(
        "--live-rpm-limit",
        type=float,
        default=900.0,
        help="gateway arm's request-bucket ceiling, live mode only. Measured "
        "real response headers for this account/model showed "
        "x-ratelimit-limit-requests=7000 with a ~7-minute reset window (roughly "
        "1000 req/min) -- generous, and NOT the binding constraint (see "
        "--live-tpm-limit). This default (900) is set comfortably under that "
        "measured request ceiling so requests are never why the gateway arm "
        "paces itself; tokens-per-minute is. An earlier version of this flag "
        "defaulted to 25, extrapolated from a single misread header on the "
        "first live attempt -- see benchmarks/README.md, 'Live results', for "
        "that mistake and the header values that corrected it. The naive arm "
        "has no rate limiting at all, live or simulated: that asymmetry is "
        "the variable under test.",
    )
    live.add_argument(
        "--live-tpm-limit",
        type=float,
        default=5_500.0,
        help="gateway arm's token-per-minute bucket, live mode only. Measured "
        "real response headers for this account/model showed "
        "x-ratelimit-limit-tokens=6000 with a fast-refilling window -- this "
        "is the actual binding constraint for this account (also matches "
        "groq_provider()'s own default tpm_limit=6000 in "
        "src/llm_gateway/providers/groq.py). This default (5500) is set just "
        "under that measured ceiling.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    live = args.provider == "groq"
    if args.arm_cooldown is None:
        args.arm_cooldown = 90.0 if live else 0.0

    print(build_header(args))
    print()
    print(LIVE_EXPECTATION if live else EXPECTATION)

    result = asyncio.run(run_all(args))
    naive_ok = valid_runs(result["naive"], "naive")
    gateway_ok = valid_runs(result["gateway"], "gateway")
    naive_summary = summarize(naive_ok, result["n_prompts"])
    gateway_summary = summarize(gateway_ok, result["n_prompts"])

    print(render_table(naive_summary, gateway_summary, result["n_prompts"], args.runs))

    if args.markdown:
        print()
        print(render_markdown(naive_summary, gateway_summary, result["n_prompts"], args.runs))

    if live:
        print(f"\nlive HTTP calls made this invocation: {result['live_calls_used']} "
              f"(ceiling was --max-live-calls {args.max_live_calls})")

    if args.json_path:
        dump_json(args.json_path, args, result, naive_summary, gateway_summary)
        print(f"\nraw per-run numbers written to {args.json_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
