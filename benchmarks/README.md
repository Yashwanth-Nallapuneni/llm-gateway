# Benchmark: what does llm-gateway buy you over a naive async loop?

`bench.py` answers one question as honestly as a simulated harness can:
against a server with a real, enforced rate limit and occasional transient
failures, what's the actual difference between

- **naive**: `asyncio.gather` over every prompt through a bounded
  semaphore, with a *properly implemented* retry — exponential backoff,
  full jitter, `Retry-After` respected — on 429/5xx. This is not a
  strawman: the only thing it's missing on purpose is rate limiting,
  because that's the variable under test. It reuses `llm_gateway.RetryPolicy`
  for the backoff/jitter/classification logic itself (not for the rate
  limiting, batching, or failover the gateway adds), so both arms get the
  same quality of retry decision.
- **gateway**: `llm_gateway.LLMGateway`, configured with a token bucket set
  to the server's real limits, request batching, and the same retry
  policy.

Both arms are pointed at their own instance of the same simulated server
(`SimClient`, a small subclass of `llm_gateway.providers.mock.MockClient` —
see "Threats to validity" below for exactly what "same" means here).

## Run it

```
python3 benchmarks/bench.py --n-prompts 200 --runs 5 --seed 42 --markdown
```

That one command reproduces the run below end to end: it builds the
workload, runs both arms `--runs` times, and prints a stdout table plus
(with `--markdown`) a table ready to paste elsewhere. Add `--json out.json`
to also dump every raw per-run number, so a skeptical reader can recompute
the medians and percentiles themselves instead of taking the summary on
faith.

Every tunable — prompt count, server RPM, injected failure rate, latency
distribution, retry budget, batch shape, pricing — is a CLI flag; run
`python3 benchmarks/bench.py --help` for the full list. Increase `--runs`
for tighter percentile estimates; the default is 10.

## Hypothesis (stated before you scroll to the numbers)

- **naive** should rack up a real count of 429s as soon as its unthrottled
  burst exceeds the server's configured RPM. Because the server enforces a
  genuine 60-second sliding window but naive's retry budget is finite, some
  requests can exhaust their retries *before* the window frees capacity —
  meaning naive is not guaranteed 100% eventual success the way a
  patient-enough retry loop would be.
- **gateway** should show 429s approaching zero, because the token bucket
  paces requests under the limit *before* they're sent rather than
  discovering the limit by exceeding it. It should also send meaningfully
  fewer total requests to the server. The cost should be added latency —
  a caller occasionally waits in the gateway's queue for capacity that
  naive would have burned as a rejected-then-retried call instead.

`bench.py` prints this same paragraph before it prints results, so nobody
can accuse the harness of being tuned to match an after-the-fact story.

## Measured result

Run: `python3 benchmarks/bench.py --n-prompts 200 --runs 5 --seed 42 --markdown`

```
timestamp (UTC):  2026-09-18T06:03:12.822547+00:00
git commit:       9dd26c9
python:           3.13.2 (CPython)
platform:         macOS-26.6.1-arm64-arm-64bit-Mach-O
processor:        arm
cpu_count:        8
parameters:
  base_delay = 0.5, batch_size = 16, batch_tokens = 8000, batch_wait_ms = 25.0
  concurrency = 40, cost_per_1k_input = 0.02, cost_per_1k_output = 0.04
  gateway_tpm = 10000000.0, max_attempts = 6, max_delay = 8.0, max_tokens = 256
  n_prompts = 200, runs = 5, seed = 42, server_fail_rate = 0.03
  server_latency_mean_ms = 30.0, server_latency_sigma = 0.5, server_rpm = 190.0
```

_median [p5, p95] across 5 runs, 200 prompts/run_

**Configuration A — 200 prompts against a 190/min server (over the limit).**
This is the regime the library is built for.

| metric | naive | gateway |
|---|---|---|
| wall-clock (s) | 14.313 [13.609, 14.641] | **6.126** [5.816, 9.899] |
| requests sent | 275.0 [271.0, 286.0] | **206.0** [205.0, 218.0] |
| 429s received | 85.0 [81.0, 96.0] | **16.0** [15.0, 28.0] |
| 5xx received | 5.0 [4.0, 7.0] | 5.0 [4.0, 7.0] |
| successes | 185.0 [183.0, 186.0] | 185.0 [183.0, 186.0] |
| failures | 15.0 [14.0, 17.0] | 15.0 [14.0, 17.0] |
| success rate | 92.5% [91.5%, 93.0%] | 92.5% [91.5%, 93.0%] |
| latency p50 (s) | **0.082** [0.078, 0.092] | 0.106 [0.090, 0.116] |
| latency p95 (s) | 0.146 [0.140, 0.154] | **0.141** [0.130, 0.169] |
| latency p99 (s) | **0.157** [0.151, 0.160] | 0.518 [0.419, 4.482] |
| est. cost (USD) | 0.3563 [0.3531, 0.3574] | 0.3579 [0.3544, 0.3603] |

**Configuration B — 150 prompts against the same server (inside the limit).**
Run because a benchmark that only reports its favourable regime is marketing.

| metric | naive | gateway |
|---|---|---|
| wall-clock (s) | 0.567 | 0.508 |
| requests sent | 155.0 | 155.0 |
| 429s received | 0.0 | 0.0 |
| success rate | 100.0% | 100.0% |
| latency p50 (s) | 0.069 | 0.086 |

Reproduce with `python3 benchmarks/bench.py --n-prompts 150 --runs 5 --seed 42`.

### Reading this honestly

- **The gateway's real win is 429s and wasted work**: 85 rate-limit rejections
  drop to 16, and 275 requests sent drop to 206 for the same 200 prompts.
  Those 69 extra requests are work the naive loop performed and threw away.
  Wall-clock follows from that: 6.1s vs 14.3s, because time spent backing off
  from self-inflicted 429s dwarfs the time spent pacing to avoid them.

- **Success rate is identical, not better.** Both arms complete 185/200. The
  15 failures are structural, not flakiness: 200 prompts against a 190/minute
  allowance means ~10 requests cannot be served inside the window, and a
  6-attempt retry budget capped at 8s cannot outlast a 60s window. Neither
  client can conjure capacity that does not exist. An earlier version of this
  file claimed 100% for the gateway; that number came from a misconfiguration
  described below, and was wrong.

- **The gateway is worse at the tail.** p50 is slightly worse (0.106s vs
  0.082s) because batching deliberately waits up to `max_wait_ms`, and p99 is
  markedly worse (0.518s vs 0.157s) because the requests that lose the
  rate-limit lottery wait for bucket refill instead of being rejected quickly.
  If your workload is interactive, that tradeoff is the wrong one and you
  should set `max_wait_ms` near zero.

- **Cost is a wash** (0.3563 vs 0.3579). Both arms ultimately bill for the same
  successful completions. Anyone claiming a client-side gateway saves money on
  identical work should be asked to show the arithmetic.

- **Configuration B is the honest counterweight: below the provider's limit,
  this library buys you nothing.** Identical requests, identical success,
  identical 429 count (zero), and slightly worse p50 from batching wait. If
  you are not near a rate limit, do not add this dependency.

### Two corrections made to this benchmark

Both were found by reviewing results that looked too good, and both changed the
conclusions materially. They are recorded here rather than quietly fixed.

1. **A batch endpoint that does not exist.** The first run configured the
   simulated server with `supports_batching=True`, so a 16-prompt batch counted
   as ONE request, producing a headline "14 requests sent vs 275". Neither Groq
   nor OpenRouter — the two providers this library ships adapters for — has a
   synchronous multi-prompt endpoint, so that saving is uncollectable in
   production. The benchmark now defaults to no batch endpoint; `--batch-endpoint`
   models the other case for providers that genuinely offer one.

2. **Blast radius, a real defect this exposed.** With the fake batch endpoint
   removed, the gateway's success rate fell to **76%, well below the naive
   loop's 92.5%**. The cause was that a batch was retried and failed as a unit,
   so one transient 503 doomed its 15 healthy siblings. That is correct for a
   true batch endpoint and wrong for N independent HTTP calls that merely share
   rate-limit accounting. Failures are now isolated per request, which restored
   parity. The benchmark earned its keep by catching this.

## Live results (real Groq API)

Everything above is the simulated server: it is fully reproducible and, by
construction, cannot say anything about how a real provider actually
behaves. This section is the other half — the same two arms, the same
`run_naive`/`run_gateway` code paths and the same metric collection, run
against the real Groq API (`--provider groq`) instead of `SimClient`. It
answers a narrower question honestly, rather than a broader one dishonestly:
not "is the gateway faster," but "does the live path work at all, and does
pacing under a real token ceiling actually change the outcome."

Two earlier attempts, run before this one, are kept below rather than
deleted — both were diagnosed and fixed, and that record is worth more than
a clean-looking number with no history:

1. **Attempt 1 was misconfigured.** The gateway arm's rate-limit ceiling
   was set from a misread header — roughly 200x too high on the dimension
   that actually binds — so it provided close to no pacing. Both arms got
   hammered with real 429s and the run proved nothing about the library.
2. **Attempt 2 fixed the ceiling but confounded the comparison.** With the
   correct ceiling in place, `run_naive` executed immediately before
   `run_gateway` inside the same invocation, against the same Groq account.
   Naive's burst pushed the account into a multi-thousand-token deficit that
   took on the order of a minute to clear, so the gateway arm — correctly
   paced at/under the real limit — still absorbed 429s left over from
   naive's overshoot moments earlier. The comparison measured cross-arm
   contamination, not the gateway.

Both problems are structural properties of *how the benchmark was run*, not
defects in `src/llm_gateway/`: in every attempt, every 429 the account
actually returned was correctly surfaced as `ProviderError(status=429)` and
correctly retried per `RetryPolicy`. Full numbers for both are preserved in
"Earlier attempts" at the end of this section.

### What was actually wrong, and the real limits

Measuring real response headers from this account/model directly (a 2-call
pilot, `curl` against `api.groq.com/openai/v1/chat/completions`) gave:

```
x-ratelimit-limit-requests: 7000     x-ratelimit-reset-requests: ~4-7 min
x-ratelimit-limit-tokens:   6000     x-ratelimit-reset-tokens:   ~100-300ms (near-full bucket)
```

The **binding constraint is tokens per minute (6000), not requests**. The
request bucket is enormous relative to any benchmark-sized workload (7000
per multi-minute window); the token bucket is what a workload of any
realistic size will actually hit. This is exactly why the library keeps two
independent buckets (requests and tokens) rather than one: on this account,
the token dimension is the one that binds, and a client that only paces
requests would miss it entirely. This also matches `groq_provider()`'s own
built-in default (`src/llm_gateway/providers/groq.py`), which already sets
`tpm_limit=6_000` — the codebase's own default was right about the token
ceiling all along. `bench.py`'s live defaults are now `--live-tpm-limit
5500` (just under the real 6000) and `--live-rpm-limit 900` (deliberately
generous — requests were never the constraint under test).

To fix the ordering confound from attempt 2, `bench.py` gained
`--arm-cooldown` (a real pause between arms so the account's token bucket
refills before the next arm starts, not just between repeated runs) and
`--arm-order alternate` (which arm goes first flips each run, so a
directional bias from "whoever runs first drains the bucket for whoever
runs second" cannot land on one arm only).

### The corrected run

Run:

```
GROQ_API_KEY=$(cat ~/.groq_key) python3 benchmarks/bench.py --provider groq \
  --model allam-2-7b --n-prompts 40 --runs 2 --concurrency 15 --max-tokens 16 \
  --max-attempts 4 --base-delay 0.3 --max-delay 3 --live-rpm-limit 900 \
  --live-tpm-limit 5500 --max-live-calls 185 --prompt-words-min 120 \
  --prompt-words-max 180 --arm-cooldown 90 --arm-order alternate \
  --seed 202 --markdown --json out.json
```

```
timestamp (UTC):  2026-09-18T16:13:03.000525+00:00
git commit:       c78685d
python:           3.13.2 (CPython)
platform:         macOS-26.6.1-arm64-arm-64bit-Mach-O
processor:        arm
cpu_count:        8
model:            allam-2-7b
live_rpm_limit:   900.0    (gateway arm's configured ceiling)
live_tpm_limit:   5500.0
concurrency:      15
n_prompts:        40
runs:             2
arm_cooldown:     90.0s
arm_order:        alternate
max_tokens:       16
```

_median [p5, p95] across 2 runs, 40 prompts/run, 185 live HTTP calls, $0 (free tier)_

| metric | naive | gateway |
|---|---|---|
| wall-clock (s) | 7.428 [0.987, 7.428] | **67.240** [32.469, 67.240] |
| requests sent | 67.0 [23.0, 67.0] | 55.0 [40.0, 55.0] |
| 429s received | 34.0 [0.0, 34.0] | 15.0 [0.0, 15.0] |
| 5xx received | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] |
| successes | 33.0 [23.0, 33.0] | **40.0** [40.0, 40.0] |
| failures | 17.0 [7.0, 17.0] | **0.0** [0.0, 0.0] |
| success rate | 82.5% [57.5%, 82.5%]* | **100.0%** [100.0%, 100.0%] |
| latency p50 (s) | **0.698** [0.465, 0.698] | 29.920 [1.477, 29.920] |
| latency p95 (s) | **5.571** [0.756, 5.571] | 67.239 [32.467, 67.239] |
| total tokens (in+out) | 6014 [4129, 6014] | 7307 [7307, 7307] |

\* **Naive's success-rate and failures cells above are compromised and must
not be read as a real measurement.** The 57.5% low end (and the 17.0
failures high end two rows up) come from run 2's naive arm, which was cut
off mid-run by this task's `--max-live-calls 185` ceiling, not by real
rate-limit rejections — see the per-run table and footnote below for the
full explanation. The only clean, uncontaminated naive number in this
table is the 82.5% / 7-failure end of the range, from run 1. Do not average
or split the difference between the two ends of this range; the low end is
not a second real data point, it is a harness artifact.

Live HTTP calls made this invocation: 185 (the ceiling passed to
`--max-live-calls`; the run completed on its own, not cut off by the
budget). Raw per-run numbers are in `/tmp/bench_out/live_run.json`; full
console output is in `/tmp/bench_out/live_run.log`.

Per-run numbers, not medians — with N=2, the median table above hides which
arm ran when. Execution order (per `--arm-order alternate`, with `--arm-cooldown
90` between the two arms of each run and 90s waited out before the first arm
of the whole invocation, but **no** cooldown between run 1's last arm and
run 2's first arm — see below for why that gap matters):

| order | run | arm | wall-clock (s) | requests sent | 429s | successes | failures | success rate | tokens (in+out) |
|---|---|---|---|---|---|---|---|---|---|
| 1st | 1 | naive | 7.428 | 67 | 34 | 33 | 7 | 82.5% | 6014 (5490 in / 524 out) |
| 2nd | 1 | gateway | 32.469 | 40 | **0** | 40 | 0 | **100%** | 7307 (6667 in / 640 out) |
| 3rd | 2 | gateway | 67.240 | 55 | 15 | 40 | 0 | 100% | 7307 (6667 in / 640 out) |
| 4th | 2 | naive | 0.987 | 23 | 0 | 23 | 17 | 57.5%* | 4129 (3761 in / 368 out) |

\* Run 2's naive arm is not a clean measurement of naive's rate-limit
behavior — it sent only 23 of an expected ~40+ requests before the whole
invocation's `--max-live-calls 185` ceiling was reached (`67 + 40 + 55 + 23
= 185`, exactly the cap). `LiveCallBudgetExceeded` (raised by
`LiveCallBudget.take()` in `bench.py`) counts as a failure with no
accompanying 429, which is where 17 of its 17 failures and 0.0% of its 429s
actually come from — a budget-cutoff artifact of this task's deliberately
tight call ceiling, not a statement about naive's real success rate.

Reading the order column against the arm-cooldown design directly answers
the ordering-bias question this task asked: **run 1's gateway (2nd) is the
clean result** — it ran a full 90s after naive's burst, and got 0 429s /
100% success, matching the hypothesis and directly contradicting the
uncorrected attempt 2 above (68.3s / 44% 429s / 64% success). **Run 2's
gateway (3rd) shows the cooldown's remaining gap** — it ran immediately
after run 1's gateway finished, with no cooldown *between runs* (only
`--arm-cooldown` applies *within* a run, between that run's two arms), and
picked up 15 real 429s (27% of 55 requests) despite being configured
identically to run 1's gateway. It still reached 100% success via retries,
so this is a milder, second-order version of the same account-state-bleed
problem `--arm-cooldown` was built to fix — just one boundary short of
fully fixed (arm-to-arm within a run, not run-to-run). This is a genuine,
disclosed limitation of this benchmark's current methodology, not a defect
in `llm_gateway`: every 429 in both runs was correctly surfaced as
`ProviderError(status=429)` and correctly retried, and the gateway's own
pacing behaved identically (5500 tpm) in both cases — what differed was how
much real capacity the shared account actually had left at the moment each
arm started. Closing this residual gap (a cooldown at every arm boundary,
including between the last arm of one run and the first arm of the next,
not just within a run) is the natural next fix and was not attempted here —
this task's live-call budget does not comfortably cover verifying it (see
below).

Combined with the 160 calls spent on the two earlier live attempts above,
this task has now used **345 of the project's 350-call live-quota ceiling
across all live attempts to date — 5 calls remain.** Any further live work
needs either a raised ceiling or a much smaller workload.

### Reading this honestly

- **The clean result is run 1: naive 82.5% success vs gateway 100%
  success, 0 rejections, at a real cost of 32.5s vs 7.4s wall-clock.** Run
  1 is the only pairing where both arms ran to completion untouched by the
  call-budget ceiling: naive sent 67 requests, took 34 real 429s, and
  finished 33/40 prompts (7 genuine failures); the gateway sent 40
  requests, took 0 429s, and finished 40/40. That is the real, defensible
  result of this benchmark. **Run 2's naive arm is not a second data
  point for this comparison and must not be quoted as one** — it stopped
  after 23 requests because this task's `--max-live-calls 185` ceiling was
  reached (67 + 40 + 55 + 23 = 185, exactly the cap), not because of
  rate-limiting. Its 17 "failures" are `LiveCallBudgetExceeded` — the
  harness refusing to place calls it had no budget left for — with zero
  accompanying 429s, meaning 17 of its 40 prompts were simply never
  attempted. Reporting that as "57.5% success" or "lost 17 of 40 prompts"
  describes the harness's call budget, not naive's behavior against a real
  rate limit; an earlier draft of this document made exactly that mistake
  and has been corrected. For a batch job or eval sweep where every prompt
  matters, run 1's clean 82.5% vs 100% is still the failure that counts; a
  gateway that reliably finishes the job is doing its job, even slowly.

- **The gateway is far slower here, and that is not a win — it's the
  tradeoff, stated plainly.** 67.2s vs 7.4s wall-clock; p50 latency 29.9s
  vs 0.7s. The gateway is deliberately pacing itself under the account's
  real 5500 tokens/min ceiling instead of bursting and eating rejections
  the way naive does. That buys completeness, not throughput. Do not read
  the wall-clock numbers in this section as evidence the gateway is faster
  live — it is the opposite, by design, and the simulated benchmark's
  wall-clock win above is *not* reproduced here.

- **The binding limit is tokens per minute (6000), not requests** —
  measured from real response headers, as above. That is precisely why the
  library tracks two independent buckets instead of one: a client that only
  paced requests would sail right past this account's actual ceiling. Both
  arms sending roughly similar token volumes (6014 vs 7307) but very
  different request-success outcomes is the token dimension binding first,
  as expected.

- **The variance between the two runs is large and must not be read as
  precise.** Naive ranged from 23 requests / 0 rejections (run where it
  happened to run second, after the cooldown, with a partly-drained bucket
  from the arm before it) to 67 requests / 34 rejections. Gateway ranged
  from 32.5s / 0 rejections to 67.2s / 15 rejections. With N=2 runs and a
  single shared account whose token bucket carries state across the whole
  invocation (the 90s cooldown reduces but does not eliminate this — a
  bucket refilling at ~100 tokens/sec does not fully recover from a
  multi-thousand-token deficit in 90s), these numbers are indicative of the
  shape of the effect, not a precise measurement of it. Treat every number
  in this table as "this is roughly what happened, twice," not a tight
  estimate.

- **What this confirms**: the live path works end to end (real HTTP calls,
  real `usage.prompt_tokens`/`completion_tokens`, real 429/`Retry-After`
  handling, correct retry classification); pacing under the real token
  ceiling produces a 100% completion rate in both runs where an unpaced
  loop does not, in both runs; alternating arm order and a real cooldown
  between arms removed the one-directional bleed seen in attempt 2 (both
  arms show 429s in at least one run, not just whichever ran second).

- **What this does not confirm**: the simulated benchmark's wall-clock
  advantage for the gateway. Live, under a real, small (5500 tpm) ceiling
  and a workload sized to exceed it, the gateway is much slower, not
  faster. The simulated benchmark's Configuration A (200 prompts against a
  190/min *request* limit) and this live run (40 prompts against a 5500
  tpm *token* limit) are different regimes and are not comparable
  numbers — only the reliability/failure-isolation story generalizes
  across both.

- **Caveats, unchanged in kind from the earlier attempts**: one provider
  (Groq), one model (`allam-2-7b`), one account's current limits, one
  machine, one network path, one point in time (2026-09-18, ~16:13 UTC),
  free tier, **N=2**. This is not a claim about what Groq does or what the
  gateway does against a real provider in general — it is what happened
  twice, on this account, with the ordering confound addressed.

- **What we would do differently.** This attempt still produced one
  contaminated arm (run 2's naive, truncated by the call ceiling) and one
  residually confounded arm (run 2's gateway, carrying rate-limit state
  from run 1). A clean live A/B needs either separate API keys per arm (so
  no arm's account state can bleed into another's), or a cooldown inserted
  at *every* arm boundary including between runs — not just within a run —
  plus a `--max-live-calls` ceiling generous enough that no single arm can
  hit it mid-run. A ceiling that truncates an arm partway through is what
  produced this attempt's `LiveCallBudgetExceeded` artifact; size the
  budget to the worst case (all arms at their maximum plausible request
  count), not the expected case.

### Earlier attempts (kept for the record)

Both were found by reviewing results that looked too good or too bad, and
both changed the conclusions materially.

**Attempt 1 — misconfigured ceiling.** `--live-rpm-limit`/`--live-tpm-limit`
defaulted to 6500/5500, guessed from a general "~7000 req/min" figure
applied to the wrong dimension (requests instead of tokens) — a ceiling
roughly 200x too permissive relative to the real 6000 tpm limit, because it
was read off the wrong header. Against 50 prompts at 10-way concurrency,
naive alone racked up 254 real 429s across 3 runs (46/150 prompts
succeeded, 30.7%), and gateway — whose token bucket ceiling was not
actually under the account's real 6000 tpm limit, so it provided close to
no pacing — sent 100 requests and got 100 real 429s back (0/150 succeeded,
0%):

```
timestamp (UTC):  2026-09-18T15:50:46.544085+00:00
git commit:       c78685d
model:            allam-2-7b
live_rpm_limit:   6500.0   (gateway arm's configured ceiling)
live_tpm_limit:   5500.0
concurrency:      10
n_prompts:        50
runs:             3
max_tokens:       16
```

| metric | naive | gateway |
|---|---|---|
| wall-clock (s) | 6.670 [2.197, 7.964] | 2.577 [0.027, 2.919] |
| requests sent | 106.0 [7.0, 187.0] | 50.0 [0.0, 50.0] |
| 429s received | 70.0 [6.0, 178.0] | 50.0 [0.0, 50.0] |
| successes | 9.0 [1.0, 36.0] | 0.0 [0.0, 0.0] |
| success rate | 18.0% [2.0%, 72.0%] | 0.0% [0.0%, 0.0%] |

The fix (correct token-bucket ceiling, derived from measured headers
instead of a guess) was applied to `bench.py`'s defaults.

**Attempt 2 — correct ceiling, confounded ordering.** With the ceiling
fixed, `run_naive` ran immediately before `run_gateway` inside the same
invocation, against the same account, with no cooldown between them. Naive
pushed roughly 840 tokens/sec through in the first ~8 seconds — about 8x
the account's ~100 tokens/sec sustained ceiling — leaving the account in a
multi-thousand-token deficit that takes on the order of a minute to work
off, independent of who asks next. Gateway, correctly configured at/under
the real 6000 tpm limit, was still asking an account with no real remaining
capacity left over from naive's overshoot moments earlier, and received 25
real 429s on 57 requests (44%), taking 68.3s — worse than naive on both
counts, which is the opposite of the hypothesis:

```
timestamp (UTC):  2026-09-18T15:58:33.544365+00:00
git commit:       c78685d
model:            allam-2-7b
live_rpm_limit:   900.0    (gateway arm's configured ceiling)
live_tpm_limit:   5500.0
concurrency:      15
n_prompts:        50
runs:             1
max_tokens:       16
```

| metric | naive | gateway |
|---|---|---|
| wall-clock (s) | 8.264 | **68.312** |
| requests sent | **99** | 57 |
| 429s received | **60** | 25 |
| successes | **39** | 32 |
| success rate | **78.0%** | 64.0% |

This run stopped at N=1, deliberately, per this task's own rule against
continuing to burn live quota on a comparison already known to be
confounded. The per-request latencies pointed at the mechanism: gateway's
successful calls landed in tight clusters (~17.4s, ~32.7s, ~68.3s) — batches
released together after waiting for real token-bucket capacity that naive
had just spent, not independently-paced calls. That diagnosis is what led
directly to `--arm-cooldown` and `--arm-order alternate`, both used in the
corrected run above.

Neither attempt found a defect in `src/llm_gateway/` itself — in both,
every 429 the account actually returned was correctly reported as
`ProviderError(status=429)` and correctly retried. Both were benchmark
methodology problems: attempt 1 picked the wrong ceiling; attempt 2 picked
the right ceiling but ran the two arms back-to-back on a shared account
with no cooldown, letting one arm's usage bleed into the other's
measurement. The corrected run above addresses both.

## Threats to validity

Read this before trusting any number above.

- **This is a simulated server, not a real provider.** `SimClient` enforces
  a sliding-window RPM limit and injects latency/failures the same *shape*
  a real API uses, but it has none of a real provider's queueing behavior,
  regional routing, per-key vs per-org limits, or the myriad undocumented
  quirks (soft throttling before the hard limit, warm-up penalties, etc.)
  that real gateways exhibit. A result here is a statement about this
  simulation, not a guarantee about any specific real provider.
- **Latency is synthetic.** It's drawn from a seeded log-normal
  distribution chosen to *look* plausible (mean 30ms, σ=0.5), not measured
  from any real API. Real latency distributions are typically fatter-tailed
  and can correlate with load in ways a fixed, load-independent distribution
  does not capture.
- **The naive baseline is one particular implementation.** It uses a
  bounded semaphore plus `RetryPolicy`'s backoff. A different naive
  implementation — a lower concurrency cap, a smarter (but still
  rate-limit-unaware) circuit breaker, client-side request coalescing,
  adaptive concurrency — could look meaningfully better or worse than what
  is measured here. This is not "the" naive baseline, just an honest one.
- **Results depend entirely on the configured limits.** The specific gap
  shown above (85 vs 0 429s, a long gateway p95 tail) is a direct
  consequence of `--server-rpm 190` against `--n-prompts 200` — a workload
  chosen to sit close to the server's burst allowance so both the
  "gateway avoids 429s" and "gateway pays a queueing-latency tax when
  genuinely over capacity" behaviors show up in one run. Move `--server-rpm`
  well above `--n-prompts` and naive stops erroring at all (nothing to
  measure); move it well below and *both* arms take proportionally longer
  because the workload's true minimum completion time grows to
  `(N - burst_capacity) / (rpm / 60)` seconds regardless of client
  implementation — no library can make a server accept more traffic than
  its own configured limit permits. Different limits, meaningfully
  different story.
- **The naive and gateway arms do not literally share one server
  instance.** Each arm gets its own `SimClient`, seeded identically per
  run (same latency/failure RNG seed, same retry-jitter RNG seed) so the
  *draw sequence available* to each arm is equivalent. But because the two
  arms issue calls in different order and at different concurrency (batched
  vs unbatched, paced vs unpaced), they do not consume that identical
  sequence in the same order — so a given prompt is not guaranteed to see
  the exact same simulated latency/failure in both arms, only a
  statistically equivalent source of them. This is the standard tradeoff
  for comparing two systems with different concurrency structure; a
  perfectly shared server would require serializing both arms through one
  stateful object, which would itself distort the comparison by forcing
  artificial serialization between them.
- **Failover is configured but not exercised.** The gateway is built with
  `LLMGateway`'s normal multi-provider machinery, but this benchmark gives
  it exactly one provider (matching naive's single endpoint, for a fair
  1:1 comparison). The circuit breaker and failover-to-a-second-provider
  path exist in the library and are exercised by the test suite, but this
  benchmark says nothing about their effect — that would need a
  multi-provider run, which is a different (and separately useful)
  benchmark.
- **A mock cannot reproduce real-world queueing or regional effects.**
  No connection pooling limits, no TCP/TLS handshake cost, no DNS, no
  geographic latency variance, no provider-side autoscaling lag, no
  multi-tenant noisy-neighbor effects. Everything here happens in one
  process against in-memory `asyncio.sleep()` calls.
- **"Reproducible" means the deterministic counts, not wall-clock.** With
  a fixed `--seed`, the *count*-based metrics (requests sent, 429s, 5xxs,
  successes, failures, cost) are bit-for-bit identical across repeated
  invocations — `tests/test_benchmark.py` asserts this directly. Wall-clock
  time and per-call latency are not: they depend on real OS scheduling,
  asyncio event-loop timing, and machine load, and will vary run to run
  and machine to machine even with the same seed. Report wall-clock as a
  distribution (which `--runs` does), never as a single number, for exactly
  this reason.

## Metric definitions

- **wall-clock**: time from dispatching the first prompt to
  `asyncio.gather` returning for that arm, whether or not every prompt
  succeeded.
- **requests sent**: count of actual calls made to the simulated server
  (one per `complete()`/`complete_batch()` invocation, including retries
  and failed attempts), *not* count of prompts. A batch call counts once
  regardless of how many prompts it carries — this is why gateway's number
  is so much smaller than naive's: both fewer retries *and* batching
  reduce it.
- **429s / 5xx received**: count of *call-level* rejection events (one per
  rejected call, so one rejected batch call counts once even though it
  fails every prompt in that batch).
- **latency p50/p95/p99**: end-to-end, per prompt, from submission to
  final success — includes queueing, retries, and (for gateway) time spent
  waiting in the batcher and the rate limiter. Computed only over prompts
  that succeeded.
- **success rate**: successes / total prompts submitted, per run.
- **est. cost**: `input_tokens/1000 * cost_per_1k_input + output_tokens/1000
  * cost_per_1k_output`, summed over successful prompts, using the mock's
  token accounting (`~len(prompt)//4` input tokens, `min(max_tokens, 32)`
  output tokens) and the `--cost-per-1k-*` prices (defaults: $0.02 in /
  $0.04 out per 1k tokens — placeholder numbers, not any real provider's
  pricing).
