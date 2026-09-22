# Roadmap

Researched 2026-09-11; provider limits, free tiers and prices change often,
so re-check anything marked (verify) before relying on it.

## Current state

`llm_gateway` is a zero-runtime-dependency async layer for pacing, retrying,
batching and routing requests across LLM providers. 234 tests pass; CI runs
pytest, ruff and mypy on Python 3.11-3.14 and builds/validates the package
on every push and pull request. The package is published on PyPI as
[`aiollm-gateway`](https://pypi.org/project/aiollm-gateway/) (v0.1.0,
released via Trusted Publishing from a tagged GitHub Actions run).

Providers: `MockProvider` (in-process, no network), `GroqProvider` and
`OpenRouterProvider` (real HTTP adapters with error mapping and
`Retry-After`/rate-limit-header parsing). Of the two live adapters, only
Groq has been run against its real API (three times; see
[benchmarks/README.md](benchmarks/README.md#live-results-real-groq-api) for
all three in full, including two methodology failures kept on record).
OpenRouter has only ever been exercised against a fake HTTP server in tests,
never the live API.

Other components: a persistent run store (`RunStore`, SQLite-backed, used
for resuming a crashed sweep and for idempotent replay), a budget ledger
that reserves worst-case cost and settles on actual usage
(`BudgetLedger`/`BudgetExceeded`), a circuit breaker, and a CLI
(`llm-gateway run`) that reads a `.jsonl`/`.txt` file or stdin, supports
`--dry-run` cost estimation, `--budget`, and `--provider {mock,groq,openrouter}`.
The simulated benchmark (`benchmarks/bench.py`) compares a naive
semaphore-bounded loop against the gateway on a seeded mock server with a
real enforced rate limit; see [benchmarks/README.md](benchmarks/README.md)
for the full method and numbers, including the honest reversal on
latency p99.

## Not yet done

- **OpenRouter has never been exercised against the live API.** All
  OpenRouter coverage today is unit tests against a fake HTTP server.
- **Clean live A/B methodology.** The one clean live comparison (Groq, run
  1) is uncontaminated, but a documented residual confound remains: a
  90-second cooldown is applied only *between arms within a run*, not
  between one run's last arm and the next run's first arm, so account
  rate-limit state can carry across run boundaries (this is what happened
  to run 2's gateway arm). The fix is either separate API keys per arm, or
  a cooldown enforced at every arm boundary including across runs, plus a
  `--max-live-calls` ceiling generous enough that no arm can be truncated
  mid-run (run 2's naive arm was cut short this way, not by real rejections).
- **`ruff format` not enforced in CI.** `.github/workflows/ci.yml` runs
  `ruff check` but not `ruff format --check`; the comment there explains
  that turning it on today would reformat about ten files with no
  functional benefit, and asks for a deliberate `ruff format` pass first.

## Next

Ranked by expected value, effort estimates from the original research pass:

1. **OpenRouter live coverage**: run the adapter against the real
   OpenRouter API at least once, the same way Groq was validated, and fix
   whatever that exposes. This is the most direct way to close the gap
   called out above.
2. **Provider Batch API offload** (12-16h): route eligible requests to a
   provider's async, file-based batch endpoint (roughly half price, 24h
   turnaround window) when the caller's deadline allows it. Depends on the
   durable run store for tracking batch job IDs across process restarts,
   which now exists (`RunStore`).
3. **AIMD adaptive limiting** (6-10h): when a provider gives no rate-limit
   headers to sync against, back off multiplicatively on rejection and
   grow additively on success, the same pattern used by TCP congestion
   control, Google SRE client-side throttling and Netflix's
   concurrency-limits library.
4. **`inspect_ai` `ModelAPI` adapter** (6-10h): worth building only if this
   gateway is used to drive `inspect_ai` evals; otherwise skip.

## Deliberately not doing

- **Streaming.** Low value for the eval-sweep workload this library targets,
  and it conflicts with batching: batching deliberately *adds* up to
  `max_wait_ms` before dispatch, which lands directly on time-to-first-token,
  the one metric streaming exists to minimize. Streaming also breaks TPM
  settlement (usage arrives only in the last chunk), and makes retries
  unsafe once partial output has already reached the caller.
- **Hedged requests.** Pays tokens twice, and the duplicate requests share a
  rate-limit bucket, so their latency is correlated rather than independent;
  the usual benefit of hedging (an unlucky slow request being masked by a
  faster duplicate) mostly does not materialize under a shared limit.

## Prior art and where this differs

For a production app, use LiteLLM or a gateway like Portkey. Those projects
are mature and widely deployed, so it's worth being specific about why this
one exists alongside them.

Batch evals across DeepInfra, Novita and OpenRouter failed in ways generic
routers don't model: logprobs silently dropped, strict-JSON support varying
by upstream, different limit models (TPM vs concurrency). LiteLLM's router
filters on context window but not on logprobs. OpenRouter's
`require_parameters` does, but only inside OpenRouter.

What this gateway adds: token-bucket math, full-jitter backoff, and a
breaker state machine, implemented with the tradeoffs stated up front rather
than hidden. Buckets are per-process, where LiteLLM uses Redis for shared
state across processes. There is no distributed state and no streaming
support.

The niche this fills: capability-filtered routing across providers, an
eval-workload focus, and zero-dependency code that fits in one sitting.

## Sources

Competitive: docs.litellm.ai/docs/routing · openrouter.ai/docs/features/provider-routing ·
portkey.ai/docs/product/ai-gateway/circuit-breaker · developer.konghq.com/plugins/ai-rate-limiting-advanced
Providers: openrouter.ai/docs/api-reference/{limits,errors} · console.groq.com/docs/rate-limits ·
docs.deepinfra.com/account/rate-limits · docs.together.ai/docs/{rate-limits,logprobs} · docs.novita.ai/guides/llm-rate-limits
Shipping: docs.pypi.org/trusted-publishers · docs.astral.sh/uv/guides/package · github.com/charmbracelet/vhs ·
github.com/squidfunk/mkdocs-material/issues/8523 · huggingface.co/docs/hub/en/spaces-overview
Extensions: sre.google/sre-book/handling-overload · github.com/Netflix/concurrency-limits ·
aws.amazon.com/blogs/architecture/exponential-backoff-and-jitter · inspect.aisi.org.uk/providers.html
