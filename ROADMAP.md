# Roadmap

Where this project goes next, in order. Researched 2026-09-11; provider
limits, free tiers and prices change often — re-check anything marked (verify)
before relying on it.

## Where it stands

Done: all four core modules, breaker, queue, metrics, mock provider, 83 tests,
strip/rebuild tooling, explainers, examples. Local git only.

The honest weak spot: **everything has only ever run against a mock.** Every
phase below is ordered to close that gap before adding features.

---

## Phase 0 — Learn it (you, ~10–15h) · gate for everything on the resume

Nothing below matters in an interview if you cannot rebuild the core.

- [ ] Read `LEARNING_PATH.md` order: EXPLAIN → source → tests, one module per sitting
- [ ] `scripts/strip.py` → rebuild → diff, once per module
- [ ] Answer every `ASK:` question out loud without looking

## Phase 1 — Correctness fixes (~4h)

Found while auditing the code; all are real gaps against live APIs.

- [x] **Concurrency cap.** `gateway._run` spawns unbounded dispatch tasks. Add a
      per-provider `asyncio.Semaphore` (`max_concurrency`). DeepInfra limits by
      *concurrent requests* (200/model), not RPM — without this you cannot model it.
- [x] **Short batch responses.** `zip(batch, responses)` silently truncates; a
      provider returning fewer responses leaves callers hanging forever. Check
      lengths and fail the unmatched futures.
- [x] **End-to-end latency.** Metrics record provider-call time, but the spec asks
      for end-to-end p50/p95/p99. Record `now - enqueued_at` instead.
- [x] **Composite request hack.** `_dispatch` overwrites `composite.max_tokens`
      with a total-token estimate. Give the router an explicit requirements object.
- [ ] Inject `clock` into `LLMGateway` like every other component.

## Phase 2 — Ship it publicly (~12–16h)

- [ ] **Rename.** `llm-gateway` is taken on PyPI. Unregistered as of research
      (verify with a TestPyPI upload — a 404 is not a guarantee):
      `aiollm-gateway`, `llm-flowgate`, `llm-batchgate`, `llm-throttle`.
- [ ] MIT `LICENSE`, full `pyproject` metadata, `py.typed`
- [ ] Install tooling (none present locally): `gh`, `uv`, `ruff`, `mypy`
- [ ] Public GitHub repo; pin it on your profile
- [ ] GitHub Actions: pytest on **3.11–3.14**, ruff lint+format, mypy
- [ ] Codecov badge (free for public repos)
- [ ] Release via **PyPI Trusted Publishing** (OIDC, no API token): TestPyPI first,
      then tag `v0.1.0`. Create the pending publisher right before publishing —
      it does not reserve the name.
- [ ] README first screen: one-line pitch, badges, a **vhs** terminal GIF of
      `failover_demo.py`, 10-line quickstart

Skip: Fly.io/Railway (no real free tier), Hugging Face Gradio/Docker Spaces
(now paid), Render (1-min cold start), a new Material for MkDocs site
(end-of-life Nov 2026), issue templates / CONTRIBUTING (solo project).

## Phase 3 — Real providers (~12–16h)

All five candidates are OpenAI-compatible, so **one** `httpx` adapter
(`pip install aiollm-gateway[http]`) with per-provider tweaks.

| Provider | Role in this project | Key quirk |
|---|---|---|
| **Groq** | $0 CI smoke tests | Free plan, real `x-ratelimit-*` headers; reset given as durations (`7.66s`); **no logprobs** |
| **OpenRouter** | Main adapter (your research stack). Not a model host: an aggregator that forwards to upstreams like DeepInfra and Novita, so it is a second routing layer behind this one | Send `provider.require_parameters: true` when a request needs logprobs/strict JSON, or they silently drop; 402 even on free models when out of credit |
| **Together / Novita** | Logprobs coverage | Together sends `logprobs` as an integer |
| **DeepInfra** | Concurrency-limited provider | 200 concurrent/model, no RPM/TPM — needs Phase 1's semaphore |

- [ ] Map errors → `ProviderError(status, retry_after)`; parse `Retry-After`,
      Groq durations, Together `x-ratelimit-reset`
- [ ] **Header sync:** correct the local token buckets from `x-ratelimit-remaining-*`
      on every response (~5h). Fixes drift when several processes share a key.
- [ ] **Redefine "batching" honestly.** Only async 24h file-based batch endpoints
      exist (OpenRouter has none). Against chat APIs, `supports_batching=False`
      and the batcher's grouping becomes coordinated concurrent dispatch. Update
      README and explainers to say so.
- [ ] Live tests behind `@pytest.mark.live`, skipped without an env var; default
      `addopts = "-m 'not live'"`. Separate `workflow_dispatch` workflow using
      GitHub Environment secrets, `max_tokens ≤ 16`, OpenRouter per-key spend cap.

## Phase 4 — Proof (~8–10h)

This is what turns "a learning project" into "measured results".

- [ ] `benchmarks/`: naive async loop vs gateway. Report **429 count, wall time,
      cost, p50/p99**. Credible method: seeded mock with stated limits and
      latency, N ≥ 10 runs, median with p5–p95, hardware + Python + commit hash,
      an honest baseline (same concurrency, same retries), one-command reproduce.
- [ ] One small **real** run (Groq free tier) alongside the mock numbers
- [ ] Blog post: the design tradeoffs + benchmark + the four bugs found. Personal
      site or dev.to; avoid Medium's paywall.

## Phase 5 — Extensions, ranked

| # | Extension | Effort | Why this rank |
|---|---|---|---|
| 1 | **Persistent queue + cached replay** (SQLite WAL, idempotency key = hash of model+params+prompt) | 10–14h | Biggest real win for eval sweeps: crashed 20k-prompt runs resume, re-scoring costs $0. Strong at-least-once / idempotency interview material |
| 2 | **Budget ceiling** (reserve worst-case cost, settle on actual usage) | 4–6h | Same reserve/settle shape as the token bucket; stops runaway spend |
| 3 | **Provider Batch API offload** (50% off, 24h window) | 12–16h | Route to the batch tier when the deadline allows. Depends on #1 for durable job IDs |
| 4 | **AIMD adaptive limiting** when headers are absent | 6–10h | Best interview talking point (TCP congestion control, Google SRE client-side throttling, Netflix concurrency-limits) |
| — | inspect_ai `ModelAPI` adapter | 6–10h | Only if you use inspect_ai for real evals |
| ✗ | Streaming | 12–20h | Low value for eval sweeps; conflicts with batching (see below) |
| ✗ | Hedged requests | — | Pays tokens twice; duplicates share a rate bucket so their latency is correlated |

**Why batching and streaming conflict** (a likely interview question): batching
deliberately *adds* up to `max_wait_ms` before dispatch, which lands directly on
time-to-first-token — the one metric streaming exists to minimize. Streaming
also breaks TPM settlement (usage arrives in the last chunk), and makes retries
unsafe once partial output has reached the caller.

---

## Positioning: "why not just use LiteLLM?"

Expect this question. Answer honestly:

1. **Concede:** for a production app, use LiteLLM or a gateway like Portkey.
2. **Motivation:** batch evals across DeepInfra, Novita and OpenRouter failed in
   ways generic routers don't model — logprobs silently dropped, strict-JSON
   support varying by upstream, different limit models (TPM vs concurrency).
   LiteLLM's router filters on context window but not on logprobs; OpenRouter's
   `require_parameters` does, but only inside OpenRouter.
3. **Depth:** token-bucket math, full-jitter backoff, breaker state machine —
   and the tradeoffs, e.g. buckets are per-process where LiteLLM uses Redis.
4. **Limits, stated first:** no distributed state, no streaming.

Genuine niche: capability-filtered routing across providers + eval-workload
focus + zero-dependency code you can read in one sitting.

## Resume

Keep the current bullets off until Phase 0 is done. After Phases 3–4, rewrite
them around **real** numbers from the benchmark, and say "concurrent dispatch"
rather than "batching" for chat-API providers.

## Sources

Competitive: docs.litellm.ai/docs/routing · openrouter.ai/docs/features/provider-routing ·
portkey.ai/docs/product/ai-gateway/circuit-breaker · developer.konghq.com/plugins/ai-rate-limiting-advanced
Providers: openrouter.ai/docs/api-reference/{limits,errors} · console.groq.com/docs/rate-limits ·
docs.deepinfra.com/account/rate-limits · docs.together.ai/docs/{rate-limits,logprobs} · docs.novita.ai/guides/llm-rate-limits
Shipping: docs.pypi.org/trusted-publishers · docs.astral.sh/uv/guides/package · github.com/charmbracelet/vhs ·
github.com/squidfunk/mkdocs-material/issues/8523 · huggingface.co/docs/hub/en/spaces-overview
Extensions: sre.google/sre-book/handling-overload · github.com/Netflix/concurrency-limits ·
aws.amazon.com/blogs/architecture/exponential-backoff-and-jitter · inspect.aisi.org.uk/providers.html
