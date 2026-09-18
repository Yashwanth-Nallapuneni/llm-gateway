# Resume material — llm-gateway

**Do not use these bullets on a live resume until the owner has rebuilt the
five stripped core functions from scratch (`scripts/strip.py` — see
`LEARNING_PATH.md`, Phase 0 in `ROADMAP.md`).** These bullets describe the
design and the measured results correctly, but an interviewer who reads
"rate limiter" or "retry policy" on a resume will ask to see it rebuilt on a
whiteboard. Until that rebuild-and-diff loop has actually been done once per
module, these claims are not personally defensible yet — the code is real,
the practiced recall of it is not.

## Project title (one line, with stack)

```
llm-gateway — async LLM client gateway (Python, asyncio, httpx; zero runtime deps)
```
(82 characters)

## 3-bullet version

Character counts below include the leading `•`-equivalent bullet marker's
text only (not a rendered LaTeX `\item`) — measure your own template's
bullet glyph separately if it isn't zero-width.

1. **(101 chars)** Found & fixed batching blast-radius bug via own simulated benchmark: success 76%→92.5%, matched naive
2. **(95 chars)** Built async Python LLM gateway, zero runtime deps; live Groq run: 100% success vs naive's 82.5%
3. **(103 chars)** Designed dual token-bucket limiter, jittered retry/backoff, deadline batching, capability-aware routing

Plain text, ready to paste:

```
Found & fixed batching blast-radius bug via own simulated benchmark: success 76%→92.5%, matched naive
Built async Python LLM gateway, zero runtime deps; live Groq run: 100% success vs naive's 82.5%
Designed dual token-bucket limiter, jittered retry/backoff, deadline batching, capability-aware routing
```

Notes on sourcing, so nothing here has to be taken on faith:

- Bullet 1's 76%→92.5% figures are from the **simulated** benchmark, and the
  bullet says so explicitly ("simulated benchmark") so it can't be mistaken
  for the live numbers in bullet 2
  (`benchmarks/README.md`, "Two corrections made to this benchmark," #2):
  before the fix, a batch was retried and failed as a unit, so one transient
  503 took down its 15 healthy siblings, dropping success to 76% against the
  naive loop's 92.5%. Isolating failures per request "restored parity" with
  that 92.5% naive figure — the doc does not print an exact post-fix gateway
  percentage, so "matched naive" is the accurate claim, not a specific number.
- Bullet 2's 100%/82.5% figures are from the **live** Groq run — the bullet
  names "Groq" explicitly so it can't be mistaken for bullet 1's simulated
  numbers — run 1 only, the one clean, uncontaminated pairing
  (`benchmarks/README.md`, "Live results (real Groq API)" → "The corrected
  run"): naive 33/40 (82.5%, 34 real 429s), gateway 40/40 (100%, 0
  rejections). The retracted 57.5%/"lost 17 of 40" figure from run 2's
  truncated naive arm is deliberately excluded.
- Bullet 3 is architecture, not a benchmark claim, and is supported directly
  by the README's architecture diagram and the four `*.EXPLAIN.md` files.

## 4-bullet version (one bullet may wrap to two lines)

```
Built a zero-runtime-dependency async Python gateway for LLM APIs (Groq, OpenRouter
  adapters) that paces requests under provider rate limits, retries transient
  failures, batches independent prompts, and routes around unhealthy providers.
Designed dual token-bucket rate limiting (request + token dimensions), full-jitter
  exponential backoff with Retry-After handling, deadline-bounded batching, and
  capability-filtered provider routing (logprobs, strict-JSON, context window).
Found and fixed a batching blast-radius defect using a self-built benchmark against
  a simulated rate-limited server: grouped retries failed 16 healthy prompts for one
  transient 503, dropping success to 76%; isolating per-request failures fixed it.
Validated live against the real Groq API: 100% success with 0 rate-limit rejections
  vs. a naive asyncio loop's 82.5% (34 real 429s), at a disclosed cost of slower
  wall-clock — the gateway trades throughput for completeness by design.
```

## Talking points (5 likely interviewer questions)

**1. "Why not just use LiteLLM (or Portkey, or another off-the-shelf
gateway)?"**
Concede it first: for a production app, use LiteLLM or a gateway like
Portkey. This project exists because batch evals run across DeepInfra,
Novita and OpenRouter failed in ways generic routers don't model — logprobs
silently dropped, strict-JSON support varying by upstream, different limit
models (TPM vs. concurrency per provider). LiteLLM's router filters on
context window but not on logprobs; OpenRouter's own `require_parameters`
does, but only inside OpenRouter. The genuine niche is capability-filtered
routing across providers, an eval-workload focus, and zero-dependency code
small enough to read in one sitting — not a claim of being better in
general. (This is the project's own stated positioning — see the "why not
just use LiteLLM?" section of `ROADMAP.md`; don't improvise a different
answer.)

**2. "Walk me through the batching bug you found — what actually broke, and
why?"**
The batcher grouped independent prompts into one dispatch to cut round
trips, and originally retried a failed batch as a single unit. That's
correct when a batch is a true atomic operation against a real batch
endpoint. It's wrong here, because these "batches" are just client-side
coalescing of independent HTTP calls that share rate-limit accounting —
there is no real batch endpoint behind either Groq or OpenRouter. A
transient 503 on one dispatch took down every prompt in that dispatch,
including ones that would have succeeded on their own, and a self-built
benchmark measured the gateway losing to a naive per-request loop (76% vs.
92.5% success) because of it. The fix was isolating failure handling to the
individual request inside a batch, not the batch as a whole.

**3. "How do you know these numbers are real and not benchmark noise?"**
Two ways. First, by disclosing the mistakes, not just the final numbers: an
earlier simulated run inflated its headline using a batch endpoint that
doesn't exist on any provider this library ships adapters for, and an
earlier live run misread a rate-limit figure into a live A/B, cutting the
naive arm's run short and producing an invalid 57.5% figure that was caught
and corrected in the README rather than left standing. Second, the honest
result includes the case where the library helps nothing: below a
provider's rate limit, gateway and naive are a dead heat on every metric —
if a benchmark only shows its favorable case, that's marketing, not
measurement.

**4. "What's the actual tradeoff — when would you *not* use this?"**
Two costs, stated plainly rather than hidden: tail latency, because batching
deliberately waits up to `max_wait_ms` and a throttled request queues for
capacity instead of failing fast (simulated p99: 0.157s naive vs. 0.518s
gateway); and, live, wall-clock itself — pacing under a real ~5,500
tokens/min ceiling took 32.5s vs. naive's 7.4s to finish the same 40-prompt
run. It buys completeness (100% vs. 82.5% success live), not speed. For an
interactive, latency-sensitive path, or any workload comfortably under the
provider's rate limit, this library adds a dependency and buys nothing.

**5. "What would you do differently, or what's still missing?"**
No distributed rate-limit state (buckets are per-process; LiteLLM uses
Redis for this). No streaming, which is in direct tension with batching —
batching adds latency to first token, exactly what streaming exists to
minimize, and retries become unsafe once partial streamed output has
reached the caller. Live testing is currently one provider (Groq), one
model, one account, N=2 runs — broader real-endpoint coverage (more
providers, tighter live variance, a cooldown at every arm boundary instead
of only within a run) is the explicit next milestone in `ROADMAP.md`.
