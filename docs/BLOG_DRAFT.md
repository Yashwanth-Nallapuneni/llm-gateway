# My benchmark caught my library losing. Twice.

I built `llm-gateway` after running batch evals across DeepInfra, Novita and
OpenRouter and getting tired of writing the same ad-hoc `time.sleep()` calls
and manual provider-switching logic every time. It's a small async Python
layer that sits between your code and a provider's API: it paces requests
under rate limits, retries what's worth retrying, groups prompts to cut round
trips, and routes around a provider that starts failing.

This post isn't "I built a library." It's about what happened when I finally
measured it against a naive implementation instead of assuming it was better.
The first measurement was dishonest by accident. The second was honest, and it
showed my library losing to a plain `asyncio.gather` loop — because of a real
design defect in the code, not in the benchmark. Finding that, and fixing it,
is the most useful thing that came out of this project.

## The problem this actually solves

Driving LLM APIs at volume, four things go wrong, and they compound.

Providers cap requests-per-minute and tokens-per-minute; exceed either and you
get a `429`, and naive retry-on-429 makes it worse. Transient failures —
`502`, `503`, resets — are recoverable, but only with correct backoff.
Sending hundreds of small independent prompts one at a time pays full network
and queueing latency on each, when the provider would happily take sixteen of
them together. And capability fragmentation is the sneaky one: one provider
returns logprobs but not strict JSON schema, another does schema but not
logprobs, and choosing by hand per request is the kind of bookkeeping that
works fine for a week and then quietly routes 4,000 rows of an eval through
the wrong endpoint.

## What the library does

Two token buckets per provider — one for requests, one for tokens — because a
provider that limits on tokens-per-minute will happily let you blow past it if
you're only counting requests. Retries use exponential backoff with full
jitter, `Retry-After` respected when the provider sends one. A batcher
coalesces requests that arrive close together, flushing on whichever of three
conditions hits first: batch size, cumulative token estimate, or a deadline
anchored to the first request in the batch. A router filters providers by hard
capability constraints (logprobs, strict JSON, context window) first, then
health (circuit breaker state), then ranks survivors by rate-limit headroom
before cost. All of it is client-side — no proxy, no server, zero runtime
dependencies.

## The batching defect the benchmark exposed

Here's the part worth reading this post for.

I built a benchmark to compare the gateway against a naive `asyncio.gather`
loop with a properly-implemented retry — full jitter, `Retry-After` respected,
exponential backoff — the only thing missing on purpose was rate limiting,
since that's the variable under test. First run, against a simulated server
enforcing 190 requests/minute with 200 prompts, I configured the simulated
provider with `supports_batching=True`. The result: a 16-prompt batch counted
as one request to the server, and the headline came out as roughly 14
requests sent versus 275 for the naive loop.

That number is fake. Neither Groq nor OpenRouter — the two providers this
library ships adapters for — has a synchronous multi-prompt endpoint. The
"savings" that headline implied is uncollectable against any real API this
library actually talks to. I turned the fake batch endpoint off and re-ran.

With the fake endpoint gone, the gateway's success rate fell to **76%,
against the naive loop's 92.5%**. My own library lost, on the metric that
matters most, to the thing it was supposed to improve on.

The cause: my batcher retried a failed batch as a single unit. Group sixteen
independent prompts into one dispatch, and if that dispatch comes back with a
transient `503`, the retry logic failed all sixteen together — including the
fifteen that had nothing wrong with them. That behavior is *correct* for a
true batch endpoint, where a batch really is one atomic unit of work and a
partial retry isn't meaningful. It is wrong here, because these sixteen
prompts were never atomic — they were independent HTTP calls that happened to
share rate-limit accounting through client-side coalescing. One bad request
should not take fifteen good ones down with it. The fix was to isolate
failures per request inside a batch rather than per batch: a `503` on request
7 of 16 now retries request 7 alone. That restored parity with the naive
loop. The benchmark earned its keep catching this — it's the kind of bug you
don't find by reading the code, because the code's logic ("retry the failed
batch") reads as reasonable until you ask what "the batch" actually is
against a chat API with no batch endpoint.

## The measurement bugs, because I made more than one

The fake batch endpoint above was the first measurement bug: a benchmark
number that looked great and was uncollectable in production. I caught and
corrected it, and the corrected simulated numbers are the ones in the
project's README.

There was a second one, in the live run against the real Groq API. An early
version of the live benchmark ran a naive arm and a gateway arm back to back
on the same free-tier account with a fixed call budget across the whole
invocation (`--max-live-calls`). In one run, the naive arm got cut off partway
through — it had sent 23 of an expected ~40+ requests when the shared
invocation's call budget ran out. The harness recorded the 17 unattempted
prompts as failures with zero accompanying `429`s, because they were never
sent, not rejected. An earlier draft of the benchmark's README read that as
"57.5% success" and "lost 17 of 40 prompts" for the naive arm. That's wrong.
It describes the harness's call budget, not naive's behavior against a real
rate limit. The figure was corrected once caught, and it should never be
quoted — I'm noting it here specifically so it doesn't quietly resurface.

Two unrelated measurement failures, same root cause: a number that looked
newsworthy and hadn't been checked against what actually produced it.

## What actually happened live

Before any of the benchmark work, the live tests turned up something dumber
and more basic: the shipped default model, `llama-3.3-70b-versatile`, did not
appear in Groq's live `/models` listing at all. It had been deprecated. The
default had been wrong since whenever Groq retired it, and nothing caught
that until someone actually called the API. It's now `allam-2-7b`, verified
live, and there's a dedicated test (`test_default_model_is_live`) that reads
the provider's own default via `inspect.signature` and checks it against a
live models listing, specifically so this can't rot silently again.

The second live finding was subtler. Some of Groq's available chat models
(`openai/gpt-oss-20b`, `gpt-oss-120b`, `gpt-oss-safeguard-20b`) are reasoning
models. At a small `max_tokens` — 16, in these tests — a reasoning model
spends its entire token budget on an internal `reasoning` field and returns
an **empty** `content` string with `finish_reason: "length"`. My library was
treating that the same as any other response: it had no way to distinguish "I
was truncated before I said anything" from "I have nothing to say." Those are
completely different failure modes for a caller, and collapsing them is a
real gap. It's why the shipped default model is a conventional non-reasoning
model with no such ambiguity, and it's a flagged, open item rather than
something quietly patched over.

## The honest numbers

Simulated, 200 prompts against a 190 req/min server, median of 5 runs: 429s
drop from 85 to 16, requests sent drop from 275 to 206, wall-clock drops from
14.3s to 6.1s. Success rate is *identical* at 92.5% for both arms — the
library doesn't create capacity that isn't there, it just wastes less effort
finding out the capacity isn't there. Tail latency is worse for the gateway:
p99 of 0.157s naive versus 0.518s gateway, because batching waits and
throttled requests queue instead of failing fast. Run the same comparison at
150 prompts, under the limit, and it's a dead heat on every metric — **below
a provider's rate limit, this library buys you nothing.** If you're not near
a limit, don't add it.

Live, against the real Groq API, free tier, one clean run (a second run was
contaminated by cross-arm account-state bleed and is documented but not used
as a comparison point): naive succeeded on 33/40 prompts (82.5%, after 34 real
429s), the gateway succeeded on 40/40 (100%, zero rejections). That's the
real result. It came at a real cost: 32.5s wall-clock for the gateway versus
7.4s for naive. The gateway was paced under the account's real ~5500
tokens/minute ceiling instead of bursting and eating rejections, and that
buys completeness, not speed. The simulated benchmark's wall-clock advantage
does not reproduce live — live, the gateway is slower, by design, full stop.

## What I'd do differently

I'd write the benchmark's threats-to-validity section before writing the
benchmark, not after seeing a number I liked. Both bugs here — the fake batch
endpoint and the call-budget artifact — were caught by asking "does this
number make sense" after the fact, which means they could just as easily not
have been caught. I'd also run the live A/B with separate API keys per arm
from the start; sharing one account's rate-limit state across arms is what
produced both the ordering confound in an earlier live attempt and the
residual gap in the corrected one, where a gateway run immediately after
another gateway run still picked up 15 real 429s from state the two runs
should not have shared.

None of this makes the library revolutionary. It makes token-bucket pacing
under two dimensions and per-request failure isolation defensible claims
instead of assumed ones, and it turned two real bugs — one a defect in the
code, one a defect in how I measured the code — into a documented record
instead of a quietly fixed diff. That record, more than the library itself,
is what I'd point an interviewer at.
