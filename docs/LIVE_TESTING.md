# Live testing against the real Groq API

Everything under `tests/` except `tests/test_live_groq.py` runs entirely
offline, against mocks or `httpx.MockTransport`. `test_live_groq.py` is the
one file that makes real network calls to Groq, and it only runs when you
explicitly ask for it.

## Get a free key

1. Sign up at https://console.groq.com (free tier, no card required).
2. Create an API key under "API Keys" in the console.
3. Save it locally:

   ```bash
   echo 'gsk_...' > ~/.groq_key
   chmod 600 ~/.groq_key
   ```

   Never commit this file, never put the key in a repo file, and never paste
   it directly into a shell command that gets logged in your history --
   read it from `~/.groq_key` instead, as below.

## Run the live tests

```bash
GROQ_API_KEY=$(cat ~/.groq_key) python3 -m pytest tests/ -m live -q
```

A plain `python3 -m pytest tests/ -q` never touches the network: this
project's `pyproject.toml` sets `addopts = "-m 'not live'"`, so the `live`
marker is excluded by default. Every test in `test_live_groq.py` also carries
its own `skipif` on `GROQ_API_KEY` being unset, so the file is a no-op (all
skips) for anyone who checks this repo out without a key -- the offline
suite, `ruff`, and `mypy` all stay green either way.

## CI: a green run means the tests actually ran

`.github/workflows/live.yml` (manual `workflow_dispatch`, plus a weekly
Monday canary) runs `pytest -m live -q --junit-xml=live-results.xml` against
the real Groq API. Locally, "all skipped" and "all passed" both look like a
plain green `pytest` exit code -- that ambiguity is exactly what would make
the workflow's badge meaningless from the outside, since the run log itself
isn't publicly readable. So the workflow does not trust pytest's exit code
by itself: a second step parses `live-results.xml` and requires at least
**5** live tests to have actually **passed** (not skipped).

That floor of 5, not 6, accounts for `test_real_429_with_retry_after`, which
is a *permanent*, intentional `pytest.skip()` (see "What each live test
checks" below) -- so a fully successful run looks like 5 passed + 1 skipped,
never 6 passed. If `GROQ_API_KEY` is missing or wrong, every live test
self-skips, `passed` comes back as `0`, and the job fails with:

```
only 0 live test(s) actually passed (need at least 5) -- live tests were
skipped. Is the GROQ_API_KEY secret configured for the 'live-tests'
environment? A green run must mean the tests actually executed against the
real Groq API, not that they self-skipped.
```

The job output also always prints a one-line summary (`total=... passed=...
skipped=... failed=... errored=...` plus the names of any skipped tests) so
a maintainer can see at a glance what happened -- the API key itself is
never printed, echoed, or dumped. If `tests/test_live_groq.py` gains or
loses `@pytest.mark.live` tests, update `MIN_LIVE_PASSED` in the workflow's
"Check live test results actually ran" step to match.

## Why live tests are excluded by default

- They need a real API key and network access, neither of which CI or a
  fresh clone can assume.
- They cost real (if tiny) quota against your Groq account.
- They are inherently non-deterministic: a real API can be slow, rate-limit
  you, or change its available models out from under you (see below).

The offline suite is what proves the library's logic is correct in
isolation; `test_live_groq.py` is what proves that logic actually holds up
against the real thing on the wire.

## Cost

$0 on Groq's free tier. Every live test caps `max_tokens` at 16, and the
whole file makes a small, fixed number of real calls (well under 30) each
time you run it.

## The model-deprecation gotcha

Don't hardcode a Groq model name from memory or from old docs/code -- Groq
retires models regularly, and a name that worked six months ago can now
return an error. Before writing or fixing live tests, ask the API itself:

```bash
GROQ_API_KEY=$(cat ~/.groq_key) curl -s \
  -H "Authorization: Bearer $GROQ_API_KEY" \
  https://api.groq.com/openai/v1/models
```

When this suite was written, `groq_provider`'s advertised default model,
`llama-3.3-70b-versatile`, did **not** appear in that list at all -- it was
dead. `groq_provider`'s `model=` default has since been corrected to
`allam-2-7b` (verified live 2026-09-18, see below), but treat that as a
temporary fact, not a permanent one: `tests/test_live_groq.py` does not rely
on the default staying correct -- every test still passes
`model="allam-2-7b"` explicitly, and a dedicated test,
`test_default_model_is_live`, asserts that whatever `groq_provider` defaults
to right now is still present in a live `/models` listing. That test is
what will fail, loudly, the day this default rots again.

`allam-2-7b` was picked over the other available chat models for two
reasons:

- It is small (7B) and cheap, and Groq's free tier reported generous limits
  for it (7000 requests/min, 6000 tokens/min, observed live).
- Some of the other available chat models (`openai/gpt-oss-20b`,
  `openai/gpt-oss-120b`, `openai/gpt-oss-safeguard-20b`) are reasoning
  models: at `max_tokens=16` they spend the entire token budget on an
  internal `reasoning` field and return an **empty** `content` string with
  `finish_reason: "length"`. That silently breaks a test asserting on
  non-empty output text, for reasons that have nothing to do with the
  library itself. `allam-2-7b` returns real, non-empty content immediately
  at `max_tokens=16`.

(A third candidate, `qwen/qwen3.8-27b`, was also live at the time of
writing and did return non-empty content at `max_tokens=16` in a spot check
-- but Qwen3 is a hybrid reasoning model family that can enable thinking
mode depending on request/version details, so it was not trusted as a
default; `allam-2-7b` is a conventional, non-reasoning chat model with no
such ambiguity.)

If Groq deprecates `allam-2-7b` too, re-run the `curl` command above, pick
another small non-reasoning chat model from the current list, verify it
returns non-empty content at `max_tokens=16`, and update both `LIVE_MODEL`
in `tests/test_live_groq.py` and the `model=` default in
`src/llm_gateway/providers/groq.py` (`groq_provider`'s docstring there spells
out exactly what to check again). `test_default_model_is_live` will catch it
if you forget the second half.

### rpm/tpm defaults

`groq_provider`'s `rpm_limit`/`tpm_limit` defaults (30 / 6,000) predate this
investigation and were never tied to a specific model. Live headers observed
for the current default model, `allam-2-7b`, were much higher: ~7000
requests/min and ~6000 tokens/min. `tpm_limit=6_000` turned out to already be
a reasonable match; `rpm_limit=30` was left deliberately conservative rather
than raised to match `allam-2-7b` specifically, because these two defaults
apply to whatever model a caller passes, and other Groq models (especially
larger/popular ones) have historically reported rpm figures much closer to
the old 30/min number. See the comment above `rpm_limit`/`tpm_limit` in
`groq.py` for the full reasoning. Either way, both numbers are an
approximation -- `_wire_header_sync` syncs the real live `x-ratelimit-*`
headers into the limiter on every response, which is what actually governs
behavior after the first call.

## What each live test checks

- **A real completion end to end** -- through `LLMGateway`, not just the raw
  HTTP client: non-empty text, positive token counts, `provider == "groq"`.
- **The default model is still live** -- reads `groq_provider`'s own `model=`
  default via `inspect.signature` (so the test can't drift from the real
  default) and checks it against a live `GET /openai/v1/models` listing.
  This is the test that would have caught the original
  `llama-3.3-70b-versatile` bug.
- **Rate-limit headers actually move the local buckets** -- the provider is
  configured with a local rpm/tpm far above Groq's real limit, so
  `TokenBucket.sync`'s `min(local, server)` rule (see `rate_limit.py`) can
  only produce a smaller number by actually applying Groq's
  `x-ratelimit-remaining-requests` / `x-ratelimit-remaining-tokens` headers.
  Observed live headers were the standard OpenAI-compatible set:
  `x-ratelimit-limit-requests`, `x-ratelimit-remaining-requests`,
  `x-ratelimit-reset-requests` (Groq's own compound-duration format, e.g.
  `"24.685s"`), and the `-tokens` equivalents.
- **A bad key is a 401, and is not retried** -- uses a deliberately invalid
  literal key, never a mangled version of the real one.
- **A real 429 with a real `Retry-After`** -- attempted, but skipped. Groq's
  free-tier limits for `allam-2-7b` (7000 rpm / 6000 tpm) are high enough
  that reliably provoking a genuine 429 within a small, cost-conscious test
  budget isn't realistic without burning a large slice of the daily quota,
  so this test is skipped with that reasoning rather than hammering the API.
- **The CLI against the real provider** -- runs
  `python3 -m llm_gateway.cli run ... --provider groq --yes --max-tokens 16`
  as a subprocess over a two-prompt file and checks the JSONL output parses
  and contains real text.

# Live testing against the real OpenRouter API

`tests/test_live_openrouter.py` mirrors `test_live_groq.py`: it makes real
calls to OpenRouter, only under `pytest -m live`, only with
`OPENROUTER_API_KEY` set, and every test carries its own `skipif` so the
file is a no-op for anyone without a key.

## Get a key

1. Sign up at https://openrouter.ai and create a key under Settings > Keys.
2. Save it locally, the same way as the Groq key:

   ```bash
   echo 'sk-or-...' > ~/.openrouter_key
   chmod 600 ~/.openrouter_key
   ```

   Never commit this file or paste the key into a command that gets logged
   -- read it from `~/.openrouter_key` instead.

## Run the live tests

```bash
OPENROUTER_API_KEY=$(cat ~/.openrouter_key) python3 -m pytest tests/test_live_openrouter.py -m live -q
```

## Model choice and cost (verified 2026-09-24)

`LIVE_MODEL` is `meta-llama/llama-3.1-8b-instruct`, a cheap paid model
(observed pricing: $0.05 / M input tokens, $0.08 / M output tokens), not one
of OpenRouter's free (`:free`-suffixed) models. That was a deliberate choice
after checking the free models actually available that day: almost all of
them (`liquid/lfm-2.5-2.6b:free`, `nvidia/nemotron-3.5-lightning:free`,
`cohere/north-mini-code:free`, `qwen/qwen3.8-27b:free`, and others) are
reasoning models that spend a 16-token budget on an internal `reasoning`
field and come back with `content: null`, and a couple
(`google/gemma-4-26b-a4b-it:free`, `poolside/laguna-xs-2.1:free`) returned a
real HTTP 429 from their shared free-tier pool on an ordinary request.
`meta-llama/llama-3.1-8b-instruct` is a conventional, non-reasoning model
that supports logprobs and returns clean content immediately at
`max_tokens=16`; its cost for one 16-token completion was observed at
$0.00000046. The whole exploration plus the live test file together moved
account usage from $0.388871 to $0.393383 -- about $0.0045 total, well
inside the $1.00 budget for this work.

`LIVE_REASONING_FREE_MODEL` is `liquid/lfm-2.5-2.6b:free`, used
specifically to exercise the empty-content/`was_truncated` path against a
real reasoning model, the same trap documented in
`src/llm_gateway/providers/groq.py`'s `GroqClient` docstring for Groq's
`openai/gpt-oss-*` models. Observed live: `content: null`,
`finish_reason: "length"` at `max_tokens=16`.

## Headers observed

Verified live 2026-09-24: OpenRouter's `/chat/completions` responses do
**not** include any `x-ratelimit-*` headers -- only `x-generation-id`,
standard CORS/transport headers, and Cloudflare's own headers. This is a
real difference from Groq, which sends
`x-ratelimit-{limit,remaining,reset}-{requests,tokens}` on every response.
`sync_from_headers` (rate_limit.py) already handles this correctly: it
looks for specific header names and is a no-op when they're absent, so an
OpenRouter provider's local `rpm_limit`/`tpm_limit` configuration keeps
governing pacing for the life of the process, never getting corrected by
the server the way a Groq provider's does.
`test_rate_limit_headers_sync_is_a_safe_no_op` in `test_live_openrouter.py`
guards this: it asserts the bucket's *capacity* is untouched by a real call
(only its current level drops from normal consumption), which is what would
break if a future change to `sync_from_headers` assumed every
OpenAI-compatible provider sends these headers.

## What each live test checks

- **A real completion end to end** -- through `LLMGateway`, non-empty text,
  positive token counts, `provider == "openrouter"`, and that
  `was_truncated` agrees with `finish_reason`.
- **Rate-limit header sync is a safe no-op** -- see "Headers observed"
  above.
- **A bad model id is a 400, and is not retried** -- OpenRouter reports an
  unrecognized model id as HTTP 400 (`"... is not a valid model ID"`),
  unlike Groq's 401 for a bad key; both are correctly non-retryable.
- **A bad key is a 401, and is not retried** -- a deliberately invalid
  literal key, never a mangled version of the real one.
- **Logprobs round-trip on a supporting model** -- `needs_logprobs=True`
  against `meta-llama/llama-3.1-8b-instruct` returns real per-token logprob
  values; `require_parameters` (forced on automatically by
  `OpenRouterClient.extra_body_params` whenever a request needs logprobs or
  strict JSON -- see `src/llm_gateway/providers/openrouter.py`) does not
  itself break an ordinary supporting model.
- **`require_parameters` rejects a model without logprobs support** --
  `amazon/nova-micro-v1` (verified live to not list `logprobs` in its
  `supported_parameters`) with `needs_logprobs=True` comes back as a
  routing failure (`"No endpoints found that can handle the requested
  parameters"`, HTTP 404) rather than a silent 200 missing the logprobs
  field. This is the whole point of `require_parameters` -- see the module
  docstring in `openrouter.py`.
- **A free reasoning model can return empty content** -- see "Model choice
  and cost" above; proves `was_truncated`/`finish_reason` survive a real
  OpenRouter response the same way they do for Groq's reasoning models.
- **The CLI against the real provider** -- runs
  `python3 -m llm_gateway.cli run ... --provider openrouter --yes
  --max-tokens 16 --budget 0.05` as a subprocess over a two-prompt file and
  checks the JSONL output parses and contains real text.

## Defect found and fixed during this investigation

`OpenRouterClient._raise_for_status` (in `src/llm_gateway/providers/openrouter.py`)
used to call `self._notify_headers(response)` a second time when mapping a
402 (out of credit) response, on top of the call `OpenAICompatibleClient.complete`
already makes for every response before `_raise_for_status` runs. That
double-invoked `on_headers` -- and therefore the limiter's
`sync_from_headers` -- twice for the same response. The extra call has been
removed; `tests/test_http_provider.py::test_openrouter_402_notifies_headers_exactly_once`
is a regression test for it (offline, via `httpx.MockTransport`, since
provoking a real 402 live would mean deliberately draining account credit).
