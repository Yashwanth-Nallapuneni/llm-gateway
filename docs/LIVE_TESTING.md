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
