"""Tests against the real OpenRouter API.

Everything else in this suite runs offline (mocks, httpx.MockTransport). This
file is the only place besides test_live_groq.py that makes real network
calls, and only when explicitly asked for: `pytest -m live`, with
`OPENROUTER_API_KEY` set in the environment. A plain `pytest` run never
touches this file's network path -- `addopts = "-m 'not live'"` in
pyproject.toml excludes it by default, and every test below also carries its
own `skipif` so the suite stays green for anyone who checks this repo out
without a key.

Model choice: see the "OpenRouter" section of docs/LIVE_TESTING.md.
`LIVE_MODEL` (meta-llama/llama-3.1-8b-instruct) is a cheap paid model, not a
free one -- OpenRouter's `:free` models observed at the time of writing were
almost all reasoning models that burn a small `max_tokens` budget on an
internal `reasoning` field and return empty `content` (see
`test_free_reasoning_model_can_return_empty_content` below), and several
returned upstream 429s from their shared free pool on an ordinary request.
`meta-llama/llama-3.1-8b-instruct` is a conventional non-reasoning model,
supports logprobs, and costs a fraction of a cent for this whole file's
calls (observed: ~$0.0000005 per 16-token completion).

Cost discipline: every request below caps `max_tokens` at 16, and the whole
file makes well under 20 real calls. See docs/LIVE_TESTING.md for the
observed spend.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from llm_gateway import LLMGateway, LLMRequest, ProviderError, RetryPolicy
from llm_gateway.providers.openrouter import openrouter_provider

# Conventional (non-reasoning) chat model, cheap, supports logprobs. Verified
# live 2026-09-24 -- see docs/LIVE_TESTING.md if this needs replacing.
LIVE_MODEL = "meta-llama/llama-3.1-8b-instruct"

# A free-tier model that spends its token budget on an internal `reasoning`
# field before emitting `content` -- used only to exercise the
# empty-content/was_truncated path. Free, so it costs nothing beyond the
# small chance of an upstream 429 from the shared free pool (harmless: the
# test just wants to see either behavior tied to a real request).
LIVE_REASONING_FREE_MODEL = "liquid/lfm-2.5-2.6b:free"

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")

requires_key = pytest.mark.skipif(
    not OPENROUTER_API_KEY,
    reason="OPENROUTER_API_KEY is not set; export it to run live OpenRouter tests",
)


@pytest.mark.live
@requires_key
async def test_real_completion_end_to_end() -> None:
    """A real prompt, through the full gateway stack, gets a real answer."""
    assert OPENROUTER_API_KEY is not None  # for mypy; requires_key already checked this
    provider = openrouter_provider(OPENROUTER_API_KEY, model=LIVE_MODEL)
    gateway = LLMGateway(providers=[provider])
    async with gateway:
        response = await gateway.submit(
            LLMRequest(prompt="Reply with the single word: hello", max_tokens=16)
        )

    assert response.text.strip() != ""
    assert response.provider == "openrouter"
    assert response.input_tokens > 0
    assert response.output_tokens > 0
    assert response.finish_reason in ("stop", "length")
    assert response.was_truncated == (response.finish_reason == "length")


@pytest.mark.live
@requires_key
async def test_rate_limit_headers_sync_is_a_safe_no_op() -> None:
    """OpenRouter, unlike Groq, does not send `x-ratelimit-*` response
    headers on chat completions (observed live 2026-09-24: only
    `x-generation-id` and standard transport headers came back -- no
    `x-ratelimit-limit-requests` or similar). `sync_from_headers` must not
    crash or corrupt the local buckets just because the headers it looks
    for are absent; the local configured limits keep governing pacing.

    This is a real behavioral difference from Groq worth guarding: a future
    change to `sync_from_headers` that assumed every OpenAI-compatible
    provider sends these headers would silently do nothing here, which is
    correct, but a change that raised on missing headers would break every
    OpenRouter call.
    """
    assert OPENROUTER_API_KEY is not None
    provider = openrouter_provider(
        OPENROUTER_API_KEY, model=LIVE_MODEL, rpm_limit=60, tpm_limit=100_000
    )
    gateway = LLMGateway(providers=[provider])
    before = provider.limiter.requests.capacity
    async with gateway:
        await gateway.submit(LLMRequest(prompt="Reply with the word: ok", max_tokens=16))

    # No server-reported limit exists to sync in, so the bucket's capacity
    # (not just its current level) must be unchanged -- only local
    # consumption from the one call above should have happened.
    assert provider.limiter.requests.capacity == before
    assert provider.limiter.requests.available < before  # normal consumption
    assert provider.limiter.requests.available > before - 2  # not corrupted to near-zero


@pytest.mark.live
@requires_key
async def test_bad_model_id_is_400_and_not_retried() -> None:
    """A model id OpenRouter doesn't recognize surfaces as ProviderError(400)
    and RetryPolicy correctly refuses to retry it."""
    assert OPENROUTER_API_KEY is not None
    bad_provider = openrouter_provider(
        OPENROUTER_API_KEY, model="not-a-real-vendor/does-not-exist"
    )
    gateway = LLMGateway(providers=[bad_provider])
    with pytest.raises(ProviderError) as exc_info:
        async with gateway:
            await gateway.submit(LLMRequest(prompt="hello", max_tokens=16))

    assert exc_info.value.status == 400
    assert RetryPolicy().should_retry(exc_info.value, 0) is False


@pytest.mark.live
@requires_key
async def test_bad_key_is_401_and_not_retried() -> None:
    """An invalid key surfaces as ProviderError(status=401) and is not
    retried -- a 401 means the request is wrong, not that the server is
    momentarily unavailable."""
    bad_provider = openrouter_provider("sk-or-invalid-not-a-real-key", model=LIVE_MODEL)
    gateway = LLMGateway(providers=[bad_provider])
    with pytest.raises(ProviderError) as exc_info:
        async with gateway:
            await gateway.submit(LLMRequest(prompt="hello", max_tokens=16))

    assert exc_info.value.status == 401
    assert RetryPolicy().should_retry(exc_info.value, 0) is False


@pytest.mark.live
@requires_key
async def test_logprobs_round_trip_on_a_supporting_model() -> None:
    """`needs_logprobs=True` against a model that actually supports logprobs
    (LIVE_MODEL does) comes back with real per-token logprob values, and
    `require_parameters` (set automatically by OpenRouterClient whenever a
    request needs logprobs or strict JSON -- see openrouter.py) does not
    itself break an ordinary supporting model."""
    assert OPENROUTER_API_KEY is not None
    provider = openrouter_provider(OPENROUTER_API_KEY, model=LIVE_MODEL)
    gateway = LLMGateway(providers=[provider])
    async with gateway:
        response = await gateway.submit(
            LLMRequest(
                prompt="Reply with the single word: hello",
                max_tokens=8,
                needs_logprobs=True,
            )
        )

    assert response.logprobs is not None
    assert len(response.logprobs) > 0


@pytest.mark.live
@requires_key
async def test_require_parameters_rejects_model_without_logprobs() -> None:
    """Requesting logprobs from a model that doesn't support them, with
    `require_parameters` forced on, is a routing failure (no upstream
    honors every requested parameter) rather than a silently-dropped
    field. This is the whole point of `require_parameters`: without it,
    OpenRouter could route to an endpoint that just ignores `logprobs` and
    return a normal 200 missing the data -- with it, the gap becomes a
    normal, catchable ProviderError instead.

    `amazon/nova-micro-v1` was verified live 2026-09-24 to not list
    `logprobs` among its `supported_parameters` in OpenRouter's `/models`
    listing.
    """
    assert OPENROUTER_API_KEY is not None
    provider = openrouter_provider(OPENROUTER_API_KEY, model="amazon/nova-micro-v1")
    gateway = LLMGateway(providers=[provider])
    with pytest.raises(ProviderError) as exc_info:
        async with gateway:
            await gateway.submit(
                LLMRequest(prompt="hi", max_tokens=8, needs_logprobs=True)
            )

    # OpenRouter reports this as "no endpoints found" (404) rather than a
    # 400 -- both are non-retryable, which is what actually matters here.
    assert exc_info.value.status in (400, 404, 422)
    assert RetryPolicy().should_retry(exc_info.value, 0) is False


@pytest.mark.live
@requires_key
async def test_free_reasoning_model_can_return_empty_content() -> None:
    """Some OpenRouter models (including several free-tier ones) spend part
    of a small `max_tokens` budget on an internal `reasoning` field before
    emitting `content`, the same trap documented for Groq's reasoning
    models in providers/groq.py. `LLMResponse.finish_reason` /
    `was_truncated` are what let a caller tell "truncated by reasoning"
    apart from "model legitimately produced nothing" -- this test proves
    that distinction survives a real OpenRouter response.

    `liquid/lfm-2.5-2.6b:free` was observed live 2026-09-24 returning
    `content: null` with `finish_reason: "length"` at `max_tokens=16`. If
    OpenRouter's routing changes what backs this free id and it stops
    reproducing the trap, this test still passes either way (it does not
    assert content is empty, only that was_truncated matches finish_reason
    honestly) -- see the assertions below.
    """
    assert OPENROUTER_API_KEY is not None
    provider = openrouter_provider(
        OPENROUTER_API_KEY,
        model=LIVE_REASONING_FREE_MODEL,
        rpm_limit=20,
        tpm_limit=20_000,
    )
    gateway = LLMGateway(providers=[provider])
    async with gateway:
        response = await gateway.submit(
            LLMRequest(prompt="Reply with the single word: hello", max_tokens=16)
        )

    # The one thing that must always hold: was_truncated is derived purely
    # from finish_reason, never from whether text happens to be empty.
    assert response.was_truncated == (response.finish_reason == "length")
    if response.text.strip() == "":
        assert response.was_truncated, (
            "got empty content without finish_reason == 'length' -- either "
            "the free model changed behavior or something upstream of "
            "was_truncated broke"
        )


@pytest.mark.live
@requires_key
def test_cli_runs_against_real_openrouter() -> None:
    """The CLI's `run` subcommand, driven exactly as a user would, against
    the real provider."""
    assert OPENROUTER_API_KEY is not None
    with tempfile.TemporaryDirectory() as tmp_dir:
        prompts_path = Path(tmp_dir) / "prompts.txt"
        prompts_path.write_text(
            "Reply with the single word: sun\nReply with the single word: moon\n"
        )

        repo_root = Path(__file__).resolve().parent.parent
        env = dict(os.environ)
        env["OPENROUTER_API_KEY"] = OPENROUTER_API_KEY
        env["PYTHONPATH"] = str(repo_root / "src")

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "llm_gateway.cli",
                "run",
                str(prompts_path),
                "--provider",
                "openrouter",
                "--model",
                LIVE_MODEL,
                "--yes",
                "--max-tokens",
                "16",
                "--budget",
                "0.05",
            ],
            capture_output=True,
            text=True,
            env=env,
            timeout=60,
        )

    assert result.returncode == 0, result.stderr

    lines = [line for line in result.stdout.splitlines() if line.strip()]
    assert len(lines) == 2
    for line in lines:
        row = json.loads(line)
        assert "error" not in row
        assert row["provider"] == "openrouter"
        assert row["text"].strip() != ""


# asyncio_mode = "auto" (pyproject.toml) runs the `async def` tests above as
# coroutines automatically; this sanity check is here only so a stray
# `asyncio` import isn't flagged as unused by a linter that doesn't know
# that.
def test_asyncio_mode_sanity() -> None:
    assert asyncio.iscoroutinefunction(test_real_completion_end_to_end)
