"""Tests against the real Groq API.

Everything else in this suite runs offline (mocks, httpx.MockTransport). This
file is the only place that makes real network calls, and it only does so
when explicitly asked for: `pytest -m live`, with `GROQ_API_KEY` set in the
environment. A plain `pytest` run never touches this file's network path --
`addopts = "-m 'not live'"` in pyproject.toml excludes it by default, and
every test below also carries its own `skipif` so the suite stays green for
anyone who checks this repo out without a key.

Model choice: see docs/LIVE_TESTING.md for how `allam-2-7b` was picked and
why the package's previously-advertised default (`llama-3.3-70b-versatile`)
turned out to be dead -- `groq_provider`'s default has since been corrected
to `allam-2-7b` too, but every test here still passes `model=LIVE_MODEL`
explicitly rather than relying on the default, both to control cost/behaviour
and because *any* hardcoded default can rot the moment Groq retires a model
-- see `test_default_model_is_live` below, which exists specifically to
catch that.

Cost discipline: every request below caps `max_tokens` at 16, and the whole
file makes well under 30 real calls, to stay inside Groq's free tier.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import httpx
import pytest

from llm_gateway import LLMGateway, LLMRequest, ProviderError, RetryPolicy
from llm_gateway.providers.groq import GROQ_BASE_URL, groq_provider

LIVE_MODEL = "allam-2-7b"

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

requires_key = pytest.mark.skipif(
    not GROQ_API_KEY,
    reason="GROQ_API_KEY is not set; export it to run live Groq tests",
)


@pytest.mark.live
@requires_key
async def test_real_completion_end_to_end() -> None:
    """A real prompt, through the full gateway stack, gets a real answer."""
    assert GROQ_API_KEY is not None  # for mypy; requires_key already checked this
    provider = groq_provider(GROQ_API_KEY, model=LIVE_MODEL)
    gateway = LLMGateway(providers=[provider])
    async with gateway:
        response = await gateway.submit(
            LLMRequest(prompt="Reply with the single word: hello", max_tokens=16)
        )

    assert response.text.strip() != ""
    assert response.provider == "groq"
    assert response.input_tokens > 0
    assert response.output_tokens > 0


@pytest.mark.live
@requires_key
async def test_default_model_is_live() -> None:
    """The model `groq_provider` defaults to, if nobody overrides it, must
    actually exist on Groq right now.

    This is the test that would have caught the original bug: the package
    shipped with `model: str = "llama-3.3-70b-versatile"` as the default for
    `groq_provider()`, but that model had been retired from Groq entirely --
    anyone calling `groq_provider(api_key)` with no explicit `model=` got a
    hard failure. Every other test in this file sidesteps that risk by
    passing `model=LIVE_MODEL` explicitly; this one instead reads the
    default straight off `groq_provider`'s own signature (so it can't
    silently drift from the test) and checks it against a live
    `GET /openai/v1/models` listing.
    """
    assert GROQ_API_KEY is not None
    default_model = inspect.signature(groq_provider).parameters["model"].default
    assert isinstance(default_model, str) and default_model, (
        "groq_provider's `model` parameter has no plain string default to check"
    )

    async with httpx.AsyncClient(
        base_url=GROQ_BASE_URL, headers={"Authorization": f"Bearer {GROQ_API_KEY}"}
    ) as http_client:
        response = await http_client.get("/models")
    response.raise_for_status()
    live_model_ids = {entry["id"] for entry in response.json()["data"]}

    assert default_model in live_model_ids, (
        f"groq_provider's default model {default_model!r} does not appear in "
        f"Groq's live /models listing -- it has likely been deprecated/removed. "
        f"Pick a new default (small, non-reasoning, verified to return non-empty "
        f"content at max_tokens=16 -- see docs/LIVE_TESTING.md) and update both "
        f"groq_provider's `model=` default and LIVE_MODEL above."
    )


@pytest.mark.live
@requires_key
async def test_rate_limit_headers_sync_local_buckets() -> None:
    """After a real call, the local bucket reflects Groq's own
    x-ratelimit-remaining-requests header, not our locally-configured guess.

    This is the seam proven so far only against a fake server in
    test_http_provider.py (a synthetic MockTransport response with hand-set
    headers). Here the header comes from Groq itself.

    `TokenBucket.sync` deliberately only ever tightens a bucket (it takes
    `min(local, server)` -- see rate_limit.py), so a locally-configured
    limit lower than the server's real limit would never move. To actually
    observe the sync, the local rpm/tpm are configured deliberately high
    (far above Groq's real free-tier limit for this model, observed at 7000
    rpm / 6000 tpm for allam-2-7b) so the server's smaller "remaining" number
    is what wins the min().
    """
    assert GROQ_API_KEY is not None
    provider = groq_provider(
        GROQ_API_KEY, model=LIVE_MODEL, rpm_limit=100_000, tpm_limit=1_000_000
    )
    gateway = LLMGateway(providers=[provider])
    async with gateway:
        await gateway.submit(LLMRequest(prompt="Reply with the word: ok", max_tokens=16))

    # If this fails, either Groq stopped sending x-ratelimit-remaining-requests
    # on this plan, or its real limit for this model now exceeds the
    # deliberately-inflated local default above -- report that rather than
    # loosening the assertion.
    assert provider.limiter.requests.available < 100_000, (
        "requests bucket is still at (or near) the locally-configured "
        "capacity; sync_from_headers does not appear to have applied "
        "Groq's x-ratelimit-remaining-requests header"
    )
    assert provider.limiter.tokens.available < 1_000_000, (
        "tokens bucket is still at (or near) the locally-configured "
        "capacity; sync_from_headers does not appear to have applied "
        "Groq's x-ratelimit-remaining-tokens header"
    )


@pytest.mark.live
@requires_key
async def test_bad_key_is_401_and_not_retried() -> None:
    """An invalid key surfaces as ProviderError(status=401) and RetryPolicy
    correctly refuses to retry it -- a 401 means the request is wrong, not
    that the server is momentarily unavailable."""
    bad_provider = groq_provider("sk-invalid-not-a-real-key", model=LIVE_MODEL)
    gateway = LLMGateway(providers=[bad_provider])
    with pytest.raises(ProviderError) as exc_info:
        async with gateway:
            await gateway.submit(LLMRequest(prompt="hello", max_tokens=16))

    assert exc_info.value.status == 401
    assert RetryPolicy().should_retry(exc_info.value, 0) is False


@pytest.mark.live
@requires_key
async def test_real_429_with_retry_after() -> None:
    """Try to provoke a genuine 429 by bursting requests past the free-tier
    per-minute limit, and check the Retry-After Groq sends back actually
    parses.

    Groq's smallest text models sit at several thousand requests/minute on
    the free tier (observed: 7000 rpm / 6000 tpm for allam-2-7b), so
    reliably provoking a 429 within the ~30-call budget for this whole file
    is not realistic without either a much tighter model or burning a
    meaningful slice of the daily quota. Rather than hammer the API to force
    one, this test is skipped with the concrete numbers observed -- see
    docs/LIVE_TESTING.md.
    """
    pytest.skip(
        "Groq's free-tier limits for allam-2-7b (7000 rpm / 6000 tpm, observed "
        "live) are too high to provoke a genuine 429 within this suite's "
        "~30-call budget without burning a large slice of the daily quota. "
        "Not attempted; see docs/LIVE_TESTING.md."
    )


@pytest.mark.live
@requires_key
def test_cli_runs_against_real_groq() -> None:
    """The CLI's `run` subcommand, driven exactly as a user would, against
    the real provider."""
    assert GROQ_API_KEY is not None
    with tempfile.TemporaryDirectory() as tmp_dir:
        prompts_path = Path(tmp_dir) / "prompts.txt"
        prompts_path.write_text(
            "Reply with the single word: sun\nReply with the single word: moon\n"
        )

        repo_root = Path(__file__).resolve().parent.parent
        env = dict(os.environ)
        env["GROQ_API_KEY"] = GROQ_API_KEY
        env["PYTHONPATH"] = str(repo_root / "src")

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "llm_gateway.cli",
                "run",
                str(prompts_path),
                "--provider",
                "groq",
                "--model",
                LIVE_MODEL,
                "--yes",
                "--max-tokens",
                "16",
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
        assert row["provider"] == "groq"
        assert row["text"].strip() != ""


# asyncio_mode = "auto" (pyproject.toml) runs the `async def` tests above as
# coroutines automatically; this sanity check is here only so a stray
# `asyncio` import isn't flagged as unused by a linter that doesn't know
# that.
def test_asyncio_mode_sanity() -> None:
    assert asyncio.iscoroutinefunction(test_real_completion_end_to_end)
