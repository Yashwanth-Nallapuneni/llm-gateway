# Contributing

## Dev setup

```bash
pip install -e ".[dev,http]"
```

`dev` pulls in pytest, pytest-asyncio, pytest-cov, ruff, and mypy (it
already includes httpx, but `http` is listed explicitly above since it's
the extra you'd install standalone to use the HTTP-backed providers).
Requires Python 3.11+.

## Checks CI runs

See `.github/workflows/ci.yml`. Three jobs, all must pass:

- **test**: `pytest tests/ -q --cov=llm_gateway --cov-report=xml --cov-report=term`,
  across Python 3.11-3.14.
- **lint**: `ruff check .`, `ruff format --check .`, and `mypy`.
- **build**: `python -m build`, then `twine check dist/*`.

Run the same commands locally before opening a PR.

## Live tests

`tests/test_live_groq.py` and `tests/test_live_openrouter.py` make real
network calls against the Groq and OpenRouter APIs. They're excluded by
default (`pyproject.toml` sets `addopts = "-m 'not live'"`) and each test
also self-skips if its API key env var isn't set, so a plain
`pytest tests/ -q` never touches the network. They need a real API key
and are opt-in; see `docs/LIVE_TESTING.md` for how to get a free Groq key
and run them.

## Project principles

- Zero runtime dependencies. `httpx` is optional (the `http` extra) and
  imported lazily, only by the code paths that need it, so importing
  `llm_gateway` without `http` installed still works for anything that
  doesn't touch the network.
- Keep things small and easy to follow. Prefer the straightforward
  implementation over the clever one.
- Every bug fix comes with a test that fails without the fix and passes
  with it.

## Adding a provider

`src/llm_gateway/providers/groq.py` is the model to follow: it's a thin
configuration layer over `OpenAICompatibleClient` in
`src/llm_gateway/providers/http.py`, pinning a base URL and describing
the provider's `ProviderCapabilities` (e.g. which params it doesn't
support). If the provider's API isn't OpenAI-compatible, look at
`src/llm_gateway/providers/base.py` for the `Provider` interface instead.

See `docs/ARCHITECTURE.md` for how a request flows through the gateway
and where a new provider plugs in.
