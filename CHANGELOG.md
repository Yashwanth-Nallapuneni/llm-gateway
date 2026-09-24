# Changelog

All notable changes to this project are documented in this file, in
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) style.

## [0.2.0] - unreleased

### Added

- OpenRouter adapter validated against the real OpenRouter API (previously
  only tested against a fake HTTP server). See
  `docs/LIVE_TESTING.md` for model choice, headers observed and cost.
- Opt-in adaptive rate limiting (`LLMGateway(..., adaptive=True)`, CLI
  `--adaptive`): halves the request rate on a 429 and grows it back slowly,
  for providers such as OpenRouter that send no rate-limit headers to sync
  against.
- CLI multi-provider failover: `--provider groq,openrouter` runs several
  providers under one gateway with the library's existing routing and
  failover, `--model` accepts `name=value` pairs when more than one
  provider is given.
- Per-request timeout: `LLMRequest(timeout_s=...)` and CLI `--timeout`,
  raising `RequestTimeout` on expiry (covers queueing and retries, not just
  the network call).
- `MetricsSink.to_dict()` and CLI `--metrics-json PATH` for a
  machine-readable metrics snapshot alongside the human-readable report.
- CLI progress line on stderr during a run, and
  `examples/timeouts_and_adaptive.py`.
- `llm-gateway models --provider groq|openrouter [--contains TEXT]` lists
  the live model IDs a provider currently offers, sorted, via
  `OpenAICompatibleClient.list_models()` (GET `{base_url}/models`) — a way
  to find a model that hasn't been retired since a default was picked.
- `.github/workflows/live.yml` now also runs the OpenRouter live tests when
  an optional `OPENROUTER_API_KEY` secret is configured on the `live-tests`
  environment (self-skips cleanly without it; the existing 5-test pass
  floor is unaffected either way). See `docs/LIVE_TESTING.md` for setup and
  the roughly half-cent-per-run cost.

### Fixed

- `OpenRouterClient._raise_for_status` no longer notifies rate-limit headers
  twice for the same response when mapping a 402 (out-of-credit) error.
- Live benchmark: a cooldown now applies at every arm boundary, including
  across runs, and a run cut short by the live-call ceiling is excluded
  from the summary instead of averaged in.
- CLI: `run --dry-run` no longer requires an API key, since a dry run makes
  no network calls; the missing-API-key and missing-`--model` messages for
  a multi-provider `--provider` list now match what's actually allowed in
  that case (no mention of `--api-key`, and the `name=value` `--model`
  syntax) instead of the single-provider wording.
- `aclose()` no longer leaves requests that were still queued waiting
  forever; they fail with `GatewayError`. An unexpected error while
  dispatching a batch now reaches that batch's callers instead of being
  lost in a background task.
- A half-open provider that was ranked but not used gets its trial call
  back, so it can no longer stay half-open forever.
- `RetryPolicy.delay_for` no longer raises `OverflowError` for very large
  attempt numbers.
- `Reservation.settle()` with a negative cost or token count no longer
  leaves the reservation stuck; it stays open and can be released.

## [0.1.0] - 2026-09-21

First release. Async gateway with rate limiting, retry with backoff,
batching, capability-aware routing, a circuit breaker, a durable run store,
a budget ledger, and a CLI, with Groq and Mock providers and Groq validated
against the real API.
