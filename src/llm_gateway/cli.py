"""Command-line interface: use the gateway without writing Python.

`llm-gateway run prompts.jsonl` reads a file of prompts, drives them through
`LLMGateway` (batching, rate limiting, retry, failover) exactly as the
library does when called from code, and writes one JSON object per line to
stdout (or `--output FILE`), preserving input order. `gateway.metrics.report()`
goes to stderr by default so `--output -` piping stays clean.

Kept thin on purpose: this module parses arguments, builds an `LLMGateway`
from the library's own public constructors, and formats results. It adds no
new behaviour of its own -- the guarantees (order preservation, per-request
failure isolation, metrics) all come from the library.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any, TextIO

from . import (
    Batcher,
    BudgetExceeded,
    BudgetLedger,
    LLMGateway,
    LLMRequest,
    LLMResponse,
    RetryPolicy,
    RunStore,
    TokenBudgetExceeded,
    __version__,
)
from .providers.base import Provider
from .providers.mock import MockProvider

PROVIDER_CHOICES = ("mock", "groq", "openrouter")

_ENV_VAR_FOR_PROVIDER = {
    "groq": "GROQ_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
}


class CliError(Exception):
    """A usage or configuration problem: report it and exit(2), no traceback.

    Distinguished from a per-prompt provider failure (which is reported as an
    error line in the output and does not abort the run) -- a CliError means
    the run never got a fair chance to start at all.
    """


# --------------------------------------------------------------------------
# Input parsing
# --------------------------------------------------------------------------


@dataclass(slots=True)
class PromptItem:
    """One line of input, normalized. `id` defaults to its position."""

    id: str
    prompt: str
    max_tokens: int | None = None
    priority: int | None = None
    needs_logprobs: bool | None = None
    needs_strict_json: bool | None = None
    model: str | None = None


def _parse_jsonl_lines(lines: Iterable[str]) -> list[PromptItem]:
    items: list[PromptItem] = []
    for lineno, raw in enumerate(lines, start=1):
        line = raw.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            raise CliError(f"invalid JSON on line {lineno}: {exc.msg}") from exc
        if not isinstance(obj, dict):
            raise CliError(
                f"line {lineno}: expected a JSON object, got {type(obj).__name__}"
            )
        if "prompt" not in obj or not isinstance(obj["prompt"], str):
            raise CliError(f'line {lineno}: missing required string field "prompt"')
        items.append(
            PromptItem(
                id=str(obj.get("id", len(items))),
                prompt=obj["prompt"],
                max_tokens=obj.get("max_tokens"),
                priority=obj.get("priority"),
                needs_logprobs=obj.get("needs_logprobs"),
                needs_strict_json=obj.get("needs_strict_json"),
                model=obj.get("model"),
            )
        )
    return items


def _parse_txt_lines(lines: Iterable[str]) -> list[PromptItem]:
    items: list[PromptItem] = []
    for raw in lines:
        line = raw.rstrip("\n").rstrip("\r")
        if not line.strip():
            continue
        items.append(PromptItem(id=str(len(items)), prompt=line))
    return items


def read_input(path: str) -> list[PromptItem]:
    """Read prompts from a `.jsonl` file, a `.txt` file, or stdin (`-`).

    `.jsonl`: one JSON object per line, `{"prompt": ...}` plus optional
    `id`/`max_tokens`/`priority`/`needs_logprobs`/`needs_strict_json`/`model`.
    `.txt`: one prompt per line, nothing else.
    `-`: read stdin; format is auto-detected from the first non-blank line
    (valid JSON object -> jsonl rules, anything else -> one prompt per line).
    """
    if path == "-":
        text = sys.stdin.read()
        lines = text.splitlines()
        first = next((line for line in lines if line.strip()), None)
        if first is not None:
            try:
                looks_jsonl = isinstance(json.loads(first), dict)
            except json.JSONDecodeError:
                looks_jsonl = False
        else:
            looks_jsonl = False
        return _parse_jsonl_lines(lines) if looks_jsonl else _parse_txt_lines(lines)

    if not os.path.isfile(path):
        raise CliError(f"input file not found: {path}")

    try:
        with open(path, encoding="utf-8") as f:
            text = f.read()
    except OSError as exc:
        raise CliError(f"could not read {path}: {exc}") from exc

    if path.endswith(".jsonl"):
        return _parse_jsonl_lines(text.splitlines())
    if path.endswith(".txt"):
        return _parse_txt_lines(text.splitlines())
    raise CliError(f"unrecognized input extension for {path!r}: use .jsonl or .txt")


# --------------------------------------------------------------------------
# Building the gateway
# --------------------------------------------------------------------------


def _no_none(**kwargs: Any) -> dict[str, Any]:
    """Drop None values so unset CLI flags fall through to the library's own
    defaults instead of the CLI silently re-declaring (and risking drifting
    from) them."""
    return {k: v for k, v in kwargs.items() if v is not None}


def _parse_model_map(model_arg: str | None, provider_names: list[str]) -> dict[str, str]:
    """`--model` is either a plain value (single provider only) or, for
    multiple providers, `name=value` pairs separated by commas, e.g.
    `groq=allam-2-7b,openrouter=x/y`. Returns {provider_name: model}."""
    if model_arg is None:
        return {}
    if len(provider_names) == 1:
        return {provider_names[0]: model_arg}
    mapping: dict[str, str] = {}
    for part in model_arg.split(","):
        if "=" not in part:
            raise CliError(
                f"--model {model_arg!r}: with multiple providers, use "
                "name=value pairs, e.g. --model groq=allam-2-7b,openrouter=x/y"
            )
        name, _, value = part.partition("=")
        name = name.strip()
        if name not in provider_names:
            raise CliError(
                f"--model names {args_provider_hint(provider_names)}, got {name!r}"
            )
        mapping[name] = value.strip()
    return mapping


def args_provider_hint(provider_names: list[str]) -> str:
    return "one of " + ", ".join(provider_names)


def build_provider(
    name: str,
    args: argparse.Namespace,
    model: str | None,
    *,
    dry_run: bool = False,
    multi: bool = False,
) -> Provider:
    """Build a single provider by name, using that provider's own API key
    env var and (optionally) its own model from --model.

    `dry_run` means no network call will ever be made for this provider (see
    `--dry-run`), so a missing API key is not fatal -- a placeholder is used
    instead, purely so the provider object can be built for cost estimation.
    `multi` means more than one --provider was given, which changes what a
    couple of error messages should say (--api-key doesn't apply, and
    --model needs the name=value syntax).
    """
    common = _no_none(
        rpm_limit=args.rpm,
        tpm_limit=args.tpm,
        max_concurrency=args.max_concurrency,
    )

    if name == "mock":
        return MockProvider(name="mock", **common)

    env_var = _ENV_VAR_FOR_PROVIDER[name]
    api_key = args.api_key or os.environ.get(env_var)
    if not api_key:
        if dry_run:
            api_key = "dry-run-placeholder"
        elif multi:
            raise CliError(f"no API key for provider {name!r}: set {env_var}")
        else:
            raise CliError(
                f"no API key for provider {name!r}: set {env_var} or pass --api-key"
            )

    try:
        from . import groq_provider, openrouter_provider
    except ImportError as exc:
        raise CliError(
            "the http extra is required for a real provider: "
            "pip install aiollm-gateway[http]"
        ) from exc

    if name == "groq":
        built: Provider = groq_provider(api_key, **_no_none(model=model, **common))
        return built

    # openrouter
    if not model:
        if multi:
            raise CliError(
                "--model is required for --provider openrouter; with multiple "
                "providers use --model openrouter=<model id>"
            )
        raise CliError("--model is required for --provider openrouter")
    built = openrouter_provider(api_key, model=model, **common)
    return built


def build_providers(args: argparse.Namespace) -> list[Provider]:
    """Build every provider named in --provider, in priority order.

    --provider takes a comma-separated list (e.g. "groq,openrouter"); a
    single name keeps working exactly as before. Each provider looks up its
    own API key env var. --api-key and a single-value --model apply only
    when a single provider is given; with multiple providers, --model must
    use name=value pairs (see _parse_model_map) and --api-key is rejected
    outright, since one key cannot serve two different providers' env vars.
    """
    names = [n.strip() for n in args.provider.split(",") if n.strip()]
    if not names:
        raise CliError("--provider: at least one provider name is required")
    for n in names:
        if n not in PROVIDER_CHOICES:
            raise CliError(
                f"--provider: unknown provider {n!r}, choose from {PROVIDER_CHOICES}"
            )

    if len(names) > 1 and args.api_key:
        raise CliError(
            "--api-key can only be used with a single --provider; with "
            "multiple providers each one reads its own env var "
            f"({_ENV_VAR_FOR_PROVIDER})"
        )

    model_map = _parse_model_map(args.model, names)
    multi = len(names) > 1
    dry_run = getattr(args, "dry_run", False)
    return [
        build_provider(n, args, model_map.get(n), dry_run=dry_run, multi=multi)
        for n in names
    ]


def build_request(item: PromptItem, args: argparse.Namespace) -> LLMRequest:
    kwargs: dict[str, Any] = {"prompt": item.prompt}
    max_tokens = item.max_tokens if item.max_tokens is not None else args.max_tokens
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    # args.model is a per-provider map ("groq=x,openrouter=y") when more than
    # one --provider is given, and is not a sensible per-request override in
    # that case -- each provider already got its own model in build_provider.
    single_provider = "," not in args.provider
    model = item.model or (args.model if single_provider else None)
    if model is not None:
        kwargs["model"] = model
    if item.priority is not None:
        kwargs["priority"] = item.priority
    if item.needs_logprobs is not None:
        kwargs["needs_logprobs"] = item.needs_logprobs
    if item.needs_strict_json is not None:
        kwargs["needs_strict_json"] = item.needs_strict_json
    if args.timeout is not None:
        kwargs["timeout_s"] = args.timeout
    return LLMRequest(**kwargs)


# --------------------------------------------------------------------------
# Cost estimation / guard
# --------------------------------------------------------------------------


def estimate(
    items: list[PromptItem], args: argparse.Namespace, providers: list[Provider]
) -> tuple[int, float, str]:
    """Token/cost estimate. With more than one provider, cost is the worst
    case: whichever listed provider is most expensive for this workload.
    Simple and safe -- the real run may end up cheaper if a cheaper
    provider serves most of the traffic, never more expensive than this."""
    total_tokens = 0
    input_tokens = 0
    output_tokens = 0
    for item in items:
        req = build_request(item, args)
        input_tokens += req.estimated_input_tokens()
        output_tokens += req.max_tokens
        total_tokens += req.estimated_total_tokens()
    worst_provider = max(
        providers, key=lambda p: p.estimated_cost(input_tokens, output_tokens)
    )
    cost = worst_provider.estimated_cost(input_tokens, output_tokens)
    return total_tokens, cost, worst_provider.name


def _confirm_cost(
    cost: float, provider_name: str, *, assume_yes: bool, quiet: bool = False
) -> None:
    """The cost guard for non-mock providers.

    Always runs unless `--yes` is given. When stdin is not a TTY there is no
    way to prompt, and proceeding silently is exactly the "skip it
    accidentally" failure mode this guard exists to prevent -- so a
    non-interactive run without `--yes` is refused outright rather than
    auto-confirmed. `--quiet` only silences the acknowledgement line for the
    `--yes` case; it never skips the interactive prompt or the refusal above.
    """
    if assume_yes:
        if not quiet:
            print(
                f"estimated cost: ${cost:.4f} ({provider_name}) -- proceeding (--yes)",
                file=sys.stderr,
            )
        return
    if not sys.stdin.isatty():
        raise CliError(
            f"refusing to call paid provider {provider_name!r} non-interactively "
            f"without --yes (estimated cost: ${cost:.4f})"
        )
    answer = input(
        f"About to call {provider_name!r}. Estimated cost: ${cost:.4f}. Proceed? [y/N] "
    )
    if answer.strip().lower() not in ("y", "yes"):
        raise CliError("aborted: cost not confirmed")


# --------------------------------------------------------------------------
# Running
# --------------------------------------------------------------------------


def _response_to_dict(item: PromptItem, resp: LLMResponse) -> dict[str, Any]:
    return {
        "id": item.id,
        "prompt": item.prompt,
        "text": resp.text,
        "provider": resp.provider,
        "input_tokens": resp.input_tokens,
        "output_tokens": resp.output_tokens,
        "latency_s": resp.latency_s,
        "attempts": resp.attempts,
    }


def _error_to_dict(item: PromptItem, exc: BaseException) -> dict[str, Any]:
    return {
        "id": item.id,
        "prompt": item.prompt,
        "error": str(exc),
        "error_status": getattr(exc, "status", None),
    }


async def _submit_one(
    gateway: LLMGateway, item: PromptItem, request: LLMRequest
) -> tuple[PromptItem, LLMResponse | None, BaseException | None]:
    try:
        resp = await gateway.submit(request)
        return item, resp, None
    except Exception as exc:
        return item, None, exc


class _Progress:
    """A single self-overwriting progress line on stderr.

    Only active when stderr is a real terminal -- writing carriage-return
    updates to a file or a pipe would just leave junk in the output, so this
    is a no-op whenever `stream.isatty()` is false. Updates are throttled to
    a few times a second so a fast run doesn't spend its time repainting a
    line no one can read anyway.
    """

    def __init__(
        self, stream: TextIO, total: int, *, clock: Callable[[], float] | None = None
    ) -> None:
        self._clock = clock or time.monotonic
        self.stream = stream
        self.total = total
        self.enabled = stream.isatty()
        self.ok = 0
        self.failed = 0
        self._started = self._clock()
        self._last_shown = 0.0

    def update(self, *, ok: bool) -> None:
        if ok:
            self.ok += 1
        else:
            self.failed += 1
        if not self.enabled:
            return
        now = self._clock()
        done = self.ok + self.failed
        if now - self._last_shown < 0.2 and done < self.total:
            return
        self._last_shown = now
        width = len(str(self.total))
        elapsed = now - self._started
        self.stream.write(
            f"\r[{done:>{width}}/{self.total}] ok {self.ok}  failed {self.failed}  "
            f"elapsed {elapsed:.1f}s"
        )
        self.stream.flush()

    def finish(self) -> None:
        if self.enabled:
            self.stream.write("\n")
            self.stream.flush()


async def run_gateway(
    providers: list[Provider],
    items: list[PromptItem],
    args: argparse.Namespace,
    err: TextIO = sys.stderr,
) -> tuple[list[dict[str, Any]], bool, str, dict[str, Any]]:
    """Drive every prompt through the gateway. Returns (rows, any_failed, report, metrics)."""
    batcher = Batcher(
        **_no_none(max_batch_size=args.batch_size, max_wait_ms=args.max_wait_ms)
    )
    retry = RetryPolicy(**_no_none(max_attempts=args.max_attempts))

    store: RunStore | None = None
    if args.store is not None:
        # Opening the file (and reclaiming any `in_flight` rows a previous
        # crashed run left behind -- see store.py) is a few quick sqlite
        # statements, done once before any provider call, so it costs
        # nothing a real sweep would notice.
        store = RunStore(args.store)

    budget: BudgetLedger | None = None
    if args.budget is not None:
        budget = BudgetLedger(
            args.budget, **_no_none(max_tokens_total=args.max_total_tokens)
        )

    gateway = LLMGateway(
        providers=providers,
        batcher=batcher,
        retry=retry,
        store=store,
        budget=budget,
        adaptive=args.adaptive,
    )

    # Every prompt gets a row no matter how the run ends: `_submit_one` turns
    # any failure into an error row instead of letting it propagate out of
    # `asyncio.gather` and take every other prompt's result down with it.
    rows: list[dict[str, Any]] = []
    any_failed = False
    budget_skipped = 0
    progress = _Progress(err, len(items)) if not args.quiet else None

    async def _submit_and_track(
        item: PromptItem, req: LLMRequest
    ) -> tuple[PromptItem, LLMResponse | None, BaseException | None]:
        result = await _submit_one(gateway, item, req)
        if progress is not None:
            progress.update(ok=result[2] is None)
        return result

    try:
        async with gateway:
            requests = [build_request(item, args) for item in items]
            results = await asyncio.gather(
                *(
                    _submit_and_track(item, req)
                    for item, req in zip(items, requests, strict=True)
                )
            )
            for item, resp, exc in results:
                if exc is not None:
                    any_failed = True
                    if isinstance(exc, (BudgetExceeded, TokenBudgetExceeded)):
                        budget_skipped += 1
                    rows.append(_error_to_dict(item, exc))
                else:
                    assert resp is not None
                    rows.append(_response_to_dict(item, resp))
    finally:
        if progress is not None:
            progress.finish()
        if store is not None:
            await store.aclose()

    metrics_dict = gateway.metrics.to_dict()
    report = gateway.metrics.report()
    if store is not None:
        report += (
            f"\nstore: {gateway.served_from_store} served from store, "
            f"{gateway.freshly_called} freshly called"
        )
    if budget is not None:
        report += f"\nbudget: ${budget.spent:.4f} spent of ${budget.limit_usd:.4f} limit"
        if budget_skipped:
            report += (
                f"; {budget_skipped} of {len(items)} prompt(s) skipped (budget exceeded)"
            )
    return rows, any_failed, report, metrics_dict


# --------------------------------------------------------------------------
# models
# --------------------------------------------------------------------------


async def _list_models(provider: str, api_key: str) -> list[str]:
    """Return the live model IDs a provider currently offers, sorted."""
    if provider == "mock":
        return ["mock"]
    from . import groq_provider, openrouter_provider
    from .providers.http import OpenAICompatibleClient

    if provider == "groq":
        prov = groq_provider(api_key)
    else:
        prov = openrouter_provider(api_key, model="placeholder")
    client = prov.client
    assert isinstance(client, OpenAICompatibleClient)
    try:
        return await client.list_models()
    finally:
        await client.aclose()


def _models_command(args: argparse.Namespace, out: TextIO) -> int:
    if args.provider == "mock":
        api_key = ""
    else:
        env_var = _ENV_VAR_FOR_PROVIDER[args.provider]
        api_key = os.environ.get(env_var, "")
        if not api_key:
            raise CliError(f"no API key for provider {args.provider!r}: set {env_var}")
    models = asyncio.run(_list_models(args.provider, api_key))
    if args.contains is not None:
        models = [m for m in models if args.contains in m]
    for model in models:
        print(model, file=out)
    return 0


# --------------------------------------------------------------------------
# argparse
# --------------------------------------------------------------------------


RUN_EXAMPLES = """examples:
  llm-gateway run prompts.txt --provider mock
  llm-gateway run prompts.jsonl --provider groq --dry-run
  llm-gateway run prompts.jsonl --provider groq,openrouter \\
      --model groq=allam-2-7b,openrouter=meta-llama/llama-3.1-8b-instruct \\
      --budget 1.00 --store sweep.db --output out.jsonl
  llm-gateway run prompts.jsonl --provider groq --store sweep.db --resume
"""


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="llm-gateway",
        description="Send a file of prompts through llm-gateway without writing Python.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command")

    run = subparsers.add_parser(
        "run",
        help="Run every prompt in INPUT through the gateway.",
        epilog=RUN_EXAMPLES,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    run.add_argument(
        "input",
        help="Path to a .jsonl or .txt file of prompts, or - for stdin.",
    )
    run.add_argument(
        "--provider",
        default="mock",
        help=(
            "One provider, or a comma-separated priority list, e.g. "
            "'groq,openrouter' (routing and failover between them come from "
            "the library). Choices for each entry: " + ", ".join(PROVIDER_CHOICES) + "."
        ),
    )
    run.add_argument(
        "--api-key",
        default=None,
        help="API key (prefer an env var). Only valid with a single --provider.",
    )
    run.add_argument(
        "--model",
        default=None,
        help=(
            "Model name. With a single --provider, a plain value. With "
            "multiple providers, name=value pairs separated by commas, e.g. "
            "'groq=allam-2-7b,openrouter=x/y'."
        ),
    )
    run.add_argument("--max-tokens", type=int, default=None)
    run.add_argument("--batch-size", type=int, default=None, dest="batch_size")
    run.add_argument("--max-wait-ms", type=float, default=None, dest="max_wait_ms")
    run.add_argument("--rpm", type=float, default=None)
    run.add_argument("--tpm", type=float, default=None)
    run.add_argument("--max-concurrency", type=int, default=None, dest="max_concurrency")
    run.add_argument("--max-attempts", type=int, default=None, dest="max_attempts")
    run.add_argument(
        "--adaptive",
        action="store_true",
        help=(
            "Halve the request rate after a 429 and recover slowly; useful "
            "for providers without rate-limit headers such as OpenRouter."
        ),
    )
    run.add_argument(
        "--timeout",
        type=float,
        default=None,
        metavar="SECONDS",
        help="Overall per-prompt timeout in seconds, including queueing and retries.",
    )
    run.add_argument(
        "--budget",
        type=float,
        default=None,
        metavar="USD",
        help=(
            "Spending ceiling in USD for this run. Once committed spend would "
            "cross it, further prompts fail fast with a budget-exceeded error "
            "instead of being sent to the provider; prompts already in flight "
            "are allowed to finish, and every result obtained before the "
            "ceiling was hit is still written out."
        ),
    )
    run.add_argument(
        "--max-total-tokens",
        type=int,
        default=None,
        dest="max_total_tokens",
        help="Optional total token ceiling for this run. Requires --budget.",
    )
    run.add_argument("--output", default=None, help="Write JSONL here instead of stdout.")
    run.add_argument(
        "--limit", type=int, default=None, help="Only process the first N prompts."
    )
    run.add_argument(
        "--quiet", action="store_true", help="Suppress progress/guard chatter on stderr."
    )
    run.add_argument(
        "--no-metrics", action="store_true", help="Do not print the metrics report."
    )
    run.add_argument(
        "--metrics-json",
        default=None,
        dest="metrics_json",
        metavar="PATH",
        help="Write a machine-readable metrics snapshot (JSON) to PATH.",
    )
    run.add_argument(
        "--dry-run", action="store_true", help="Parse and estimate only; no calls."
    )
    run.add_argument(
        "--yes", action="store_true", help="Skip the cost confirmation prompt."
    )
    run.add_argument(
        "--store",
        default=None,
        help=(
            "Path to a SQLite file recording every prompt's outcome. With this "
            "set, a prompt already completed in that file is served from it "
            "instead of calling the provider again -- crash the run and rerun "
            "the same command to pick up where it left off."
        ),
    )
    run.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Required alongside --store when the store file already has rows in "
            "it, to confirm this run is meant to continue that sweep rather than "
            "collide with an unrelated one."
        ),
    )

    models = subparsers.add_parser(
        "models", help="List the model IDs a provider currently offers."
    )
    models.add_argument(
        "--provider",
        required=True,
        choices=PROVIDER_CHOICES,
        help="Which provider to query.",
    )
    models.add_argument(
        "--contains", default=None, help="Only print model IDs containing this substring."
    )

    return parser


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------


def _check_store_flags(args: argparse.Namespace) -> None:
    """`--resume` only means something alongside `--store`, and an existing,
    non-empty store file is ambiguous without it: is this run meant to
    continue that sweep, or did it just reuse a stale path by accident? A
    fresh or absent path needs no confirmation -- there is nothing yet to
    collide with.
    """
    if args.resume and args.store is None:
        raise CliError("--resume requires --store PATH")
    if args.store is None:
        return
    has_existing_rows = os.path.isfile(args.store) and os.path.getsize(args.store) > 0
    if has_existing_rows and not args.resume:
        raise CliError(
            f"--store {args.store!r} already has data in it; pass --resume to "
            "continue that run, or point --store at a fresh path"
        )


def _check_budget_flags(args: argparse.Namespace) -> None:
    if args.max_total_tokens is not None and args.budget is None:
        raise CliError("--max-total-tokens requires --budget USD")
    if args.budget is not None and args.budget <= 0:
        raise CliError(f"--budget must be > 0, got {args.budget}")
    if args.max_total_tokens is not None and args.max_total_tokens <= 0:
        raise CliError(f"--max-total-tokens must be > 0, got {args.max_total_tokens}")


def _check_writable(*paths: str | None) -> None:
    """Fail before any prompt is sent if an output file cannot be written,
    so a run is never paid for and then lost at the end."""
    for path in paths:
        if path is None:
            continue
        parent = os.path.dirname(os.path.abspath(path))
        if not os.path.isdir(parent) or not os.access(parent, os.W_OK):
            raise CliError(f"cannot write {path}: directory missing or not writable")


def _run_command(args: argparse.Namespace, out: TextIO, err: TextIO) -> int:
    _check_store_flags(args)
    _check_budget_flags(args)
    _check_writable(args.output, args.metrics_json)
    items = read_input(args.input)
    if args.limit is not None:
        if args.limit < 0:
            raise CliError(f"--limit must be >= 0, got {args.limit}")
        items = items[: args.limit]
    if not items:
        raise CliError("no prompts to run: input was empty")

    providers = build_providers(args)
    provider_names = [p.name for p in providers]
    requested_names = [n.strip() for n in args.provider.split(",") if n.strip()]
    any_paid = any(name != "mock" for name in requested_names)

    if args.dry_run:
        total_tokens, cost, worst_name = estimate(items, args, providers)
        print(f"prompts: {len(items)}", file=err)
        print(f"estimated total tokens: {total_tokens}", file=err)
        if any_paid:
            if len(providers) > 1:
                print(
                    f"estimated cost: ${cost:.4f} (worst case -- assumes every "
                    f"prompt goes to the most expensive listed provider, {worst_name!r})",
                    file=err,
                )
            else:
                print(f"estimated cost: ${cost:.4f}", file=err)
        return 0

    if any_paid:
        _, cost, worst_name = estimate(items, args, providers)
        _confirm_cost(
            cost,
            worst_name if len(providers) > 1 else provider_names[0],
            assume_yes=args.yes,
            quiet=args.quiet,
        )

    if not args.quiet:
        print(
            f"running {len(items)} prompt(s) through {','.join(provider_names)}...",
            file=err,
        )

    rows, any_failed, report, metrics_dict = asyncio.run(
        run_gateway(providers, items, args, err)
    )

    if args.output is None or args.output == "-":
        for row in rows:
            out.write(json.dumps(row) + "\n")
    else:
        try:
            with open(args.output, "w", encoding="utf-8") as output_stream:
                for row in rows:
                    output_stream.write(json.dumps(row) + "\n")
        except OSError as exc:
            raise CliError(f"could not write {args.output}: {exc}") from exc

    if args.metrics_json is not None:
        try:
            with open(args.metrics_json, "w", encoding="utf-8") as metrics_stream:
                json.dump(metrics_dict, metrics_stream)
        except OSError as exc:
            raise CliError(f"could not write {args.metrics_json}: {exc}") from exc

    if not args.no_metrics:
        print(report, file=err)

    return 1 if any_failed else 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help(sys.stderr)
        return 2

    if args.command == "run":
        try:
            return _run_command(args, sys.stdout, sys.stderr)
        except CliError as exc:
            print(f"llm-gateway: error: {exc}", file=sys.stderr)
            return 2
        except KeyboardInterrupt:
            print("llm-gateway: interrupted", file=sys.stderr)
            return 2

    if args.command == "models":
        try:
            return _models_command(args, sys.stdout)
        except CliError as exc:
            print(f"llm-gateway: error: {exc}", file=sys.stderr)
            return 2

    parser.error(f"unknown command {args.command!r}")  # argparse.error exits the process


if __name__ == "__main__":
    raise SystemExit(main())
