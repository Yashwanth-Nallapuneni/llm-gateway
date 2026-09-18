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
from collections.abc import Iterable
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
            raise CliError(f"line {lineno}: expected a JSON object, got {type(obj).__name__}")
        if "prompt" not in obj or not isinstance(obj["prompt"], str):
            raise CliError(f"line {lineno}: missing required string field \"prompt\"")
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


def build_provider(args: argparse.Namespace) -> Provider:
    common = _no_none(
        rpm_limit=args.rpm,
        tpm_limit=args.tpm,
        max_concurrency=args.max_concurrency,
    )

    if args.provider == "mock":
        return MockProvider(name="mock", **common)

    env_var = _ENV_VAR_FOR_PROVIDER[args.provider]
    api_key = args.api_key or os.environ.get(env_var)
    if not api_key:
        raise CliError(
            f"no API key for provider {args.provider!r}: "
            f"set {env_var} or pass --api-key"
        )

    try:
        from . import groq_provider, openrouter_provider
    except ImportError as exc:
        raise CliError(
            "the http extra is required for a real provider: "
            "pip install aiollm-gateway[http]"
        ) from exc

    if args.provider == "groq":
        built: Provider = groq_provider(
            api_key,
            **_no_none(model=args.model, **common),
        )
        return built

    # openrouter
    if not args.model:
        raise CliError("--model is required for --provider openrouter")
    built = openrouter_provider(
        api_key,
        model=args.model,
        **common,
    )
    return built


def build_request(item: PromptItem, args: argparse.Namespace) -> LLMRequest:
    kwargs: dict[str, Any] = {"prompt": item.prompt}
    max_tokens = item.max_tokens if item.max_tokens is not None else args.max_tokens
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    model = item.model or args.model
    if model is not None:
        kwargs["model"] = model
    if item.priority is not None:
        kwargs["priority"] = item.priority
    if item.needs_logprobs is not None:
        kwargs["needs_logprobs"] = item.needs_logprobs
    if item.needs_strict_json is not None:
        kwargs["needs_strict_json"] = item.needs_strict_json
    return LLMRequest(**kwargs)


# --------------------------------------------------------------------------
# Cost estimation / guard
# --------------------------------------------------------------------------


def estimate(
    items: list[PromptItem], args: argparse.Namespace, provider: Provider
) -> tuple[int, float]:
    total_tokens = 0
    input_tokens = 0
    output_tokens = 0
    for item in items:
        req = build_request(item, args)
        input_tokens += req.estimated_input_tokens()
        output_tokens += req.max_tokens
        total_tokens += req.estimated_total_tokens()
    cost = provider.estimated_cost(input_tokens, output_tokens)
    return total_tokens, cost


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


async def run_gateway(
    provider: Provider,
    items: list[PromptItem],
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], bool, str]:
    """Drive every prompt through the gateway. Returns (rows, any_failed, report)."""
    batcher = Batcher(
        **_no_none(max_batch_size=args.batch_size, max_wait_ms=args.max_wait_ms)
    )
    retry = RetryPolicy(**_no_none(max_attempts=args.max_attempts))

    store: RunStore | None = None
    if args.store is not None:
        # Opening the file (and reclaiming any `in_flight` rows left by a
        # previous, crashed process -- see store.py) is a handful of quick
        # sqlite statements against what is, at worst, a file with a few
        # tens of thousands of rows. It runs once, before any provider call
        # is even dispatched, so blocking the not-yet-busy event loop for it
        # costs nothing a real sweep would notice.
        store = RunStore(args.store)

    budget: BudgetLedger | None = None
    if args.budget is not None:
        budget = BudgetLedger(
            args.budget, **_no_none(max_tokens_total=args.max_total_tokens)
        )

    gateway = LLMGateway(
        providers=[provider], batcher=batcher, retry=retry, store=store, budget=budget
    )

    # Every prompt gets a row in `rows` no matter how the run ends: a
    # BudgetExceeded (or any other) failure is caught per-prompt by
    # `_submit_one` and turned into an error row, never left to propagate
    # out of `asyncio.gather` and take the results already collected for
    # every other prompt down with it. A ceiling hit at prompt 401 of 1000
    # must not cost the 400 results already in hand.
    rows: list[dict[str, Any]] = []
    any_failed = False
    budget_skipped = 0
    try:
        async with gateway:
            requests = [build_request(item, args) for item in items]
            results = await asyncio.gather(
                *(
                    _submit_one(gateway, item, req)
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
        if store is not None:
            await store.aclose()

    report = gateway.metrics.report()
    if store is not None:
        report += (
            f"\nstore: {gateway.served_from_store} served from store, "
            f"{gateway.freshly_called} freshly called"
        )
    if budget is not None:
        report += (
            f"\nbudget: ${budget.spent:.4f} spent of ${budget.limit_usd:.4f} limit"
        )
        if budget_skipped:
            report += (
                f"; {budget_skipped} of {len(items)} prompt(s) skipped "
                "(budget exceeded)"
            )
    return rows, any_failed, report


# --------------------------------------------------------------------------
# argparse
# --------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="llm-gateway",
        description="Send a file of prompts through llm-gateway without writing Python.",
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )
    subparsers = parser.add_subparsers(dest="command")

    run = subparsers.add_parser(
        "run", help="Run every prompt in INPUT through the gateway."
    )
    run.add_argument(
        "input",
        help="Path to a .jsonl or .txt file of prompts, or - for stdin.",
    )
    run.add_argument("--provider", choices=PROVIDER_CHOICES, default="mock")
    run.add_argument("--api-key", default=None, help="API key (prefer an env var).")
    run.add_argument("--model", default=None)
    run.add_argument("--max-tokens", type=int, default=None)
    run.add_argument("--batch-size", type=int, default=None, dest="batch_size")
    run.add_argument("--max-wait-ms", type=float, default=None, dest="max_wait_ms")
    run.add_argument("--rpm", type=float, default=None)
    run.add_argument("--tpm", type=float, default=None)
    run.add_argument("--max-concurrency", type=int, default=None, dest="max_concurrency")
    run.add_argument("--max-attempts", type=int, default=None, dest="max_attempts")
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
    run.add_argument("--limit", type=int, default=None, help="Only process the first N prompts.")
    run.add_argument("--quiet", action="store_true", help="Suppress progress/guard chatter on stderr.")
    run.add_argument("--no-metrics", action="store_true", help="Do not print the metrics report.")
    run.add_argument("--dry-run", action="store_true", help="Parse and estimate only; no calls.")
    run.add_argument("--yes", action="store_true", help="Skip the cost confirmation prompt.")
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


def _run_command(args: argparse.Namespace, out: TextIO, err: TextIO) -> int:
    _check_store_flags(args)
    _check_budget_flags(args)
    items = read_input(args.input)
    if args.limit is not None:
        items = items[: args.limit]
    if not items:
        raise CliError("no prompts to run: input was empty")

    provider = build_provider(args)

    if args.dry_run:
        total_tokens, cost = estimate(items, args, provider)
        print(f"prompts: {len(items)}", file=err)
        print(f"estimated total tokens: {total_tokens}", file=err)
        if args.provider != "mock":
            print(f"estimated cost: ${cost:.4f}", file=err)
        return 0

    if args.provider != "mock":
        _, cost = estimate(items, args, provider)
        _confirm_cost(cost, args.provider, assume_yes=args.yes, quiet=args.quiet)

    if not args.quiet:
        print(f"running {len(items)} prompt(s) through {args.provider}...", file=err)

    rows, any_failed, report = asyncio.run(run_gateway(provider, items, args))

    if args.output is None or args.output == "-":
        for row in rows:
            out.write(json.dumps(row) + "\n")
    else:
        with open(args.output, "w", encoding="utf-8") as output_stream:
            for row in rows:
                output_stream.write(json.dumps(row) + "\n")

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

    parser.error(f"unknown command {args.command!r}")  # argparse.error exits the process


if __name__ == "__main__":
    raise SystemExit(main())
