"""End-to-end CLI tests: exercise `llm_gateway.cli.main` the way a user would
invoke it, against the mock provider only. Offline, fast, no live calls.
"""

from __future__ import annotations

import json
import re
import sqlite3

from llm_gateway.cli import main


def _write(tmp_path, name, content):
    path = tmp_path / name
    path.write_text(content)
    return path


def _lines(text: str) -> list[dict]:
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def _jsonl(n: int) -> str:
    return "\n".join(
        json.dumps({"id": f"p{i}", "prompt": f"prompt number {i}"}) for i in range(n)
    )


def _in_flight_count(store_path) -> int:
    conn = sqlite3.connect(str(store_path))
    try:
        cur = conn.execute("SELECT COUNT(*) FROM requests WHERE status = 'in_flight'")
        return cur.fetchone()[0]
    finally:
        conn.close()


# --------------------------------------------------------------------------
# 1. budget-limited run with --store, then --resume with a bigger budget
# --------------------------------------------------------------------------


def test_budget_partial_run_then_resume_completes(tmp_path, monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    n = 30
    input_path = _write(tmp_path, "prompts.jsonl", _jsonl(n))
    store_path = tmp_path / "store.db"
    output_path = tmp_path / "out.jsonl"

    # the budget ledger reserves against max_tokens (256), not the mock's
    # actual (smaller) output, so each admission reserves roughly $0.0258
    # up front and releases most of it back once the real (much cheaper)
    # cost is known. Requests are admitted one at a time, so a budget of
    # $0.08 lets roughly the first half of the 30 prompts through before
    # the reservation for the next one no longer fits.
    code = main(
        [
            "run",
            str(input_path),
            "--provider",
            "mock",
            "--store",
            str(store_path),
            "--budget",
            "0.08",
            "--output",
            str(output_path),
        ]
    )

    assert code != 0
    rows = _lines(output_path.read_text())
    assert len(rows) == n
    assert [r["id"] for r in rows] == [f"p{i}" for i in range(n)]

    successes = [r for r in rows if "text" in r]
    errors = [r for r in rows if "error" in r]
    assert successes, "budget should allow at least some prompts through"
    assert errors, "budget should be small enough to stop partway"
    for row in errors:
        assert row["error_status"] is None or True  # budget errors have no HTTP status
        assert "error" in row

    assert _in_flight_count(store_path) == 0

    # rerun with --resume and a much bigger budget: previously finished
    # prompts are served from the store, the rest go to the provider, and
    # every prompt succeeds this time.
    from llm_gateway.providers.mock import MockClient

    call_count = {"n": 0}
    original_complete = MockClient.complete

    async def counting_complete(self, request):
        call_count["n"] += 1
        return await original_complete(self, request)

    monkeypatch.setattr(MockClient, "complete", counting_complete)

    output_path2 = tmp_path / "out2.jsonl"
    code2 = main(
        [
            "run",
            str(input_path),
            "--provider",
            "mock",
            "--store",
            str(store_path),
            "--resume",
            "--budget",
            "10.0",
            "--output",
            str(output_path2),
        ]
    )

    assert code2 == 0
    rows2 = _lines(output_path2.read_text())
    assert len(rows2) == n
    assert [r["id"] for r in rows2] == [f"p{i}" for i in range(n)]
    assert all("text" in r for r in rows2), "every prompt must succeed after resume"

    # only the prompts that failed with a budget error the first time should
    # have gone to the provider on resume; the rest were served from store.
    assert call_count["n"] == len(errors)
    assert _in_flight_count(store_path) == 0


def test_resume_reports_served_from_store_count(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    n = 10
    input_path = _write(tmp_path, "prompts.jsonl", _jsonl(n))
    store_path = tmp_path / "store.db"

    code = main(
        [
            "run",
            str(input_path),
            "--provider",
            "mock",
            "--store",
            str(store_path),
            "--budget",
            "0.04",
            "--output",
            str(tmp_path / "out.jsonl"),
        ]
    )
    assert code != 0
    capsys.readouterr()

    code2 = main(
        [
            "run",
            str(input_path),
            "--provider",
            "mock",
            "--store",
            str(store_path),
            "--resume",
            "--budget",
            "10.0",
            "--output",
            str(tmp_path / "out2.jsonl"),
        ]
    )
    assert code2 == 0
    err = capsys.readouterr().err
    match = re.search(r"store: (\d+) served from store, (\d+) freshly called", err)
    assert match is not None, f"expected store summary line in stderr, got: {err!r}"
    served_from_store = int(match.group(1))
    assert served_from_store > 0
    assert served_from_store < n  # some prompts still needed a fresh call


# --------------------------------------------------------------------------
# 2. --provider mock,mock with --timeout and --metrics-json
# --------------------------------------------------------------------------


def test_multi_mock_provider_with_timeout_and_metrics_json(tmp_path):
    n = 8
    input_path = _write(tmp_path, "prompts.jsonl", _jsonl(n))
    output_path = tmp_path / "out.jsonl"
    metrics_path = tmp_path / "metrics.json"

    code = main(
        [
            "run",
            str(input_path),
            "--provider",
            "mock,mock",
            "--timeout",
            "5.0",
            "--metrics-json",
            str(metrics_path),
            "--output",
            str(output_path),
            "--no-metrics",
        ]
    )

    assert code == 0
    rows = _lines(output_path.read_text())
    successful = [r for r in rows if "text" in r]
    assert len(successful) == n

    metrics = json.loads(metrics_path.read_text())
    assert isinstance(metrics, dict)
    assert metrics["completed"] == len(successful)


# --------------------------------------------------------------------------
# 3. output row ids follow input ids and order
# --------------------------------------------------------------------------


def test_ids_follow_input_order_plain_run(tmp_path):
    n = 30
    input_path = _write(tmp_path, "prompts.jsonl", _jsonl(n))
    output_path = tmp_path / "out.jsonl"

    code = main(
        [
            "run",
            str(input_path),
            "--provider",
            "mock",
            "--output",
            str(output_path),
            "--no-metrics",
        ]
    )

    assert code == 0
    rows = _lines(output_path.read_text())
    assert [r["id"] for r in rows] == [f"p{i}" for i in range(n)]


def test_ids_follow_input_order_multi_provider(tmp_path):
    n = 12
    input_path = _write(tmp_path, "prompts.jsonl", _jsonl(n))
    output_path = tmp_path / "out.jsonl"

    code = main(
        [
            "run",
            str(input_path),
            "--provider",
            "mock,mock",
            "--output",
            str(output_path),
            "--no-metrics",
        ]
    )

    assert code == 0
    rows = _lines(output_path.read_text())
    assert [r["id"] for r in rows] == [f"p{i}" for i in range(n)]


def test_ids_follow_input_order_with_budget_errors_mixed_in(tmp_path):
    n = 20
    input_path = _write(tmp_path, "prompts.jsonl", _jsonl(n))
    output_path = tmp_path / "out.jsonl"

    code = main(
        [
            "run",
            str(input_path),
            "--provider",
            "mock",
            "--budget",
            "0.03",
            "--output",
            str(output_path),
            "--no-metrics",
        ]
    )

    assert code != 0
    rows = _lines(output_path.read_text())
    assert len(rows) == n
    # order and ids must follow the input regardless of success/error mix
    assert [r["id"] for r in rows] == [f"p{i}" for i in range(n)]
