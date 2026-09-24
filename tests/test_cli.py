"""CLI tests. Everything here runs offline against the mock provider."""

from __future__ import annotations

import json

import pytest

from llm_gateway import __version__
from llm_gateway.cli import main


def _write(tmp_path, name, content):
    path = tmp_path / name
    path.write_text(content)
    return path


def _lines(text: str) -> list[dict]:
    return [json.loads(line) for line in text.splitlines() if line.strip()]


# --------------------------------------------------------------------------
# happy path
# --------------------------------------------------------------------------


def test_happy_path_writes_output_file(tmp_path, capsys):
    input_path = _write(
        tmp_path,
        "prompts.jsonl",
        "\n".join(
            json.dumps({"id": f"p{i}", "prompt": f"hello {i}"}) for i in range(5)
        ),
    )
    output_path = tmp_path / "out.jsonl"

    code = main(
        ["run", str(input_path), "--provider", "mock", "--output", str(output_path)]
    )

    assert code == 0
    rows = _lines(output_path.read_text())
    assert len(rows) == 5
    for i, row in enumerate(rows):
        assert row["id"] == f"p{i}"
        assert row["prompt"] == f"hello {i}"
        assert row["provider"] == "mock"
        assert "text" in row
        assert row["attempts"] >= 1

    captured = capsys.readouterr()
    assert "LLM GATEWAY METRICS" in captured.err


def test_order_preserved(tmp_path):
    input_path = _write(
        tmp_path,
        "prompts.jsonl",
        "\n".join(json.dumps({"prompt": f"item {i}"}) for i in range(20)),
    )
    output_path = tmp_path / "out.jsonl"

    code = main(
        ["run", str(input_path), "--output", str(output_path), "--no-metrics"]
    )

    assert code == 0
    rows = _lines(output_path.read_text())
    assert [r["prompt"] for r in rows] == [f"item {i}" for i in range(20)]
    assert [r["id"] for r in rows] == [str(i) for i in range(20)]


# --------------------------------------------------------------------------
# input formats
# --------------------------------------------------------------------------


def test_txt_input(tmp_path):
    input_path = _write(tmp_path, "prompts.txt", "first prompt\nsecond prompt\n")
    output_path = tmp_path / "out.jsonl"

    code = main(
        ["run", str(input_path), "--output", str(output_path), "--no-metrics"]
    )

    assert code == 0
    rows = _lines(output_path.read_text())
    assert [r["prompt"] for r in rows] == ["first prompt", "second prompt"]


def test_stdin_input(tmp_path, monkeypatch, capsys):
    import io

    monkeypatch.setattr(
        "sys.stdin", io.StringIO("stdin prompt one\nstdin prompt two\n")
    )
    output_path = tmp_path / "out.jsonl"

    code = main(["run", "-", "--output", str(output_path), "--no-metrics"])

    assert code == 0
    rows = _lines(output_path.read_text())
    assert [r["prompt"] for r in rows] == ["stdin prompt one", "stdin prompt two"]


def test_limit(tmp_path):
    input_path = _write(
        tmp_path,
        "prompts.jsonl",
        "\n".join(json.dumps({"prompt": f"item {i}"}) for i in range(10)),
    )
    output_path = tmp_path / "out.jsonl"

    code = main(
        [
            "run",
            str(input_path),
            "--output",
            str(output_path),
            "--limit",
            "3",
            "--no-metrics",
        ]
    )

    assert code == 0
    rows = _lines(output_path.read_text())
    assert len(rows) == 3


# --------------------------------------------------------------------------
# dry run
# --------------------------------------------------------------------------


def test_dry_run_makes_no_calls(tmp_path, capsys):
    input_path = _write(
        tmp_path,
        "prompts.jsonl",
        "\n".join(json.dumps({"prompt": f"item {i}"}) for i in range(4)),
    )

    code = main(["run", str(input_path), "--dry-run"])

    assert code == 0
    captured = capsys.readouterr()
    assert "prompts: 4" in captured.err
    assert "estimated total tokens" in captured.err
    # dry-run must never produce result rows.
    assert captured.out == ""


def test_dry_run_shows_cost_for_paid_provider(tmp_path, monkeypatch, capsys):
    input_path = _write(tmp_path, "prompts.jsonl", json.dumps({"prompt": "hi"}))
    monkeypatch.setenv("GROQ_API_KEY", "test-key-value")

    code = main(["run", str(input_path), "--provider", "groq", "--dry-run"])

    assert code == 0
    captured = capsys.readouterr()
    assert "estimated cost" in captured.err
    assert "test-key-value" not in captured.err
    assert "test-key-value" not in captured.out


# --------------------------------------------------------------------------
# missing API key
# --------------------------------------------------------------------------


def test_missing_api_key_exits_2(tmp_path, monkeypatch, capsys):
    input_path = _write(tmp_path, "prompts.jsonl", json.dumps({"prompt": "hi"}))
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    code = main(["run", str(input_path), "--provider", "groq"])

    assert code == 2
    captured = capsys.readouterr()
    assert "GROQ_API_KEY" in captured.err
    # Nothing resembling a key value should ever be echoed.
    assert "sk-" not in captured.err
    assert "sk-" not in captured.out


def test_missing_api_key_openrouter_exits_2(tmp_path, monkeypatch, capsys):
    input_path = _write(tmp_path, "prompts.jsonl", json.dumps({"prompt": "hi"}))
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    code = main(
        [
            "run",
            str(input_path),
            "--provider",
            "openrouter",
            "--model",
            "some/model",
        ]
    )

    assert code == 2
    captured = capsys.readouterr()
    assert "OPENROUTER_API_KEY" in captured.err


# --------------------------------------------------------------------------
# per-prompt failure isolation
# --------------------------------------------------------------------------


def test_failing_prompt_does_not_abort_run(tmp_path, monkeypatch):
    """A single bad prompt yields an error line and exit 1, others succeed.

    We can't inject a failure through JSONL fields (the mock provider's
    failure knobs are constructor args, not per-request), so instead we
    monkeypatch MockClient.complete to fail on one specific prompt text.
    """
    from llm_gateway.providers.mock import MockClient

    original_complete = MockClient.complete

    async def flaky_complete(self, request):
        if request.prompt == "boom":
            from llm_gateway.types import ProviderError

            raise ProviderError("synthetic failure", status=400, provider=self.name)
        return await original_complete(self, request)

    monkeypatch.setattr(MockClient, "complete", flaky_complete)

    input_path = _write(
        tmp_path,
        "prompts.jsonl",
        "\n".join(
            json.dumps({"prompt": p}) for p in ["good 1", "boom", "good 2"]
        ),
    )
    output_path = tmp_path / "out.jsonl"

    code = main(
        [
            "run",
            str(input_path),
            "--output",
            str(output_path),
            "--no-metrics",
            "--max-attempts",
            "1",
            "--batch-size",
            "1",
        ]
    )

    assert code == 1
    rows = _lines(output_path.read_text())
    assert len(rows) == 3
    by_prompt = {r["prompt"]: r for r in rows}
    assert "text" in by_prompt["good 1"]
    assert "text" in by_prompt["good 2"]
    assert "error" in by_prompt["boom"]
    assert by_prompt["boom"]["error_status"] == 400


# --------------------------------------------------------------------------
# --version
# --------------------------------------------------------------------------


def test_version(capsys):
    with pytest.raises(SystemExit) as exc_info:
        main(["--version"])
    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert __version__ in captured.out


# --------------------------------------------------------------------------
# malformed input
# --------------------------------------------------------------------------


def test_malformed_json_line_is_reported_clearly(tmp_path, capsys):
    input_path = _write(
        tmp_path,
        "prompts.jsonl",
        '{"prompt": "ok"}\nnot json at all\n',
    )

    code = main(["run", str(input_path)])

    assert code == 2
    captured = capsys.readouterr()
    assert "line 2" in captured.err
    assert "Traceback" not in captured.err


def test_missing_prompt_field_is_reported_clearly(tmp_path, capsys):
    input_path = _write(tmp_path, "prompts.jsonl", json.dumps({"not_prompt": "x"}))

    code = main(["run", str(input_path)])

    assert code == 2
    captured = capsys.readouterr()
    assert "prompt" in captured.err
    assert "Traceback" not in captured.err


# --------------------------------------------------------------------------
# multiple providers (--provider a,b)
# --------------------------------------------------------------------------


def test_multi_provider_failover(tmp_path, monkeypatch):
    """Two mock providers, the first always failing: every prompt still
    succeeds (via the second) and the output row names the provider that
    actually served it."""
    from llm_gateway import cli as cli_module
    from llm_gateway.providers.mock import MockClient

    real_mock_provider = cli_module.MockProvider
    calls = {"n": 0}

    def fake_mock_provider(name="mock", **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            client = MockClient(name="primary", fail_status=503)
            return real_mock_provider(name="primary", client=client, failure_threshold=1000)
        client = MockClient(name="backup")
        return real_mock_provider(name="backup", client=client)

    monkeypatch.setattr(cli_module, "MockProvider", fake_mock_provider)

    input_path = _write(
        tmp_path,
        "prompts.jsonl",
        "\n".join(json.dumps({"prompt": f"item {i}"}) for i in range(6)),
    )
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
    assert len(rows) == 6
    assert all("text" in r for r in rows)
    assert all(r["provider"] == "backup" for r in rows)


def test_api_key_rejected_with_multiple_providers(tmp_path, capsys):
    input_path = _write(tmp_path, "prompts.jsonl", json.dumps({"prompt": "hi"}))

    code = main(
        [
            "run",
            str(input_path),
            "--provider",
            "mock,mock",
            "--api-key",
            "some-key",
        ]
    )

    assert code == 2
    captured = capsys.readouterr()
    assert "--api-key" in captured.err
    assert "single" in captured.err


def test_single_provider_still_works_unchanged(tmp_path):
    """--provider mock (no comma) behaves exactly as before."""
    input_path = _write(
        tmp_path,
        "prompts.jsonl",
        "\n".join(json.dumps({"prompt": f"item {i}"}) for i in range(3)),
    )
    output_path = tmp_path / "out.jsonl"

    code = main(
        ["run", str(input_path), "--provider", "mock", "--output", str(output_path), "--no-metrics"]
    )

    assert code == 0
    rows = _lines(output_path.read_text())
    assert len(rows) == 3
    assert all(r["provider"] == "mock" for r in rows)
