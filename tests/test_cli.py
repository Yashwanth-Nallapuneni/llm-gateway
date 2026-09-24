"""CLI tests. Everything here runs offline against the mock provider."""

from __future__ import annotations

import io
import json
import sys

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
        "\n".join(json.dumps({"id": f"p{i}", "prompt": f"hello {i}"}) for i in range(5)),
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

    code = main(["run", str(input_path), "--output", str(output_path), "--no-metrics"])

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

    code = main(["run", str(input_path), "--output", str(output_path), "--no-metrics"])

    assert code == 0
    rows = _lines(output_path.read_text())
    assert [r["prompt"] for r in rows] == ["first prompt", "second prompt"]


def test_stdin_input(tmp_path, monkeypatch, capsys):
    import io

    monkeypatch.setattr("sys.stdin", io.StringIO("stdin prompt one\nstdin prompt two\n"))
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


def test_dry_run_works_without_api_key(tmp_path, monkeypatch, capsys):
    """A dry run makes no network calls, so it must not require a key."""
    input_path = _write(tmp_path, "prompts.jsonl", json.dumps({"prompt": "hi"}))
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    code = main(["run", str(input_path), "--provider", "groq", "--dry-run"])

    assert code == 0
    captured = capsys.readouterr()
    assert "prompts: 1" in captured.err
    assert "estimated cost" in captured.err
    assert captured.out == ""


def test_dry_run_works_without_api_key_multi_provider(tmp_path, monkeypatch, capsys):
    input_path = _write(tmp_path, "prompts.jsonl", json.dumps({"prompt": "hi"}))
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    code = main(
        [
            "run",
            str(input_path),
            "--provider",
            "groq,openrouter",
            "--model",
            "openrouter=some/model",
            "--dry-run",
        ]
    )

    assert code == 0
    captured = capsys.readouterr()
    assert "prompts: 1" in captured.err


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


def test_missing_api_key_multi_provider_does_not_mention_api_key_flag(
    tmp_path, monkeypatch, capsys
):
    """With multiple providers, --api-key can't be used, so the missing-key
    message should only point at the env var, not at the disallowed flag."""
    input_path = _write(tmp_path, "prompts.jsonl", json.dumps({"prompt": "hi"}))
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key-value")

    code = main(
        [
            "run",
            str(input_path),
            "--provider",
            "groq,openrouter",
            "--model",
            "openrouter=some/model",
        ]
    )

    assert code == 2
    captured = capsys.readouterr()
    assert "GROQ_API_KEY" in captured.err
    assert "--api-key" not in captured.err


def test_missing_model_multi_provider_shows_multi_syntax(tmp_path, monkeypatch, capsys):
    """--model is required for openrouter; with multiple providers the
    message should show the name=value syntax, not the single-provider form."""
    input_path = _write(tmp_path, "prompts.jsonl", json.dumps({"prompt": "hi"}))
    monkeypatch.setenv("GROQ_API_KEY", "test-key-value")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key-value")

    code = main(["run", str(input_path), "--provider", "groq,openrouter"])

    assert code == 2
    captured = capsys.readouterr()
    assert "openrouter=<model id>" in captured.err


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
        "\n".join(json.dumps({"prompt": p}) for p in ["good 1", "boom", "good 2"]),
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
            return real_mock_provider(
                name="primary", client=client, failure_threshold=1000
            )
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
    assert len(rows) == 3
    assert all(r["provider"] == "mock" for r in rows)


# --------------------------------------------------------------------------
# progress line
# --------------------------------------------------------------------------


class _TTYStringIO(io.StringIO):
    """A StringIO that claims to be a terminal, for progress-line tests."""

    def isatty(self) -> bool:
        return True


def test_progress_not_printed_when_stderr_is_not_a_tty(tmp_path, capsys):
    input_path = _write(
        tmp_path,
        "prompts.jsonl",
        "\n".join(json.dumps({"prompt": f"p{i}"}) for i in range(5)),
    )

    code = main(["run", str(input_path), "--output", str(tmp_path / "out.jsonl")])

    assert code == 0
    captured = capsys.readouterr()
    assert "\r" not in captured.err


def test_progress_not_printed_with_quiet(tmp_path, monkeypatch):
    input_path = _write(
        tmp_path,
        "prompts.jsonl",
        "\n".join(json.dumps({"prompt": f"p{i}"}) for i in range(5)),
    )
    fake_err = _TTYStringIO()
    monkeypatch.setattr(sys, "stderr", fake_err)

    code = main(
        ["run", str(input_path), "--output", str(tmp_path / "out.jsonl"), "--quiet"]
    )

    assert code == 0
    assert "\r" not in fake_err.getvalue()


def test_progress_printed_on_a_tty(tmp_path, monkeypatch):
    input_path = _write(
        tmp_path,
        "prompts.jsonl",
        "\n".join(json.dumps({"prompt": f"p{i}"}) for i in range(5)),
    )
    fake_err = _TTYStringIO()
    monkeypatch.setattr(sys, "stderr", fake_err)

    code = main(["run", str(input_path), "--output", str(tmp_path / "out.jsonl")])

    assert code == 0
    output = fake_err.getvalue()
    assert "\r" in output
    assert "ok" in output
    assert "failed" in output
    # final update always lands, even with the 0.2s throttle
    assert "[5/5]" in output


def test_progress_does_not_pollute_stdout(tmp_path, capsys):
    input_path = _write(
        tmp_path,
        "prompts.jsonl",
        "\n".join(json.dumps({"prompt": f"p{i}"}) for i in range(5)),
    )

    code = main(["run", str(input_path), "--no-metrics"])

    assert code == 0
    captured = capsys.readouterr()
    rows = _lines(captured.out)
    assert len(rows) == 5
    assert "\r" not in captured.out


# --------------------------------------------------------------------------
# --adaptive
# --------------------------------------------------------------------------


def test_adaptive_flag_is_wired_to_the_gateway(tmp_path, monkeypatch):
    input_path = _write(tmp_path, "prompts.jsonl", json.dumps({"prompt": "hi"}))

    seen = {}
    from llm_gateway import cli as cli_module

    real_gateway_cls = cli_module.LLMGateway

    def spy_gateway(*args, **kwargs):
        seen["adaptive"] = kwargs.get("adaptive")
        return real_gateway_cls(*args, **kwargs)

    monkeypatch.setattr(cli_module, "LLMGateway", spy_gateway)

    code = main(
        [
            "run",
            str(input_path),
            "--output",
            str(tmp_path / "out.jsonl"),
            "--adaptive",
            "--no-metrics",
        ]
    )

    assert code == 0
    assert seen["adaptive"] is True


def test_adaptive_defaults_to_false(tmp_path, monkeypatch):
    input_path = _write(tmp_path, "prompts.jsonl", json.dumps({"prompt": "hi"}))

    seen = {}
    from llm_gateway import cli as cli_module

    real_gateway_cls = cli_module.LLMGateway

    def spy_gateway(*args, **kwargs):
        seen["adaptive"] = kwargs.get("adaptive")
        return real_gateway_cls(*args, **kwargs)

    monkeypatch.setattr(cli_module, "LLMGateway", spy_gateway)

    code = main(
        ["run", str(input_path), "--output", str(tmp_path / "out.jsonl"), "--no-metrics"]
    )

    assert code == 0
    assert seen["adaptive"] is False


# --------------------------------------------------------------------------
# models command
# --------------------------------------------------------------------------


def test_models_command_mock(capsys):
    code = main(["models", "--provider", "mock"])
    assert code == 0
    out = capsys.readouterr().out
    assert out.splitlines() == ["mock"]


def test_models_command_prints_sorted_ids_and_filters(monkeypatch, capsys):
    import llm_gateway.cli as cli_module

    async def fake_list_models(provider, api_key):
        assert api_key == "test-key"
        return ["a-model", "allam-2-7b", "z-model"]

    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setattr(cli_module, "_list_models", fake_list_models)

    code = main(["models", "--provider", "groq"])
    assert code == 0
    assert capsys.readouterr().out.splitlines() == ["a-model", "allam-2-7b", "z-model"]

    code = main(["models", "--provider", "groq", "--contains", "allam"])
    assert code == 0
    assert capsys.readouterr().out.splitlines() == ["allam-2-7b"]


def test_models_command_missing_api_key_exits_2(monkeypatch, capsys):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    code = main(["models", "--provider", "groq"])
    assert code == 2
    assert "GROQ_API_KEY" in capsys.readouterr().err


# --------------------------------------------------------------------------
# input/flag validation edge cases
# --------------------------------------------------------------------------


def test_negative_limit_is_rejected(tmp_path, capsys):
    input_path = _write(
        tmp_path,
        "prompts.jsonl",
        "\n".join(json.dumps({"prompt": f"item {i}"}) for i in range(5)),
    )
    code = main(
        ["run", str(input_path), "--provider", "mock", "--limit", "-2", "--no-metrics"]
    )
    assert code == 2
    assert "--limit" in capsys.readouterr().err


def test_zero_budget_is_rejected_cleanly(tmp_path, capsys):
    input_path = _write(tmp_path, "prompts.jsonl", json.dumps({"prompt": "hi"}) + "\n")
    code = main(
        [
            "run",
            str(input_path),
            "--provider",
            "mock",
            "--budget",
            "0",
            "--no-metrics",
        ]
    )
    assert code == 2
    assert "--budget" in capsys.readouterr().err


def test_negative_budget_is_rejected_cleanly(tmp_path, capsys):
    input_path = _write(tmp_path, "prompts.jsonl", json.dumps({"prompt": "hi"}) + "\n")
    code = main(
        [
            "run",
            str(input_path),
            "--provider",
            "mock",
            "--budget",
            "-5",
            "--no-metrics",
        ]
    )
    assert code == 2
    assert "--budget" in capsys.readouterr().err


def test_unwritable_output_path_is_rejected_cleanly(tmp_path, capsys):
    input_path = _write(tmp_path, "prompts.jsonl", json.dumps({"prompt": "hi"}) + "\n")
    bad_output = tmp_path / "no_such_dir" / "out.jsonl"
    code = main(
        [
            "run",
            str(input_path),
            "--provider",
            "mock",
            "--output",
            str(bad_output),
            "--no-metrics",
        ]
    )
    assert code == 2
    assert "cannot write" in capsys.readouterr().err


def test_unwritable_metrics_json_path_is_rejected_cleanly(tmp_path, capsys):
    input_path = _write(tmp_path, "prompts.jsonl", json.dumps({"prompt": "hi"}) + "\n")
    bad_metrics = tmp_path / "no_such_dir" / "metrics.json"
    code = main(
        [
            "run",
            str(input_path),
            "--provider",
            "mock",
            "--metrics-json",
            str(bad_metrics),
            "--no-metrics",
        ]
    )
    assert code == 2
    assert "cannot write" in capsys.readouterr().err


# --------------------------------------------------------------------------
# jsonl edge cases
# --------------------------------------------------------------------------


def test_jsonl_blank_lines_are_skipped(tmp_path):
    input_path = _write(
        tmp_path,
        "prompts.jsonl",
        '\n\n{"prompt": "a"}\n\n{"prompt": "b"}\n\n',
    )
    output_path = tmp_path / "out.jsonl"
    code = main(["run", str(input_path), "--output", str(output_path), "--no-metrics"])
    assert code == 0
    rows = _lines(output_path.read_text())
    assert [r["prompt"] for r in rows] == ["a", "b"]


def test_jsonl_non_string_prompt_is_rejected(tmp_path, capsys):
    input_path = _write(tmp_path, "prompts.jsonl", json.dumps({"prompt": 123}))
    code = main(["run", str(input_path)])
    assert code == 2
    err = capsys.readouterr().err
    assert "prompt" in err
    assert "Traceback" not in err


def test_jsonl_extra_fields_are_ignored(tmp_path):
    input_path = _write(
        tmp_path,
        "prompts.jsonl",
        json.dumps({"prompt": "hi", "unexpected_field": "whatever", "another": 1}),
    )
    output_path = tmp_path / "out.jsonl"
    code = main(["run", str(input_path), "--output", str(output_path), "--no-metrics"])
    assert code == 0
    rows = _lines(output_path.read_text())
    assert rows[0]["prompt"] == "hi"


def test_jsonl_custom_id_preserved(tmp_path):
    input_path = _write(
        tmp_path, "prompts.jsonl", json.dumps({"id": "custom-123", "prompt": "hi"})
    )
    output_path = tmp_path / "out.jsonl"
    code = main(["run", str(input_path), "--output", str(output_path), "--no-metrics"])
    assert code == 0
    rows = _lines(output_path.read_text())
    assert rows[0]["id"] == "custom-123"


def test_jsonl_unicode_round_trips(tmp_path):
    prompt = "你好世界 \U0001f600 café"
    input_path = _write(
        tmp_path, "prompts.jsonl", json.dumps({"prompt": prompt}, ensure_ascii=False)
    )
    output_path = tmp_path / "out.jsonl"
    code = main(["run", str(input_path), "--output", str(output_path), "--no-metrics"])
    assert code == 0
    rows = _lines(output_path.read_text())
    assert rows[0]["prompt"] == prompt


def test_empty_jsonl_file_is_rejected_cleanly(tmp_path, capsys):
    input_path = _write(tmp_path, "prompts.jsonl", "")
    code = main(["run", str(input_path)])
    assert code == 2
    err = capsys.readouterr().err
    assert "empty" in err
    assert "Traceback" not in err


def test_jsonl_blank_lines_only_is_rejected_as_empty(tmp_path, capsys):
    input_path = _write(tmp_path, "prompts.jsonl", "\n\n\n")
    code = main(["run", str(input_path)])
    assert code == 2
    assert "empty" in capsys.readouterr().err


# --------------------------------------------------------------------------
# txt edge cases
# --------------------------------------------------------------------------


def test_txt_crlf_trailing_newline_and_blank_lines(tmp_path):
    input_path = _write(
        tmp_path, "prompts.txt", "first\r\n\r\nsecond\r\n\r\n\r\nthird\r\n"
    )
    output_path = tmp_path / "out.jsonl"
    code = main(["run", str(input_path), "--output", str(output_path), "--no-metrics"])
    assert code == 0
    rows = _lines(output_path.read_text())
    assert [r["prompt"] for r in rows] == ["first", "second", "third"]


# --------------------------------------------------------------------------
# --store / --resume
# --------------------------------------------------------------------------


def test_resume_when_everything_already_done_does_not_call_provider_again(
    tmp_path, monkeypatch
):
    from llm_gateway.providers.mock import MockClient

    input_path = _write(
        tmp_path,
        "prompts.jsonl",
        "\n".join(json.dumps({"prompt": p}) for p in ["a", "b"]),
    )
    store_path = tmp_path / "store.db"
    output_path = tmp_path / "out.jsonl"

    code = main(
        [
            "run",
            str(input_path),
            "--output",
            str(output_path),
            "--no-metrics",
            "--store",
            str(store_path),
        ]
    )
    assert code == 0

    calls = {"n": 0}
    original_complete = MockClient.complete

    async def counting_complete(self, request):
        calls["n"] += 1
        return await original_complete(self, request)

    monkeypatch.setattr(MockClient, "complete", counting_complete)

    output_path2 = tmp_path / "out2.jsonl"
    code = main(
        [
            "run",
            str(input_path),
            "--output",
            str(output_path2),
            "--no-metrics",
            "--store",
            str(store_path),
            "--resume",
        ]
    )
    assert code == 0
    assert calls["n"] == 0, "resume after everything is done must not call the provider"
    rows = _lines(output_path2.read_text())
    assert [r["prompt"] for r in rows] == ["a", "b"]


def test_store_without_resume_on_existing_file_is_rejected(tmp_path):
    input_path = _write(tmp_path, "prompts.jsonl", json.dumps({"prompt": "a"}))
    store_path = tmp_path / "store.db"

    code = main(["run", str(input_path), "--no-metrics", "--store", str(store_path)])
    assert code == 0

    code = main(["run", str(input_path), "--no-metrics", "--store", str(store_path)])
    assert code == 2


# --------------------------------------------------------------------------
# Ctrl-C / interruption mid-run
# --------------------------------------------------------------------------


def test_dry_run_flags_a_zero_estimate_as_unpriced(tmp_path, capsys, monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    input_path = tmp_path / "p.txt"
    input_path.write_text("hello\n")
    code = main(["run", str(input_path), "--provider", "groq", "--dry-run"])
    assert code == 0
    assert "no per-token prices are configured" in capsys.readouterr().err
