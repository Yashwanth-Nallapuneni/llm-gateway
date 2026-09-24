"""Per-request `timeout_s`: offline, deterministic, using MockProvider latency."""

from __future__ import annotations

import json

import pytest

from llm_gateway import BudgetLedger, LLMGateway, LLMRequest, RequestTimeout
from llm_gateway.cli import main
from llm_gateway.providers.mock import MockProvider


async def test_request_finishing_in_time_succeeds():
    provider = MockProvider("a", latency=0.01)
    gw = LLMGateway(providers=[provider])
    resp = await gw.submit(LLMRequest(prompt="hi", timeout_s=1.0))
    assert resp.text
    await gw.aclose()


async def test_slow_request_raises_request_timeout():
    provider = MockProvider("a", latency=0.2)
    gw = LLMGateway(providers=[provider])
    with pytest.raises(RequestTimeout):
        await gw.submit(LLMRequest(prompt="hi", timeout_s=0.01))
    await gw.aclose()


async def test_no_timeout_is_unchanged_behaviour():
    provider = MockProvider("a", latency=0.01)
    gw = LLMGateway(providers=[provider])
    resp = await gw.submit(LLMRequest(prompt="hi"))
    assert resp.text
    await gw.aclose()


async def test_budget_reservation_released_after_timeout():
    provider = MockProvider("a", latency=0.2)
    budget = BudgetLedger(1000.0)
    gw = LLMGateway(providers=[provider], budget=budget)

    with pytest.raises(RequestTimeout):
        await gw.submit(LLMRequest(prompt="hi", max_tokens=8, timeout_s=0.01))

    # The dispatch that outlived the caller's wait keeps running in the
    # background; give it a moment to settle the reservation it holds.
    # Poll instead of one fixed sleep so a slow CI machine does not flake.
    import asyncio

    for _ in range(100):
        if budget.outstanding_count == 0:
            break
        await asyncio.sleep(0.05)
    # No reservation is left dangling: the background dispatch settled (or
    # released) it even though the caller stopped waiting.
    assert budget.outstanding_count == 0
    await gw.aclose()


def _write(tmp_path, name, content):
    path = tmp_path / name
    path.write_text(content)
    return path


def test_cli_timeout_marks_slow_prompts_as_errors(tmp_path, monkeypatch):
    from llm_gateway.providers.mock import MockClient

    original_complete = MockClient.complete

    async def slow_complete(self, request):
        if request.prompt == "slow":
            import asyncio

            await asyncio.sleep(0.3)
        return await original_complete(self, request)

    monkeypatch.setattr(MockClient, "complete", slow_complete)

    input_path = _write(
        tmp_path,
        "prompts.jsonl",
        "\n".join(json.dumps({"prompt": p}) for p in ["fast", "slow"]),
    )
    output_path = tmp_path / "out.jsonl"

    code = main(
        [
            "run",
            str(input_path),
            "--output",
            str(output_path),
            "--timeout",
            "0.05",
            "--batch-size",
            "1",
        ]
    )

    assert code == 1
    rows = [json.loads(line) for line in output_path.read_text().splitlines()]
    assert len(rows) == 2
    by_prompt = {row["prompt"]: row for row in rows}
    assert "text" in by_prompt["fast"]
    assert "error" in by_prompt["slow"]


async def test_store_row_settled_after_timeout(tmp_path):
    """A request that times out while store-backed must not leave its row
    stuck `in_flight` forever once the background dispatch finishes."""
    import asyncio

    from llm_gateway import RunStore
    from llm_gateway.providers.mock import MockProvider

    store = RunStore(tmp_path / "run.db")
    provider = MockProvider("a", latency=0.2)
    gw = LLMGateway(providers=[provider], store=store)

    with pytest.raises(RequestTimeout):
        await gw.submit(LLMRequest(prompt="hi", timeout_s=0.01))

    # Give the background dispatch (which keeps running after the caller's
    # wait_for gives up) time to actually finish the provider call.
    for _ in range(100):
        counts = await store.counts()
        if counts.get("in_flight", 0) == 0:
            break
        await asyncio.sleep(0.05)

    counts = await store.counts()
    assert counts.get("in_flight", 0) == 0, (
        f"row stuck in_flight after timeout settled: {counts}"
    )
    await gw.aclose()
    await store.aclose()
