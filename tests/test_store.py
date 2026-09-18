from __future__ import annotations

import asyncio
import contextlib

from llm_gateway import LLMGateway, LLMRequest, RetryPolicy, RunStore, idempotency_key
from llm_gateway.providers.mock import MockClient, MockProvider


def fast_retry(**kw) -> RetryPolicy:
    kw.setdefault("base_delay", 0.001)
    kw.setdefault("max_delay", 0.01)
    return RetryPolicy(**kw)


def make_gateway(store: RunStore | None, client: MockClient | None = None) -> LLMGateway:
    provider = MockProvider("a", client=client)
    return LLMGateway(providers=[provider], retry=fast_retry(), store=store)


# --------------------------------------------------------------------------
# idempotency key
# --------------------------------------------------------------------------


def test_key_changes_with_scored_fields():
    base = LLMRequest("hello", max_tokens=10, model="m1")
    variants = [
        LLMRequest("goodbye", max_tokens=10, model="m1"),
        LLMRequest("hello", max_tokens=11, model="m1"),
        LLMRequest("hello", max_tokens=10, model="m2"),
        LLMRequest("hello", max_tokens=10, model="m1", needs_logprobs=True),
        LLMRequest("hello", max_tokens=10, model="m1", needs_strict_json=True),
    ]
    base_key = idempotency_key(base)
    for variant in variants:
        assert idempotency_key(variant) != base_key


def test_key_ignores_priority_and_metadata():
    base = LLMRequest("hello", max_tokens=10, model="m1")
    same = LLMRequest(
        "hello", max_tokens=10, model="m1", priority=9, metadata={"row": 42}
    )
    assert idempotency_key(base) == idempotency_key(same)


# --------------------------------------------------------------------------
# fresh store / no-store path
# --------------------------------------------------------------------------


async def test_fresh_store_path_works(tmp_path):
    store = RunStore(tmp_path / "run.db")
    gw = make_gateway(store)
    async with gw:
        resp = await gw.submit(LLMRequest("hello"))
    assert "hello" in resp.text
    assert gw.freshly_called == 1
    assert gw.served_from_store == 0
    await store.aclose()


async def test_no_store_path_unaffected():
    client = MockClient("a")
    gw = make_gateway(None, client=client)
    async with gw:
        resp1 = await gw.submit(LLMRequest("hello"))
        resp2 = await gw.submit(LLMRequest("hello"))
    assert client.calls == 2
    assert gw.served_from_store == 0
    assert gw.freshly_called == 0
    assert "hello" in resp1.text
    assert "hello" in resp2.text


# --------------------------------------------------------------------------
# resume: completed prompts are not re-called
# --------------------------------------------------------------------------


async def test_completed_prompt_not_recalled_on_second_run(tmp_path):
    db_path = tmp_path / "run.db"
    client = MockClient("a")

    store1 = RunStore(db_path)
    gw1 = make_gateway(store1, client=client)
    async with gw1:
        await gw1.submit(LLMRequest("hello"))
    await store1.aclose()
    assert client.calls == 1

    # New process, same file, same underlying client instance standing in
    # for "the provider": if resume worked, it must not be touched again.
    store2 = RunStore(db_path)
    gw2 = make_gateway(store2, client=client)
    async with gw2:
        resp = await gw2.submit(LLMRequest("hello"))
    assert client.calls == 1
    assert gw2.served_from_store == 1
    assert gw2.freshly_called == 0
    assert "hello" in resp.text
    await store2.aclose()


async def test_partial_run_resumes_only_remainder(tmp_path):
    db_path = tmp_path / "run.db"
    client = MockClient("a")

    store1 = RunStore(db_path)
    gw1 = make_gateway(store1, client=client)
    async with gw1:
        await gw1.submit(LLMRequest("one"))
        await gw1.submit(LLMRequest("two"))
    await store1.aclose()
    assert client.calls == 2

    store2 = RunStore(db_path)
    gw2 = make_gateway(store2, client=client)
    async with gw2:
        results = await gw2.submit_many(
            [LLMRequest("one"), LLMRequest("two"), LLMRequest("three")]
        )
    assert client.calls == 3  # only "three" made a fresh call
    assert gw2.served_from_store == 2
    assert gw2.freshly_called == 1
    assert [r.text for r in results] == [
        f"[a] {p}" for p in ("one", "two", "three")
    ]
    await store2.aclose()


# --------------------------------------------------------------------------
# crash simulation: in_flight rows are reclaimed
# --------------------------------------------------------------------------


async def test_in_flight_rows_reclaimed_after_crash(tmp_path):
    db_path = tmp_path / "run.db"
    client = MockClient("a")

    # Simulate a crash: reserve a row (marks it in_flight) and then simply
    # abandon this store object without ever calling complete()/fail() --
    # standing in for the process dying mid-call.
    crashed_store = RunStore(db_path)
    request = LLMRequest("hello")
    key = idempotency_key(request)
    raced = await crashed_store.reserve(request, key)
    assert raced is None
    counts = await crashed_store.counts()
    assert counts.get("in_flight") == 1
    crashed_store.close()

    fresh_store = RunStore(db_path)
    counts_after_open = await fresh_store.counts()
    assert counts_after_open.get("in_flight", 0) == 0
    assert counts_after_open.get("pending") == 1

    gw = make_gateway(fresh_store, client=client)
    async with gw:
        resp = await gw.submit(request)
    assert client.calls == 1
    assert gw.freshly_called == 1
    assert "hello" in resp.text
    await fresh_store.aclose()


# --------------------------------------------------------------------------
# failures are recorded and retried on resume
# --------------------------------------------------------------------------


async def test_failure_recorded_and_retried_on_resume(tmp_path):
    db_path = tmp_path / "run.db"

    failing_client = MockClient("a", fail_status=503)
    store1 = RunStore(db_path)
    provider = MockProvider("a", client=failing_client)
    gw1 = LLMGateway(providers=[provider], retry=fast_retry(max_attempts=1), store=store1)
    async with gw1:
        with contextlib.suppress(Exception):
            await gw1.submit(LLMRequest("hello"))
    counts = await store1.counts()
    assert counts.get("failed") == 1
    await store1.aclose()

    healthy_client = MockClient("a")
    store2 = RunStore(db_path)
    gw2 = make_gateway(store2, client=healthy_client)
    async with gw2:
        resp = await gw2.submit(LLMRequest("hello"))
    assert healthy_client.calls == 1
    assert gw2.freshly_called == 1
    assert "hello" in resp.text
    await store2.aclose()


# --------------------------------------------------------------------------
# concurrency
# --------------------------------------------------------------------------


async def test_concurrent_submits_do_not_corrupt_or_deadlock(tmp_path):
    db_path = tmp_path / "run.db"
    client = MockClient("a")
    store = RunStore(db_path)
    gw = make_gateway(store, client=client)

    prompts = [LLMRequest(f"prompt-{i % 20}") for i in range(80)]
    async with gw:
        results = await asyncio.wait_for(
            asyncio.gather(*(gw.submit(r) for r in prompts)), timeout=10
        )
    assert len(results) == 80
    counts = await store.counts()
    assert counts.get("done", 0) == 20
    assert sum(counts.values()) == 20
    await store.aclose()


async def test_store_reflects_response_fields(tmp_path):
    db_path = tmp_path / "run.db"
    store = RunStore(db_path)
    gw = make_gateway(store)
    async with gw:
        resp = await gw.submit(LLMRequest("hello", max_tokens=10))
    cached = await store.get_response(idempotency_key(LLMRequest("hello", max_tokens=10)))
    assert cached is not None
    assert cached.text == resp.text
    assert cached.provider == resp.provider
    assert cached.input_tokens == resp.input_tokens
    assert cached.output_tokens == resp.output_tokens
    await store.aclose()
