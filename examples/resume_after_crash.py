"""Resuming a sweep after a crash using RunStore.

Run:  python examples/resume_after_crash.py

A RunStore backs the gateway with a SQLite file. The first "process" only
gets partway through a sweep before it stops -- standing in for a crash.
The second "process" opens the same store file and runs the exact same
prompts again: everything already recorded comes straight from the store,
and only the unfinished prompts actually call the provider.
"""

from __future__ import annotations

import asyncio
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from llm_gateway import LLMGateway, LLMRequest
from llm_gateway.providers.mock import MockClient, MockProvider
from llm_gateway.store import RunStore

N_REQUESTS = 20
N_BEFORE_CRASH = 12


def make_requests() -> list[LLMRequest]:
    return [
        LLMRequest(prompt=f"sweep prompt #{i}", max_tokens=32) for i in range(N_REQUESTS)
    ]


async def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        store_path = Path(tmp) / "sweep.db"
        client = MockClient("sweep_co")

        # First run: only submit the first N_BEFORE_CRASH prompts, then stop
        # without finishing the rest -- simulating a process that died
        # partway through the sweep.
        store = RunStore(store_path, run_id="sweep-1")
        gw = LLMGateway(providers=[MockProvider("sweep_co", client=client)], store=store)
        async with gw:
            await gw.submit_many(make_requests()[:N_BEFORE_CRASH])
        store.close()

        print(f"first run: submitted {N_BEFORE_CRASH}/{N_REQUESTS} prompts, then stopped")
        print(f"provider calls so far: {client.calls}\n")

        # Second run: same store file, same prompts, run again from scratch.
        # The first N_BEFORE_CRASH prompts hash to the same idempotency key
        # they did before, so they are served from the store; only the
        # remaining ones reach the provider.
        store2 = RunStore(store_path, run_id="sweep-2")
        gw2 = LLMGateway(
            providers=[MockProvider("sweep_co", client=client)], store=store2
        )
        async with gw2:
            responses = await gw2.submit_many(make_requests())
        counts = await store2.counts()
        store2.close()

    assert len(responses) == N_REQUESTS
    print("second run: resumed the same sweep")
    print(f"served from store (no provider call): {gw2.served_from_store}")
    print(f"freshly called the provider: {gw2.freshly_called}")
    print(f"provider calls total across both runs: {client.calls}")
    print(f"store row counts by status: {counts}")


if __name__ == "__main__":
    asyncio.run(main())
