"""A dollar ceiling that refuses work once the run would overspend.

Run:  python examples/budget_ceiling.py

A BudgetLedger is given a small ceiling and handed to the gateway. More
work is submitted than the ceiling allows, so some requests succeed and
the rest are refused with BudgetExceeded before they ever reach the
provider.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from llm_gateway import LLMGateway, LLMRequest
from llm_gateway.budget import BudgetExceeded, BudgetLedger
from llm_gateway.providers.mock import MockProvider

N_REQUESTS = 40
CEILING_USD = 0.20


async def main() -> None:
    ledger = BudgetLedger(limit_usd=CEILING_USD)
    provider = MockProvider(
        "priced_co",
        cost_per_1k_input=0.50,
        cost_per_1k_output=1.00,
    )
    gw = LLMGateway(providers=[provider], budget=ledger)

    requests = [
        LLMRequest(prompt=f"priced prompt #{i}", max_tokens=16) for i in range(N_REQUESTS)
    ]

    async with gw:
        results = await asyncio.gather(
            *(gw.submit(r) for r in requests), return_exceptions=True
        )

    succeeded = [r for r in results if not isinstance(r, Exception)]
    refused = [r for r in results if isinstance(r, BudgetExceeded)]
    other_errors = [
        r
        for r in results
        if isinstance(r, Exception) and not isinstance(r, BudgetExceeded)
    ]

    print(f"submitted {N_REQUESTS} requests against a ${CEILING_USD:.2f} budget")
    print(f"succeeded: {len(succeeded)}")
    print(f"refused with BudgetExceeded: {len(refused)}")
    if other_errors:
        print(f"other errors (unexpected): {len(other_errors)}")
    print(f"final spent: ${ledger.spent:.6f} of ${ledger.limit_usd:.2f} ceiling")
    print(f"outstanding reservations at the end: {ledger.outstanding_count}")


if __name__ == "__main__":
    asyncio.run(main())
