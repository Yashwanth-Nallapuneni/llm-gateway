# Examples

All of these run offline against `MockProvider` -- no API key needed.

- `bulk_eval.py` -- 500 prompts across two providers with different cost and
  capability profiles, routed by the batcher and metrics reported at the end.
  Run: `python examples/bulk_eval.py`
- `failover_demo.py` -- a primary provider starts failing mid-run, the
  circuit breaker trips and traffic moves to a backup, then the primary
  recovers. Run: `python examples/failover_demo.py`
- `timeouts_and_adaptive.py` -- a flaky provider returns a couple of 429s;
  `adaptive=True` halves the request rate in response, and each request
  carries a `timeout_s`. Run: `python examples/timeouts_and_adaptive.py`
- `resume_after_crash.py` -- a sweep stops partway through with a
  `RunStore` behind it, then runs again from the same store file: finished
  prompts come back with no provider call, only the rest are freshly
  called. Run: `python examples/resume_after_crash.py`
- `budget_ceiling.py` -- more work is submitted than a small
  `BudgetLedger` ceiling allows, so some requests succeed and the rest are
  refused with `BudgetExceeded` before reaching the provider. Run:
  `python examples/budget_ceiling.py`
