"""In-memory metrics. No Prometheus, no external deps."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    # Nearest-rank. Exact enough for a report table, and it never interpolates
    # a latency that no request actually experienced.
    k = max(0, min(len(ordered) - 1, round(p / 100.0 * len(ordered) + 0.5) - 1))
    return ordered[k]


@dataclass
class ProviderMetrics:
    attempted: int = 0
    succeeded: int = 0
    failed: int = 0
    retries: int = 0
    # Keyed by status class ("4xx", "5xx", "429", "timeout", ...).
    failures_by_class: Counter[str] = field(default_factory=Counter)
    blocked_seconds: float = 0.0
    batch_sizes: list[int] = field(default_factory=list)
    latencies: list[float] = field(default_factory=list)
    cost: float = 0.0


class MetricsSink:
    def __init__(self) -> None:
        self._providers: dict[str, ProviderMetrics] = defaultdict(ProviderMetrics)
        self.submitted = 0
        self.completed = 0
        self.rejected = 0

    def _p(self, name: str) -> ProviderMetrics:
        return self._providers[name]

    # -- recording -----------------------------------------------------

    def record_attempt(self, provider: str) -> None:
        self._p(provider).attempted += 1

    def record_success(self, provider: str, latency: float, cost: float = 0.0) -> None:
        m = self._p(provider)
        m.succeeded += 1
        m.latencies.append(latency)
        m.cost += cost
        self.completed += 1

    def record_failure(self, provider: str, status: int | None) -> None:
        m = self._p(provider)
        m.failed += 1
        if status is None:
            key = "transport"
        elif status == 429:
            key = "429"
        else:
            key = f"{status // 100}xx"
        m.failures_by_class[key] += 1

    def record_retry(self, provider: str) -> None:
        self._p(provider).retries += 1

    def record_blocked(self, provider: str, seconds: float) -> None:
        self._p(provider).blocked_seconds += seconds

    def record_batch(self, provider: str, size: int) -> None:
        self._p(provider).batch_sizes.append(size)

    # -- reporting -----------------------------------------------------

    def batch_histogram(self) -> Counter[int]:
        h: Counter[int] = Counter()
        for m in self._providers.values():
            h.update(m.batch_sizes)
        return h

    def to_dict(self) -> dict[str, Any]:
        """Plain JSON-serialisable snapshot of everything report() shows."""
        providers: dict[str, Any] = {}
        for name, m in sorted(self._providers.items()):
            providers[name] = {
                "succeeded": m.succeeded,
                "failed": m.failed,
                "retries": m.retries,
                "failures_by_class": dict(m.failures_by_class),
                "blocked_seconds": m.blocked_seconds,
                "cost": m.cost,
                "latency_p50": _percentile(m.latencies, 50),
                "latency_p95": _percentile(m.latencies, 95),
                "latency_p99": _percentile(m.latencies, 99),
                "batch_sizes": list(m.batch_sizes),
            }

        failures: Counter[str] = Counter()
        for m in self._providers.values():
            failures.update(m.failures_by_class)

        hist = self.batch_histogram()
        total_reqs = sum(s * c for s, c in hist.items())
        total_batches = sum(hist.values())

        return {
            "submitted": self.submitted,
            "completed": self.completed,
            "rejected": self.rejected,
            "providers": providers,
            "failures_by_class": dict(sorted(failures.items())),
            "batch_histogram": {str(k): v for k, v in sorted(hist.items())},
            "mean_batch_size": total_reqs / total_batches if total_batches else 0.0,
        }

    def report(self) -> str:
        d = self.to_dict()
        lines: list[str] = []
        lines.append("=" * 78)
        lines.append("LLM GATEWAY METRICS")
        lines.append("=" * 78)
        lines.append(
            f"submitted={d['submitted']}  completed={d['completed']}  "
            f"rejected={d['rejected']}"
        )
        lines.append("")
        header = (
            f"{'provider':<14}{'ok':>6}{'fail':>6}{'retry':>7}"
            f"{'p50':>9}{'p95':>9}{'p99':>9}{'blocked':>10}{'cost':>10}"
        )
        lines.append(header)
        lines.append("-" * len(header))
        for name, p in d["providers"].items():
            lines.append(
                f"{name:<14}{p['succeeded']:>6}{p['failed']:>6}{p['retries']:>7}"
                f"{p['latency_p50']:>9.3f}"
                f"{p['latency_p95']:>9.3f}"
                f"{p['latency_p99']:>9.3f}"
                f"{p['blocked_seconds']:>10.2f}"
                f"{p['cost']:>10.4f}"
            )

        if d["failures_by_class"]:
            lines.append("")
            lines.append("failures by class: " + ", ".join(
                f"{k}={v}" for k, v in d["failures_by_class"].items()
            ))

        hist = d["batch_histogram"]
        if hist:
            lines.append("")
            lines.append("batch size distribution")
            widest = max(hist.values())
            for size in sorted(hist, key=int):
                count = hist[size]
                bar = "#" * max(1, int(40 * count / widest))
                lines.append(f"  {size:>3} | {bar} {count}")
            total_batches = sum(hist.values())
            lines.append(
                f"  mean batch size: {d['mean_batch_size']:.2f} "
                f"over {total_batches} dispatches"
            )
        lines.append("=" * 78)
        return "\n".join(lines)
