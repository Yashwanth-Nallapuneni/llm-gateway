"""In-memory metrics. No Prometheus, no external deps."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field


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

    def report(self) -> str:
        lines: list[str] = []
        lines.append("=" * 78)
        lines.append("LLM GATEWAY METRICS")
        lines.append("=" * 78)
        lines.append(
            f"submitted={self.submitted}  completed={self.completed}  "
            f"rejected={self.rejected}"
        )
        lines.append("")
        header = (
            f"{'provider':<14}{'ok':>6}{'fail':>6}{'retry':>7}"
            f"{'p50':>9}{'p95':>9}{'p99':>9}{'blocked':>10}{'cost':>10}"
        )
        lines.append(header)
        lines.append("-" * len(header))
        for name, m in sorted(self._providers.items()):
            lines.append(
                f"{name:<14}{m.succeeded:>6}{m.failed:>6}{m.retries:>7}"
                f"{_percentile(m.latencies, 50):>9.3f}"
                f"{_percentile(m.latencies, 95):>9.3f}"
                f"{_percentile(m.latencies, 99):>9.3f}"
                f"{m.blocked_seconds:>10.2f}"
                f"{m.cost:>10.4f}"
            )

        failures: Counter[str] = Counter()
        for m in self._providers.values():
            failures.update(m.failures_by_class)
        if failures:
            lines.append("")
            lines.append("failures by class: " + ", ".join(
                f"{k}={v}" for k, v in sorted(failures.items())
            ))

        hist = self.batch_histogram()
        if hist:
            lines.append("")
            lines.append("batch size distribution")
            widest = max(hist.values())
            for size in sorted(hist):
                count = hist[size]
                bar = "#" * max(1, int(40 * count / widest))
                lines.append(f"  {size:>3} | {bar} {count}")
            total_reqs = sum(s * c for s, c in hist.items())
            total_batches = sum(hist.values())
            lines.append(
                f"  mean batch size: {total_reqs / total_batches:.2f} "
                f"over {total_batches} dispatches"
            )
        lines.append("=" * 78)
        return "\n".join(lines)
