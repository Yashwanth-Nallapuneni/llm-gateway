"""Tests for MetricsSink.to_dict() and its JSON-serialisability."""

from __future__ import annotations

import json

from llm_gateway.metrics import MetricsSink


def _known_run() -> MetricsSink:
    m = MetricsSink()
    m.submitted = 3
    m.rejected = 1
    m.record_attempt("mock")
    m.record_success("mock", latency=0.1, cost=0.01)
    m.record_attempt("mock")
    m.record_success("mock", latency=0.3, cost=0.02)
    m.record_attempt("mock")
    m.record_failure("mock", status=429)
    m.record_retry("mock")
    m.record_batch("mock", 2)
    m.record_batch("mock", 3)
    return m


def test_to_dict_matches_known_run():
    m = _known_run()
    d = m.to_dict()

    assert d["submitted"] == 3
    assert d["completed"] == 2
    assert d["rejected"] == 1

    p = d["providers"]["mock"]
    assert p["succeeded"] == 2
    assert p["failed"] == 1
    assert p["retries"] == 1
    assert p["failures_by_class"] == {"429": 1}
    assert p["cost"] == 0.03
    assert p["latency_p50"] in (0.1, 0.3)
    assert p["batch_sizes"] == [2, 3]

    assert d["failures_by_class"] == {"429": 1}
    assert d["batch_histogram"] == {"2": 1, "3": 1}
    assert d["mean_batch_size"] == 2.5


def test_to_dict_empty_sink():
    d = MetricsSink().to_dict()
    assert d["providers"] == {}
    assert d["failures_by_class"] == {}
    assert d["batch_histogram"] == {}
    assert d["mean_batch_size"] == 0.0


def test_to_dict_is_json_serialisable():
    d = _known_run().to_dict()
    text = json.dumps(d)
    assert json.loads(text) == d


def test_report_and_to_dict_agree():
    m = _known_run()
    report = m.report()
    d = m.to_dict()
    assert f"submitted={d['submitted']}" in report
    assert "429=1" in report
