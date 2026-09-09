from __future__ import annotations

import pytest

from llm_gateway.breaker import CircuitBreaker, State
from llm_gateway.types import CircuitOpenError


def make(clock, **kw):
    kw.setdefault("failure_threshold", 5)
    kw.setdefault("recovery_timeout", 30.0)
    return CircuitBreaker(clock=clock, **kw)


def test_opens_at_threshold(clock):
    cb = make(clock)
    for _ in range(4):
        cb.record_failure()
        assert cb.state is State.CLOSED
    cb.record_failure()
    assert cb.state is State.OPEN


def test_success_resets_the_consecutive_count(clock):
    cb = make(clock)
    for _ in range(4):
        cb.record_failure()
    cb.record_success()
    for _ in range(4):
        cb.record_failure()
    assert cb.state is State.CLOSED


def test_open_breaker_rejects_without_calling_the_provider(clock):
    cb = make(clock)
    for _ in range(5):
        cb.record_failure()
    assert cb.allows_request() is False
    with pytest.raises(CircuitOpenError):
        cb.check()


def test_half_open_permits_exactly_one_probe(clock):
    cb = make(clock)
    for _ in range(5):
        cb.record_failure()
    clock.advance(30)
    assert cb.state is State.HALF_OPEN
    assert cb.allows_request() is True
    assert cb.allows_request() is False  # second caller is refused


def test_half_open_success_closes(clock):
    cb = make(clock)
    for _ in range(5):
        cb.record_failure()
    clock.advance(31)
    cb.allows_request()
    cb.record_success()
    assert cb.state is State.CLOSED
    assert cb.allows_request() is True


def test_half_open_failure_reopens_and_restarts_the_timer(clock):
    cb = make(clock)
    for _ in range(5):
        cb.record_failure()
    clock.advance(31)
    cb.allows_request()
    cb.record_failure()
    assert cb.state is State.OPEN
    clock.advance(29)
    assert cb.state is State.OPEN  # timer restarted, not resumed
    clock.advance(2)
    assert cb.state is State.HALF_OPEN


def test_failures_while_open_do_not_restart_the_recovery_timer(clock):
    """Stragglers landing after the trip must not postpone recovery."""
    cb = make(clock)
    for _ in range(5):
        cb.record_failure()
    clock.advance(20)
    cb.record_failure()  # a call that was already in flight when it tripped
    clock.advance(11)
    assert cb.state is State.HALF_OPEN
