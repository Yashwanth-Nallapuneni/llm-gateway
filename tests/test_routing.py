from __future__ import annotations

import pytest

from llm_gateway.providers.mock import MockProvider
from llm_gateway.routing import ProviderRouter
from llm_gateway.types import LLMRequest, NoEligibleProviderError


def logprob_provider(**kw):
    return MockProvider("logprobs_co", supports_logprobs=True, **kw)


def json_provider(**kw):
    return MockProvider("json_co", supports_strict_json=True, **kw)


def test_capability_filter_excludes_provider_lacking_logprobs():
    router = ProviderRouter()
    providers = [json_provider(), logprob_provider()]
    chosen = router.select(LLMRequest("hi", needs_logprobs=True), providers)
    assert chosen.name == "logprobs_co"


def test_capability_filter_excludes_provider_lacking_strict_json():
    router = ProviderRouter()
    providers = [logprob_provider(), json_provider()]
    chosen = router.select(LLMRequest("hi", needs_strict_json=True), providers)
    assert chosen.name == "json_co"


def test_context_window_counts_prompt_plus_max_tokens():
    router = ProviderRouter()
    small = MockProvider("small", max_context_tokens=1000)
    big = MockProvider("big", max_context_tokens=100_000)
    # ~250 prompt tokens + 900 completion = 1150 > 1000
    req = LLMRequest("x" * 1000, max_tokens=900)
    assert router.select(req, [small, big]).name == "big"


def test_open_breaker_excludes_provider():
    router = ProviderRouter()
    a, b = MockProvider("a"), MockProvider("b")
    for _ in range(5):
        a.breaker.record_failure()
    assert router.select(LLMRequest("hi"), [a, b]).name == "b"


def test_headroom_ranking_prefers_the_less_loaded_provider():
    router = ProviderRouter()
    loaded = MockProvider("loaded", rpm_limit=100, cost_per_1k_output=0.01)
    free = MockProvider("free", rpm_limit=100, cost_per_1k_output=1.00)
    # Drain the cheap one to ~10% headroom.
    assert loaded.limiter.requests.try_acquire(90)
    chosen = router.select(LLMRequest("hi"), [loaded, free])
    assert chosen.name == "free", "headroom must dominate cost"


def test_cost_breaks_ties_between_equally_free_providers():
    router = ProviderRouter()
    pricey = MockProvider("pricey", cost_per_1k_input=1.0, cost_per_1k_output=2.0)
    cheap = MockProvider("cheap", cost_per_1k_input=0.01, cost_per_1k_output=0.02)
    assert router.select(LLMRequest("hi"), [pricey, cheap]).name == "cheap"


def test_priority_breaks_ties_between_equal_cost_providers():
    router = ProviderRouter()
    a = MockProvider("a", priority=0)
    b = MockProvider("b", priority=10)
    assert router.select(LLMRequest("hi"), [a, b]).name == "b"


def test_no_eligible_provider_names_the_eliminating_constraint():
    router = ProviderRouter()
    plain = MockProvider("plain")
    broken = MockProvider("broken", supports_logprobs=True)
    for _ in range(5):
        broken.breaker.record_failure()

    with pytest.raises(NoEligibleProviderError) as excinfo:
        router.select(LLMRequest("hi", needs_logprobs=True), [plain, broken])

    reasons = excinfo.value.reasons
    assert "logprobs" in reasons["plain"]
    assert "circuit breaker" in reasons["broken"]
    assert "plain" in str(excinfo.value) and "broken" in str(excinfo.value)


def test_router_never_falls_back_to_an_incapable_provider():
    router = ProviderRouter()
    with pytest.raises(NoEligibleProviderError):
        router.select(LLMRequest("hi", needs_logprobs=True), [MockProvider("plain")])


def test_select_all_returns_ranked_survivors():
    router = ProviderRouter()
    a = MockProvider("a", cost_per_1k_output=1.0)
    b = MockProvider("b", cost_per_1k_output=0.1)
    c = MockProvider("c", supports_logprobs=False)
    for _ in range(5):
        c.breaker.record_failure()
    ranked = router.select_all(LLMRequest("hi"), [a, b, c])
    assert [p.name for p in ranked] == ["b", "a"]


def test_select_all_returns_empty_when_nothing_matches():
    router = ProviderRouter()
    assert router.select_all(
        LLMRequest("hi", needs_logprobs=True), [MockProvider("plain")]
    ) == []
