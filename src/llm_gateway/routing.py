"""Capability-aware provider selection.

Read `routing.EXPLAIN.md` before this file.
"""

from __future__ import annotations

from .providers.base import Provider
from .types import LLMRequest, NoEligibleProviderError


class ProviderRouter:
    """Pick the provider for a request: filter hard, then rank soft."""

    def __init__(self, prefer_cheap: bool = True) -> None:
        self.prefer_cheap = prefer_cheap

    # ------------------------------------------------------------------
    # Stage 1: capability filter
    # ------------------------------------------------------------------

    def _capability_miss(self, req: LLMRequest, provider: Provider) -> str | None:
        """Return the name of the unmet hard constraint, or None."""
        caps = provider.capabilities

        # WHY: capability is a HARD constraint, not a preference, and so it is
        #      evaluated before anything else. This is the DeepInfra/Novita
        #      split from the author's eval runs, generalized: one endpoint
        #      returns token logprobs but will not enforce a JSON schema, the
        #      other enforces the schema but drops logprobs. A request that
        #      needs logprobs is not "better served" by a provider without
        #      them -- it is not served at all.
        # ALT: scoring capability as a heavy penalty instead of a filter. That
        #      is strictly worse: under enough load or cost pressure the
        #      penalty gets outweighed and the request silently routes to a
        #      provider that cannot satisfy it.
        # ASK: Why is capability a filter rather than a term in the score?
        if req.needs_logprobs and not caps.supports_logprobs:
            return "needs logprobs, provider does not support them"
        if req.needs_strict_json and not caps.supports_strict_json:
            return "needs strict JSON schema, provider does not support it"

        # TRAP: the context check must count prompt AND completion. A provider
        #       with a 4k window cannot serve a 3.5k prompt asking for 1k of
        #       output -- the failure arrives mid-generation as a truncated
        #       response or a 400, long after routing has moved on.
        # ASK: What do you compare against max_context_tokens -- prompt size,
        #      or prompt plus max_tokens?
        needed = req.estimated_total_tokens()
        if needed > caps.max_context_tokens:
            return (
                f"needs ~{needed} tokens of context, provider caps at "
                f"{caps.max_context_tokens}"
            )
        return None

    # ------------------------------------------------------------------
    # Stage 3: ranking
    # ------------------------------------------------------------------

    def _score(self, provider: Provider) -> tuple:
        """Sort key over the survivors. Lower sorts first."""
        # WHY: headroom dominates cost. A marginally cheaper provider whose
        #      bucket is empty does not save money -- it converts money saved
        #      into seconds of blocking in acquire(), and under a sustained
        #      sweep that is the difference between finishing and not. Cost
        #      only breaks ties between providers that can both take the work
        #      right now.
        # ALT: ranking on cost first gives you a stampede onto the cheapest
        #      provider: everything queues on one bucket while the others sit
        #      idle, and the aggregate throughput of the fleet collapses to
        #      that of its cheapest member.
        # ASK: Why rank on rate-limit headroom before cost?
        #
        # TRAP: headroom is read from the buckets, not from a success counter.
        #       It is a *predictive* signal -- it says the next call will not
        #       block -- whereas error counts are retrospective and only tell
        #       you about limits you have already blown through.
        headroom = provider.limiter.headroom

        # Bucket the headroom into coarse bands before comparing.
        # WHY: raw floats make the comparison hyper-sensitive -- 0.81 vs 0.80
        #      is noise, yet it would deterministically pin all traffic to one
        #      provider and never let cost or priority matter at all. Rounding
        #      to bands means "meaningfully more free" wins, and near-ties fall
        #      through to the cheaper option.
        # ASK: Why quantize the headroom score instead of comparing raw floats?
        band = round(headroom * 10)

        cost = provider.blended_cost_per_1k() if self.prefer_cheap else 0.0

        # Negate the band because sort() is ascending and more headroom is
        # better; cost and priority are both "smaller is better" already.
        return (-band, cost, -provider.priority, provider.name)

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def select(self, req: LLMRequest, providers: list[Provider]) -> Provider:
        """Choose a provider, or raise NoEligibleProviderError."""
        # Reasons are accumulated per provider as it is eliminated. This dict
        # is the entire value of the error path -- see the TRAP below.
        reasons: dict[str, str] = {}
        eligible: list[Provider] = []

        for provider in providers:
            miss = self._capability_miss(req, provider)
            if miss is not None:
                reasons[provider.name] = miss
                continue

            # WHY: health is checked after capability but before ranking. A
            #      provider with an open breaker is known-bad right now, so
            #      spending a score on it is wasted -- and, worse, an open
            #      breaker usually means an empty error budget, which makes its
            #      rate-limit headroom look *excellent* precisely because
            #      nothing is getting through.
            # ALT: penalizing unhealthy providers in the score instead of
            #      excluding them -- see the ALT above; same failure, and here
            #      the bad signal actively rewards the broken provider.
            # ASK: Why does a down provider look attractive to a headroom-based
            #      ranker, and how do you stop that?
            if not provider.breaker.allows_request():
                reasons[provider.name] = (
                    f"circuit breaker is {provider.breaker.state.value}"
                )
                continue

            eligible.append(provider)

        if not eligible:
            # TRAP: the two tempting alternatives here are both worse than
            #       raising. Returning None makes every call site grow a
            #       None-check that someone will forget. Falling back to "the
            #       first provider anyway" is the genuinely dangerous one: the
            #       caller asked for logprobs, gets a well-formed response with
            #       `logprobs=None`, and finds out hours later when the
            #       analysis script divides by a missing field -- by which
            #       point the run is finished and the money is spent.
            # WHY: the error names which constraint eliminated which provider,
            #      because "no eligible provider" alone is unactionable. The
            #      operator needs to know whether to raise a limit, fix a key,
            #      or add a capability.
            # ASK: What is the worst thing a router can do when nothing matches?
            raise NoEligibleProviderError(reasons)

        eligible.sort(key=self._score)
        return eligible[0]

    def select_all(self, req: LLMRequest, providers: list[Provider]) -> list[Provider]:
        """Every eligible provider, best first.

        WHY: failover needs the ordered remainder, not just the winner. Calling
             select() again after a failure would re-run the same scoring and,
             until the breaker trips, hand back the same provider that just
             failed.
        ASK: How does the gateway fail over without re-selecting the provider
             that just failed?
        """
        eligible = [
            p
            for p in providers
            if self._capability_miss(req, p) is None and p.breaker.allows_request()
        ]
        if not eligible:
            return []
        eligible.sort(key=self._score)
        return eligible
