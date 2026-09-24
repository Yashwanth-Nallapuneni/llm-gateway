"""Capability-aware provider selection."""

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

        # Capability is a hard constraint, checked before anything else. A
        # request that needs logprobs is not served by a provider without
        # them, so this must filter candidates out rather than just score
        # them down -- otherwise, under enough load, the scoring penalty
        # gets outweighed and the request silently routes somewhere that
        # can't actually satisfy it.
        if req.needs_logprobs and not caps.supports_logprobs:
            return "needs logprobs, provider does not support them"
        if req.needs_strict_json and not caps.supports_strict_json:
            return "needs strict JSON schema, provider does not support it"

        # Counts prompt tokens plus expected completion tokens, since a
        # provider can run out of context mid-generation even if the
        # prompt alone would have fit.
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

    def _score(self, provider: Provider) -> tuple[int, float, int, str]:
        """Sort key over the survivors. Lower sorts first."""
        # Headroom (how much rate-limit capacity is free) is ranked before
        # cost. A cheaper provider with an empty bucket doesn't actually
        # save money -- it just blocks in acquire() -- so ranking on cost
        # first would stampede all traffic onto the cheapest provider and
        # leave the others idle.
        headroom = provider.limiter.headroom

        # Round headroom into coarse bands before comparing. Comparing raw
        # floats would treat noise like 0.81 vs 0.80 as a real difference
        # and pin all traffic to one provider; bands let only a meaningful
        # gap in headroom win, with cost breaking near-ties.
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
        # Records why each eliminated provider was rejected, so the error
        # below can explain itself instead of just saying "none worked".
        reasons: dict[str, str] = {}
        eligible: list[Provider] = []

        for provider in providers:
            miss = self._capability_miss(req, provider)
            if miss is not None:
                reasons[provider.name] = miss
                continue

            # Checked after capability but before ranking. A provider with
            # an open circuit breaker is known-bad right now, and since
            # nothing is getting through it, its rate-limit headroom would
            # look artificially excellent -- so it must be excluded, not
            # merely scored down.
            if not provider.breaker.allows_request():
                reasons[provider.name] = (
                    f"circuit breaker is {provider.breaker.state.value}"
                )
                continue

            eligible.append(provider)

        if not eligible:
            # Raising here beats returning None (every caller would need a
            # forgettable None-check) or silently falling back to the first
            # provider anyway (which could return a well-formed response
            # missing a capability the caller actually needed). The error
            # carries the per-provider reasons so the operator knows whether
            # to raise a limit, fix a key, or add a capability.
            raise NoEligibleProviderError(reasons)

        eligible.sort(key=self._score)
        return eligible[0]

    def select_all(self, req: LLMRequest, providers: list[Provider]) -> list[Provider]:
        """Every eligible provider, best first.

        Failover needs the whole ordered list, not just the winner --
        calling select() again after a failure would just hand back the
        same provider, since its breaker hasn't tripped yet.
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
