"""Token-bucket rate limiting.

Read `rate_limit.EXPLAIN.md` before this file.

Every non-obvious decision below carries a WHY / ALT / TRAP / ASK tag. The
tags are the point of this module: the code is ~60 lines and the reasoning
behind it is what an interviewer actually probes.
"""

from __future__ import annotations

import asyncio
import re
import time
from collections.abc import Awaitable, Callable, Mapping


class TokenBucket:
    """Capacity `C` tokens, refilled at `refill_rate` tokens/sec.

    Allows a burst of up to `C` while bounding the long-run average to
    `refill_rate`. O(1) memory and O(1) time per acquisition.
    """

    def __init__(
        self,
        capacity: float,
        refill_rate: float,
        *,
        clock: Callable[[], float] | None = None,
        sleep: Callable[[float], Awaitable[None]] | None = None,
    ) -> None:
        """capacity: burst allowance. refill_rate: sustained tokens/sec.

        `clock` and `sleep` exist only so tests can drive a fake clock; the
        defaults are the real thing.
        """
        if capacity <= 0:
            raise ValueError("capacity must be > 0")
        if refill_rate <= 0:
            raise ValueError("refill_rate must be > 0")

        self.capacity = float(capacity)
        self.refill_rate = float(refill_rate)

        # WHY: monotonic() is immune to system clock adjustments -- NTP sync,
        #      DST, an operator running `date -s`. It only ever moves forward.
        # ALT: time.time() is wall-clock. If NTP steps the clock backward
        #      mid-run, `elapsed` goes negative, the bucket refills by a
        #      negative amount, and the limiter silently locks up until wall
        #      time catches back up to where it was.
        # ASK: Why monotonic() instead of time.time() in a rate limiter?
        self._clock = clock or time.monotonic
        self._sleep = sleep or asyncio.sleep

        # WHY: start full. A process that has just booted has consumed none of
        #      the provider's budget, so it is entitled to the full burst.
        # ALT: start empty and make callers wait C/r seconds for no reason.
        self._tokens = self.capacity
        self._last_refill = self._clock()

        # WHY: the read-modify-write of (_tokens, _last_refill) is not atomic.
        #      Between reading _tokens and writing it back there is an await
        #      point in acquire(), so two coroutines can both observe "enough
        #      tokens" and both deduct, issuing 2x the allowed traffic.
        # ALT: no lock at all "because asyncio is single-threaded" -- true but
        #      irrelevant: single-threaded does not mean uninterrupted, it
        #      means interrupted only at awaits, and this method has one.
        # ASK: Does single-threaded asyncio need locks? When?
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _refill(self) -> None:
        """Bring the bucket up to date. Caller must hold the lock."""
        now = self._clock()
        elapsed = now - self._last_refill

        # WHY: lazy refill -- compute what *would* have accumulated since the
        #      last call, rather than a background task ticking every N ms.
        # ALT: a background asyncio task adding tokens on a timer. Costs one
        #      task per bucket, drifts under event-loop congestion, and needs
        #      explicit shutdown handling or it leaks on every gateway close.
        # ASK: How do you refill a token bucket without a background timer?
        self._tokens = min(self.capacity, self._tokens + elapsed * self.refill_rate)

        # TRAP: the min() clamp above is load-bearing. Without it an idle
        #       bucket accrues unbounded tokens, and the first burst after a
        #       quiet hour blows straight through the provider's limit -- the
        #       exact 429 storm the limiter exists to prevent.
        # ASK: What breaks if you drop the min(capacity, ...) clamp?

        self._last_refill = now

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    async def acquire(self, tokens: float = 1.0) -> None:
        """Block until `tokens` available, then deduct.

        Must not busy-wait. Must be correct under concurrent callers.
        """
        # TRAP: a request larger than the bucket can never be satisfied -- the
        #       clamp in _refill() caps _tokens at capacity, so the loop below
        #       would sleep forever. Fail loudly at the boundary instead of
        #       hanging a coroutine with no traceback.
        # ASK: What happens if someone asks for more tokens than capacity?
        if tokens > self.capacity:
            raise ValueError(
                f"requested {tokens} tokens but bucket capacity is {self.capacity}"
            )

        # WHY: a loop, not a single sleep-then-take. Between computing the wait
        #      and waking up, another coroutine can consume the very tokens we
        #      were waiting for. Re-check after every sleep.
        # TRAP: sleeping for `deficit / refill_rate` and then deducting without
        #       re-checking is the classic over-issue bug: N waiters all wake
        #       up "entitled" to the same tokens and all deduct.
        # ASK: Why does the sleep sit inside a re-check loop?
        while True:
            async with self._lock:
                self._refill()
                if self._tokens >= tokens:
                    self._tokens -= tokens
                    return
                deficit = tokens - self._tokens
                # WHY: sleep exactly as long as the missing tokens take to
                #      refill. Precise, and no CPU burned in the meantime.
                # ALT: poll on a fixed 10ms tick -- busy-waits, adds up to 10ms
                #      of latency per acquisition, and scales badly with waiters.
                wait = deficit / self.refill_rate

            # TRAP: this await is deliberately OUTSIDE the `async with`. Holding
            #       the lock across a sleep serializes every waiter behind the
            #       first one: waiter #2 cannot even *check* the bucket until
            #       waiter #1 has slept its full duration, so throughput
            #       collapses to one acquisition per sleep instead of one per
            #       refill interval. This release/reacquire boundary is the
            #       subtlest bug in the module.
            # ASK: Why must the lock not be held across `await asyncio.sleep()`?
            await self._sleep(wait)

    def try_acquire(self, tokens: float = 1.0) -> bool:
        """Non-blocking. Deduct and return True, or return False untouched."""
        # WHY: no lock and no await. This function contains no await point, so
        #      the event loop cannot interleave another coroutine in the middle
        #      of it -- the read-modify-write is already atomic with respect to
        #      other coroutines on this loop.
        # ALT: making it `async` and taking the lock. That would make it
        #      awaitable, which is exactly wrong for a routing hot path that
        #      wants to *peek* at headroom without yielding control.
        # TRAP: this reasoning only holds for a single event loop. Sharing one
        #       bucket across threads needs a threading.Lock instead.
        # ASK: Why does try_acquire not need the lock but acquire does?
        if tokens > self.capacity:
            return False
        self._refill()
        if self._tokens >= tokens:
            self._tokens -= tokens
            return True
        return False

    def sync(self, remaining: float, reset_in: float | None = None) -> None:
        """Reconcile the local count with a provider's `x-ratelimit-remaining`.

        `remaining` is the provider's own count of what is left in its
        window, as of the instant it built the response we just received.
        `reset_in`, if given, is how many seconds the provider says are left
        until that window resets; see the WHY block below for why it is
        accepted but deliberately not used to adjust refill timing.

        WHY: this method exists at all because the local bucket is a model,
             not a measurement -- it is seeded from a number a human typed
             into config and then only ever updated by our own guesses about
             elapsed time. If the configured limit is wrong, or another
             process/replica is spending the same provider key, the model
             drifts away from what the provider actually enforces, and we
             only find out via a 429. The provider hands us ground truth on
             every single response; this is where we fold it back in.
        ASK: Why does the bucket need correcting from response headers at all
             if it already tracks tokens locally?
        """
        # WHY: refresh local accounting to "now" before comparing against the
        #      server's number. Without this, _tokens reflects whatever state
        #      it was left in at the last acquire/refill call, which could be
        #      arbitrarily stale relative to `self._clock()` -- comparing a
        #      current server number against a stale local one would make
        #      the min() below meaningless.
        self._refill()

        # TRAP: min(), never max(). The server's number is already stale by
        #       the time we see it -- it describes the instant the response
        #       was generated, and any request we fired after that one (but
        #       before this one's response arrived) has already spent tokens
        #       the server hasn't told us about yet. Trusting a *larger*
        #       server number would hand back tokens we've already spent
        #       locally, and the very next burst would sail past the real
        #       limit straight into a 429 -- precisely the failure this
        #       method exists to prevent. Taking the minimum means sync()
        #       can only ever make the bucket more conservative, never less;
        #       that asymmetry is the whole point and is what makes it safe
        #       to call unconditionally on every response.
        # ALT: overwrite unconditionally (`self._tokens = remaining`).
        #      Simpler, but a single reordered or delayed response landing
        #      after a fresher one would silently hand back tokens and undo
        #      every deduction made since -- drift in the unsafe direction.
        # ASK: Why min() instead of just trusting the server's number
        #      outright?
        self._tokens = min(self._tokens, remaining)

        # WHY: clamp defensively. A provider bug, a transient negative
        #      remaining count, or a remaining count that -- because two
        #      processes configured the same key with different limits --
        #      exceeds our own configured capacity, must not corrupt the
        #      invariant every other method relies on: 0 <= _tokens <=
        #      capacity. A bucket that reports available() > capacity would
        #      break the router's headroom fraction (headroom > 1.0).
        # TRAP: clamping the lower bound matters just as much as the upper
        #       one -- a negative _tokens would make the next acquire()'s
        #       `deficit = tokens - self._tokens` larger than it should be,
        #       but more importantly would make try_acquire() and available()
        #       report nonsense to callers that assume tokens are >= 0.
        self._tokens = max(0.0, min(self.capacity, self._tokens))

        # ASK: What is `reset_in` for, and why doesn't it touch the refill
        #      math above?
        # WHY: it is accepted, and intentionally NOT folded into `_tokens` or
        #      `_last_refill`, because this bucket models a *continuous*
        #      refill (elapsed_seconds * refill_rate), while the provider's
        #      `reset_in` describes a *discrete* fixed window that jumps back
        #      to full capacity at one instant. Those two models disagree
        #      about the shape of the curve between now and the reset
        #      instant, not just its endpoints, so there is no single correct
        #      way to fold a discrete deadline into a continuous rate without
        #      guessing at the provider's internal window algorithm.
        # ALT: schedule `_tokens = capacity` at `now + reset_in` (e.g. via a
        #      timer, or by snapping `_last_refill` backward to fake extra
        #      elapsed time). Rejected on two counts: (1) this module is
        #      deliberately timer-free -- see `_refill`'s WHY -- and a
        #      scheduled callback here reintroduces the exact background-task
        #      lifecycle/leak risk lazy refill exists to avoid; (2) faking
        #      elapsed time would fight the min() above the next time sync()
        #      is called mid-window with a smaller `remaining`, since a
        #      forced-full bucket would simply get clamped back down anyway
        #      -- the complexity would buy nothing durable.
        # TRAP: an unused parameter can look like a bug in review. It isn't
        #       one here -- `sync_from_headers` already has to parse
        #       `reset_in` out of the response (Groq/OpenAI-style headers
        #       pair every `-remaining-` with a `-reset-`), and this keeps
        #       that parsed value going somewhere explicit and documented
        #       instead of being silently dropped at the call site, which
        #       would look like the parsing itself was the bug.

    @property
    def available(self) -> float:
        """Current token count, refreshed. Used by the router for headroom."""
        self._refill()
        return self._tokens

    @property
    def headroom(self) -> float:
        """Available tokens as a fraction of capacity, in [0, 1].

        WHY: the router compares buckets of different sizes (600 rpm against
             150k tpm). Raw token counts are not comparable across dimensions;
             a normalized fraction is.
        """
        return self.available / self.capacity

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"TokenBucket(capacity={self.capacity}, rate={self.refill_rate}, "
            f"available={self._tokens:.2f})"
        )


# ----------------------------------------------------------------------
# header parsing for sync_from_headers
# ----------------------------------------------------------------------

# WHY: Groq (and some gateways) send reset windows as Go-style duration
#      strings -- "7.66s", "2m59.56s", "1h2m3s", "120ms" -- rather than a
#      bare number of seconds. OpenAI sends bare seconds. One regex handles
#      every ordered combination of hours/minutes/seconds/milliseconds, all
#      optional, so a plain "7.66s" and a compound "1h2m3s" both match.
# TRAP: "m" (minutes) and "ms" (milliseconds) share a prefix. Without the
#       `(?!s)` lookahead after the minutes group, "120ms" would be
#       misparsed as failing to match minutes only by luck of group order;
#       the lookahead makes that explicit and correct instead of accidental.
# ASK: How do you tell "m" (minutes) apart from "ms" (milliseconds) in a
#      single regex without two passes?
_DURATION_RE = re.compile(
    r"^(?:(?P<h>\d+(?:\.\d+)?)h)?"
    r"(?:(?P<m>\d+(?:\.\d+)?)m(?!s))?"
    r"(?:(?P<s>\d+(?:\.\d+)?)s)?"
    r"(?:(?P<ms>\d+(?:\.\d+)?)ms)?$"
)


def _parse_float(value: str) -> float | None:
    """Best-effort float parse. Returns None instead of raising.

    WHY: a header value is untrusted input from a network peer. `float()`
         raising ValueError/TypeError deep inside a response-handling path
         is exactly the kind of thing that must never take down a request
         that otherwise succeeded -- the response body the caller wants is
         already in hand by the time headers are synced.
    """
    try:
        return float(value.strip())
    except (ValueError, AttributeError):
        return None


def _parse_duration(value: str) -> float | None:
    """Parse a plain number of seconds OR a Go-style duration string.

    ASK: Why try a bare float before the duration regex?
    WHY: plain numeric reset values ("120") are the common case (OpenAI-
         compatible APIs) and a numeric string like "120" would otherwise
         also satisfy a loosely-written duration regex as "no unit,
         therefore zero" -- trying float() first avoids that ambiguity
         entirely rather than special-casing it inside the regex.
    """
    try:
        stripped = value.strip()
    except AttributeError:
        return None
    try:
        return float(stripped)
    except ValueError:
        pass

    match = _DURATION_RE.fullmatch(stripped)
    # TRAP: `_DURATION_RE` matches the empty string (every group is
    #       optional), so a blank/garbage header would silently parse as
    #       "0 seconds" without the `any(match.groups())` guard below --
    #       indistinguishable from a provider that legitimately sent "the
    #       window resets right now". Require at least one unit to have
    #       actually matched before trusting the result.
    if not match or not any(match.groups()):
        return None

    hours = float(match.group("h") or 0.0)
    minutes = float(match.group("m") or 0.0)
    seconds = float(match.group("s") or 0.0)
    millis = float(match.group("ms") or 0.0)
    return hours * 3600.0 + minutes * 60.0 + seconds + millis / 1000.0


class ProviderLimiter:
    """The two independent buckets a provider needs.

    WHY: request-rate and token-rate are independent limits. A workload of many
         tiny prompts hits RPM first; a workload of few huge prompts hits TPM
         first. One bucket cannot express both, so both are checked and both
         are deducted before dispatch.
    ASK: Why two buckets per provider instead of one?
    """

    def __init__(
        self,
        rpm_limit: float,
        tpm_limit: float,
        *,
        clock: Callable[[], float] | None = None,
        sleep: Callable[[float], Awaitable[None]] | None = None,
    ) -> None:
        # WHY: capacity == the per-minute limit, refill == limit/60 per second.
        #      That reproduces the provider's own model: a full minute's budget
        #      may be spent as a burst, but the sustained rate is the limit.
        self.requests = TokenBucket(rpm_limit, rpm_limit / 60.0, clock=clock, sleep=sleep)
        self.tokens = TokenBucket(tpm_limit, tpm_limit / 60.0, clock=clock, sleep=sleep)

    async def acquire(self, n_requests: float = 1.0, n_tokens: float = 0.0) -> None:
        # TRAP: acquire the request bucket first, then the token bucket. Doing
        #       it in a consistent order everywhere is what keeps two buckets
        #       from deadlocking against each other under contention.
        await self.requests.acquire(n_requests)
        if n_tokens > 0:
            await self.tokens.acquire(min(n_tokens, self.tokens.capacity))

    @property
    def headroom(self) -> float:
        """The binding constraint: a provider is only as free as its tightest
        dimension, so take the minimum rather than an average."""
        return min(self.requests.headroom, self.tokens.headroom)

    def sync_from_headers(self, headers: Mapping[str, str]) -> None:
        """Correct both buckets from a provider's `x-ratelimit-*` response headers.

        Recognises the common OpenAI-compatible spellings
        (`x-ratelimit-remaining-requests`, `x-ratelimit-remaining-tokens`,
        `x-ratelimit-reset-requests`, `x-ratelimit-reset-tokens`) plus the
        unqualified `x-ratelimit-remaining` / `x-ratelimit-reset` some
        gateways send when they don't distinguish the two dimensions.

        WHY: this is the only place in the module that touches attacker- (or
             at least peer-) controlled strings. Everywhere else, callers
             pass floats they already own. A header is text some remote
             server chose to send, in whatever casing, spelling, and format
             it likes -- this method's whole job is to turn that into either
             a valid call to `TokenBucket.sync` or nothing at all.
        ASK: Why does this method exist separately from `TokenBucket.sync`?
        """
        # WHY: case-insensitive lookup by building a lowered copy once.
        #      HTTP header names are case-insensitive per RFC 7230, but
        #      `Mapping[str, str]` gives no such guarantee -- an httpx or
        #      aiohttp headers object normalizes this for you, but a plain
        #      dict (as tests, and some hand-rolled adapters, will pass)
        #      does not. Doing this once up front is also simpler than
        #      re-scanning per lookup.
        # TRAP: this whole method must never raise -- it runs on the
        #       response path of every single call, success or not. A
        #       malformed `headers` argument (not actually a mapping of
        #       str->str, e.g. a Mock or a bytes-keyed dict from a lower-
        #       level HTTP client) must degrade to "nothing synced", not an
        #       exception that would take down an otherwise-successful
        #       response.
        # ASK: Why wrap the entire body in a bare `except Exception`, rather
        #      than only around the float/duration parsing?
        try:
            lowered = {str(k).lower(): v for k, v in headers.items()}
        except Exception:
            return

        def first(*names: str) -> str | None:
            for name in names:
                value = lowered.get(name)
                if value is not None:
                    return value
            return None

        # WHY: try the dimension-specific header first, fall back to the
        #      unqualified one. A provider that sends both is telling us the
        #      qualified one is the more precise answer for that dimension;
        #      a provider that sends only the unqualified pair (some
        #      gateways don't separate RPM/TPM) still gets *some* correction
        #      instead of none.
        req_remaining = first("x-ratelimit-remaining-requests", "x-ratelimit-remaining")
        req_reset = first("x-ratelimit-reset-requests", "x-ratelimit-reset")
        tok_remaining = first("x-ratelimit-remaining-tokens", "x-ratelimit-remaining")
        tok_reset = first("x-ratelimit-reset-tokens", "x-ratelimit-reset")

        try:
            if req_remaining is not None:
                remaining = _parse_float(req_remaining)
                if remaining is not None:
                    reset_in = _parse_duration(req_reset) if req_reset is not None else None
                    self.requests.sync(remaining, reset_in)

            if tok_remaining is not None:
                remaining = _parse_float(tok_remaining)
                if remaining is not None:
                    reset_in = _parse_duration(tok_reset) if tok_reset is not None else None
                    self.tokens.sync(remaining, reset_in)
        except Exception:
            # TRAP: unparseable or missing values must be ignored silently,
            #       per spec -- but "unparseable" also covers surprises the
            #       regex/float paths above didn't anticipate (e.g. a header
            #       value that isn't a str at all because a caller passed a
            #       non-conforming mapping). Catching here, not just
            #       returning None from the parsers, is the difference
            #       between "this header is ignored" and "this response
            #       handler crashed because of a header".
            return
