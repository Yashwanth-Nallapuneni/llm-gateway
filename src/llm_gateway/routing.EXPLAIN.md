# Provider routing — read this before `routing.py`

## The problem

Providers are not interchangeable, and they are not equally healthy, and they
are not equally free right now. Three different kinds of difference, and the
router's job is to keep them straight — because the correct way to handle each
one is different.

The concrete version, from batch evaluations run across DeepInfra, Novita and
OpenRouter: one endpoint returns token logprobs but will not enforce a JSON
schema. Another enforces the schema but drops logprobs. Choosing by hand,
per request, is exactly the kind of bookkeeping that works fine for a week and
then quietly puts 4,000 rows of an eval through the wrong endpoint.

## The algorithm

**Stage 1 — filter by capability.** Hard constraint. A request that needs
logprobs cannot go to a provider without them; there is no amount of cheapness
or availability that changes that. Context window is checked here too, and it
must count prompt *plus* `max_tokens` — a 4k-window provider cannot serve a
3.5k prompt asking for 1k of output, and that failure arrives mid-generation,
long after routing has moved on.

**Stage 2 — filter by health.** Exclude providers whose circuit breaker is open.
Also a hard filter, for a reason that is easy to miss: see below.

**Stage 3 — rank the survivors.** Sort on rate-limit headroom first, then cost,
then configured priority.

**Stage 4 — if nothing survives, raise**, naming which constraint eliminated
which provider.

## Why headroom outranks cost

A marginally cheaper provider whose token bucket is empty does not save you
money. It converts money saved into seconds spent blocking inside `acquire()`,
and under a sustained sweep that is the difference between finishing and not.

Rank on cost first and you get a stampede: every request queues on the cheapest
provider while the others sit idle, and the aggregate throughput of the whole
fleet collapses to that of its cheapest member. Headroom first, cost as the
tiebreak between providers that can both take the work *right now*, is the
ordering that keeps every provider busy.

Headroom is also the right *kind* of signal: it is predictive. It says the next
call will not block. Error counters are retrospective — they only tell you about
limits you have already blown through.

One wrinkle: headroom is quantized into coarse bands before comparison. Raw
floats are hypersensitive — 0.81 versus 0.80 is noise, yet it would pin all
traffic to one provider and never let cost or priority matter at all.

## Why health is a filter, not a penalty

Here is the trap. A provider that is hard-down has an *excellent* rate-limit
headroom, precisely because nothing is getting through to consume it. A ranker
that scores health as a penalty term is fighting its own headroom signal, and
whenever the penalty loses, traffic routes preferentially to the most broken
provider in the fleet.

The same argument applies to capability. Score it as a heavy penalty and, under
enough load or cost pressure, the penalty eventually gets outweighed and a
request silently routes somewhere that cannot satisfy it. Hard constraints get
filters. Preferences get scores. Mixing them is how you get a router that is
correct in testing and wrong under load.

## Why failure must be loud

When nothing survives, there are three things the router could do, and two of
them are bad.

Returning `None` makes every call site grow a null check somebody will forget.

Falling back to "the first provider anyway" is the genuinely dangerous one. The
caller asked for logprobs. They get back a perfectly well-formed response with
`logprobs=None`. Nothing raises. They find out hours later, when the analysis
script divides by a missing field — at which point the run is finished, the
money is spent, and the results are garbage that *looks* like data.

So: raise, and name the reason per provider. "No eligible provider" alone is
unactionable; the operator needs to know whether to raise a limit, fix a key, or
add a capability. `NoEligibleProviderError` carries a `{provider: reason}` map
for exactly that.

## Three questions an interviewer will ask

**Why is capability a filter rather than a term in the score?** Because a score
can be outweighed. A request that needs logprobs is not merely better served
elsewhere; it cannot be served at all by a provider without them.

**Why does a down provider look attractive to a headroom-based ranker?** Its
buckets are full — nothing is getting through to drain them. That is why health
must exclude rather than penalize.

**What is the worst thing a router can do when nothing matches?** Silently fall
back to an incapable provider. The caller gets a response missing the field they
required and does not find out until much later.
