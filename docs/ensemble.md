# Ensemble consensus

Ask the same prompt many times, reduce the replies to one answer plus an explicit
confidence, and refuse to average across a genuine disagreement.

This is **opt-in per call**. Without an `EnsembleSpec` nothing on this page runs and every
existing surface behaves exactly as before.

## Why

A measurement study asked one model an identical prompt 1000 times at `temperature=0`. Of
the 977 replies that parsed, **652 were distinct** parameter vectors — yet **98.6% reached
the same decision**. The dispersion is not sampling temperature; temperature was already
zero. It is nondeterminism in the serving stack, and it is not available to be turned off.

So a single draw is a poor estimate of the parameters and a good estimate of the decision.
An ensemble is how you get both, with the disagreement reported rather than hidden.

The sharpest case is this package's own product. Scored 100 times on one identical prompt,
`p_memorized` had mean 0.211, sd 0.218, and range 0.000–0.760. A single scoring's 95% band
spans two thirds of the unit interval — while multiplying deployed exposure directly through
`memguard_confidence = raw_confidence · (1 − p_memorized)`. As a multiplier, one draw is
close to uninformative.

## What it actually does

Concretely, one call does this:

1. **Draws** `spec.draws` replies to the same prompt, in waves of `max_workers`, under the
   generation settings you declare.
2. **Triages** each reply into usable, transport failure, parse failure, or
   projection failure — four categories, kept separate because they mean different things.
3. **Refuses** rather than returning, if the request budget is exhausted, too few draws are
   usable, transport failures exceed your threshold, or the credential is rejected.
4. **Canonically orders** the surviving draws by content, so nothing downstream depends on
   the order replies happened to arrive in.
5. **Measures agreement** on the decision your `decide` callback returns, with an interval.
6. **Tests each component** for two separated clusters — *before* estimating anything.
7. **Estimates a location** only for components that pass that test. A flagged component is
   named and left without a location, on purpose.
8. **Selects a representative draw** that is always one you actually received, never a
   synthesised combination.
9. **Hashes** the draw set so the whole reduction can be replayed and checked later.

Steps 6 and 7 are the ordering that carries the design. Estimating first and checking after
would produce a number for a component that has no single answer.

## Sizing: two different questions

`draws` controls two things that need **different** sample sizes, and sizing for one
silently under-sizes for the other:

| question | helper | at 95% |
|---|---|---|
| how precisely do I know the *agreement*? | `smallest_certifiable_n(0.95)` | 52 draws one-sided, 73 two-sided |
| can a *component split* even be detected? | `smallest_detectable_split_n(cluster_positions=...)` | grows with cluster width |

The default `draws=64` is sized for agreement precision. **Split detection needs more.**
Both helpers report a floor, not a promise: below the value, the outcome is impossible;
above it, it is possible but still subject to sampling noise.

Measured on the shipped corpus, whose split is unambiguous at full size:

| parsed draws | split undetected |
|---|---|
| 32 | 12.0% |
| 48 | 9.5% |
| **61** | **5.5%** |
| **64** | **3.6%** |
| 96 | 2.9% |
| 128 | 0.8% |
| 256 | 0.1% |

**Size against `n_parsed`, not against `draws`.** The test only sees replies that survived
transport, parsing, and your projection — so a prompt with a 5% failure rate configured at
`draws=64` is really operating at 61, where the miss rate is half again as high. That row
is in the table for a reason.

In a small fraction of the misses the reported location lands on the wrong side of zero
with nothing flagged. That is the failure mode worth sizing against: not a wrong interval,
but a confident single value for a component that has no single answer. Read
`component_verdicts` (below) rather than trusting an empty `multimodal`.

## Scoring a prompt over many draws

```python
from recall_guard import EnsembleSpec, MemoryGuardedScorer

scorer = MemoryGuardedScorer.calibrate(...)

result = scorer.score_ensemble(
    "your prompt",
    spec=EnsembleSpec(
        draws=64, max_workers=8, min_parsed=24,
        max_tokens=2048,              # match production -- see below
    ),
    conservative_quantile=0.9,        # optional
)

result.p_memorized_point            # <-- the exposure multiplier
result.p_memorized_ci               # descriptive spread of the draws
result.p_memorized_conservative     # upper quantile, when asked for
result.agreement, result.agreement_ci
result.n_requested, result.n_parsed, result.fail_counts
result.draws_sha256
```

### Which number multiplies exposure

**`p_memorized_point`.** Nothing else.

`result.consensus` is a real observed draw, and `consensus.p_memorized` is *its* score. The
two differ — the representative draw is selected by rank, the point estimate reduces every
draw. `consensus` is evidence you can show an auditor; it is not the multiplier.

### Why the mean rather than the median

Attenuation is linear in the score, so the mean is unbiased for expected attenuation. The
score's distribution is right-skewed, so its median sits materially lower and withholds
less exposure — the risk-increasing direction. On the measured data the mean withholds
21.1% of exposure and the median 13.6%: a **1.55× difference in the size of the haircut**.

For the same reason no symmetric trimming is applied to contamination scores. The upper
tail *is* the contamination evidence; trimming it discards the signal.

If you would rather withhold too much than too little, pass `conservative_quantile`.

## Match your production generation settings

`max_tokens` and `temperature` default to `None`, meaning the client's own defaults.
**Set them to whatever your production path uses.**

An ensemble drawn at a different token budget is not measuring the production decision,
which is the entire point of the feature. On a reasoning model the budget is not a detail:
the chain of thought consumes it, and the reply is truncated before the payload you parse.
Those truncated replies surface as `projection` failures, which reads like "the model gave
bad answers" when the real cause is "the ensemble asked a different question".

Measured on one reasoning model, the same prompt, 64 draws each:

| draw budget | parsed |
|---|---|
| 512 (client default) | 31/64 — 48% |
| 2048 (that caller's production setting) | 61/64 — 95% |

A parse rate that low is at least visible. The quieter hazard is a production budget near
the default: a mild dip, and no signal that the ensemble is sampling a different regime.

Both settings are recorded on the result, next to `draws_sha256`, because "under what
settings" is part of what an audit artifact has to carry.

## Reducing a caller-supplied decision

For anything other than the contamination score, supply two callbacks:

```python
from recall_guard import EnsembleSpec, generate_ensemble

result = generate_ensemble(
    lm, "your prompt",
    EnsembleSpec(draws=64, grid=0.1),
    decide=lambda reply: parse_verdict(reply.content),        # hashable
    components=lambda reply: parse_components(reply.content), # named scalars
)
```

Two callbacks, not one, because they answer different questions. `decide` yields a
categorical outcome and drives agreement; `components` yields ordered scalars and drives
location. A single callback cannot serve both, and conflating them is how a reported
consensus ends up naming a different reading than the reported location.

## What it refuses to do

**It reports what the split test saw, for every component.** `result.component_verdicts`
carries a verdict per component, flagged or not — a `separated=False` with masses at
31%/58% is a very different situation from one with no mass either side, and only you can
judge which matters. A `None` verdict means the check did not run at all: no lattice
declared, too few draws, or clusters too thin for a gap to mean anything. Do not read an
empty `multimodal` as "checked and fine".

**It will not report a location inside a gap.** Each component is checked for two separated
clusters *before* any location estimate is computed, and a flagged component is named in
`result.multimodal` and given no location at all. This ordering is load-bearing: on the
measured data no symmetric trim fraction escapes the gap, so the location estimator is
simply the wrong instrument there. Averaging across a 36/63 split returns a value the model
emitted 5 times in 977 attempts — worse than a single draw, because it launders a real
disagreement into false precision.

**It will not report a consensus computed from survivors.** Four separate refusals raise
rather than return: the request budget is exhausted, too few draws were usable, transport
failures exceeded the configured share, or the credential was rejected. A failed ensemble
reports **no** score — never `0.0`, which would mean "pass 100% of exposure through".

**It will not accept a configuration that cannot work.** An agreement target no draw count
can certify is rejected at construction, naming the smallest feasible count. Certifying
0.95 needs at least 52 unanimous draws one-sided, or 73 two-sided — you must declare which
tail you mean.

## Reading the reported numbers honestly

- **The interval assumes independent draws.** Draws against one serving stack are plausibly
  *not* independent — batching, cache reuse, and node affinity all couple requests issued
  together — and positive dependence makes every interval **narrower than its label**.
  `result.draw_dependence` measures this so you can tell; it does not correct for it.
- **Agreement is conditional on the draws that parsed.** Timeouts and token-budget
  truncation censor long generations, so the surviving draws are a length-conditioned
  subsample of unknown direction. Read `n_parsed` against `n_requested`, always.
- **`p_memorized_ci` is descriptive, not inferential.** It is the observed spread of the
  draws you took, by order statistic. There is no bootstrap and no seed, which is what lets
  a stored draw set replay exactly.
- **Cluster detection finds separated clusters only.** Two overlapping modes with no gap
  between them are invisible to it. Known, accepted, not a bug.
- **A shared reference draw understates dispersion.** The reference model is drawn once and
  reused, so every draw's score is correlated through it. This pushes the reported spread in
  the *same* direction as undetected draw dependence — the two caveats compound rather than
  offset.
- **Every default is provisional.** All of them were calibrated against a single rebalance
  date at a crisis onset, chosen because it was the hard case. Whether they generalise to
  calm regimes is unmeasured, which is why each threshold is a field rather than a literal.

## Cost

Ensemble mode multiplies request consumption by the draw count — and by the retry count on
failure, and again by two if a reference model is configured. Find out before you run:

```python
from recall_guard import estimate_cost

estimate_cost(spec, max_retries=2, has_reference=True).worst_case_requests
```

Set `max_total_requests` on the spec to make exhaustion an error rather than a bill. The
endpoint is rate-limited and 429-prone; retries are jittered and honour `Retry-After`, but
**you are responsible for your own quota and terms**.

Raw per-draw replies are **not** retained by default. Turning `retain_draws` on holds every
response in memory, which at 20 top-logprobs per token runs to roughly 12 MB per draw —
about 1.5 GB for a 128-draw ensemble.

## Shelf life

The feature exists because the serving stack is nondeterministic *within* a session. The
distribution being sampled also moves *between* sessions.

Observed on one model: the same prompt against the same model id, two days apart, shifted
one component's median from −0.60 to −0.20 (KS p ≈ 4e-12) while three other components were
statistically unchanged. The mass above the split point went from 12% to 48% — enough that
a fresh ensemble correctly flagged a separated component that a stored corpus said was
unimodal.

Three consequences:

- **A cached consensus is not a fresh one.** `draws_sha256` pins *which* draws produced a
  result; `sampled_at` pins *when*. A result replayed from a stored draw set has
  `sampled_at = None`, because a replay was not sampled.
- **Threshold tuning ages.** Any `mass_min`, `density_ratio`, or `agreement_target` tuned
  against one corpus is tuned against one window of that model's behaviour. This is the
  concrete reason the defaults are documented as provisional.
- **A stored corpus is not ground truth for a fresh run.** A detection that disagrees with
  your archive may be the archive being stale rather than the detector being wrong. Check
  the fresh distribution before assuming a false positive — that mistake has already cost
  someone a debugging cycle.

## Replay

The reduction is a pure function of the draw set: no I/O, no clock, no randomness, no seed.
Draws are canonically ordered by content before anything is computed, and every tie breaks
on content alone — so a stored set replays to a bit-identical result, including the
representative draw and the digest.

`draws_sha256` covers **reply text only**, with a version tag and a separator that stops two
draws being concatenated into one. Logprob structures are excluded: their key ordering comes
from the provider's JSON and is not stable across servers or library versions.

```python
from recall_guard import reduce_draws

replayed = reduce_draws(stored_draws, spec, decide=..., components=...)
assert replayed.draws_sha256 == recorded_hash
```
