# Research & Design Decisions — ensemble-consensus

## Summary

- **Feature**: `ensemble-consensus`
- **Discovery Scope**: Complex Integration — a defect repair in the model client, a new set of
  pure statistical primitives, and a new surface on the public scorer, constrained by an
  existing layering gate, an exact-equality packaging contract, and a bit-for-bit parity test.
- **Key Findings**:
  - The concurrency defect is broader than the source proposal states and is a regression from
    a prior hardening task; the existing test cannot detect it but must not be changed.
  - The proposal's statistical design does not survive contact with its own measurements. Its
    lattice premise, its reduction, its stopping rule, its defaults, and its representative-draw
    rule each fail against the data that motivated them.
  - Ensembling the model call alone is sufficient. The downstream scoring pipeline is already
    bit-deterministic, so no membership-inference or calibrator code changes.

## Research Log

### The pacing defect: scope, provenance, and the correct repair

- **Context**: §9 of the source proposal reports that the client serialises concurrent calls
  and gates the whole feature on fixing it.
- **Sources Consulted**: `recall_guard/core/nvidia_lm.py`, `tests/core/test_nvidia_lm.py`,
  `git blame`/`git log -L` on the pacing block, `.kiro/specs/review-hardening/`, plus direct
  measurement against the real module with a mocked transport.
- **Findings**:
  - The lock acquisition is **unconditional**; only the sleep is gated on a positive interval.
    Serialisation therefore occurs at the default setting where pacing is *disabled*, which is
    what every internal caller uses. The proposal understates the blast radius.
  - Measured, twice independently: 8 requests across 8 workers with a 0.20 s transport and
    pacing off produce a peak of **1** in-flight request and **1.61 s** wall clock; after the
    repair, peak **8** and **0.20 s**.
  - Every fan-out in the package funnels through one executor, so `max_workers` is inert at
    ten call sites, including the CLI's documented default of 8.
  - **Provenance**: the lock was introduced by `review-hardening` task 2.3, whose commit
    literally re-indents the pre-existing request into a new `with` block. That task satisfied
    its Req 2.6 by destroying concurrency. Before it, the pacing block was outside any lock:
    concurrency worked and pacing raced. The regression is scoping-only.
  - The latency-coupled stamp — recording the clock *after* the response returns — predates the
    lock. So the repair changes two things that arrived at different times.
  - **The existing concurrency test passes both before and after the repair and must not be
    modified.** It asserts only a lower bound on request-start gaps against an instant
    transport, so it is structurally blind to serialisation. But it is the only assertion that
    rejects the naive repair variant that stamps the current clock instead of the reserved
    slot: that variant fails it 5 runs out of 5. It is a necessary keeper and an insufficient
    guard.
  - **Correction found during verification**: rate-limited retries are *already* paced today,
    because a 429 is a successful request that stamps the clock before status checking raises.
    Only transport-level exceptions skip the stamp, and a real timeout self-paces by costing
    the full timeout. The genuinely exposed case is a fast-failing connection error. An earlier
    finding that framed this as a 429-safety defect was overstated.
- **Implications**: phase 1 is a slot-reservation scheme, a documented start-to-start contract,
  three new tests, and an explicit decision about the traffic increase that follows from making
  `--max-workers` real.

### The emission lattice premise

- **Context**: the proposal's configuration, scale floor, reduction, and multimodality test all
  assume replies land on a 0.1 lattice.
- **Sources Consulted**: the study's raw draw CSVs, recomputed rather than read from the
  proposal's tables.
- **Findings**: **17.6% of emitted values are off the 0.1 lattice.** The true lattice is 0.01
  with heavy 0.05 structure. The second-most-common value on the tightest axis occurs in 41% of
  its draws and is off-lattice. The proposal's histogram is a 0.1-binned *view*, read back as a
  physical property of the model. Separately, the contamination score is continuous — 96
  distinct values in 100 draws — so the entire lattice apparatus is inapplicable to the consumer
  the proposal itself ranks first.
- **Implications**: the lattice becomes caller-declared and optional, with a reported adherence
  diagnostic so a caller who declares the wrong one finds out.

### Robust location under the measured pathologies

- **Findings**:
  - The scale floor the proposal asks for is **inert on four of five axes**, binding only where
    the median absolute deviation is exactly zero. It does nothing for the over-flagging
    problem the proposal attributes to it, which is a breakdown problem rather than a scale
    problem — no floor of any size touches it.
  - The floor's usual justification, that quantization injects a dispersion of the step over
    the square root of twelve, is **wrong as stated**: a perfectly deterministic value
    quantizes to a single lattice point and shows zero observed dispersion. The correct
    framing is an *identifiability* floor — any true dispersion far below the step produces
    observations on one or two lattice points and is indistinguishable from zero, so an
    estimate below that level carries no information.
  - A caveat that must reach the spec: the floor only does anything **because the declared
    lattice is coarser than the real one**. Evaluated against the true 0.01 lattice, the
    pinned axis returns to the divide-by-zero-equivalent flag rate. It is a smoothing knob,
    not a resolution limit, and a caller who declares the lattice honestly gets no protection
    from it.
  - **No symmetric trim fraction rescues the bimodal axis.** Trimming converges toward the
    median, not toward a mode; even at 25% the estimate lands inside the measured gap. (25% is
    a convention — the interquartile mean — not a maximum: any fraction below one half is
    valid. Note also that the breakdown point of a symmetric trimmed mean equals its trim
    fraction, so 25% gives 0.25, *half* the median's 0.5. The case for that fraction rests on
    measured proximity to the majority mode, not on breakdown.) The proposal's own 10% choice
    sits a few hundredths from the raw mean it was meant to improve on. Trimming is a
    heavy-tail remedy; this is a mixture problem.
  - Winsorizing is strictly worse — it reproduces the raw mean's pathology almost exactly.
  - Harrell–Davis needs an incomplete beta function that numpy does not expose, and its
    smoothing is what puts the estimate back in the gap.
  - **Grid-snapping the location is not a neutral display convention.** Measured over thousands
    of resampled ensembles, it degrades accuracy on all five axes and introduces a systematic
    bias on the best-estimated one, because that axis's estimate sits near a bin edge and always
    rounds the same way. It moves the downstream decision statistic by an amount roughly 40% as
    large as the entire spread between candidate estimators — an unremarked implementation
    detail with more consequence than the choice the proposal agonises over.
- **Implications**: multimodality gates location; location is reported unsnapped; any snapped
  value is a separate, separately-named quantity.

### Multimodality detection

- **Findings**: all three classical tests fail on this data because they assume continuity and
  this data is a discrete mixture.
  - **Hartigan's dip inverts**, and this can be proved without trusting any dip
    implementation. A unimodal distribution is continuous except possibly at its mode, so it
    absorbs at most one atom; the second-largest atom mass therefore lower-bounds the dip,
    while a best-uniform fit upper-bounds it. On the measured data those bounds do not
    overlap: the pinned axis is at least 0.207 while the genuinely bimodal axis is at most
    0.185. The dip is detecting tie mass, not modality — which is exactly backwards on lattice
    data. (Independent verification also showed a reported dip value exceeding the theoretical
    maximum of one quarter, so specific dip *numbers* from any hand-rolled implementation
    should be treated as unreliable; the inequality above is what the spec should cite.) It is
    also seconds-slow on continuous data at the operating point and GPL-2 in every mature
    implementation, against an MIT package.
  - **The bimodality coefficient** flags the tightly-converged axis 36% of the time at low draw
    counts. Its real detector is "skewed and platykurtic", which any bounded or concentrated
    distribution satisfies.
  - **Silverman's test** returns the bootstrap resolution floor on every axis — total false
    positive.
  - **The proposal's own heuristic is the only viable candidate**, being defined natively on the
    lattice so ties are its input rather than its failure mode. But its stated mass threshold
    false-positives on a third axis in 13–31% of ensembles; the minority-mass gap between that
    axis and the genuinely bimodal one is the natural separator, and raising the threshold
    accordingly gives clean separation from roughly fifty draws onward.
- **Implications**: ship the retuned heuristic, expose all three constants as configuration, and
  document the known false-negative class — overlapping modes without a gap are invisible to it.

### Intervals and sequential stopping

- **Findings**:
  - Wilson is the correct interval and the proposal is right here. The naive
    normal-approximation interval **degenerates to zero width at unanimous agreement**, which
    is the typical case at this agreement level — it would report certainty from two dozen
    draws and fire the stopping rule instantly and always. Wilson needs only a square root and
    a normal quantile the standard library already provides.
  - Correction worth recording: the *exact* interval is *not* out of reach. For integer counts
    the binomial tail is a finite sum, so it is a short bisection in pure standard library —
    verified against a reference to fourteen decimal places. More importantly, the
    continuity-corrected Wilson variant is **more conservative than the exact interval** at the
    unanimous operating point, so choosing it "because it is the conservative one" overshoots
    exactness. If conservatism is wanted for a stopping rule, the exact interval is the
    principled choice and is also tighter there.
  - **The tail convention must be pinned, and the proposal never states it.** One-sided, a
    unanimous sample needs 52 draws to certify the target and the rule does fire within the
    proposed cap; two-sided it needs 73 and the cap is exhausted on essentially every prompt.
    The proposal's minimum is infeasible under *either* convention, which is the finding that
    survives — but a feasibility check is meaningless until the tail is declared.
  - **The proposal's stated minimum cannot stop under any convention** (52 one-sided, 73
    two-sided, against its stated 24). Raising the target to be "more careful" produces a
    large cost increase and, at the highest target, zero behavioural change — silently.
  - **Continuous monitoring inflates the error rate about 3.5×** — a nominal 5% becomes a
    measured 17.6%. The reported interval is invalid twice over: selection bias from stopping on
    a favourable wander, and a coverage guarantee that holds only at fixed sample size.
  - **Always-valid alternatives are correct but unusable here.** The measured normal-mixture
    confidence sequence delivers zero miscoverage but needs roughly seven times the proposed cap
    to clear the target; the tighter betting bound also falls short at the cap.
  - **Sequential stopping does not pay for itself.** Best case 18–26% saving, and **zero** for
    the only variant that is statistically valid — with a corrected multiplicity threshold the
    schedule's last look *is* the cap, so it never stops early at all. Meanwhile the point
    estimate is stable well below the proposed cap.
- **Implications**: fixed draw count by default, with the interval then exactly valid; early
  stopping ships as an opt-in fixed schedule with a multiplicity correction and a descriptive
  label.

### The representative draw

- **Findings**: "the draw nearest the decision-space median" is ill-posed here.
  - **At even draw counts a tie exists with probability one, by construction.** The median of
    an even sample is the midpoint of the two middle order statistics, and both are exactly
    equidistant from it. The 73–82% figure measured under exact float equality is an artifact
    of last-bit rounding in that midpoint; with a tolerance the rate is 100%. So the headline
    output is decided by thread completion order in *every* even-count ensemble — this breaks
    the replay contract structurally, not probabilistically.
  - **One callback cannot serve both reductions.** Agreement requires a categorical outcome;
    "nearest the median" requires an ordered scalar. The proposal supplies one projection and
    never says which it is. (An earlier reading of this as a mass tie among 964 draws assumed
    the categorical interpretation; under the scalar reading the proposal actually intends,
    that particular counterexample does not hold. The under-specification is the real defect.)
  - A weaker consistency problem survives on the measured data regardless: the proposal's own
    representative draw disagrees with the per-component median on at least one component,
    with no documented precedence between them.
  - Selecting by minimum decision-space distance is also **ill-conditioned** rather than
    merely tied: many draws sit at near-identical decision distance while lying far apart in
    parameter space, so the selection is unstable under small changes to the projection.
- **Implications**: the projection must declare whether it is categorical or scalar — these are
  two different algorithms. Selection is modal class first, then a within-class parameter-space
  medoid, with ties resolved only by draw contents.

### Determinism hazards

- **Findings**: enumerated and measured — draw arrival order; hash canonicalization; float
  summation order (the same 977 values in different orders yield two distinct results, because
  pairwise summation depends on array layout); the trim index rule; modal ties at rates of
  10–20%; representative-draw ties at 73–82%; even-count medians landing off-lattice in up to
  35% of small ensembles; lattice half-step rounding, where the three obvious spellings disagree
  and the library rounding is inconsistent with itself because the division is inexact in
  binary; cluster-split selection; and the stopping point itself, which under per-draw checking
  depends on arrival order and is therefore unreplayable.
- **Implications**: every one resolves by a rule depending only on draw contents. Exact
  summation replaces order-dependent means. No randomness anywhere on the consensus path, so no
  seed needs storing. Also noted: the endpoint payload carries no seed field, so reproducibility
  is a property of the stored draw set, never of a re-execution.

### The contamination score as the priority consumer

- **Findings**:
  - The feature computation and the calibrator are both **bit-deterministic** — verified over
    hundreds and thousands of repeated calls on identical inputs. The measured dispersion is
    entirely upstream in the serving stack. Ensembling the model call is therefore sufficient
    and no membership-inference code changes.
  - **The median is the wrong default.** Attenuation is linear in the score, so the mean is
    unbiased for expected attenuation; measured, the median of this right-skewed sample sits
    materially lower. Stated as a haircut rather than a pass-through — which is the framing a
    risk reader needs — the mean withholds 21.1% of exposure and the median 13.6%, a **1.55×
    difference in the size of the haircut**. (That the median falls below the mean here is a
    measured fact about this sample, not a theorem; the linearity argument for the mean is
    independent of skew and stands on its own.)
  - A trap in the same area: symmetric trimming moves this estimate **down**, toward the
    unsafe side. So "robustify it" and "be conservative" point in opposite directions here,
    and a conservative estimate must be an explicit upper quantile or bound, never a trimmed
    mean.
  - **Symmetric trimming is actively wrong here**: the right tail is the contamination evidence
    the score exists to report.
  - A useful algebraic fact: because the calibrator is monotone in its linear form, the median
    of the per-draw scores *is* the score of an actual draw at odd draw counts. The mean is not.
  - Reproducibility is relative to a pinned baseline and calibrator; a fresh calibration is
    itself nondeterministic because it drives the model. Ensembling does not fix that.
- **Implications**: report a point estimate, an interval, and an optional conservative upper
  quantile; default the point estimate to the unbiased-for-attenuation choice and say why.

### Failure and cost behaviour

- **Findings**:
  - A rejected credential currently aborts by **substring-matching digits in the error text**,
    and the abort happens after every request has already completed — so one false positive
    discards 127 successful draws and the caller pays for all of them.
  - Retry backoff has **no jitter and ignores any endpoint-supplied delay**, so concurrently
    rate-limited draws retry in unison; with the current retry budget an ensemble can collapse
    within seconds, after which a naive aggregator reports a confident consensus over a handful
    of survivors.
  - Retaining full per-draw responses is roughly 12 MB per draw at the study's token budget,
    because 20 top-logprobs are requested per token — about **1.5 GB** for a full ensemble.
  - Worst-case request counts are understated by the proposal by up to sixfold once retries and
    a reference model are included.
  - Draw failures are **not missing at random**: timeouts preferentially kill long generations,
    and the features divide by reply length, so the surviving subset is systematically shifted
    and measured agreement is biased upward.
- **Implications**: status-based credential classification with early abort; jitter and
  retry-delay handling; incremental reduction with retention off by default; a request budget
  and a pre-execution estimate; and failure counts as reported statistics rather than log lines.

## Architecture Pattern Evaluation

| Option | Description | Strengths | Risks / Limitations | Notes |
|--------|-------------|-----------|---------------------|-------|
| **A. All new modules in `core/`** | Pure statistics and executor at the bottom layer | No gate change, no new boundary stanzas, reachable from every upper layer; precedent exists in the hand-rolled bootstrap module | None identified | **Selected** — verified clean against the gate |
| B. New sibling layer at the middle order | A dedicated `ensemble` layer | Conceptually tidy | Equal-order siblings are forbidden both directions, so the reduction becomes unreachable from the membership-inference layer where a later variance estimator would want it; needs six new boundary stanzas | Rejected |
| C. New layer below `core` | A strict leaf | Universally reachable | Cannot import the model client, so the executor cannot live there; invalidates the "core is the bottom" narrative for no gain | Rejected |
| D. Unregistered new package directory | `recall_guard/ensemble/` | Superficially clean | **The gate skips it entirely** — an unregistered top directory yields no source order and every one of its imports is ignored, so it would be architecturally unconstrained with CI green | Rejected as a trap |

## Design Decisions

### Decision: Reservation rather than exclusion for pacing

- **Context**: the prior hardening task satisfied its pacing requirement by serialising.
- **Alternatives Considered**: 1. Remove the lock (races on shared state, regresses the prior
  requirement). 2. Stamp the current clock and sleep outside the lock (fails the existing test
  5/5, because waiters read a slot a still-sleeping thread already consumed).
  3. Reserve the slot under the lock, sleep and send outside it.
- **Selected Approach**: option 3. The lock covers three arithmetic operations, so the critical
  section scales with neither the interval nor the round-trip time.
- **Rationale**: preserves the prior requirement's guarantee while removing the serialisation;
  verified against the full suite and the architecture gate.
- **Trade-offs**: the pacing contract tightens from end-to-start to start-to-start, becoming
  latency-independent. This is the better semantics and matches how server-side limits are
  enforced, but it is a behaviour change that must be documented. Making worker counts real
  also makes the CLI's default an actual eightfold traffic increase.
- **Follow-up**: the source proposal's sketch is algebraically equivalent to the selected form;
  the spec adopts the explicit formulation for clarity, not because the sketch was defective.

### Decision: The lattice is declared, optional, and diagnosed

- **Rationale**: the assumed lattice holds for only 82.4% of measured values, and the priority
  consumer is continuous.
- **Trade-offs**: callers who genuinely have a lattice must say so; in exchange nobody silently
  snaps continuous data.

### Decision: Multimodality gates location

- **Rationale**: no trim fraction rescues the mixture case, so the location estimator is dead
  code on exactly the component that motivated the design. The proposal presents the two as
  parallel features; they are sequential and the order carries the correctness.

### Decision: Fixed draw count by default

- **Alternatives Considered**: continuous Wilson checking (measured 3.5× error inflation);
  always-valid confidence sequences (correct, but need roughly seven times the cap);
  fixed-schedule with multiplicity correction (valid, saves nothing at the measured agreement);
  fixed count with an exactly valid interval.
- **Selected Approach**: fixed count, with the corrected fixed schedule available as an
  explicitly-labelled opt-in whose interval is marked descriptive.
- **Rationale**: it is simultaneously cheaper than the only valid sequential variant and
  honestly reportable. Recommending an always-valid bound without stating it can never fire at
  this cap would be recommending a feature that never fires.

### Decision: Composition for the ensembled score type

- **Alternatives Considered**: appending optional fields to the existing score; subclassing it.
- **Selected Approach**: a new type containing an unmodified single-draw score.
- **Rationale**: appending forces an unresolvable ambiguity about whether the score field holds
  the representative draw's value or the reduced location, and two consumers will read it two
  ways. Subclassing compares unequal to an identically-valued base instance in both directions,
  because dataclass equality compares classes — a silent trap for downstream fixtures.
  Composition makes the parity contract hold by construction.
- **Trade-offs**: one more public type. Every collection field must be a tuple to preserve
  hashability.

## Risks & Mitigations

- **Correlated draws** — batching, cache reuse, and node affinity plausibly induce positive
  correlation, which makes every reported interval narrower than its label and the stopping rule
  fire too early. Mitigated for now by documenting the independence assumption; a measured
  correlation diagnostic is recorded as an open question rather than shipped.
- **Traffic increase on repair** — making worker counts real turns a documented default of 8
  into genuine concurrency. Mitigated by shipping jitter and retry-delay handling in the same
  phase and documenting the change.
- **Accidental large runs** — a single flag can take a run from hundreds of requests to six
  figures. Mitigated by a required budget that raises on exhaustion, a pre-execution estimate,
  and explicit quota documentation.
- **Memory** — full draw retention is about 1.5 GB per ensemble. Mitigated by incremental
  reduction with retention off by default.
- **Constants tuned on one date** — every default descends from a single crisis-onset rebalance.
  Mitigated by exposing them as configuration and documenting them as provisional.
- **Silent gate bypass** — a new unregistered package directory would be invisible to the
  architecture check. Mitigated by placing everything in an already-registered layer.
- **Pre-existing ceiling violations** — the complexity tool already fails on the current tree and
  is not run in CI, so acceptance must be "no new violations" rather than "clean".

## References

- `Global_Macro_AI_Factors/docs/ensemble_prompt_consensus.md` — the source design proposal.
- The study's raw draw and guard CSVs — every measured figure in this document was recomputed
  from these rather than transcribed from the proposal's tables. One transcription hazard was
  found and avoided: the proposal's vector tables use a component ordering that differs from its
  own earlier table, so positional transcription silently swaps two components.
- `.kiro/specs/review-hardening/` — carried as **precedent**, not as binding requirement IDs:
  those requirements take the review-hardening work as their subject and do not constrain this
  spec by their own terms. The genuinely binding constraints are the declared dependency set in
  `pyproject.toml`, the invariants in the steering document, and the enforcing subprocess test
  that asserts a clean import pulls in no plotting or backtest modules. Req 2.6's pacing
  behaviour is binding in the form that matters — the test that enforces it.
- `.kiro/steering/recall-guard-package.md` — lean runtime, layering, and lazy-import invariants.
- Brown, Cai & DasGupta (2001) on interval estimation for a binomial proportion — the basis for
  preferring the score interval at small samples with a proportion near one.
