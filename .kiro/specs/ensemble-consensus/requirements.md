# Requirements Document

## Introduction

`recall_guard` currently takes one model draw per decision. A 1000-draw measurement study
at `temperature=0` against one production model established that the serving stack is
nondeterministic: 652 of 977 parsed replies were distinct parameter vectors while 98.6% of
them reached the same decision. On the package's own product — the contamination score
`p_memorized`, which multiplies deployed exposure through `memguard_confidence =
raw_confidence · (1 − p_memorized)` — 100 scorings of one identical prompt produced mean
0.211, sd 0.218, range 0.000–0.760. A single scoring's 95% interval spans two thirds of the
unit interval, so as an exposure multiplier it is close to uninformative. Temperature is
already 0; lowering it is not available as a mitigation.

This feature adds an opt-in path that asks the same prompt N times, reduces the draws to one
representative answer plus an explicit and auditable confidence, and refuses to average
across a genuine disagreement. Callers that do not opt in observe no change whatsoever.

Two conditions gate it. First, the LM client currently holds its pacing lock across the
blocking HTTP request, so every concurrent call through one client serialises — and because
the lock is taken unconditionally, this happens even at the default setting where pacing is
disabled. Every `max_workers` argument in the package is therefore inert today. Second, the
source design proposal's statistical machinery does not survive contact with its own
measurements: its emission-grid premise holds for only 82.4% of measured values, its early
stopping rule is either invalid or worthless depending on how it is corrected, its proposed
reduction lands inside the very distributional trough it was designed to avoid, and its
representative-draw rule is a tie with certainty at even draw counts and therefore not
replayable. Those corrections are part of this feature's scope, not deviations from it.

## Boundary Context

- **In scope**: honest concurrency in the LM client; an opt-in, per-call N-draw execution
  path; reduction of a draw set to a location estimate, an agreement measure, and an
  interval; detection and non-collapse of multimodal components; an ensembled contamination
  score; per-draw failure accounting; request budgeting and cost estimation before
  execution; deterministic, replayable reduction over a persisted draw set.
- **Out of scope**: the downstream macro-domain material — the five-axis loadings vector,
  the `REGIME_ASSET_EXPOSURE` asset map, and the `factor_run.v1` / `replay_audit` artifact
  schema — which remain owned by the consuming project; changes to the statistical objective
  of `p_memorized`; new membership-inference features; prompt redesign; new providers; any
  change to the single-draw result for callers who do not opt in.
- **Adjacent expectations**: consumers supply the decision projection that turns a reply into
  a decision, and own their own persistence and artifact schema. `recall_guard` provides the
  reduction, the diagnostics, and a content hash they can reference; it does not define their
  artifact format. This feature must not introduce a second contamination-scoring path: the
  ensemble is a wrapper over the existing scoring behavior, not a parallel implementation of
  it.

## Requirements

### Requirement 1: Genuine concurrency in the LM client

**Objective:** As an operator running any fan-out through `recall_guard`, I want concurrent
model calls to actually execute concurrently, so that the worker count I configure has the
effect its documentation promises.

#### Acceptance Criteria

1. When multiple threads issue requests through one LM client, the client shall allow those
   requests to be in flight simultaneously rather than completing them one at a time.
2. While request pacing is configured to a positive minimum interval, the client shall
   continue to space the start of successive requests by at least that interval.
3. Where request pacing is disabled, the client shall impose no mutual exclusion between
   concurrent requests.
4. When a request attempt fails and is retried, the client shall apply the configured pacing
   contract to the retry attempt.
5. The client shall define the pacing interval as the spacing between the starts of
   successive requests, and this definition shall be stated in its published documentation.
6. The recall_guard test suite shall contain a regression test that fails when concurrent
   requests are serialised and passes when they overlap.
7. When the fan-out entry points are executed with a worker count greater than one, the
   observed wall-clock duration shall scale with the worker count rather than with the
   request count.
8. The recall_guard command-line interface shall default to a single worker, so that upgrading
   to the repaired client does not silently multiply an existing operator's request rate.
9. While the repaired client is in use, the recall_guard package shall apply randomized
   variation to retry delays, and this behavior shall be available in the same release as the
   concurrency repair rather than a later one.

### Requirement 2: Opt-in ensemble execution

**Objective:** As a consumer of `recall_guard`, I want ensembling to be an explicit per-call
choice, so that I can pay for it where a decision warrants it and remain on the cheap path
everywhere else without any hidden change in behavior.

#### Acceptance Criteria

1. Where no ensemble configuration is supplied, the recall_guard scoring and generation
   surfaces shall behave exactly as they do today, including their return types.
2. The recall_guard ensemble path shall be selected by an explicit argument at the call
   site, and shall not be selectable by environment variable, configuration file, or any
   other process-wide setting.
3. The recall_guard package shall not supply an implicit default ensemble configuration; a
   caller who wants an ensemble shall construct and pass one.
4. When an ensemble configuration is constructed with an internally inconsistent or
   unsatisfiable setting, the recall_guard package shall reject it at construction time with
   an error naming the offending setting.
5. If a requested agreement target could not be certified at the requested draw count even
   under unanimous agreement, then the recall_guard package shall reject the configuration at
   construction time rather than accept one whose interval can never clear the target.
6. The recall_guard package shall expose the smallest draw count at which a requested
   agreement target could be certified, so that a caller can see the cost implication of the
   target before executing.
7. The recall_guard published documentation shall describe every default value carried by the
   ensemble configuration as provisional and calibrated on a single measurement date.

### Requirement 3: Reduction of a draw set to a location estimate

**Objective:** As a consumer, I want the ensemble's summary of a component to be a value the
model could plausibly hold, so that I never act on a number the model effectively never
emitted.

#### Acceptance Criteria

1. When a component's draws pass the multimodality check, the recall_guard ensemble path
   shall report a robust location estimate for that component.
2. The recall_guard ensemble path shall report the location estimate without snapping it to
   any lattice, and shall report any lattice-snapped value as a separate, clearly named
   quantity.
3. Where a caller declares an emission lattice, the recall_guard ensemble path shall report
   the fraction of observed draws that actually lie on that lattice.
4. If a caller declares no emission lattice, then the recall_guard ensemble path shall
   complete the reduction without requiring one, and shall report that lattice-dependent
   checks did not run.
5. If a caller declares a non-positive emission lattice, then the recall_guard package shall
   reject the configuration.
6. The recall_guard ensemble path shall report the representative draw as an actual observed
   draw and never as a synthesized combination of draws.
7. When more than one draw qualifies as the representative draw, the recall_guard ensemble
   path shall select among them by a rule that depends only on the draw contents and not on
   the order in which the draws arrived.

### Requirement 4: Refusal to collapse a disagreeing component

**Objective:** As a risk owner, I want the ensemble to tell me when the model is choosing
between two incompatible readings rather than estimating one value, so that a real
disagreement is never laundered into false precision.

#### Acceptance Criteria

1. The recall_guard ensemble path shall test each reduced component for separated clusters
   before computing any location estimate for it.
2. When a component is found to hold separated clusters, the recall_guard ensemble path
   shall name that component in its result.
3. When a component is found to hold separated clusters, the recall_guard ensemble path
   shall not report a location estimate that falls between those clusters.
4. The recall_guard ensemble path shall default to flagging a disagreeing component and
   returning control to the caller, and shall never silently average across one.
5. Where a caller explicitly requests it, the recall_guard ensemble path shall instead raise
   an error on a disagreeing component.
6. The thresholds governing cluster detection shall be caller-configurable rather than fixed
   in the implementation.
7. The recall_guard published documentation shall state that the detection identifies
   separated clusters only, and that overlapping modes without a gap are not detected.

### Requirement 5: Agreement measurement and honest intervals

**Objective:** As an auditor, I want the confidence an ensemble reports to mean what its
label says, so that a stated interval is not systematically narrower than its true coverage.

#### Acceptance Criteria

1. The recall_guard ensemble path shall measure agreement on the decision produced by a
   caller-supplied projection, not on the parameter values themselves.
2. The recall_guard ensemble path shall report an interval around the measured agreement
   that remains within the zero-to-one range and is never zero-width at unanimous agreement.
3. The recall_guard ensemble path shall run to a fixed draw count, so that the reported
   agreement interval is valid at its stated confidence level without correction.
4. The recall_guard published documentation shall state that the reported interval assumes
   draws are independent, and that correlated draws make it narrower than its label.
5. If a caller supplies a projection that raises on a draw, then the recall_guard ensemble
   path shall record that draw as a distinct projection failure rather than as a transport or
   parse failure.
6. The recall_guard ensemble configuration shall require the interval's tail convention to be
   declared, and shall evaluate any feasibility check against that declared convention.
7. The recall_guard ensemble path shall report a measure of dependence between draws collected
   at different points in the run, so that a caller can detect when the independence
   assumption behind the reported interval is violated rather than only being warned of it.
8. The recall_guard ensemble path shall record, for each draw, the collection group it came
   from, so that the dependence measure is reproducible from the stored draw set alone.

### Requirement 6: Ensembled contamination score

**Objective:** As a consumer who scales exposure by the contamination score, I want that
score estimated from an ensemble with a reported interval, so that a single near-uninformative
reading cannot silently set my position size.

#### Acceptance Criteria

1. Where an ensemble configuration is supplied to the guarded scoring surface, the
   recall_guard package shall return a contamination score reduced over the draw set together
   with an interval around it.
2. The recall_guard package shall derive each draw's contamination score through the existing
   single-draw scoring behavior and reduce the resulting scores, rather than combining
   intermediate quantities and scoring once.
3. When an ensemble of exactly one draw is requested, the resulting contamination score shall
   be identical to the score the single-draw path returns for the same reply.
4. The recall_guard package shall not apply a symmetric trimming reduction to contamination
   scores, because the upper tail of that distribution is the evidence the score exists to
   report.
5. The recall_guard package shall default the reported point estimate to an estimator that is
   unbiased for the expected attenuation, and shall document why a lower estimator would
   under-attenuate exposure.
6. Where a caller requests it, the recall_guard package shall report a conservative upper
   quantile of the contamination score in addition to the point estimate.
7. When a reference model is configured, the recall_guard package shall state whether the
   reference draw varies per ensemble draw or is held fixed, and shall account for the
   resulting request count.
8. The existing test that pins the guarded score to the batch evaluation path shall remain
   unchanged and passing.
9. The recall_guard package shall designate exactly one reported contamination value as the
   exposure multiplier, and its published documentation shall state that the representative
   draw's own score is evidence and is not the multiplier.
10. Where a single reference-model draw is shared across the ensemble, the recall_guard
    published documentation shall state that the reported dispersion understates the true
    dispersion, because every draw's score is then correlated through that shared reference.
11. The recall_guard published documentation shall express the difference between candidate
    contamination estimators as the resulting difference in withheld exposure, rather than
    only as a difference in the score itself.

### Requirement 7: Failure accounting and refusal to mint false success

**Objective:** As an operator, I want an ensemble that mostly failed to be reported as a
failure, so that a confident-looking consensus is never computed from a handful of survivors.

#### Acceptance Criteria

1. The recall_guard ensemble path shall report the number of draws requested and the number
   that produced a usable reply as separate quantities.
2. The recall_guard ensemble path shall report failure counts broken down by the failure
   categories the package already distinguishes, rather than as a single total.
3. If the number of usable draws falls below the configured minimum, then the recall_guard
   ensemble path shall report a failure rather than a consensus computed over the survivors.
4. If every draw fails, then the recall_guard ensemble path shall report a failure with no
   contamination score, and shall not report a score of zero.
5. If the proportion of transport failures exceeds a configured threshold, then the
   recall_guard ensemble path shall report a failure rather than a high-agreement result over
   the remainder.
6. When a draw fails because the endpoint rejected the credential, the recall_guard package
   shall classify it from the response status rather than by matching text in the error
   message.
7. When the credential is rejected, the recall_guard ensemble path shall stop issuing further
   draws for that ensemble rather than completing the full draw count.
8. The recall_guard published documentation shall state that draw failures are not
   independent of the answer, and that a measured agreement is conditional on the draws that
   parsed.

### Requirement 8: Cost visibility and request budgeting

**Objective:** As an operator paying per request, I want the cost of an ensemble to be
knowable before it runs and bounded while it runs, so that enabling the feature cannot
produce an accidental six-figure request run.

#### Acceptance Criteria

1. The recall_guard ensemble path shall accept a total request budget and shall stop with an
   error when that budget is exhausted.
2. The recall_guard package shall provide a way to obtain the worst-case request count and an
   estimated duration for a given ensemble configuration without issuing any request.
3. The reported worst-case request count shall account for retry attempts and for a
   configured reference model, not only for the nominal draw count.
4. The recall_guard ensemble path shall default its concurrency to the value already used
   elsewhere in the package rather than to a higher value.
5. When the endpoint signals that the caller is being rate-limited, the recall_guard package
   shall vary its retry delays randomly so that concurrent draws do not retry in unison.
6. When the endpoint supplies a retry delay, the recall_guard package shall honour that delay.
7. The recall_guard published documentation shall state that ensemble mode multiplies request
   consumption by the draw count, that the endpoint is rate-limited, and that the operator is
   responsible for their own quota and terms.
8. The recall_guard ensemble path shall not retain full per-draw response data by default,
   and its published documentation shall state the memory cost of retaining it.

### Requirement 9: Deterministic, replayable reduction

**Objective:** As an auditor, I want a stored draw set to reduce to the same consensus every
time it is replayed, so that an ensemble result can be verified after the fact without
re-querying the model.

#### Acceptance Criteria

1. The recall_guard ensemble reduction shall be a function of the draw set alone, producing
   identical output for identical input.
2. The recall_guard ensemble reduction shall not depend on the order in which draws were
   received.
3. The recall_guard ensemble reduction shall contain no random resampling, and shall
   therefore require no random seed.
4. When two or more candidates tie at any selection step, the recall_guard ensemble path shall
   resolve the tie by a rule that depends only on draw contents.
5. The recall_guard ensemble path shall report a content hash over the ordered draw set, and
   its published documentation shall state exactly what the hash covers.
6. The recall_guard test suite shall contain a test that reduces a fixed stored draw set and
   asserts an identical result, without contacting any model.

### Requirement 10: Preservation of the existing package boundary

**Objective:** As a maintainer, I want this feature to land without loosening the constraints
the package already enforces, so that its lean runtime and layering guarantees survive.

#### Acceptance Criteria

1. The recall_guard package shall add no new runtime dependency.
2. The recall_guard package shall continue to import without pulling in the plotting or
   backtest stacks.
3. The recall_guard package shall continue to satisfy its architectural layering check, and
   any new code location shall be registered with that check rather than placed where the
   check does not inspect it.
4. The recall_guard package shall introduce no new violation of its configured complexity and
   function-length ceilings.
5. The recall_guard package's existing public types shall remain importable and constructible
   as they are today.
6. The recall_guard published documentation shall describe the ensemble surface, and the
   documentation build shall remain strict and passing.
