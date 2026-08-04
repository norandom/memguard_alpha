# Implementation Plan

Tasks follow the design's dependency order: the concurrency repair first, because without it
every fan-out in the package runs one request at a time and an ensemble is unusable; then the
pure statistics, which have no I/O and can be built and tested independently; then
configuration, execution and reduction; then the contamination-score integration; and finally
replay, boundary, and documentation validation.

- [x] 1. Repair concurrency in the model client
- [x] 1.1 Replace mutual-exclusion pacing with slot reservation
  - Change the pacing mechanism so the shared lock covers only the reservation bookkeeping and is released before the client waits or issues the request.
  - Define the pacing interval as the spacing between the starts of successive requests, and record the reserved send time rather than the observed completion time.
  - Ensure an idle client cannot accumulate credit and then issue a burst.
  - Ensure each retry attempt reserves its own pacing slot, so a retried request is spaced like any other rather than firing immediately.
  - Lower the command-line worker default to one, so that upgrading does not silently multiply an existing operator's request rate against a rate-limited endpoint.
  - Done looks like concurrent requests through one client overlapping in flight, with the configured minimum spacing still observed between request starts and no longer inflated by request latency.
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.8_
  - _Boundary: core.nvidia_lm, harness.runner_
- [x] 1.2 Harden retry behaviour under rate limiting
  - Add randomized variation to retry delays so concurrently rate-limited requests do not all retry at the same instant.
  - Honour a retry delay supplied by the endpoint when one is present.
  - Surface the response status on the raised failure so a rejected credential can be classified from it rather than by matching text in the error message, and consume that status at the classification site.
  - Release this together with the pacing repair; shipping real concurrency before jitter would put synchronized retry storms in front of operators in the interval between the two.
  - Done looks like concurrently rate-limited requests retrying at dispersed times, and an error message containing digits that resemble an authorization status no longer being treated as a credential rejection.
  - _Requirements: 1.9, 7.6, 8.5, 8.6_
  - _Boundary: core.nvidia_lm, harness.scorer_
- [x] 1.3 Add concurrency regression coverage
  - Add a test that counts how many requests are simultaneously in flight when several workers share one client with pacing disabled, and requires genuine overlap.
  - Add a test that request starts remain spaced by the configured interval and are not stretched by request latency.
  - Add a test that a failed attempt still consumes a pacing slot before its retry, injecting a connection-level failure rather than a rate-limited response.
  - Preserve the existing concurrent-pacing test's discriminating power; it is the only assertion that rejects a reservation scheme which stamps the current clock instead of the reserved slot. If scheduler jitter makes it flaky once requests are no longer serialised, widen its tolerance or assert on reserved slots rather than weakening it into an assertion the naive variant would also pass.
  - Done looks like the new overlap test failing against the pre-change client and passing after it, with the whole existing client test file still green.
  - _Requirements: 1.6, 1.7_
  - _Boundary: core.nvidia_lm_

- [x] 2. Build the pure consensus statistics
- [x] 2.1 Vendor the measured pathologies as a test fixture
  - Copy the measured draw set and contamination scores into the test fixtures directory, carrying only the numeric columns — the draw index, the parse flag, the component values, and the scores — so no prompt or reply text is duplicated.
  - Record the provenance of the data in the fixture directory, including that it comes from a single rebalance date.
  - Add a loader helper the statistics tests can share.
  - Done looks like the pathology corpus loadable from within this repository with no dependency on any sibling project, at a size in line with the existing fixtures.
  - _Requirements: 3.1, 4.1_
  - Note: 2.2-2.5 lost their `(P)` markers during implementation. All four share the single `core.consensus` module, so they contend on one file and are not parallel-safe; the annotation was wrong when written.
- [x] 2.2 Agreement measurement and intervals
  - Implement a score-based proportion interval that stays within the zero-to-one range and never collapses to zero width at unanimous agreement, plus its continuity-corrected variant.
  - Obtain the normal quantile from the standard library rather than adding a numerical dependency.
  - Cover the unanimous case at several sample sizes, and record why an interval built on the naive normal approximation is rejected.
  - Done looks like a unanimous sample of any size producing a non-degenerate interval bounded inside zero and one.
  - _Requirements: 5.2, 5.3_
  - _Boundary: core.consensus_
- [x] 2.3 Lattice handling and adherence reporting
  - Implement optional lattice snapping with a single pinned rounding direction that does not depend on floating-point division by the lattice step.
  - Implement a diagnostic reporting the fraction of observed values that actually lie on a declared lattice.
  - Cover every half-step tie value across the unit range, asserting one consistent rounding direction.
  - Done looks like a caller who declares a lattice the data does not follow seeing a low adherence figure rather than silently mis-snapped values.
  - _Requirements: 3.3, 3.5_
  - _Boundary: core.consensus_
- [x] 2.4 Robust location estimation
  - Implement the location estimators the design permits, with order-independent summation so a reordered input cannot change the last bit of the result.
  - Pin the index rule for any trimming so the retained core is reproducible.
  - Use the measured pathologies as the test corpus: a component whose median absolute deviation is exactly zero, and a component where every symmetric trim fraction lands inside the observed gap.
  - Done looks like the same values in any order producing a bit-identical location estimate.
  - _Requirements: 3.1, 9.1, 9.2, 9.3_
  - _Boundary: core.consensus_
- [x] 2.5 Separated-cluster detection
  - Implement detection of two clusters separated by a gap, parameterized by minimum cluster mass, minimum gap width, and the density ratio between cluster peaks and the gap.
  - Return a defined outcome indicating the check did not run when no lattice is declared, rather than silently reporting no clusters.
  - Pin the selection rule when several candidate splits qualify, using only the values themselves.
  - Cover the measured bimodal component at several draw counts as a required detection, and the tightly-converged component as a required non-detection.
  - Done looks like the bimodal component being flagged and the converged component not being flagged, at draw counts spanning the intended operating range, and a reduction with no declared lattice completing while reporting that the check did not run.
  - _Requirements: 3.4, 4.1, 4.6, 9.4_
  - _Boundary: core.consensus_

- [x] 3. Build ensemble configuration, execution and reduction
- [x] 3.1 Ensemble configuration with validation and feasibility checking
  - Define the ensemble configuration as an explicitly constructed value with no package-supplied default instance.
  - Reject inconsistent settings at construction time with an error naming the offending setting, following the client's validate-on-construct convention rather than silently clamping.
  - Reject an agreement target that could never be reached within the requested maximum draw count even under unanimous agreement, and expose the earliest draw count at which the target could be reached.
  - Document every default value as provisional and calibrated on a single measurement date.
  - Done looks like an unreachable agreement target being refused at construction with a message naming the smallest feasible draw count, instead of being accepted and never firing.
  - Require the interval's tail convention to be declared, and evaluate the feasibility check against that convention rather than an implied one.
  - _Requirements: 2.3, 2.4, 2.5, 2.6, 2.7, 3.5, 4.6, 5.6_
  - _Boundary: core.ensemble_
- [x] 3.2 Cost estimation and request budgeting
  - Provide a way to obtain the worst-case request count and an estimated duration for a configuration without issuing any request.
  - Include retry attempts and a configured reference model in the worst-case count, not only the nominal draw count.
  - Enforce a total request budget with a counter that stops the ensemble with an error when exhausted.
  - Default the concurrency to the value already used elsewhere in the package rather than a higher one.
  - Done looks like a caller obtaining a worst-case request count and duration before any network activity, and an ensemble that exceeds its budget stopping with an error rather than continuing.
  - _Requirements: 8.1, 8.2, 8.3, 8.4_
  - _Boundary: core.ensemble_
- [x] 3.3 Wave-based draw execution and failure triage
  - Execute draws in waves sized to the configured concurrency, preserving submission order within each wave.
  - Label each draw with the wave it was collected in, and carry that label alongside the draw so a later dependence measure can be recomputed from stored data alone.
  - Triage each result into a usable draw, a transport failure, a parse failure, or a caller-projection failure, keeping the projection failure distinct from the other two.
  - Record the number of draws requested and the number usable as separate quantities, with failure counts broken down by the categories the package already distinguishes.
  - Stop the ensemble with a failure when usable draws fall below the configured minimum, when every draw fails, or when transport failures exceed the configured proportion.
  - Stop issuing further draws when the endpoint rejects the credential, rather than paying for the remaining draws.
  - Do not retain full per-draw response data unless explicitly requested; reduce incrementally instead.
  - Done looks like an ensemble in which most draws failed reporting a failure with no contamination score, rather than a confident consensus computed from the survivors.
  - _Requirements: 5.5, 5.8, 7.1, 7.2, 7.3, 7.4, 7.5, 7.7, 8.8_
  - _Boundary: core.ensemble_
- [x] 3.4 Deterministic reduction, tie-breaking and draw-set hashing
  - Sort the collected draws into a canonical order derived from their contents before any reduction step, so arrival order cannot influence any result.
  - Run the separated-cluster check before computing any location estimate, and omit the location for any component the check flags, so no reported location falls inside a detected gap.
  - Report the location unsnapped, and report any lattice-snapped value as a separately named quantity.
  - Select the representative draw as an actual observed draw, using the modal decision class and then a within-class parameter-space selection, with ties resolved only by draw contents.
  - Compute a content hash over the ordered sequence of reply texts with an explicit version tag, separator and encoding, excluding timing and thread identity.
  - Done looks like the same draw set, shuffled, reducing to a bit-identical result including the same representative draw and the same hash.
  - _Requirements: 3.2, 3.6, 3.7, 4.2, 4.3, 4.4, 4.5, 5.1, 9.4, 9.5_
  - _Boundary: core.ensemble_

- [x] 3.5 Draw-dependence diagnostic
  - Measure the association between decision labels across collection-group boundaries, and report it alongside the agreement interval.
  - Compute it from the stored group labels rather than from arrival order, so it replays identically from a persisted draw set; keep the labels out of the content hash, which continues to cover reply text only.
  - Report an undefined result rather than a misleading zero when there are too few groups to measure.
  - Do not adjust the reported interval from this measurement; the diagnostic exists to make the independence assumption falsifiable, not to correct for it.
  - Done looks like a draw set whose decisions cluster by collection group reporting visible dependence, and an independently-shuffled set reporting none, with both reproducible from stored data.
  - _Requirements: 5.7, 5.8_
  - _Boundary: core.consensus, core.ensemble_

- [ ] 4. Integrate the ensembled contamination score
- [ ] 4.1 Ensembled score type and entry point
  - Add a result type that contains an unmodified single-draw guarded score as its representative draw alongside the reduced score, its interval, agreement, flagged components and counters, using only hashable collection types.
  - Add a scoring entry point that takes an explicitly supplied ensemble configuration and leaves the existing scoring signatures and return types untouched.
  - Derive each draw's contamination score through the existing single-draw scoring behaviour and reduce the resulting scores, rather than combining intermediate quantities and scoring once.
  - Default the reported point estimate to the estimator that is unbiased for the expected attenuation, forbid symmetric trimming of contamination scores, and offer a conservative upper quantile on request.
  - Hold the reference-model draw fixed across the ensemble, account for it in the request count, and record in the docstring that a shared reference correlates every draw's score and therefore understates the reported dispersion.
  - Done looks like an ensemble of exactly one draw producing a contamination score identical to the single-draw path for the same reply, with the existing parity test unchanged and passing.
  - Designate exactly one reported value as the exposure multiplier, rank the representative draw so the two agree where possible, and document that the representative draw's own score is evidence rather than the multiplier.
  - _Requirements: 2.1, 2.2, 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7, 6.8, 6.9_
  - _Boundary: harness.scorer_
- [ ] 4.2 Failure semantics for the ensembled score
  - Report no contamination score when the ensemble fails, and never substitute zero, which would mean passing full exposure through.
  - Carry the failure category breakdown and the requested-versus-usable counts onto the returned score.
  - Cover the cases where all draws fail, where most draws fail, and where one draw returns a rejected credential.
  - Done looks like a failed ensemble returning an absent contamination score with its failure counts intact, matching how the single-draw path already signals failure.
  - _Requirements: 7.1, 7.2, 7.3, 7.4_
  - _Boundary: harness.scorer_
- [ ] 4.3 Expose the new surface on the public API
  - Re-export the new public names from the package root and the core layer, keeping the name lists that act as typo guards in step with the imports.
  - Use absolute, module-direct imports in the new modules, avoiding relative imports and root-level re-imports that the architecture check does not inspect and that would introduce a cycle.
  - Done looks like the new names importable from the package root with the existing public types still importable and constructible exactly as before.
  - _Requirements: 2.1, 10.5_
  - _Depends: 3.4, 4.1_
  - _Boundary: core.ensemble, harness.scorer_

- [ ] 5. Validate the feature end to end
- [ ] 5.1 Replay determinism verification
  - Add a test that reduces a fixed stored draw set to an identical result without contacting any model, including the representative draw and the content hash.
  - Confirm the reduction contains no random resampling and therefore requires no seed.
  - Done looks like a stored draw set replaying to a bit-identical result across repeated runs and across input reorderings, with no model call.
  - _Requirements: 9.1, 9.2, 9.3, 9.6_
  - _Depends: 3.4_
  - _Boundary: core.ensemble, core.consensus_
- [ ] 5.2 Package boundary verification
  - Confirm the runtime dependency set is unchanged and that importing the package still avoids the plotting and backtest stacks.
  - Confirm the architectural layering check passes with the new modules in a registered location rather than one the check skips.
  - Confirm no new violation of the configured complexity and function-length ceilings, measured against the existing baseline rather than against a clean tree.
  - Done looks like the dependency and layering checks passing unchanged and the complexity report showing no new violations beyond those already present.
  - _Requirements: 10.1, 10.2, 10.3, 10.4_
  - _Depends: 4.3_
- [ ] 5.3 Document the ensemble surface and its operational cost
  - Describe the ensemble surface, the pacing contract change, and the meaning of each reported diagnostic.
  - State that the reported interval assumes independent draws and is narrower than its label when draws are correlated, that agreement is conditional on the draws that parsed, and that draw failures are not independent of the answer. Explain what the dependence diagnostic measures and how to read it.
  - Present the choice between contamination estimators as a difference in withheld exposure rather than a difference in score, with a worked example on the measured data.
  - State that a shared reference-model draw correlates every draw's score, that this understates reported dispersion, and that it compounds with rather than offsets undetected draw dependence.
  - State that the separated-cluster detection finds only clusters with a gap between them, and does not detect overlapping modes.
  - State that ensemble mode multiplies request consumption by the draw count, that the endpoint is rate-limited, that the operator owns their own quota and terms, and what retaining full per-draw data costs in memory.
  - Done looks like the strict documentation build passing with the ensemble surface described and each of these caveats present.
  - _Requirements: 4.7, 5.4, 6.10, 6.11, 7.8, 8.7, 8.8, 10.6_
  - _Depends: 4.3_

## Implementation Notes

Carried forward from completed tasks. Read these before starting a task whose boundary
overlaps.

**From 1.1–1.3 (concurrency repair):**

- The client now raises `LMHTTPError`, a `RuntimeError` subclass carrying `status_code`
  (`None` for transport-level failures). Later tasks that triage draws should read that
  attribute rather than matching text; `except RuntimeError` handlers are unaffected.
- Retry backoff is now fully jittered, so retry timing is nondeterministic by design. Any
  test asserting on backoff must patch `time.sleep` or set `retry_backoff_s=0.0`.
- `tests/core/test_nvidia_lm.py::test_concurrent_pacing_enforces_min_interval` must stay. It
  is the only assertion that rejects a reservation scheme stamping the current clock instead
  of the reserved slot, and it passes both before and after the repair.
- `generate_many` calls `lm.generate(prompt)` with no arguments and drives its executor with
  `ex.map`, which blocks until every future resolves. The ensemble executor in task 3.3 must
  be a new primitive — waves, budget enforcement, and credential abort are all inexpressible
  over a call that returns everything at once.
- The LM test doubles across the suite (`_FakeLM.generate(self, prompt, temperature=0.0)`)
  accept no token-budget parameter. Changing the executor's call shape breaks them; the
  measured dispersion occurs at the shipped defaults, so there is no reason to.
- Task 1.2's `_Boundary:` was corrected during review: status-based credential classification
  cannot be done inside `core.nvidia_lm` alone, because the classifier lives in
  `harness.scorer`. Check boundary annotations against where the behaviour actually lives.
- **Open decision, deliberately not taken here.** The library fan-out defaults
  (`generate_many`, `score_many`, `calibrate`) remain `max_workers=8`. Only the CLI default
  was lowered, per Req 1.8. A library consumer therefore goes from one effective request at a
  time to eight on upgrade. Left alone because the downstream consumer built a thread-local
  workaround specifically to obtain this concurrency — and that workaround is now unnecessary
  and unsafe, since independent pacers multiply the provider rate limit by the worker count.
  Worth notifying that project when this ships.
