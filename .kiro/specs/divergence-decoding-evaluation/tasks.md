# Implementation tasks: divergence decoding evaluation

Task status is recorded below. Each numbered executable task is a focused work unit
targeting 1–3 hours; corpus availability and hardware validation may require
external work beyond that estimate. Execute in dependency order. No parallel
markers are used because shared contracts and the experimental runtime evolve
together. Live execution requires explicit approval of its concrete budget.

Every code task includes focused Docker verification and review of idiomatic
Python, role-specific names, units and array axes. No new DataFrames are required.
Do not mark a task complete solely because its scaffolding exists.

## 1. Contracts and execution environment

- [x] 1.1 Define experiment contracts and credential-free planning
  - Add typed plan, corpus, auxiliary, operation and item contracts in the experimental package; pin mandatory target validation and schema versions.
  - Implement local-only planning with unresolved prerequisites reported explicitly; validate output limits, budgets and phase identity.
  - Done when offline fixtures round-trip, invalid target/settings fail before model access, and planning performs no downloads or inference.
  - _Boundary: Experimental contracts and CLI planning_
  - _Depends: None_
  - _Requirements: 1, 4, 6, 7_

- [x] 1.2 Add the optional locked Docker experiment environment
  - Select compatible pinned inference packages in the existing lock and add an optional target to the canonical Dockerfile; exclude experimental code from the library wheel.
  - Record image identity and dependency versions. Verify lightweight imports without inference packages and the experimental import path in Docker.
  - Done when the optional image builds and imports, default library dependency isolation passes, and no weights were fetched by build/tests.
  - _Boundary: Optional environment and packaging_
  - _Depends: 1.1_
  - _Requirements: 1, 4, 7_

## 2. Corpus admission and auxiliary artifacts

Tasks 2.1 and 2.2 proceeded using the validated development image during task
1.2's dependency download. Its build/import acceptance has now passed independent
review; the temporary scheduling exception has ended.

- [x] 2.1 Validate temporal provenance and partition boundaries
  - Implement manifest checks for rights references, publication/revision evidence, content hashes, source/window counts and duplicated held-out documents.
  - Record semantic audit attestations and intentional shared fact exposure separately; generate the deterministic matched-size control manifest with actual size discrepancy.
  - Done when date/revision/overlap failures are retained and valid fixture corpora reproduce source and period counts.
  - _Boundary: Corpus preparation_
  - _Depends: 1.1_
  - _Requirements: 2, 7_

- [ ] 2.2 Fit and persist the two sparse trigram estimators
  - Implement the designed interpolation with outgoing-transition denominators, document resets, short-prefix backoff and finite full-vocabulary vectors.
  - Enforce fitting volume/memory caps, preserve tokenizer identity and serialize sparse counts without unsafe object deserialization.
  - Done when hand-calculated tiny corpora verify normalization, unseen contexts and document-end behavior, and artifact reload reproduces scores.
  - _Boundary: Auxiliary estimator_
  - _Depends: 2.1_
  - _Requirements: 1, 2, 3, 4, 7_

## 3. Real-model callbacks and steering integration

- [ ] 3.1 Implement the GPT-OSS runtime callback
  - Load only explicit local pinned artifacts, apply the pinned Harmony template and expose final-position full logits through the existing ModelSource contract.
  - Validate token mappings/output-head slots and manage per-arm caches with prefix-extension checks and resets; classify unsupported precision/kernel settings.
  - Done when fake-model fixtures verify cache reset/rebuild and slot compatibility and the adapter imports in the optional Docker image. Real-device acceptance remains task 6.2.
  - _Boundary: Main-model runtime adapter_
  - _Depends: 1.2_
  - _Requirements: 1, 3, 4, 7_

- [ ] 3.2 Integrate auxiliary callbacks, Harmony handling and greedy decoding
  - Reuse decode_greedy and combine_logits, preserving full-prefix semantics and finite scores; neutralize structural/header/padded-slot auxiliary differences according to the design.
  - Parse final-channel output and record absent final answers, truncation and unmapped-token failures. Capture same-prefix changed-token evidence and paired changed answers without publishing raw reasoning.
  - Done when deterministic fixtures prove zero-alpha/sham parity, header transitions, EOS handling and nonzero steering while preserving failure states.
  - _Boundary: Explicit decoder integration_
  - _Depends: 2.2, 3.1_
  - _Requirements: 3, 5, 6, 7_

## 4. Bounded execution and persistence

- [ ] 4.1 Implement phase admission and durable operation accounting
  - Validate frozen plan hashes and per-phase limits; write intents and terminal records, reject existing destinations and retain unknown/incomplete work.
  - Include fitting, screening, development, parity, controls and evaluation in resource ledgers; report unknown costs explicitly.
  - Done when offline failures prove no silent replay, no budget reset, atomic summaries and accurate requested/completed/unperformed counts.
  - _Boundary: Orchestration and persistence_
  - _Depends: 1.1, 2.2_
  - _Requirements: 4, 6, 7_

- [ ] 4.2 Integrate cancellation and isolated worker termination
  - Check deadlines/cancellation in callbacks and before new operations; parent owns the ledger and terminates an unresponsive worker after the frozen grace period.
  - Preserve DecodeError role/step and distinguish cancelled or incomplete work from a successful partial answer.
  - Done when fake blocked workers and callback failures verify bounded termination and retained evidence without requiring a GPU.
  - _Boundary: Explicit runtime/orchestration integration_
  - _Depends: 3.2, 4.1_
  - _Requirements: 3, 4, 6_

## 5. Experimental protocol and replayable analysis

- [ ] 5.1 Implement screening, development selection and paired scheduling
  - Freeze screening thresholds, alpha grid, selection rule and task strata before execution; halt efficacy interpretation without a baseline recall signal or qualifying development alpha.
  - Alternate held-out arm order deterministically and keep labels out of model inputs; keep recall and financial-direction metrics distinct.
  - Done when fixture runs demonstrate no held-out tuning, correct 8+8 paired scheduling and explicit inconclusive outcomes.
  - _Boundary: Experimental protocol_
  - _Depends: 3.2, 4.2_
  - _Requirements: 2, 5, 7_

- [ ] 5.2 Implement paired analysis and evidence reports
  - Add all-planned/common-usable metrics, lexical recall, task/cohort coverage and paired cohort-stratified bootstrap with frozen seed/count.
  - Apply gap/utility criteria only when matching evidence is valid; flag tiny/degenerate samples, expose control effects, unknown costs and unavailable MCS.
  - Done when fixtures with different arm failures and invalid matching reproduce denominators and unavailable outcomes from saved artifacts without model access.
  - _Boundary: Analysis and reporting_
  - _Depends: 5.1_
  - _Requirements: 5, 6, 7_

- [ ] 5.3 Validate the complete offline workflow and write usage documentation
  - Exercise planning through fake fitting/generation/analysis in Docker; verify package isolation, architecture, focused tests and Ruff.
  - Document separate execution phases, model limits, artifact schemas and honest efficacy interpretation using COBESY; run strict MkDocs for public documentation changes.
  - Done when an offline end-to-end receipt and source-matched review show requirements coverage. Perform one broader regression at this integration boundary if changes justify it.
  - _Boundary: Explicit offline integration and documentation_
  - _Depends: 5.2_
  - _Requirements: 1, 2, 3, 4, 5, 6, 7_

## 6. Separately gated real-model validation and pilot

- [ ] 6.1 Prepare a concrete runtime-smoke resource proposal
  - Inventory available hardware read-only; pin model/tokenizer revisions and propose download size, precision, context/output limits, operation counts and maximum compute time/cost.
  - Price paid resources from current sources where needed; list unknowns rather than infer capacity from checkpoint size. Include a declared memory margin, worker grace period and capped fitting of tiny synthetic auxiliary corpora solely for smoke validation.
  - Done when the exact proposed download and smoke operations are reviewable. Do not download weights or provision paid resources until approved.
  - _Boundary: Resource preflight proposal_
  - _Depends: 1.2, 3.1_
  - _Requirements: 1, 4, 6_

- [ ] 6.2 Run the approved bounded real-model smoke
  - After approval of 6.1, acquire pinned artifacts and measure cold/warm latency, peak host/device memory and parsing at declared limits.
  - Fit the approved tiny synthetic smoke auxiliaries and verify cached/uncached behavior, zero-alpha parity, sham cancellation and same-prefix nonzero steering with real GPT-OSS logits. These artifacts do not use or replace study corpora.
  - Done when measured receipts establish runtime feasibility; failure leaves the pilot blocked without model/precision substitution. Record no claim of mitigation efficacy.
  - _Boundary: Real-runtime acceptance_
  - _Depends: 5.3, 6.1_
  - _Requirements: 1, 3, 4, 6_

- [ ] 6.3 Prepare a concrete data proposal and frozen split manifests
  - Identify accessible sources with auditable dates and rights; construct candidate older/newer and control manifests without fitting them.
  - Specify screening criteria and independent screening/development/evaluation splits, the 8+8 pilot, labels and task-matching rules; preserve human audit decisions.
  - Done when a source-backed proposal is reviewable with volume estimates, or a precise missing-source/audit prerequisite is recorded; unresolved prerequisites keep this task open.
  - _Boundary: Study data preparation_
  - _Depends: 2.1_
  - _Requirements: 2, 5, 6_

- [ ] 6.4 Prepare the complete study execution budget and freeze the plan
  - Combine audited corpus proposal and smoke measurements into fitting, screening, development, controls and held-out phase caps, including all preprocessing and generation work.
  - Freeze revisions, manifests, parser, settings, alpha-selection criteria, task matching and the 32-generation held-out subtotal; retain separate total study accounting.
  - Done when the full plan is reviewable and any required human corpus audits are complete. Obtain explicit execution approval before study-corpus fitting or study generation.
  - _Boundary: Study admission proposal_
  - _Depends: 6.3_
  - _Requirements: 1, 2, 4, 5, 6_

- [ ] 6.5 Fit and verify the approved study auxiliary artifacts
  - Fit the older, newer and matched-size control artifacts within the approved study caps; verify corpus/tokenizer hashes, counts and reload parity.
  - Done when reproducible fitted artifacts and actual resource receipts are saved, or a terminal fitting failure is reported and subsequent phases remain unperformed.
  - _Boundary: Study fitting acceptance_
  - _Depends: 6.4_
  - _Requirements: 1, 2, 3, 4, 6_

- [ ] 6.6 Execute screening and development gates
  - Run the frozen screening and development protocol within its approved caps; record the baseline recall signal and qualifying alpha or an inconclusive result.
  - Done when the alpha decision and consumed budget are saved without held-out access; a failed gate explicitly blocks efficacy evaluation.
  - _Boundary: Study screening and development_
  - _Depends: 6.5_
  - _Requirements: 4, 5, 6_

- [ ] 6.7 Execute the gated held-out pilot and publish its outcome
  - Proceed to held-out generation only if screening/development gates pass and approved budget remains; otherwise publish the terminal inconclusive outcome with the held-out phase unperformed.
  - Analyze saved outcomes, preserve incomplete/unperformed phases, and publish dated evidence with README/MkDocs links and no inflated semantic-lookahead claim.
  - Done when the authorized study's actual terminal outcome is documented and replayable; an inconclusive result is valid, but omitted phases must remain visibly unperformed.
  - _Boundary: Explicit live study integration and reporting_
  - _Depends: 6.6_
  - _Requirements: 1, 2, 3, 4, 5, 6, 7_
