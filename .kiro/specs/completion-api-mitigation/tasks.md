# Implementation tasks: configurable completion mitigation

Status: requirements, design and tasks approved by the user on 2026-10-30. Implementation in progress; checkboxes require reviewed evidence.

Each executable task is scoped to roughly 1-3 hours of engineering effort; provider wait time and manual fidelity audits may add elapsed time. Execute sequentially to avoid overlapping adapter, scorer and ensemble edits. Live tasks may remain blocked by missing credentials or corpus coverage; never mark them done from offline tests alone.

## Quality contract for every task

Use idiomatic Python and names matching domain meaning and representation. Document array axes and DataFrame row grain, columns, units and alignment. Preserve lightweight imports and project layering. Use Ruff, tests and `scripts/check_architecture.py`; Sentrux is not a gate. Keep each code unit focused; split responsibilities instead of adding speculative abstractions. For data/report-only tasks, review schema and provenance; production-code size guidelines do not apply. Existing requirement-mandated documentation and live measurement are included despite the generic task template's code-only default.

- [ ] 1. Foundation and compatibility
- [ ] 1.1 Verify provider contracts and freeze compatibility fixtures
  - Check current official OpenAI, Anthropic, Mistral and NIM protocol/model documentation; record supported parameter mappings, capability evidence, unknowns and endpoint variants. Create sanitized success/error fixtures and pin any genuinely required dependencies.
  - Done when fixture tests distinguish absent logprobs, rejected parameters and authentication failures, and existing scorer/ensemble baselines are recorded.
  - Verify the quality contract for this task's code or artifacts, including semantic names and explicit evidence availability.
  - _Boundary: Provider adapters_
  - _Requirements: 1.4, 1.5, 7.1, 7.2, 7.4, 7.5_

- [ ] 1.2 Implement explicit model and response contracts
  - Add typed configuration, capability states, optional token evidence and legacy response adaptation. Validate feature combinations without silently changing them; expose a configuration inspection result.
  - Done when old response construction still works and incompatible requests fail before batch calls with specific reasons.
  - Verify the quality contract for this task's code or artifacts, including semantic names and explicit evidence availability.
  - _Boundary: Completion contracts_
  - _Requirements: 1.3, 3.5, 7.1, 7.2, 7.3, 7.5, 7.7_

- [ ] 1.3 Implement physical-request budgets and admission
  - Count probes, retries, verification, rewriting and target calls under a shared thread-safe limit. Preserve current start-spacing semantics, bound retries and redact credentials from errors.
  - Done when concurrent/retry tests never exceed the budget or compress request starts, and partial stage usage remains inspectable.
  - Verify the quality contract for this task's code or artifacts, including semantic names and explicit evidence availability.
  - _Boundary: Request control_
  - _Requirements: 3.1, 3.2, 3.3, 3.4, 8.6_

- [ ] 1.4 Define experiment protocol and persisted record schemas
  - Define versioned protocol, item/arm/draw identity, request events, rewrite records, completion markers and restart validation before orchestration consumes them.
  - Done when round-trip fixtures preserve partial failures and reject incompatible resume inputs without issuing calls.
  - Verify the quality contract for this task's code or artifacts, including semantic names and explicit evidence availability.
  - _Boundary: Experiment contracts_
  - _Requirements: 3.1, 3.3, 5.1, 5.5_

- [ ] 2. Provider integration
- [ ] 2.1 Implement OpenAI-compatible and NVIDIA completion support
  - Use audited mappings for OpenAI-compatible, Mistral and NIM configurations; permit absent logprobs only on explicitly selected text paths. Integrate shared request admission and actual access probes.
  - Done when fixtures exercise text and token-evidence modes, unsupported explicit parameters remain visible and legacy NVIDIA calls preserve behavior.
  - Verify the quality contract for this task's code or artifacts, including semantic names and explicit evidence availability.
  - _Boundary: Provider adapters_
  - _Depends: 1.1, 1.2, 1.3_
  - _Requirements: 1.4, 1.5, 3.1, 7.2, 7.4, 7.5_

- [ ] 2.2 Implement native Anthropic completion support
  - Normalize native Anthropic messages using the audited protocol contract. Preserve usage/finish metadata and distinguish runtime, transport and capability errors.
  - Done when the native Anthropic path satisfies the common contract using offline fixtures, including truncated output and no-logprobs responses.
  - Verify the quality contract for this task's code or artifacts, including semantic names and explicit evidence availability.
  - _Boundary: Provider adapters_
  - _Depends: 1.1, 1.2, 1.3_
  - _Requirements: 1.4, 1.5, 3.1, 3.4, 7.1, 7.4, 7.7, 7.9_

- [ ] 2.3 Adapt caller-supplied completion functions
  - Validate caller text results, optional usage and exception translation. Expose an explicit request-admission hook; report internal retry counts as unverified when the callback cannot disclose them and disallow strict-budget live runs through such callbacks.
  - Done when callback fixtures support text-only responses and distinguish callback runtime failures from model capability errors.
  - Verify the quality contract for this task's code or artifacts, including semantic names and explicit evidence availability.
  - _Boundary: Callback adapter_
  - _Depends: 1.2, 1.3_
  - _Requirements: 1.4, 3.1, 3.3, 7.7, 7.9_

- [ ] 3. Ensemble and scoring integration
- [ ] 3.1 Generalize existing answer ensembles to optional token evidence
  - Adapt existing generation/reduction contracts without a second reducer. Preserve tie-breaking, minimum draws, failure handling and replay; expose agreement independently from MCS.
  - Done when identical legacy draws replay identically and text-only draws produce valid answer statistics with missing MCS explicitly marked.
  - Verify the quality contract for this task's code or artifacts, including semantic names and explicit evidence availability.
  - _Boundary: Ensemble core_
  - _Depends: 1.2, 2.1, 2.2, 2.3_
  - _Requirements: 7.7, 8.1, 8.2, 8.3, 8.4_

- [ ] 3.2 Add configured clients and calibration identity to the scorer
  - Introduce the optional client path while preserving legacy construction. Require complete feature evidence; bind fitted calibration to endpoint/model/settings/intervention identity and reject mismatch.
  - Done when legacy parity passes and text-only or mismatched scoring is rejected without fabricated probabilities.
  - Verify the quality contract for this task's code or artifacts, including semantic names and explicit evidence availability.
  - _Boundary: Scorer integration_
  - _Depends: 1.2, 2.1_
  - _Requirements: 3.5, 7.6, 8.2, 9.3_

- [ ] 4. Rewriting and workflow
- [ ] 4.1 Implement versioned evidence rewriting
  - Separate instructions, evidence and as-of data; generate neutral conversational evidence with preservation constraints and no added external facts. Keep original/rewrite hashes and stage provenance.
  - Done when deterministic fixtures verify task instructions are unchanged and all failures remain explicit.
  - Verify the quality contract for this task's code or artifacts, including semantic names and explicit evidence availability.
  - _Boundary: Rewrite policy_
  - _Depends: 2.1, 2.2, 2.3_
  - _Requirements: 1.1, 1.3, 2.1, 2.2, 2.3, 3.1_

- [ ] 4.2 Implement fidelity verification and rejection
  - Combine deterministic factual-token checks with structured semantic verification. Treat malformed verification as failure, retain discrepancies and never present acceptance as proof.
  - Done when fixtures reject changed dates, numbers, entities, omitted qualifications and injected instructions before target inference.
  - Verify the quality contract for this task's code or artifacts, including semantic names and explicit evidence availability.
  - _Boundary: Rewrite policy_
  - _Requirements: 2.3, 2.4, 2.5, 2.6_

- [ ] 4.3 Integrate configured single and ensemble execution
  - Compose adapters, shared request budget, optional rewrite and existing ensembles in one workflow. Share one accepted rewrite per item; retain draw associations and explicit unavailable fields.
  - Done when original/rewritten single/ensemble fixtures complete through one configuration and disabled mitigation makes zero rewrite/verification calls.
  - Verify the quality contract for this task's code or artifacts, including semantic names and explicit evidence availability.
  - _Boundary: Workflow integration_
  - _Depends: 1.3, 2.1, 2.2, 2.3, 3.1, 4.2_
  - _Requirements: 1.1, 1.2, 7.3, 7.7, 8.1, 8.4, 8.5, 8.6_

- [ ] 4.4 Integrate intervention-matched scoring
  - Apply the same rewrite pipeline to calibration data when scored mitigation is selected, score genuine evidence per draw and reuse existing scored reduction. Preserve ordinary scorer entry points.
  - Done when mismatched unsteered calibration cannot score rewritten generation and valid scored draws retain existing aggregation semantics.
  - Verify the quality contract for this task's code or artifacts, including semantic names and explicit evidence availability.
  - _Boundary: Scorer and workflow integration_
  - _Depends: 3.2, 4.3_
  - _Requirements: 7.6, 8.2, 8.3, 9.3_

- [ ] 5. Experiment execution and statistics
- [ ] 5.1 Implement four-condition execution and record identity
  - Generate paired original/rewrite target draws with balanced arm order; designate draw zero before outcomes as single response. Retain item/arm/draw identity, failure records and rewrite associations; prevent accidental duplicate calls on resume.
  - Done when saved fixtures contain all planned conditions and missing draws remain visible with unchanged target/reduction settings.
  - Verify the quality contract for this task's code or artifacts, including semantic names and explicit evidence availability.
  - _Boundary: Experiment orchestration_
  - _Depends: 1.4, 4.3, 4.4_
  - _Requirements: 3.1, 3.3, 4.1, 4.2, 4.5, 8.5, 8.7, 9.1, 9.2_

- [ ] 5.2 Implement paired statistics with explicit denominators
  - Compute all-attempted and valid-pair outcomes, item-cluster bootstrap intervals, agreement, fidelity/parse rates and usage. Preserve dependencies when single responses reuse ensemble draws.
  - Done when analytic fixtures reproduce paired effects and failure-heavy runs cannot improve reported mitigation by dropping bad rows.
  - Verify the quality contract for this task's code or artifacts, including semantic names and explicit evidence availability.
  - _Boundary: Experiment statistics_
  - _Requirements: 3.2, 4.2, 4.3, 4.4, 4.5, 4.8, 9.2, 9.6_

- [ ] 5.3 Implement portable reports and offline metric replay
  - Persist redacted manifest, settings, stage events, original/rewrite text and response records. Record unknown effective settings/cost and distinguish each effect and provider-validation level.
  - Done when a fresh offline process reproduces reported metrics from exported artifacts without API access or absolute machine paths.
  - Verify the quality contract for this task's code or artifacts, including semantic names and explicit evidence availability.
  - _Boundary: Experiment reporting_
  - _Depends: 1.4, 5.1, 5.2_
  - _Requirements: 3.1, 3.2, 3.4, 5.3, 5.4, 5.5, 5.6, 6.2, 9.3, 9.6_

- [ ] 5.4 Add the explicit experiment command and protocol validation
  - Wire the established versioned experiment configuration to the command with model settings, conditions, data hashes, draw count, utility tolerance, audit selection and hard call budget. Keep historical cohort logic separate from existing post-cutoff guard.
  - Done when CLI dry-run validates configuration and displays planned calls/capabilities without starting a batch; invalid protocols fail clearly.
  - Verify the quality contract for this task's code or artifacts, including semantic names and explicit evidence availability.
  - _Boundary: CLI and experiment configuration_
  - _Depends: 1.4, 5.3_
  - _Requirements: 1.5, 4.6, 4.7, 5.1, 7.2, 8.7_

- [ ] 6. Measurement preparation and execution
- [ ] 6.1 Prepare sourced models and matched evaluation inputs
  - Verify official cutoffs for exact selected models including Anthropic when selected; record source URLs, retrieval dates and alias uncertainty. Build matched pre/post cohorts and hashes without weakening existing cutoff rules.
  - Done when cohort validation reports inclusion/exclusion counts, source/format differences and any unsupported historical inference.
  - Verify the quality contract for this task's code or artifacts, including semantic names and explicit evidence availability.
  - _Boundary: Experiment inputs_
  - _Depends: 5.4_
  - _Requirements: 4.6, 4.7, 5.1, 9.4, 9.5_

- [ ] 6.2 Preflight live access and freeze the pilot protocol
  - Select one logprob-capable model and one actual completion-only frontier configuration; confirm credentials without persisting them. Freeze sample size, n, fidelity audit and request budget before batch execution.
  - Done when actual inference probes pass and a concrete protocol identifies every planned condition and measurement gap; absent credentials leave this task incomplete.
  - Verify the quality contract for this task's code or artifacts, including semantic names and explicit evidence availability.
  - _Boundary: Live experiment setup_
  - _Depends: 6.1_
  - _Requirements: 1.5, 3.3, 5.1, 5.3, 5.6, 9.7_

- [ ] 6.3 Fit matching calibration for scored pilot conditions
  - Validate calibration/evaluation item separation and fit original and rewritten calibration on the logprob-capable model with matching settings and rewrite method. Include calibration and fidelity verification calls in the predeclared total budget; text-only conditions keep MCS unavailable.
  - Done when both calibration identities, valid counts and quality are recorded before scored evaluation; weak or missing calibration remains a visible limitation.
  - Verify the quality contract for this task's code or artifacts, including semantic names and explicit evidence availability.
  - _Boundary: Live calibration_
  - _Depends: 4.4, 6.2_
  - _Requirements: 3.3, 5.1, 7.6, 9.3_

- [ ] 6.4 Run the bounded paired pilot and audit rewrites
  - Execute four conditions on shared items and retain all failures. Review the preselected stratified fidelity sample, record judgments and disagreements, and reject efficacy claims if fidelity coverage is insufficient.
  - Done when artifacts contain both capability modes, all planned conditions, audit decisions and actual physical call/usage totals; incomplete runs remain labeled incomplete.
  - Verify the quality contract for this task's code or artifacts, including semantic names and explicit evidence availability.
  - _Boundary: Live measurement_
  - _Depends: 6.3_
  - _Requirements: 2.6, 4.1, 4.2, 4.4, 5.2, 5.6, 8.7, 9.1, 9.7_

- [ ] 6.5 Analyze measured effects and publish a dated local report
  - Recompute effects from saved data, separate rewriting/aggregation/scoring changes and report uncertainty, utility tolerance and fidelity limits. Link portable results from docs and README without presenting a pilot as proof of mitigation.
  - Done when report values reconcile with records and clearly state supported findings, inconclusive findings and missing provider coverage.
  - Verify the quality contract for this task's code or artifacts, including semantic names and explicit evidence availability.
  - _Boundary: Measurement report_
  - _Depends: 5.3, 6.4_
  - _Requirements: 4.3, 4.8, 5.3, 5.4, 5.5, 9.2, 9.3, 9.6, 9.7_

- [ ] 7. Adoption and final verification
- [ ] 7.1 Add notebook and application examples
  - Demonstrate explicit provider configuration, unsupported parameter errors, optional n draws, rewrite selection and missing MCS. Execute the notebook in a running kernel using deterministic completion fixtures; cite Jane Street in the engineering guide.
  - Done when examples run without shell-only setup or event-loop conflicts and accurately explain available capabilities and failure handling.
  - Verify the quality contract for this task's code or artifacts, including semantic names and explicit evidence availability.
  - _Boundary: Examples and documentation_
  - _Depends: 5.4_
  - _Requirements: 6.5, 7.8, 7.9, 8.2_

- [ ] 7.2 Verify legacy compatibility and readable public contracts
  - Run focused integration/regression tests followed by the full offline suite, Ruff and architecture checker. Review every new public contract and array/DataFrame alignment.
  - Done when required gates pass with fresh evidence, legacy calls retain behavior, no hidden network calls occur in tests and outstanding live gaps are listed separately.
  - Verify the quality contract for this task's code or artifacts, including semantic names and explicit evidence availability.
  - _Boundary: Final integration verification_
  - _Depends: 7.1_
  - _Requirements: 1.2, 3.4, 5.6, 6.1, 6.2, 6.3, 6.4, 6.5, 7.8, 7.9, 8.2_


- [ ] 7.3 Verify native platforms, notebook execution and docs build
  - Run the supported Windows/Linux Python matrix, notebook kernel smoke and strict documentation build after compatibility checks. Record commands and environments; keep live-provider coverage separate.
  - Done when platform and notebook checks pass with fresh evidence and all documented examples and links build successfully.
  - Verify the quality contract for this task's code or artifacts, including semantic names and explicit evidence availability.
  - _Boundary: Platform verification_
  - _Depends: 7.2_
  - _Requirements: 5.6, 6.1, 6.2, 6.5, 7.8, 7.9_

