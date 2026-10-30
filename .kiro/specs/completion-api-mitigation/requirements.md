# Requirements Document

## Introduction

Recall Guard adopters need an optional mitigation step for frontier models
accessed through ordinary completion APIs, including OpenAI and Anthropic,
without token logprobs or control over generation. The existing scorer measures
contamination-related signals after generation; the divergence decoder requires
next-token logits. This feature will offer one configurable model workflow for
generation, optional ensembles, available scoring evidence and optional
rewriting. Users explicitly choose capabilities and interventions; the system
validates those choices against the endpoint.

The user explicitly requested implementation and measurement on 2026-10-30.
The proposed first method is a fact-preserving narrative/podcast-style rewrite,
inspired by the separate rewriting experiment in
[Jane Street's blog](https://blog.janestreet.com/mitigating-memorization-in-llms/).
This is an experimental intervention, not an equivalent implementation of the
logit equation or a promise that memorization will decrease.

## Boundary context

- In scope: opt-in rewriting, frontier completion integration, inspection of
  original and rewritten evidence, failure reporting and paired measurement.
- In scope: support OpenAI, Mistral and Anthropic completion services and a caller-supplied
  completion function for additional providers. Live validation requires access
  to the particular provider and model; compatibility must not imply live validation.
- Out of scope: weight updates, model hosting, logits reconstruction, automatic
  selection of investments, and a new probability-of-membership estimator.
- Adjacent expectations: callers supply task instructions, source evidence,
  as-of date and evaluation labels. Rewriting cannot undo leakage already in
  the input. Existing logprob-based MCS scores remain unavailable where their
  required inputs are missing.

## Requirements

### Requirement 1: Optional completion-only intervention

**Objective:** As an application developer, I want to use mitigation with my
chosen completion provider without changing existing scoring behavior.

#### Acceptance criteria
1. When mitigation is enabled, Recall Guard shall obtain a rewritten evidence
   text and submit accepted evidence with the original task instructions to
   the selected target model without requiring token probabilities.
2. When mitigation is disabled, Recall Guard shall preserve existing call
   behavior and make no rewriting or verification calls.
3. The feature shall allow independently selected rewriting and target models.
4. Where a caller supplies a completion function, the feature shall accept
   its text results without requiring a particular provider SDK.
5. If a required capability or credential is unavailable, the feature shall
   report the affected stage before scheduling the evaluation batch.

### Requirement 2: Preserve evidence and temporal boundaries

**Objective:** As an evaluator, I want a change in presentation to preserve
the information the model was allowed to use.

#### Acceptance criteria
1. When rewriting evidence, the feature shall instruct the rewriting model
   to preserve factual claims, entities, dates, quantities, qualifications and
   uncertainty, and to add no external facts or predicted outcomes.
2. The feature shall keep task instructions and output-format requirements
   outside the rewriteable evidence.
3. When a rewrite is returned, the feature shall retain both texts and the
   outcome of fidelity checks, including checks that could not be performed.
4. If checks detect added, removed or changed facts, the feature shall reject
   the rewrite and report the reason before target inference.
5. If rewrite generation or verification fails, the feature shall report a
   failed intervention rather than silently label an original-prompt answer
   as mitigated.
6. The feature shall label automated fidelity acceptance as a check result,
   not a guarantee of semantic equivalence or absence of future knowledge.

### Requirement 3: Inspectable results and bounded calls

**Objective:** As an operator, I want to understand what ran and control its cost.

#### Acceptance criteria
1. When a run completes or fails, the feature shall record model identifiers,
   generation settings, rewrite-method version, timestamps, prompt hashes,
   stage outcomes, latency and provider-reported usage when available.
2. If usage or price information is unavailable, the feature shall mark cost
   as unknown rather than zero or invent a monetary estimate.
3. When a configured call or retry limit is reached, the feature shall stop
   issuing further calls and preserve partial results with an explicit status.
4. The feature shall exclude credentials and authorization headers from
   persisted results and errors.
5. Where logprobs are unavailable, the feature shall report the existing MCS
   score as unavailable rather than derive it from self-reported confidence.

### Requirement 4: Paired measurement and honest denominators

**Objective:** As a researcher, I want to measure whether the intervention
changes recall-related behavior while preserving useful performance.

#### Acceptance criteria
1. When an experiment runs, the evaluator shall compare original and rewritten
   evidence for the same item IDs using the same target model and recorded
   generation settings, with balanced arm ordering.
2. The evaluator shall report task performance, parsing success, fidelity
   acceptance, failures, latency and usage separately for each arm.
3. The evaluator shall report paired performance differences with uncertainty
   and disclose the resampling unit and eligible-pair count.
4. The evaluator shall report all-attempted outcomes as well as valid-pair
   outcomes so failed rewrites or responses cannot appear as mitigation gains.
5. Where repeated completions are configured, the evaluator shall retain draw
   identities and account for shared item identity in uncertainty estimates.
6. When historical evaluation claims are made, the evaluator shall report
   matched pre-cutoff and post-cutoff cohorts and the documented cutoff basis.
7. If a model's cutoff cannot be established, the evaluator shall allow a
   general intervention comparison but withhold cutoff-based recall conclusions.
8. The evaluator shall withhold a mitigation-success conclusion when the
   observed change is explained by failed responses, rejected rewrites or
   degradation beyond a predeclared useful-performance tolerance.

### Requirement 5: Evidence required before claiming effectiveness

**Objective:** As an adopter, I want reproducible evidence that distinguishes
implementation correctness from a research result.

#### Acceptance criteria
1. Before a live evaluation, the evaluation protocol shall record dataset
   hashes, cohort definitions, task metric, fidelity-audit method, utility
   tolerance, repetition count and call budget.
2. The measurement shall include a reviewable fidelity audit of rewritten
   evidence and disclose its sample selection and disagreements with automated checks.
3. When publishing results, the report shall distinguish implemented provider
   support, offline-tested support and providers actually measured live.
4. The report shall label historical performance gaps and confidence changes
   as proxies, and shall not treat lower accuracy alone as proof of reduced recall.
5. When saving results, the feature shall preserve sufficient inputs and
   outputs to recompute reported metrics without fresh provider calls.
6. The feature shall keep live measurements separate from offline tests and
   report a missing live credential as an unmeasured provider, not a passed test.

### Requirement 6: Understandable, portable integration

**Objective:** As a maintainer, I want an idiomatic implementation that remains
usable on native Windows and understandable to engineering adopters.

#### Acceptance criteria
1. The implementation shall support the project's Python versions on Windows
   and Linux without requiring WSL or a local GPU.
2. The implementation shall use explicit UTF-8 for persisted text artifacts.
3. The implementation shall use names that distinguish original evidence,
   rewritten evidence, model responses and evaluation records.
4. Where arrays or DataFrames are used, the implementation shall document
   axes, row grain, column meanings, units and pairing/alignment rules.
5. The documentation shall provide an opt-in example, failure handling and
   a reproducible paired-evaluation command, with a citation to Jane Street
   and an explicit distinction from divergence decoding.

### Requirement 7: Explicit configuration and capabilities

**Objective:** As a user, I want to configure a model once and choose its
supported parameters and features in a notebook or application.

#### Acceptance criteria
1. When configuring a model, the feature shall accept provider/protocol,
   endpoint URL, model identifier, credential reference, generation parameters
   and declared capabilities without inferring protocol solely from a URL.
2. Before a batch runs, the feature shall expose requested capabilities as
   supported, unsupported or unverified, including logprobs and generation controls.
3. If a selected feature requires unsupported inputs, the feature shall report
   the incompatible choice before batch execution and require an explicit
   configuration change rather than silently switch methods.
4. If the provider rejects an explicit parameter, the feature shall identify
   the parameter and error rather than silently drop it.
5. Where defaults are used, the feature shall record omitted parameters and
   distinguish requested values from confirmed effective values.
6. Where valid logprobs and matching calibration are available, the workflow
   shall allow existing MIA/MCS scoring alongside the chosen intervention.
7. Where logprobs are unavailable, the workflow shall allow generation,
   parsing, ensemble answer statistics and optional rewriting, with missing
   MIA/MCS fields explicitly marked unavailable.
8. When used in Jupyter, the workflow shall preserve the same model configuration
   and feature choices without requiring shell-only setup.
9. If a notebook execution constraint prevents a call, the feature shall
   distinguish that runtime error from an unsupported model parameter.

### Requirement 8: Existing ensemble integration

**Objective:** As a user, I want an ensemble of n completions with or without
rewriting or logprobs through the existing workflow.

#### Acceptance criteria
1. When an ensemble is requested, the feature shall accept an explicit draw
   count and reuse existing ensemble configuration and reduction semantics
   wherever the corresponding inputs are available.
2. The feature shall preserve existing scorer and ensemble entry points for
   current integrations and document added optional configuration.
3. When aggregating completion-only responses, the feature shall distinguish
   answer agreement and dispersion from memorization scores and shall not
   require fabricated logprob fields for supported statistics.
4. When draws fail, the feature shall report requested, attempted and usable
   draw counts and enforce the configured minimum usable count.
5. When rewriting and ensembles are combined, the feature shall record whether
   draws share a rewrite or use multiple rewrites and retain each draw's association.
6. The feature shall count rewrite, verification, target and retry calls against
   request budgets rather than treat n target draws as total cost.
7. When original and rewritten ensembles are compared, the evaluator shall use
   equal draw counts, reduction settings and target generation settings and
   identify unpaired or missing draws.

### Requirement 9: Measure the contribution of each option

**Objective:** As an adopter, I want separate evidence for rewriting,
ensembling and available scoring.

#### Acceptance criteria
1. When measuring the combined workflow, the evaluator shall support original
   single response, original ensemble, rewritten single response and rewritten
   ensemble as four explicit conditions.
2. The evaluator shall distinguish rewriting effects from aggregation effects
   using shared evaluation items for each comparison.
3. Where logprob scoring is available, the evaluator shall report confidence
   discounts and calibration quality without attributing them to rewriting.
4. When a documented cutoff exists for the exact model, including an Anthropic
   model, the evaluator shall use its sourced registry entry for cohort selection
   and preserve the source URL and cutoff date in provenance.
5. If a model alias has ambiguous training history, the evaluator shall record
   uncertainty rather than inherit another model family's cutoff.
6. When presenting measured effects, the report shall include resource usage
   alongside task outcomes so the cost of additional requests is visible.
7. The live evaluation shall cover at least one logprob-capable configuration
   and one completion-only configuration; if access is missing, the report
   shall identify incomplete coverage rather than mark measurement complete.

## Requirements review

Reviewed for coverage, numeric requirement IDs, EARS structure, failure paths,
provider scope and measurement boundaries. Nine areas cover invocation,
fidelity, provenance/cost, paired statistics, evidence, portability, explicit
capabilities, existing ensemble integration and comparisons of feature effects.
Implementation mechanisms and fidelity-check algorithms belong in design.
The user approved requirements with the capability-driven configuration and
ensemble clarification on 2026-10-30. These revisions incorporate that direction.
Design and tasks still require phase review. No mitigation measurement has
yet been performed.
