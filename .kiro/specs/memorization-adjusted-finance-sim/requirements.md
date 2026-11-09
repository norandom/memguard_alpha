# Requirements Document

## Project Description (Input)
**Who has the problem.** Quant researchers and Recall Guard adopters who run finance
simulations (backtests, event studies, forecasting) with LLMs. Historical accuracy
can come from recall of training data rather than reasoning, which inflates
pre-cutoff results (look-ahead bias).

**Current situation.** Recall Guard estimates and calibrates `p_memorized`, and
the release-1.0 research established the expected null: on a clean SEC EDGAR
corpus, five models across four runtimes show no predictive capacity, with
single-draw scoring and with full-logit document likelihood (register R26, R30,
R38, R40, R41). Exact full logits and logprobs are available for Gemma on local
hardware: llama.cpp on the iGPU and Ryzen AI OGA on the NPU (R35, R36, R39).
There is no importable API that runs a finance simulation and adjusts each
answer for memorization in the background, no positive control where
memorization is known to exist, and no test of whether the adjustment or the
ensemble matters.

**What should change.** An importable API that calls a model, returns its answer
and, in the background, adjusts the answer's confidence for memorization using
the calibrated `p_memorized` computed from the response's own logprobs (top-k) or
full logits, with and without the ensemble strategy (which reduces serving
nondeterminism). A study must show statistically whether the adjustment and the
ensemble each matter, with a positive control, a negative control, paired
pre-registered tests, a convergence check of top-k against full logits, and data
visualization with one clear worked example published in the docs.

## Introduction

This feature turns Recall Guard's calibrated memorization estimate into an
importable tool for finance simulations, and tests whether it earns its place.

An adopter calls one function with a prompt. Recall Guard returns the model's
answer as a structured decision, together with a memorization estimate for that
answer and a memorization-adjusted version of the decision. The adjustment
scales the decision's magnitude by one minus the estimate, the rule already used
by the adopter project Global_Macro_AI_Factors (`guarded_tilt = raw_tilt ·
(1 − p_memorized)`). The ensemble mode samples the same prompt many times on the
same model and reduces the draws to one consensus decision and one point
estimate, because both the answer and the estimate vary between calls even at
temperature 0: in that project, 100 scorings of one prompt gave a mean
`p_memorized` of 0.211 with a standard deviation of 0.218 and a range of 0.00 to
0.76.

The study needs a setting where memorization is known to exist, otherwise no
adjustment can show an effect. The positive control renders the same FRED macro
row twice: an identifying prompt (real date, asset names, raw levels) that
invites recall, and an anonymised point-in-time prompt that does not. Before a
model's training cutoff the identifying prompt can draw on memory of what
happened next; after the cutoff it cannot. The adjustment should remove the extra
pre-cutoff advantage of the identifying prompt. The negative control is the
SEC EDGAR setting from release 1.0, where nothing should change.

The worked example is a monthly macro rebalance in the style of
Global_Macro_AI_Factors: five regime loadings from a FRED macro row, mapped to
tilts on four anonymised assets.

## Boundary Context

- **In scope**: an importable single-call and ensemble API for decisions with
  background memorization adjustment; memorization estimates from top-k
  logprobs and from full logits; calibration per model, estimate source and
  settings; the positive-control (identifying versus anonymised) and
  negative-control (SEC EDGAR) studies; the statistical tests and their reports;
  the top-k versus full-logit convergence analysis; one worked macro example with
  figures published in the docs; local Gemma hosts and optional Modal.
- **Out of scope**: changes to the Global_Macro_AI_Factors repository (it adopts
  the API later in its own change); removing memorization from a model
  (rewriting, divergence decoding, fine-tuning); trading advice or live trading;
  new model hosting runtimes beyond those verified in release 1.0; hard gating of
  decisions on a single-draw estimate.
- **Adjacent expectations**: model hosts (Ollama, llama.cpp, FastFlowLM, Ryzen AI
  OGA, Modal) are run and stopped as in release 1.0, natively on Windows behind
  authenticated, auto-stopping gateways, while Recall Guard runs in Linux
  containers. FRED supplies macro series and a documented price source supplies
  realized asset returns. Reports follow `docs/results/template.md` and
  `.kiro/steering/research-reports.md`.

## Requirements

### Requirement 1: Guarded decision for a single call
**Objective:** As a quant researcher, I want one call that returns the model's decision together with a memorization estimate and an adjusted decision, so that I can use LLM outputs in a simulation without hand-wiring the guard.

#### Acceptance Criteria
1. When an adopter submits a prompt with a configured model and decision definition, Recall Guard shall return the raw decision, the memorization estimate, the adjusted decision and a status in one result.
2. The Recall Guard adjustment shall scale each component of the decision's magnitude by one minus the memorization estimate, clipped to the range 0 to 1, and shall leave the decision's sign unchanged.
3. If the response carries no usable token evidence, Recall Guard shall return the raw decision unchanged, mark the estimate as unavailable and state the reason.
4. If the calibrator for the configured model and estimate source is marked weak, Recall Guard shall return the raw decision unchanged and state that the adjustment was skipped because of calibrator quality.
5. If the response cannot be parsed into the decision definition, Recall Guard shall return a failure status with the reason and no decision.
6. The Recall Guard result shall record the model identity, estimate source, generation settings, calibrator identity and a hash of the prompt.
7. The Recall Guard shall not withhold or zero a decision on the basis of a single-draw memorization estimate.

### Requirement 2: Decision definitions
**Objective:** As an adopter, I want to describe what my simulation asks the model for, so that the same API serves different finance tasks.

#### Acceptance Criteria
1. The Recall Guard shall accept a decision definition that states how a response is parsed into named numeric components and, optionally, how components map to one summary decision value.
2. Where a decision definition includes a summary mapping, Recall Guard shall report the summary value for the raw and the adjusted decision.
3. When an adopter supplies an invalid decision definition, Recall Guard shall reject it before any model request and name the problem.
4. The Recall Guard shall ship one decision definition for the worked example: five regime loadings in the range -1 to 1, mapped to tilts on four anonymised assets and to one defensive-spread value.

### Requirement 3: Memorization estimate sources
**Objective:** As a researcher, I want the estimate to come from the response's own token evidence, either top-k logprobs or full logits, so that I can choose the source my host supports and know which one was used.

#### Acceptance Criteria
1. Where the host returns top-k logprobs, Recall Guard shall compute the memorization estimate from the response's own top-k logprobs.
2. Where the host returns full next-token logits or full-vocabulary logprobs, Recall Guard shall compute the memorization estimate from the full distribution.
3. The Recall Guard shall label every estimate with its source (top-k with k, or full distribution) and shall not mix sources within one calibrator.
4. If a configured source is not available from the selected host, Recall Guard shall refuse the configuration before any model request and name the available sources.
5. When full-distribution values are given as logits, Recall Guard shall convert them to log-probabilities per position before computing features.

### Requirement 4: Calibration identity
**Objective:** As a researcher, I want calibrators bound to the model, source and settings they were fitted on, so that an estimate is never computed with the wrong calibrator.

#### Acceptance Criteria
1. The Recall Guard shall bind each calibrator to the model identity, estimate source, generation settings and calibration corpus it was fitted on.
2. If an adopter applies a calibrator to a response with a different model identity, estimate source or generation settings, Recall Guard shall refuse to compute the estimate and state the mismatch.
3. When a calibrator is fitted, Recall Guard shall report its held-out discrimination with an interval and mark it weak when the discrimination is below the configured threshold.

### Requirement 5: Ensemble mode
**Objective:** As a quant researcher, I want the ensemble to give me the calibrated model's decision and estimate with less serving nondeterminism, so that a simulation does not depend on one lucky or unlucky draw.

#### Acceptance Criteria
1. Where ensemble mode is selected, Recall Guard shall draw the same prompt the configured number of times with the same model and settings.
2. Where ensemble mode is selected, Recall Guard shall return one consensus decision, one point memorization estimate as the mean of the per-draw estimates, the per-component spread, the decision agreement and the counts of requested, parsed and failed draws.
3. Where ensemble mode is selected, Recall Guard shall compute the adjusted decision from the consensus decision and the point estimate.
4. If fewer draws parse than the configured minimum, Recall Guard shall return a failure status with the counts and no decision.
5. If a decision component is split into separated clusters across draws, Recall Guard shall report that component as unresolved and shall not report a single value for it.
6. The Recall Guard shall record the draws so that the ensemble result can be recomputed without new model requests.

### Requirement 6: Positive control (identifying versus anonymised prompts)
**Objective:** As a researcher, I want a setting where memorization is known to exist, so that the study can show whether the adjustment removes it.

#### Acceptance Criteria
1. The Study shall render each FRED macro row as an identifying prompt (real decision date, asset names, raw levels) and as an anonymised point-in-time prompt that differ only in those identifying elements.
2. The Study shall classify each decision date as before or after the evaluated model's training cutoff using the project's sourced cutoff registry.
3. When the Study evaluates a model, it shall measure the identifying-minus-anonymised difference in the memorization estimate and in decision accuracy separately before and after the cutoff.
4. If the evaluated model has no sourced cutoff in the registry, the Study shall exclude that model and state why.
5. The Study shall use, for each decision date, only macro values that were available on that date where FRED provides vintages, and shall label any series used without vintages as revised data.
6. The Study shall define, in the frozen plan, decision accuracy against realized next-period returns of the four assets from a documented price source, and shall use the same definition for every variant.

### Requirement 7: Negative control
**Objective:** As a researcher, I want evidence that the adjustment does not discount honest answers, so that adopters can trust it where there is nothing to detect.

#### Acceptance Criteria
1. The Study shall run every variant on the release-1.0 SEC EDGAR setting, where no memorization is detectable.
2. When the negative-control results are reported, the Study shall state the size of the change each variant makes against no adjustment, with an interval.
3. If a variant changes negative-control decisions by more than the pre-registered tolerance, the Study shall report that variant as discounting honest answers.

### Requirement 8: Variants and statistical tests
**Objective:** As a principal investigator, I want pre-registered paired tests of each variant, so that a significant result is not a product of choosing the analysis after seeing the data.

#### Acceptance Criteria
1. The Study shall compare four variants on the same items: no adjustment, adjustment from top-k logprobs, adjustment from full logits, and ensemble with adjustment.
2. The Study shall freeze the items, variants, metrics, tolerance, seeds, resample counts, significance level and decision rule in a dated plan with content hashes before it sends any study request.
3. The Study shall test, per variant, the change in the pre-cutoff minus post-cutoff accuracy gap of identifying prompts, with a paired bootstrap interval.
4. The Study shall test, per variant, the difference in Brier score or log loss against no adjustment on realized outcomes, with a paired bootstrap interval.
5. The Study shall test whether ensemble mode reduces the run-to-run variance of the decision and of the memorization estimate, by comparing two independent half-ensembles of each prompt with single draws.
6. The Study shall test each model's decisions against a random-answerer null over the same decision space and report the test statistic and p-value.
7. The Study shall adjust p-values across variants with the Holm method and report raw and adjusted values.
8. The Study shall report effect sizes with intervals and the achieved power for the pre-registered effect size.
9. If a pre-registered item or metric cannot be computed, the Study shall report it as missing with the reason and shall not substitute another.

### Requirement 9: Top-k versus full-logit convergence
**Objective:** As a researcher, I want to know how far top-k logprob estimates are from full-logit estimates within one model, so that I can judge when top-k is good enough.

#### Acceptance Criteria
1. The Study shall compute, for the same responses and the same model, the memorization features and estimates from top-k logprobs for several values of k and from the full distribution.
2. The Study shall report, per feature and per k, the agreement with the full-distribution value and the top-k probability mass covered.
3. The Study shall report how the downstream adjusted decision changes between top-k and full-distribution estimates.
4. The Study shall compare values only within one model and one set of weights, and shall state that values from different builds or families are not comparable.

### Requirement 10: Worked example and visualization
**Objective:** As an adopter, I want one clear worked example with figures, so that I can see the effect, or its absence, at a glance.

#### Acceptance Criteria
1. The Study shall publish one worked example in the docs: a monthly macro rebalance with FRED rows, run through all four variants for the positive and the negative control.
2. The Study shall include in the worked example a figure of the pre-cutoff versus post-cutoff accuracy gap of identifying and anonymised prompts before and after adjustment, with intervals.
3. The Study shall include in the worked example reliability diagrams of the memorization estimate per variant.
4. The Study shall include in the worked example a figure of the per-draw spread of decisions and estimates against the ensemble result.
5. The Study shall include in the worked example a figure of top-k versus full-distribution agreement as k grows.
6. The Study shall state each figure's data, sample size and interval method in its caption, and shall make each figure readable without color alone.
7. The Study shall make the worked example reproducible offline from recorded draws and scores, without new model requests.

### Requirement 11: Execution, cost and reporting
**Objective:** As a principal investigator, I want every run bounded, approved and reported the same way, so that the study is auditable and costs are controlled.

#### Acceptance Criteria
1. The Study shall run local models through the verified local hosts and shall stop every host and gateway it starts, verifying that no process or listener remains.
2. When a paid run is requested, the Study shall present the request count, token limits and maximum cost, and shall not start it without an approval record that matches the frozen plan.
3. If a run exceeds its frozen request, token, time or cost limit, the Study shall stop admission, keep the partial results and report the run as incomplete.
4. The Study shall report every result in a dated report that follows the project report template, with a numbered research-register entry and a link from the docs navigation.
5. The Study shall report failed, stopped and excluded runs with their causes.
6. The Recall Guard API shall work from an installed package without the research data, hosts or study code.
