# Requirements: endpoint-based divergence calibration and evaluation

> Scope split, 2026-11-05: local endpoint support with mandatory log probabilities
> is specified separately in [local-ai-provider](../local-ai-provider/requirements.md).
> Opus and the completion-only frontier alternative remain independent concerns.
> This study's broader requirements do not make that alternative a prerequisite
> for the local-provider extension. The November 5 NIM attempts are operationally
> classified as "NIM unavailable"; their diagnostic evidence remains unchanged.

Revision 2, 2026-11-01. Supersedes revision 1. User authorized the scope amendment;
this generated revision awaits review before revised design and implementation.
See [scope amendment](scope-amendment.md).

## Context and scope

Configure endpoints, source statistical evidence, tune a divergence correction,
freeze it, and measure held-out effects. Reuse the existing calibration workflow
where applicable. ROC-AUC measures discrimination, not calibration or mitigation.
The completed mathematical decoder remains unchanged.

Included: capability assessment, auxiliary provenance, correction tuning,
approximation disclosure, paired measurement, optional separately reported
ensembles, and request accounting. GPT-OSS-20B remains mandatory in study plans.
Anthropic is included as a completion-only target: target logprobs are not an
admission requirement for the sample-based correction route. Its statistical
estimator and correction mechanism must be specified and validated during design.
Excluded: mandatory self-hosting or GPU provisioning, automatic method/model
substitution, and treating rewriting as divergence decoding. No new calls are
authorized. Existing completion-only mitigation remains a separate option.

## Requirement 1: Endpoint evidence and capability identity

1. When preparing a study, the evaluator shall include openai/gpt-oss-20b and record each endpoint, model identity, revision when available, requested and observed settings, observation time and unknown fields.
2. When assessing a source, the evaluator shall distinguish full next-token scores, partial candidate scores, sampled text and unavailable evidence.
3. When assessing support, the evaluator shall report evidence availability separately from the ability to apply corrections before token selection and continue the selected prefix.
4. If a capability is unsupported or unverified, the evaluator shall report that state without silently substituting another model, mitigation or hosting arrangement.
5. The evaluator shall support planning without credentials, inference, downloads or a local accelerator.
6. When configuring Anthropic for completion-based statistical correction, the evaluator shall accept sampled completions without requiring target logits or logprobs, and record the explicitly selected model and sampling settings.

## Requirement 2: Auxiliary statistical sourcing

1. When selecting auxiliaries, the evaluator shall record older/newer roles, source identities, exposure windows, training or fitting provenance, tokenizer compatibility evidence and unknown provenance.
2. If advertised cutoff dates are the only temporal evidence, the evaluator shall mark temporal attribution unverified rather than infer a controlled knowledge contrast.
3. Where auxiliaries are fitted from corpora, the evaluator shall record rights, publication/revision evidence, content identities, counts by source and period, fitting policy and controls for size and source-composition effects.
4. When preparing splits, the evaluator shall exclude held-out prompt/answer documents from fitting and tuning and distinguish intentional factual exposure from accidental leakage.
5. When calculating token-level corrections, the evaluator shall require aligned token events and the same effective prefix across sources, or identify a separately validated approximation.

## Requirement 3: Correction and estimation semantics

1. When complete compatible log probabilities are available, the evaluator shall permit their use without requiring raw unnormalized logits.
2. Where partial candidate scores are used, the evaluator shall label the correction approximate and report candidate coverage, missing mass when known and treatment of omitted tokens.
3. Where probabilities are estimated from samples, the evaluator shall report sampling settings, counts, uncertainty, zero-count treatment and a predeclared precision or stopping criterion.
4. If evidence is insufficient, the evaluator shall report insufficient evidence rather than fabricate missing scores or equate ensemble agreement with next-token probabilities.
5. When reporting results, the evaluator shall distinguish exact execution of the declared mathematical adjustment from evidence of successful mitigation.
6. If the endpoint cannot apply the correction during generation, the evaluator shall distinguish offline score analysis and completion-sample correction from an executed token-level divergence-decoding intervention.
7. Where completion-sample correction is evaluated, the evaluator shall identify the auxiliary evidence, correction rule, sampling uncertainty and effective sample size, without claiming that target next-token probabilities were recovered or that sequence reweighting reproduces token-level divergence decoding.

## Requirement 4: Development calibration and held-out evaluation

1. Before tuning, the evaluator shall freeze development and held-out splits, candidate correction strengths including zero, selection rules, utility tolerance and memorization-related outcomes.
2. When tuning correction strength, the evaluator shall use development data only and select against both memorization-related outcomes and retained utility.
3. If no setting meets development criteria, the evaluator shall report no qualifying correction without using held-out outcomes to select one.
4. Before held-out execution, the evaluator shall freeze the correction, sources, evidence mode, prompts, parser and generation settings.
5. When reporting calibration, the evaluator shall distinguish correction-strength tuning, probability-estimation uncertainty and the existing contamination-score calibrator; ROC-AUC shall be labeled discrimination.
6. If source identity or relevant settings change, the evaluator shall flag calibration as condition-mismatched until revalidated.

## Requirement 5: Paired effects and uncertainty

1. When evaluating mitigation, the evaluator shall compare unchanged and corrected conditions on the same held-out items under matching recorded settings and a predeclared arm schedule.
2. When interpreting temporal mitigation, the evaluator shall require baseline recall evidence and report temporal and cohort confounds; lower historical accuracy alone shall not establish success.
3. When reporting effects, the evaluator shall retain planned, completed, parsed and common-usable denominators, failures, cohort-specific metrics and paired uncertainty.
4. Where ensembles are evaluated, the evaluator shall report their draw budgets and results separately from single answers and the divergence correction effect.
5. If repeats are deterministic or samples inadequate for the claim, the evaluator shall report that limitation without asserting independent evidence or proven efficacy.
6. If condition-matched calibration and valid token evidence are absent, the evaluator shall report MCS unavailable rather than substitute confidence or agreement.

## Requirement 6: Bounded, understandable execution

1. Before live execution, the evaluator shall require a concrete approved budget covering eligibility, source and auxiliary calls, sampling, tuning, retries and held-out evaluation.
2. When executing, the evaluator shall respect provider quotas, bounded waiting and cancellation, retaining completed, failed, ambiguous and unperformed work without silent replay.
3. When reporting usage, the evaluator shall distinguish physical requests from logical items and preserve unknown costs as unknown.
4. When publishing results, the evaluator shall preserve plans, source provenance, calibration decisions and per-item evidence without credentials, reporting what ran, where and with which limitations.
5. The implementation shall retain the lean portable library and use idiomatic names matching roles, representations and units, with documented array axes and DataFrame row grain and alignment.
6. The documentation shall distinguish mathematical implementation, endpoint feasibility and measured efficacy, citing Jane Street and the underlying research.
