# Requirements: real-model divergence decoding evaluation

## Context and scope

Recall Guard adopters need measured evidence of real-model steering and its
recall/utility tradeoff. Existing tests establish mathematical correctness;
existing NIM measurements establish unadjusted scoring behavior. Neither
demonstrates divergence mitigation efficacy.

The user approved the [model selection](../divergence-decoding/model-selection.md):
`openai/gpt-oss-20b` is mandatory, with two proposed trigram auxiliary models
using its exact tokenizer. This specification covers a single-answer feasibility
experiment. Docker is the execution environment. The existing mathematical
decoder is a dependency; its completed specification is not reopened.

Included: optional experimental execution, corpus provenance, auxiliary fitting,
paired generation, recall/utility evaluation and reproducible reporting.
Excluded: main-model training, neural auxiliary training, cross-tokenizer
bridges, synthetic rewriting, stochastic ensembles, production serving and
automatic provider substitution. Neither lower confidence nor lexical recall
suppression establishes broad financial lookahead mitigation.

The proposed auxiliary windows end on 2019-12-31 and 2024-06-30. Actual source
access, licensing, timestamp quality and compute capacity are unresolved inputs
to design; they must be resolved before an executable run is approved.

## Requirement 1: Explicit experimental identity

**Objective:** Operators can identify exactly which model and intervention ran.

1. The evaluation runner shall include `openai/gpt-oss-20b` as the main target and unadjusted baseline in every study.
2. When preparing a plan, the runner shall record immutable model revisions, tokenizer identity, auxiliary artifact hashes, corpus hashes, generation settings, precision, software revision and runtime identity.
3. When creating auxiliary artifacts, the runner shall identify them as locally fitted trigram models rather than published neural checkpoints.
4. If a required model or artifact is unavailable, the runner shall report the missing prerequisite and shall not substitute another model or method.
5. When comparing with saved hosted NIM results, the report shall identify hosted and self-hosted conditions separately and shall not assert numerical equivalence from a shared model name.

## Requirement 2: Controlled auxiliary exposure

**Objective:** Researchers can audit the older/newer contrast and its confounds.

1. When admitting a corpus, the runner shall require source identity, usage rights, publication and revision evidence, exposure window and content hashes.
2. If a source lacks adequate temporal provenance or violates the declared exposure window, the runner shall reject it with a recorded reason.
3. When fitting auxiliaries, the runner shall use the same tokenizer, estimator policy and text processing, with the newer corpus containing the declared older corpus plus the declared later material.
4. The runner shall record document and token counts for each period and source, and a same-period matched-size control for corpus-size effects.
5. When validating data splits, the runner shall reject duplicated held-out prompt/answer documents across fitting, development and evaluation partitions and shall record deliberately shared factual exposure separately from text duplication.
6. The runner shall exclude post-June-2024 utility events from both auxiliary fits and exclude held-out evaluation outcomes from alpha selection.
7. When constructing prompts, the runner shall preserve the declared as-of information boundary; later information intentionally supplied to the newer auxiliary corpus shall remain identified as experimental exposure.

## Requirement 3: Compatible and verifiable token steering

**Objective:** Operators can establish that the intended equation controls real generation.

1. Before generation, the runner shall verify matching vocabulary indices, token meanings, special-token handling and compatible prefix encoding across all three roles.
2. When advancing generation, the runner shall obtain finite full-vocabulary scores for the same selected-token prefix and apply the existing older-minus-newer logit combination.
3. When an auxiliary encounters an unseen context or token, the runner shall apply its frozen finite-score policy and record that policy in the plan.
4. When alpha is zero, the validation shall require token-for-token parity with the unadjusted main-model loop under identical settings.
5. When identical auxiliary scores are supplied, the validation shall require cancellation parity with the unadjusted loop.
6. If compatibility, finite-score, formatting or parity validation fails, the runner shall stop before efficacy evaluation and preserve the failure evidence.
7. When nonzero-alpha generation completes, the runner shall report changed-token and changed-answer counts without requiring every answer to change.

## Requirement 4: Bounded and reproducible execution

**Objective:** Operators can review resource use before running the experiment.

1. When preparing a plan, the runner shall work without credentials, model downloads, auxiliary fitting or inference and shall enumerate missing prerequisites.
2. Before execution, the runner shall require an approved plan containing dataset sizes, maximum fitting volume, generation counts, context and output limits, wall-time and compute-cost limits, and the selected hardware/runtime.
3. When validating runtime feasibility, the runner shall measure peak memory and elapsed time for the declared settings rather than treat checkpoint size as runtime capacity.
4. If a resource limit is reached or the caller cancels, the runner shall stop admitting new work and retain completed, failed and unperformed work separately.
5. When a run is interrupted, the runner shall preserve its operation ledger and shall not silently replay ambiguous work or overwrite an existing result directory.
6. The experimental runner shall execute in Docker without requiring native Windows-specific business logic or adding inference dependencies to the library's default installation.

## Requirement 5: Paired recall and utility evaluation

**Objective:** Researchers can distinguish steering from evidence of a useful mitigation.

1. Before held-out evaluation, the runner shall freeze the nonzero alpha selected on separate development data, the selection rule, prompts, parser, cohort assignment and paired arm order.
2. When evaluating an item, the runner shall compare unadjusted and adjusted generation using the same main-model revision, prompt and generation limits.
3. The feasibility study shall include eight declared historical items and eight declared post-cutoff utility items, and shall label its results exploratory.
4. Before interpreting historical mitigation, the report shall require a demonstrated baseline recall signal on a separately declared screening task; absent that signal, it shall mark mitigation efficacy inconclusive.
5. When reporting results, the runner shall provide correct/planned, correct/usable, parse coverage, retained failures, paired common-usable results and cohort-specific metrics.
6. The runner shall report the change in historical-versus-post-cutoff accuracy gap and post-cutoff utility change with item-paired, cohort-stratified uncertainty intervals.
7. When testing the proposed efficacy criterion, the runner shall require the upper 95% bound for gap change to be below zero and the lower 95% bound for post-cutoff utility change to exceed minus five percentage points; it shall not describe a small feasibility sample as adequate proof solely because point estimates satisfy the criterion.
8. If apparent improvement is explained by rejected or unusable answers, the report shall expose that loss and shall not classify it as preserved useful performance.
9. When measuring lexical recall suppression, the report shall distinguish that result from semantic recall and financial lookahead efficacy.

## Requirement 6: Honest scores and result artifacts

**Objective:** Adopters can inspect results without treating unavailable evidence as zero.

1. When publishing a report, the runner shall preserve the frozen plan, provenance, per-item outcomes, per-operation resource ledger and analysis settings in portable artifacts without credentials.
2. When reporting confidence, the runner shall distinguish stated confidence, generated-token probabilities and calibrated contamination scores.
3. If condition-matched independent calibration and valid token evidence are unavailable, the runner shall report MCS unavailable rather than reuse an unadjusted calibrator or substitute agreement.
4. Where condition-matched MCS is reported, the runner shall identify calibration data, held-out quality and whether token probabilities describe the adjusted or original distribution.
5. When repeated greedy runs produce identical answers, the report shall identify them as deterministic repeats rather than independent ensemble evidence.
6. The report shall identify unknown monetary cost or usage as unknown and shall retain requested, completed and unperformed counts separately.

## Requirement 7: Maintainable integration and scope

**Objective:** Maintainers can understand and verify the experimental addition.

1. The implementation shall reuse existing divergence mathematics and result-reduction contracts where applicable rather than introduce competing formulas.
2. The implementation shall use idiomatic Python and names distinguishing model roles, token IDs, logits, probabilities, counts, durations and memory units.
3. Where arrays or DataFrames are used, the implementation shall document array axes and DataFrame row grain, indices, column meanings and alignment.
4. When validating the feature, the checks shall cover compatibility failure, unseen auxiliary contexts, zero-alpha identity, sham cancellation, interruption accounting and paired-analysis denominators using focused offline tests.
5. The documentation shall cite the motivating research and state that the initial trigram configuration is a feasibility adaptation with limited evidence for semantic unlearning.
