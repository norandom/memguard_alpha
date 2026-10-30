# Requirements Document

## Introduction

Researchers and downstream consumers of recall_guard need an optional additional measure to reduce reliance on memorized historical outcomes. The current guard detects contamination using MIA features and calibrated scores; it does not intervene in token generation.

This feature adds experimental divergence decoding: main-model logits are adjusted by alpha times the difference between older-cutoff and newer-cutoff auxiliary logits before sampling. It complements existing contamination measurement and confidence discounting. It does not certify unlearning, establish a new knowledge cutoff, or turn recall_guard into a forecaster.

The motivation is [Jane Street's exploratory study](https://blog.janestreet.com/mitigating-memorization-in-llms/) (September 30, 2026). Its reported cricket results motivate evaluation; they do not establish effectiveness for this project's models or datasets.

## Boundary Context

- **In scope:** explicit opt-in decoding, compatibility validation, configuration-specific calibration, single-draw and ensemble scoring, provenance, failure reporting, and controlled evaluation.
- **Out of scope:** training auxiliary models from scratch, distillation, natural-language autoencoders, prompt rewriting, guaranteed temporal knowledge removal, and changes to downstream portfolio policies.
- **Adjacent expectations:** operators supply accessible models with documented training cutoffs and a runtime that controls token selection. The current completion-only NVIDIA client cannot supply that capability. The optional reference model remains a scoring reference, distinct from the two decoding auxiliaries. Consumers retain responsibility for exposure and fallback policies.
- **Design dependencies:** selection of a runtime and compatible checkpoints, distribution semantics for MIA features, and evaluation protocol must be resolved with evidence during design. No particular checkpoint, hardware configuration, performance gain, or latency target is assumed.

## Requirements

### Requirement 1: Explicit opt-in and compatibility

**Objective:** As an existing consumer, I want mitigation to be optional, so that adopting the package does not change my current workflow.

#### Acceptance Criteria

1. When divergence decoding is not selected, recall_guard shall preserve existing scoring, failure, and ensemble behavior.
2. When a caller selects divergence decoding, recall_guard shall require an explicit configuration identifying the main model, older auxiliary, newer auxiliary, steering strength, and generation settings.
3. The recall_guard package shall keep inference dependencies needed only for divergence decoding out of the default installation and package-root import.
4. Where divergence decoding is enabled, recall_guard shall identify the feature as experimental and state that its output is not a certificate of freedom from memorization.

### Requirement 2: Configuration and capability validation

**Objective:** As an operator, I want unsupported configurations rejected clearly, so that an apparently steered result cannot conceal an incompatible execution path.

#### Acceptance Criteria

1. When validating a divergence configuration, recall_guard shall require identifiable model revisions and documented cutoff dates with the older auxiliary cutoff strictly preceding the newer auxiliary cutoff.
2. When validating model metadata, recall_guard shall distinguish operator-supplied cutoff claims from independently documented evidence.
3. If the selected runtime cannot adjust token selection before generation or provide the probability information required by the selected scoring contract, recall_guard shall reject the configuration before starting calibration or scored generation.
4. If model vocabularies, token meanings, special tokens, context limits, or prompt representations cannot satisfy the shared-prefix decoding contract, recall_guard shall reject the configuration with the incompatible capability identified.
5. If alpha is non-finite or outside the supported range declared by the selected backend, recall_guard shall reject the configuration.
6. When validation fails, recall_guard shall report the reason without silently substituting another model, configuration, or unsteered execution mode.

### Requirement 3: Divergence decoding behavior

**Objective:** As a researcher, I want the specified intervention applied during generation, so that I can evaluate the actual divergence-decoding method.

#### Acceptance Criteria

1. When generating a token in divergence mode, recall_guard shall use main logits plus alpha times older-auxiliary logits minus newer-auxiliary logits at the same generated prefix, with corresponding token meanings aligned.
2. When generating successive tokens, recall_guard shall condition all three models on the same previously selected continuation.
3. When producing probabilities for sampling or scoring, recall_guard shall identify the distribution used and apply its documented normalization and generation transformations consistently.
4. When alpha is zero, recall_guard shall reproduce the unsteered main-model next-token distribution under the same backend, prompt, and generation settings within a declared numerical tolerance.
5. If a required model fails or produces values from which no valid next-token distribution can be obtained, recall_guard shall stop that draw and report failure without returning it as a successful mitigated result.
6. When generation reaches the configured output limit or a stopping condition, recall_guard shall preserve the completion status needed to distinguish completed and truncated output.

### Requirement 4: Calibration and contamination-score integrity

**Objective:** As a consumer, I want scores calibrated for the actual generation process, so that steering does not create misleading contamination estimates.

#### Acceptance Criteria

1. When calibrating divergence mode, recall_guard shall use the selected decoding configuration and declared feature-distribution contract for calibration and subsequent scoring.
2. When identifying a calibration, recall_guard shall bind it to model revisions, alpha, prompt-format identity, generation settings, feature semantics, and calibration-corpus identity.
3. If a scoring configuration does not match its calibration identity, recall_guard shall reject its use as a calibrated divergence score and identify the mismatch.
4. When scoring a steered response, recall_guard shall use the existing MIA feature and MCS calibration definitions, with explicit probability provenance rather than an unrelated replacement contamination formula.
5. If required probability evidence is unavailable or calibration is unavailable or weak, recall_guard shall expose the corresponding failure or quality status and shall not describe the result as reliable mitigation evidence.
6. When reporting a steered contamination score, recall_guard shall retain the existing confidence-discount semantics and distinguish the score from evidence that the intervention reduced memorization.
7. When comparing baseline and steered outputs, recall_guard shall not classify lower confidence or lower calibrated contamination scores alone as demonstrated mitigation.

### Requirement 5: Scoring and ensemble integration

**Objective:** As a library consumer, I want mitigation to compose with supported scoring modes, so that I can use it without changing the meaning of ensemble results.

#### Acceptance Criteria

1. Where divergence decoding is supported, recall_guard shall support single-prompt, multiple-prompt, and ensemble scoring with the same configuration validation and calibration rules.
2. When executing an ensemble, recall_guard shall use one effective divergence configuration and calibration identity for all draws included in that ensemble.
3. If draws have incompatible decoding or calibration identities, recall_guard shall reject their combination into one calibrated ensemble result.
4. When reducing a steered ensemble, recall_guard shall preserve existing failure accounting, parsed-draw denominators, component verdicts, and the distinction between the exposure score and the representative draw's score.
5. When replaying a stored steered draw set, recall_guard shall preserve deterministic, order-independent reduction without claiming that fresh model generation is deterministic.

### Requirement 6: Provenance and operational visibility

**Objective:** As an operator, I want to audit what ran and its cost, so that I can reproduce comparisons and detect configuration drift.

#### Acceptance Criteria

1. When returning a divergence result, recall_guard shall expose the effective execution mode, three model identities and revisions, cutoff metadata, alpha, generation settings, backend identity, calibration identity, and execution timestamp.
2. When recording probability-derived features, recall_guard shall identify whether their probabilities came from the main or adjusted distribution and the transformations included.
3. When evaluating a run, recall_guard shall record dataset identities and content hashes, prompt-format identity, per-draw completion or failure status, and effective configuration identity.
4. When reporting execution cost, recall_guard shall report elapsed time and available usage measurements, identify unavailable measurements explicitly, and distinguish estimates from observed values.
5. If configuration or serving revisions change, recall_guard shall expose the changed identity rather than represent the new run as identical to the prior configuration.

### Requirement 7: Controlled evaluation and evidence limits

**Objective:** As a researcher, I want a controlled evaluation, so that apparent mitigation can be distinguished from loss of useful capability or dataset shift.

#### Acceptance Criteria

1. When evaluating divergence decoding, recall_guard shall compare unsteered, alpha-zero, and nonzero-alpha conditions using the same held-out examples and matched main-model generation settings.
2. When constructing temporal evaluation groups, recall_guard shall report their relationship to all three model cutoffs, including the interval between auxiliary cutoffs and the post-main-cutoff control group.
3. When reporting temporal comparisons, recall_guard shall describe matching or stratification of relevant dataset characteristics and identify remaining distribution-shift limitations.
4. When selecting alpha or calibrating scores, recall_guard shall keep tuning and calibration examples separate from the final held-out evaluation and report the split identities.
5. When reporting evaluation results, recall_guard shall include sample counts, failures and exclusions, contamination-score quality, task utility, confidence behavior, output validity, latency, available resource cost, and uncertainty estimates for comparisons.
6. When assigning a mitigation evaluation verdict, recall_guard shall apply criteria recorded before final held-out evaluation, including preserved out-of-sample utility and evidence beyond confidence reduction.
7. If evidence is insufficient, utility deteriorates beyond the declared tolerance, or temporal controls are confounded, recall_guard shall report those findings rather than claim demonstrated mitigation.
8. The recall_guard evaluation documentation shall state that pre-cutoff membership is an exposure proxy rather than proof that an example was memorized, and shall qualify score-quality claims accordingly.

### Requirement 8: Feasibility and supported-use documentation

**Objective:** As a maintainer, I want feasibility constraints made visible, so that implementation and adoption claims do not outrun the available models and runtime.

#### Acceptance Criteria

1. When documenting a divergence configuration as supported, recall_guard shall provide evidence of accessible compatible checkpoints, cutoff provenance, runtime capability, required resources, and an end-to-end generation and scoring check.
2. If suitable checkpoints or runtime capabilities are unavailable, recall_guard shall document the unmet dependency without presenting mocked execution as a supported working configuration.
3. When documenting setup, recall_guard shall describe optional dependencies, configuration, calibration, unsupported capabilities, failure behavior, and the limits of the evaluation evidence.
4. When documenting supported configurations, recall_guard shall distinguish technical execution support from empirically demonstrated mitigation effectiveness.

## Requirements Review

The draft was reviewed for scope coverage, numeric requirement IDs, EARS syntax, observable acceptance criteria, failure handling, and boundary continuity. Runtime selection, checkpoint selection, numerical tolerances, feature-distribution details, and the predeclared evaluation protocol remain design work; they do not imply approved implementation choices. Requirements approved on 2026-10-30 through the explicit kiro-spec-design -y command.


