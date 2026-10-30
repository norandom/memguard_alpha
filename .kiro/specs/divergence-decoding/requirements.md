# Requirements — Divergence Decoding MVP

## Introduction

Add an optional experimental mathematical layer for callers who already control model logits. The operation is main + alpha * (older - newer), applied before selecting each token. A minimal greedy decoding loop demonstrates its use with caller-supplied model callbacks. Existing recall_guard scoring and NVIDIA inference remain unchanged.

The user accepted this narrowed milestone on 2026-10-30. This revision supersedes the earlier 45-criterion scope; previous documents are preserved under archive/full-scope/.

## Boundary Context

- In scope: logits combination, input validation, a shared-prefix greedy loop, explicit stop/failure behavior, tests and usage documentation.
- Out of scope: model loading/hosting, tokenizer translation, checkpoint research, sampling strategies beyond greedy, calibration, p_memorized integration, ensemble/replay, manifests and empirical mitigation verdicts.
- Caller responsibilities: supply compatible model callbacks, identical token-ID meanings and prompt tokenization, declared context limits, and appropriate auxiliary checkpoints. The package validates declared compatibility, not training history or model internals.
- The existing completion-only NVIDIA API cannot use this layer without a different caller-controlled generation runtime.

## Requirements

### Requirement 1: Mathematical operation

**Objective:** As a caller with logits, I want a standalone steering function that can be used in my generation loop.

#### Acceptance Criteria

1. When supplied three nonempty equal-length finite one-dimensional logit vectors and finite nonnegative alpha, the divergence layer shall return main + alpha * (older - newer).
2. When alpha is zero, the divergence layer shall return an independent copy of the main logits without changing their values.
3. If vectors have invalid shapes, unequal lengths, non-finite values or invalid alpha, the divergence layer shall reject the call with an actionable validation error.
4. If the adjusted result is non-finite, the divergence layer shall reject the call.
5. The divergence layer shall leave caller input arrays unchanged.

### Requirement 2: Minimal decoding integration

**Objective:** As a caller controlling three models, I want a small greedy loop that supplies the same continuation to each model.

#### Acceptance Criteria

1. When starting decoding, the decoder shall require three callbacks with matching declared vocabulary identities and sizes, compatible declared EOS tokens, and positive context limits.
2. When producing each token, the decoder shall call all three callbacks with the same prompt-plus-generated token IDs and select the maximum adjusted logit, breaking ties by lowest token ID.
3. If the prompt and requested output budget exceed any declared model context limit, the decoder shall reject the request before invoking callbacks.
4. When an EOS token is selected or the output budget is reached, the decoder shall return generated token IDs and the corresponding finish reason.
5. If a callback fails or returns invalid logits, the decoder shall report the failed model role and step and shall not return a successful partial result or silently fall back.

### Requirement 3: Optionality and honest scope

**Objective:** As a recall_guard consumer, I want an experimental utility that does not change existing behavior or overstate its evidence.

#### Acceptance Criteria

1. When the new utility is not invoked, recall_guard shall preserve existing scorer, client and ensemble behavior.
2. The divergence layer shall require no additional runtime dependencies beyond the current default installation.
3. The divergence documentation shall describe required caller-controlled logits access, compatibility responsibilities, and the limitation of the current completion-only NVIDIA client.
4. The divergence documentation shall label the utility experimental and shall not present mathematical correctness as evidence of reduced memorization.
5. The divergence utility shall return no calibrated contamination score and shall not reuse an existing calibration for adjusted generation.

### Requirement 4: Correctness evidence

**Objective:** As a maintainer, I want focused tests and an executable example that verify the bounded feature.

#### Acceptance Criteria

1. When validating the mathematical layer, the test suite shall verify a hand-calculated nonzero-alpha example, zero-alpha identity, input nonmutation and invalid-input rejection.
2. When validating the decoder, the test suite shall verify shared prefixes, deterministic tie handling, context rejection before callback invocation, stopping and callback failure behavior.
3. The documentation shall include an executable callback-based example that requires no downloaded models or credentials and identifies its synthetic outputs as correctness evidence only.


### Requirement 5: Idiomatic code and semantic naming

**Objective:** As a maintainer, I want idiomatic Python and NumPy with names that match the data, so that mathematical operations and transformations are easy to review.

#### Acceptance Criteria

1. The implementation shall use idiomatic Python and NumPy operations, focused functions and explicit types without introducing unnecessary abstractions or dependencies.
2. The implementation shall name variables according to their domain meaning and current representation, distinguishing model sources, logits, probabilities, token IDs and counts; array axes and shapes shall be documented where not evident from the contract.
3. When generating implementation tasks and reviewing their completion, the task specification and reviewer shall explicitly check idiomatic code and semantic naming, including meaningful DataFrame column/index names if DataFrames are involved, without adding DataFrames to this milestone.
