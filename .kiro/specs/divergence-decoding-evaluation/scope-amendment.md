# Scope amendment: endpoint evidence, correction calibration, held-out effects

Date: 2026-11-01. The user requested this amendment after agreeing to the
collect, tune, freeze and evaluate workflow. It supersedes the hosting-first
evaluation selection, design and uncompleted task sequence.

## Governing decision

Recall Guard consumes configured model endpoints and statistical sources.
Local hosting and GPU capacity are not prerequisites. Establish available
evidence and correction controls first. Tokenizers define token events, not
conditional model probabilities.

The statistical relationship is:

    adjusted_probability ∝ target_probability * (older_probability / newer_probability)^alpha

All terms refer to the same prefix and aligned token events. Complete log
probabilities differ from logits by per-prefix constants that cancel after
normalization. Temperature, truncation and renormalization still need verification.
Partial candidates and sample-derived estimates require explicit approximation
validation. Calibration cannot recover omitted information or create generation
controls.

Auxiliary sources may be documented endpoints or fitted statistical artifacts.
Advertised cutoff dates alone do not establish a controlled temporal contrast.
Trigrams remain an optional estimator, with semantic limitations, not a mandatory
choice. Source provenance and confounds must be recorded.

## Supersession map

| Earlier decision | Current disposition |
| --- | --- |
| Mandatory GPT-OSS-20B | Retained in endpoint-based study plans; unsupported states explicit |
| Mandatory local Transformers/GPU runtime | Superseded; optional separately justified research route |
| Fixed local trigram pair and dates | Optional historical candidate pending source assessment |
| Hardware smoke as first gate | Replaced by endpoint capability and statistical-source assessment |
| Fixed 8+8 pilot and efficacy thresholds | Historical proposal; revised design must justify and freeze sample sizes and criteria |
| Single greedy answers only | Single-answer baseline retained; ensembles separately identified and budgeted when evaluated |
| Completed mathematical decoder | Unchanged; no added efficacy claim |
| Completed tasks 1.1, 1.2, 2.1, 2.2 | Historical completion preserved; reuse assessed during design |
| Uncompleted evaluation tasks | Superseded; not executable under earlier approvals |

## Revised flow and boundaries

Capability assessment -> statistical source validation -> development tuning of
alpha -> freeze -> paired held-out evaluation.

Reuse existing calibration orchestration where applicable. Correction tuning is
separate from the contamination classifier. AUC measures discrimination.
Mitigation requires a recall-sensitive outcome and preserved task utility with
paired uncertainty. Sample-derived probability precision is a separate estimation
question and needs its own request budget.

The original requirements, design, tasks and metadata are preserved in
[the revision-1 archive](archive/2026-11-01-local-runtime/).
Existing code, reviews, receipts and the optional Docker image remain.
Removing optional hosting dependencies requires a concrete cleanup assessment;
this amendment neither deletes them nor makes them required.

## Status and next phase

### Anthropic completion-only inclusion

User clarification: include the Anthropic endpoint without requiring target
logprobs. Target completions can supply samples; auxiliary sources supply the
statistical contrast. The proposed design must define the sample-level correction
and its estimand, calibration and uncertainty. Candidate reweighting or resampling
is a research route, not an established implementation or efficacy result.
This route must remain distinct from exact token-level divergence decoding and
from synthetic rewriting. It does not require reconstruction of the target's
full next-token distribution. The specific Anthropic model and live request
budget remain explicit run-plan choices; existing credentials do not authorize
additional inference.

Requirements revision 2 is generated, not approved on the user's behalf.
Earlier approvals apply to archived revision 1. Revised design and executable
tasks must follow requirements review. Design must resolve endpoint score
semantics, continuation and correction controls, auxiliary sources, approximation
validation and budgets. No provider support is presumed, no new inference is
authorized, and completion rewriting remains a separate explicit intervention.

## Sources

- [Jane Street: Mitigating memorization in LLMs](https://blog.janestreet.com/mitigating-memorization-in-llms/): temporal auxiliary contrast and separate rewriting experiments.
- [Merchant and Levy: Divergence Decoding, sections 3.1–3.2](https://arxiv.org/html/2605.31293v1#S3): score adjustment and equivalent probability-ratio formulation.

Neither source establishes compatibility with every hosted completion endpoint.
