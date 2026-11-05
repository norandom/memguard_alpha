> Scope amended 2026-11-01. The local-runtime proposal below is historical.
> The current plan starts with configured endpoints: verify statistical evidence
> and correction controls, validate auxiliary sources, tune correction strength on
> development data, freeze it, and measure paired held-out effects. Complete
> compatible log probabilities can replace raw logits; partial or sampled evidence
> requires explicit approximation validation. GPU hosting is not a prerequisite.
> GPT-OSS-20B remains included. No additional live execution is authorized.

# Test divergence decoding with open weights and a controlled generation loop

Proposal dated **2026-11-01**. Status: proposed experiment; no additional live
requests, model downloads, compute provisioning or implementation are authorized
by this document. Existing results are documented
[without ensembles](results/2026-11-01-recall-guard-single.md) and
[with ensembles](results/2026-11-01-recall-guard-ensemble.md).

## Open weights make the direct experiment possible

**Divergence decoding** is the name of the token-level method in James Somers'
[Mitigating memorization in LLMs](https://blog.janestreet.com/mitigating-memorization-in-llms/)
(Jane Street, September 30, 2026). Recall Guard implements its logit combination
and a caller-controlled greedy decoding loop. It has not yet measured its
mitigation efficacy on real models.

```text
adjusted_logits = main_logits + alpha * (older_aux_logits - newer_aux_logits)
```

A known tokenizer is a useful starting point for an open-weight experiment.
The runtime must also obtain all three full next-token vectors for the same
prefix and select the next token from the adjusted vector. Each selected token
then extends the prefix supplied to all three models.

| Configuration | Can test exact divergence decoding? | Needed access |
| --- | --- | --- |
| Open weights in a controlled local or self-hosted runtime | Yes, once compatible models are selected | Full logits, aligned token IDs and caller-controlled selection |
| Hosted endpoint with a server-side implementation of the method | Yes | Verified server-side three-model decoding and provenance |
| Hosted NIM completion with returned logprobs only | No through that interface | Returned evidence does not give token-selection control |
| Ordinary Anthropic, OpenAI or Mistral text completions | No through that interface | Use a separate completion-compatible intervention |

Publicly known tokenizers do not imply identical token IDs across models.
For the first implementation, select a shared tokenizer and verify vocabulary,
token-ID meanings, special tokens and prefix encoding; reject mismatches.
Freeze model revisions, tokenizer hashes, templates, precision and generation
settings. Compatible older/newer auxiliaries also need documented training
histories supporting the intended knowledge window. Open weights alone do not
establish their cutoffs. **Selection of that model trio and its compute budget
is the remaining prerequisite for a real divergence experiment.** Docker supplies
the reproducible execution environment; weights and model inference compute
are separate requirements from this library's API-based scoring path.

## First prove the intervention runs, then measure its effect

### Selected first configuration

`openai/gpt-oss-20b` is the required main target and baseline. For the first
feasibility experiment, select two proposed **trigram language models fitted
with its exact tokenizer**, rather than assume a compatible older/newer pair of
published neural checkpoints exists. Proposed artifacts are
`gpt-oss-token-trigram-older-v1` and `gpt-oss-token-trigram-newer-v1`; neither is
trained yet. Propose dated corpora ending December 2019 and June 2024 respectively,
with common sources and processing, controlled overlap, and size-matched controls.
These dates describe auxiliary corpus exposure, not verified main-model membership.

This is a direct logit-combination experiment using small probabilistic models.
It can establish real-model steering and test targeted recall, but cannot alone
validate semantic financial lookahead mitigation. Published
[auxiliary-model research](https://arxiv.org/html/2605.31293v1) finds limited
trigram utility on more complex questions. A neural auxiliary experiment remains
a separately budgeted escalation if the first baseline is inadequate. Start
without ensembles because repeated greedy answers add no independent evidence.

The [official GPT-OSS model card](https://deploymentsafety.openai.com/gpt-oss)
documents June 2024 as the family cutoff. `gpt-oss-120b` therefore does not supply
an older temporal checkpoint merely by being a different size. The official
[inference implementation](https://github.com/openai/gpt-oss) provides the starting
point for a Harmony-compatible controlled loop. Actual hardware capacity, model
revision, precision and peak memory remain to be verified before execution.

### Feasibility and evaluation sequence

1. Run an offline integration check through Recall Guard's existing decoder and
   real-model callbacks. Verify identical prefixes, finite aligned vectors,
   deterministic greedy selection and cancellation. At `alpha=0`, require
   token-for-token parity with the unadjusted main-model loop under identical
   numerical settings. Record any numerical nondeterminism before evaluation.
2. Use a separate development corpus to choose one nonzero `alpha` and freeze
   it before held-out outcomes. Record the proposed grid and selection criterion;
   never select the best held-out result. A nonzero setting need not change every
   answer, so report changed-token and changed-answer counts.
3. Freeze disjoint calibration, development and held-out evaluation inputs.
   For recall-sensitive evaluation, use events in the documented older-to-newer
   exposure window; for utility controls, use events after the main and newer
   auxiliary cutoffs. Match domain, task, source format, difficulty and outcome
   balance where possible. Keep information available after each as-of date out
   of prompts, calibration inputs and rewrite instructions. Audit retrospective
   source revisions and report remaining temporal confounds.
4. Compare the same main model at `alpha=0` and the frozen nonzero `alpha`.
   Keep prompts, output projection, precision and token limits fixed. Interleave
   paired conditions using a frozen order. A useful third diagnostic is a
   sham adjustment with identical auxiliary vectors, which must cancel.
5. First run a 16-item feasibility pilot with eight items per cohort. Its purpose
   is to expose integration and coverage failures. Propose a larger study only
   after using pilot uncertainty to justify sample size for the decision below.

## Report two views without confusing aggregation with mitigation

| Generation condition | Single-answer view | Ensemble view |
| --- | --- | --- |
| Unadjusted main model | Predesignated draw 0 | Three draws, including draw 0 |
| Divergence decoding | Predesignated draw 0 | Three draws, including draw 0 |

The existing decoder is greedy. Repeating identical deterministic requests is
an execution check, not independent evidence or a source of ensemble diversity.
If a sampling adapter is proposed, specify and validate its token selection,
seeds and generation contract before implementation and run approval. Until
then, label identical greedy ensembles as a diagnostic; do not promise an
ensemble gain. Reuse Recall Guard's existing reduction rather than add a new
aggregation formula.

For each view report correct/planned and correct/usable, coverage, failure
reasons, confidence, latency, token counts and actual resource usage. Report
paired effects within cohorts and the change in the pre/post accuracy gap:

```text
gap_change = (mitigated_pre - mitigated_post) - (original_pre - original_post)
utility_change = mitigated_post - original_post
```

Negative `gap_change` is consistent with reducing the historical advantage; it
does not prove memory removal. A general loss of task ability can also lower
historical accuracy. Report item-paired, cohort-stratified bootstrap intervals
and retain every arm/draw together during resampling. Report all-planned and
common-usable results so filtering cannot manufacture an effect.

The proposed efficacy decision requires the upper 95% interval bound for
`gap_change` to be below zero and the lower bound for `utility_change` to exceed
**-5 percentage points**. Freeze that tolerance and the analysis before the
held-out run; feasibility pilots cannot establish it reliably. A weak original
historical advantage makes mitigation efficacy inconclusive on that task.
Even a positive result remains evidence about this corpus and runtime.

MCS is a secondary outcome requiring independent calibration matched to each
generation condition. Report held-out calibration quality and its limitations;
do not reuse an unadjusted calibrator to claim a causal reduction in MCS. If
generated-token logprobs cannot be collected with the scorer's required
semantics, report MCS unavailable. State whether reported probabilities come
from the adjusted distribution or the original main model. Neither confidence
nor agreement substitutes for token evidence.

## Test synthetic rewriting separately on completion-only APIs

The blog also studies **synthetic rewriting**, including podcast-style
presentation. This is a separate method from divergence decoding. Recall
Guard's completion-only adaptation preserves factual information rather than
removing quoted material. It can be tested through ordinary completions without
logprobs; MCS remains unavailable when genuine token evidence is absent.

Before another live rewrite study, replace the overrestrictive fidelity policy
in a separately reviewed specification. Validate it offline on accepted
equivalent presentations and deliberately corrupted facts, numbers, dates,
negations and attribution. Use a canonical fact representation, preserving
as-of boundaries and qualifications, with semantic checks for residual changes.
Have outcome-blinded human review assess the policy on development examples;
freeze it before new held-out generation. Never retroactively accept the
October 31 rejected rewrites as efficacy evidence.

Then compare original and accepted rewritten inputs, each with draw 0 and a
three-draw ensemble. Share one accepted rewrite across an item's draws. Report
fidelity rejection and verifier disagreement as first-class outcomes. A revised
policy's acceptance rate alone is not success: the same gap and utility
criteria, usable-pair counts and uncertainty apply.

A 16-item, three-draw, two-condition feasibility study needs at most 96 target
generations plus 16 rewrite generations and up to 16 verifier calls: **128 calls
before access probes or any logprob calibration**. Draw 0 adds no calls.
This is a planning subtotal, not an approved budget. Declare every physical
request, probe, retry and calibration allowance before a live run. For NIM use
serial starts at least 2 seconds apart, accounting for other clients sharing
the user-specified 40 RPM quota; stop on rate-limit responses with retries
disabled. Keep this proposal separate from prior consumed request allowances.

