> SUPERSEDED evaluation proposal (2026-11-01): the mandatory local runtime and fixed auxiliary selection below are historical. GPT-OSS-20B remains included. The [endpoint-based amendment](../divergence-decoding-evaluation/scope-amendment.md) and its requirements govern the follow-up. The completed mathematical MVP is unchanged.

# Model selection for the real-model evaluation follow-up

Date: **2026-11-01**. Status: proposed selection for the next specification;
not an amendment to the completed mathematical MVP's implementation approval.
User constraint: **`openai/gpt-oss-20b` is always included**.

## Selected first experiment

| Role | Selected model | Status |
| --- | --- | --- |
| Main target and unadjusted baseline | `openai/gpt-oss-20b`, official open weights in a caller-controlled Docker runtime | Existing model; exact revision and runtime precision must be pinned |
| Older auxiliary | `gpt-oss-token-trigram-older-v1`, a proposed locally fitted trigram language model using the main tokenizer | New experimental artifact, not a published checkpoint |
| Newer auxiliary | `gpt-oss-token-trigram-newer-v1`, the same estimator fitted on an expanded dated corpus | New experimental artifact, not a published checkpoint |

These auxiliary names are proposed artifact identifiers, not Hugging Face model
IDs. They must not be advertised as existing trained models or measured results.
Their vocabulary and token-ID meanings come from the exact main-model tokenizer.
Finite smoothed next-token log probabilities are valid auxiliary logits for the
existing combination. Normalization contributes only a per-prefix constant,
which does not change next-token softmax or greedy selection. The estimator,
smoothing, unseen contexts and special-token policy still need design review.

The selected first experiment tests whether a cheap aligned auxiliary contrast
can steer real `gpt-oss-20b` generation and suppress targeted recall without
collapsing utility. It is not an assumed solution to semantic financial lookahead.

## Why these choices follow from our results

- October 30 NIM results establish an existing scoring baseline for this target:
  held-out calibration AUC 0.876, 312/330 parsed, 170/312 correct. Near-baseline
  market-direction accuracy does not establish an exploitable recall advantage.
  A recall-sensitive screening task is required before efficacy measurement.
- The October 31 Opus pilot was 11/14 correct in both single and ensemble views
  on common usable items. It does not justify making ensembles a prerequisite.
  Start with single greedy answers; identical greedy repeats are diagnostics.
- November 1 NIM access succeeded once, but fitting and evaluation were blocked
  by gateway failures. Keep the same model identity while moving token selection
  into a runtime we control. Hosted versus self-hosted results are separate
  conditions; do not claim numerical parity with NIM.
- All 16 Opus rewrites were rejected. That argues for separating rewriting from
  the direct decoding test, not for treating rewritten answers as valid evidence.

See the saved [single-answer evidence](../../../docs/results/2026-11-01-recall-guard-single.md)
and [ensemble evidence](../../../docs/results/2026-11-01-recall-guard-ensemble.md).

## Corpus windows define the auxiliary contrast

Propose an older corpus ending **2019-12-31**, and a newer corpus consisting
of that same corpus plus matched material from **2020-01-01 through 2024-06-30**.
The latter boundary follows the official main-model June 2024 cutoff; it is not
proof of the main model's exact training membership. Both boundaries must be
reviewed with the actual data manifest before requirements approval.

Fit both auxiliaries from scratch with the same tokenizer and estimation policy.
Publication dates and corpus hashes establish their controlled exposure, rather
than inferring knowledge windows from checkpoint release dates. Match sources
and text processing, record token counts, and include a same-period matched-size
control to expose corpus-size or domain-shift effects. Reject missing timestamps,
future revisions that invalidate the boundary, and corpus overlap with held-out
prompt/answer documents. Targeted recall examples should be held-out paraphrases
of facts whose exposure is declared, not accidental duplicated evaluation text.

The newer auxiliary intentionally sees later historical information; this is the
steering contrast. Original prompts and older auxiliary inputs must respect their
declared information boundaries. Hold out all post-June-2024 utility events from
both auxiliary fits. No evaluation outcome is used for alpha selection.

## Evidence and limitations of trigram auxiliaries

Merchant and Levy's [Divergence Decoding: Inference-Time Unlearning via Auxiliary Models](https://arxiv.org/html/2605.31293v1)
studies small neural and trigram auxiliaries. Section 4.4 finds the trigram
variant useful in some near-verbatim settings but poor on TOFU's more complex
questions. This supports selecting it as a low-cost feasibility baseline;
it does not establish semantic lookahead mitigation. The follow-up spec must
report that limitation even if the integration works.

Jane Street's [memorization study](https://blog.janestreet.com/mitigating-memorization-in-llms/)
motivates the older-minus-newer steering direction and separate synthetic rewrite
experiment. This selected trio adapts that idea; it is not a reproduction of its
neural auxiliary experiment.

## Alternatives excluded from this first run

| Candidate | Decision |
| --- | --- |
| `openai/gpt-oss-120b` as the older auxiliary | Exclude: the official family model card gives June 2024 as the cutoff; size is not a temporal exposure contrast |
| Arbitrary Qwen, Llama or OLMo checkpoints as direct auxiliaries | Exclude: different tokenizers cannot be combined by vector index; a cross-tokenizer bridge would be a separate method extension |
| Two copies of `gpt-oss-20b` as auxiliaries | Use only for cancellation/sham checks; their difference is zero |
| Neural auxiliaries trained with the GPT-OSS tokenizer | Possible later escalation if the trigram baseline is inadequate; architecture, corpora, training and compute require a separate concrete proposal |
| Claude Opus 5.5 | Retain existing completion-only results as context; not a source of next-token vectors for this experiment |

The [official model card](https://deploymentsafety.openai.com/gpt-oss) documents
the family cutoff and a 12.8 GiB released checkpoint for the 20b model. The
[official implementation](https://github.com/openai/gpt-oss) provides inference
code and requires Harmony formatting. Checkpoint size is not peak runtime memory.
Select a GPU/runtime with measured headroom for weights, KV cache, activations
and full vocabulary outputs; Docker alone supplies none of that capacity.

## Requirements handoff

The next spec should cover an optional experimental runner, auxiliary fitting
and provenance, Harmony-compatible callbacks, zero-alpha and sham parity,
real-token steering evidence, recall/utility evaluation and honest unavailable
scores. Reuse the existing mathematical layer. Keep inference dependencies out
of the library's default installation. Use idiomatic code with array axes and
units named explicitly. Sampling, ensembles and neural-auxiliary training are
separate follow-ups unless explicitly added during requirements review.

No model weights were downloaded and no training, provisioning or inference was
performed for this selection. Remaining decisions are corpus access/licensing,
runtime feasibility, exact revisions, maximum context/output lengths and compute
budget. These are inputs to the proposed experiment, not completed checks.
