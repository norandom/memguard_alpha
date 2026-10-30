# Integration findings for the task plan

Reviewed 2026-10-30. This is an integration-focused design proposal; provider
wire-contract verification is explicitly the first implementation task.

## Existing code evidence

- `core/nvidia_lm.py` hardcodes the NVIDIA endpoint and always requests
  logprobs. Its parser requires token evidence. Making that evidence optional
  in the existing scorer would conceal invalid feature inputs; use an explicit
  completion contract and compatibility adapter instead.
- `core/ensemble.py` already provides `EnsembleSpec`, `generate_ensemble` and
  pure reduction with caller projections, failure limits and content-based
  tie-breaking. The response typing is tied to NVIDIA's `CompletionResult`,
  but answer-level aggregation need not depend on token evidence.
- `harness/scorer.py` already supplies single, batch and ensemble scoring.
  Its ensemble path scores each response before reduction. Preserve that
  behavior and existing constructors while adding optional configured clients.
- Existing ensemble request limits count logical draw calls; transport retries
  happen inside the client. The new workflow needs shared physical-request
  accounting rather than assuming the existing limit bounds provider spend.
- Existing generation is synchronous and uses a thread pool. Retaining that
  interface avoids forcing a nested event loop in Jupyter.

## Decisions and scope

Use one configured-client contract and retain existing reduction logic. Avoid
a provider framework or a second ensemble implementation. Expose missing MCS
separately from available answer statistics. First rewrite scope is one accepted
rewrite per item shared by target draws. Multiple rewrites are not required to
measure the initial intervention and would confound rewrite variability with
target sampling.

Use paired item-level uncertainty. Predesignate the first draw for single-answer
comparisons and disclose its reuse within the ensemble. Preserve failures in
all-attempted metrics. The historical experiment has its own cohort validation;
it must not relax the existing post-cutoff harness guard.

## Unverified dependencies and measurement risks

Exact provider/model parameter support must be verified from official docs and
actual access probes. A generic endpoint URL is insufficient. The currently
established credential is for NVIDIA; no Anthropic/OpenAI live access is assumed.
Documented cutoffs must be sourced for exact selected model IDs. Corpus matching,
rewrite fidelity and alias ambiguity can limit historical conclusions.

Automated fidelity checks can miss semantic changes. The measurement includes
a separately recorded manual audit; its outcome cannot be inferred from the
rewriter or verifier accepting its own output. A small pilot measures effects
and execution, not conclusive de-memorization.

## Workflow notes

Used the Kiro task-generation and design guidance from the locally available
QOpenBB skill copies because this repository has no local skill/template tree.
The user explicitly requested task generation after requirements approval, so
the proposed design and tasks are supplied together for review. Neither design
nor task approval is inferred from that request. Documentation and measurement
tasks are retained because the user and requirements explicitly require them.
The project's removal of Sentrux takes precedence over generic template rules.
