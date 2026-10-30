# Design: configurable completion mitigation

Status: requirements, design and tasks approved by the user on 2026-10-30.
Implementation is in progress.

## Boundaries and integration

Extend the current Python workflow rather than create an independent ensemble
implementation. Core owns model configuration, completion contracts, transport
and pure reduction. Harness owns mitigation orchestration, optional scoring and
evaluation. MIA continues to require genuine token evidence. The mathematical
divergence decoder remains a separate opt-in interface.

Existing NVIDIA construction and `MemoryGuardedScorer` calls remain supported.
Introduce an optional configured-client path to scorer calibration. A new
`RecallWorkflow` composes a completion client, optional matching scorer and
optional rewrite policy. It exposes single and ensemble execution without
forcing calibration on text-only users. Existing `score_ensemble`, `EnsembleSpec`
and `generate_ensemble` retain their current semantics for existing callers.

## Contracts

- `ModelConfig`: protocol, endpoint URL, model ID, credential reference,
  explicitly requested generation parameters, declared capabilities, timeout,
  retry and pacing settings. Credentials are resolved at runtime and excluded
  from serialization. No global model selection or notebook-only defaults.
- `CapabilityReport`: per-feature supported/unsupported/unverified status with
  evidence source. Validate declarations locally, then probe required features
  with actual generation before scheduling a batch. A successful response does
  not establish that an unechoed temperature or seed was honored.
- `ModelResponse`: content, optional token evidence, usage, finish reason,
  requested settings, confirmed settings and request identity. Missing evidence
  is `None`, never fabricated zeros. Preserve existing `CompletionResult`
  construction and imports through an explicit legacy adapter.
- `CompletionClient.generate`: synchronous, typed completion contract usable
  from Jupyter's running event loop without calling `asyncio.run`. Normalize
  protocol-specific requests and responses at the provider boundary.
- `CallBudget`: shared thread-safe admission of every physical request,
  including retries, probes, rewriting and verification. Reject before issuing
  an over-budget request. Record logical draws separately from physical calls.
- `WorkflowResult`: original evidence, accepted rewrite if any, answer/draws,
  fidelity status, optional guarded scores, explicit unavailable reasons,
  effective method and provenance. Failure records preserve completed stages.

Protocol selection is explicit: OpenAI-compatible chat, native Anthropic
messages, or caller callback. Mistral and NIM support is validated against the
selected protocol/model; a URL alone is not a capability declaration. The
implementation foundation task verifies current official provider contracts
and builds sanitized fixtures before adapters are written. Unsupported
parameters produce actionable errors; no retry strips them automatically.
The implementation must not claim support for every provider endpoint variant.
Caller callbacks must expose physical-request admission for strict-budget runs.
If they hide internal retries, record accounting as unverified and reject their
use in a strict-budget live experiment rather than claim an enforceable cap.

## Rewrite and fidelity path

Inputs separate task instructions, source evidence and as-of date. Version the
rewrite template. First mode rewrites evidence into a neutral conversational
presentation while preserving names, numbers, dates, claims and uncertainty.
It does not remove quotations or facts to imitate an external experiment.

Use deterministic entity/date/quantity checks plus a structured semantic
comparison by a separately configured verifier completion client. Neither
method is a guarantee. Malformed, missing or negative verification rejects the
rewrite before target generation. Retain checks and both texts for review.
Treat source text as evidence, not instructions to the rewriter or verifier.

The initial workflow uses one accepted rewrite per item shared by its target
draws. Record `rewrite_scope=shared_per_item`; reject unsupported scope values.
Multiple rewrites per item are outside this first implementation. This keeps
rewrite variability distinct from target sampling and satisfies the requirement
to disclose rewrite/draw association.

Disabled rewriting issues no rewrite or verification calls. No silent fallback
turns failed mitigation into a successful original-prompt answer.

## Scoring and ensemble compatibility

Generalize the existing ensemble reducer's response typing to a structural
content-bearing response contract; preserve exact return objects and legacy
behavior. Use caller projections for answer labels and numeric components.
Answer consensus remains distinct from MCS estimates. Minimum usable draws,
failure proportions, content tie-breaking and retained-draw replay stay intact.

The optional configured-client scorer path validates the complete token evidence
needed by existing features. Logprobs alone do not imply feature compatibility.
Score each draw independently before reducing scored outputs; do not score
averaged features. Bind calibration to model, endpoint, generation settings and
intervention/template identity. Mismatches fail explicitly. Separate calibration
is needed for rewritten generation, applying the same rewrite pipeline to the
calibration corpora; no reuse of the existing unsteered fit by default.

## Evaluation protocol

Four conditions: original single, original n-draw ensemble, rewritten single,
rewritten n-draw ensemble. Use identical target settings and item IDs. Alternate
original/rewrite target ordering across items. Generate n target draws per
evidence arm and predesignate draw 0 as the single-response condition; do not
select a successful draw after observing results. Single/ensemble observations
are dependent by design and the report must state this reuse.

Store one row per item/arm/draw plus rewrite records and request events. Bootstrap
paired differences by item, preserving all its arms and draws together. Report
accuracy, valid-pair accuracy, parse and fidelity acceptance, agreement,
latency and usage. All-attempted accuracy counts failures as unsuccessful task
outcomes; it is not interpreted as recall reduction. Confidence metrics use
only explicitly defined confidence fields. Unknown monetary cost stays unknown.

Predeclare exact models, sourced cutoffs, cohort matching, sample/draw counts,
utility tolerance, fidelity audit selection and maximum physical requests.
Use a versioned experiment config. Historical cohorts require model-specific
documented cutoffs; uncertain aliases permit general comparison only. Keep
historical mitigation evaluation separate from the existing post-cutoff-only
harness guard, which must not be weakened.

Audit a deterministic, stratified sample of accepted and rejected rewrites
manually before interpreting efficacy; retain selections and judgments. An
automated model judge cannot serve as independent proof of factual fidelity.
An exploratory pilot may establish execution and estimate effects, but cannot
be labeled a powered mitigation-success result. Missing access, unmatched
cohorts, inadequate pairs or missing audits remain explicit measurement gaps.

## File structure and responsibility plan

| Boundary | New or modified paths | Responsibility |
| --- | --- | --- |
| Completion contracts | `recall_guard/core/completion.py` | Configuration, response and capability types |
| Request control | `recall_guard/core/request_control.py` | Shared budget, pacing and retry admission |
| Provider adapters | `recall_guard/core/providers.py`, `core/nvidia_lm.py` | Protocol translation, callback and legacy compatibility |
| Ensemble core | `recall_guard/core/ensemble.py` | Response typing and existing reduction integration |
| Rewrite policy | `recall_guard/harness/rewriting.py` | Versioned transformation and fidelity checks |
| Workflow | `recall_guard/harness/workflow.py` | Compose capabilities, generation, rewriting and ensembles |
| Scorer integration | `recall_guard/harness/scorer.py` | Optional configured client and calibration identity |
| Experiment orchestration | `recall_guard/harness/intervention.py` | Four conditions and resumable records without automatic duplicate calls |
| Experiment contracts | `recall_guard/harness/intervention_types.py` | Versioned protocol, record identities and completion/resume markers |
| Experiment statistics | `recall_guard/harness/intervention_metrics.py` | Paired item-level metrics and uncertainty |
| Experiment reporting | `recall_guard/harness/intervention_report.py` | Portable records, manifest and dated results |
| CLI and inputs | `scripts/run_intervention.py`, `data/experiments/`, `data/cutoffs.yaml` | Explicit protocol, sourced model metadata and budget |
| Adoption and tests | `docs/`, `examples/`, corresponding `tests/core` and `tests/harness` | Notebook example, regression and live-report artifacts |

Define experiment contracts before execution and persistence consume them.
Prepare separate original/rewrite calibrations before the scored live pilot,
verify calibration/evaluation separation and include those calls in its budget.

No new runtime dependency is assumed: reuse requests, NumPy and existing
statistics. Add dependencies only if the protocol audit demonstrates a need.
Keep lightweight imports and architecture direction. Split a module if it
gains multiple responsibilities; avoid a generic provider/plugin framework.

## Validation and requirement traceability

Configuration/adapters cover 1 and 7; rewrite checks cover 2; shared request
control and reporting cover 3; evaluator/statistics cover 4 and 9; protocol and
reports cover 5; native tests, examples and naming review cover 6; ensemble
and scorer integration cover 8. Task mappings enumerate acceptance criteria.

Offline checks use sanitized provider fixtures, recorded draws and adverse
responses: auth rejection, unsupported parameters, missing token evidence,
budget exhaustion, fidelity failures and unparseable completions. Verify legacy
scorer parity, reduction replay and no extra calls when features are disabled.
Run notebook smoke in a live kernel and native Windows/Linux tests. Live checks
require an authenticated logprob-capable model and a completion-only frontier
configuration; a forced text-only run on the same model is a useful control,
but does not demonstrate a second provider's support.

## Review status

This design is approved for the requested task plan. Provider wire
contracts and exact experiment inventory are explicit foundation deliverables,
not claimed verified facts. The user approved this design and its task plan on 2026-10-30.
