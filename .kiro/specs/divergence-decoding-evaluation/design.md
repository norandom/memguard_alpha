# Design: bounded real-model divergence evaluation

## Boundary commitments

An optional experiment uses `openai/gpt-oss-20b` and two tokenizer-aligned
trigram estimators. It owns corpus validation, fitting, model callbacks, paired
execution and reporting. It calls the existing `combine_logits` and
`decode_greedy`; it adds no alternative decoding equation or public scorer.

Main-model weights remain fixed. Neural auxiliary training, stochastic sampling,
cross-tokenizer translation, hosted serving and rewriting are outside this
design. First-run MCS is explicitly unavailable: adding condition-matched
calibration is a later extension. Prior NIM calibration is never imported.

The first result is a feasibility and lexical-recall experiment. Semantic
financial lookahead remains unproven even if that experiment succeeds.

## Components and dependency direction

```text
CLI / experiment orchestration
    -> manifests, budgets, artifact persistence, paired analysis
    -> trigram fitting and callbacks / GPT-OSS callback
    -> existing core ModelSource, decode_greedy, combine_logits
```

Use a small `experiments/divergence_evaluation/` package, excluded from the
distributed library, with `contracts.py`, `corpora.py`, `trigram.py`,
`model_runtime.py`, `runner.py`, `analysis.py` and `__main__.py`. Contracts and
corpus/analysis helpers must import without GPU libraries. Torch/Transformers
imports occur only in the experimental runtime module. No core imports from
experiments; no new service, queue or generic provider abstraction.

Extend the canonical Dockerfile with an optional experiment target using a
dedicated optional dependency group in the existing project lock. Select and
lock compatible Torch, Transformers and GPT-OSS kernel versions during the
runtime feasibility task. The normal library installation does not install
them. Require a recorded image digest and model revision; never run a moving
`latest` image as a measured configuration.

The selected backend is a direct Transformers causal-LM forward pass, under
inference mode, returning the final-position logits. Its API supports GPT-OSS;
the exact hardware/kernel/precision combination is not yet validated. Default
quantized weights are a candidate, not a promise that the machine can load them.
Do not silently dequantize or switch precision after an out-of-memory failure.

## Contracts and artifact layout

Use frozen dataclasses, explicit enums and JSON serialization; no DataFrames
are needed. Arrays name their representation and units. A vocabulary score
vector has shape `(vocabulary_size,)`, axis 0 indexed by the main token ID.

| Contract | Required information |
| --- | --- |
| `CorpusDocument` | document/source ID, local content hash, rights reference, publication/revision timestamps, cohort, split, fact-exposure IDs |
| `AuxiliaryManifest` | tokenizer mapping hash, estimator version, smoothing, source hashes, counts by source/window, fitting limits |
| `EvaluationPlan` | main revision, image identity, tokenizer and template hashes, precision, settings, alpha selection protocol, splits, parser, budgets, hardware |
| `OperationRecord` | stable operation ID, phase/item/arm, intent timestamp, completion status, elapsed time, generated-token count, memory/cost evidence |
| `ItemResult` | item/cohort/arm, answer and parse state, correctness, token-change evidence, finish reason, unavailable-score reasons |

Each new run directory contains `plan.json`, `provenance.json`,
`operations.jsonl`, `items.jsonl`, `metrics.json` and `report.md`. Auxiliary
artifacts have their own manifests and sparse counts; avoid pickle and load
NumPy data with object deserialization disabled. Write intent before work,
flush terminal records before the next operation, and write final summaries by
atomic replacement. Refuse existing output directories. Partial work stays
failed/cancelled/incomplete; it is never resumed implicitly.

## Corpus preparation and fitting

Require operator-supplied local corpus manifests. There is no automatic web
scraper. Preparation validates rights references, timestamps and revisions,
content hashes, partitions and duplication. Missing evidence blocks admission.
Human audit supplies semantic near-duplicate and factual-exposure judgments;
exact hashes alone are not represented as semantic validation.

Older material ends 2019-12-31; newer adds matched material through 2024-06-30.
Produce source-stratified counts and a deterministic same-period, equal-token-
budget control, using the frozen seed and whole-document selection with actual
size discrepancy reported. Do not silently clip documents to manufacture dates
or matching. Exact corpus sources and sizes must be specified in the run plan.

Trigram counts are sparse mappings from the last two token IDs to next-token
counts. Reset at document boundaries; do not learn cross-document transitions.
Use interpolated smoothing, with the same frozen `beta=1` for both estimators:

```text
p1(t) = (count(t) + 1) / (total_tokens + vocabulary_size)
p2(t | b) = (count(b,t) + beta * p1(t)) / (count(b) + beta)
p3(t | a,b) = (count(a,b,t) + beta * p2(t | b)) / (count(a,b) + beta)
auxiliary_logits = log(p3)
```

Denominators count outgoing transitions for each context, not all occurrences
of a token at document end. Short prefixes back off to p2/p1. An unseen context
backs off exactly; all token probabilities are positive. Build one dense vector
per callback from sparse counts, not a dense vocabulary-cubed table. Serialize
the policy and document resets. Fitting enforces document, token and memory
caps and reports actual counts without silently truncating the approved corpus.

Harmony structural tokens and unused output slots must receive identical
auxiliary log-scores in both roles, making their adjustment zero. The first
version neutralizes auxiliary adjustment while the prefix is inside Harmony
headers; content tokens in analysis and final channels can be steered. This
policy is explicit experimental behavior and must be tested at boundaries.

## Tokenization, main callback and decoding

Pin the complete tokenizer mapping, added tokens, Harmony template and terminal
token policy. Validate against the model's output-head size. Preserve padded
output positions; do not renumber IDs or truncate vectors. Padded slots receive
zero auxiliary difference. If the main model selects an unmapped token, record
a generation failure instead of decoding it as valid text.

Format each prompt once using the pinned Harmony template, with tools disabled
and explicit low reasoning effort. All roles receive exactly those token IDs.
The main callback owns a KV cache for one arm. It verifies each requested prefix
extends the cached prefix by one selected token, otherwise clears and rebuilds
the cache. Reset between items and arms. Validate cached versus uncached logits
and greedy output on a short fixture before measured execution.

The callback checks cancellation and deadlines before each forward pass, records
the operation step, and returns finite logits as a CPU vector. Callback failures
are wrapped by the existing `DecodeError`; the runner preserves the role/step
and uses its own cancellation state to classify cancellation correctly. It does
not convert a partial decode into success. A worker process can be terminated
after the declared grace period if a GPU call fails to return; the parent owns
the durable ledger and marks that operation incomplete.

Use `decode_greedy` unchanged. Compare alpha-zero output against an independent
unadjusted greedy forward loop with the same precision, cache policy, terminal
tokens and token limit. Compare the sham condition with identical auxiliary
callbacks. If parity cannot be reproduced, stop before the efficacy pilot.
Parse only the Harmony final channel, preserving final-answer absence and length
exhaustion as failures. Retain reasoning-channel token counts, but do not publish
raw reasoning text as part of the report.

Changed-token evidence compares baseline and adjusted argmax at the same prefix
during the adjusted trajectory. Changed-answer counts compare completed paired
arms. Do not align token positions from diverged trajectories as if they shared
a prefix. Zero-alpha baseline uses the same wrapper and logging path.

## Lifecycle and resources

Expose separate operations: `plan`, `validate-corpora`, `fit-auxiliaries`,
`check-runtime`, `run-development`, `run-evaluation`, `analyze`. `plan` reads only
local configuration and reports unresolved fields; it constructs no model client
and downloads nothing. Executing operations requires a frozen plan hash and
explicit phase budget; no phase auto-starts a more expensive one.

Plan limits include max documents/tokens for fitting, maximum operations and
generated tokens by phase, context length, wall seconds, peak host/device memory,
GPU-hour cap and monetary cap with a recorded rate where paid compute is used.
For local compute with no billing evidence, report monetary cost unknown and
enforce the time/resource limits. There is no inferred zero-dollar cost.

Candidate pilot limits are 4096 total context tokens and 1024 generated tokens
per answer; they need runtime and parsing validation before freezing. The
16-item, two-arm evaluation needs 32 generations. Development, screening,
parity checks and controls have separate declared budgets; 32 is not the total
study allowance. NIM request allowances do not finance this experiment.

Measure warm and cold latency, host/device peak memory and token throughput at
the planned upper bounds. If no suitable hardware is available, produce an
actionable feasibility report without changing the main target. Model downloads
and paid provisioning require their own concrete execution approval.

## Experiment and analysis

Separate screening, development and held-out evaluation. Screening requires
audited historical fact questions showing that the unchanged main model can
recall the targeted facts; market-direction accuracy alone is not that signal.
Freeze screening criteria before screening, and never promote screened examples
into the held-out evaluation set. Lack of a baseline recall signal yields an
inconclusive feasibility report.

Development tests alpha in `{0.25, 0.5, 1.0}` against zero, preferring the smallest
alpha that reduces the declared recall metric while preserving development
utility within five percentage points. If none qualifies, report no qualifying
setting; do not select on the held-out outcomes. This small grid is a design
choice, not a literature-derived optimal setting.

Held-out evaluation uses eight historical and eight post-cutoff items, frozen
parsing and deterministic alternating arm order. Recall questions and financial
direction questions are separate task strata with separate metrics; never pool
their accuracy into one efficacy number. Dataset manifest freezes task identity,
answers, information boundaries and matching evidence before execution.

Report all-planned and common-usable paired results, cohort coverage, lexical
recall metrics, output failures and utility. Compute gap change as
`(adjusted_pre - adjusted_post) - (original_pre - original_post)` only for
matched tasks with validated cohort evidence. Otherwise report it unavailable.
Use 2000 paired-item bootstrap resamples within cohorts, seed 0, retaining both
arms together. Report percentile 95% intervals and flag degeneracy/tiny samples.
The approved criterion is upper gap-change bound below zero and lower utility-
change bound above -0.05. Passing is exploratory evidence on this sample, never
a general claim of memory removal. Report same-period control effects to expose
size/source confounds. MCS and improved confidence calibration are not claimed.

## Validation and traceability

| Requirements | Design evidence / verification |
| --- | --- |
| 1 | Plan identity, mandatory target rejection tests, hosted/local labeling |
| 2 | Manifest checks, date/revision failures, split overlap, corpus/control counts |
| 3 | Tiny vocabulary hand calculations, unseen contexts, cache parity, zero/sham parity, actual changed-token trace |
| 4 | Credential-free planning, admission limits, cancelled worker ledger, memory feasibility receipt |
| 5 | Frozen development selection, paired fixtures with differing failures, cohort bootstrap and unavailable-gap cases |
| 6 | Artifact replay, no secret output, unavailable MCS/cost and condition semantics |
| 7 | Dependency/import checks, focused Docker tests, Ruff, code review for role/shape/unit naming |

Offline tests use tiny deterministic callbacks and count corpora; they do not
download weights. A separately approved real-model smoke verifies backend,
Harmony and hardware before the pilot. Rebuild strict docs when public guides
change. No full native-platform matrix is required.

## Outstanding execution prerequisites

Before a live plan can be approved, select licensed dated sources and audit
their revisions, pin model/tokenizer and dependency revisions, identify actual
hardware, verify memory and parsing feasibility, and price the full phase budget.
These are implementation/preflight deliverables; the design does not claim they
have already passed.
