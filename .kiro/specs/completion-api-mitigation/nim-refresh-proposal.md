# NIM calibration and ensemble refresh — 2026-11-01

Prepared for the user's request to update NIM open-weight model tests using
Docker. This is a separate dated baseline refresh; it does not overwrite the
October30/31 results or reuse an exhausted eligibility allowance.

## Concrete first run

Use NVIDIA-hosted `openai/gpt-oss-20b`, the existing configured model. Its
[NVIDIA model card](https://build.nvidia.com/openai/gpt-oss-20b/modelcard) and
[API reference](https://docs.api.nvidia.com/nim/reference/openai-gpt-oss-20b)
were checked on2026-11-01. A public listing does not establish account eligibility
or logprob support; actual inference must establish those first.

Keep generation settings fixed: max_tokens2048, temperature0, logprobs=true,
top_logprobs20. No automatic model switch, parameter substitution or text-only
fallback. All calls use this NIM model; no paid auxiliary provider or local model
hosting is introduced. Exact divergence decoding and rewriting are inactive in
this baseline refresh, explicitly recorded in the report.

Reuse and freshly validate the frozen October31 inputs:20 in-sample proxy
calibration items,20 out-of-sample controls and16 evaluation items. Labels and
cohort names stay out of model-facing prompts through the existing label-free
input conversion. Input hashes are captured in the executable offline plan.

1. Probe the configured API and genuine token evidence before fitting. Missing
   credentials, explicit auth/access rejection or unsupported capabilities stop
   calibration and evaluation. Timeout/504 leaves eligibility unverified.
2. Fit a fresh original-response calibrator through the existing workflow,
   with min_valid10, min_auc0.6, seed0 and one worker. Retain weak/failed fit
   evidence; do not silently reuse October30's fit or change to an unscored run.
3. Run the16 items with three serial draws each and the existing ensemble reducer,
   requiring at least two parsed draws. Draw0 is the predesignated single-answer
   condition, so no extra single requests are added. Retain failures and report
   all-planned and common-usable denominators separately.

## Proposed hard request caps

| Phase | Nominal physical requests | Maximum |
| --- | ---: | ---: |
| Eligibility | 1 | 2 |
| Fresh original calibration, including its actual probes | 62 | 80 |
| Sixteen three-draw ensembles, including fresh phase probe | 49 | 64 |
| Total | 112 | 146 |

Every actual probe/generation counts through the existing CallBudget; there is
no second consumption ledger. No automatic retries or status polls. The extra
eligibility capacity does not authorize automatic replay of ambiguous inference.
Nonempty output directories and unresolved prior intents prevent restart.

Transport retains 1.5s request-start spacing and uses the existing bounded ReadinessPolicy: connect10s, socket-idle600s,
controlled-wait1200s per operation, zero polls and no ambiguous replay. New calls
stop after a3600s admission window. An already admitted synchronous request may
finish after that window within the transport's cooperative bounds; record actual
elapsed time and any partial/unperformed phase honestly. No local heartbeat is
reported as provider progress or evidence of a valid key.

## Outputs and interpretation

The driver is `scripts/measure_nim_refresh.py`; `--plan` is offline and requires
no credentials. Explicit `--run` requires a matching `--plan-file` and `--output` before credential lookup, and writes a new directory under
`runs/container/2026-11-01-nim-refresh`, with plan/input hashes, write-ahead intent,
actual physical events, fit quality/fingerprint and item/draw outcomes. Use the
existing per-call evidence and reducer; no generic runner or resume system.

The Docker run mounts the existing .env read-only and resolves only the selected
NVIDIA_API_KEY through python-dotenv, so quoted dotenv values do not become
literal quotes in an Authorization header. The image and exported artifacts
contain no credentials. Offline planning never reads that file.

This measures current API availability, calibration quality and baseline
single-versus-ensemble behavior on a small reused corpus. It does not demonstrate
a Jane Street mitigation effect or provide a precise causal contamination rate.
Fits are used for this run; do not claim a reusable fitted model export unless
one is actually produced by the existing API. Publish a dated report and link
from README only after results, including failures, are available.

## Authorization boundary

Offline preparation/testing is authorized by the user's current request. Before
the larger live run, present this concrete146-request ceiling for selection.
The separately approved eight-call fixed/adaptive readiness pilot has a different
purpose and cannot silently fund this calibration/ensemble measurement. Old
budgets/results remain immutable; no allowance is replenished by this proposal.

## Offline verification

Existing input provenance recomputation passed in Docker: 16 evaluation, 20
in-sample and 20 out-of-sample items. Fifteen focused Docker tests passed in
4.73s, including a full fake-transport run through real stock admission and
scoring (112 physical requests), draw-zero reuse, one failed item with the other
15 retained, key rejection, missing logprobs, quoted credential parsing, frozen
plan mismatch and admission-window enforcement. Ruff and architecture passed.
Independent review APPROVED after correcting pacing across the deadline with
the existing CancellationToken guard. No core changes or live calls were made.

## Live authorization

User approved the 146-request ceiling on 2026-11-01T15:22:42.6349154+00:00. Scope: NIM openai/gpt-oss-20b eligibility, fresh original calibration and single/three-draw comparison; no paid auxiliary providers, no transfer from the readiness allowance.

## First attempt and optional explicit follow-up

The approved first attempt consumed one physical eligibility request and returned HTTP 504 after302.24s. Credential eligibility remains unverified; no calibration/evaluation calls ran. Raw evidence is published in docs/results/2026-11-01-nim-refresh/. Automatic replay remains disabled.

A separately selected new attempt may use retry-plan.json with eligibility cap1, calibration80, evaluation64: maximum145 remaining, plus the prior consumed1 equals the original146 ceiling. No call allowance is reset. The helper's lower-cap option passed18 Docker tests and independent review. One additional eligibility operation would require explicit user replay permission because the first provider outcome is unknown; no second request has been sent.
