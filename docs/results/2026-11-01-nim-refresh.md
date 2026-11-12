# NIM refresh of GPT-OSS-20B calibration and ensembles, 2026-11-01

> Register: R05 · Status: failed · Track: NIM

NIM accepted access once; calibration then stopped at a gateway timeout. Across
two attempts, three physical requests ran: initial eligibility returned HTTP 504
after 302.24 seconds, explicitly approved follow-up eligibility succeeded after
189.54 seconds with provider-reported token evidence, and the fresh calibration
probe then returned HTTP 504 after 302.21 seconds. No classifier was fitted and
no evaluation item ran, so there are no new NIM ensemble or calibration metrics
to interpret.

## Question

Can a fresh calibration and single-versus-ensemble evaluation of NVIDIA-hosted
`openai/gpt-oss-20b` run within an approved request ceiling?

## Setup

Target: NVIDIA-hosted `openai/gpt-oss-20b`. Execution: the locked Recall Guard
Docker development image, using an explicitly approved ceiling of 146 physical
requests.

The frozen settings were `max_tokens=2048`, `temperature=0`, logprobs=true and
`top_logprobs=20`. Transport used a 10s connection limit, 600s socket-idle limit and
1200s controlled-wait allowance per operation. New request admission was bounded
by a 3600s run window. Requests were serial with 1.5s minimum start spacing.

The plan allowed zero automatic retries and disabled ambiguous replay. Its
eligibility ceiling was 2 requests, calibration 80 and evaluation 64, total 146.
Historical plans retain 1.5s spacing; the updated runner uses 2s for future plans.

What the updated Docker tests establish: the new measurement harness reuses
RecallWorkflow calibration, the existing ensemble reducer and CallBudget. It
strips local labels/cohorts from model inputs, binds execution to an offline
plan, refuses ambiguous output-directory reuse and stops calibration after
failed eligibility. Local projection failures are retained without silently
dropping the other evaluation items.

Fifteen focused Docker tests passed before this live attempt. A later
reduced-budget option passed 18 focused tests and independent review; it permits
a future explicitly selected 145-request remaining ceiling without resetting the
already consumed request. The offline fixture's 112 requests are a test of
accounting, not 112 live model calls. Ruff and architecture checks passed.
The existing frozen input-provenance recomputation also passed in Docker.

The run used measurement commit `d0beae3` and the image identified in provenance,
including preserved uncommitted readiness changes.

## Data

| Stage | Planned | Observed |
| --- | --- | --- |
| Eligibility | Actual configured inference before calibration | One physical POST; HTTP 504; `unknown_outcome` / `gateway_timeout` |
| Fresh original calibration | 20 in-sample proxies and 20 out-of-sample controls, plus baseline/probes | Not run |
| Original single answers | Draw 0 of each planned three-draw ensemble | Not run |
| Original ensembles | 16 items, three serial draws each | Not run |
| Rewriting / divergence decoding | Explicitly disabled in this baseline refresh | Inactive |

This table records the first attempt. The planned single and ensemble views
share draw0, so they would not require separate single-answer requests.

## Results

The approved follow-up verified access but did not reach fitting:

| Physical request | Result | Latency |
| --- | --- | --- |
| First eligibility | HTTP 504 | 302.24s |
| Follow-up eligibility | Accepted; token evidence available | 189.54s |
| Calibration preflight | HTTP 504 | 302.21s |

The successful response contained final text `OK`, 30 provider-reported token
evidence positions and reported usage of 71 prompt plus 30 completion tokens.
Access was accepted at the successful probe; the latest failed operation does
not independently verify credential state.

Combined accounting is 3/146 used, leaving 143 unused: no further eligibility
allowance, 79 calibration requests and 64 evaluation requests. Unused allowance
does not authorize replay. No further automatic requests will be made.

First attempt: the gateway response ended the operation before the client limit.
The actual failed exchange lasted 302.2363s; run elapsed time was 302.3262s.
The observation recorded no answer or reasoning progress. A local 1s backoff
recommendation was retained, but no retry or provider status poll was performed.
The server returned 504 before the configured socket-idle limit; the run admission
window was not exhausted. Only one eligibility request was consumed in that
attempt. Its 145-request remaining balance was subsequently reduced to 143 by the
approved follow-up.

No 0/16 accuracy, MCS estimate or ensemble gain is reported: these items were
unperformed rather than incorrect predictions. The previous
[October 30 NIM measurement](2026-10-30-nim-measurement.md) remains the latest
successful NIM calibration/evaluation. The
[October 31 ensemble results](2026-10-31-completion-api-pilot.md) belong to Opus,
not this NIM model.

## Limits

- No independent token-alignment audit or scoring conclusion follows from the
  successful probe. Effective generation settings were not echoed. Usage for the
  two failed requests and total monetary cost remain unknown.
- The first-attempt evidence does not identify the underlying service fault or
  prove that backend inference never ran.
- The remaining 145 after the first attempt were not a fresh allowance or
  permission for automatic replay, another model or paid providers. Any explicit
  follow-up must retain the consumed request and preserve the combined ceiling.
  The separate eight-call readiness pilot was not used.
- This is an update to the hosted open-weight model tests. It does not make the
  hosted API expose the next-token selection loop. Jane Street's
  [memorization article](https://blog.janestreet.com/mitigating-memorization-in-llms/)
  discusses divergence decoding and rewritten inputs; neither intervention was
  active here, so no mitigation-effect conclusion follows. Neither the spacing
  change nor the follow-up supplies mitigation efficacy evidence. See the
  [single](2026-11-01-recall-guard-single.md) and
  [ensemble](2026-11-01-recall-guard-ensemble.md) summaries and the
  [divergence decoding experiment proposal](../research/divergence-decoding-evaluation.md).
- Passing local tests does not accept the unfinished readiness feature or turn
  this failed eligibility check into model validation.

## Evidence

The follow-up artifacts are copied byte-for-byte from the local run:

- [Frozen remaining-budget plan](2026-11-01-nim-refresh/followup-01/plan.json)
- [Final request ledger](2026-11-01-nim-refresh/followup-01/receipt.json)
- [Failed calibration summary](2026-11-01-nim-refresh/followup-01/calibration.json)
- [Eligibility intent](2026-11-01-nim-refresh/followup-01/eligibility/request-001.intent.json)
- [Successful eligibility evidence](2026-11-01-nim-refresh/followup-01/eligibility/request-001.receipt.json)
- [Calibration probe intent](2026-11-01-nim-refresh/followup-01/calibration/request-001.intent.json)
- [Calibration probe failure](2026-11-01-nim-refresh/followup-01/calibration/request-001.receipt.json)

First-attempt saved evidence, inspectable without credentials:

- [Frozen configuration, input hashes and budgets](2026-11-01-nim-refresh/plan.json)
- [Final phase/request receipt](2026-11-01-nim-refresh/receipt.json)
- [Write-ahead eligibility intent](2026-11-01-nim-refresh/eligibility-intent.json)
- [Observed gateway failure and credential state](2026-11-01-nim-refresh/eligibility-receipt.json)
- [Code/image provenance and actual request totals](2026-11-01-nim-refresh/provenance.json)

The raw artifacts are retained as exported, including UTC timestamps and the
provider request identity. Token usage and monetary cost were not reported and
remain unknown. Credentials were supplied only through an explicit read-only
runtime secret mount and are absent from these exports. No old reports or
results were overwritten.
