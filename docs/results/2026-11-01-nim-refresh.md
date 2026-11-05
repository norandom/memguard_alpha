# NIM accepted access once; calibration then stopped at a gateway timeout

Measurement date: **2026-11-01**. Target: NVIDIA-hosted
`openai/gpt-oss-20b`. Execution: the locked Recall Guard Docker development
image, using an explicitly approved ceiling of 146 physical requests.

Across two attempts, **three physical requests** ran. Initial eligibility
returned HTTP 504 after 302.24 seconds. Explicitly approved follow-up eligibility
succeeded after 189.54 seconds, with provider-reported token evidence. The fresh
calibration probe then returned HTTP 504 after 302.21 seconds. No classifier
was fitted and no evaluation item ran. There are no new NIM ensemble or
calibration metrics to interpret. Access was accepted at the successful probe;
the latest failed operation does not independently verify credential state.

## The approved follow-up verified access but did not reach fitting

| Physical request | Result | Latency |
| --- | --- | --- |
| First eligibility | HTTP 504 | 302.24s |
| Follow-up eligibility | Accepted; token evidence available | 189.54s |
| Calibration preflight | HTTP 504 | 302.21s |

The successful response contained final text `OK`, 30 provider-reported token
evidence positions and reported usage of 71 prompt plus 30 completion tokens.
No independent token-alignment audit or scoring conclusion follows from that
probe. Effective generation settings were not echoed. Usage for the two failed
requests and total monetary cost remain unknown.

Combined accounting is **3/146 used**, leaving 143 unused: no further eligibility
allowance, 79 calibration requests and 64 evaluation requests. Unused allowance
does not authorize replay. No further automatic requests will be made.

The follow-up artifacts are copied byte-for-byte from the local run:

- [Frozen remaining-budget plan](2026-11-01-nim-refresh/followup-01/plan.json)
- [Final request ledger](2026-11-01-nim-refresh/followup-01/receipt.json)
- [Failed calibration summary](2026-11-01-nim-refresh/followup-01/calibration.json)
- [Eligibility intent](2026-11-01-nim-refresh/followup-01/eligibility/request-001.intent.json)
- [Successful eligibility evidence](2026-11-01-nim-refresh/followup-01/eligibility/request-001.receipt.json)
- [Calibration probe intent](2026-11-01-nim-refresh/followup-01/calibration/request-001.intent.json)
- [Calibration probe failure](2026-11-01-nim-refresh/followup-01/calibration/request-001.receipt.json)

The following sections preserve first-attempt details. Its 145-request remaining
balance was subsequently reduced to 143 by the approved follow-up. Historical
plans retain 1.5s spacing; the updated runner uses 2s for future plans. Neither
change supplies mitigation efficacy evidence. See the
[single](2026-11-01-recall-guard-single.md) and
[ensemble](2026-11-01-recall-guard-ensemble.md) summaries and the
[divergence decoding experiment proposal](../divergence-decoding-evaluation.md).

## First attempt: what actually ran

| Stage | Planned | Observed |
| --- | --- | --- |
| Eligibility | Actual configured inference before calibration | One physical POST; HTTP 504; `unknown_outcome` / `gateway_timeout` |
| Fresh original calibration | 20 in-sample proxies and 20 out-of-sample controls, plus baseline/probes | Not run |
| Original single answers | Draw 0 of each planned three-draw ensemble | Not run |
| Original ensembles | 16 items, three serial draws each | Not run |
| Rewriting / divergence decoding | Explicitly disabled in this baseline refresh | Inactive |

The planned single and ensemble views share draw0, so they would not require
separate single-answer requests. No 0/16 accuracy, MCS estimate or ensemble gain
is reported: these items were unperformed rather than incorrect predictions.
The previous [October 30 NIM measurement](2026-10-30-nim-measurement.md) remains
the latest successful NIM calibration/evaluation. The
[October 31 ensemble results](2026-10-31-completion-api-pilot.md) belong to Opus,
not this NIM model.

## The gateway response ended the operation before the client limit

The frozen settings were `max_tokens=2048`, `temperature=0`, logprobs=true and
`top_logprobs=20`. Transport used a 10s connection limit, 600s socket-idle limit and
1200s controlled-wait allowance per operation. New request admission was bounded
by a 3600s run window. Requests were serial with 1.5s minimum start spacing.

The actual failed exchange lasted 302.2363s; run elapsed time was 302.3262s.
The observation recorded no answer or reasoning progress. A local 1s backoff
recommendation was retained, but no retry or provider status poll was performed.
The server returned 504 before the configured socket-idle limit; the run admission
window was not exhausted. This evidence does not identify the underlying service
fault or prove that backend inference never ran.

The plan allowed zero automatic retries and disabled ambiguous replay. Its
eligibility ceiling was 2 requests, calibration 80 and evaluation 64, total 146.
Only one eligibility request was consumed. The remaining 145 are not a fresh
allowance or permission for automatic replay, another model or paid providers.
Any explicit follow-up must retain this consumed request and preserve the
combined ceiling. The separate eight-call readiness pilot was not used.

## What the updated Docker tests establish

The new measurement harness reuses RecallWorkflow calibration, the existing
ensemble reducer and CallBudget. It strips local labels/cohorts from model
inputs, binds execution to an offline plan, refuses ambiguous output-directory
reuse and stops calibration after failed eligibility. Local projection failures
are retained without silently dropping the other evaluation items.

Fifteen focused Docker tests passed before this live attempt. A later
reduced-budget option passed 18 focused tests and independent review; it permits
a future explicitly selected 145-request remaining ceiling without resetting the
already consumed request. The offline fixture's 112 requests are a test of
accounting, **not 112 live model calls**. Ruff and architecture checks passed.
The existing frozen input-provenance recomputation also passed in Docker.

This is an update to the hosted open-weight model tests. It does not make the
hosted API expose the next-token selection loop. Jane Street's
[memorization article](https://blog.janestreet.com/mitigating-memorization-in-llms/)
discusses divergence decoding and rewritten inputs; neither intervention was
active here, so no mitigation-effect conclusion follows.

## Inspect the saved evidence without credentials

- [Frozen configuration, input hashes and budgets](2026-11-01-nim-refresh/plan.json)
- [Final phase/request receipt](2026-11-01-nim-refresh/receipt.json)
- [Write-ahead eligibility intent](2026-11-01-nim-refresh/eligibility-intent.json)
- [Observed gateway failure and credential state](2026-11-01-nim-refresh/eligibility-receipt.json)
- [Code/image provenance and actual request totals](2026-11-01-nim-refresh/provenance.json)

The raw artifacts are retained as exported, including UTC timestamps and the
provider request identity. Token usage and monetary cost were not reported and
remain unknown. Credentials were supplied only through an explicit read-only
runtime secret mount and are absent from these exports.

The run used measurement commit `d0beae3` and the image identified in provenance,
including preserved uncommitted readiness changes. Passing local tests does not
accept the unfinished readiness feature or turn this failed eligibility check
into model validation. No old reports or results were overwritten.
