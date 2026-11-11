# Completion API pilot: rewrite mitigation and three-draw ensemble with Opus, 2026-10-31

> Register: R02 · Status: failed · Track: Completion

Rejected rewrites leave mitigation unmeasured: this pilot does not demonstrate
reduced memorization. The workflow generated 16 rewrites with Claude Opus 5.5,
but the frozen fidelity gate rejected every candidate before rewritten target
inference, leaving zero valid original versus rewritten answer pairs. Original
evidence produced 11 correct single answers and 12 correct ensemble answers
across 16 items; that small aggregation comparison is useful for inspecting the
workflow, but it does not establish an accuracy gain or a mitigation benefit.

## Question

Does a completion-API presentation rewrite reduce recall while preserving useful
performance, and how does a three-draw ensemble compare with a single answer on
the same items?

## Setup

Software: unreleased Recall Guard development checkout; the completion workflow
described here is not part of v0.5.0.

The completion-only Anthropic run is saved and replayable. The planned NVIDIA
NIM logprob-capable run could not pass its current eligibility check, so fresh
calibration and two-mode measurement remain incomplete. An adopter can inspect
the retained failures and reproduce these metrics offline; this report provides
no evidence for using the rewrite as an effective mitigation.

Intervention: the method is `neutral-conversation-v1`, with one candidate rewrite
per item, shared by its planned target draws. The deterministic
`factual-tokens-v1` checker requires exact preservation of extracted token
multisets. The intervention adapts the separate podcast-rewriting experiment in
James Somers' [Mitigating memorization in LLMs](https://blog.janestreet.com/mitigating-memorization-in-llms/)
(Jane Street, September 30, 2026). That experiment also removed player quotes.
Recall Guard's rewrite must preserve quotations and factual qualifications.
This is an adaptation, not a replication. It also differs from the post's
logit-divergence method: presentation rewriting does not apply the next-token
logit equation, and no divergence decoding ran in this pilot.

Live coverage and resources remain distinct from implementation. The development
checkout implements native Anthropic Messages, OpenAI-compatible chat and caller
callback paths, with offline fixture checks. Those checks do not establish live
support for every provider or endpoint.

| Configuration | Live evidence in this study | Scoring and intervention limits |
| --- | --- | --- |
| Anthropic `claude-opus-5-5` | Target generation, rewrite generation and three role access probes | No genuine token logprobs or MCS; no item-level semantic verification or accepted rewritten target answers |
| NVIDIA NIM `openai/gpt-oss-20b` | Three failed eligibility requests | Current logprob eligibility unproven; no fresh calibration or evaluation |
| OpenAI service, Mistral service, caller callbacks | Not measured live | Offline compatibility is not live-provider validation |

The Opus configuration uses native API version `2026-10-21` at
`https://api.anthropic.com/v1/messages`, requesting `max_tokens=2048` only,
with timeout 90 seconds, zero retries and request starts spaced by at least
1.5 seconds. Temperature and seed were not sent. Accepted parameters and
unechoed effective values are distinct; effective controls and omitted
defaults remain unknown. Target arm order alternates by frozen item index;
each arm plans three draws with the same settings and shared rewrite.

| Opus role, including its access probe | Physical requests | Reported input tokens | Reported output tokens | Sum of request latency, seconds |
| --- | ---: | ---: | ---: | ---: |
| Target | 49 | 15,665 | 6,098 | 136.939 |
| Rewriter | 17 | 8,817 | 3,654 | 64.471 |
| Verifier | 1 | 17 | 4 | 1.328 |
| Total | 67 | 24,499 | 9,756 | 202.738 |

All 67 physical Opus requests succeeded: 48 original target draws, 16 rewrite
generations and three role probes. The single and ensemble views overlap, so
their request totals must not be added. Latency sums measure request work,
not elapsed wall time. Monetary cost is unknown; no billing receipt or price
table was used. No Anthropic MCS fit, confidence discount or memorization
probability is available, and no fresh NIM fit was completed.

NIM first timed out after 90.453 seconds, then after 90.297 seconds. Following
the operator's latency clarification, a documented post-observation transport
amendment changed only its timeout from 90 to 600 seconds. The final request
returned HTTP 504 at 302.453 seconds, before that client timeout. Its endpoint
remained `https://integrate.api.nvidia.com/v1/chat/completions`, with the same
model, `max_tokens=2048`, `temperature=0`, `logprobs=true`,
`top_logprobs=20`, zero retries and pacing of 1.5 seconds. These observations
do not identify invalid credentials, insufficient balance, an unsupported model
or the underlying service fault.

## Data

The evaluation contains eight pre-cutoff and eight post-cutoff items, with
eight UP and eight DOWN outcomes overall. Each cohort/ticker/direction stratum
contains two items. A constant UP or DOWN answer therefore scores 50% on all
16. The task predicts the next session's quoted-close direction from five
completed daily closes for SPY or IAU. Instructions, output format and the
as-of boundary stay outside rewriteable evidence. The outcome is retained for
evaluation and excluded from model input.

The source is a retrospective Yahoo Finance chart snapshot. Selection used a
frozen SHA256 rank, outcome-balanced strata and disjoint six-session windows
for each cohort/ticker. Prices and labels use quoted closes rounded half up to
USD 0.01. These are not total returns. The preparation excludes recorded
corporate-action windows, invalid bars and ties, but source revisions, split
adjustment and incomplete action metadata remain limitations. Outcome balancing
and deterministic selection do not make this a representative forecasting
sample or a contemporaneous archive.

The frozen registry uses 2026-06-30 for `claude-opus-5-5`, based on the June
2026 cutoff in [Anthropic's model overview](https://platform.claude.com/docs/en/models/opus-5-5/overview),
and 2024-06-30 for `openai/gpt-oss-20b`, based on June 2024 in section 2.4 of
the [OpenAI model card](https://arxiv.org/html/2508.10925v1). Month-only dates
use the month's final day. The saved provenance records the exact model
identities, source URLs and review of later post-training information. Neither
cutoff establishes training membership or guarantees exclusion from later
tuning, retrieval or serving systems.

## Results

Original answers support a limited aggregation comparison. Each condition
contains the same 16 planned item IDs. The single response is predesignated draw
0 of the three-draw ensemble, so the two conditions share responses and are
dependent. All-attempted accuracy treats an unusable answer or failed
intervention as an unsuccessful task outcome.

| Condition | Correct / all planned items | Usable answers | Correct / usable answers |
| --- | --- | --- | --- |
| Original single | 11/16 (68.75%) | 15/16 | 11/15 |
| Original ensemble | 12/16 (75.00%) | 15/16 | 12/15 |
| Rewritten single | 0/16 successful task outcomes | 0/16 | Unavailable |
| Rewritten ensemble | 0/16 successful task outcomes | 0/16 | Unavailable |

The two original conditions fail on different items. Draw 0 was unusable for
`price-e61647df3c766eb4bd36`; the ensemble was unusable for
`price-f9a0273a435521825a03`, where only one of three draws parsed and the
minimum usable count was two. Comparing 11/15 with 12/15 as though they were
the same 15 items would hide this difference. Across the 14 items usable in
both conditions, both conditions are correct on 11/14.

| Paired comparison, ensemble minus single | Eligible items | Difference | 95% bootstrap interval |
| --- | ---: | ---: | --- |
| Original, all attempted | 16 | +6.25 percentage points | [0.00, 18.75] percentage points |
| Original, common usable pairs | 14 | 0.00 percentage points | [0.00, 0.00], degenerate |
| Original, self-reported confidence | 14 | +0.01047619 | [-0.00309524, 0.02785714] probability units |
| Rewriting, single or ensemble, valid pairs | 0 | Unavailable | Unavailable |

These are percentile paired-item bootstrap intervals: 2,000 resamples, seed 0,
level 0.95, with each item's arms and draws kept together. Resampling occurs
within each eligible comparison without cohort stratification. Draws are not
independent resampling units. Confidence is the model's stated probability of
its chosen direction. For an ensemble, these report metrics use the mean of the
explicit confidence components available in its draws. This statistic is
separate from core consensus confidence and MCS; it does not establish
calibrated uncertainty.

The saved all-attempted rewriting differences are -68.75 percentage points for
single responses (95% interval [-87.50, -43.75]) and -75.00 percentage points
for ensembles ([-93.75, -56.09]). They count rejected interventions as failed
task outcomes. They do not measure model accuracy after accepted rewrites or
the effect of rewriting on recall. The frozen utility tolerance is 0.05
probability units, or five percentage points. With no accepted rewritten
answers, this run cannot establish preservation of useful performance within
that tolerance.

Of 48 physical original target draws, 45 parsed successfully. All 48 succeeded
at transport. Three retained projection failures contain explanatory prose
around a JSON object; this run's strict JSON projection rejected them. The report
preserves those decisions rather than reparsing favorable answers after
observation.

The fidelity audit explains rejection without overriding it. The checker
rejected 16/16 candidates. Semantic verification ran on 0/16 items, and rewritten
target generation made zero requests. The verifier's live coverage is an access
probe only. All 16 candidate hashes differ from their source hashes, but that
does not establish a usable presentation intervention.

The frozen audit selected a census of all 16 planned items before generation.
An AI agent inspected every source/candidate pair and identified no material
factual discrepancy in 14. Two pairs need human judgment about possible changes
to attribution or qualification: wording that attributed a revision caveat to
the source, and wording that narrowed a general revision caveat to possible
past revisions. Repeated as-of dates, equivalent date spelling and capitalized
prose words contributed to disagreement with the token gate.

Budget: eligibility consumed all four requests (three NIM failures and one
Anthropic success). Calibration's 220-request cap and NIM evaluation's
144-request cap remain unused. Opus evaluation used 67/144. The source plan's
overall cap is 512, with no automatic transfer of unused phase budgets.

## Limits

- The degenerate interval describes this tiny observed sample; it is not
  evidence of equivalence, adequate power or stable performance on new items.
- The fidelity audit is AI inspection, not independent human proof of
  equivalence or absence of future knowledge. It does not override the frozen
  gate. Every candidate remains rejected, and apparent loss of task accuracy from
  these failures cannot support a mitigation-success conclusion. A future
  experiment would need a separately declared fidelity policy and adequate
  accepted pairs before measuring efficacy.
- Although the CLI input checks allow these sourced cohorts, the saved analyzer
  explicitly marks historical inference unavailable because it does not validate
  historical matching and cutoff evidence. This report does not manufacture a
  historical gap or difference-of-gaps estimate. Temporal accuracy and confidence
  differences would remain recall-related proxies even with validated cohorts.
- Tasks 6.2, 6.3 and the full two-mode task 6.4 remain incomplete, including
  requirement 9.7. The earlier [October 30 NIM measurement](2026-10-30-nim-measurement.md)
  and the separate 49-call, all-UP early Opus pilot are different experiments and
  do not fill these gaps.

## Evidence

The [saved metrics](2026-10-31-completion-api/anthropic-evaluation-live/metrics.json)
retain condition, cohort and paired-effect denominators. The
[audit](2026-10-31-completion-api/anthropic-evaluation-live/evidence/fidelity-audit.json)
retains the texts, extracted checks, judgments and disagreements.

The [latest NIM receipt](2026-10-31-completion-api/eligibility/latency-01/receipt.json)
is a valid failed receipt, not an ambiguous attempt. Its SHA256 is
`18c29de4ec1da3b99b4bc7489b21329b3da7a9198c4534a6f0e0206bedfa17f9`.

The following ten files are byte-identical copies of the saved live bundle.
The manifest's completed status describes terminal execution records; it does
not mean successful mitigation or complete provider coverage. Provider origin
is caller-attested. File hashes establish integrity, not provider authentication.

- [Manifest](2026-10-31-completion-api/anthropic-evaluation-live/manifest.json)
- [Records, including inputs, answers, draws and request events](2026-10-31-completion-api/anthropic-evaluation-live/records.json)
- [Recomputed metrics](2026-10-31-completion-api/anthropic-evaluation-live/metrics.json)
- [Frozen Anthropic protocol](2026-10-31-completion-api/anthropic-evaluation-live/evidence/anthropic-protocol.json)
- [Study plan with input hashes and model provenance](2026-10-31-completion-api/anthropic-evaluation-live/evidence/study-plan.json)
- [Cutoff registry](2026-10-31-completion-api/anthropic-evaluation-live/evidence/cutoff-registry.json)
- [Eligibility receipt used for the Opus run](2026-10-31-completion-api/anthropic-evaluation-live/evidence/eligibility-receipt.json)
- [Fidelity audit](2026-10-31-completion-api/anthropic-evaluation-live/evidence/fidelity-audit.json)
- [Reconciliation](2026-10-31-completion-api/anthropic-evaluation-live/evidence/reconciliation.json)
- [Operator run attestation and source revision](2026-10-31-completion-api/anthropic-evaluation-live/evidence/operator-run-attestation.json)

The separate [NIM timeout amendment](2026-10-31-completion-api/eligibility/latency-amendment.json)
and [latest receipt](2026-10-31-completion-api/eligibility/latency-01/receipt.json)
preserve the later eligibility outcome. Their source-bound verification needs
the original repository input chain, not just those two downloads.

From this development checkout with dependencies installed, this command works
in PowerShell and bash and makes no provider calls:

```text
uv run --no-sync python scripts/run_intervention.py --replay docs/results/2026-10-31-completion-api/anthropic-evaluation-live
```

It verifies the saved record and metric byte hashes, loads the retained inputs
and recomputes the saved analysis. The bundle can move to another directory;
pass that directory to `--replay`. Replay uses saved responses and settings,
without resolving provider credentials or refitting a classifier. A fresh
moved-copy check with socket networking blocked and inherited provider keys
removed reproduced the metrics exactly.

To verify the failed NIM receipt against its full repository input chain:

```text
uv run --no-sync python -m scripts.prepare_completion_latency verify-receipt --directory data/experiments/2026-10-31-completion-api
```

The expected result is `status=failed`, budget used 4 and remaining 0, with
exit code 1. That exit records failed live eligibility after successful receipt
validation; it is not a passed provider test. Neither command schedules a new
eligibility request.
