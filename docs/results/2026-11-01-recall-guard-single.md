# Without ensembles: detection is measured; mitigation is not

> **Later evidence (2026-11-08):** this report's calibration corpus separates its classes by publication date and source as well as by memorization, so the AUC reflects corpus separation. On a clean SEC EDGAR corpus, held-out results show no single-draw skill for GPT-OSS-20B or three local models. See [Phase A](2026-11-08-phase-a-single-draw.md) and the [research register](../research-register.md) (R24, R26, R30).


Evidence reviewed: **2026-11-01**. This page records single-answer results.
The [ensemble companion](2026-11-01-recall-guard-ensemble.md) records aggregation
separately. These are different models and corpora, not a cross-model ranking.

## NIM provides a contamination-related signal

The October 30 `openai/gpt-oss-20b` measurement used one answer per evaluation
prompt, without rewriting or divergence decoding.

| Measurement | Observed result | Interpretation |
| --- | --- | --- |
| Held-out calibration ROC-AUC | 0.876 | Separation of supplied exposure-proxy labels |
| Evaluation coverage | 312/330 parsed | 18 retained parse failures |
| Directional accuracy | 170/312, 54.49% | Did not pass the accuracy gate against the majority baseline |
| Mean raw confidence | 0.5992 | Model-reported confidence on parsed answers |
| Mean discounted confidence | 0.2219 | Arithmetic effect of the Recall Guard discount |
| Mean classifier output | 0.6283 | Not an observed contamination rate |

Recall Guard applies `raw_confidence * (1 - p_memorized)` without changing the
predicted direction. Lower confidence therefore does not establish reduced
lookahead or improved confidence calibration. Dates do not establish training
membership; source, formatting and temporal differences can explain label
separation. See the [original report and records](2026-10-30-nim-measurement.md).

## Opus provides a completion-only baseline

On October 31, Claude Opus 5.5's predesignated draw 0 was correct on **11/16**
planned items, with **15/16** usable answers and **11/15** correct usable answers.
On the 14 items usable in both single and ensemble views, it was correct on
**11/14**, the same as the ensemble. Draw 0 belongs to the ensemble's three
draws, so the comparison is paired and dependent.

All 16 proposed rewrites failed the frozen fidelity gate before rewritten
target inference. There are **zero valid mitigation pairs**. Anthropic token
logprobs and MCS were unavailable; self-reported confidence does not replace
them. See the [pilot and replay evidence](2026-10-31-completion-api-pilot.md).

## The November 1 refresh adds access evidence, not efficacy evidence

Across two NIM attempts, three physical requests ran: eligibility returned 504,
explicit follow-up eligibility succeeded, then calibration preflight returned
504. No classifier was fitted and no evaluation item ran. The successful probe
established access and provider-reported token evidence at that moment.

The [refresh report](2026-11-01-nim-refresh.md) retains the outcomes. Neither
this refresh nor the older single-answer results demonstrate that divergence
decoding or synthetic rewriting reduces lookahead. The
[proposed experiment](../divergence-decoding-evaluation.md) defines that test.
