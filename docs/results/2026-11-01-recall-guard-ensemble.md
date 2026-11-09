# Lookahead evidence with ensembles, 2026-11-01

> Register: R07 · Status: measured · Track: Summary

With ensembles, aggregation is measured; reduced lookahead is not. An ensemble
combines repeated answers; it does not itself remove memorized knowledge. On the
Opus pilot, the three-draw ensemble's all-planned difference over the single
answer is +6.25 percentage points with a 95% interval of [0.00, 18.75], and zero
on common usable items; this small dependent sample does not demonstrate an
aggregation benefit.

## Question

What do the existing runs show when repeated answers are aggregated, as opposed
to single answers? Compare the
[single-answer companion](2026-11-01-recall-guard-single.md).

## Setup

Evidence reviewed: 2026-11-01. The October 31 Claude Opus 5.5 pilot planned
three serial draws per item. The single condition reuses predesignated draw 0,
without extra requests.

## Data

The 16 planned items of the October 31 Opus pilot, and the November 1
`openai/gpt-oss-20b` NIM refresh.

## Results

The Opus pilot has a paired three-draw comparison:

| Original-input view | Correct / planned | Usable / planned | Correct / usable |
| --- | --- | --- | --- |
| Single | 11/16 (68.75%) | 15/16 | 11/15 |
| Three-draw ensemble | 12/16 (75.00%) | 15/16 | 12/15 |
| Single on common usable items | 11/14 | 14/14 | 11/14 |
| Ensemble on common usable items | 11/14 | 14/14 | 11/14 |

- The unusable item differs between views. The all-planned difference is
  +6.25 percentage points, with a paired-item bootstrap 95% interval of
  [0.00, 18.75] percentage points. The common-usable difference is zero.
- Of 48 original target draws, all succeeded at transport and 45 parsed.
- All 16 rewrites were rejected before rewritten target inference. Consequently
  there are no valid single or ensemble mitigation comparisons, and no Anthropic
  MCS values.

NIM ensemble evidence is still missing. The November 1 `openai/gpt-oss-20b`
refresh stopped at calibration preflight after three physical requests across
two attempts. No evaluation draws ran and no ensemble metrics exist. October
30's NIM results were single-answer results; they cannot supply an ensemble
comparison.

## Limits

- The degenerate interval for those 14 pairs does not establish equivalence.
  This small dependent sample does not demonstrate an aggregation benefit.
- Agreement and stated confidence are not memorization estimates.
- Do not add the overlapping single and ensemble request counts.

Measure mitigation separately from aggregation: the next experiment crosses
original versus mitigated generation with single versus three-draw reduction.
Each generation condition shares its draw 0 with its ensemble. This separates an
intervention's effect from aggregation's effect. See the
[divergence decoding evaluation proposal](../divergence-decoding-evaluation.md).

## Evidence

The [original pilot](2026-10-31-completion-api-pilot.md) includes saved metrics,
failure identities, fidelity audit and bootstrap details. NIM refresh:
[report](2026-11-01-nim-refresh.md).
