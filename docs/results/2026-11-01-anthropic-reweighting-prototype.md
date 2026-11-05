# Anthropic completion-reweighting prototype: stopped at sample-support gate

Date: 2026-11-01. Model: claude-opus-5-5 through the existing Anthropic Messages
adapter. Target logprobs were neither requested nor required. Docker ran the
existing client and a small statistical calculation; no models were hosted.

## Decision

Do not build out the sample-reweighting workflow on this evidence. The first
eight completions all chose A, leaving no observed alternative for the correction
to act on. The predeclared support gate stopped the experiment before the second
prompt. This is a fail-early result for this setup and request budget, not proof
that completion-based correction is generally impossible.

## What ran

| Measure | Result |
| --- | --- |
| Physical requests | 9 of maximum 17 |
| Eligibility | Successful |
| Named hypothetical prompt | 8 completed, 8 parsed |
| A / B counts | 8 / 0 |
| Anonymous control | Not run after gate failure |
| A fraction, alpha 0 / 0.5 / 1 | 1.0 / 1.0 / 1.0 |
| Effective sample size at each alpha | 8 |
| Observed probability change | 0 |
| Monetary cost | Unknown; provider token usage retained |
| MCS | Unavailable |
| Memorization mitigation efficacy | Not tested |

High effective sample size does not rescue missing candidate support: all weights
were equal because every observed answer belonged to the same category. An
unobserved answer cannot acquire probability through reweighting observed samples.

The hypothetical prompt explicitly supplied an 80% chance for Brazil and 20%
for France before the 1998 final, instructing the model to ignore the actual
historical result and simulate one outcome. All replies chose Brazil. This does
not demonstrate memorized-outcome bias. A completion instruction to simulate a
random draw also does not establish that the provider samples the requested
categorical distribution.

Even independent ideal 80/20 draws produce eight A answers with probability
0.8^8, about 16.8%. The observed outcome is therefore not evidence of deterministic
provider behavior. A descriptive 95% Wilson interval for the A proportion is
approximately [0.676, 1.000], assuming independent identically distributed draws.
The raw receipt's plug-in standard error of zero is a boundary artifact, not
certainty. Independence and stationary effective generation settings were not
verified.

## What this prototype does and does not establish

Four offline checks verified zero-alpha identity, an analytic two-category
weighting result, unit-weight ESS and inability to create an absent category.
Ruff passed. Live eligibility and response parsing worked without target logprobs.

The auxiliary contrast used synthetic categorical counts: older A80/B20 and newer
A20/B80. These are transparent test inputs, not fitted temporal models or evidence
that real auxiliary sourcing works. The correction computes a self-normalized
weighted distribution over sampled answers. It is not exact token-level
divergence decoding. No alpha was fitted and no held-out mitigation study ran.

## Consequence for the specification

Keep full implementation gated. Before another live pilot, identify an auditable
auxiliary source and a recall-sensitive task with an independently defined utility
control, then justify the sampling budget and a support/precision rule. Do not
increase requests or change prompts until a success appears under the same pilot.
The anonymous control and remaining eight calls are unperformed, not automatically
authorized follow-up work.

The original mathematical decoder remains valid. This prototype tests the proposed
completion-sample route only; it neither validates nor invalidates token-level
divergence decoding.

## Reproducible evidence

- [Frozen plan and prompts](2026-11-01-anthropic-reweighting-prototype/plan.json)
- [Terminal receipt](2026-11-01-anthropic-reweighting-prototype/receipt.json)
- [Physical request ledger](2026-11-01-anthropic-reweighting-prototype/requests.json)
- [Responses](2026-11-01-anthropic-reweighting-prototype/responses.json)

These files are byte-for-byte copies of the local run artifacts. The source
entry point is scripts/probe_completion_reweighting.py.

The motivation is [Jane Street's memorization study](https://blog.janestreet.com/mitigating-memorization-in-llms/).
The [divergence-decoding probability-ratio formulation](https://arxiv.org/html/2605.31293v1#S3)
motivates the contrast; it does not establish equivalence between this categorical
prototype and their token-level method.
