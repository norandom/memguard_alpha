Sources: macro rows from FRED (Federal Reserve Bank of St. Louis) as known on each decision date, including BAA10Y, Moody's Seasoned Baa Corporate Bond Yield Relative to Yield on 10-Year Treasury Constant Maturity, copyright Moody's, cited and not reproduced; realized next-month returns from Financial Modeling Prep (FMP) end-of-day adjusted closes (licensed, not published) with DTB3 from FRED as the cash leg; EDGAR outlook prompts from SEC EDGAR filings. Replayed offline from the recorded draws and scores by scripts/finance_sim/replay.py (task 8.4) with no model request; the JSON document with the per-item rows stays private under runs/finance-sim/analysis/. Derived statistics only: no reply text, no realized return and no raw licensed series.

# Pre-registered analysis: accuracy, losses and multiple testing

Arm sampling (probe arm), stage full_run, model google/gemma-3-4b-it, plan f7603eeb8efdbf74645d658f2bd733a35b3834ba7c90b74d00c6ac2e6b5332fe.

This arm supplies the top_k versus full_distribution comparison and the convergence analysis (7.12) and a small paired check of the gap, with all six variants on its frozen probe evaluation months; the pilot and power gate (7.8) gates this arm only.

Planned months: 20 (15 on or before the model's cutoff, 5 after); pilot items left out: 0; planned results missing: 0.

Intervals: 95 percent non-circular moving-block percentile bootstrap over months, block length 6 months, 2000 resamples, seed 20261110; tested metrics carry two-sided p-values, Holm-adjusted across their variants (alpha 0.05); pre-registered effect 0.1.

Derived statistics only: no reply text, no realized returns and no raw licensed series.

## estimate_difference_pre_cutoff

Requirement 6.3. Identifying minus anonymised p_memorized, mean over the rows of the period; rows: the arm's planned evaluation months whose identifying and anonymised renderings both carry the value, pilot items excluded; interval: non-circular moving-block percentile bootstrap over the rows in plan order (recall_guard.core.significance.block_bootstrap) with the plan's block_length, n_resamples, seed and confidence 1 - alpha; a resample whose needed period has no month is dropped and counted; a measurement (Requirement 6.3), not a test, so no p-value.

| Variant | Statistic | Interval | Months (pre, post) | Dropped resamples | Missing |
|---|---|---|---|---|---|
| single_none | missing | missing | 0 (0, 0) |  | not applicable: the no-adjustment variant carries no memorization estimate (its draws pass without estimates by construction, status estimate_unavailable), so it has no estimate difference |
| single_top_k | 0.221634 | [0.118157, 0.275174] | 9 (4, 5) | 0 | 11 planned months without the value in both renderings (reasons in the JSON) |
| single_full_distribution | 0.733721 | [0.517274, 0.764642] | 9 (4, 5) | 0 | 11 planned months without the value in both renderings (reasons in the JSON) |
| ensemble_none | missing | missing | 0 (0, 0) |  | not applicable: the no-adjustment variant carries no memorization estimate (its draws pass without estimates by construction, status estimate_unavailable), so it has no estimate difference |
| ensemble_top_k | 0.068397 | [0.053202, 0.0978145] | 9 (4, 5) | 0 | 11 planned months without the value in both renderings (reasons in the JSON) |
| ensemble_full_distribution | 0.624133 | [0.494879, 0.638805] | 9 (4, 5) | 0 | 11 planned months without the value in both renderings (reasons in the JSON) |

## estimate_difference_post_cutoff

Requirement 6.3. Identifying minus anonymised p_memorized, mean over the rows of the period; rows: the arm's planned evaluation months whose identifying and anonymised renderings both carry the value, pilot items excluded; interval: non-circular moving-block percentile bootstrap over the rows in plan order (recall_guard.core.significance.block_bootstrap) with the plan's block_length, n_resamples, seed and confidence 1 - alpha; a resample whose needed period has no month is dropped and counted; a measurement (Requirement 6.3), not a test, so no p-value.

| Variant | Statistic | Interval | Months (pre, post) | Dropped resamples | Missing |
|---|---|---|---|---|---|
| single_none | missing | missing | 0 (0, 0) |  | not applicable: the no-adjustment variant carries no memorization estimate (its draws pass without estimates by construction, status estimate_unavailable), so it has no estimate difference |
| single_top_k | 0.0525273 | [-0.0484964, 0.0525273] | 9 (4, 5) | 0 | 11 planned months without the value in both renderings (reasons in the JSON) |
| single_full_distribution | 0.618387 | [0.508662, 0.782538] | 9 (4, 5) | 0 | 11 planned months without the value in both renderings (reasons in the JSON) |
| ensemble_none | missing | missing | 0 (0, 0) |  | not applicable: the no-adjustment variant carries no memorization estimate (its draws pass without estimates by construction, status estimate_unavailable), so it has no estimate difference |
| ensemble_top_k | 0.0935117 | [0.0761158, 0.105323] | 9 (4, 5) | 0 | 11 planned months without the value in both renderings (reasons in the JSON) |
| ensemble_full_distribution | 0.55441 | [0.55441, 0.768922] | 9 (4, 5) | 0 | 11 planned months without the value in both renderings (reasons in the JSON) |

## accuracy_difference_pre_cutoff

Requirement 6.3. Identifying minus anonymised correctness (1 correct, 0 not), mean over the rows of the period; rows: the arm's planned evaluation months whose identifying and anonymised renderings both carry the value, pilot items excluded; interval: non-circular moving-block percentile bootstrap over the rows in plan order (recall_guard.core.significance.block_bootstrap) with the plan's block_length, n_resamples, seed and confidence 1 - alpha; a resample whose needed period has no month is dropped and counted; a measurement (Requirement 6.3), not a test, so no p-value.

| Variant | Statistic | Interval | Months (pre, post) | Dropped resamples | Missing |
|---|---|---|---|---|---|
| single_none | 0 | [0, 0] | 9 (4, 5) | 0 | 11 planned months without the value in both renderings (reasons in the JSON) |
| single_top_k | 0 | [0, 0] | 9 (4, 5) | 0 | 11 planned months without the value in both renderings (reasons in the JSON) |
| single_full_distribution | 0 | [0, 0] | 9 (4, 5) | 0 | 11 planned months without the value in both renderings (reasons in the JSON) |
| ensemble_none | missing | missing | 5 (4, 1) |  | 5 months carry the value in both renderings, fewer than the frozen block length of 6 months, so the moving-block bootstrap is not defined |
| ensemble_top_k | missing | missing | 5 (4, 1) |  | 5 months carry the value in both renderings, fewer than the frozen block length of 6 months, so the moving-block bootstrap is not defined |
| ensemble_full_distribution | missing | missing | 5 (4, 1) |  | 5 months carry the value in both renderings, fewer than the frozen block length of 6 months, so the moving-block bootstrap is not defined |

## accuracy_difference_post_cutoff

Requirement 6.3. Identifying minus anonymised correctness (1 correct, 0 not), mean over the rows of the period; rows: the arm's planned evaluation months whose identifying and anonymised renderings both carry the value, pilot items excluded; interval: non-circular moving-block percentile bootstrap over the rows in plan order (recall_guard.core.significance.block_bootstrap) with the plan's block_length, n_resamples, seed and confidence 1 - alpha; a resample whose needed period has no month is dropped and counted; a measurement (Requirement 6.3), not a test, so no p-value.

| Variant | Statistic | Interval | Months (pre, post) | Dropped resamples | Missing |
|---|---|---|---|---|---|
| single_none | 0 | [-0.333333, 0] | 9 (4, 5) | 0 | 11 planned months without the value in both renderings (reasons in the JSON) |
| single_top_k | 0 | [-0.333333, 0] | 9 (4, 5) | 0 | 11 planned months without the value in both renderings (reasons in the JSON) |
| single_full_distribution | 0 | [-0.333333, 0] | 9 (4, 5) | 0 | 11 planned months without the value in both renderings (reasons in the JSON) |
| ensemble_none | missing | missing | 5 (4, 1) |  | 5 months carry the value in both renderings, fewer than the frozen block length of 6 months, so the moving-block bootstrap is not defined |
| ensemble_top_k | missing | missing | 5 (4, 1) |  | 5 months carry the value in both renderings, fewer than the frozen block length of 6 months, so the moving-block bootstrap is not defined |
| ensemble_full_distribution | missing | missing | 5 (4, 1) |  | 5 months carry the value in both renderings, fewer than the frozen block length of 6 months, so the moving-block bootstrap is not defined |

## did_accuracy_gap

Requirements 8.3, 8.8. Pre-cutoff minus post-cutoff accuracy difference through recall_guard.core.significance.did_gap(correct, identifying, pre_cutoff) over both renderings of every row; rows: the arm's planned evaluation months whose identifying and anonymised renderings both carry the value, pilot items excluded; interval: non-circular moving-block percentile bootstrap over the rows in plan order (recall_guard.core.significance.block_bootstrap) with the plan's block_length, n_resamples, seed and confidence 1 - alpha; a resample whose needed period has no month is dropped and counted; two-sided p = min(1, 2 min(1 + #boot <= 0, 1 + #boot >= 0) / (1 + n_kept)) over the kept resamples (the rule of paired_loss_difference), Holm-adjusted across the variants.

| Variant | Statistic | Interval | p raw | p Holm | Months (pre, post) | Dropped resamples | Missing |
|---|---|---|---|---|---|---|---|
| single_none | 0 | [0, 0.333333] | 1 | 1 | 9 (4, 5) | 0 | 11 planned months without the value in both renderings (reasons in the JSON) |
| single_top_k | 0 | [0, 0.333333] | 1 | 1 | 9 (4, 5) | 0 | 11 planned months without the value in both renderings (reasons in the JSON) |
| single_full_distribution | 0 | [0, 0.333333] | 1 | 1 | 9 (4, 5) | 0 | 11 planned months without the value in both renderings (reasons in the JSON) |
| ensemble_none | missing | missing | missing | missing | 5 (4, 1) |  | 5 months carry the value in both renderings, fewer than the frozen block length of 6 months, so the moving-block bootstrap is not defined |
| ensemble_top_k | missing | missing | missing | missing | 5 (4, 1) |  | 5 months carry the value in both renderings, fewer than the frozen block length of 6 months, so the moving-block bootstrap is not defined |
| ensemble_full_distribution | missing | missing | missing | missing | 5 (4, 1) |  | 5 months carry the value in both renderings, fewer than the frozen block length of 6 months, so the moving-block bootstrap is not defined |

Holm family: single_none, single_top_k, single_full_distribution.

## brier_difference

Requirements 8.4, 8.8. Mean loss difference against the no-adjustment variant of the same draw mode through recall_guard.core.significance.paired_loss_difference(loss of the variant, loss of the reference), the loss of the plan's adjusted scored probability against the outcome as the plan's scored_probability.losses define it; rows: the arm's planned evaluation months whose two renderings both carry the scored probabilities of the variant and its reference and an outcome, one column per rendering, pilot items excluded; moving-block percentile bootstrap with the plan's block_length, n_resamples, seed and confidence 1 - alpha; its two-sided p-value, Holm-adjusted across the variants.

| Variant | Statistic | Interval | p raw | p Holm | Months (pre, post) | Dropped resamples | Missing |
|---|---|---|---|---|---|---|---|
| single_none | missing | missing | missing | missing | 0 (0, 0) |  | not applicable: the no-adjustment variant is the reference the adjusted variants of its draw mode are compared against, so it has no difference against itself |
| single_top_k | -0.256675 | [-0.267328, -0.140812] | 0.0009995 | 0.003998 | 9 (4, 5) | 0 | 11 planned months without the value in both renderings (reasons in the JSON) |
| single_full_distribution | -0.20454 | [-0.260033, -0.116801] | 0.0009995 | 0.003998 | 9 (4, 5) | 0 | 11 planned months without the value in both renderings (reasons in the JSON) |
| ensemble_none | missing | missing | missing | missing | 0 (0, 0) |  | not applicable: the no-adjustment variant is the reference the adjusted variants of its draw mode are compared against, so it has no difference against itself |
| ensemble_top_k | -0.206459 | [-0.225697, -0.114942] | 0.0009995 | 0.003998 | 9 (4, 5) | 0 | 11 planned months without the value in both renderings (reasons in the JSON) |
| ensemble_full_distribution | -0.19236 | [-0.196468, -0.0801512] | 0.0009995 | 0.003998 | 9 (4, 5) | 0 | 11 planned months without the value in both renderings (reasons in the JSON) |

Holm family: single_top_k, single_full_distribution, ensemble_top_k, ensemble_full_distribution.

## log_loss_difference

Requirements 8.4, 8.8. Mean loss difference against the no-adjustment variant of the same draw mode through recall_guard.core.significance.paired_loss_difference(loss of the variant, loss of the reference), the loss of the plan's adjusted scored probability against the outcome as the plan's scored_probability.losses define it; rows: the arm's planned evaluation months whose two renderings both carry the scored probabilities of the variant and its reference and an outcome, one column per rendering, pilot items excluded; moving-block percentile bootstrap with the plan's block_length, n_resamples, seed and confidence 1 - alpha; its two-sided p-value, Holm-adjusted across the variants.

| Variant | Statistic | Interval | p raw | p Holm | Months (pre, post) | Dropped resamples | Missing |
|---|---|---|---|---|---|---|---|
| single_none | missing | missing | missing | missing | 0 (0, 0) |  | not applicable: the no-adjustment variant is the reference the adjusted variants of its draw mode are compared against, so it has no difference against itself |
| single_top_k | -2.03915 | [-2.07286, -1.24894] | 0.0009995 | 0.003998 | 9 (4, 5) | 0 | 11 planned months without the value in both renderings (reasons in the JSON) |
| single_full_distribution | -1.48115 | [-1.73872, -0.912421] | 0.0009995 | 0.003998 | 9 (4, 5) | 0 | 11 planned months without the value in both renderings (reasons in the JSON) |
| ensemble_none | missing | missing | missing | missing | 0 (0, 0) |  | not applicable: the no-adjustment variant is the reference the adjusted variants of its draw mode are compared against, so it has no difference against itself |
| ensemble_top_k | -1.55789 | [-1.72901, -0.945544] | 0.0009995 | 0.003998 | 9 (4, 5) | 0 | 11 planned months without the value in both renderings (reasons in the JSON) |
| ensemble_full_distribution | -1.36695 | [-1.37656, -0.734597] | 0.0009995 | 0.003998 | 9 (4, 5) | 0 | 11 planned months without the value in both renderings (reasons in the JSON) |

Holm family: single_top_k, single_full_distribution, ensemble_top_k, ensemble_full_distribution.

## achieved_power

Requirement 8.9. Missing: fewer than two months carry the noise statistic before or after the cutoff, so the pooled standard deviation and the achieved power cannot be computed.

Read as a t-test approximation under independent months: the one-sided pooled two-sample t test of simulated_power_two_sample at the pre-registered effect, with the realised pre-cutoff and post-cutoff month counts and the pooled within-period standard deviation of the noise variant's identifying minus anonymised correctness; the confirmatory test of the DiD gap is the two-sided moving-block bootstrap, which allows dependence between neighbouring months, so this power approximates the power of that test.

Months without the noise statistic: 15.

## half_ensemble_variance_summary

Deferred: an ensemble metric (Requirements 8.5 to 8.7) that the ensemble-metrics analysis (task 7.11) computes from the same results; not part of this analysis

## half_ensemble_variance_p_memorized

Deferred: an ensemble metric (Requirements 8.5 to 8.7) that the ensemble-metrics analysis (task 7.11) computes from the same results; not part of this analysis

## first_draw_rate

Deferred: an ensemble metric (Requirements 8.5 to 8.7) that the ensemble-metrics analysis (task 7.11) computes from the same results; not part of this analysis

## random_answerer_null

Deferred: an ensemble metric (Requirements 8.5 to 8.7) that the ensemble-metrics analysis (task 7.11) computes from the same results; not part of this analysis
