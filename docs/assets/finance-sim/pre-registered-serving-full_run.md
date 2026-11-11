Sources: macro rows from FRED (Federal Reserve Bank of St. Louis) as known on each decision date, including BAA10Y, Moody's Seasoned Baa Corporate Bond Yield Relative to Yield on 10-Year Treasury Constant Maturity, copyright Moody's, cited and not reproduced; realized next-month returns from Financial Modeling Prep (FMP) end-of-day adjusted closes (licensed, not published) with DTB3 from FRED as the cash leg; EDGAR outlook prompts from SEC EDGAR filings. Replayed offline from the recorded draws and scores by scripts/finance_sim/replay.py (task 8.4) with no model request; the JSON document with the per-item rows stays private under runs/finance-sim/analysis/. Derived statistics only: no reply text, no realized return and no raw licensed series.

# Pre-registered analysis: accuracy, losses and multiple testing

Arm serving (primary arm), stage full_run, model google/gemma-3-4b-it, plan f7603eeb8efdbf74645d658f2bd733a35b3834ba7c90b74d00c6ac2e6b5332fe.

This arm supplies the pre-registered tests: the pre- minus post-cutoff accuracy gap, the Brier and log-loss differences and the negative control, on the none and top_k variants over the full evaluation set, with its achieved power.

Planned months: 90 (68 on or before the model's cutoff, 22 after); pilot items left out: 0; planned results missing: 180.

Intervals: 95 percent non-circular moving-block percentile bootstrap over months, block length 6 months, 2000 resamples, seed 20261110; tested metrics carry two-sided p-values, Holm-adjusted across their variants (alpha 0.05); pre-registered effect 0.1.

Derived statistics only: no reply text, no realized returns and no raw licensed series.

## estimate_difference_pre_cutoff

Requirement 6.3. Identifying minus anonymised p_memorized, mean over the rows of the period; rows: the arm's planned evaluation months whose identifying and anonymised renderings both carry the value, pilot items excluded; interval: non-circular moving-block percentile bootstrap over the rows in plan order (recall_guard.core.significance.block_bootstrap) with the plan's block_length, n_resamples, seed and confidence 1 - alpha; a resample whose needed period has no month is dropped and counted; a measurement (Requirement 6.3), not a test, so no p-value.

| Variant | Statistic | Interval | Months (pre, post) | Dropped resamples | Missing |
|---|---|---|---|---|---|
| single_none | missing | missing | 0 (0, 0) |  | not applicable: the no-adjustment variant carries no memorization estimate (its draws pass without estimates by construction, status estimate_unavailable), so it has no estimate difference |
| single_top_k | missing | missing | 0 (0, 0) |  | no planned month carries the value in both renderings |
| single_full_distribution | missing | missing | 0 (0, 0) |  | missing for the serving arm: the full-distribution host serves the local GGUF, which are different weights from the served model, and values from different builds are not comparable (Requirement 9.4) |
| ensemble_none | missing | missing | 0 (0, 0) |  | not applicable: the no-adjustment variant carries no memorization estimate (its draws pass without estimates by construction, status estimate_unavailable), so it has no estimate difference |
| ensemble_top_k | missing | missing | 0 (0, 0) |  | no planned month carries the value in both renderings |
| ensemble_full_distribution | missing | missing | 0 (0, 0) |  | missing for the serving arm: the full-distribution host serves the local GGUF, which are different weights from the served model, and values from different builds are not comparable (Requirement 9.4) |

## estimate_difference_post_cutoff

Requirement 6.3. Identifying minus anonymised p_memorized, mean over the rows of the period; rows: the arm's planned evaluation months whose identifying and anonymised renderings both carry the value, pilot items excluded; interval: non-circular moving-block percentile bootstrap over the rows in plan order (recall_guard.core.significance.block_bootstrap) with the plan's block_length, n_resamples, seed and confidence 1 - alpha; a resample whose needed period has no month is dropped and counted; a measurement (Requirement 6.3), not a test, so no p-value.

| Variant | Statistic | Interval | Months (pre, post) | Dropped resamples | Missing |
|---|---|---|---|---|---|
| single_none | missing | missing | 0 (0, 0) |  | not applicable: the no-adjustment variant carries no memorization estimate (its draws pass without estimates by construction, status estimate_unavailable), so it has no estimate difference |
| single_top_k | missing | missing | 0 (0, 0) |  | no planned month carries the value in both renderings |
| single_full_distribution | missing | missing | 0 (0, 0) |  | missing for the serving arm: the full-distribution host serves the local GGUF, which are different weights from the served model, and values from different builds are not comparable (Requirement 9.4) |
| ensemble_none | missing | missing | 0 (0, 0) |  | not applicable: the no-adjustment variant carries no memorization estimate (its draws pass without estimates by construction, status estimate_unavailable), so it has no estimate difference |
| ensemble_top_k | missing | missing | 0 (0, 0) |  | no planned month carries the value in both renderings |
| ensemble_full_distribution | missing | missing | 0 (0, 0) |  | missing for the serving arm: the full-distribution host serves the local GGUF, which are different weights from the served model, and values from different builds are not comparable (Requirement 9.4) |

## accuracy_difference_pre_cutoff

Requirement 6.3. Identifying minus anonymised correctness (1 correct, 0 not), mean over the rows of the period; rows: the arm's planned evaluation months whose identifying and anonymised renderings both carry the value, pilot items excluded; interval: non-circular moving-block percentile bootstrap over the rows in plan order (recall_guard.core.significance.block_bootstrap) with the plan's block_length, n_resamples, seed and confidence 1 - alpha; a resample whose needed period has no month is dropped and counted; a measurement (Requirement 6.3), not a test, so no p-value.

| Variant | Statistic | Interval | Months (pre, post) | Dropped resamples | Missing |
|---|---|---|---|---|---|
| single_none | missing | missing | 0 (0, 0) |  | no planned month carries the value in both renderings |
| single_top_k | missing | missing | 0 (0, 0) |  | no planned month carries the value in both renderings |
| single_full_distribution | missing | missing | 0 (0, 0) |  | missing for the serving arm: the full-distribution host serves the local GGUF, which are different weights from the served model, and values from different builds are not comparable (Requirement 9.4) |
| ensemble_none | missing | missing | 0 (0, 0) |  | no planned month carries the value in both renderings |
| ensemble_top_k | missing | missing | 0 (0, 0) |  | no planned month carries the value in both renderings |
| ensemble_full_distribution | missing | missing | 0 (0, 0) |  | missing for the serving arm: the full-distribution host serves the local GGUF, which are different weights from the served model, and values from different builds are not comparable (Requirement 9.4) |

## accuracy_difference_post_cutoff

Requirement 6.3. Identifying minus anonymised correctness (1 correct, 0 not), mean over the rows of the period; rows: the arm's planned evaluation months whose identifying and anonymised renderings both carry the value, pilot items excluded; interval: non-circular moving-block percentile bootstrap over the rows in plan order (recall_guard.core.significance.block_bootstrap) with the plan's block_length, n_resamples, seed and confidence 1 - alpha; a resample whose needed period has no month is dropped and counted; a measurement (Requirement 6.3), not a test, so no p-value.

| Variant | Statistic | Interval | Months (pre, post) | Dropped resamples | Missing |
|---|---|---|---|---|---|
| single_none | missing | missing | 0 (0, 0) |  | no planned month carries the value in both renderings |
| single_top_k | missing | missing | 0 (0, 0) |  | no planned month carries the value in both renderings |
| single_full_distribution | missing | missing | 0 (0, 0) |  | missing for the serving arm: the full-distribution host serves the local GGUF, which are different weights from the served model, and values from different builds are not comparable (Requirement 9.4) |
| ensemble_none | missing | missing | 0 (0, 0) |  | no planned month carries the value in both renderings |
| ensemble_top_k | missing | missing | 0 (0, 0) |  | no planned month carries the value in both renderings |
| ensemble_full_distribution | missing | missing | 0 (0, 0) |  | missing for the serving arm: the full-distribution host serves the local GGUF, which are different weights from the served model, and values from different builds are not comparable (Requirement 9.4) |

## did_accuracy_gap

Requirements 8.3, 8.8. Pre-cutoff minus post-cutoff accuracy difference through recall_guard.core.significance.did_gap(correct, identifying, pre_cutoff) over both renderings of every row; rows: the arm's planned evaluation months whose identifying and anonymised renderings both carry the value, pilot items excluded; interval: non-circular moving-block percentile bootstrap over the rows in plan order (recall_guard.core.significance.block_bootstrap) with the plan's block_length, n_resamples, seed and confidence 1 - alpha; a resample whose needed period has no month is dropped and counted; two-sided p = min(1, 2 min(1 + #boot <= 0, 1 + #boot >= 0) / (1 + n_kept)) over the kept resamples (the rule of paired_loss_difference), Holm-adjusted across the variants.

| Variant | Statistic | Interval | p raw | p Holm | Months (pre, post) | Dropped resamples | Missing |
|---|---|---|---|---|---|---|---|
| single_none | missing | missing | missing | missing | 0 (0, 0) |  | no planned month carries the value in both renderings |
| single_top_k | missing | missing | missing | missing | 0 (0, 0) |  | no planned month carries the value in both renderings |
| single_full_distribution | missing | missing | missing | missing | 0 (0, 0) |  | missing for the serving arm: the full-distribution host serves the local GGUF, which are different weights from the served model, and values from different builds are not comparable (Requirement 9.4) |
| ensemble_none | missing | missing | missing | missing | 0 (0, 0) |  | no planned month carries the value in both renderings |
| ensemble_top_k | missing | missing | missing | missing | 0 (0, 0) |  | no planned month carries the value in both renderings |
| ensemble_full_distribution | missing | missing | missing | missing | 0 (0, 0) |  | missing for the serving arm: the full-distribution host serves the local GGUF, which are different weights from the served model, and values from different builds are not comparable (Requirement 9.4) |

Holm family: none.

## brier_difference

Requirements 8.4, 8.8. Mean loss difference against the no-adjustment variant of the same draw mode through recall_guard.core.significance.paired_loss_difference(loss of the variant, loss of the reference), the loss of the plan's adjusted scored probability against the outcome as the plan's scored_probability.losses define it; rows: the arm's planned evaluation months whose two renderings both carry the scored probabilities of the variant and its reference and an outcome, one column per rendering, pilot items excluded; moving-block percentile bootstrap with the plan's block_length, n_resamples, seed and confidence 1 - alpha; its two-sided p-value, Holm-adjusted across the variants.

| Variant | Statistic | Interval | p raw | p Holm | Months (pre, post) | Dropped resamples | Missing |
|---|---|---|---|---|---|---|---|
| single_none | missing | missing | missing | missing | 0 (0, 0) |  | not applicable: the no-adjustment variant is the reference the adjusted variants of its draw mode are compared against, so it has no difference against itself |
| single_top_k | missing | missing | missing | missing | 0 (0, 0) |  | no planned month carries the value in both renderings |
| single_full_distribution | missing | missing | missing | missing | 0 (0, 0) |  | missing for the serving arm: the full-distribution host serves the local GGUF, which are different weights from the served model, and values from different builds are not comparable (Requirement 9.4) |
| ensemble_none | missing | missing | missing | missing | 0 (0, 0) |  | not applicable: the no-adjustment variant is the reference the adjusted variants of its draw mode are compared against, so it has no difference against itself |
| ensemble_top_k | missing | missing | missing | missing | 0 (0, 0) |  | no planned month carries the value in both renderings |
| ensemble_full_distribution | missing | missing | missing | missing | 0 (0, 0) |  | missing for the serving arm: the full-distribution host serves the local GGUF, which are different weights from the served model, and values from different builds are not comparable (Requirement 9.4) |

Holm family: none.

## log_loss_difference

Requirements 8.4, 8.8. Mean loss difference against the no-adjustment variant of the same draw mode through recall_guard.core.significance.paired_loss_difference(loss of the variant, loss of the reference), the loss of the plan's adjusted scored probability against the outcome as the plan's scored_probability.losses define it; rows: the arm's planned evaluation months whose two renderings both carry the scored probabilities of the variant and its reference and an outcome, one column per rendering, pilot items excluded; moving-block percentile bootstrap with the plan's block_length, n_resamples, seed and confidence 1 - alpha; its two-sided p-value, Holm-adjusted across the variants.

| Variant | Statistic | Interval | p raw | p Holm | Months (pre, post) | Dropped resamples | Missing |
|---|---|---|---|---|---|---|---|
| single_none | missing | missing | missing | missing | 0 (0, 0) |  | not applicable: the no-adjustment variant is the reference the adjusted variants of its draw mode are compared against, so it has no difference against itself |
| single_top_k | missing | missing | missing | missing | 0 (0, 0) |  | no planned month carries the value in both renderings |
| single_full_distribution | missing | missing | missing | missing | 0 (0, 0) |  | missing for the serving arm: the full-distribution host serves the local GGUF, which are different weights from the served model, and values from different builds are not comparable (Requirement 9.4) |
| ensemble_none | missing | missing | missing | missing | 0 (0, 0) |  | not applicable: the no-adjustment variant is the reference the adjusted variants of its draw mode are compared against, so it has no difference against itself |
| ensemble_top_k | missing | missing | missing | missing | 0 (0, 0) |  | no planned month carries the value in both renderings |
| ensemble_full_distribution | missing | missing | missing | missing | 0 (0, 0) |  | missing for the serving arm: the full-distribution host serves the local GGUF, which are different weights from the served model, and values from different builds are not comparable (Requirement 9.4) |

Holm family: none.

## achieved_power

Requirement 8.9. Missing: fewer than two months carry the noise statistic before or after the cutoff, so the pooled standard deviation and the achieved power cannot be computed.

Read as a t-test approximation under independent months: the one-sided pooled two-sample t test of simulated_power_two_sample at the pre-registered effect, with the realised pre-cutoff and post-cutoff month counts and the pooled within-period standard deviation of the noise variant's identifying minus anonymised correctness; the confirmatory test of the DiD gap is the two-sided moving-block bootstrap, which allows dependence between neighbouring months, so this power approximates the power of that test.

Months without the noise statistic: 90.

## half_ensemble_variance_summary

Deferred: an ensemble metric (Requirements 8.5 to 8.7) that the ensemble-metrics analysis (task 7.11) computes from the same results; not part of this analysis

## half_ensemble_variance_p_memorized

Deferred: an ensemble metric (Requirements 8.5 to 8.7) that the ensemble-metrics analysis (task 7.11) computes from the same results; not part of this analysis

## first_draw_rate

Deferred: an ensemble metric (Requirements 8.5 to 8.7) that the ensemble-metrics analysis (task 7.11) computes from the same results; not part of this analysis

## random_answerer_null

Deferred: an ensemble metric (Requirements 8.5 to 8.7) that the ensemble-metrics analysis (task 7.11) computes from the same results; not part of this analysis

## Missing and excluded items

- macro-2016-02 identifying: no stored result: the stage holds no result for this item and rendering, so the run is incomplete
- macro-2016-02 anonymised: no stored result: the stage holds no result for this item and rendering, so the run is incomplete
- macro-2016-03 identifying: no stored result: the stage holds no result for this item and rendering, so the run is incomplete
- macro-2016-03 anonymised: no stored result: the stage holds no result for this item and rendering, so the run is incomplete
- macro-2016-04 identifying: no stored result: the stage holds no result for this item and rendering, so the run is incomplete
- macro-2016-04 anonymised: no stored result: the stage holds no result for this item and rendering, so the run is incomplete
- macro-2016-05 identifying: no stored result: the stage holds no result for this item and rendering, so the run is incomplete
- macro-2016-05 anonymised: no stored result: the stage holds no result for this item and rendering, so the run is incomplete
- macro-2016-06 identifying: no stored result: the stage holds no result for this item and rendering, so the run is incomplete
- macro-2016-06 anonymised: no stored result: the stage holds no result for this item and rendering, so the run is incomplete
- macro-2016-09 identifying: no stored result: the stage holds no result for this item and rendering, so the run is incomplete
- macro-2016-09 anonymised: no stored result: the stage holds no result for this item and rendering, so the run is incomplete
- macro-2016-11 identifying: no stored result: the stage holds no result for this item and rendering, so the run is incomplete
- macro-2016-11 anonymised: no stored result: the stage holds no result for this item and rendering, so the run is incomplete
- macro-2016-12 identifying: no stored result: the stage holds no result for this item and rendering, so the run is incomplete
- macro-2016-12 anonymised: no stored result: the stage holds no result for this item and rendering, so the run is incomplete
- macro-2017-02 identifying: no stored result: the stage holds no result for this item and rendering, so the run is incomplete
- macro-2017-02 anonymised: no stored result: the stage holds no result for this item and rendering, so the run is incomplete
- macro-2017-04 identifying: no stored result: the stage holds no result for this item and rendering, so the run is incomplete
- macro-2017-04 anonymised: no stored result: the stage holds no result for this item and rendering, so the run is incomplete
- macro-2017-05 identifying: no stored result: the stage holds no result for this item and rendering, so the run is incomplete
- macro-2017-05 anonymised: no stored result: the stage holds no result for this item and rendering, so the run is incomplete
- macro-2017-06 identifying: no stored result: the stage holds no result for this item and rendering, so the run is incomplete
- macro-2017-06 anonymised: no stored result: the stage holds no result for this item and rendering, so the run is incomplete
- macro-2017-08 identifying: no stored result: the stage holds no result for this item and rendering, so the run is incomplete
- macro-2017-08 anonymised: no stored result: the stage holds no result for this item and rendering, so the run is incomplete
- macro-2017-09 identifying: no stored result: the stage holds no result for this item and rendering, so the run is incomplete
- macro-2017-09 anonymised: no stored result: the stage holds no result for this item and rendering, so the run is incomplete
- macro-2017-11 identifying: no stored result: the stage holds no result for this item and rendering, so the run is incomplete
- macro-2017-11 anonymised: no stored result: the stage holds no result for this item and rendering, so the run is incomplete
- macro-2018-01 identifying: no stored result: the stage holds no result for this item and rendering, so the run is incomplete
- macro-2018-01 anonymised: no stored result: the stage holds no result for this item and rendering, so the run is incomplete
- macro-2018-02 identifying: no stored result: the stage holds no result for this item and rendering, so the run is incomplete
- macro-2018-02 anonymised: no stored result: the stage holds no result for this item and rendering, so the run is incomplete
- macro-2018-03 identifying: no stored result: the stage holds no result for this item and rendering, so the run is incomplete
- macro-2018-03 anonymised: no stored result: the stage holds no result for this item and rendering, so the run is incomplete
- macro-2018-05 identifying: no stored result: the stage holds no result for this item and rendering, so the run is incomplete
- macro-2018-05 anonymised: no stored result: the stage holds no result for this item and rendering, so the run is incomplete
- macro-2018-06 identifying: no stored result: the stage holds no result for this item and rendering, so the run is incomplete
- macro-2018-06 anonymised: no stored result: the stage holds no result for this item and rendering, so the run is incomplete
- macro-2018-08 identifying: no stored result: the stage holds no result for this item and rendering, so the run is incomplete
- macro-2018-08 anonymised: no stored result: the stage holds no result for this item and rendering, so the run is incomplete
- macro-2018-10 identifying: no stored result: the stage holds no result for this item and rendering, so the run is incomplete
- macro-2018-10 anonymised: no stored result: the stage holds no result for this item and rendering, so the run is incomplete
- macro-2018-12 identifying: no stored result: the stage holds no result for this item and rendering, so the run is incomplete
- macro-2018-12 anonymised: no stored result: the stage holds no result for this item and rendering, so the run is incomplete
- macro-2019-01 identifying: no stored result: the stage holds no result for this item and rendering, so the run is incomplete
- macro-2019-01 anonymised: no stored result: the stage holds no result for this item and rendering, so the run is incomplete
- macro-2019-02 identifying: no stored result: the stage holds no result for this item and rendering, so the run is incomplete
- macro-2019-02 anonymised: no stored result: the stage holds no result for this item and rendering, so the run is incomplete
- macro-2019-04 identifying: no stored result: the stage holds no result for this item and rendering, so the run is incomplete
- macro-2019-04 anonymised: no stored result: the stage holds no result for this item and rendering, so the run is incomplete
- macro-2019-08 identifying: no stored result: the stage holds no result for this item and rendering, so the run is incomplete
- macro-2019-08 anonymised: no stored result: the stage holds no result for this item and rendering, so the run is incomplete
- macro-2019-10 identifying: no stored result: the stage holds no result for this item and rendering, so the run is incomplete
- macro-2019-10 anonymised: no stored result: the stage holds no result for this item and rendering, so the run is incomplete
- macro-2019-11 identifying: no stored result: the stage holds no result for this item and rendering, so the run is incomplete
- macro-2019-11 anonymised: no stored result: the stage holds no result for this item and rendering, so the run is incomplete
- macro-2019-12 identifying: no stored result: the stage holds no result for this item and rendering, so the run is incomplete
- macro-2019-12 anonymised: no stored result: the stage holds no result for this item and rendering, so the run is incomplete
- macro-2020-01 identifying: no stored result: the stage holds no result for this item and rendering, so the run is incomplete
- macro-2020-01 anonymised: no stored result: the stage holds no result for this item and rendering, so the run is incomplete
- macro-2020-04 identifying: no stored result: the stage holds no result for this item and rendering, so the run is incomplete
- macro-2020-04 anonymised: no stored result: the stage holds no result for this item and rendering, so the run is incomplete
- macro-2020-05 identifying: no stored result: the stage holds no result for this item and rendering, so the run is incomplete
- macro-2020-05 anonymised: no stored result: the stage holds no result for this item and rendering, so the run is incomplete
- macro-2020-06 identifying: no stored result: the stage holds no result for this item and rendering, so the run is incomplete
- macro-2020-06 anonymised: no stored result: the stage holds no result for this item and rendering, so the run is incomplete
- macro-2020-07 identifying: no stored result: the stage holds no result for this item and rendering, so the run is incomplete
- macro-2020-07 anonymised: no stored result: the stage holds no result for this item and rendering, so the run is incomplete
- macro-2020-08 identifying: no stored result: the stage holds no result for this item and rendering, so the run is incomplete
- macro-2020-08 anonymised: no stored result: the stage holds no result for this item and rendering, so the run is incomplete
- macro-2020-09 identifying: no stored result: the stage holds no result for this item and rendering, so the run is incomplete
- macro-2020-09 anonymised: no stored result: the stage holds no result for this item and rendering, so the run is incomplete
- macro-2020-12 identifying: no stored result: the stage holds no result for this item and rendering, so the run is incomplete
- macro-2020-12 anonymised: no stored result: the stage holds no result for this item and rendering, so the run is incomplete
- macro-2021-01 identifying: no stored result: the stage holds no result for this item and rendering, so the run is incomplete
- macro-2021-01 anonymised: no stored result: the stage holds no result for this item and rendering, so the run is incomplete
- macro-2021-02 identifying: no stored result: the stage holds no result for this item and rendering, so the run is incomplete
- macro-2021-02 anonymised: no stored result: the stage holds no result for this item and rendering, so the run is incomplete
- macro-2021-05 identifying: no stored result: the stage holds no result for this item and rendering, so the run is incomplete
- macro-2021-05 anonymised: no stored result: the stage holds no result for this item and rendering, so the run is incomplete
- macro-2021-06 identifying: no stored result: the stage holds no result for this item and rendering, so the run is incomplete
- macro-2021-06 anonymised: no stored result: the stage holds no result for this item and rendering, so the run is incomplete
- macro-2021-07 identifying: no stored result: the stage holds no result for this item and rendering, so the run is incomplete
- macro-2021-07 anonymised: no stored result: the stage holds no result for this item and rendering, so the run is incomplete
- macro-2021-08 identifying: no stored result: the stage holds no result for this item and rendering, so the run is incomplete
- macro-2021-08 anonymised: no stored result: the stage holds no result for this item and rendering, so the run is incomplete
- macro-2021-11 identifying: no stored result: the stage holds no result for this item and rendering, so the run is incomplete
- macro-2021-11 anonymised: no stored result: the stage holds no result for this item and rendering, so the run is incomplete
- macro-2021-12 identifying: no stored result: the stage holds no result for this item and rendering, so the run is incomplete
- macro-2021-12 anonymised: no stored result: the stage holds no result for this item and rendering, so the run is incomplete
- macro-2022-02 identifying: no stored result: the stage holds no result for this item and rendering, so the run is incomplete
- macro-2022-02 anonymised: no stored result: the stage holds no result for this item and rendering, so the run is incomplete
- macro-2022-03 identifying: no stored result: the stage holds no result for this item and rendering, so the run is incomplete
- macro-2022-03 anonymised: no stored result: the stage holds no result for this item and rendering, so the run is incomplete
- macro-2022-04 identifying: no stored result: the stage holds no result for this item and rendering, so the run is incomplete
- macro-2022-04 anonymised: no stored result: the stage holds no result for this item and rendering, so the run is incomplete
- macro-2022-05 identifying: no stored result: the stage holds no result for this item and rendering, so the run is incomplete
- macro-2022-05 anonymised: no stored result: the stage holds no result for this item and rendering, so the run is incomplete
- macro-2022-08 identifying: no stored result: the stage holds no result for this item and rendering, so the run is incomplete
- macro-2022-08 anonymised: no stored result: the stage holds no result for this item and rendering, so the run is incomplete
- macro-2022-09 identifying: no stored result: the stage holds no result for this item and rendering, so the run is incomplete
- macro-2022-09 anonymised: no stored result: the stage holds no result for this item and rendering, so the run is incomplete
- macro-2022-10 identifying: no stored result: the stage holds no result for this item and rendering, so the run is incomplete
- macro-2022-10 anonymised: no stored result: the stage holds no result for this item and rendering, so the run is incomplete
- macro-2022-11 identifying: no stored result: the stage holds no result for this item and rendering, so the run is incomplete
- macro-2022-11 anonymised: no stored result: the stage holds no result for this item and rendering, so the run is incomplete
- macro-2023-04 identifying: no stored result: the stage holds no result for this item and rendering, so the run is incomplete
- macro-2023-04 anonymised: no stored result: the stage holds no result for this item and rendering, so the run is incomplete
- macro-2023-05 identifying: no stored result: the stage holds no result for this item and rendering, so the run is incomplete
- macro-2023-05 anonymised: no stored result: the stage holds no result for this item and rendering, so the run is incomplete
- macro-2023-06 identifying: no stored result: the stage holds no result for this item and rendering, so the run is incomplete
- macro-2023-06 anonymised: no stored result: the stage holds no result for this item and rendering, so the run is incomplete
- macro-2023-08 identifying: no stored result: the stage holds no result for this item and rendering, so the run is incomplete
- macro-2023-08 anonymised: no stored result: the stage holds no result for this item and rendering, so the run is incomplete
- macro-2023-09 identifying: no stored result: the stage holds no result for this item and rendering, so the run is incomplete
- macro-2023-09 anonymised: no stored result: the stage holds no result for this item and rendering, so the run is incomplete
- macro-2023-11 identifying: no stored result: the stage holds no result for this item and rendering, so the run is incomplete
- macro-2023-11 anonymised: no stored result: the stage holds no result for this item and rendering, so the run is incomplete
- macro-2023-12 identifying: no stored result: the stage holds no result for this item and rendering, so the run is incomplete
- macro-2023-12 anonymised: no stored result: the stage holds no result for this item and rendering, so the run is incomplete
- macro-2024-01 identifying: no stored result: the stage holds no result for this item and rendering, so the run is incomplete
- macro-2024-01 anonymised: no stored result: the stage holds no result for this item and rendering, so the run is incomplete
- macro-2024-02 identifying: no stored result: the stage holds no result for this item and rendering, so the run is incomplete
- macro-2024-02 anonymised: no stored result: the stage holds no result for this item and rendering, so the run is incomplete
- macro-2024-03 identifying: no stored result: the stage holds no result for this item and rendering, so the run is incomplete
- macro-2024-03 anonymised: no stored result: the stage holds no result for this item and rendering, so the run is incomplete
- macro-2024-04 identifying: no stored result: the stage holds no result for this item and rendering, so the run is incomplete
- macro-2024-04 anonymised: no stored result: the stage holds no result for this item and rendering, so the run is incomplete
- macro-2024-05 identifying: no stored result: the stage holds no result for this item and rendering, so the run is incomplete
- macro-2024-05 anonymised: no stored result: the stage holds no result for this item and rendering, so the run is incomplete
- macro-2024-06 identifying: no stored result: the stage holds no result for this item and rendering, so the run is incomplete
- macro-2024-06 anonymised: no stored result: the stage holds no result for this item and rendering, so the run is incomplete
- macro-2024-07 identifying: no stored result: the stage holds no result for this item and rendering, so the run is incomplete
- macro-2024-07 anonymised: no stored result: the stage holds no result for this item and rendering, so the run is incomplete
- macro-2024-09 identifying: no stored result: the stage holds no result for this item and rendering, so the run is incomplete
- macro-2024-09 anonymised: no stored result: the stage holds no result for this item and rendering, so the run is incomplete
- macro-2024-10 identifying: no stored result: the stage holds no result for this item and rendering, so the run is incomplete
- macro-2024-10 anonymised: no stored result: the stage holds no result for this item and rendering, so the run is incomplete
- macro-2024-11 identifying: no stored result: the stage holds no result for this item and rendering, so the run is incomplete
- macro-2024-11 anonymised: no stored result: the stage holds no result for this item and rendering, so the run is incomplete
- macro-2024-12 identifying: no stored result: the stage holds no result for this item and rendering, so the run is incomplete
- macro-2024-12 anonymised: no stored result: the stage holds no result for this item and rendering, so the run is incomplete
- macro-2025-01 identifying: no stored result: the stage holds no result for this item and rendering, so the run is incomplete
- macro-2025-01 anonymised: no stored result: the stage holds no result for this item and rendering, so the run is incomplete
- macro-2025-02 identifying: no stored result: the stage holds no result for this item and rendering, so the run is incomplete
- macro-2025-02 anonymised: no stored result: the stage holds no result for this item and rendering, so the run is incomplete
- macro-2025-03 identifying: no stored result: the stage holds no result for this item and rendering, so the run is incomplete
- macro-2025-03 anonymised: no stored result: the stage holds no result for this item and rendering, so the run is incomplete
- macro-2025-04 identifying: no stored result: the stage holds no result for this item and rendering, so the run is incomplete
- macro-2025-04 anonymised: no stored result: the stage holds no result for this item and rendering, so the run is incomplete
- macro-2025-05 identifying: no stored result: the stage holds no result for this item and rendering, so the run is incomplete
- macro-2025-05 anonymised: no stored result: the stage holds no result for this item and rendering, so the run is incomplete
- macro-2025-06 identifying: no stored result: the stage holds no result for this item and rendering, so the run is incomplete
- macro-2025-06 anonymised: no stored result: the stage holds no result for this item and rendering, so the run is incomplete
- macro-2025-07 identifying: no stored result: the stage holds no result for this item and rendering, so the run is incomplete
- macro-2025-07 anonymised: no stored result: the stage holds no result for this item and rendering, so the run is incomplete
- macro-2025-08 identifying: no stored result: the stage holds no result for this item and rendering, so the run is incomplete
- macro-2025-08 anonymised: no stored result: the stage holds no result for this item and rendering, so the run is incomplete
- macro-2025-09 identifying: no stored result: the stage holds no result for this item and rendering, so the run is incomplete
- macro-2025-09 anonymised: no stored result: the stage holds no result for this item and rendering, so the run is incomplete
- macro-2025-10 identifying: no stored result: the stage holds no result for this item and rendering, so the run is incomplete
- macro-2025-10 anonymised: no stored result: the stage holds no result for this item and rendering, so the run is incomplete
- macro-2025-11 identifying: no stored result: the stage holds no result for this item and rendering, so the run is incomplete
- macro-2025-11 anonymised: no stored result: the stage holds no result for this item and rendering, so the run is incomplete
- macro-2025-12 identifying: no stored result: the stage holds no result for this item and rendering, so the run is incomplete
- macro-2025-12 anonymised: no stored result: the stage holds no result for this item and rendering, so the run is incomplete
- macro-2026-01 identifying: no stored result: the stage holds no result for this item and rendering, so the run is incomplete
- macro-2026-01 anonymised: no stored result: the stage holds no result for this item and rendering, so the run is incomplete
- macro-2026-02 identifying: no stored result: the stage holds no result for this item and rendering, so the run is incomplete
- macro-2026-02 anonymised: no stored result: the stage holds no result for this item and rendering, so the run is incomplete
- macro-2026-03 identifying: no stored result: the stage holds no result for this item and rendering, so the run is incomplete
- macro-2026-03 anonymised: no stored result: the stage holds no result for this item and rendering, so the run is incomplete
- macro-2026-04 identifying: no stored result: the stage holds no result for this item and rendering, so the run is incomplete
- macro-2026-04 anonymised: no stored result: the stage holds no result for this item and rendering, so the run is incomplete
- macro-2026-05 identifying: no stored result: the stage holds no result for this item and rendering, so the run is incomplete
- macro-2026-05 anonymised: no stored result: the stage holds no result for this item and rendering, so the run is incomplete
- macro-2026-06 identifying: no stored result: the stage holds no result for this item and rendering, so the run is incomplete
- macro-2026-06 anonymised: no stored result: the stage holds no result for this item and rendering, so the run is incomplete
