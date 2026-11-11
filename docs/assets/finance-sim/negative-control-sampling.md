Sources: macro rows from FRED (Federal Reserve Bank of St. Louis) as known on each decision date, including BAA10Y, Moody's Seasoned Baa Corporate Bond Yield Relative to Yield on 10-Year Treasury Constant Maturity, copyright Moody's, cited and not reproduced; realized next-month returns from Financial Modeling Prep (FMP) end-of-day adjusted closes (licensed, not published) with DTB3 from FRED as the cash leg; EDGAR outlook prompts from SEC EDGAR filings. Replayed offline from the recorded draws and scores by scripts/finance_sim/replay.py (task 8.4) with no model request; the JSON document with the per-item rows stays private under runs/finance-sim/analysis/. Derived statistics only: no reply text, no realized return and no raw licensed series.

# Negative control: the change each adjustment makes against no adjustment

Arm sampling, stage negative_control, plan f7603eeb8efdbf74645d658f2bd733a35b3834ba7c90b74d00c6ac2e6b5332fe.

Releases: 40 frozen held-out control releases over 9 filing quarters (releases per quarter: 2024Q3 1, 2024Q4 5, 2025Q1 7, 2025Q2 4, 2025Q3 2, 2025Q4 6, 2026Q1 8, 2026Q2 2, 2026Q3 5); results stored: 40; planned results missing: 0; stage complete.

Tolerance 0.1. Intervals: 95 percent non-circular moving-block percentile bootstrap over filing quarters, block length 2 quarters, 2000 resamples, seed 20261110; the statistic's interval and the signed interval come from the same resamples.

Outcome rule: per variant with an adjustment, against the frozen tolerance: 'no material change' when the interval's upper bound is below the tolerance; 'discounting honest answers' (the flag) when its lower bound is above the tolerance; 'inconclusive' otherwise. Variants flagged as discounting honest answers: none.

| variant | draw_mode | adjustment | reference | outcome | flagged | releases | mean_absolute_change | interval_low | interval_high | signed_mean_change | signed_interval_low | signed_interval_high | sign_flips | sign_flip_share | dropped_resamples | statuses | missing |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| single_top_k | single | top_k | single_none | no material change | false | 40 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | adjustment_skipped_weak_calibrator 40 |  |
| single_full_distribution | single | full_distribution | single_none | no material change | false | 40 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | adjustment_skipped_weak_calibrator 40 |  |
| ensemble_top_k | ensemble | top_k | ensemble_none | no material change | false | 40 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | adjustment_skipped_weak_calibrator 40 |  |
| ensemble_full_distribution | ensemble | full_distribution | ensemble_none | no material change | false | 40 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | adjustment_skipped_weak_calibrator 40 |  |

Method: per release, the change of the outlook = adjusted minus raw, the raw outlook being the no-adjustment variant's of the same draw mode on the same draws; statistic: mean absolute change over the releases with both values, with the signed mean change and the share of releases whose sign flips beside it; interval: non-circular moving-block percentile bootstrap over filing quarters (rows are the quarters holding a frozen release, in calendar order; each resample draws ceil(Q / L) start quarters uniformly from 0..Q - L, takes L consecutive quarters from each, truncates to Q quarters and pools their releases), with the plan's seed, n_resamples, negative_control_block_length and confidence 1 - alpha; a resample with no computable release is dropped and counted. Statuses count the guarded statuses of the variant over the releases with a result; a weak calibrator or unavailable estimate leaves the raw outlook unchanged (change 0), so the counts stand beside every outcome.

The outlook prompt ends with the JSON example {"outlook": 0.2}, a mild positive anchor that is the same for every arm and variant, so it cancels in the per-release change against no adjustment.

Data: the held-out control releases of the SEC EDGAR corpus rendered as outlook prompts (public filings); no FRED series, FMP price or BAA10Y value enters the negative control. The study's positive-control outcomes come from Financial Modeling Prep (FMP) end-of-day adjusted closes (licensed, not published) and its macro rows from FRED including BAA10Y (Moody's Seasoned Baa Corporate Bond Yield Relative to Yield on 10-Year Treasury Constant Maturity, copyright Moody's, cited and not reproduced); no such value is listed here.

Derived statistics only: no release text, no reply text, no per-release row and no raw licensed series (the per-release rows stay in the private JSON).
