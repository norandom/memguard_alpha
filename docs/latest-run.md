# Latest run results

Snapshot from two local example runs. Regenerate by re-running `./start.sh` (recall-guard check) or `scripts/run_cmmd_backtest.py` (`cmmd` backtest). A fixed seed keeps the local sampling and bootstrap steps stable, but hosted model outputs and upstream price data can still move the numbers.

## Recall-guard check — `runs/20260429T073229Z`

Source: `runs/20260429T073229Z/summary.csv`.

| Model | Raw Acc (95% CI) | MemGuard Acc (95% CI) | MCS-AUC | Parse % | Score | Survives gates | Warnings |
| --- | --- | --- | --- | --- | --- | --- | --- |
| meta/llama-3.1-8b-instruct | 0.486 (0.430–0.542) | 0.486 (0.430–0.542) | 1.000 | 86.7% | 0.000 | no | not-better-than-baseline |
| meta/llama-3.2-3b-instruct | 0.535 (0.480–0.590) | 0.535 (0.480–0.590) | 0.996 | 99.1% | 0.000 | no | not-better-than-baseline |
| openai/gpt-oss-20b | 0.549 (0.492–0.603) | 0.549 (0.492–0.603) | 0.976 | 95.5% | 0.000 | no | not-better-than-baseline |
| microsoft/phi-4-mini-instruct | 0.319 (0.244–0.388) | 0.319 (0.244–0.388) | 0.855 | 48.5% | 0.000 | no | parse-unreliable; not-better-than-baseline |
| *majority baseline* | 0.539 (0.485–0.597) | — | — | — | — | — | — |

No model beat the majority baseline on this stream, so all four score 0 and none survive the gates. On this eval set (next-day ETF direction) that is not surprising; see [What this measures](index.md) in the project overview.

## `cmmd` backtest — `runs/cmmd_20260429T104542Z`

Single-model run (`openai/gpt-oss-20b`). Sources: `summary.csv`, `is_oos_gap.csv`, `backtest_summary.csv` in the run directory.

| Metric | Value |
| --- | --- |
| Raw Acc (95% CI) | 0.514 (0.463–0.569) |
| MCS-AUC | 0.784 |
| Pre-cutoff accuracy (n=253, <= 2024-06-30) | 0.490 (0.427–0.553) |
| Post-cutoff accuracy (n=60, > 2024-06-30) | 0.617 (0.500–0.750) |
| Pre − Post gap | −0.127 |

| Strategy | Sharpe (95% CI) | Mean daily bps | Max drawdown | Total return | Signals used |
| --- | --- | --- | --- | --- | --- |
| raw_alpha | −1.14 (−1.67 to −0.28) | −0.98 | −21.6% | −21.7% | 313 |
| cmmd | −1.01 (−1.51 to −0.21) | −0.81 | −18.2% | −18.3% | 250 |

`cmmd` filtering (threshold 0.278) dropped 63 higher-score signals and reduced the drawdown and loss relative to `raw_alpha`. Both variants still lose money on this stream.

## Metric definitions

| Metric | Column(s) | Meaning |
| --- | --- | --- |
| Raw accuracy | `raw_acc_point/lo/hi` | Share of parse-OK rows where `predicted_direction == target_direction`, with bootstrap 95% CI. |
| MemGuard accuracy | `memguard_acc_point/lo/hi` | Same denominator as raw accuracy. The MemGuard penalty discounts confidence but does not flip the predicted direction, so the two coincide in this spec; the column exists so a confidence-thresholded variant can diverge later. |
| MCS-AUC | `mcs_auc_point/lo/hi` | AUC of the classifier on held-out `(p_memorized, label)` pairs from the shipped calibration split. High values mean the classifier separated those two corpora cleanly for that model. |
| Parse success rate | `parse_success_rate`, `parse_failures` | Fraction of model responses that parsed into a usable prediction; the complement counts failures. |
| Composite score | `score` | `memguard_acc_lo * mcs_auc_point * parse_success_rate`. Zeroed for models that fail a gate. |
| Survives gates | `survives_gates` | `true` only if parse rate >= 0.8, MCS-AUC >= 0.6, and the model's accuracy lower CI clears the majority baseline's upper CI. Non-survivors are excluded from the top-3 ranking. |
| Warnings | `warnings` | Gate/quality flags: `parse-unreliable` (parse < 0.8), `weak-calibration` (MCS-AUC < 0.6), `not-better-than-baseline` (accuracy CI does not clear the majority baseline). |
| Majority baseline | `__majority_baseline__` row | Accuracy of always predicting the majority direction of the eval stream. |
| Pre/post-cutoff accuracy & gap | `is_oos_gap.csv` | Accuracy on rows dated before vs after the model's training cutoff, with bootstrap CIs. The gap is a descriptive split, not proof of memorization by itself. |
| Cohen's d | `cohens_d.csv` | Per-MIA-feature effect size between the pre-cutoff and post-cutoff raw feature values, reported with the holdout MCS-AUC for context. |
| Sharpe / bps / drawdown | `backtest_summary.csv` | Annualized Sharpe ratio, mean daily return in basis points, maximum drawdown, and total return for each backtest variant, with bootstrap CIs. |
| `cmmd` threshold | `backtest_summary.csv` | The `p_memorized` quantile cutoff used by this backtest run; signals above it are excluded from the `cmmd` variant. |
