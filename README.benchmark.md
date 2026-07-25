# Recall Guard benchmark

Benchmark date: **2026-04-29**

This page pulls the model benchmark material out of `README.md` and keeps it in one place.
The figures below come from two local runs against the same 330-prompt, 10-year `SWDA.L / XLK / IAU` eval set:

- `runs/20260429T073229Z/`: recall-guard check, four-model shortlist
- `runs/cmmd_20260429T070616Z/`: single-model `cmmd` backtest for `openai/gpt-oss-20b`

These are local run artifacts, not tracked repository files. Re-run the commands in `README.md` to regenerate them in your own checkout.

## Recall-guard benchmark

| Model | Parse % | Raw Acc (95% CI) | MemGuard Acc (95% CI) | MCS-AUC | Score | Survives gates | Warnings |
| --- | ---: | --- | --- | ---: | ---: | --- | --- |
| `openai/gpt-oss-20b` | 95.5% | 0.5492 [0.4921–0.6032] | 0.5492 [0.4921–0.6032] | 0.976 | 0.000 | no | `not-better-than-baseline` |
| `meta/llama-3.2-3b-instruct` | 99.1% | 0.5352 [0.4801–0.5902] | 0.5352 [0.4801–0.5902] | 0.996 | 0.000 | no | `not-better-than-baseline` |
| `meta/llama-3.1-8b-instruct` | 86.7% | 0.4860 [0.4301–0.5420] | 0.4860 [0.4301–0.5420] | 1.000 | 0.000 | no | `not-better-than-baseline` |
| `microsoft/phi-4-mini-instruct` | 48.5% | 0.3187 [0.2437–0.3875] | 0.3187 [0.2437–0.3875] | 0.854 | 0.000 | no | `parse-unreliable`, `not-better-than-baseline` |
| `__majority_baseline__` | — | 0.5394 [0.4848–0.5970] | — | — | — | — | `always-predict-the-majority-class` |

### Quick read

- `gpt-oss-20b` has the highest point estimate, but its interval still overlaps the majority baseline.
- `phi-4-mini` parses much less reliably than the other models, so its estimate comes from a smaller usable subset.
- High MCS-AUC means the classifier separated the repository's shipped calibration split cleanly for that model. It does **not** prove that a specific eval prompt was memorized.

## Pre-cutoff vs post-cutoff split

| Model | Cutoff | Pre-cutoff acc (95% CI) | Post-cutoff acc (95% CI) | Gap |
| --- | --- | --- | --- | ---: |
| `openai/gpt-oss-20b` | 2024-06-30 | 0.560 [0.502–0.622] | 0.617 [0.500–0.750] | -0.057 |

On this run, `gpt-oss-20b` is slightly better on post-cutoff rows than on pre-cutoff rows.
The same run also shows large raw-feature effect sizes in `cohens_d.md` (`loss` d = +2.23, `zlib_ratio` d = +2.00, `min_k_pp` d = -2.15), so the classifier is seeing a real difference between the two buckets even though it does not show up as higher pre-cutoff directional accuracy on this eval stream.

## `cmmd` backtest benchmark

| Variant | Sharpe (95% CI) | Mean daily bps (95% CI) | Max drawdown | Total return | Signals used |
| --- | --- | --- | ---: | ---: | ---: |
| `raw_alpha` | -1.560 [-1.980, -0.622] | -1.14 [-1.66, -0.55] | -24.49% | -24.54% | 319 |
| `cmmd` | -1.360 [-1.791, -0.455] | -0.97 [-1.48, -0.40] | -21.30% | -21.36% | 255 |

Additional run-level facts for `runs/cmmd_20260429T070616Z`:

| Metric | Value |
| --- | --- |
| Signal model | `openai/gpt-oss-20b` |
| MCS holdout AUC | 0.728 |
| Parse rate | 96.7% |
| `p_memorized` threshold | 0.1598 |
| Relative Sharpe change `(cmmd - raw_alpha) / raw_alpha` | -12.87% |

`cmmd` still loses money on this stream, but less badly than `raw_alpha`. In this repository's single-model thresholded variant, filtering high-score rows changes the portfolio, but it does not make the strategy profitable on this universe and date span.

## Metric explanations

| Metric | Meaning |
| --- | --- |
| **Parse %** | Fraction of model responses that parsed into a usable directional prediction. Lower values mean more rows were dropped before scoring. |
| **Raw Acc** | Share of parse-OK rows where `predicted_direction == target_direction`, with a bootstrap 95% confidence interval. |
| **MemGuard Acc** | Accuracy after the confidence penalty is computed. In this spec it matches Raw Acc because the penalty rescales confidence, not the predicted class. |
| **MCS-AUC** | Area under the ROC curve for the per-model classifier on held-out calibration data. Higher values mean the classifier separated the repository's shipped in-sample and out-of-sample calibration corpora more cleanly. |
| **Score** | Composite ranking score: `memguard_acc_lo * mcs_auc_point * parse_success_rate`. Models that fail a gate are zeroed out. |
| **Survives gates** | Whether the model cleared the parse-rate, calibration-quality, and majority-baseline gates. Only survivors can enter the top-3 ranking. |
| **Warnings** | Gate or quality flags such as `parse-unreliable`, `weak-calibration`, or `not-better-than-baseline`. |
| **Pre-cutoff / Post-cutoff acc** | Accuracy split by whether the eval row date is on/before or after the model's documented training cutoff. This is descriptive, not proof of memorization by itself. |
| **Gap** | `pre-cutoff accuracy - post-cutoff accuracy`. A positive value can be a memorization clue; a negative or near-zero value means the split does not show that pattern on this run. |
| **Sharpe** | Annualized Sharpe ratio of the backtest variant. Higher is better; negative values mean the strategy lost return relative to its volatility. |
| **Mean daily bps** | Mean daily return in basis points. 1 bp = 0.01%. |
| **Max drawdown** | Largest peak-to-trough loss over the backtest. |
| **Total return** | End-to-end portfolio return over the backtest period. |
| **Signals used** | Number of parse-OK rows that reached the backtest variant after any filtering. |
| **`p_memorized` threshold** | Quantile cutoff used by the `cmmd` variant for that run. Rows above it are excluded. |

## Source files

- Main benchmark summary in the docs site: `docs/latest-run.md`
- Overview and workflow docs: `README.md`
- Raw local artifacts referenced above: `runs/20260429T073229Z/` and `runs/cmmd_20260429T070616Z/`
