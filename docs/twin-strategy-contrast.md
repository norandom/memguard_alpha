# The twin-strategy contrast: pricing look-ahead recall

This page describes the identification strategy behind the PIT vs non-PIT comparison used in the external consumer pipeline. The [PIT architecture](pit-architecture.md) page describes the five-layer stack; this page explains the twin design that sits around it.

The measurement logic is useful even when the exact run artifacts live outside this repository: compare two strategies that share the same pipeline and differ mainly in whether the prompt exposes identifying context.

## The two strategies

| | PIT deployable | Non-PIT diagnostic |
| --- | --- | --- |
| Role | Deployable prompt discipline | Measurement control; not deployable |
| Prompt | Anonymized: z-scored macro state, assets as `Asset_A–D` + category, no date, no tickers ("you do not know what year it is") | Same base text with date, real tickers, and raw macro levels inserted |
| Recall exposure | Reduced by prompt design | Deliberately reintroduced |
| Everything else | Shared: same model, rebalance dates, prices, optimizer stack, recall guard, and simulator | Shared |

The identifying prompt is produced by inserting the recall-enabling blocks into the anonymized prompt. That keeps the two variants close enough for a useful contrast, while still leaving the usual caveat that prompt length, formatting, and token-position effects can also change model behaviour.

Each twin runs the same monthly walk-forward loop: prompt → macro-axis loadings → Black-Litterman views → recall-guarded magnitude adjustment → HRP-CVaR base with cash pin → `w = 0.7·HRP + 0.3·BL` → target-percent rebalancing simulation (vectorbt).

## The measurement: contamination premium

The contrast harness pairs the twins date by date and reports the non-PIT minus PIT gap over the full stream rather than at a single date.

| Quantity | Meaning |
| --- | --- |
| `p_memorized_*_delta` | Gap in the repository's per-prompt contamination score between the identifying and anonymized variants of the same prompt date. |
| `p_memorized_paired_d` | Cohen's d of the paired per-date deltas. |
| `<metric>_delta` | Head-to-head gap for shared portfolio metrics such as return, volatility, Sharpe, drawdown, turnover, or timing diagnostics. |

Read the premium as a diagnostic contrast, not as deployable alpha. If the identifying twin outperforms the PIT twin, that extra performance belongs in the "prompt gave the model more context" bucket first; only more targeted controls can narrow that down further.

## Four evidence channels

The twin comparison is stronger when several channels move in the same direction:

1. **Logprob signature** — the per-call `p_memorized` score from the MIA features and per-model classifier.
2. **Realized performance** — both twins go through the same allocation and simulation pipeline, so their deltas show the downstream portfolio effect.
3. **Stated predictions** — a plain directional evaluation of what the model says, independent of the logprob-based score.
4. **The training-cutoff boundary** — extending both twins beyond the model's published training cutoff tests whether any premium survives on later dates.

Agreement across these channels is suggestive. It still does not collapse every alternative explanation into a single cause on its own.

## How `p_memorized` is determined, per call

Each inference call is scored individually:

1. The generation runs with per-token log-probabilities enabled.
2. Membership-inference features are computed on the reply's token stream: mean loss, Min-K%, Min-K%++, and the zlib ratio, with optional `ref_delta` when a reference model is used.
3. The features are standardized against a per-model baseline built from the out-of-sample control corpus.
4. A logistic-regression classifier trained on the repository's in-sample vs out-of-sample calibration split maps the vector to `p_memorized ∈ [0, 1]`.
5. The guard scales the view's exposure magnitude by that score.

In this repository, the score is always applied. `MCSCalibrator.is_weak` is reported as a quality flag, but the package does not automatically skip the discount when the calibrator is weak.

The model-selection screen is a separate one-time statistic computed before deployment. It selects which model to run; the per-call score determines how much to discount each output.

## Framing

The twin design is a way to turn a vague concern about look-ahead contamination into a repeatable contrast. It does not make the causal story trivial. What it does give you is a structured before/after comparison: same pipeline, same dates, same prices, different prompt exposure.
