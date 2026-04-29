# cmmd-backtest — Requirements

## Introduction

This spec adds a portfolio backtest on top of the `honest-model-ranking`
harness. The goal is to take the per-model accuracy numbers we already
produce and run them through a long-short trading strategy, with and
without an MCS-based filter, so we can see whether dropping the rows
the model probably memorised actually moves Sharpe. The harness can't
make that comparison today, and it is the paper's headline finding.

The backtest reuses the harness signal path unchanged: same MIA
features, same calibrator, same parser. Recall avoidance comes from
the calibrated `p_memorized` per row. Not a date filter, not a second
model.

Two smaller artifacts come along for the ride because the harness has
the data and the paper relies on them: per-feature Cohen's d (paper
Table 3) and the IS-vs-OOS accuracy gap reported in the README
sample-run section (paper Figure 5).

## Boundary Context

- **In scope**: per-(model, MIA-feature) Cohen's d artifact; IS/OOS-gap
  numbers in the README; eval-set builder for the new universe (SWDA.L,
  XLK, IAU); long-short cross-sectional backtest with daily rebalance;
  raw-alpha and CMMD-filtered variants on the same signal stream;
  output artifacts (Sharpe table, equity curves, daily-returns CSV);
  BIL used as the cash leg.
- **Out of scope**: multi-LLM ensembles (single model); MIA detectors
  outside MCS; intraday rebalance; futures or options overlays; tax
  accounting; live trading; signals for BIL (BIL holds residual cash,
  the model is never asked to predict it).
- **Adjacent expectations**:
  - Reuses `harness/`, `mia/`, `dataset/`, `core/` from the
    `honest-model-ranking` spec without changes.
  - Reuses `data/cutoffs.yaml`. The backtest deliberately accepts
    IS-dated rows because filtering them is the whole feature, so the
    cutoff guard does not run on the backtest's eval set.
  - The single signal model is `openai/gpt-oss-20b`. Its documented
    training cutoff is 2024-06-30, so the eval set has to straddle
    that date to give CMMD anything to filter.

## Requirements

### Requirement 1: Per-feature Cohen's d artifact

**Objective:** As a researcher comparing the harness's MIA separation to
MemGuard-Alpha Table 3, I want a per-(model, MIA-feature) Cohen's d
breakdown written to each run directory, so that the per-feature claims
in the paper become directly auditable against my own runs.

#### Acceptance Criteria

1. When a harness run completes, the cmmd-backtest analyser shall write
   `cohens_d.csv` and `cohens_d.md` into the run directory.
2. When the analyser computes Cohen's d for a feature, it shall use the
   IS-memorized vs OOS-control distributions of that feature's raw
   (non-standardised) value, with `pooled_std` as the denominator.
3. When a feature's IS distribution or OOS distribution has fewer than
   two valid samples for a given model, the analyser shall record
   `Cohen_d = NaN` and a `note = "insufficient samples"` for that row
   instead of failing the artifact.
4. The analyser shall report Cohen's d for each of the five MIA
   features (Loss, Min-K%, Min-K%++, zlib ratio, ref-delta) for every
   model in the run.
5. The artifact shall include each model's combined MCS-AUC alongside
   the per-feature numbers so a reader can see how much the composite
   gains over the best single feature.

### Requirement 2: IS-vs-OOS gap surfaced in README

**Objective:** As a researcher reading the README, I want the per-model
IS-vs-OOS accuracy crossover stated alongside the sample-run accuracy
table, so that the paper's headline finding (Figure 5) and my run's
findings live on the same page.

#### Acceptance Criteria

1. When a harness run completes, `scripts/analyze_is_oos_gap.py` shall
   produce `is_oos_gap.csv` and `is_oos_gap.md` in the run directory
   without manual intervention.
2. The gap analyser shall report per-model `IS_acc`, `OOS_acc`, and
   `gap = IS_acc - OOS_acc` with bootstrap 95% CIs on each accuracy
   estimate.
3. If a model has fewer than ten parse-OK IS rows or ten parse-OK OOS
   rows, the analyser shall mark its gap as `unreliable` and emit a
   note rather than a numeric estimate.
4. The README "Sample run" section shall include a table of per-model
   IS-vs-OOS gap numbers immediately after the headline accuracy table.
5. The README sample-run text shall explicitly cite MemGuard-Alpha
   Section 5.3's IS 40.8 → 52.5% / OOS 47 → 42% finding so the reader
   can compare.

### Requirement 3: Three-asset eval-set builder

**Objective:** As an operator running the backtest, I want a deterministic
eval-set builder that emits prompts for three risk assets across a date
range that straddles gpt-oss-20b's training cutoff, so the backtest has
both IS and OOS rows to filter and the run is reproducible.

#### Acceptance Criteria

1. The eval-set builder shall emit prompts for `SWDA.L`, `XLK`, and
   `IAU` only; it shall NOT emit prompts for `BIL`.
2. The eval-set builder shall sample at least 100 distinct trading
   days from the trailing 10-year window (`[today − 10y, today]`),
   producing at least 300 prompts total (100 days × 3 tickers).
3. The eval-set builder shall draw both pre-2024-07-01 and
   post-2024-07-01 trading days so the resulting eval set contains
   both IS and OOS rows for `openai/gpt-oss-20b`.
4. The eval-set builder shall use a fixed random seed and shall print
   the seed in its stdout summary so a re-run with the same seed
   produces an identical file.
5. If FMP returns fewer than 100 valid trading days for any of the
   three tickers in the window, the builder shall fail with a clear
   error message identifying the under-supplied ticker.
6. The eval-set builder shall persist `metadata.ticker` and
   `metadata.date` on every row so the backtest layer can group
   signals by date and ticker without re-deriving them.

### Requirement 4: Single-model signal generation

**Objective:** As a researcher, I want the cmmd-backtest pipeline to use
exactly one LLM (gpt-oss-20b) and to generate every signal through the
existing harness path, so that the backtest's inputs cannot diverge from
what `summary.csv` reports.

#### Acceptance Criteria

1. The cmmd-backtest signal generator shall route every prompt through
   the existing `evaluate_model` function so the parser, MIA features,
   MCS, and `p_memorized` per row are identical to a normal harness run.
2. The cmmd-backtest pipeline shall not retry an LLM call that already
   produced a parseable response; a parse failure remains a parse
   failure for both the table and the backtest.
3. Where MCS calibration fails for the model (uncalibrated baseline or
   AUC below the harness's threshold), the cmmd-backtest pipeline shall
   abort with a clear message and shall NOT run any backtest.
4. When the model emits a parse-OK row, the cmmd-backtest pipeline
   shall record the row's `direction`, `confidence`, and
   `p_memorized` keyed by `(date, ticker)`.
5. The cmmd-backtest pipeline shall record but NOT modify the row's
   `direction`; MCS only decides whether the row is allowed into the
   filtered backtest, never what direction the row carries.

### Requirement 5: Long-short cross-sectional backtest

**Objective:** As a researcher, I want a daily-rebalanced long-short
backtest over the three risk assets with a cash leg, so that the
harness's per-row signals translate into a portfolio metric directly
comparable to MemGuard-Alpha Table 4.

#### Acceptance Criteria

1. The backtest engine shall rebalance once per trading day on the
   close, using the signals dated for that day.
2. When a row's parsed direction is `+1`, the backtest engine shall
   take a long position in that ticker for the next trading day; when
   `-1`, a short position; when `0`, no position.
3. The backtest engine shall set each ticker's per-day position weight
   to `direction × confidence`, so a row with `Direction: 1,
   Confidence: 0.9` weighs 9× a row with `Direction: 1,
   Confidence: 0.1`.
4. When the sum of absolute position weights across the universe on a
   given day is less than 1.0, the backtest engine shall allocate the
   residual capital `(1.0 - sum_abs_weights)` to `BIL` and shall earn
   `BIL`'s daily return on that residual.
5. When the sum of absolute position weights on a given day exceeds
   1.0, the backtest engine shall scale every position proportionally
   so the sum of absolute weights equals 1.0 (no leverage above 1×).
6. The backtest engine shall apply 15 basis points of transaction cost
   on the absolute change in each position weight from the previous
   trading day, charged to the next day's net return.
7. The backtest engine shall report annualised Sharpe, mean daily
   return in basis points, max drawdown, and total return for the
   strategy, alongside bootstrap 95% CIs on Sharpe and mean daily
   return.
8. If for a given trading day the eval set contains zero parse-OK
   rows for the universe, the backtest engine shall hold 100% in
   `BIL` for that day and continue.

### Requirement 6: CMMD filtering versus raw alpha

**Objective:** As a researcher, I want to see the same model's signal
stream traded twice — once raw, once with the top-MCS rows filtered —
so the Sharpe gain attributable to recall avoidance is directly visible.

#### Acceptance Criteria

1. The cmmd-backtest pipeline shall produce two side-by-side backtests
   from a single signal stream: one labelled `raw_alpha` (no
   filtering) and one labelled `cmmd` (filtered).
2. The CMMD filter shall drop rows whose `p_memorized` falls in the
   top quintile (top 20%) of the signal stream's empirical
   `p_memorized` distribution.
3. When a row is dropped by the CMMD filter, the cmmd-backtest engine
   shall treat that `(date, ticker)` slot as if no signal existed —
   the corresponding capital sits in BIL for that day.
4. The cmmd-backtest pipeline shall not regenerate the signal stream
   for the CMMD variant; both variants must consume the identical
   per-row records produced in Requirement 4.
5. The cmmd-backtest pipeline shall report Sharpe and mean daily return
   for both variants and shall compute the relative Sharpe improvement
   `(cmmd - raw) / raw` in the summary artifact.

### Requirement 7: Backtest output artifacts

**Objective:** As an operator, I want the backtest to write its results
to the run directory in formats the existing notebooks and the README
can consume, so the new layer fits the existing artifact discipline.

#### Acceptance Criteria

1. When a backtest completes, the cmmd-backtest engine shall write
   `backtest_summary.csv` and `backtest_summary.md` containing the
   side-by-side metrics for both variants.
2. The cmmd-backtest engine shall write `equity_curves.csv` containing
   per-day equity values for `raw_alpha`, `cmmd`, and a passive
   `buy_and_hold_swda` benchmark, all normalised to start at 1.0.
3. The cmmd-backtest engine shall write `daily_returns.csv` containing
   per-day net returns (in basis points) for each variant.
4. The cmmd-backtest engine shall write `equity_curves.png` rendering
   the same three series, with annotations for the largest drawdown
   on each variant.
5. Where the existing harness manifest is written, the cmmd-backtest
   engine shall extend it with a `backtest` block recording the
   universe, the signal model, the cost assumption, the filter
   quintile, and the artifact paths just enumerated.
6. If any backtest artifact cannot be written (disk full, permission
   denied), the cmmd-backtest engine shall fail with a non-zero exit
   code and shall NOT write a partially populated manifest.

### Requirement 8: Date safety and reproducibility

**Objective:** As a reviewer auditing the backtest, I want every run to
record enough provenance that I can re-derive its inputs and re-run it
deterministically, even when the eval set straddles the model's
training cutoff.

#### Acceptance Criteria

1. The cmmd-backtest pipeline shall accept eval-set rows whose dates
   precede the model's training cutoff (IS rows) without aborting,
   because filtering those rows IS the feature.
2. The cmmd-backtest pipeline shall record in the run manifest, for
   every row, whether the row was IS or OOS for the signal model.
3. The cmmd-backtest pipeline shall record the random seed used for
   bootstrap CIs and signal-generation parallelism, and re-running
   with the same seed and the same input files shall reproduce
   identical Sharpe and mean daily return numbers within numerical
   round-off.
4. When the eval set's date span does not include both IS and OOS
   rows (e.g., user supplied an OOS-only eval set), the
   cmmd-backtest pipeline shall log a warning that CMMD filtering
   will have nothing to remove and shall continue.

### Requirement 9: Failure handling

**Objective:** As an operator, I want the cmmd-backtest pipeline to
degrade gracefully when individual rows fail, rather than aborting an
otherwise-good run, but to abort cleanly when a structural problem
makes the result meaningless.

#### Acceptance Criteria

1. If a single signal row fails to parse, the cmmd-backtest engine
   shall drop that row from BOTH backtests and continue.
2. If FMP price data is missing for a `(date, ticker)` pair on a
   trading day where signals exist for that ticker, the
   cmmd-backtest engine shall drop the affected positions for that
   day and continue.
3. If MCS calibration fails for the signal model, the cmmd-backtest
   pipeline shall abort with a non-zero exit code BEFORE running
   either backtest.
4. If fewer than 30 parse-OK rows survive across the full backtest
   horizon, the cmmd-backtest engine shall write the artifacts but
   shall mark the summary with a `low-row-count` warning so the
   numbers are not over-interpreted.
