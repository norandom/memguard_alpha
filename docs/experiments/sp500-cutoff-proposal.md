# Proposed S&P 500 cutoff experiment

Date: 2026-11-01. Status: research proposal, not an executed measurement or live
request authorization.

## Decision this experiment should support

Determine whether named historical S&P 500 windows give a model an extra
prediction advantage before its documented cutoff. Use observed closes as ground
truth, then account for market regime, scale, missing outputs and temporal
dependence. This is a test of contamination-related behavior, not a trading
strategy or a demonstration of divergence mitigation.

We can measure variance, but raw price variance and repeated-answer variance are
not suitable primary signals. Price levels trend; return volatility changes.
A biased model can have small error variance. Measure errors against truth, their
bias and dispersion, and a paired identity advantage.

## Data and model conditions

Use the S&P 500 price-index daily close, not SPY, an adjusted ETF price or a
total-return series. FRED identifies its SP500 series as daily closing values and
offers ten years of history. Its notes also identify S&P ownership and
redistribution restrictions. Keep licensed/raw observations out of public result
bundles unless permitted; retain retrieval metadata, hashes and analysis code.
See [FRED SP500 notes](https://fred.stlouisfed.org/series/SP500).

Required data fields: observation date, index close, source, retrieval time,
source-series identity and file hash. Retain missing days as missing and use
actual trading observations; do not forward-fill weekends as additional outcomes.
Use the last completed, available observation as the data boundary.

| Target | Documented cutoff | Conservative cohort boundary |
| --- | --- | --- |
| openai/gpt-oss-20b through NIM | June 2024 | 2024-06-30 |
| claude-opus-5-5 through Anthropic | June 2026 | 2026-06-30 |

Sources checked 2026-11-01:
[GPT-OSS model card](https://deploymentsafety.openai.com/gpt-oss),
[Opus model documentation](https://platform.claude.com/docs/en/models/opus-5-5/overview).
The registry already uses these month-end boundaries. Vendor cutoff dates are
exposure proxies, not exact training-membership labels. Pin the requested model,
observed identity and settings and retain unknown effective settings.

Analyze each model against its own cutoff. The same calendar date can be
post-cutoff for GPT-OSS and pre-cutoff for Opus. This design does not support
ranking model abilities from unmatched cohorts.

## Three conditions per target date

| Condition | Input | Requested output | What it probes |
| --- | --- | --- | --- |
| Direct recall | S&P 500 identity and target trading date; no actual closes | That date's close, or explicit unknown | Date-specific numerical recall |
| Identified continuation | S&P 500 identity, real dates, preceding 20 closes rebased to start at 100 | Next rebased close, or unknown | Ability to use the numeric history plus identity |
| Anonymous continuation | The exact same rebased values, sequential indices instead of dates, no series identity | Next rebased close, or unknown | Ability to use the numeric history without explicit identity |

Both continuation arms use identical numeric values and formatting. Anonymization
must not also change numerical scale, precision or history length. No context
contains the target close, future return, realized outcome or future observation.
Each request is a fresh conversation; browsing, tools and retrieval are disabled.
Randomize arm order with a frozen seed. Label answers by date and arm, never by
completion order.

For a history starting at t-20, supply rebased observations through t-1:

    rebased_close_s = 100 * close_s / close_(t-20)

Convert the predicted rebased close to an implied one-day log return:

    predicted_return_t = log(predicted_rebased_close_t / rebased_close_(t-1))
    observed_return_t = log(close_t / close_(t-1))

Explicit dates may improve calendar reasoning without memorization. Price paths
can identify a series even when anonymized. These controls reduce confounds; they
do not establish a perfect counterfactual.

## Primary measurement and the proposed signal

Estimate volatility from the 19 observed one-day returns inside the supplied
20-close history. Choose a numerical floor before execution and expose it in the
plan; never estimate scale from future observations. Define normalized loss:

    loss_arm,t = abs(predicted_return_arm,t - observed_return_t) / trailing_volatility_t

Compare each model with the unchanged-close baseline, whose predicted return is
zero. Track signed error and squared error as secondary measures. Price-level
correlation or R-squared alone is misleading for persistent index levels.

The paired identity advantage is:

    identity_advantage_t = loss_anonymous,t - loss_identified,t

Positive values mean identifying the history helped. The proposed cohort
diagnostic is:

    cutoff_contrast = mean(identity_advantage_pre) - mean(identity_advantage_post)

A positive contrast is consistent with extra historical information helping
before cutoff. It is not automatically a memorization probability or a buy/sell
signal. Per-date advantages are noisy observations, not calibrated production
scores.

For direct recall, report exact matches at the source's published precision,
absolute relative error, abstentions and coverage. Freeze additional tolerance
bands before execution. Exact-date recall and forecast performance are separate
outcomes; do not combine them into a single unvalidated score.

## Sampling and confound checks

- Exclude the cutoff month and a further 30-calendar-day buffer on both sides.
  Use available post-cutoff history to set equally long pre/post windows. Opus
  currently has only a short post-cutoff period; do not claim a large study from it.
- Select dates before querying. Stratify on trailing volatility computed only
  from prior observations, and report balance and sample counts. Avoid selecting
  only famous crashes or extreme days.
- Reserve development and held-out target-date blocks. Do not tune thresholds,
  prompt wording or weights on held-out answers. Guard against adjacent-date
  overlap when defining splits.
- Retain overlapping input windows in the dependence record. Use calendar-block
  resampling for confidence intervals, not independent row bootstrap. Small
  numbers of blocks make intervals unreliable and must be reported.
- Evaluate placebo cutoff contrasts away from the documented boundary in a
  larger follow-up. A shared discontinuity across arbitrary dates suggests a
  market-regime or sampling explanation.
- An optional shifted-date arm could help distinguish calendar cues, but changes
  the information supplied. It requires a separate plan, interpretation and
  budget; it is not included in the first request count.
- No ensemble repetitions are required for this screening. Multiple distinct
  dates supply the observations. An ensemble experiment would be a separate arm.

## Missing outputs and stopping rules

Keep requested, completed, parsed, abstained and common-usable counts for each
cohort and arm. No reply is not a zero-error forecast. Primary paired diagnostics
use common-usable items and must be accompanied by coverage and missingness
sensitivity summaries; selective refusals may explain an apparent effect.

Stop on invalid credentials, billing rejection, quota responses or a frozen
request/deadline cap. Stop the scientific progression if outputs are not parseable
enough for paired analysis or the pilot has too few usable dates. Provider
availability failure is not evidence that the scientific hypothesis failed.

## Small pilot before a wider measurement

First prepare the source snapshot, date-selection manifest, prompts, parser and
offline scoring checks. Then present the exact request and cost budget.

Proposed mechanics pilot: four pre-cutoff and four post-cutoff dates per model,
three conditions per date, one response per condition. For two models this is
48 generations plus one eligibility request per model: maximum 50 physical
requests with no retries. It checks feasibility and coverage; it cannot establish
a reliable cutoff effect.

If that works, a separately approved expansion could use 16 dates per cohort
per model: 192 generations plus two eligibility requests, 194 total. This remains
exploratory; uncertainty and available independent blocks govern interpretation.
These are separate proposed budgets, not additions to earlier NIM allowances.

NIM remains paced below the user's 40 RPM ceiling: one worker, at least two
seconds between request starts, no hidden retries. Both providers retain their
own usage ledgers and bounded waiting. Token limits, prompts, costs and exact dates
must be frozen in the execution proposal. No requests are authorized by this page.

## Connection to divergence work

A detectable pre-cutoff identity advantage gives us a target behavior to test
against future mitigation. It does not construct an auxiliary next-token
distribution. Likewise, the existing FOMC lexical references should not be
assumed to correct daily index values.

The immediate sequence is: validate this measurement, establish whether there is
a repeatable unwanted signal, then evaluate an explicitly specified intervention.
If no useful signal appears, record that outcome before building mitigation
infrastructure. See the [experiment track](../research/experiment-track.md).
