# Recall Guard

Two CLIs that grade language models on a financial-prediction task without giving them credit for what they memorised.

- `./start.sh` runs the **recall-guard check**: per-model accuracy, MIA features, MCS-AUC, and a top-3 ranking with bootstrap CIs.
- `scripts/run_cmmd_backtest.py` runs the **CMMD backtest**: turns one model's signal stream into a long-short portfolio over a 10-year window, with and without filtering the rows the MCS classifier flags as memorised.

The recall-guard check is the per-model leaderboard. The backtest takes the leaderboard's signal model and asks the harder question: does dropping the rows that look memorised actually improve trading Sharpe? Both run from the same harness path, so the numbers stay anchored to the same prompt stream.

## What this measures (and why coin-flip accuracy is the point)

Recall Guard is not a forecaster and is not trying to be. On a "did this ETF close up or down" task, near-coin-flip directional accuracy is the **expected, correct** result: without financial modelling, an LLM's predictive quality on raw price direction cannot beat chance, and a tool that reported otherwise would be measuring leaked lookahead rather than skill. What the harness actually measures is honesty — whether a model's apparent edge comes from reasoning it can defend out-of-sample or from text it memorised before its training cutoff. The MIA features and the per-model MCS `p_memorized` turn that distinction into a number.

That number is the product. It is meant to be consumed downstream as a contamination / lookahead score on AI-derived signals — see [Use as a package](#use-as-a-package).

## Setup

You need Python 3.14, [uv](https://github.com/astral-sh/uv), an NVIDIA chat-completions API key, and an FMP API key.

```bash
uv sync
cat > .env <<EOF
NVIDIA_API_KEY=...
FMP_API_KEY=...
EOF
```

`uv sync` installs everything from `pyproject.toml` into `.venv/`. No system pip, no manual virtualenv.

## Workflow 1: Run the recall-guard check

```bash
./start.sh
```

Defaults to the 3-asset eval set (`SWDA.L / XLK / IAU`, 330 prompts), both calibration corpora, the cutoff registry, and a four-model shortlist (`llama-3.1-8b`, `llama-3.2-3b`, `gpt-oss-20b`, `phi-4-mini`). 8 parallel workers, 1.5 s pacing.

Override anything inline:

```bash
SHORTLIST="meta/llama-3.1-8b-instruct" ./start.sh
EVAL_SET=data/eval/etf_direction_multiyear.jsonl OUT_DIR=runs/quick ./start.sh
./start.sh --no-reference          # extra harness flags pass through
```

You get five files in `OUT_DIR` (defaults to `runs/<UTC-timestamp>/`):

| File | What it is |
| --- | --- |
| `top3.md` | The ranking, plus a "why fewer than three" section when gates kicked models out. |
| `summary.csv` | Per-model raw acc, MemGuard acc, MCS-AUC with 95% CIs, parse rate, score, warnings. |
| `records.jsonl` | One row per (model, prompt) with logprobs, the five MIA features, and `p_memorized`. |
| `manifest.json` | Seed, input file hashes, artifact paths. |
| `shortlist.json` | Only when the smoke gate ran (i.e., you used `--candidates` instead of `--shortlist`). |

To open the figures:

```bash
uv run jupyter-lab notebooks/visualize_run.ipynb
```

Point the `RUN_DIR` cell at your run, execute. You get MIA-feature distributions, MCS calibration curves, accuracy and MCS-AUC with bootstrap CIs, and the composite ranking, all read straight from the artifacts.

## Workflow 2: Run the CMMD backtest

```bash
uv run python scripts/run_cmmd_backtest.py
```

This is the end-to-end pipeline:

1. Build `data/eval/etf_portfolio.jsonl` if it is missing (10-year window, three risk tickers × 110 prompts).
2. Run the harness on `openai/gpt-oss-20b` against that eval set.
3. Compute per-(model, MIA-feature) Cohen's d.
4. Run `scripts/analyze_is_oos_gap.py` for the IS-vs-OOS accuracy gap.
5. Pull EOD prices for `{SWDA.L, XLK, IAU, BIL}` from FMP for the eval-set's date span.
6. Run two backtests on the same prices: `raw_alpha` (every parse-OK row) and `cmmd` (top-quintile `p_memorized` rows dropped). Daily rebalance, 1× leverage cap, 15 bps round-trip cost.
7. Write artifacts and extend `manifest.json` with the `backtest` block.

You get this in the run dir (default `runs/cmmd_<UTC-timestamp>/`):

| File | What it is |
| --- | --- |
| `records.jsonl`, `summary.csv`, `top3.md`, `manifest.json` | Standard harness artifacts (single-model run for `gpt-oss-20b`). The manifest now carries a `backtest` block with the universe, cmmd quantile + threshold, fees, seed, n_is_rows / n_oos_rows, and artifact paths. |
| `cohens_d.csv`, `cohens_d.md` | Per-feature Cohen's d on the raw MIA values, IS vs OOS, with the model's holdout MCS-AUC for context. |
| `is_oos_gap.csv`, `is_oos_gap.md` | Per-model IS / OOS accuracy with bootstrap 95% CIs and the gap. |
| `backtest_summary.csv`, `backtest_summary.md` | Sharpe, mean daily bps, max drawdown, total return for both variants, with bootstrap CIs and the relative Sharpe improvement. |
| `equity_curves.csv`, `equity_curves.png` | Daily equity for `raw_alpha`, `cmmd`, and a passive `buy_and_hold_swda` benchmark, normalised to 1.0 at the first trading day. |
| `daily_returns.csv` | Daily returns in basis points for `raw_alpha` and `cmmd`. |

If the run aborts before writing artifacts:

| Exit | Reason |
| --- | --- |
| 4 | MCS calibration failed for `gpt-oss-20b`. The harness writes the warning to `summary.csv`; the orchestrator refuses to write a misleading `backtest` block on top of it. |
| 5 | `analyze_is_oos_gap.py` preconditions failed (records.jsonl missing, cutoffs missing, or no eval rows have an ISO date). |
| 6 | `BacktestArtifactError` on the artifact write phase (disk full, permission denied). The run dir is rolled back to the pre-write state. |
| 7 | `PriceFetchError` from FMP (HTTP error, or fewer than 30 aligned trading days across the 4-ticker inner-join). |

Re-run the script any time. The eval set is deterministic for a fixed seed (default 0), so re-runs reproduce the same prompt stream and, given the same model and prices, the same numbers within bootstrap noise.

## What "10-year window" means

The eval-set builder samples trading days from `[today − 10 years, today]`, stratified across the gpt-oss-20b cutoff (2024-06-30) so both halves are populated. `scripts/build_etf_portfolio_eval.py` recomputes the start date at import time, so re-running the orchestrator next month picks up the trailing decade automatically. FMP's `historical-price-eod/light` endpoint caps history at 5 years unless `from`/`to` are supplied; both the eval-set builder and `portfolio.prices.fetch_universe_prices` pass them so the full 10 years comes back.

## Sample run

Two reference runs from 2026-04-29 against the same 330-prompt, 10-year SWDA.L / XLK / IAU eval set:

- `runs/20260429T073229Z/` — recall-guard check, four-model shortlist.
- `runs/cmmd_20260429T070616Z/` — CMMD backtest, single signal model (`gpt-oss-20b`, MCS holdout AUC 0.728, parse rate 96.7%).

### Per-model accuracy (recall-guard check)

| Model | Parse % | Raw Acc (95% CI) | MCS-AUC | Warnings |
| --- | ---: | --- | ---: | --- |
| `openai/gpt-oss-20b`            | 95.5% | **0.5492** [0.4921–0.6032] | 0.976 | not-better-than-baseline |
| `meta/llama-3.2-3b-instruct`    | 99.1% | 0.5352 [0.4801–0.5902]     | 0.996 | not-better-than-baseline |
| `meta/llama-3.1-8b-instruct`    | 86.7% | 0.4860 [0.4301–0.5420]     | 1.000 | not-better-than-baseline |
| `microsoft/phi-4-mini-instruct` | 48.5% | **0.3187** [0.2437–0.3875] | 0.854 | parse-unreliable, not-better-than-baseline |
| `__majority_baseline__`         | —     | 0.5394 [0.4848–0.5970]     | —     | always-predict-the-majority-class |

Notes worth keeping:

- `gpt-oss-20b` is the only model whose point estimate beats the always-predict-up baseline (0.5492 vs 0.5394). The 95% CIs overlap, so it is not a "p<0.05" winner — but it sits where MemGuard-Alpha's own models live (40–52% directional accuracy across contamination quintiles).
- `phi-4-mini` at 0.3187 is statistically anti-skilled. Its CI [0.244, 0.388] does not overlap the baseline's lower bound (0.485). The parse-unreliable warning at 48.5% means the estimate comes from a smaller-than-usual sample, but the direction is real; on a three-class task this reads as a flipped +1/−1 mapping rather than noise.
- MCS-AUC near 1.0 for the three larger models means the calibrator can almost perfectly separate IS-memorised text from OOS-control text. `phi-4-mini` at 0.854 is the outlier — its log-probability signature is harder to discriminate.

### IS-vs-OOS memorization gap

| Model | Cutoff | n_IS | n_OOS | IS Acc (95% CI) | OOS Acc (95% CI) | Gap |
| --- | --- | ---: | ---: | --- | --- | ---: |
| `openai/gpt-oss-20b` | 2024-06-30 | 259 | 60 | 0.560 [0.502–0.622] | 0.617 [0.500–0.750] | **−0.057** |

`n_IS + n_OOS = 319`, matching the parse-OK row count in `records.jsonl`. 11 rows parse-failed and are excluded.

> _MemGuard-Alpha Section 5.3 reports IS accuracy for ChatGPT rising 40.8 → 52.5% and OOS accuracy falling 47 → 42% over the same evaluation. A large positive gap is the memorisation signature; a small or zero gap with healthy OOS accuracy is the desired honest behaviour._

The gap is again *negative* — gpt-oss-20b does slightly better on rows it could not have memorised than on rows it could. The Cohen's d artifact in the same run dir reports large effect sizes on the raw MIA features (`loss` d = +2.23, `zlib_ratio` d = +2.00, `min_k_pp` d = −2.15), so the calibrator is detecting something MIA-shaped — just not the directional-accuracy gap the paper highlights. See `cohens_d.md` for the full feature breakdown.

### Backtest

10-year long-short portfolio over `{SWDA.L, XLK, IAU}` with BIL as the cash leg, daily rebalance, 1× leverage cap, 15 bps round-trip cost. `raw_alpha` uses every parse-OK row; `cmmd` drops the top quintile by `p_memorized` (empirical threshold 0.1598 on this run). Bootstrap 95% CIs from 1000 resamples.

| Variant | Sharpe (95% CI) | Mean daily bps (95% CI) | Max drawdown | Total return | n signals |
| --- | --- | --- | ---: | ---: | ---: |
| `raw_alpha` | −1.560 [−1.980, −0.622] | −1.14 [−1.66, −0.55] | −24.49% | −24.54% | 319 |
| `cmmd`      | −1.360 [−1.791, −0.455] | −0.97 [−1.48, −0.40] | −21.30% | −21.36% | 255 |

Relative Sharpe improvement `(cmmd − raw_alpha) / raw_alpha`: **−12.87%** — the CMMD-filtered Sharpe is *less negative* than `raw_alpha`'s. Both variants are net-losing on this universe and date span, so the "improvement" is a smaller loss rather than alpha. The paper's headline finding (CMMD lifts a borderline-positive Sharpe further into the green) does not reproduce on a single-model gpt-oss-20b run against three liquid ETFs over ten years; consistent with the negative IS−OOS gap above.

Equity curves for both variants plus the passive `buy_and_hold_swda` benchmark are saved to `runs/cmmd_20260429T070616Z/equity_curves.png`. Re-run `scripts/run_cmmd_backtest.py` to regenerate.

This non-result is the expected outcome, not a failure of the method: a directional ETF signal with no financial modelling behind it has no honest alpha to harvest, so filtering memorised rows cannot manufacture one. The method is working — it is declining to invent skill that is not there.

## Use as a package

The longer-term goal is to consume Recall Guard as a library, not only as two CLIs. [Global_Macro_AI_Factors](https://github.com/norandom/Global_Macro_AI_Factors) builds AI macro/risk factors; its Track A is a DSPy agent that emits Black-Litterman views from anonymised, z-scored macro state and deliberately never sees a date, a year, or a real ticker. That is recall-avoidance enforced by construction. Recall Guard supplies the missing half: a *measured* `p_memorized` per prompt, derived from per-token logprobs that DSPy hides, so contamination becomes an observable instead of an assumption. The inference is honest by the same mechanism the leaderboard uses; the factor pipeline downstream decides what to do with the score.

Packaging this cleanly — installable as `recall_guard`, importable on Python 3.12, behind a small stable façade over the `NvidiaLM` + control-baseline + MCS stack — is specced under [`.kiro/specs/recall-guard-package/`](./.kiro/specs/recall-guard-package/). Until that lands, import from the `src.*` layout in an editable checkout.

## Caveats

The MCS classifier is only as good as the calibration corpora. The shipped IS corpus has 40 rows from 2020 and 2023 (FMP's older archive is thin); OOS has 100. If a model's MCS-AUC falls below 0.6 the harness flags it `weak-calibration` and drops it from the top-3 list. Check `summary.csv` after each run.

NVIDIA's free-tier 70B endpoints are queue-heavy. Use `--min-call-interval 1.5` or higher for stability. The 8B–20B size range is a better starting point.

The eval set you ship to the harness defines what "skill" means. The bundled ETF builder asks "did this ETF close higher or lower than the previous trading day," which is close to a coin flip even for an oracle — and on this task that is the correct answer, not a defect (see [What this measures](#what-this-measures-and-why-coin-flip-accuracy-is-the-point)). The harness is measuring memorisation honesty on that stream, not trying to win it. If you want to study reasoning skill instead, ship an eval set that asks something a model can actually reason about (read this earnings report, predict the reaction); the memorisation machinery is identical either way.

The CMMD backtest uses gpt-oss-20b only. Other models are evaluated by the recall-guard check but do not feed the backtest; the design keeps the comparison "raw vs cmmd on the same signal stream" rather than "model A vs model B".

## See also

- [`Qualified_Models.md`](./Qualified_Models.md) — per-model training-cutoff registry with sources.
- [`papers/2603.26797v1.md`](./papers/2603.26797v1.md) — the MemGuard-Alpha paper.
- [`.kiro/specs/honest-model-ranking/`](./.kiro/specs/honest-model-ranking/) — recall-guard spec.
- [`.kiro/specs/cmmd-backtest/`](./.kiro/specs/cmmd-backtest/) — backtest spec.
- [`.kiro/specs/recall-guard-package/`](./.kiro/specs/recall-guard-package/) — packaging spec (use as a library for AI macro-factor analysis).
