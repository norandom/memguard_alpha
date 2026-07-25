# Recall Guard

Two CLIs for scoring language-model responses on a financial prompt stream and attaching a per-prompt contamination score.

- `./start.sh` runs the **recall-guard check**: per-model accuracy, MIA features, MCS-AUC, and a top-3 ranking with bootstrap CIs.
- `scripts/run_cmmd_backtest.py` runs the repository's **`cmmd` backtest**: one model's signal stream goes through a long-short portfolio over a 10-year window, with and without dropping the rows whose `p_memorized` score lands in the top quintile for that run.

The recall-guard check is the per-model leaderboard. The backtest then asks a narrower question: on this prompt stream, does filtering high-score rows change the downstream portfolio at all? Both workflows share the same harness path, so the artifacts stay comparable.

## What this measures

Recall Guard is not a forecaster. On a "did this ETF close up or down" task, near-coin-flip directional accuracy is an unsurprising result. The useful output here is not alpha. It is the per-prompt `p_memorized` score and the supporting artifacts around it.

In this repository, `p_memorized` is a model-specific score produced by a logistic classifier trained to separate the shipped in-sample and out-of-sample calibration corpora. It is useful as a contamination flag for downstream analysis. It is not proof that a prompt was memorized, and it should be read in the context of the calibration data used to train it.

## Install (users)

Recall Guard ships as a package on GitHub Releases (no PyPI). Pin a release tag; Python 3.12 or newer:

```bash
uv add "recall-guard @ git+https://github.com/norandom/memguard_alpha.git@v0.1.1"
# or, with pip: install the wheel attached to the release
pip install https://github.com/norandom/memguard_alpha/releases/download/v0.1.1/recall_guard-0.1.1-py3-none-any.whl
```

Then `from recall_guard import MemoryGuardedScorer` and bring your own NVIDIA API key at calibration time. The runtime dependency set is lean (`numpy`, `scikit-learn`, `rich`, `pyyaml`, `requests`, `python-dotenv`); plotting and backtest extras are opt-in (`recall-guard[backtest]`).

## Develop (contributors)

You need Python 3.12, [uv](https://github.com/astral-sh/uv), an NVIDIA chat-completions API key, and an FMP API key.

```bash
uv sync
cat > .env <<EOF
NVIDIA_API_KEY=...
FMP_API_KEY=...
EOF
uv run pytest -q   # 258 tests, offline
```

`uv sync` installs everything from `pyproject.toml` into `.venv/`, including the dev group.

## Workflow 1: run the recall-guard check

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

The default `--shortlist` path writes four files to `OUT_DIR` (defaults to `runs/<UTC-timestamp>/`):

| File | What it is |
| --- | --- |
| `top3.md` | The ranking, plus a "why fewer than three" section when gates remove models. |
| `summary.csv` | Per-model raw acc, MemGuard acc, MCS-AUC, parse rate, score, and warnings. |
| `records.jsonl` | One row per `(model, prompt)` with parsed output, derived MIA features, standardized features, and `p_memorized`. |
| `manifest.json` | Seed, input file hashes, artifact paths, and run metadata. |

If you use `--candidates` instead of `--shortlist`, the harness also writes `shortlist.json`.

To open the figures:

```bash
uv run jupyter-lab notebooks/visualize_run.ipynb
```

Point the `RUN_DIR` cell at your run and execute. The notebook reconstructs the figures from the artifacts on disk.

## Workflow 2: run the `cmmd` backtest

```bash
uv run python scripts/run_cmmd_backtest.py
```

This is the end-to-end pipeline:

1. Build `data/eval/etf_portfolio.jsonl` if it is missing (10-year window, three risk tickers × 110 prompts).
2. Run the harness on `openai/gpt-oss-20b` against that eval set.
3. Compute per-(model, MIA-feature) Cohen's d.
4. Run `scripts/analyze_is_oos_gap.py` for the pre-cutoff vs post-cutoff accuracy gap.
5. Pull EOD prices for `{SWDA.L, XLK, IAU, BIL}` from FMP for the eval-set date span.
6. Run two backtests on the same prices: `raw_alpha` (every parse-OK row) and `cmmd` (rows above the run's top-quintile `p_memorized` threshold dropped). Daily rebalance, 1× leverage cap, 15 bps round-trip cost.
7. Write artifacts and extend `manifest.json` with the `backtest` block.

You get this in the run dir (default `runs/cmmd_<UTC-timestamp>/`):

| File | What it is |
| --- | --- |
| `records.jsonl`, `summary.csv`, `top3.md`, `manifest.json` | Standard harness artifacts for the single-model `gpt-oss-20b` run. The manifest also carries a `backtest` block with the universe, quantile/threshold, fees, seed, and artifact paths. |
| `cohens_d.csv`, `cohens_d.md` | Per-feature Cohen's d on the raw MIA values, split by the cutoff-based date buckets. |
| `is_oos_gap.csv`, `is_oos_gap.md` | Per-model pre-cutoff / post-cutoff accuracy with bootstrap CIs and the gap. |
| `backtest_summary.csv`, `backtest_summary.md` | Sharpe, mean daily bps, max drawdown, total return for both variants, with bootstrap CIs and the relative Sharpe change. |
| `equity_curves.csv`, `equity_curves.png` | Daily equity for `raw_alpha`, `cmmd`, and a passive `buy_and_hold_swda` benchmark, normalized to 1.0 at the first trading day. |
| `daily_returns.csv` | Daily returns in basis points for `raw_alpha` and `cmmd`. |

If the run aborts before writing backtest artifacts:

| Exit | Reason |
| --- | --- |
| 4 | MCS calibration failed for `gpt-oss-20b`. The harness writes the warning to `summary.csv`; the orchestrator does not add a misleading `backtest` block on top of that. |
| 5 | `analyze_is_oos_gap.py` preconditions failed (`records.jsonl` missing, cutoffs missing, or no eval rows have an ISO date). |
| 6 | `BacktestArtifactError` on the artifact write phase (disk full, permission denied). The run dir is rolled back to the pre-write state. |
| 7 | `PriceFetchError` from FMP (HTTP error, or fewer than 30 aligned trading days across the 4-ticker inner join). |

Re-run the script any time. A fixed seed keeps the local sampling and bootstrap steps stable for a fixed input set. Hosted model responses, provider changes, and upstream price-data corrections can still move the resulting numbers.

## What "10-year window" means

The eval-set builder samples trading days from `[today − 10 years, today]`, stratified across the `gpt-oss-20b` cutoff (`2024-06-30`) so both halves are populated. `scripts/build_etf_portfolio_eval.py` recomputes the start date at import time, so re-running the orchestrator next month picks up the trailing decade automatically. The builder and the price fetcher both pass explicit `from` / `to` dates to FMP.

## Sample run

The examples below come from two local runs from 2026-04-29 against the same 330-prompt, 10-year `SWDA.L / XLK / IAU` eval set:

- `runs/20260429T073229Z/`: recall-guard check, four-model shortlist.
- `runs/cmmd_20260429T070616Z/`: single-model `cmmd` backtest (`gpt-oss-20b`, MCS holdout AUC 0.728, parse rate 96.7%).

These paths are local run artifacts, not tracked repository files. Re-run the commands above to regenerate equivalents in your own checkout.

### Per-model accuracy (recall-guard check)

| Model | Parse % | Raw Acc (95% CI) | MCS-AUC | Warnings |
| --- | ---: | --- | ---: | --- |
| `openai/gpt-oss-20b`            | 95.5% | **0.5492** [0.4921–0.6032] | 0.976 | not-better-than-baseline |
| `meta/llama-3.2-3b-instruct`    | 99.1% | 0.5352 [0.4801–0.5902]     | 0.996 | not-better-than-baseline |
| `meta/llama-3.1-8b-instruct`    | 86.7% | 0.4860 [0.4301–0.5420]     | 1.000 | not-better-than-baseline |
| `microsoft/phi-4-mini-instruct` | 48.5% | **0.3187** [0.2437–0.3875] | 0.854 | parse-unreliable, not-better-than-baseline |
| `__majority_baseline__`         | —     | 0.5394 [0.4848–0.5970]     | —     | always-predict-the-majority-class |

Notes:

- `gpt-oss-20b` is the only model whose point estimate lands above the always-predict-up baseline (0.5492 vs 0.5394), but its interval still overlaps the baseline interval.
- `phi-4-mini` performs poorly here, and the low parse rate means the estimate is based on a smaller usable subset than the other models.
- MCS-AUC near 1.0 means the classifier cleanly separated the repository's shipped IS and OOS calibration corpora for that model. It does **not** by itself prove that the model memorized a specific eval prompt.

### Pre-cutoff vs post-cutoff split

`openai/gpt-oss-20b` shows a negative gap in this run: pre-cutoff accuracy is lower than post-cutoff accuracy.

| Model | Cutoff | Pre-cutoff acc (95% CI) | Post-cutoff acc (95% CI) | Gap |
| --- | --- | --- | --- | ---: |
| `openai/gpt-oss-20b` | 2024-06-30 | 0.560 [0.502–0.622] | 0.617 [0.500–0.750] | **−0.057** |

The same run also shows large effect sizes on the raw MIA features (`loss` d = +2.23, `zlib_ratio` d = +2.00, `min_k_pp` d = −2.15), so the classifier is picking up a real difference between the two calibration buckets. On this eval stream, that difference does not show up as higher pre-cutoff directional accuracy.

### Backtest

10-year long-short portfolio over `{SWDA.L, XLK, IAU}` with BIL as the cash leg, daily rebalance, 1× leverage cap, 15 bps round-trip cost. `raw_alpha` uses every parse-OK row; `cmmd` drops rows above the run's top-quintile `p_memorized` threshold (empirical threshold 0.1598 on this run). Bootstrap 95% CIs from 1000 resamples.

| Variant | Sharpe (95% CI) | Mean daily bps (95% CI) | Max drawdown | Total return | n signals |
| --- | --- | --- | ---: | ---: | ---: |
| `raw_alpha` | −1.560 [−1.980, −0.622] | −1.14 [−1.66, −0.55] | −24.49% | −24.54% | 319 |
| `cmmd`      | −1.360 [−1.791, −0.455] | −0.97 [−1.48, −0.40] | −21.30% | −21.36% | 255 |

Relative Sharpe change `(cmmd − raw_alpha) / raw_alpha`: **−12.87%**. The filtered variant is still losing money, but less badly than `raw_alpha` on this run.

This repository's single-model thresholded variant does not turn the strategy profitable on this universe and date span. It is evidence about this implementation, this model, and this prompt stream; it is not a claim about every CMMD-style setup.

Equity curves for both variants plus the passive `buy_and_hold_swda` benchmark are saved in the run dir as `equity_curves.png`. Re-run `scripts/run_cmmd_backtest.py` to regenerate them locally.

## Use as a package

The longer-term use case is to consume Recall Guard as a library rather than only through the two CLIs. [Global_Macro_AI_Factors](https://github.com/norandom/Global_Macro_AI_Factors) is one consumer: it builds AI macro/risk factors and can feed prompts through `recall_guard` to get a per-call `p_memorized` score and a discounted confidence.

Since v0.1.1 this is packaged: `recall_guard` installs from the GitHub Release (wheel + sdist, hatchling build) with a small stable façade over the `NvidiaLM` + control-baseline + MCS stack. See [Install (users)](#install-users) above; the packaging spec lives under [`.kiro/specs/recall-guard-package/`](./.kiro/specs/recall-guard-package/).

## Caveats

The MCS classifier is only as good as the calibration corpora. The shipped IS corpus is small, and the shipped IS/OOS split is also confounded by source, formatting, and publication period. Treat `p_memorized` as an implementation-specific score, not ground truth.

NVIDIA's free-tier 70B endpoints are queue-heavy. Use `--min-call-interval 1.5` or higher for stability. The 8B–20B size range is a better starting point.

The eval set you ship to the harness defines what "skill" means. The bundled ETF builder asks a next-day direction question that leaves little room for genuine forecasting edge. On this task the harness is mainly useful for comparing contamination-related signals across models and prompts. If you want to study reasoning skill instead, ship an eval set that asks something the model can plausibly reason about; the contamination machinery is the same either way.

The `cmmd` backtest uses `gpt-oss-20b` only. Other models are evaluated by the recall-guard check but do not feed that portfolio run; the design keeps the comparison focused on one signal stream.

## Citation

Recall Guard is an independent implementation and evaluation of the MemGuard-Alpha method. The sample run above documents where this repository's implementation did and did not line up with the paper's reported behaviour.

Roy, A., & Roy, D. (2026). MemGuard-Alpha: Detecting and Filtering Memorization-Contaminated Signals in LLM-Based Financial Forecasting via Membership Inference and Cross-Model Disagreement. arXiv:2603.26797.

```bibtex
@misc{roy2026memguardalpha,
  title         = {MemGuard-Alpha: Detecting and Filtering Memorization-Contaminated
                   Signals in LLM-Based Financial Forecasting via Membership Inference
                   and Cross-Model Disagreement},
  author        = {Roy, Anisha and Roy, Dip},
  year          = {2026},
  eprint        = {2603.26797},
  archivePrefix = {arXiv},
}
```

The point-in-time evaluation problem this library addresses is also benchmarked independently by Look-Ahead-Bench:

> Benhenda, M. (2026). *Look-Ahead-Bench: a Standardized Benchmark of Look-ahead Bias in Point-in-Time LLMs for Finance.* arXiv:2601.13770.

```bibtex
@misc{benhenda2026lookaheadbench,
  title  = {Look-Ahead-Bench: a Standardized Benchmark of Look-ahead Bias in
            Point-in-Time LLMs for Finance},
  author = {Benhenda, Mostapha},
  year   = {2026},
  eprint = {2601.13770},
  archivePrefix = {arXiv},
  primaryClass = {cs.AI}
}
```

## See also

- [`Qualified_Models.md`](./Qualified_Models.md): per-model training-cutoff registry with sources.
- [`papers/2603.26797v1.md`](./papers/2603.26797v1.md): the MemGuard-Alpha paper.
- [`.kiro/specs/honest-model-ranking/`](./.kiro/specs/honest-model-ranking/): recall-guard spec.
- [`.kiro/specs/cmmd-backtest/`](./.kiro/specs/cmmd-backtest/): backtest spec.
- [`.kiro/specs/recall-guard-package/`](./.kiro/specs/recall-guard-package/): packaging spec.
