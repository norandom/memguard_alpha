# Recall Guard

[![DOI](https://zenodo.org/badge/1221761081.svg)](https://doi.org/10.5281/zenodo.21557232)

Two CLIs for scoring language-model responses on a financial prompt stream and attaching a per-prompt contamination score.

- `./start.sh` runs the **recall-guard check**: per-model accuracy, MIA features, MCS-AUC, and a top-3 ranking with bootstrap CIs.
- `scripts/run_cmmd_backtest.py` runs the repository's **`cmmd` backtest**: one model's signal stream goes through a long-short portfolio over a 10-year window, with and without dropping the rows whose `p_memorized` score lands in the top quintile for that run.

The recall-guard check is the per-model leaderboard. The backtest then asks a narrower question: on this prompt stream, does filtering high-score rows change the downstream portfolio at all? Both workflows share the same harness path, so the artifacts stay comparable.

## What this measures

Recall Guard is not a forecaster. On a "did this ETF close up or down" task, near-coin-flip directional accuracy is an unsurprising result. The useful output here is not alpha. It is the per-prompt `p_memorized` score and the supporting artifacts around it.

In this repository, `p_memorized` is a model-specific score produced by a logistic classifier trained to separate the shipped in-sample and out-of-sample calibration corpora. It is useful as a contamination flag for downstream analysis. It is not proof that a prompt was memorized, and it should be read in the context of the calibration data used to train it.

## Install (users)

Recall Guard ships from its Git repository and as a wheel on GitHub Releases (no PyPI). Python 3.12 or newer. In a uv-managed project, declare it as a dependency against a released tag:

```bash
uv add "recall-guard @ git+https://github.com/norandom/memguard_alpha.git@v0.2.0"
# with the plotting/backtest extras:
uv add "recall-guard[backtest] @ git+https://github.com/norandom/memguard_alpha.git@v0.2.0"
```

Or install the release wheel into an environment of its own:

```bash
uv venv
uv pip install https://github.com/norandom/memguard_alpha/releases/download/v0.2.0/recall_guard-0.2.0-py3-none-any.whl
```

Then `from recall_guard import MemoryGuardedScorer` and bring your own NVIDIA API key at calibration time. The runtime dependency set is lean (`numpy`, `scikit-learn`, `rich`, `pyyaml`, `requests`, `python-dotenv`); plotting and backtest extras are opt-in (`recall-guard[backtest]`).

## Develop (contributors)

To work on the code, check out the repository and sync the development environment:

```bash
git clone https://github.com/norandom/memguard_alpha.git
cd memguard_alpha
uv sync
cat > .env <<EOF
NVIDIA_API_KEY=...
FMP_API_KEY=...
EOF
uv run pytest -q   # 311 tests, offline
```

You need Python 3.12 and [uv](https://github.com/astral-sh/uv). `uv sync` installs everything from `pyproject.toml` into `.venv/`, including the dev group.

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
6. Run two backtests on the same prices: `raw_alpha` (every parse-OK row) and `cmmd` (the top-quintile `p_memorized` slice dropped by rank). Daily rebalance, 1× leverage cap, 15 bps round-trip cost.
7. Write artifacts and extend `manifest.json` with the `backtest` block.

You get this in the run dir (default `runs/cmmd_<UTC-timestamp>/`):

| File | What it is |
| --- | --- |
| `records.jsonl`, `summary.csv`, `top3.md`, `manifest.json` | Standard harness artifacts for the single-model `gpt-oss-20b` run. The manifest also carries a `backtest` block with the universe, quantile/threshold, fees, seed, and artifact paths. |
| `cohens_d.csv`, `cohens_d.md` | Per-feature Cohen's d on the raw MIA values, split by the cutoff-based date buckets. |
| `is_oos_gap.csv`, `is_oos_gap.md` | Per-model pre-cutoff / post-cutoff accuracy with bootstrap CIs and the gap. |
| `backtest_summary.csv`, `backtest_summary.md` | Sharpe, mean daily bps, max drawdown, total return for both variants, with bootstrap CIs and the relative Sharpe change. |
| `equity_curves.csv`, `equity_curves.png` | Daily equity for `raw_alpha`, `cmmd`, and a passive `buy_and_hold_swda` benchmark. The curves are cumulative products of the daily returns, so the fee-bearing variants show the day-0 entry fee and the terminal value matches the summary's total return; the fee-free benchmark starts at exactly 1.0. |
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

## Benchmark

The model benchmark now lives in [`README.benchmark.md`](./README.benchmark.md).
It includes the benchmark date, the per-model table, the `cmmd` backtest table, and a metric glossary.

## Use as a package

The longer-term use case is to consume Recall Guard as a library rather than only through the two CLIs. [Global_Macro_AI_Factors](https://github.com/norandom/Global_Macro_AI_Factors) is one consumer: it builds AI macro/risk factors and can feed prompts through `recall_guard` to get a per-call `p_memorized` score and a discounted confidence.

Since v0.1.2 this is packaged: `recall_guard` installs from the GitHub Release (wheel + sdist, hatchling build) with a small stable façade over the `NvidiaLM` + control-baseline + MCS stack. See [Install (users)](#install-users) above; the packaging spec lives under [`.kiro/specs/recall-guard-package/`](./.kiro/specs/recall-guard-package/).

## Caveats

The MCS classifier is only as good as the calibration corpora. The shipped IS corpus is small, and the shipped IS/OOS split is also confounded by source, formatting, and publication period. Treat `p_memorized` as an implementation-specific score, not ground truth.

NVIDIA's free-tier 70B endpoints are queue-heavy. Use `--min-call-interval 1.5` or higher for stability. The 8B–20B size range is a better starting point.

The eval set you ship to the harness defines what "skill" means. The bundled ETF builder asks a next-day direction question that leaves little room for genuine forecasting edge. On this task the harness is mainly useful for comparing contamination-related signals across models and prompts. If you want to study reasoning skill instead, ship an eval set that asks something the model can plausibly reason about; the contamination machinery is the same either way.

The `cmmd` backtest uses `gpt-oss-20b` only. Other models are evaluated by the recall-guard check but do not feed that portfolio run; the design keeps the comparison focused on one signal stream.

## Citation

### Recall Guard

If you want to cite the software itself, use the Zenodo DOI. Cite the version you actually ran; the all-versions DOI always resolves to the latest release.

- v0.1.2: <https://doi.org/10.5281/zenodo.21557233>
- all versions: <https://doi.org/10.5281/zenodo.21557232>

`CITATION.cff` carries the same metadata for GitHub's "Cite this repository" widget.

### Underlying papers

Recall Guard is based on and evaluates ideas from MemGuard-Alpha and addresses the point-in-time evaluation problem also benchmarked by Look-Ahead-Bench.

**MemGuard-Alpha**

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

**Look-Ahead-Bench**

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
