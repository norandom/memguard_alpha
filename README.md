# Recall Guard

A small CLI that ranks language models on a financial-prediction task without crediting them for what they memorized.

## The problem

A model trained on web text has read years of market commentary. Ask it "did SPY close up on 2023-06-15" and it can recall the answer instead of reasoning about it. That looks like skill in a backtest. It isn't.

## The fix

This follows the MemGuard-Alpha paper. For each candidate model, train a small classifier (the MCS classifier) that recognizes that model's own log-probability signature when it's regurgitating from memory versus when it's reasoning. Use the classifier to discount confidence on prompts that look memorized. Rank models by what's left.

## What's in the repo

- `harness.py` — the CLI. One mode: load eval set → smoke shortlist → control baselines → MCS train → evaluate → rank → top-3.
- `start.sh` — convenience wrapper around `harness.py` with sensible defaults; override any input via env vars.
- `src/dataset/fmp_corpora.py` — pulls news from FMP and writes the IS-memorized and OOS-control calibration corpora.
- `src/core/` — HTTP client (NVIDIA chat completions with logprobs), JSONL loader with cutoff guard, bootstrap CI helper, run manifest.
- `src/mia/` — the five MIA features (Loss, Min-K%, Min-K%++, zlib ratio, ref-delta), per-model control baseline, MCS calibrator.
- `src/harness/` — smoke gate, evaluator, ranker, report writers, plot helpers, runner.
- `notebooks/qualification.ipynb` — methodology with twelve LaTeX equations and figure templates.
- `notebooks/method_overview.ipynb` — what's in each calibration corpus and why.
- `notebooks/visualize_run.ipynb` — load a finished run directory and render the paper-ready figures.

## Install

Python 3.14, [uv](https://github.com/astral-sh/uv), an NVIDIA chat-completions API key, an FMP API key (only needed if you rebuild the calibration corpora).

```bash
uv sync
cat > .env <<EOF
NVIDIA_API_KEY=...
FMP_API_KEY=...
EOF
```

That's it — `uv sync` resolves and installs everything from `pyproject.toml` into `.venv/`. No global pip, no manual virtualenv.

## Run

```bash
./start.sh
```

Defaults to the multiyear ETF eval set, both calibration corpora, the registry cutoffs, and a 4-model shortlist (`llama-3.1-8b`, `llama-3.2-3b`, `gpt-oss-20b`, `phi-4-mini`) with 8 parallel workers and 1.5 s rate-limit pacing. Override any input inline:

```bash
SHORTLIST="meta/llama-3.1-8b-instruct" ./start.sh
EVAL_SET=data/eval/etf_direction.jsonl OUT_DIR=runs/quick ./start.sh
./start.sh --no-reference        # extra harness flags pass through
```

Each run writes five files into `OUT_DIR` (default `runs/<UTC-timestamp>/`):

- `top3.md` — the ranking, plus a "why fewer than three" section when gates kicked models out.
- `summary.csv` — every model's CIs (raw acc, MemGuard acc, MCS-AUC), parse-success rate, score, warnings.
- `records.jsonl` — per-prompt logprobs and MIA feature values.
- `manifest.json` — seed, input hashes, artifact paths.
- `shortlist.json` — present only when the smoke gate ran (with `--candidates`).

If you ever need to rebuild the inputs:

```bash
uv run python -m src.dataset.fmp_corpora build         # calibration corpora
uv run python scripts/build_etf_multiyear_eval.py      # ETF eval set
```

## See results

Open `notebooks/visualize_run.ipynb`, point its `RUN_DIR` cell at the run you want, and execute. Two ways to do that:

```bash
# Option A — execute headless, then open the rendered notebook
uv run jupyter nbconvert --to notebook --execute notebooks/visualize_run.ipynb \
  --output visualize_run.ipynb
uv run jupyter-lab notebooks/visualize_run.ipynb

# Option B — interactive: open lab and step through the cells
uv run jupyter-lab notebooks/visualize_run.ipynb
```

The notebook renders MIA-feature distributions, MCS calibration curves, accuracy + MCS-AUC with bootstrap CIs, and the composite ranking — all from the artifacts in `OUT_DIR`.

## Sample run

200-prompt multiyear ETF eval set (SPY, QQQ, GLD, URTH, sampled across 2020–2026), 4-model shortlist, NVIDIA free tier:

| Model | Parse % | Raw Acc (95% CI) | MCS-AUC | vs Baseline (0.5350) |
|---|---|---|---|---|
| `openai/gpt-oss-20b`            | 92.0%  | **0.5380** [0.4674–0.6088] | 0.984 | point above baseline; CIs overlap |
| `meta/llama-3.2-3b-instruct`    | 99.0%  | 0.5303 [0.4596–0.6010]     | 1.000 | indistinguishable from baseline |
| `meta/llama-3.1-8b-instruct`    | 49.0%  | 0.4694 [0.3773–0.5612]     | 1.000 | parse-unreliable, below baseline |
| `microsoft/phi-4-mini-instruct` | 100.0% | **0.3000** [0.2400–0.3650] | 0.963 | **statistically anti-skilled** (CI doesn't overlap baseline) |
| `__majority_baseline__`         | —      | 0.5350 [0.4600–0.6001]     | —     | always-predict-the-majority-class |

Two findings worth keeping:

- `gpt-oss-20b` is the only model whose point estimate beats the always-predict-up baseline. The 95% CI still overlaps, so this is *not* a "p<0.05" winner — but it sits exactly where MemGuard-Alpha's own models live (the paper reports 40–52% directional accuracy across contamination quintiles on the same kind of task).
- `phi-4-mini` at 0.30 is reliably *anti-skilled*. Its CI [0.24–0.37] does not overlap the baseline, so the "wrongness" is real, not noise. On a 3-class task that means it's consistently picking against the right direction — likely a prompt/training quirk that flips the +1 / -1 mapping.

The MCS-AUC values near 1.0 indicate that the calibrator can perfectly separate IS-memorized text from OOS-control text for these models — exactly the memorization signal the paper documents.

### IS-vs-OOS memorization gap

Per-(model, eval-row) split by whether the row's resolution date falls *inside* (IS) or *after* (OOS) the model's training cutoff. A large positive `IS − OOS` gap is the memorization signature; a small or zero gap with healthy OOS accuracy is the desired honest behaviour.

Live `runs/cmmd_20260429T064026Z/` against `openai/gpt-oss-20b` (eval set: 330 prompts × {SWDA.L, XLK, IAU} straddling the 2024-06-30 cutoff):

| Model | Cutoff | n_IS | n_OOS | IS Acc (95% CI) | OOS Acc (95% CI) | Gap |
| --- | --- | ---: | ---: | --- | --- | ---: |
| `openai/gpt-oss-20b` | 2024-06-30 | 194 | 121 | 0.505 [0.438–0.577] | 0.612 [0.521–0.702] | **−0.106** |

`n_IS + n_OOS = 315`, matching the `parse_ok=True` row count in `records.jsonl` (15 of 330 rows parse-failed and are excluded).

> _MemGuard-Alpha Section 5.3 reports IS accuracy for ChatGPT rising 40.8 → 52.5% and OOS accuracy falling 47 → 42% over the same evaluation. A large positive gap is the memorization signature; a small/zero gap with healthy OOS accuracy is the desired honest behaviour._

The negative gap on this run is the *opposite* of the paper's directional-trade contamination story: gpt-oss-20b actually does better on rows it could not have memorised (post-2024-06-30) than on rows it could. The Cohen's d artifact in the same run dir still shows a large IS↔OOS separation on the raw MIA features (`loss` d=+1.84, `zlib_ratio` d=+1.83), so the calibrator is detecting *something* — it just is not the directional-accuracy gap the paper flagged. Re-run `scripts/run_cmmd_backtest.py` to regenerate against your own gpt-oss-20b output.

### Backtest

Daily long/short signals from gpt-oss-20b are turned into a long-short cross-sectional portfolio over `{SWDA.L, XLK, IAU}` (BIL as the cash leg) with leverage capped at 1×, daily rebalance on the close, and 15 bps round-trip transaction cost. `raw_alpha` uses every parse-OK row; `cmmd` drops the top-quintile rows by MCS-derived `p_memorized`. Bootstrap 95% CIs use 1000 resamples on Sharpe and mean daily return.

Live numbers from `runs/cmmd_20260429T064026Z/backtest_summary.md`:

| Variant | Sharpe (95% CI) | Mean daily bps (95% CI) | Max drawdown | Total return | n signals |
| --- | --- | --- | ---: | ---: | ---: |
| `raw_alpha` | −2.030 [−2.750, −0.722] | −2.77 [−4.29, −1.28] | −29.15% | −29.04% | 315 |
| `cmmd`      | −1.800 [−2.648, −0.539] | −2.27 [−3.62, −0.88] | −24.46% | −24.52% | 252 (cmmd_threshold = 0.347) |

Relative Sharpe improvement `(cmmd − raw_alpha) / raw_alpha`: **−11.3%** — meaning the CMMD-filtered Sharpe is *less negative* than `raw_alpha`'s. Both variants are net-losing on this universe and date span, so the "improvement" is a smaller loss rather than alpha. The paper's headline finding (CMMD lifts a borderline-positive Sharpe further into the green) does not reproduce on a single-model gpt-oss-20b run against this 3-asset ETF universe; this is consistent with the negative IS−OOS gap above.

Equity curves for both variants plus the passive `buy_and_hold_swda` benchmark are saved to `runs/cmmd_20260429T064026Z/equity_curves.png` alongside the CSV/MD artifacts. Re-run `scripts/run_cmmd_backtest.py` to regenerate.

## Caveats

The MCS classifier is only as good as the calibration corpora. The shipped IS corpus has 40 rows from 2020 and 2023 (FMP's older archive is thin); OOS has 100. If a model's MCS-AUC falls below 0.6 the harness flags it `weak-calibration` and drops it from the top-3 list. Check `summary.csv` after each run to see what fired.

NVIDIA's free-tier 70B endpoints are queue-heavy. Use `--min-call-interval 1.5` or higher for stability. The 8B–20B size range is a better starting point.

The eval set you ship to the harness defines what "skill" means. The included ETF builder asks "did this ETF close higher or lower than the previous trading day," which is close to a coin flip even for an oracle. Better eval sets ask things the model can actually reason about (read this earnings report, predict the reaction).

## See also

- [`Qualified_Models.md`](./Qualified_Models.md) — the per-model training-cutoff registry, with sources.
- [`papers/2603.26797v1.md`](./papers/2603.26797v1.md) — the MemGuard-Alpha paper this builds on.
- [`.kiro/specs/honest-model-ranking/`](./.kiro/specs/honest-model-ranking/) — the spec the harness was built from.
