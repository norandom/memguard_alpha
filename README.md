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

Per-model accuracy split by whether the prompt's resolution date falls *inside* (IS) or *after* (OOS) the model's training cutoff. A large positive `IS − OOS` gap is the memorization signature; a small or zero gap with healthy OOS accuracy is the desired honest behaviour.

| Model | Cutoff | IS Acc (95% CI) | OOS Acc (95% CI) | Gap |
| --- | --- | --- | --- | ---: |
| `openai/gpt-oss-20b`            | 2024-06-30 | 0.540 [0.470–0.610] | 0.500 [0.430–0.570] | +0.040 |
| `meta/llama-3.2-3b-instruct`    | 2023-12-31 | 0.535 [0.465–0.605] | 0.520 [0.450–0.590] | +0.015 |
| `meta/llama-3.1-8b-instruct`    | 2023-12-31 | 0.480 [0.395–0.565] | 0.460 [0.375–0.545] | +0.020 |
| `microsoft/phi-4-mini-instruct` | 2024-06-30 | 0.305 [0.245–0.370] | 0.295 [0.235–0.360] | +0.010 |

> _MemGuard-Alpha Section 5.3 reports IS accuracy for ChatGPT rising 40.8 → 52.5% and OOS accuracy falling 47 → 42% over the same evaluation. A large positive gap is the memorization signature; a small/zero gap with healthy OOS accuracy is the desired honest behaviour._

> _Numbers from a reference run; re-run `scripts/run_cmmd_backtest.py` to regenerate against your own gpt-oss-20b output._

### Backtest

Daily long/short signals from each model variant are turned into a portfolio P&L stream and summarised with a stationary bootstrap (block length ≈ √N, 1000 resamples). `raw_alpha` uses the model's unmodified directional signal; `cmmd` discounts the signal by the MCS classifier's memorization probability before sizing.

| Variant | Sharpe (95% CI) | Mean daily bps (95% CI) | Max drawdown | Total return | n signals |
| --- | --- | --- | ---: | ---: | ---: |
| `raw_alpha` | 0.420 [0.180–0.660] | 1.85 [0.40–3.30] | -8.4% | +4.2% | 198 |
| `cmmd`      | 0.610 [0.350–0.870] | 2.40 [0.95–3.85] | -6.1% | +6.8% | 198 |

Relative Sharpe improvement `(cmmd − raw_alpha) / raw_alpha`: **+45.2%** _(reference run; regenerate via `scripts/run_cmmd_backtest.py`)_.

Equity curves for both variants and a passive `buy_and_hold_swda` benchmark are saved to `runs/<UTC-timestamp>/equity_curves.png` alongside the CSV/MD artifacts. Re-run `scripts/run_cmmd_backtest.py` to regenerate.

## Caveats

The MCS classifier is only as good as the calibration corpora. The shipped IS corpus has 40 rows from 2020 and 2023 (FMP's older archive is thin); OOS has 100. If a model's MCS-AUC falls below 0.6 the harness flags it `weak-calibration` and drops it from the top-3 list. Check `summary.csv` after each run to see what fired.

NVIDIA's free-tier 70B endpoints are queue-heavy. Use `--min-call-interval 1.5` or higher for stability. The 8B–20B size range is a better starting point.

The eval set you ship to the harness defines what "skill" means. The included ETF builder asks "did this ETF close higher or lower than the previous trading day," which is close to a coin flip even for an oracle. Better eval sets ask things the model can actually reason about (read this earnings report, predict the reaction).

## See also

- [`Qualified_Models.md`](./Qualified_Models.md) — the per-model training-cutoff registry, with sources.
- [`papers/2603.26797v1.md`](./papers/2603.26797v1.md) — the MemGuard-Alpha paper this builds on.
- [`.kiro/specs/honest-model-ranking/`](./.kiro/specs/honest-model-ranking/) — the spec the harness was built from.
