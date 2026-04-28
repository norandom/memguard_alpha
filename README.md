# Recall Guard

A small CLI that ranks language models on a financial-prediction task without crediting them for what they memorized.

## The problem

A model trained on web text has read years of market commentary. Ask it "did SPY close up on 2023-06-15" and it can recall the answer instead of reasoning about it. That looks like skill in a backtest. It isn't.

## The fix

This follows the MemGuard-Alpha paper. For each candidate model, train a small classifier (the MCS classifier) that recognizes that model's own log-probability signature when it's regurgitating from memory versus when it's reasoning. Use the classifier to discount confidence on prompts that look memorized. Rank models by what's left.

## What's in the repo

- `harness.py` — the CLI. Two subcommands: `build` (full evaluation pipeline) and `replay` (reproduce a past run from its manifest).
- `src/dataset/fmp_corpora.py` — pulls news from FMP and writes the IS-memorized and OOS-control calibration corpora.
- `src/core/` — HTTP client (NVIDIA chat completions with logprobs), JSONL loader with cutoff guard, bootstrap CI helper, run manifest.
- `src/mia/` — the five MIA features (Loss, Min-K%, Min-K%++, zlib ratio, ref-delta), per-model control baseline, MCS calibrator.
- `src/harness/` — smoke gate, evaluator, ranker, report writers, plot helpers, runner.
- `notebooks/qualification.ipynb` — methodology with twelve LaTeX equations and figure templates.
- `notebooks/method_overview.ipynb` — what's in each calibration corpus and why.
- `notebooks/visualize_run.ipynb` — load a finished run directory and render the paper-ready figures.

## Setup

Python 3.14, [uv](https://github.com/astral-sh/uv), an NVIDIA chat-completions API key, an FMP API key.

```bash
uv sync
cat > .env <<EOF
NVIDIA_API_KEY=...
FMP_API_KEY=...
EOF
```

## Run

Three commands. Calibration corpora are built once. The eval set is per-experiment. The harness consumes both.

```bash
# 1. Calibration corpora (one-shot, writes data/calibration/*.jsonl)
uv run python -m src.dataset.fmp_corpora build

# 2. Eval set (the ETF starter; replace with your own builder later)
uv run python scripts/build_etf_multiyear_eval.py

# 3. Rank models
uv run python harness.py build \
  --eval-set data/eval/etf_direction_multiyear.jsonl \
  --shortlist meta/llama-3.1-8b-instruct,openai/gpt-oss-20b \
  --out-dir "runs/$(date +%Y%m%d_%H%M%S)" \
  --no-reference \
  --min-call-interval 1.0
```

Each run writes five files into `--out-dir`:

- `top3.md` — the ranking, plus a "why fewer than three" section when gates kicked models out.
- `summary.csv` — every model's CIs (raw acc, MemGuard acc, MCS-AUC), parse-success rate, score, warnings.
- `records.jsonl` — per-prompt logprobs and MIA feature values.
- `manifest.json` — seed, input hashes, ranking — for `harness replay`.
- `shortlist.json` — present only when the smoke gate ran (with `--candidates`).

## Caveats

The MCS classifier is only as good as the calibration corpora. The shipped IS corpus has 40 rows from 2020 and 2023 (FMP's older archive is thin); OOS has 100. If a model's MCS-AUC falls below 0.6 the harness flags it `weak-calibration` and drops it from the top-3 list. Check `summary.csv` after each run to see what fired.

NVIDIA's free-tier 70B endpoints are queue-heavy. Use `--min-call-interval 1.5` or higher for stability. The 8B–20B size range is a better starting point.

The eval set you ship to the harness defines what "skill" means. The included ETF builder asks "did this ETF close higher or lower than the previous trading day," which is close to a coin flip even for an oracle. Better eval sets ask things the model can actually reason about (read this earnings report, predict the reaction).

## See also

- [`Qualified_Models.md`](./Qualified_Models.md) — the per-model training-cutoff registry, with sources.
- [`papers/2603.26797v1.md`](./papers/2603.26797v1.md) — the MemGuard-Alpha paper this builds on.
- [`.kiro/specs/honest-model-ranking/`](./.kiro/specs/honest-model-ranking/) — the spec the harness was built from.
