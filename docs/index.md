# recall_guard

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21557232.svg)](https://doi.org/10.5281/zenodo.21557232)

`recall_guard` runs a NIM-hosted model on your prompt, parses a directional signal, computes membership-inference features from per-token logprobs, and returns a per-prompt `p_memorized` score plus the discounted confidence.

Every release is archived on Zenodo. The concept DOI above resolves to the latest version; per-version DOIs (v0.4.1 is [10.5281/zenodo.21806959](https://doi.org/10.5281/zenodo.21806959)) pin the exact artifact you ran. The README lists every archived version.

It is not a forecaster. On the bundled raw price-direction task, near-coin-flip accuracy is not surprising. The useful output is the contamination score, not alpha. Use `p_memorized` to discount or filter AI-derived signals that look more like the repository's in-sample calibration corpus than its out-of-sample control corpus.

## Where this fits

`recall_guard` is the measurement layer of a point-in-time (PIT) inference process. Anonymization, de-dating, and as-of data discipline reduce what a model can recall; this package measures what still leaks through and turns it into a per-prompt score. The full stack is described in [How this system achieves PIT inference](pit-architecture.md).

One example consumer is a macro overlay that multiplies each AI-generated Black-Litterman view by `(1 - p_memorized)` before it can move money, and falls back to its risk-parity core when a score or parse is missing. The boundary of this tool follows from that design: the score works as a discount, never as proof that a model is recall-free.

## Published results

That overlay is written up in **Computational Global Macro with AI for Risk and Portfolio Management** (Marius Ciepluch, SSRN working paper 7231358, 2026): <https://papers.ssrn.com/sol3/papers.cfm?abstract_id=7231358>

The paper treats point-in-time inference as an architectural requirement rather than a prompt convention: model inputs are anonymized, timestamp-free, dated strictly before each rebalance, and discounted by the Memorization Confidence Score this package computes. Bounded Black-Litterman views from three z-scored macro factors (inflation velocity, growth expectations, credit stress) tilt an HRP/CVaR baseline, followed by a Sparse Jump Model regime overlay and a macro-stress crisis gate.

On its 2016–2026 walk-forward, the de-risked line gives up CAGR — 15.4% to 9.4% — for lower annualized volatility (12.9% to 7.2%) and a shallower maximum drawdown (−19.6% to −8.2%). Sharpe is unchanged; the appraisal ratio rises from 0.81 to 1.06. Those portfolio numbers are produced by the consumer pipeline, not by this repository; what `recall_guard` contributes is the per-prompt score the views are discounted by.

## Install

`recall_guard` is distributed from its Git repository. For uv-managed projects:

```bash
uv add "recall-guard @ git+https://github.com/norandom/memguard_alpha.git@<tag-or-commit>"
```

or declare it directly:

```toml
# pyproject.toml of the consuming project
[project]
dependencies = [
    "recall-guard @ git+https://github.com/norandom/memguard_alpha.git@<tag-or-commit>",
]
```

The default install is lean: `numpy`, `scikit-learn`, `rich`, `pyyaml`, `requests`, `python-dotenv`. Optional extras: `recall-guard[backtest]` (matplotlib + vectorbt for the backtest) and `recall-guard[docs]` (this site's toolchain). Requires Python >= 3.12 and a valid `NVIDIA_API_KEY` at runtime.

## Minimal example

```python
from recall_guard import MemoryGuardedScorer

# Calibrate once per model. IS prompts were published before the model's
# documented cutoff; OOS prompts were published after it.
scorer = MemoryGuardedScorer.calibrate(
    api_key="...",                       # NVIDIA NIM key
    model="meta/llama-3.1-8b-instruct",
    is_memorized=is_prompts,             # Sequence[str]
    oos_control=oos_prompts,             # Sequence[str]
)
print(scorer.holdout_auc, scorer.is_weak)   # classifier quality on the held-out split

guarded = scorer.score("Your prompt here")
print(guarded.signal)                # parsed direction in {-1, 0, 1}
print(guarded.p_memorized)           # model-derived score in [0, 1]
print(guarded.memguard_confidence)   # raw_confidence * (1 - p_memorized)
```

`scorer.score_many(prompts)` scores a batch with parallel calls while preserving input order. A missing or rejected `NVIDIA_API_KEY` raises `ConfigurationError` rather than returning a silent failure.

## What recall_guard needs vs. what you own

| You provide | recall_guard owns |
| --- | --- |
| `NVIDIA_API_KEY` + a NIM model id | running the model and parsing the response |
| IS / OOS calibration prompt corpora | the MIA features and the per-model MCS classifier |
| the prompts you want scored | `p_memorized` and the MemGuard discount |

`recall_guard` does **not** own key provisioning, prompt construction, your factor pipeline, or any portfolio/allocation logic; those stay with the consumer. Browse the full surface under **API reference** in the navigation.
