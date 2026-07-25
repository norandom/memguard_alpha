# recall_guard

`recall_guard` runs a NIM-hosted model on your prompt, parses a directional signal, computes membership-inference features from per-token logprobs, and returns a per-prompt `p_memorized` score plus the discounted confidence.

It is not a forecaster. On the bundled raw price-direction task, near-coin-flip accuracy is not surprising. The useful output is the contamination score, not alpha. Use `p_memorized` to discount or filter AI-derived signals that look more like the repository's in-sample calibration corpus than its out-of-sample control corpus.

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
