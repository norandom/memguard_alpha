# recall_guard

Measured **inference-without-recall**. recall_guard runs a NIM-hosted model on your
prompt and returns the parsed directional signal **together with a calibrated
`p_memorized`** — a per-prompt contamination score derived from per-token logprobs —
plus the MemGuard-discounted confidence.

It is **not a forecaster**. Near-coin-flip directional accuracy on a raw price-direction
task is the expected, correct result; the product is the honesty signal, not alpha. Use
`p_memorized` to discount or filter AI-derived factor signals that look memorized rather
than reasoned.

## Install

recall_guard is distributed from its Git repository, for uv-managed projects:

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

The default install is lean — `numpy`, `scikit-learn`, `rich`, `pyyaml`, `requests`,
`python-dotenv`. Optional extras: `recall-guard[backtest]` (matplotlib + vectorbt for the
CMMD backtest) and `recall-guard[docs]` (this site's toolchain). Requires **Python ≥ 3.12**
and a valid `NVIDIA_API_KEY` at runtime.

## Minimal example

```python
from recall_guard import MemoryGuardedScorer

# Calibrate once per model. IS = prompts dated BEFORE the model's training cutoff,
# OOS = prompts dated AFTER it; the contamination calibrator learns to tell them apart.
scorer = MemoryGuardedScorer.calibrate(
    api_key="...",                       # NVIDIA NIM key
    model="meta/llama-3.1-8b-instruct",
    is_memorized=is_prompts,             # Sequence[str]
    oos_control=oos_prompts,             # Sequence[str]
)
print(scorer.holdout_auc, scorer.is_weak)   # calibrator quality

guarded = scorer.score("Your prompt here")
print(guarded.signal)                # parsed direction in {-1, 0, 1}
print(guarded.p_memorized)           # calibrated contamination probability in [0, 1]
print(guarded.memguard_confidence)   # raw_confidence * (1 - p_memorized)
```

`scorer.score_many(prompts)` scores a batch with parallel calls, preserving input order.
A missing or rejected `NVIDIA_API_KEY` raises `ConfigurationError` rather than returning a
silently invalid score.

## What recall_guard needs vs. what you own

| You provide | recall_guard owns |
| --- | --- |
| `NVIDIA_API_KEY` + a NIM model id | running the model and parsing the response |
| IS / OOS calibration prompt corpora | the five MIA features + the per-model MCS calibrator |
| the prompts you want scored | `p_memorized` and the MemGuard discount |

recall_guard does **not** own key provisioning/rotation, prompt construction, your factor
pipeline, or any portfolio/allocation logic — those stay with the consumer (for example,
`macro_framework`). Browse the full surface under **API reference** in the navigation.
