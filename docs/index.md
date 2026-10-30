# Integrate and verify Recall Guard

Recall Guard has two entry points. The scorer measures contamination-related
features in hosted model responses. The experimental divergence utility changes
next-token logits inside a generation loop you control. Choose the interface
that matches your runtime.

| Interface | Input | Output | Runtime |
| --- | --- | --- | --- |
| `MemoryGuardedScorer` | Calibration corpora and prompts | Direction, `p_memorized`, discounted confidence and failure status | NVIDIA NIM with token logprobs |
| `combine_logits` / `decode_greedy` | Compatible logits or three callbacks | Adjusted logits or generated token IDs | Caller supplied; NumPy helper runs locally |

## Install and choose a revision

Use Python >=3.12. For a released scorer in a uv-managed project:

```text
uv add "recall-guard @ git+https://github.com/norandom/memguard_alpha.git@v0.5.0"
```

The divergence helper is included in v0.5.0. Pin a release or commit to keep
your integration reproducible. The default runtime dependencies are NumPy, scikit-learn, Rich, PyYAML,
requests and python-dotenv. Extras: `backtest` and `docs`.

Every release is archived on Zenodo. The [all-versions DOI](https://doi.org/10.5281/zenodo.21557232)
resolves to the release archive; [v0.4.1](https://doi.org/10.5281/zenodo.21806959)
identifies that version.

## Integrate the scorer

Prepare two UTF-8 JSONL files with a `prompt` field on each row. Use prompts from
before the model's documented cutoff for the in-sample corpus and after it for
the out-of-sample control. Match domain and format where possible. These dates
are exposure labels, not proof of memorization.

```python
import json
import os
from pathlib import Path

from recall_guard import MemoryGuardedScorer


def load_prompts(corpus_path: Path) -> list[str]:
    return [
        json.loads(line)["prompt"]
        for line in corpus_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


scorer = MemoryGuardedScorer.calibrate(
    api_key=os.environ["NVIDIA_API_KEY"],
    model="openai/gpt-oss-20b",
    is_memorized=load_prompts(Path("data/calibration/is_memorized.jsonl")),
    oos_control=load_prompts(Path("data/calibration/oos_control.jsonl")),
)
print(scorer.holdout_auc, scorer.is_weak)

guarded = scorer.score("Direction and confidence for your evaluation prompt")
if not guarded.parse_ok:
    print("Scoring failed:", guarded.fail_reason)
elif scorer.is_weak:
    print("Calibration is weak; review the corpora before using this score.")
else:
    print(guarded.signal, guarded.p_memorized, guarded.memguard_confidence)
```

This example makes live calls and requires `NVIDIA_API_KEY`. Calibration is the
expensive setup step. `score_many(prompts)` preserves input order while making
parallel calls. `ConfigurationError` signals missing or rejected credentials;
runtime and parsing failures are exposed in result records.

The score path computes MIA features from token log probabilities, standardizes
them against the control baseline, and applies the per-model MCS classifier.
`memguard_confidence` is `raw_confidence * (1 - p_memorized)`. Held-out AUC
measures separation of the calibration labels; it does not establish that the
probabilities are ground-truth memorization rates.

## Integrate divergence decoding

Start with the [offline callback example](divergence-decoding.md). It shows the
array axis, declared vocabulary identity, complete prefix contract and stopping
behavior. Three callbacks must return logits for the same token-ID meanings.
The decoder validates declarations but cannot verify the actual tokenizer or
training history.

The equation is inspired by Jane Street's
[Mitigating memorization in LLMs](https://blog.janestreet.com/mitigating-memorization-in-llms/).
Our helper implements the mathematical operation and a greedy loop. Its tests
establish correctness; empirical mitigation and calibrated scoring require
separate work. The NVIDIA client receives completed responses and cannot apply
this adjustment retroactively.

## What recall_guard needs vs. what you own

| You provide | recall_guard owns |
| --- | --- |
| NVIDIA key, supported model and calibration prompts | Hosted calls, parsing, MIA features and the calibrated scorer |
| Compatible callbacks and vocabulary declarations | Logits validation, the equation and greedy token selection |
| Your evaluation task and fallback policy | Failure records and documented result semantics |

Recall Guard does not own prompt construction, as-of data selection, local model
hosting or portfolio allocation. For ensembles, read the
[ensemble contract](ensemble.md) before using the exposure multiplier.
[Point-in-time architecture](pit-architecture.md) explains the surrounding data
discipline.

## Run the repository workflows

From a development checkout, use the same Python commands in PowerShell and bash.
For hosted runs, set `NVIDIA_API_KEY`; the portfolio workflow also needs
`FMP_API_KEY`. Python entry points read the repository's `.env` file.

```text
uv run python harness.py --help
uv run python harness.py --eval-set data/eval/etf_portfolio.jsonl --is-memorized data/calibration/is_memorized.jsonl --oos-control data/calibration/oos_control.jsonl --cutoffs data/cutoffs.yaml --shortlist openai/gpt-oss-20b --min-call-interval 1.5
uv run python scripts/run_cmmd_backtest.py
```

The harness writes `records.jsonl`, `summary.csv`, `top3.md` and
`manifest.json` under the selected run directory. The manifest records input
hashes and settings. The portfolio workflow adds IS/OOS gap reports, feature
effect sizes, prices and portfolio artifacts. It requires the backtest
dependencies, which the checkout's development group includes.

Use [the benchmark page](benchmark.md) to interpret metric names and
[latest run results](latest-run.md) for the published artifacts. The bash
`start.sh` wrapper supplies a one-model default and other defaults; the
Python command above is a smaller explicit run.

## Windows and development

The core library and decoder use native Python/NumPy. No WSL, shell activation
script, tensor framework or GPU is required by the package. Any real-model
callback you supply has its own runtime requirements.

```text
uv sync --frozen
uv run pytest -q
uv run ruff check .
uv run python scripts/check_architecture.py
uv build
uv run mkdocs build --strict
```

The test suite is offline. Live scoring and data refresh are separate operations.
The gap-report script uses explicit UTF-8, so its Unicode reports
do not depend on the Windows locale. Architecture diagnostics use forward
slashes consistently. Request pacing waits against the actual previous
admission time, which prevents delayed wakeups from compressing later requests;
the network call runs outside that lock.

For maintenance, keep core primitives below MIA/dataset/portfolio, with harness
orchestration above them. Use idiomatic Python and NumPy with names that describe
the current value: `main_logits`, `adjusted_logits`, `prefix_token_ids` and
`generation_step`. State array axes and shapes at interfaces. DataFrames should
have meaningful column/index names, documented row grain and units, and
intentional alignment when combined. Avoid abstraction unless it makes a real
contract easier to understand. Ruff and the repository's architecture checker
are the project gates; Sentrux is not required.

## Evidence and published results

[Computational Global Macro with AI for Risk and Portfolio Management](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=7231358)
describes an example consumer of this package's confidence discount. Its
portfolio performance is not an acceptance test for Recall Guard.

Evaluate your own corpora and task before relying on the scores. If source,
format or period differences explain the classifier's separation, revise the
calibration data. For divergence decoding, if the model's useful behavior
deteriorates without independent evidence of reduced recall, the intervention
has not demonstrated its intended benefit.

## Hosted access preflight

Calibration now makes one authenticated generation probe per requested model
before scheduling corpus calls. A public model-catalog response does not prove
inference authorization. An HTTP 401 or 403 stops setup with an access error;
update the local key or its inference permissions before retrying. The current
CLI default has no reference model; request one explicitly with
`--reference-model` if it is available and suitable. On 2026-10-30,
`openai/gpt-oss-20b` was the only catalog model already covered by this
repository's cutoff registry. Historical IDs remain for interpreting older runs.
