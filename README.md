# Recall Guard

[![DOI](https://zenodo.org/badge/1221761081.svg)](https://doi.org/10.5281/zenodo.21557232)

Recall Guard helps you check whether an LLM's historical answers look contaminated
by memorized information. It returns a calibrated `p_memorized` score and a
discounted confidence that your application can use when handling AI signals.

Historical accuracy can come from recall as well as reasoning. The score is a
warning signal learned from your calibration data, not proof that an individual
answer was memorized. Recall Guard is not a forecaster.

## Choose your starting point

| Your goal | Start here | What you need |
| --- | --- | --- |
| Run checkout examples and tests in Docker | [Docker quick start](#run-the-checkout-in-docker) | Docker with Compose and a current checkout; the first example needs no keys |
| Score model responses in an application | [Scorer integration](https://norandom.github.io/memguard_alpha/) | NVIDIA API key, a model with token logprobs, and calibration prompts |
| Configure completion APIs, optional rewriting or n draws | [Completion workflow](docs/completion-api.md) | Development checkout; the default application and notebook fixtures run offline |
| Try the divergence equation without a model | [Executable example](docs/divergence-decoding.md) | Current checkout, Python and NumPy; no credentials |
| Steer your own model during generation | [Decoder contract](docs/divergence-decoding.md#caller-responsibilities-and-limits) | Three compatible callbacks that expose next-token logits |
| Inspect benchmarks or portfolio experiments | [Benchmark](README.benchmark.md) and [engineering guide](docs/index.md) | Repository data; API access for fresh model/price runs |

## Install the scorer

Use Python 3.12 or newer and [uv](https://docs.astral.sh/uv/).
The same command works in PowerShell, bash and zsh:

```text
uv add "recall-guard @ git+https://github.com/norandom/memguard_alpha.git@v0.5.0"
```

Releases provide wheels and source distributions on GitHub Releases; the package
is not published on PyPI. Optional extras are `backtest` and `docs`.

Calibrate `MemoryGuardedScorer` for your model, then call `scorer.score(prompt)`.
Read `parse_ok`, `fail_reason` and calibration quality before using
`p_memorized`. The [engineering example](docs/index.md#integrate-the-scorer)
shows the complete input path and failure handling.

## Run the checkout in Docker

Use the same locked Python environment for local examples, tests and Dagger CI.
From a current development checkout, create the `runs/container` folder, then run:

```text
docker compose build dev
docker compose run --rm dev
```

The default command runs the offline completion example. Expect an accepted
rewritten ensemble, a rejected rewrite, unavailable MCS and zero HTTP requests.
No API credentials are needed. This checkout tooling is separate from the
published v0.5.0 package.

Run tests or open the optional notebook environment:

```text
docker compose run --rm dev uv run --no-sync pytest -q
docker compose --profile notebook up notebook
```

Open the localhost URL printed by Jupyter, including its login token. Notebook
edits persist in `notebooks/`; write other artifacts to `/outputs`, which maps to
`runs/container`. Stop the notebook with Ctrl+C, then use
`docker compose --profile notebook down` to remove its container and network.
Source-code changes take effect after `docker compose build dev`.

The [engineering guide](docs/development.md) covers prerequisites, focused tests,
the lean runtime image, Dagger and explicit provider credentials. Direct Python
and existing Jupyter kernels remain supported.

## Try the completion workflow from a development checkout

The unreleased completion workflow lets you configure a target once and choose
single answers, n-draw ensembles and optional evidence rewriting. Start with the
fixed offline callbacks to inspect capabilities, failures and missing MCS:

```text
uv sync --frozen
uv run --no-sync python examples/completion_workflow.py
```

Use a checkout containing that example; these APIs are not in v0.5.0. Open
`notebooks/completion_workflow.ipynb` in the same installed Python environment
for the Jupyter path. The [engineering guide](docs/completion-api.md) explains
explicit provider settings and offline replay of the
[dated pilot](docs/results/2026-10-31-completion-api-pilot.md). The fixtures test
mechanics, and the pilot does not establish a mitigation benefit.

## Try the experimental mathematical layer

The divergence utility is included in v0.5.0. From a checkout, install only the
core dependencies:

```text
git clone https://github.com/norandom/memguard_alpha.git
cd memguard_alpha
uv sync --no-dev
uv run --no-dev python -c "from recall_guard.core.divergence import combine_logits; print(combine_logits([2, 0], [0, 3], [0, 0], alpha=1.0))"
```

Expected output: `[2. 3.]`. This verifies the equation locally. The
[full callback example](docs/divergence-decoding.md#runnable-example) shows how
the greedy decoder uses it at each token.

This optional utility follows the divergence-decoding approach described by
James Somers in Jane Street's
[Mitigating memorization in LLMs](https://blog.janestreet.com/mitigating-memorization-in-llms/)
(September 30, 2026). It combines main-model logits with the difference between
older and newer auxiliary models. Effectiveness for your models still needs
evaluation.

The hosted NVIDIA completion client does not expose the token-selection loop.
To use divergence decoding with real models, you supply compatible logits
callbacks. This utility returns token IDs; it does not produce a calibrated
contamination score.

The equation needs next-token logits inside the generation loop. An endpoint
can apply it internally and return text without logprobs, but ordinary
completion text from OpenAI, Anthropic or another provider is insufficient for
this client-side decoder. The development checkout also provides an opt-in
fact-preserving rewrite workflow for completion APIs. It is a separate
experimental method, not part of v0.5.0; the
[dated pilot](docs/results/2026-10-31-completion-api-pilot.md) records its measured
limits and incomplete provider coverage.

## Windows

The Python library, mathematical utility and greedy decoder run natively on
Windows. You do not need WSL or a GPU for the synthetic example. Use the Python
entry points in PowerShell; `start.sh` is a convenience wrapper for bash.

For a contributor checkout:

```text
uv sync --frozen
uv run --no-sync pytest -q
uv run --no-sync python harness.py --help
uv run --no-sync mkdocs serve
```

Hosted scoring needs NVIDIA credentials and network access on every operating
system. Any local runtime behind your own callbacks has its own hardware and
Windows support requirements. See [Windows and development](docs/index.md#windows-and-development).

## Understand a result before adopting it

The classifier learns the difference between the supplied in-sample and
out-of-sample corpora. Source, formatting and publication-period differences can
confound that comparison. Start with your own calibration data and inspect the
held-out quality before depending on the score.

`memguard_confidence = raw_confidence * (1 - p_memorized)`.
Your application decides what to do with weak calibration, failed parsing or
missing scores. A small score does not certify a model as recall-free.

[Global_Macro_AI_Factors](https://github.com/norandom/Global_Macro_AI_Factors)
is an example consumer. The applied workflow is described in
[Computational Global Macro with AI for Risk and Portfolio Management](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=7231358).
Its portfolio results belong to that consumer pipeline.

## NIM refresh: 2026-11-01

The Docker refresh of `openai/gpt-oss-20b` stopped at its first eligibility
request: HTTP 504 after about 302 seconds. Key eligibility remains unverified.
Fresh calibration and the planned single-versus-three-draw ensemble comparison
did not run, so the earlier results remain the latest successful measurements.
The updated harness passed its offline checks; that does not establish provider
availability. See the [dated report and raw receipts](docs/results/2026-11-01-nim-refresh.md).

## Completion API pilot: 2026-10-31

The saved Claude Opus 5.5 pilot does not demonstrate a mitigation benefit.
All 16 rewrites failed the frozen fidelity gate before rewritten target
inference, leaving zero valid rewrite pairs. Original single answers were
correct on 11/16 items and three-draw ensembles on 12/16. The two conditions
were both correct on 11/14 common usable items; this small dependent sample
does not establish an aggregation gain or equivalence.

NIM eligibility failed after two 90-second timeouts and an HTTP 504 under a
600-second timeout. Fresh logprob calibration and required two-mode live
measurement remain incomplete. Anthropic MCS is unavailable. The
[dated report and downloadable replay bundle](docs/results/2026-10-31-completion-api-pilot.md)
include denominators, paired uncertainty, the AI fidelity audit and resource
usage. The completion workflow is unreleased checkout code.

## Local measurement: 2026-10-30

A full NVIDIA NIM run with `openai/gpt-oss-20b` fitted a fresh contamination
classifier and evaluated 330 ETF prompts. Held-out calibration ROC-AUC was
0.876. The model returned 312 parseable responses (94.55%), with 170 correct
directions: 54.49% accuracy, with a 95% bootstrap interval of 48.72% to 60.26%.
It did not pass the harness's accuracy gate against the majority baseline.

The confidence discount reduced mean confidence from 0.5992 to 0.2219 on parsed
responses. That is the scorer's arithmetic effect; it does not establish that
confidence became more accurate. Corpus differences can confound the AUC, and
the fitted calibrator is used per run rather than exported for reuse.

Divergence decoding was inactive in this run. Its effect on memorization in
Recall Guard remains unmeasured. Read the
[dated technical report and downloadable results](docs/results/2026-10-30-nim-measurement.md)
for denominators, settings, limitations and the experiment needed to measure
the Jane Street feature.

## Further reading

- [Engineering guide and API reference](https://norandom.github.io/memguard_alpha/)
- [Point-in-time architecture](docs/pit-architecture.md)
- [Ensemble scoring](docs/ensemble.md)
- [Completion API configuration and notebook integration](docs/completion-api.md)
- [Model cutoff sources](Qualified_Models.md)
- [Benchmark results](README.benchmark.md)



## Citation

### Recall Guard

If you want to cite the software itself, use the Zenodo DOI. Cite the version you actually ran; the all-versions DOI always resolves to the latest release.

- v0.5.0: [GitHub Release](https://github.com/norandom/memguard_alpha/releases/tag/v0.5.0); version-specific DOI pending.
- v0.4.1: <https://doi.org/10.5281/zenodo.21806959>
- v0.4.0: <https://doi.org/10.5281/zenodo.21795891>
- v0.3.1: <https://doi.org/10.5281/zenodo.21795665>
- v0.3.0: <https://doi.org/10.5281/zenodo.21795407>
- v0.2.0: <https://doi.org/10.5281/zenodo.21637978>
- v0.1.2: <https://doi.org/10.5281/zenodo.21557233>
- all versions: <https://doi.org/10.5281/zenodo.21557232>

Releases before v0.1.2 predate the Zenodo archive and have no DOI.

`CITATION.cff` carries the same metadata for GitHub's "Cite this repository" widget.

### Applied results

Recall Guard supplies the memorization discount used by the point-in-time architecture in this working paper. It is the applied write-up of the [Global_Macro_AI_Factors](https://github.com/norandom/Global_Macro_AI_Factors) consumer linked above.

```bibtex
@misc{ciepluch2026computationalglobalmacro,
  title        = {Computational Global Macro with AI for Risk and Portfolio Management},
  author       = {Ciepluch, Marius},
  year         = {2026},
  howpublished = {SSRN working paper},
  note         = {Abstract 7231358. Written 31 July 2026, posted 10 August 2026},
  url          = {https://papers.ssrn.com/sol3/papers.cfm?abstract_id=7231358}
}
```

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
