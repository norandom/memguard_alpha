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
| Score model responses in an application | [Scorer integration](https://norandom.github.io/memguard_alpha/) | NVIDIA API key, a model with token logprobs, and calibration prompts |
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

## Windows

The Python library, mathematical utility and greedy decoder run natively on
Windows. You do not need WSL or a GPU for the synthetic example. Use the Python
entry points in PowerShell; `start.sh` is a convenience wrapper for bash.

For a contributor checkout:

```text
uv sync --frozen
uv run pytest -q
uv run python harness.py --help
uv run mkdocs serve
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

## Further reading

- [Engineering guide and API reference](https://norandom.github.io/memguard_alpha/)
- [Point-in-time architecture](docs/pit-architecture.md)
- [Ensemble scoring](docs/ensemble.md)
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
