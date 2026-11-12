# Recall Guard

Point-in-time inference from language models you host yourself, for quantitative finance research.

A language model used in a simulation knows what happened after every date before its training cutoff. That is lookahead bias: a backtest can look good because the model remembers the outcome. Recall Guard runs a prompt point in time, estimates how much of the answer is recalled, and gives the analyst that estimate to correct with.

| The period your data describes | What the recall estimate means | What to do |
| --- | --- | --- |
| On or before the model's training cutoff | The model may have seen what happened next. The estimate measures how much the reply rests on that. | Use the estimate. An answer for this period needs correction. |
| After the model's training cutoff | The model cannot have seen what happened next. An estimate above zero is noise. | Do not correct. |

Recall Guard is not a forecaster. It measures recall and reports it. How the estimate enters an analysis is the analyst's decision.

## Start here

1. Install:

    ```text
    uv add "recall-guard @ git+https://github.com/norandom/recall_guard.git@v0.6.0"
    ```

2. Run the [usage notebook](notebooks/recall_adjusted_answers.ipynb). It asks one point-in-time prompt in both evidence modes, offline first and then against live local models.
3. Read the [use cases](use-cases.md) and pick the one closest to your research.

```python
from recall_guard import Mode, ask, offline_example, worked_example

prompt, definition = worked_example()   # a point-in-time macro prompt and how its reply is read
example = offline_example()             # invented entry, calibrators and replies

result = ask(prompt, definition, **example.arguments(Mode.LOGPROBS))
print(result.answer, result.p_memorized, result.corrected)
```

`ask` returns the model's answer, the recall estimate and the answer corrected by it, together with the status of the model and route it ran on. On your own model you pass a catalogue entry and a calibrator fitted on that model.

## Use cases

One page, three research uses: what the repository provides for each and what lives in the consumer pipeline.

- [Macro timeline analysis](use-cases.md#macro-timeline-analysis)
- [Factor investment research](use-cases.md#factor-investment-research)
- [Portfolio optimization](use-cases.md#portfolio-optimization)

## Notebooks

| Notebook | What it shows |
| --- | --- |
| [Recall-adjusted answers in two modes](notebooks/recall_adjusted_answers.ipynb) | One prompt through `ask` in the logprobs and the logits mode, offline and against live local models. Start with this one. |
| [Point-in-time prompt through the front](notebooks/pit_front.ipynb) | The catalogue, the status of each model and route, and what limits an answer. |
| [Measurement flow](notebooks/flow.ipynb) | The measurement and backtest pipeline cell by cell, with intermediate state. |
| [Model qualification](notebooks/qualification.ipynb) | The statistics of the model-ranking harness, on small and synthetic examples. |
| [Completion workflow](notebooks/completion_workflow.ipynb) | A completion API configured once, with single answers, ensembles and optional rewriting, offline. |
| [Visualize a run](notebooks/visualize_run.ipynb) | Figures for a finished harness run. |

## Guides

| Guide | Read it when |
| --- | --- |
| [Point-in-time architecture](guides/pit-architecture.md) | You want the whole design: input discipline, task design, measurement, portfolio. |
| [Ensemble consensus](guides/ensemble.md) | You need one reproducible answer and estimate from many draws. |
| [Local models](guides/local-provider.md) | You run a model on your own machine. |
| [Rented GPUs on Modal](guides/modal-provider.md) | The model is too large for your machine. |
| [Scorer integration](guides/scorer-integration.md) | You score single responses in an application with `MemoryGuardedScorer`. |
| [Completion APIs](guides/completion-api.md) | Your model is behind a completion API without token evidence. |
| [Docker and development](guides/development.md) | You work on the checkout: container, tests, CI. |

## Research

| Page | What it holds |
| --- | --- |
| [Research report](research-report.md) | What has been measured so far, in one place. |
| [Research register](research-register.md) | Every dated decision, experiment and result. |
| [All dated reports](results/index.md) | The reports themselves, oldest first. |
| [Experiment track](research/experiment-track.md) | The order of experiments and why. |
| [Twin-strategy contrast](research/twin-strategy-contrast.md) | Pricing lookahead by comparing a point-in-time strategy with its identifying twin. |
| [Divergence decoding](research/divergence-decoding.md) | An experimental way to reduce recall during generation, and its [evaluation proposal](research/divergence-decoding-evaluation.md). |
| [Benchmark](benchmark.md), [latest run](research/latest-run.md), [Modal cost accounting](research/modal-cost-report.md) | Run results and receipts. |

## What is established

- The recall estimate separates prompts that identify a period from prompts that do not, on held-out prompts of the finance simulation.
- It is not yet shown to separate months inside the training range from months after the cutoff. No model and route in the shipped catalogue is qualified, so `ask` returns the answer and the estimate for them and withholds the corrected answer.
- The controls reduce and measure recall. They do not prove that lookahead has been removed.

The applied results are published as *Computational Global Macro with AI for Risk and Portfolio Management* (Marius Ciepluch, SSRN working paper 7231358, 2026): <https://papers.ssrn.com/sol3/papers.cfm?abstract_id=7231358>.

The API reference is generated from the code and is the last section of the navigation.
