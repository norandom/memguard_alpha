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
3. Read the [use cases](use-cases.md) for macro timeline analysis, factor investment research and portfolio optimization.

## The call

```python
from recall_guard import Mode, ask, offline_example, worked_example

prompt, definition = worked_example()   # a point-in-time macro prompt and how its reply is read
example = offline_example()             # invented entry, calibrators and replies

result = ask(prompt, definition, **example.arguments(Mode.LOGPROBS))
print(result.answer, result.p_memorized, result.corrected)
```

`ask` returns the model's answer, the recall estimate and the answer corrected by it, together with the status of the model and route it ran on. On your own model you pass a catalogue entry and a calibrator fitted on that model.

## Where things are

| Section | What it holds |
| --- | --- |
| [Use cases](use-cases.md) | What the repository provides for each of the three research uses |
| Notebooks | The usage notebook and the earlier walkthroughs |
| Guides | [Point-in-time architecture](guides/pit-architecture.md), [ensembles](guides/ensemble.md), [local models](guides/local-provider.md), [rented GPUs](guides/modal-provider.md), [the scorer](guides/scorer-integration.md), [completion APIs](guides/completion-api.md), [development setup](guides/development.md) |
| Research | The [research report](research-report.md), the [register](research-register.md) of every dated result, and the experiments |
| API reference | Generated from the code |

## What is established

- The recall estimate separates prompts that identify a period from prompts that do not, on held-out prompts of the finance simulation.
- It is not yet shown to separate months inside the training range from months after the cutoff. No model and route in the shipped catalogue is qualified, so `ask` returns the answer and the estimate for them and withholds the corrected answer.
- The controls reduce and measure recall. They do not prove that lookahead has been removed.

The applied results are published as *Computational Global Macro with AI for Risk and Portfolio Management* (Marius Ciepluch, SSRN working paper 7231358, 2026): <https://papers.ssrn.com/sol3/papers.cfm?abstract_id=7231358>.
