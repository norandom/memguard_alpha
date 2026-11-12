# Use cases

Recall Guard is used where a language model's answer enters a simulation over historical dates. Each use below states what this repository provides and what lives elsewhere.

## Macro timeline analysis

The question: What regime does the model read from the macro state known at each date, and which of those readings rest on recall?

The repository provides:

- A point-in-time macro prompt: the macro state as rolling z-scores, assets by letter and category, no date, no ticker, no raw level. `worked_example()` returns one with its decision definition, five loadings between -1 and 1.
- The same prompt with the date, the tickers and the raw levels put back, as a control that invites recall.
- `ask`, which returns the loadings, the recall estimate and the corrected loadings for one prompt.
- A finance simulation that runs both prompts month by month. Its results are in the [dated report](results/2026-11-11-memorization-adjusted-finance-sim.md).

Start with the [usage notebook](notebooks/recall_adjusted_answers.ipynb) and the [point-in-time architecture](guides/pit-architecture.md).

## Factor investment research

The question: Does a signal built from model answers carry information, or does it carry what the model remembers about the period?

The repository provides:

- The recall estimate per answer, to weight or filter a signal before it becomes a factor. The scorer reports `p_memorized` and a discounted confidence, `raw_confidence * (1 - p_memorized)`; see [the scorer guide](guides/scorer-integration.md).
- An evaluation harness for direction prompts on a set of ETFs, with a gate against the majority baseline. The [benchmark page](benchmark.md) shows one run.
- [Ensembles](guides/ensemble.md): many draws of one prompt reduced to one answer and one estimate, so that a signal does not change from run to run.

Elsewhere: Factor construction and testing are in the consumer pipeline, [Global_Macro_AI_Factors](https://github.com/norandom/Global_Macro_AI_Factors).

## Portfolio optimization

The question: How much of a strategy's simulated return is lookahead?

The repository provides:

- The [twin-strategy contrast](research/twin-strategy-contrast.md): a point-in-time strategy and its identifying twin share the model, the dates, the prices and the optimizer, and differ in whether the prompt identifies the period. The gap between them prices the recall.
- A backtest module, `recall_guard.portfolio`, in the optional `backtest` extra.
- Corrected answers that scale the size of a view and never change its direction.

Elsewhere: The allocation itself (views, risk model, optimizer) is in the consumer pipeline. Its walk-forward results are published in the working paper *Computational Global Macro with AI for Risk and Portfolio Management* (SSRN 7231358): <https://papers.ssrn.com/sol3/papers.cfm?abstract_id=7231358>.

## The condition all three share

The recall estimate applies to periods on or before the model's training cutoff. For periods after the cutoff the model cannot have seen what happened next, so an estimate above zero is noise and nothing is corrected. `ask` does not know which period your data describes; compare it with the model's cutoff before you use the estimate.
