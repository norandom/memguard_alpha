# Requirements Document

## Project Description (Input)
**Who has the problem.** Quant researchers and Recall Guard adopters who run finance
simulations (backtests, event studies, forecasting) with LLMs. Historical accuracy
can come from recall of training data rather than reasoning, which inflates
pre-cutoff results (look-ahead bias).

**Current situation.** Recall Guard estimates and calibrates `p_memorized`, and
the release-1.0 research established the expected null: on a clean SEC EDGAR
corpus, five models across four runtimes show no predictive capacity, with
single-draw scoring and with full-logit document likelihood (register R26, R30,
R38, R40, R41). Exact full logits and logprobs are available for Gemma on local
hardware: llama.cpp on the iGPU and Ryzen AI OGA on the NPU (R35, R36, R39).
There is no importable API that runs a finance simulation and adjusts each
answer for memorization in the background, no positive control where
memorization is known to exist, and no test of whether the adjustment or the
ensemble matters.

**What should change.** An importable API that calls a model, returns its answer
and, in the background, adjusts the answer's confidence for memorization using
the calibrated `p_memorized` computed from the response's own logprobs (top-k) or
full logits, with and without the ensemble strategy (which reduces serving
nondeterminism). A study must show statistically whether the adjustment and the
ensemble each matter:

- Positive control, where memorization is known to exist. Option A: a
  known-memorized public pre-cutoff corpus. Option B: planted canary text in a
  small fine-tune. To be chosen in requirements.
- Negative control: the current SEC EDGAR setting, where the expected result is
  no material change (no false discount of honest answers).
- Variants: no adjustment, logprob-adjusted, full-logit-adjusted, ensemble plus
  adjustment.
- Pre-registered paired tests: change in the pre- minus post-cutoff accuracy gap
  with bootstrap intervals; Brier and log-loss differences; ensemble run-to-run
  variance reduction; Holm correction across variants.
- Quantify how top-k logprob features converge to full-logit features within one
  model.
- Data visualization with one clear worked example that makes the point visible
  (for example the pre/post-cutoff gap before and after adjustment, reliability
  diagrams, top-k versus full-logit convergence), published in the docs.

Constraints: runs on local hardware (Gemma via llama.cpp on the iGPU and Ryzen AI
OGA on the NPU) with optional Modal; on Windows, Recall Guard runs in Linux
containers only and model hosts run natively; reports follow
`docs/results/template.md` and the rules in `.kiro/steering/research-reports.md`.

## Requirements
<!-- Will be generated in /kiro-spec-requirements phase -->


