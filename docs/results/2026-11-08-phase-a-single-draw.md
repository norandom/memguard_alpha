# Phase A: single-draw calibration on held-out data, 2026-11-08

> Register: R26, R30, R38 · Status: measured · Track: Local, Modal, NPU

Five models (six runs) were calibrated on one clean corpus and scored on 80 labeled rows
they never saw. None of them separates memorized from unseen press releases on
those rows, and none beats predicting the base rate. This is the expected result:
the models are not supposed to have predictive capacity here, and establishing
that is part of the research. These are single draws; the ensemble test (Phase B)
follows.

## Question

Does a freshly fitted `p_memorized` calibrator rank memorized text above unseen
text on new items (discrimination), and do its probabilities match observed
frequencies (calibration)?

## Setup

| | Local (Ollama 0.40.11, iGPU) | Modal (vLLM 0.17.0, A100) |
| --- | --- | --- |
| Models | Gemma 3 4B, Llama 3.2 3B, Llama 3.1 8B | GPT-OSS-20B, Harmony final answer |
| Generation | temperature 0, top_p 1, 32 output tokens, top-5 logprobs | temperature 0, top_p 1, `reasoning_effort: low`, 1,024 output tokens, top-5 logprobs |
| Context | 2,048 | 4,096 |
| Requests | 321 per model, 0 retries | 321, 0 retries; runner estimate USD 1.09 |
| Access | authenticated gateway, memory preflight per model | proxy auth, verified stop of both apps |

Per model: 1 probe, 40 memorized and 2 x 100 control calibration requests, 80
held-out single answers. Seed 0, `min_valid` 50, `min_auc` 0.6. Held-out
`p_memorized` comes from the same features as calibration, whether or not the
answer parses.

## Data

SEC EDGAR 8-K press releases (EX-99.1), one source for both classes, built with
`scripts/build_edgar_corpora.py` ([manifest](https://github.com/norandom/memguard_alpha/blob/main/data/edgar/manifest.json)).
Memorized rows were filed 2015-01 to 2023-03, before every model's cutoff;
control rows 2024-09 to 2026-11, after every cutoff. Calibration 40 + 100,
held-out 40 + 40, disjoint.

## Results

| Model | Calibrator holdout AUC | Held-out AUC [95 %] | Held-out Brier [95 %] | No-skill Brier | ECE | Held-out rows |
| --- | ---: | --- | --- | ---: | ---: | ---: |
| Gemma 3 4B | 0.812 | 0.574 [0.453, 0.690] | 0.253 [0.229, 0.280] | 0.250 | 0.122 | 79 |
| Llama 3.2 3B | 0.464 | 0.503 [0.381, 0.623] | 0.259 [0.242, 0.278] | 0.250 | 0.064 | 79 |
| Llama 3.1 8B | 0.544 | 0.462 [0.339, 0.598] | 0.269 [0.247, 0.293] | 0.250 | 0.090 | 79 |
| GPT-OSS-20B | 0.655 | 0.574 [0.426, 0.707] | 0.249 [0.223, 0.276] | 0.245 | 0.124 | 67 |
| Gemma 3 4B, NPU (FastFlowLM) | 0.512 | 0.599 [0.475, 0.721] | 0.243 | 0.250 | n/a | 80 |
| Gemma 4 E4B, NPU (FastFlowLM) | 0.642 | 0.438 [0.301, 0.574] | 0.255 [0.246, 0.265] | 0.248 | 0.061 | 73 |

Intervals are seeded 1,000-resample stratified bootstraps. The no-skill Brier is
that of always predicting the held-out base rate.

- Discrimination: every held-out interval includes 0.5, as expected.
- Calibration: mean predictions sit within 0.06 of the base rate, but almost
  all estimates fall between 0.3 and 0.7, so the Brier score does not beat the
  no-skill reference.
- Transfer: Gemma's calibrator looked strong on its own holdout (0.812) and
  did not carry over to new rows.
- NPU: Gemma 3 4B on the NPU agrees with its iGPU run across runtime and
  quantisation. Gemma 4 E4B uses only control rows filed after its January 2025
  cutoff (74 calibration, 33 held-out).
- GPT-OSS-20B: 44 of 321 responses had no final answer even at low reasoning
  effort, and 27 rows had duplicate top-5 alternatives; both were counted as
  failed items.

## Limits

- Single draws. Serving nondeterminism moves `p_memorized` between calls; the
  ensemble test measures how much that matters.
- 80 held-out rows per model give wide intervals; they are consistent with no
  predictive capacity but do not rule out a small effect.
- Top-5 logprobs only; full local logits or logprobs (NPU included) would
  establish the result more firmly.
- Filing date stands in for publication date.
- LLaVA-Llama3 (ctx 2048) stopped after 149 of 200 control requests because the
  host's observed model condition changed (`condition_mismatched`); no calibrator,
  no held-out scores. Phi-4 Mini is disqualified
  ([report](2026-11-08-phi4-mini-disqualified.md)).

## Evidence

Private journals and reports under `runs/container/phase-a/` (not published).
Register entries [R26, R30 and R38](../research-register.md); attempts that stopped
early are R21 and R22.
