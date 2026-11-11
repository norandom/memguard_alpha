# Gemma 4 E4B document likelihood on the NPU, 2026-11-08

> Register: R41 · Status: measured · Track: NPU

Gemma 4 E4B shows no predictive capacity on held-out documents, as the study
design expects. This is the same full-logit document-likelihood method as for
Gemma 3 4B ([report](2026-11-08-gemma3-document-likelihood.md)), run on the AMD
Ryzen AI NPU. The same method on two models and two kinds of hardware gives the
same null.

> **Later evidence (2026-11-11):** the intervals in this report are row-level stratified
> bootstraps with 1,000 resamples at 95 percent (`scripts/evaluate_document_likelihood.py`).

## Question

Does the per-token likelihood of a document under Gemma 4 E4B separate press
releases filed before its training cutoff from those filed after it, on held-out
documents?

## Setup

| Field | Value |
| --- | --- |
| Model | `amd/gemma-4-e4b-it-mobius-int4_rai_1.9.0_npu` @ 5fcdd87 (int4; file hashes match Hugging Face LFS) |
| Runtime | Ryzen AI OGA 0.18.0, NPU driver 32.0.20102.3930, natively on Windows |
| Input | raw document text with BOS, no chat template |
| Route | teacher forcing one token at a time (OGA returns last-position logits only), a fresh model load per document, the only reproducible route (register R39) |
| Exactness | two fresh processes in opposite orders give bit-identical logits; rank and logprob agree with an independent computation within 4.4e-11 |
| Hardware | the scorer's own process kept the NPU compute engine busy, mean 74 % |
| Scale | 187 documents, 67,697 scored positions, 2.96 h (7.7 tokens/s plus about 7.5 s per load) |

Features, calibrator and freezing are identical to the Gemma 3 study.

## Data

SEC EDGAR corpus. Gemma 4's cutoff is January 2025, so only control rows filed
after 2025-01-31 are used: calibration 40 memorized and 74 control, held-out 40
and 33. Memorized rows all predate 2023-03.

## Results

| Feature | Sign (from calibration) | Held-out AUC [95 %] |
| --- | ---: | --- |
| mean NLL | -1 | 0.389 [0.262, 0.516] |
| Min-K 10 % | +1 | 0.415 [0.286, 0.547] |
| Min-K 20 % | +1 | 0.417 [0.291, 0.540] |
| zlib ratio | -1 | 0.603 [0.469, 0.745] |
| mean entropy | -1 | 0.336 [0.221, 0.455] |
| mean log rank | -1 | 0.386 [0.259, 0.514] |

- Calibrator, 5-fold CV on calibration: AUC 0.481 (sd 0.070).
- Calibrator on held-out: AUC 0.470 [0.338, 0.608]; Brier 0.259 [0.243, 0.276]
  against 0.248 for predicting the base rate; ECE 0.078.
- Mean entropy is the only interval that excludes 0.5, and it lies on the side
  opposite to its calibration direction. With six features tested at 95 %, about
  one interval in four is expected to miss 0.5 by chance. It is not a usable
  signal.

Comparison with Gemma 3 4B:

| Model, hardware | Held-out calibrator AUC [95 %] | Brier | Reference |
| --- | --- | ---: | ---: |
| Gemma 3 4B, iGPU (llama.cpp) | 0.562 [0.439, 0.693] | 0.255 | 0.250 |
| Gemma 4 E4B, NPU (OGA) | 0.470 [0.338, 0.608] | 0.259 | 0.248 |

## Limits

- 73 held-out documents; a small effect is not ruled out.
- Fewer control rows than for Gemma 3, because of the later cutoff.
- AMD's int4 NPU build is its own model identity, distinct from other Gemma 4
  builds.

## Evidence

Private scores, verification and evaluation under `runs/container/gemma4-doc/`.
Code: `scripts/score_documents_oga.py`, `scripts/evaluate_document_likelihood.py
--published-after 2025-01-31`. Register entry [R41](../research-register.md).
