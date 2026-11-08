# Gemma 3 4B document likelihood from full logits, 2026-11-08

Scoring the press-release text itself, from the full 262,145-entry next-token
distribution at every position, gives the same answer as the single-draw
measurement: Gemma 3 4B shows no predictive capacity on held-out data. That is
the result the study design expects. The method is stronger than Phase A's,
because it uses every token of the candidate text and exact probabilities.

## Question

Does the per-token likelihood of a document under Gemma 3 4B separate press
releases filed before its training cutoff from those filed after it, on held-out
documents?

## Setup

| Field | Value |
| --- | --- |
| Model | `gemma3:4b`, the exact Phase A GGUF (sha256 f06e91e1…), Q4_K_M |
| Runtime | llama.cpp b12199 (372dc5ca1), Vulkan on the Radeon 890M, `-ngl 99 -fa off --swa-full`, context 2,048, one warm-up |
| Input | raw document text with BOS, no chat template |
| Route | teacher forcing, one token per decode on a growing KV cache: the computation llama-server performs per prefix |
| Exactness | matches llama-server's full-vocabulary vectors within 8e-7 (logprob) after float64 normalisation; ranks identical; repeat runs bitwise identical |
| Scale | 220 documents, 78,862 scored positions, 6,080 s |

The scorer calls the llama.cpp library from the project's native Python on the
model-host side. Features and evaluation run in the dev container, offline.

## Features and calibrator

Per document: mean negative log-likelihood, Min-K % probability (k = 10 %, 20 %),
zlib ratio (NLL sum over compressed length), mean entropy and mean log rank.
Logistic regression, class-balanced, seed 0, fitted on the calibration split
only. The feature list, seed, feature signs and coefficients were frozen in
`frozen.json` before any held-out feature was computed.

## Data

SEC EDGAR corpus ([manifest](https://github.com/norandom/memguard_alpha/blob/main/data/edgar/manifest.json)):
calibration 40 memorized and 100 control, held-out 40 and 40.

## Results

| Feature | Sign (from calibration) | Held-out AUC [95 %] |
| --- | ---: | --- |
| mean NLL | -1 | 0.415 [0.293, 0.547] |
| Min-K 10 % | +1 | 0.508 [0.383, 0.639] |
| Min-K 20 % | +1 | 0.458 [0.334, 0.596] |
| zlib ratio | -1 | 0.599 [0.462, 0.728] |
| mean entropy | -1 | 0.474 [0.356, 0.598] |
| mean log rank | -1 | 0.464 [0.343, 0.590] |

- Calibrator, 5-fold CV on calibration: AUC 0.633 (sd 0.120).
- Calibrator on held-out: AUC 0.562 [0.439, 0.693]; Brier 0.255 [0.226, 0.282]
  against 0.250 for predicting the base rate; ECE 0.097.
- Every interval includes 0.5. Mean NLL points the other way on held-out than
  on calibration, so its direction does not carry over.

## Limits

- 80 held-out documents; a small effect is not ruled out.
- Filing date stands in for publication date.
- Probabilities below about 1e-38 are clamped by llama.cpp's float32 softmax;
  this affects only the extreme tail.

## Evidence

Private scores, verification references and evaluation under
`runs/container/gemma-doc/`. Code: `scripts/score_documents_llamacpp.py`,
`recall_guard/mia/document_likelihood.py`, `scripts/evaluate_document_likelihood.py`.
Register entry [R40](../research-register.md).
