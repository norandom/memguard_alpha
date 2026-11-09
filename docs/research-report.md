# Research report: calibrating the memorization estimate, as of 2026-11-09

Recall Guard's memorization estimate behaves as the study design expects. On a
clean corpus where nothing should be detectable, it detects nothing: five models,
four runtimes and two scoring methods all give held-out discrimination at chance
and calibration at the base rate. The study now has exact, reproducible
full-vocabulary logits and logprobs from Gemma models on local hardware, on both
the integrated GPU and the NPU, which puts the next experiments on a sound
statistical footing.

Every number below comes from a dated report and a numbered entry in the
[research register](research-register.md).

## 1. Question and design

Recall Guard estimates how likely a model's answer is recalled rather than
reasoned (`p_memorized`) and calibrates that estimate, so an application can
discount recalled answers. Two properties matter:

- **Discrimination:** does the estimate rank memorized text above unseen text on
  items the calibrator never saw? Measured by held-out ROC-AUC.
- **Calibration:** do its probabilities match observed frequencies? Measured by
  the held-out Brier score against the score of always predicting the base rate,
  and by expected calibration error (ECE).

Serving is nondeterministic even at temperature 0, so one call can return a
different answer and score than the next. The [ensemble](ensemble.md) addresses
this by reducing many draws to one consensus answer and one point estimate.
Removing memorization from a model (rewriting, divergence decoding) is a
separate, optional track.

**Expected result.** The models measured here are not supposed to have predictive
capacity on this data. Establishing that the estimate stays at chance and at the
base rate when there is nothing to find is part of the research: an estimate
that "finds" memorization where none can be detected would discount honest
answers.

**Data** ([manifest](https://github.com/norandom/memguard_alpha/blob/main/data/edgar/manifest.json), R19).
SEC EDGAR 8-K press releases (EX-99.1), one public-domain source for both classes,
so only the filing date relative to each model's training cutoff separates them.
Memorized rows were filed 2015-01 to 2023-03, before every model's cutoff; control
rows 2024-09 to 2026-11, after every cutoff. Calibration 40 memorized and 100
control; held-out 40 and 40, disjoint. Gemma 4 E4B (cutoff January 2025) uses
only control rows filed after 2025-01-31: 74 calibration and 33 held-out.

**Statistics.** Calibrators are fitted on the calibration split only and frozen
before held-out evaluation. Intervals are seeded 1,000-resample stratified
bootstraps.

## 2. Results

### 2.1 Single-draw scoring (Phase A)

Features from the model's own 32-token continuation (1,024 for GPT-OSS),
top-5 logprobs, one draw per item.

| Model | Runtime and hardware | Held-out AUC [95 %] | Held-out Brier | No-skill Brier | Register |
| --- | --- | --- | ---: | ---: | --- |
| Gemma 3 4B | Ollama, iGPU | 0.574 [0.453, 0.690] | 0.253 | 0.250 | R26 |
| Gemma 3 4B | FastFlowLM, NPU | 0.599 [0.475, 0.721] | 0.243 | 0.250 | R38 |
| Gemma 4 E4B | FastFlowLM, NPU | 0.438 [0.301, 0.574] | 0.255 | 0.248 | R38 |
| Llama 3.2 3B | Ollama, iGPU | 0.503 [0.381, 0.623] | 0.259 | 0.250 | R26 |
| Llama 3.1 8B | Ollama, iGPU | 0.462 [0.339, 0.598] | 0.269 | 0.250 | R26 |
| GPT-OSS-20B | vLLM on Modal, A100 | 0.574 [0.426, 0.707] | 0.249 | 0.245 | R30 |

Every interval includes 0.5, and no Brier score beats the base-rate reference.
Mean predictions sit within 0.06 of the base rate. Gemma 3 4B agrees with itself
across runtime and quantisation. Gemma 3's calibrator reached 0.812 on its own
holdout and 0.574 on new rows, which shows why held-out evaluation is needed.
Details: [Phase A report](results/2026-11-08-phase-a-single-draw.md).

### 2.2 Document likelihood from full logits

The stronger method scores the candidate text itself: the per-token
log-likelihood of each press release under the model, computed by teacher
forcing from the full next-token distribution at every position. Features: mean
negative log-likelihood, Min-K % probability (10 % and 20 %), zlib ratio, mean
entropy and mean log rank, with a logistic calibrator.

| Model | Runtime and hardware | Positions scored | Held-out calibrator AUC [95 %] | Brier | No-skill | Register |
| --- | --- | ---: | --- | ---: | ---: | --- |
| Gemma 3 4B (exact Phase A weights) | llama.cpp, iGPU | 78,862 | 0.562 [0.439, 0.693] | 0.255 | 0.250 | R40 |
| Gemma 4 E4B (AMD int4 NPU build) | Ryzen AI OGA, NPU | 67,697 | 0.470 [0.338, 0.608] | 0.259 | 0.248 | R41 |

All twelve single-feature intervals include 0.5 except one: Gemma 4's mean
entropy (0.336 [0.221, 0.455]), which lies opposite to its calibration direction.
With six features tested at 95 %, about one run in four shows such an interval by
chance, so it is not read as a signal. Details:
[Gemma 3](results/2026-11-08-gemma3-document-likelihood.md),
[Gemma 4 on the NPU](results/2026-11-08-gemma4-document-likelihood-npu.md).

### 2.3 Earlier results

| Date | Result | Status today | Register |
| --- | --- | --- | --- |
| 2026-10-30 | GPT-OSS-20B on NIM, calibrator AUC 0.876 | The corpus separated its classes by date and source as well; read as corpus separation | R01, R24 |
| 2026-10-31 | Completion-only rewriting with Claude Opus: all 16 rewrites rejected; 3-draw ensemble +6.25 pp [0.00, 18.75] | No mitigation shown (optional track) | R02, R07 |
| 2026-11-01 | FOMC statements give an older/newer lexical signature (about +31 on rate probes) | Feasibility only; no target model | R04 |
| 2026-11-05 | S&P 500 identity contrast 0.036, no interval, four items per cohort | Too small to conclude | R10 |

## 3. Logits and logprobs on local hardware

| Source | Hardware | What it returns | Models verified | Register |
| --- | --- | --- | --- | --- |
| llama.cpp b12199 (Vulkan) | Radeon 890M iGPU | Full-vocabulary log-probabilities at every position, from the exact Ollama GGUF files | Gemma 3 4B (262,145 entries), Llama 3.2 3B (128,256) | R35, R40 |
| Ryzen AI OGA 0.18.0 | XDNA2 NPU | Full logits at every step; teacher forcing one token at a time | Llama 3.2 3B (128,256), Gemma 4 E4B (262,144) | R36, R39, R41 |
| FastFlowLM 1.0.12 | XDNA2 NPU | Top-100 logprobs (median mass per position 0.9998 or more); no raw logits | Gemma 3 4B, Gemma 4 E4B, Llama 3.2 3B | R34, R38 |
| Ollama 0.40.11 | iGPU (with `OLLAMA_IGPU_ENABLE=1`) or CPU | Top-5 logprobs of generated tokens | Gemma 3 4B, Llama 3.2 3B, Llama 3.1 8B | R12, R26 |
| vLLM 0.17.0 on Modal | A100 | Top-5 logprobs of the Harmony final answer | GPT-OSS-20B | R16, R30 |

Both full-logit routes are exact and reproducible: llama.cpp matches llama-server's
own full-vocabulary computation within 8e-7 and repeats bitwise; OGA repeats
bit-identically across processes when each document gets a fresh model load.

Logits are per model: each model scores its own vocabulary, so values compare
only within one model and tokenizer. Llama 3.2 3B and Llama 3.1 8B share a
tokenizer and could form an older/newer pair for divergence work.

All local hosts run natively on Windows behind authenticated, auto-stopping
gateways; Recall Guard itself runs in Linux containers (R18, R20).

## 4. What failed, and why

| What | Cause | Outcome | Register |
| --- | --- | --- | --- |
| GPT-OSS-20B on Ollama, HTTP 500 | 13.9 GB needed on CPU, little RAM left on a busy host | 20 GiB free-RAM preflight; use Modal or a larger machine | R15 |
| First Modal pilot rejected | Harmony logprobs were attached to final-only text (our bug) | Fixed and confirmed live | R13, R14, R16 |
| Modal Phase A attempts 1 and 2 | Reasoning truncation at 512 tokens; one evidence defect treated as fatal | Low reasoning effort, 1,024 tokens; defects now fail one item | R21, R22, R25 |
| Phi-4 Mini | Run stopped on a per-response evidence defect | Disqualified from the model set | R23 |
| LLaVA-Llama3 | The host's observed model condition changed mid-run | Stopped by design; not in the results | R32 |
| Llama 3.2 3B on FastFlowLM | Token texts did not reconstruct the answer | Set aside; OGA gives exact token IDs for this model | R37 |
| NVIDIA NIM | Repeated HTTP 504 at about 302 s, also outside Recall Guard | Treated as unavailable | R05, R09 |
| Original calibration corpus | 40 memorized rows from 3 dates, a different source from the controls | Replaced by the SEC EDGAR corpus | R24, R19 |

## 5. Limits

- 73 to 80 held-out documents per model give AUC intervals of about plus or minus
  0.12; a small effect is not ruled out.
- Single-draw results are one draw of a nondeterministic process; the ensemble
  test is still to run.
- Filing date stands in for publication date.
- Quantised builds (Q4, int4) are distinct model identities; results are reported
  per build.
- llama.cpp clamps probabilities below about 1e-38 (float32); only the extreme tail
  is affected.

## 6. Next steps

1. **Ensemble test (Phase B):** 32 draws per held-out item, analysed as two
   16-draw ensembles, to measure how much the ensemble reduces nondeterminism of
   the answer and score (R28). Modal about USD 7; inputs to be frozen and approved.
2. **Statistical power:** a larger held-out set from the same EDGAR source would
   narrow the intervals to about plus or minus 0.07.
3. **Divergence work (optional track):** an older/newer pair with a shared
   tokenizer, scored with the full-logit routes.

## Evidence

- [Research register](research-register.md): every entry R01 to R41 in date order.
- Dated reports in `docs/results/`, all in one [template](results/template.md).
- Code: `scripts/build_edgar_corpora.py`, `scripts/score_documents_llamacpp.py`,
  `scripts/score_documents_oga.py`, `recall_guard/mia/document_likelihood.py`,
  `scripts/evaluate_document_likelihood.py`, `recall_guard/core/profiles.py`.
- Private receipts and per-token scores under `runs/container/` (not published).
