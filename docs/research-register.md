# Research register

Every research decision, experiment and result for Recall Guard, numbered
consecutively in date order. Each entry has the same six fields: question, setup,
data, result, limits, evidence. Entries are added, never rewritten; when a later
entry changes an earlier conclusion, both say so.

Last updated 2026-11-08. Release plan: [release 1.0](https://github.com/norandom/memguard_alpha/blob/main/.kiro/specs/release-1.0/tasks.md).

## Purpose

Recall Guard estimates how likely an answer is recalled rather than reasoned
(`p_memorized`) and calibrates that estimate, so an application can discount
recalled answers. The research question is the calibration of that estimate:
does `p_memorized` rank memorized text above unseen text (discrimination), and do
its probabilities match observed frequencies (calibration)?

Serving stacks are nondeterministic even at temperature 0, so one call can return
a different answer and score than the next. The [ensemble](ensemble.md) gives the
calibrated model's answer with less of that nondeterminism: it draws the same
prompt many times and reduces the draws to one consensus answer and one point
estimate. Phase A measures single draws; Phase B measures whether the ensemble
makes the answer and score reproducible while keeping held-out discrimination.

Reducing memorization (rewriting, divergence decoding) is a separate, optional
track. The guard does not need it to work.

## State of evidence

1. **No predictive capacity, as the study design expects.** These models are not
   supposed to separate the classes here, and establishing that is part of the
   research. Six model runs finished Phase A on one clean corpus (R26, R30, R38);
   every held-out AUC interval includes 0.5: Gemma 3 4B 0.574 (iGPU) and 0.599
   (NPU), Gemma 4 E4B 0.438 (NPU), Llama 3.2 3B 0.503, Llama 3.1 8B 0.462,
   GPT-OSS-20B 0.574. Scoring the documents themselves from full logits gives the
   same answer for Gemma 3 4B (0.562 [0.439, 0.693], R40). With 80 held-out rows per model the
   intervals are wide, so a small effect is not ruled out.
2. **Calibration is at the no-skill level, consistent with point 1.** Held-out
   Brier scores are 0.249 to 0.269 against 0.245 to 0.250 for always predicting
   the base rate; estimates cluster near 0.5 and their mean matches the base
   rate.
3. **Calibrators do not transfer.** Calibrator holdout AUC reached 0.812 for Gemma
   but 0.574 on new rows. The earlier 0.876 for GPT-OSS on NIM (R01) used a corpus
   whose classes differ in date and source (R24) and is corpus separation.
4. **The measurement chain works.** Token evidence, memory preflight,
   authenticated auto-stopping hosts and cost-capped Modal runs are verified (R15
   to R25).
5. **Full distributions are available locally.** llama.cpp gives full-vocabulary
   logprobs for the exact Phase A weights on the iGPU (R35); Ryzen AI OGA gives
   full logits on the NPU (R36); FastFlowLM gives top-100 on the NPU (R34).
   Next: the ensemble test (Phase B) and a full-distribution analysis.
6. **Mitigation (optional track):** no method has reduced recall (R02, R03, R07).

## Claims and how they could fail

| Claim | Evidence so far | We are wrong if |
| --- | --- | --- |
| These models show no predictive capacity on this data (expected) | R26, R30: four models, all intervals include 0.5 | a held-out AUC interval excludes 0.5, or Brier beats the base-rate reference |
| `p_memorized` stays at the base rate when there is nothing to detect | R26, R30: mean prediction within 0.06 of the base rate | predictions drift away from the base rate without any separation |
| The ensemble reduces nondeterminism of answer and score | none yet (Phase B) | two 16-draw ensembles of the same prompt disagree about as much as single draws |
| A calibrator fitted on one corpus transfers to new items | R26 suggests not | ensemble holdout AUC is high but held-out AUC near 0.5 |
| Mitigation reduces recall (optional) | none | paired held-out recall does not drop at a frozen setting |

## Index

| ID | Date | Track | Status | One line |
| --- | --- | --- | --- | --- |
| R01 | 2026-10-30 | NIM | measured | GPT-OSS-20B calibration and ETF evaluation on NIM |
| R02 | 2026-10-31 | Completion | failed | Rewrite mitigation and 3-draw ensemble with Opus |
| R03 | 2026-11-01 | Divergence | failed | Opus sample reweighting prototype |
| R04 | 2026-11-01 | Divergence | feasibility | FOMC corpus gives an older/newer lexical signature |
| R05 | 2026-11-01 | NIM | failed | NIM refresh: 504s, nothing measured |
| R06 | 2026-11-01 | Summary | measured | Single-answer view of R01 and R02 |
| R07 | 2026-11-01 | Summary | measured | Ensemble view of R02 |
| R08 | 2026-11-05 | Local | feasibility | First local token-evidence probes |
| R09 | 2026-11-05 | NIM | diagnostic | NIM 504 reproduced outside Recall Guard |
| R10 | 2026-11-05 | Completion | measured | S&P 500 identity pilot, pre versus post cutoff |
| R11 | 2026-11-06 | Local | feasibility | Project-local Ollama store and sequential probes |
| R12 | 2026-11-06 | Local | feasibility | Five local models return aligned logprobs |
| R13 | 2026-11-06 | Modal | failed | First Modal pilot: eligibility rejected |
| R14 | 2026-11-06 | Modal | diagnostic | The rejection was our Harmony mapping bug |
| R15 | 2026-11-08 | Local | diagnostic | Ollama GPT-OSS-20B fits RAM only barely on this host |
| R16 | 2026-11-08 | Modal | feasibility | Live Harmony eligibility for GPT-OSS-20B |
| R17 | 2026-11-08 | Local | diagnostic | LLaVA context mismatch traced to its model file |
| R18 | 2026-11-08 | Platform | feasibility | Backend profiles, memory preflight, container-only Windows |
| R19 | 2026-11-08 | Data | decision | SEC EDGAR calibration and held-out corpus built |
| R20 | 2026-11-08 | Local | feasibility | Windows Ollama with iGPU behind an authenticated gateway |
| R21 | 2026-11-08 | Modal | failed | Phase A attempt 1 stopped after 4 requests |
| R22 | 2026-11-08 | Modal | failed | Phase A attempt 2 stopped after 4 requests |
| R23 | 2026-11-08 | Local | disqualified | Phi-4 Mini removed from the model set |
| R24 | 2026-11-08 | Data | decision | R01's corpus reclassified as confounded |
| R25 | 2026-11-08 | Platform | fix | Per-response evidence defects fail one item, not the run |
| R26 | 2026-11-08 | Local | measured | Phase A single draw: Gemma 3 4B, Llama 3.2 3B, Llama 3.1 8B |
| R27 | 2026-11-08 | Method | decision | Purpose: calibration, with the ensemble against nondeterminism |
| R28 | 2026-11-08 | Method | decision | Phase B sized at 32 draws, split 16/16 |
| R29 | 2026-11-08 | Platform | feasibility | Held-out calibration quality added to reports |
| R30 | 2026-11-08 | Modal | measured | Phase A single draw: GPT-OSS-20B |
| R31 | 2026-11-08 | NPU | decision | NPU runtime with logprobs: FastFlowLM selected |
| R32 | 2026-11-08 | Local | stopped | LLaVA-Llama3 Phase A stopped on a host condition change |
| R33 | 2026-11-08 | Method | decision | Full logits or logprobs from local models, NPU included |
| R34 | 2026-11-08 | NPU | feasibility | Three models eligible on the NPU through FastFlowLM, top-100 logprobs |
| R35 | 2026-11-08 | Local | feasibility | Full-vocabulary logprobs from the Phase A GGUF weights via llama.cpp |
| R36 | 2026-11-08 | NPU | feasibility | Full logits on the NPU via Ryzen AI OGA |
| R37 | 2026-11-08 | Method | decision | Gemma focus: document-likelihood scoring with full logits |
| R38 | 2026-11-08 | NPU | measured | Phase A single draw on the NPU: Gemma 3 4B and Gemma 4 E4B |
| R39 | 2026-11-08 | NPU | feasibility | Gemma 4 E4B full logits and teacher forcing on the NPU via OGA |
| R40 | 2026-11-08 | Local | measured | Gemma 3 4B document likelihood from full logits |

## Entries

### R01 · 2026-10-30 · GPT-OSS-20B calibration on NIM

- **Question:** a hosted scoring baseline with a freshly fitted calibrator.
- **Setup:** NVIDIA NIM `openai/gpt-oss-20b`, temperature 0, 512 tokens, seed 0, 1,000 bootstrap resamples.
- **Data:** FMP corpus, 40 memorized and 100 control rows; 330 ETF prompts.
- **Result:** calibrator holdout AUC 0.876; 312 of 330 parsed; 170 of 312 correct (54.49 %) against a 53.94 % majority baseline.
- **Limits:** the interval is a point estimate; the corpus is confounded (R24).
- **Evidence:** [report](results/2026-10-30-nim-measurement.md).

### R02 · 2026-10-31 · Completion-only rewrite mitigation

- **Question:** does rewriting reduce recall, and does a 3-draw ensemble help?
- **Setup:** Claude Opus 5.5, 67 requests; NIM eligibility failed three times.
- **Data:** 16 items, 8 before and 8 after the cutoff.
- **Result:** all 16 rewrites rejected; single 11/16 against ensemble 12/16, +6.25 pp [0.00, 18.75].
- **Limits:** small dependent sample; the NIM arm never ran.
- **Evidence:** [report](results/2026-10-31-completion-api-pilot.md).

### R03 · 2026-11-01 · Opus sample reweighting

- **Question:** can completion samples be reweighted with auxiliary counts?
- **Setup:** Claude Opus 5.5, 9 requests, no logprobs.
- **Data:** one prompt; synthetic counts.
- **Result:** no change at any alpha.
- **Limits:** synthetic counts; discard as evidence either way.
- **Evidence:** [report](results/2026-11-01-anthropic-reweighting-prototype.md).

### R04 · 2026-11-01 · FOMC lexical signature

- **Question:** does a dated corpus give an older/newer statistical signature?
- **Setup:** CPU trigram scorer, no API calls, 12 probes.
- **Data:** FOMC statements 2019 against 2019 plus 2020.
- **Result:** exposure shift of about +31 on the rate probes; zero on identical references.
- **Limits:** no held-out split, no target model.
- **Evidence:** [report](results/2026-11-01-fomc-statistical-signature.md).

### R05 · 2026-11-01 · NIM refresh

- **Question:** fresh NIM eligibility, calibration and ensemble baseline.
- **Setup:** GPT-OSS-20B in Docker, ceiling 146 requests.
- **Data:** planned 20 + 20 calibration items.
- **Result:** 3 requests, two 504s; nothing measured.
- **Limits:** a 504 does not locate the fault.
- **Evidence:** [report](results/2026-11-01-nim-refresh.md).

### R06 · 2026-11-01 · Single-answer evidence summary

- **Question:** what do R01 and R02 say about single answers?
- **Setup:** restatement, no new requests.
- **Data:** R01, R02.
- **Result:** AUC 0.876; 54.49 % correct; mean confidence 0.599 raw against 0.222 discounted.
- **Limits:** lower confidence is not reduced lookahead; R01's corpus is confounded (R24).
- **Evidence:** [report](results/2026-11-01-recall-guard-single.md).

### R07 · 2026-11-01 · Ensemble evidence summary

- **Question:** what does R02 say about ensembles?
- **Setup:** restatement, no new requests.
- **Data:** 16 items with 3 draws.
- **Result:** 11/16 against 12/16; no aggregation benefit shown.
- **Limits:** small dependent sample.
- **Evidence:** [report](results/2026-11-01-recall-guard-ensemble.md).

### R08 · 2026-11-05 · First local token-evidence probes

- **Question:** do local endpoints give scoreable token evidence?
- **Setup:** Ollama 0.40.11 on CPU, 2 probes.
- **Data:** one prompt per model.
- **Result:** Gemma 3 4B eligible; GPT-OSS-20B failed alignment; the NPU gate failed.
- **Limits:** no repeats.
- **Evidence:** [report](results/2026-11-05-local-provider-feasibility.md).

### R09 · 2026-11-05 · NIM 504 diagnostic

- **Question:** is the NIM 504 caused by Recall Guard?
- **Setup:** 2 direct requests outside Recall Guard.
- **Data:** one prompt.
- **Result:** 504 at about 302 s in both clients; the key is sent correctly.
- **Limits:** root cause unknown.
- **Evidence:** [report](results/2026-11-05-nim-request-diagnostic.md).

### R10 · 2026-11-05 · S&P 500 identity pilot

- **Question:** is there a historical identity advantage before the cutoff?
- **Setup:** Opus 25 requests, NIM 1 (504).
- **Data:** 16 dates, 48 prompts.
- **Result:** +0.011 before, -0.025 after; contrast 0.036 without an interval.
- **Limits:** four observations per cohort.
- **Evidence:** [report](results/2026-11-05-sp500-pilot.md).

### R11 · 2026-11-06 · Project-local Ollama store

- **Question:** does a project-local model store work?
- **Setup:** 3 probes at context 2,048.
- **Data:** 28 files, 22.7 GB, hashes matched.
- **Result:** Gemma eligible; GPT-OSS HTTP 500; LLaVA context 4,096.
- **Limits:** the 500 cause was unassigned (see R15).
- **Evidence:** [report](results/2026-11-06-local-model-storage.md).

### R12 · 2026-11-06 · Five-model logprob smoke

- **Question:** do five local models give aligned logprobs?
- **Setup:** 5 requests on CPU, 32 tokens.
- **Data:** one prompt per model.
- **Result:** Phi-4 Mini, Llama 3.1 8B, Llama 3.2 3B, Gemma valid; LLaVA context mismatch.
- **Limits:** partial top-k.
- **Evidence:** [report](results/2026-11-06-local-model-smoke.md).

### R13 · 2026-11-06 · First Modal pilot

- **Question:** host GPT-OSS-20B on Modal and pilot it.
- **Setup:** vLLM 0.17.0, 1 of 8 requests sent.
- **Data:** 8 prompts.
- **Result:** eligibility rejected for alignment; gross USD 0.158.
- **Limits:** reclassified by R14.
- **Evidence:** [report](results/2026-11-06-modal-pilot.md).

### R14 · 2026-11-06 · Harmony mapping repair

- **Question:** why did R13's alignment fail?
- **Setup:** 1 Modal request with token IDs, then offline replay.
- **Data:** one prompt.
- **Result:** full Harmony logprobs were attached to final-only text; the fixed adapter passes on replay.
- **Limits:** replay only; confirmed live by R16.
- **Evidence:** [report](results/2026-11-06-gpt-oss-response-mapping.md).

### R15 · 2026-11-08 · Ollama GPT-OSS-20B on a 32 GB host

- **Question:** why did R11's GPT-OSS request return HTTP 500?
- **Setup:** load-only request.
- **Data:** none.
- **Result:** Ollama drops the iGPU by default and runs on CPU; the model needs 13.9 GB, leaving 1.7 GB free. A free-RAM requirement of 20 GiB now guards the profile.
- **Limits:** the 500 body was not kept; the cause stays a hypothesis.
- **Evidence:** release plan C1, M1.4.

### R16 · 2026-11-08 · Live Harmony eligibility on Modal

- **Question:** does R14's fix hold on a new generation?
- **Setup:** 1 request through the Harmony client, after a 401 check without credentials.
- **Data:** one prompt.
- **Result:** eligible: 5 alternatives per final-answer token; runner estimate USD 0.094.
- **Limits:** one response.
- **Evidence:** release plan M2.1.

### R17 · 2026-11-08 · LLaVA context mismatch

- **Question:** why does LLaVA load at 4,096 tokens?
- **Setup:** load-only requests.
- **Data:** `llava-llama3:latest`.
- **Result:** its model file sets `num_ctx 4096`; a derived `llava-llama3:ctx2048` tag loads at 2,048.
- **Limits:** token evidence of the new tag is checked in R26.
- **Evidence:** release plan M3.1.

### R18 · 2026-11-08 · Backend profiles and container-only Windows

- **Question:** can adopters choose Ollama, Modal or NIM safely?
- **Setup:** provider profiles, `connect()`, free-RAM preflight, container execution on Windows.
- **Data:** offline tests; full suite in Linux 2,984 passed.
- **Result:** one factory for all backends; a model is never loaded without the memory it needs.
- **Limits:** NIM still unavailable.
- **Evidence:** release plan M1.

### R19 · 2026-11-08 · SEC EDGAR corpus

- **Question:** build a calibration corpus where only the date separates the classes.
- **Setup:** `scripts/build_edgar_corpora.py`, seed 0, 1,118 SEC requests.
- **Data:** 8-K press releases: memorized 2015-01 to 2023-03 (40 dates, 40 companies), control 2024-09 to 2026-11.
- **Result:** calibration 40/100 and held-out 40/40, disjoint and hashed.
- **Limits:** filing date stands in for publication date; some titles are layout noise.
- **Evidence:** [manifest](https://github.com/norandom/memguard_alpha/blob/main/data/edgar/manifest.json).

### R20 · 2026-11-08 · Windows Ollama with iGPU behind a gateway

- **Question:** can containerized Recall Guard use the GPU safely?
- **Setup:** `scripts/serve_local_ollama.ps1`: loopback Ollama with `OLLAMA_IGPU_ENABLE=1`, bearer-token gateway on the WSL adapter, 30-minute idle stop.
- **Data:** load-only check.
- **Result:** 35 of 35 layers on the Radeon 890M; 200 with the token, 401 without; stop verified.
- **Limits:** none for the measurement.
- **Evidence:** release plan M3.3.

### R21 · 2026-11-08 · Modal Phase A, attempt 1

- **Question:** as R30.
- **Setup:** 512 output tokens, default reasoning.
- **Data:** as R30.
- **Result:** stopped after 4 requests: one response spent all 512 tokens on reasoning, and the harness treated that as an unknown outcome. USD 0.12.
- **Limits:** nothing measured.
- **Evidence:** fix da2f55c.

### R22 · 2026-11-08 · Modal Phase A, attempt 2

- **Question:** as R30.
- **Setup:** `reasoning_effort: low`, 1,024 output tokens.
- **Data:** as R30.
- **Result:** stopped after 4 requests: one response failed a per-response evidence check, which the harness treated as fatal. USD 0.10.
- **Limits:** nothing measured.
- **Evidence:** fix in R25.

### R23 · 2026-11-08 · Phi-4 Mini disqualified

- **Question:** keep Phi-4 Mini after its Phase A stop?
- **Setup:** as R26.
- **Data:** as R26.
- **Result:** stopped at request 119 of 321 on a per-response evidence defect; user decision: disqualify, no rerun.
- **Limits:** the defect code was not recorded.
- **Evidence:** [report](results/2026-11-08-phi4-mini-disqualified.md).

### R24 · 2026-11-08 · R01 corpus reclassified

- **Question:** is the corpus behind R01 fit for a held-out test?
- **Setup:** audit of `data/calibration/`.
- **Data:** 40 memorized rows from 3 dates (20 on 2023-12-31, 19 on 2020-12-30); control rows from a different source.
- **Result:** label is confounded with date and source; R01's 0.876 is corpus separation. Replaced by R19.
- **Limits:** R30 re-measures GPT-OSS-20B on the new corpus, on Modal rather than NIM.
- **Evidence:** release plan D1.

### R25 · 2026-11-08 · Evidence defects fail one item

- **Question:** should one imperfect response stop a measurement?
- **Setup:** root-cause analysis of R22 with synthetic Harmony responses.
- **Data:** offline.
- **Result:** duplicate alternatives, empty output and similar defects now fail one item with a reported code; identity or settings drift still stops the run.
- **Limits:** none.
- **Evidence:** commit 0c02b1e.

### R26 · 2026-11-08 · Phase A single draw, local models

- **Question:** does a fresh calibrator separate memorized from control press releases on 80 held-out rows?
- **Setup:** Windows Ollama 0.40.11 on the iGPU through R20; context 2,048, 32 output tokens, top-5, temperature 0, 0 retries; 321 requests per model.
- **Data:** R19.
- **Result:**

  | Model | Calibrator holdout AUC | Held-out AUC [95 %] | Held-out Brier [95 %] | Reference | ECE |
  | --- | ---: | --- | --- | ---: | ---: |
  | Gemma 3 4B | 0.812 | 0.574 [0.453, 0.690] | 0.253 [0.229, 0.280] | 0.250 | 0.122 |
  | Llama 3.2 3B | 0.464 | 0.503 [0.381, 0.623] | 0.259 [0.242, 0.278] | 0.250 | 0.064 |
  | Llama 3.1 8B | 0.544 | 0.462 [0.339, 0.598] | 0.269 [0.247, 0.293] | 0.250 | 0.090 |

  LLaVA-Llama3 (ctx 2048): R32. Phi-4 Mini: R23. As the study design expects,
  no model shows predictive capacity here.
- **Limits:** single draws; partial top-5 logprobs; input token counts are upper bounds.
- **Evidence:** [report](results/2026-11-08-phase-a-single-draw.md).

### R27 · 2026-11-08 · Purpose clarified

- **Question:** what is Recall Guard for?
- **Setup:** user decision.
- **Data:** none.
- **Result:** calibration of `p_memorized`, not removal of memorization; the ensemble gives the calibrated answer with less serving nondeterminism.
- **Limits:** none.
- **Evidence:** README and [docs home](index.md).

### R28 · 2026-11-08 · Phase B sizing

- **Question:** how many draws test the ensemble?
- **Setup:** user decision.
- **Data:** none.
- **Result:** 32 draws per held-out item, analysed as two 16-draw ensembles; 2,560 requests per model; Modal about USD 7.
- **Limits:** inputs and limits still to be frozen and approved.
- **Evidence:** release plan measurement plan.

### R29 · 2026-11-08 · Held-out calibration quality

- **Question:** do the probabilities match observed frequencies?
- **Setup:** Brier score with seeded bootstrap, base-rate reference, 10-bin reliability, ECE.
- **Data:** report rows; no new requests.
- **Result:** in all reports from now on; applied to R26 and R30.
- **Limits:** none.
- **Evidence:** commit a9df539.

### R30 · 2026-11-08 · Phase A single draw, GPT-OSS-20B on Modal

- **Question:** as R26.
- **Setup:** Modal A100, Harmony final answer, `reasoning_effort: low`, 1,024 output tokens, top-5, 0 retries; 321 requests; GPU phase 1,366 s plus 133 s cold start; runner estimate USD 1.09; both apps verified stopped.
- **Data:** R19.
- **Result:** calibrator holdout AUC 0.655; held-out AUC 0.574 [0.426, 0.707] on 67 rows; Brier 0.249 [0.223, 0.276] against 0.245; ECE 0.124. 44 responses had no final answer and 27 rows had duplicate top-5 alternatives (R25 counted them as failed items).
- **Limits:** 14 % of responses gave no final answer even at low effort.
- **Evidence:** [report](results/2026-11-08-phase-a-single-draw.md).

### R31 · 2026-11-08 · NPU runtime with logprobs

- **Question:** can a model run on the Ryzen AI NPU with per-token logprobs?
- **Setup:** web research; NPU present (driver 32.0.20102.3930).
- **Data:** none yet.
- **Result:** Ollama does not use the NPU. FastFlowLM v1.0.12 runs on XDNA2 and documents `logprobs` with `top_logprobs` since v1.0.11. Selected models: `gemma3:4b`, `llama3.2:3b`, `gemma4-it:e4b` (Gemma 4, cutoff January 2025, so its control rows must postdate 2025-01-31). Installation and eligibility probes are in progress, natively on Windows.
- **Limits:** logprob behaviour on these models is not yet observed.
- **Evidence:** release plan, NPU option.

### R32 · 2026-11-08 · LLaVA-Llama3 Phase A stopped

- **Question:** as R26.
- **Setup:** as R26, `llava-llama3:ctx2048`.
- **Data:** R19.
- **Result:** stopped after 149 of 200 control requests (memorized 39 of 40 done):
  the host's observed model condition changed during the run
  (`condition_mismatched`), which stops a run by design. No calibrator, no
  held-out scores.
- **Limits:** the changed field is not yet identified.
- **Evidence:** `runs/container/phase-a/ollama-llava-llama3-ctx2048/` (private).

### R33 · 2026-11-08 · Full local logits or logprobs

- **Question:** what would establish the experiments more firmly?
- **Setup:** user decision.
- **Data:** none.
- **Result:** obtain full next-token logits or complete logprobs from local models,
  NPU included, in addition to top-k lists. This removes the partial top-k limit
  and is the basis for the divergence work.
- **Limits:** depends on runtime support (R31).
- **Evidence:** release plan, NPU option.

### R34 · 2026-11-08 · NPU eligibility through FastFlowLM

- **Question:** do NPU-served models give scoreable token evidence?
- **Setup:** FastFlowLM 1.0.12 natively on Windows, NPU driver 32.0.20102.3930
  (`flm validate` ready), loopback only; one 5-token request per model, repeated;
  temperature 0.
- **Data:** one prompt.
- **Result:** `llama3.2:3b`, `gemma3:4b` and `gemma4-it:e4b` (Q4_1) are eligible:
  aligned tokens, sorted alternatives, identical logprobs on repeat; NPU load 61 to
  94 % during requests. `top_logprobs` goes up to 100 at no measurable cost; over
  40-token generations the median top-100 mass per position is 0.9998 or more.
  No raw logits are exposed, so top-100 is the ceiling. Gemma 3 sums can exceed 1
  by up to 1.8e-6 from float32 rounding.
- **Limits:** quantised (Q4_1) builds; one short prompt per model.
- **Evidence:** release plan, NPU option; models in `models/flm/`.

### R35 · 2026-11-08 · Full-vocabulary logprobs via llama.cpp

- **Question:** can a local host give full next-token distributions for the exact
  weights measured in Phase A?
- **Setup:** llama.cpp b12199 (commit 372dc5ca1, Vulkan, installed by the user
  with winget) natively on Windows, loopback only; the Ollama GGUF blobs loaded
  read-only; `llama-server -ngl 99 -dev Vulkan0 -fa off`, `/completion` with
  `n_probs` equal to the vocabulary size, temperature 0.
- **Data:** one prompt rendered with Ollama's template.
- **Result:** every generated position returns all 262,145 (Gemma 3 4B) or
  128,256 (Llama 3.2 3B) entries; probabilities sum to 1 within 5e-6; the argmax
  is the generated token; repeat runs are byte-identical after one warm-up
  request; all layers on the Radeon 890M. A full-vocabulary response is about
  80 MB (Gemma) or 24 MB (Llama) for 3 tokens.
- **Limits:** the first request after loading can differ, so one warm-up request
  is required; flash attention changes values by about 2e-3, so it is pinned;
  probabilities below about 1e-38 floor at -3.4e38 (float32), so the extreme tail
  is not exact; the llama.cpp chat template differs from Ollama's for Llama 3.2,
  so prompts are rendered with Ollama's template.
- **Evidence:** release plan R33 follow-up.

### R36 · 2026-11-08 · Full logits on the NPU via Ryzen AI OGA

- **Question:** can the NPU give full next-token logits, not only top-k?
- **Setup:** `onnxruntime-genai-directml-ryzenai` 0.18.0 (AMD's Ryzen AI 1.9
  wheel index, no MSI, no admin) in a project-local venv, natively on Windows; NPU
  driver 32.0.20102.3930; greedy decoding, `Generator.get_logits()` per step.
- **Data:** one prompt; model `amd/Llama-3.2-3B-Instruct_rai_1.9.0_npu` at
  revision 315fc90 (public, `model.onnx` sha256 22b9b161…).
- **Result:** logits of shape (1, 1, 128256) equal to the vocabulary size at every
  step, all finite; softmax sums to 1; argmax equals the generated token up to
  end-of-sequence; repeat runs bit-identical; NPU compute engine at 92 % from our
  process; about 14 to 15 tokens/s, model load about 5 s. AMD's Gemma 4 E4B
  (262,144) and Qwen3 4B (151,936) NPU builds also declare full-vocabulary heads
  (checked by config only).
- **Limits:** logits after end-of-sequence are padding and must be ignored (stop
  on `is_done()`); the Gemma 3 "logit-pruning" build is not reachable and is
  avoided; AMD's NPU build is a different quantisation from the Ollama GGUF and
  the FastFlowLM build, so it is its own model identity.
- **Evidence:** release plan R33 follow-up; model in `models/oga/`.

### R37 · 2026-11-08 · Gemma focus and document-likelihood scoring

- **Question:** what is the statistically sound way forward?
- **Setup:** user decision.
- **Data:** none.
- **Result:** focus on Gemma. Score the candidate text itself rather than the
  model's continuation: per-token log-likelihood of each press release from full
  logits, with features such as mean negative log-likelihood, Min-K% probability
  and a compression-normalised score. Gemma 3 4B runs through llama.cpp on the iGPU
  with the exact Phase A weights (R35); Gemma 4 E4B gets a full-logit check on the
  NPU through Ryzen AI OGA (R36), then the same scoring. Same calibration and
  held-out splits (R19), same held-out metrics (R29). Runs before the paid
  ensemble test. Llama on FastFlowLM, which stopped on a token-text alignment
  check after 16 requests, is set aside.
- **Limits:** 80 held-out rows per model give AUC intervals of about plus or
  minus 0.12; a small effect may stay undetected.
- **Evidence:** release plan.

### R38 · 2026-11-08 · Phase A single draw on the NPU

- **Question:** as R26, on the NPU.
- **Setup:** FastFlowLM 1.0.12 natively on Windows, NPU, Q4_1 builds, behind the
  authenticated auto-stopping gateway; context 2,048, 32 output tokens, top-5,
  temperature 0, 0 retries; one FLM process per model, stopped and verified.
- **Data:** R19. Gemma 4 E4B (cutoff January 2025) uses only control rows filed
  after 2025-01-31: 74 calibration and 33 held-out control rows, so 262 requests.
- **Result:**

  | Model | Calibrator holdout AUC | Held-out AUC [95 %] | Held-out Brier [95 %] | Reference | Held-out rows |
  | --- | ---: | --- | --- | ---: | ---: |
  | Gemma 3 4B (NPU) | 0.512 | 0.599 [0.475, 0.721] | 0.243 | 0.250 | 80 |
  | Gemma 4 E4B (NPU) | 0.642 | 0.438 [0.301, 0.574] | 0.255 [0.246, 0.265] | 0.248 | 73 |

  The expected null holds on the NPU. Gemma 3 4B agrees with its iGPU run
  (0.574 [0.453, 0.690], R26) across runtime and quantisation. Llama 3.2 3B on
  FastFlowLM stopped after 16 requests on `token_text_alignment_failed` and is set
  aside (R37).
- **Limits:** single draws, top-5; Gemma 4 has fewer control rows.
- **Evidence:** [Phase A report](results/2026-11-08-phase-a-single-draw.md).

### R39 · 2026-11-08 · Gemma 4 E4B full logits on the NPU

- **Question:** can Gemma 4 give full logits on the NPU, at every position of a
  given document?
- **Setup:** Ryzen AI OGA 0.18.0, natively on Windows; model
  `amd/gemma-4-e4b-it-mobius-int4_rai_1.9.0_npu` at revision 5fcdd87 (public;
  file hashes match the Hugging Face LFS hashes; 19.6 GB).
- **Data:** one prompt; one 293-token press release from the held-out set.
- **Result:** logits of shape (1, 1, 262144) at every step, finite, softmax sums
  to 1, argmax equals the generated token, repeat runs bit-identical; the NPU does
  the work. OGA returns logits for the last position only, so documents are
  scored by appending one token at a time (about 6 tokens/s). Reproducible only
  with a fresh model load per document: several generators on one loaded model
  give differing decode logits (up to 1.8 in logprob).
- **Limits:** prefill and decode give slightly different logits for the same
  position (up to 0.32 in logprob, same argmax), so every document is scored the
  same way (decode route, fresh load). About 4.4 h for 220 documents.
- **Evidence:** release plan R37.

### R40 · 2026-11-08 · Gemma 3 4B document likelihood

- **Question:** does the full-distribution likelihood of the document itself
  separate memorized from control press releases on held-out data?
- **Setup:** exact Phase A GGUF on llama.cpp b12199 (iGPU), teacher forcing on
  raw text, exact against llama-server within 8e-7; features mean NLL, Min-K
  10 % and 20 %, zlib ratio, entropy, log rank; logistic calibrator on the
  calibration split, frozen before held-out.
- **Data:** R19; 220 documents, 78,862 scored positions.
- **Result:** held-out calibrator AUC 0.562 [0.439, 0.693], Brier 0.255 [0.226,
  0.282] against 0.250, ECE 0.097; single features 0.415 to 0.599, all intervals
  include 0.5. No predictive capacity, as the design expects, now with full
  distributions over the whole text.
- **Limits:** 80 held-out documents.
- **Evidence:** [report](results/2026-11-08-gemma3-document-likelihood.md).

## How to add an entry

Take the next number, give the date and track, and fill the six fields from the
run's own report. Record failures and pending runs too. If a result changes an
earlier conclusion, say so in both entries and update the state of evidence.
