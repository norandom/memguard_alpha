# Research register

Every research decision, experiment and result for Recall Guard, dated and
summarized in one shape: question, setup, data, result, limits, evidence. Entries
are append-only. A later entry may supersede an earlier one; it says so, and the
earlier entry stays.

Last updated 2026-11-08. Release plan: [release 1.0](https://github.com/norandom/memguard_alpha/blob/main/.kiro/specs/release-1.0/tasks.md).

## Purpose

Recall Guard estimates how likely an answer is recalled rather than reasoned
(`p_memorized`) and calibrates that estimate, so an application can discount
recalled answers. The research question is therefore **calibration of the
memorization estimate**: does `p_memorized` rank memorized text above unseen
text (discrimination), and do its probabilities match observed frequencies
(calibration)? Reducing memorization, by rewriting or divergence decoding, is a
separate optional track and not what the guard needs in order to work.

Serving stacks are nondeterministic even at temperature 0, so one call can
return a different answer and score than the next. The ensemble gives the answer
of the calibrated model with less of that nondeterminism: it draws the same
prompt many times at production settings and reduces the draws to one consensus
answer and one point estimate (`p_memorized_point`, the mean; [ensemble
scoring](ensemble.md), 64 draws, at least 24 parsed). Phase A measures a single
draw; Phase B measures whether the ensemble makes the answer and score
reproducible while keeping the held-out discrimination.

## State of evidence

1. **Single-draw discrimination on held-out data: not shown yet.** Phase A (the
   single-draw baseline) has held-out results for two local models. Gemma 3 4B reached 0.812 on its calibrator
   holdout but 0.574 [0.453, 0.690] on 80 new rows. Llama 3.2 3B reached 0.503
   [0.381, 0.623]. Both intervals include 0.5 (R20).
2. **Calibration quality: not measured yet.** Reports carry AUC only; Brier score
   and reliability are not computed (open item).
3. **The earlier strong number is confounded.** The 2026-10-30 NIM calibrator
   reached 0.876 (R01) on a corpus whose classes differ in date and source (R18).
   Treat it as corpus separation.
4. **The infrastructure is verified.** Token evidence works locally and on Modal
   (R12, R16), with memory preflight and authenticated, auto-stopping hosts (R15,
   R19). Two Modal attempts stopped on harness defects that are now fixed (R22, R23).
5. **Mitigation (optional track):** no rewrite or decoding method has reduced
   recall (R02, R03, R06).

## Claims and how they could fail

| Claim | Evidence so far | We are wrong if |
| --- | --- | --- |
| `p_memorized` ranks memorized above unseen text on new data | R20: two models, both intervals include 0.5 | held-out AUC intervals include 0.5 for most models |
| `p_memorized` is calibrated | none yet | reliability curve or Brier score shows probabilities far from observed frequencies |
| A calibrator fitted on one corpus transfers to new items | R20 (Gemma) suggests not | calibrator holdout AUC is high but held-out AUC is near 0.5 |
| The ensemble reduces nondeterminism of the answer and `p_memorized` | none yet (Phase B) | repeated ensembles of the same prompt disagree about as much as single draws do, or held-out AUC drops |
| Mitigation reduces recall (optional) | none | paired held-out recall does not drop at a frozen setting |

## Index

| ID | Date | Track | Status | One line | Report |
| --- | --- | --- | --- | --- | --- |
| R23 | 2026-11-08 | Modal | failed | Phase A rerun stopped after 4 requests (one evidence defect aborted calibration) | below |
| R22 | 2026-11-08 | Modal | failed | Phase A first attempt stopped after 4 requests (reasoning truncation) | below |
| R21 | 2026-11-08 | Modal | pending | Phase A rerun, low reasoning, 1,024 tokens, cap USD 2.00 | pending |
| R20 | 2026-11-08 | Local | running | Phase A, five Ollama models, 1,605 requests | pending |
| R19 | 2026-11-08 | Local | feasibility | Windows Ollama with iGPU behind an authenticated gateway | below |
| R18 | 2026-11-08 | Data | decision | Old calibration corpus confounded; SEC EDGAR corpus built | below |
| R17 | 2026-11-08 | Local | diagnostic | LLaVA context mismatch traced to its model file | below |
| R16 | 2026-11-08 | Modal | feasibility | Live Harmony eligibility for GPT-OSS-20B | below |
| R15 | 2026-11-08 | Local | diagnostic | Ollama GPT-OSS-20B load fits RAM only barely on this host | below |
| R14 | 2026-11-06 | Modal | failed | First Modal pilot: eligibility rejected | [link](results/2026-11-06-modal-pilot.md) |
| R13 | 2026-11-06 | Local | feasibility | Project-local Ollama store and sequential probes | [link](results/2026-11-06-local-model-storage.md) |
| R12 | 2026-11-06 | Local | feasibility | Five local models return aligned logprobs | [link](results/2026-11-06-local-model-smoke.md) |
| R11 | 2026-11-06 | Modal | diagnostic | GPT-OSS alignment failure was our Harmony mapping bug | [link](results/2026-11-06-gpt-oss-response-mapping.md) |
| R10 | 2026-11-05 | Track 1 | measured | S&P 500 identity pilot, pre versus post cutoff | [link](results/2026-11-05-sp500-pilot.md) |
| R09 | 2026-11-05 | NIM | diagnostic | NIM 504 reproduced outside Recall Guard | [link](results/2026-11-05-nim-request-diagnostic.md) |
| R08 | 2026-11-05 | Local | feasibility | First local token-evidence probes | [link](results/2026-11-05-local-provider-feasibility.md) |
| R07 | 2026-11-01 | Summary | measured | Single-answer view of R01 and R02 | [link](results/2026-11-01-recall-guard-single.md) |
| R06 | 2026-11-01 | Summary | measured | Ensemble view of R02 | [link](results/2026-11-01-recall-guard-ensemble.md) |
| R05 | 2026-11-01 | NIM | failed | NIM refresh: 504s, nothing measured | [link](results/2026-11-01-nim-refresh.md) |
| R04 | 2026-11-01 | Divergence | feasibility | FOMC corpus gives an older/newer lexical signature | [link](results/2026-11-01-fomc-statistical-signature.md) |
| R03 | 2026-11-01 | Divergence | failed | Opus sample reweighting prototype | [link](results/2026-11-01-anthropic-reweighting-prototype.md) |
| R02 | 2026-10-31 | Track 1 | failed | Completion-only rewrite mitigation and 3-draw ensemble | [link](results/2026-10-31-completion-api-pilot.md) |
| R01 | 2026-10-30 | NIM | measured | Hosted GPT-OSS-20B calibration and ETF evaluation | [link](results/2026-10-30-nim-measurement.md) |

## Entries

### R23 · 2026-11-08 · Modal Phase A rerun (stopped)

- **Question:** as R21.
- **Setup:** `reasoning_effort: low`, 1,024 output tokens; otherwise as R21.
- **Data:** as R21.
- **Result:** stopped after 4 of 321 requests, all HTTP 200, the longest 560
  tokens. Low effort removed the truncation. One received response failed a
  per-item evidence check, and the adapter treated that as fatal, so calibration
  stopped and the reason was discarded. Fixed in the next commit: such a response
  fails one item with its code, and `min_valid` decides. Runner estimate USD 0.10;
  both apps verified stopped. Nothing measured.
- **Limits:** the exact defect is unknown (most likely duplicate top-5
  alternatives or an empty final answer); the next run reports it.
- **Evidence:** `runs/container/phase-a/modal-gpt-oss-20b-r2/` (private).

### R22 · 2026-11-08 · Modal Phase A, first attempt (stopped)

- **Question:** as R21.
- **Setup:** as R21 with 512 output tokens and default reasoning.
- **Data:** as R21.
- **Result:** stopped after 4 of 321 requests (1 probe, 3 control rows); runner
  estimate USD 0.12; both apps verified stopped. The fourth response used all 512
  tokens on reasoning, so no final answer existed. The harness classed this as an
  unknown outcome, which stops a run by design. Fixed in da2f55c: it now counts
  as one failed item. Nothing was measured.
- **Limits:** 3 responses are too few to estimate the truncation rate.
- **Evidence:** `runs/container/phase-a/modal-gpt-oss-20b/` (private). The rerun
  (R21) uses `reasoning_effort: low` and 1,024 output tokens (user decision).

### R21 · 2026-11-08 · Modal Phase A (rerun pending approval)

- **Question:** does a fresh calibrator for GPT-OSS-20B separate memorized from
  control press releases on 80 held-out rows?
- **Setup:** profile `modal-gpt-oss-20b`, Harmony final answer, temperature 0,
  `reasoning_effort: low`, 1,024 output tokens, top-5 logprobs, 0 retries; 321 requests; GPU phase at most
  2,700 s; worst case USD 1.9976, cap USD 2.00; both apps stopped and verified.
- **Data:** R18 EDGAR corpus: calibration 40 memorized and 100 control; held-out
  40 and 40.
- **Result:** pending.
- **Limits:** partial top-5 logprobs, not full distributions.
- **Evidence:** receipt and report under `runs/container/phase-a/modal-gpt-oss-20b/` (private), dated report to follow.

### R20 · 2026-11-08 · Local Phase A (running)

- **Question:** same as R21 for Phi-4 Mini, Llama 3.2 3B, Gemma 3 4B, Llama 3.1 8B
  and LLaVA-Llama3 (ctx 2048).
- **Setup:** Windows Ollama 0.40.11 with iGPU through the R19 gateway; 2,048
  context, 32 output tokens, top-5, 0 retries; 321 requests per model, one model
  at a time, fresh memory preflight before each.
- **Data:** as R21.
- **Result:** pending.
- **Limits:** input token counts are upper bounds; the local endpoint is frozen in
  the plan hashes.
- **Evidence:** `runs/container/phase-a/<profile>/` (private), dated report to follow.

### R19 · 2026-11-08 · Windows Ollama with iGPU behind a gateway

- **Question:** can containerized Recall Guard use Windows Ollama with the GPU,
  with authentication, and leave nothing running?
- **Setup:** `scripts/serve_local_ollama.ps1`: Ollama on 127.0.0.1 with
  `OLLAMA_IGPU_ENABLE=1`; a bearer-token gateway bound only to the WSL adapter;
  30-minute idle stop.
- **Data:** load-only check of `gemma3:4b`; no generation.
- **Result:** 35 of 35 layers offloaded to the Radeon 890M through ROCm. From the
  container: 200 with the token, 401 without, 404 for pulls. Stop verified no
  process or listener left.
- **Limits:** iGPU speed under the Phase A load is not yet measured.
- **Evidence:** release plan M3.3; commit f30b33f.

### R18 · 2026-11-08 · Calibration corpus confound and SEC EDGAR replacement

- **Question:** is the calibration corpus fit for a held-out test?
- **Setup:** audit of `data/calibration/`; rebuild from SEC EDGAR 8-K press releases
  (EX-99.1) with `scripts/build_edgar_corpora.py`, seed 0, 1,118 SEC requests.
- **Data:** old set: 40 memorized rows from 3 dates (20 on 2023-12-31, 19 on
  2020-12-30), control rows from another source. A held-out FMP rebuild gave 16
  rows, all from 2020-08-07. New set: memorized 2015-01 to 2023-03 (40 dates, 40
  companies), control 2024-09 to 2026-11, one source for both.
- **Result:** the old corpus confounds label with date and source; it is kept as
  history. Phase A uses `data/edgar/` (40/100 calibration, 40/40 held-out,
  disjoint, hashed).
- **Limits:** some titles are layout noise; filing date stands in for publication date.
- **Evidence:** [data/edgar/manifest.json](https://github.com/norandom/memguard_alpha/blob/main/data/edgar/manifest.json); commit 1fe09f6.

### R17 · 2026-11-08 · LLaVA context mismatch

- **Question:** why does LLaVA load at 4,096 tokens when 2,048 is requested?
- **Setup:** load-only requests on Ollama 0.40.11; `/api/show` parameters.
- **Data:** `llava-llama3:latest`.
- **Result:** its model file sets `num_ctx 4096`, which overrides the server
  default. A derived tag `llava-llama3:ctx2048` loads at 2,048 and shares the
  weights.
- **Limits:** token evidence for the new tag is first checked in R20.
- **Evidence:** release plan M3.1.

### R16 · 2026-11-08 · Live Harmony eligibility on Modal

- **Question:** does the R11 fix hold on a fresh generation?
- **Setup:** one request through `HarmonyOpenAICompatibleClient`, 0 retries;
  unauthenticated health check refused with 401 first.
- **Data:** one eligibility prompt.
- **Result:** eligible: aligned final-answer logprobs with 5 alternatives; 73
  prompt and 30 completion tokens; 2.8 s generation, 117 s cold start; runner
  estimate USD 0.094; both apps stopped.
- **Limits:** one response; billing not reconciled.
- **Evidence:** release plan M2.1; commit 9f566cc.

### R15 · 2026-11-08 · Ollama GPT-OSS-20B on a 32 GB host

- **Question:** why did `gpt-oss:20b` return HTTP 500 locally (R13)?
- **Setup:** load-only request on Ollama 0.40.11.
- **Data:** none.
- **Result:** VRAM is not the limit: Ollama drops the iGPU by default and runs on
  CPU. The model loads with 13.9 GB of host memory, leaving 1.7 GB free. The 500
  is consistent with a busy host short of RAM; Recall Guard now asks for 20 GiB
  free and proposes Modal or a larger machine.
- **Limits:** the 500 body was not kept, so the cause stays a hypothesis.
- **Evidence:** release plan C1, M1.4.

### R14 to R01

Summaries of the reports linked in the index, in the same shape.

| ID | Question | Setup | Data | Result | Limits |
| --- | --- | --- | --- | --- | --- |
| R14 | Host GPT-OSS-20B on Modal and pilot it | vLLM 0.17.0, A100-80GB observed, 1 of 8 requests | 8 prompts, 73 to 76 tokens | eligibility rejected (alignment); gross USD 0.158 | superseded in part by R11 |
| R13 | Project-local Ollama store works | 3 probes, context 2,048 | 28 files, 22.7 GB, hashes matched | Gemma eligible; GPT-OSS HTTP 500; LLaVA context 4,096 | 500 cause unassigned (see R15) |
| R12 | Five local models give aligned logprobs | 5 requests, CPU, 32 tokens | one prompt per model | Phi-4 Mini, Llama 3.1 8B, Llama 3.2 3B, Gemma valid; LLaVA context mismatch | partial top-k, unverified normalization |
| R11 | Why GPT-OSS alignment failed | 1 Modal request with token IDs | one prompt | full Harmony logprobs attached to final-only text; fixed adapter passes on replay | replay, not a new generation (closed by R16) |
| R10 | Identity advantage before vs after cutoff | Opus 25 requests, NIM 1 (504) | 16 dates, 48 prompts | +0.011 pre, -0.025 post; contrast 0.036, no interval | 4 observations per cohort |
| R09 | Is the NIM 504 ours | 2 direct requests | one prompt | 504 at about 302 s in both clients; key sent correctly | root cause unknown |
| R08 | Do local endpoints give token evidence | Ollama CPU, 2 probes | one prompt each | Gemma eligible; GPT-OSS alignment failed; NPU gate failed | no repeats |
| R07 | Single-answer evidence summary | restates R01, R02 | R01, R02 | AUC 0.876; 170 of 312 correct vs 53.94 % baseline | lower confidence is not reduced lookahead |
| R06 | Ensemble evidence summary | restates R02 | 16 items x 3 draws | 11/16 vs 12/16; +6.25 pp [0.00, 18.75] | small dependent sample |
| R05 | Fresh NIM calibration | 3 of 146 requests | planned 20 + 20 | 504s; nothing measured | 504 does not locate the fault |
| R04 | Dated corpus gives a lexical signature | CPU trigram scorer, 0 API calls | FOMC 2019 vs 2020 | exposure shift about +31 on rate probes | no held-out split, no target model |
| R03 | Reweight Opus samples with counts | 9 Opus requests | 1 prompt, synthetic counts | no change at any alpha | synthetic; discard as evidence |
| R02 | Completion-only rewrite mitigation | Opus 67 requests | 16 items, 8 pre and 8 post cutoff | all 16 rewrites rejected; ensemble +6.25 pp, interval includes 0 | NIM arm never ran |
| R01 | Hosted GPT-OSS-20B calibration | NIM, temperature 0, 512 tokens | old FMP corpus 40 + 100; 330 ETF prompts | holdout AUC 0.876; 54.49 % correct vs 53.94 % baseline | confounded corpus (R18); point-estimate interval |

## How to add an entry

Give the next ID, date and track, then fill the six fields from the run's own
report. Record failures and pending runs as entries too. If a new result changes
an earlier conclusion, note it in both entries and update the state of evidence.
