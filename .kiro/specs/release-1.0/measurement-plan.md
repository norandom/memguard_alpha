# Measurement plan: Tracks 2 and 3 (M3.2)

Approved by the user 2026-11-08: D1 (build a labeled held-out set), D2 (source
missing cutoffs, exclude a model whose cutoff cannot be sourced), D3 (Ollama in a
WSL container) and the Phase A limits (local 1,605 requests; Modal 321 requests,
cap USD 2.00). Phase B still needs its own approval.

## Question

For each backend profile: does a freshly fitted calibrator separate memorized
from out-of-sample text on held-out data it never saw, and do ensemble draws
change that? Single-answer and ensemble results are reported separately (R3.1).
No mitigation or divergence claim is made here (R3.3); that is M6.

## Models and profiles

| Profile | Model | Host | Context | Cutoff registry |
| --- | --- | --- | ---: | --- |
| `ollama-phi4-mini` | `phi4-mini:3.8b` | local CPU | 2048 | 2024-06-30 |
| `ollama-llama3.2-3b` | `llama3.2:3b` | local CPU | 2048 | 2023-12-31 |
| `ollama-gemma3-4b` | `gemma3:4b` | local CPU | 2048 | **missing (D2)** |
| `ollama-llama3.1-8b` | `llama3.1:8b` | local CPU | 2048 | 2023-12-31 |
| new `ollama-llava-llama3-ctx2048` | `llava-llama3:ctx2048` | local CPU | 2048 | **missing (D2)** |
| `modal-gpt-oss-20b` | `openai/gpt-oss-20b`, Harmony final answer | Modal A100 | 4096 | 2024-06-30 |

Every model gets a fresh eligibility probe immediately before its batch (aligned
logprobs, 5 alternatives, observed context). Failing it skips that model; nothing
is substituted. Calibrators are per profile and never reused across profiles (R2.5).
LLaVA's new tag has not yet had a token-evidence probe (M3.1).

## Data

| Split | Source | Rows | Use |
| --- | --- | ---: | --- |
| Calibration, memorized | `data/calibration/is_memorized.jsonl` (2020-12-30..2023-12-31) | 40 | fit; internal 25 % holdout |
| Calibration, control | `data/calibration/oos_control.jsonl` (2026-04-23..27) | 100 | baseline and fit |
| **Held-out, labeled** | **new**, disjoint, same builder and date rules (D1) | 40 + 40 | evaluation only, never used to fit or tune |

Rows are filtered per model so that "memorized" rows predate that model's
cutoff and control rows postdate it; the per-model counts are reported, never
padded. Today's only held-out signal is the calibrator's internal 25 % holdout
(10 memorized rows), which produced a degenerate AUC interval in the 2026-10-30
NIM run. A separate labeled held-out set fixes that.

## Settings (frozen per profile in `plan.json`, hashed)

| | Local Ollama | Modal GPT-OSS-20B |
| --- | --- | --- |
| temperature / top_p | 0 / 1 | 0 / 1 |
| max output tokens | 32 | 512 (reasoning precedes the final answer) |
| top_logprobs | 5 | 5 |
| retries | 0 | 0 |
| prompt | calibration text as-is, `{prompt}` template | same |
| seed, min_valid, min_auc | 0, 50, 0.6 | 0, 50, 0.6 |
| ensemble | 4 draws per held-out item, min 3 parsed | same |

## Request and cost limits

Per model, with the harness counting rule (1 probe + memorized + 2 x control +
held-out x (1 + draws)):

| Phase | Logical requests per model | Local (5 models) | Modal (1 model) |
| --- | ---: | ---: | ---: |
| A. Calibration + single-answer held-out | 1 + 40 + 200 + 80 = **321** | 1,605 | 321 |
| B. Ensemble on held-out (4 draws) | 80 x 4 = **320** | 1,600 | 320 |

Phase B runs only after Phase A is reviewed, and has its own approval.

- **Local:** no money. Estimated CPU time is about 10 s per request, so Phase A
  takes about 4.5 h and Phase B about 4.5 h for all five models. Runs are
  sequential, one model loaded at a time, and each model is unloaded afterwards.
  The RAM preflight runs before each model.
- **Modal Phase A:** 321 POSTs x up to 1,112 tokens (at most 357k). The GPU phase
  is about 30 min including a 2 min cold start. At USD 0.000583/s GPU plus CPU and
  memory, that is **about USD 1.10, capped at USD 2.00**. Phase B is about USD 1.10
  more, capped at USD 2.00.
- **Total new Modal allowance requested: USD 4.00** (both phases), inside the
  USD 30/month workspace limit. The previous USD 5 pilot allowance is not reused.
  After the last Modal phase, M4.4 deletes the weights volume.

## Prerequisites before M4 (offline unless noted)

1. **D1 held-out data:** build 40 + 40 disjoint labeled rows with
   `build_calibration`. This uses the FMP API (`FMP_API_KEY` in `.env`): data
   calls, no model calls.
2. **D2 cutoffs:** add sourced registry entries for Gemma 3 4B and LLaVA-Llama3,
   with the source URL and retrieval date. If a cutoff cannot be sourced, that
   model is excluded rather than guessed.
3. **D3 local serving route:** Recall Guard runs in a container, and this host
   uses WSL NAT. Recommended: run Ollama in a Linux container in WSL from the same
   `models/` store, on a private Docker network with no exposure to Windows or the
   LAN. Take the memory snapshot inside WSL with `scripts/host_memory_snapshot.sh`.
   Alternatives: Windows Ollama bound to the WSL interface plus a firewall rule,
   or WSL mirrored networking.
4. **Modal measurement runner:** the current runner sends only the fixed 8-prompt
   pilot. M4 needs the measurement path (`measure_local_models.py` plan, journal
   and report) behind the Modal lifecycle (deploy, expiry, verified stop), and the
   manifest's GPU phase limit raised from 1200 s to cover Phase A.
5. Add the `ollama-llava-llama3-ctx2048` profile (6 GiB RAM).

## Reported per profile

Eligibility outcome; valid and failed counts per split; calibrator holdout AUC;
held-out AUC with a 1,000-resample bootstrap interval; single versus ensemble
held-out AUC with paired difference; requests, tokens and wall time; Modal cost,
estimated and then reconciled. The evidence is partial top-k (C5); results are
not compared as full distributions.

## Decisions for the user

- D1: build a labeled held-out set (recommended) or rely only on the internal holdout.
- D2: source missing cutoffs, or drop Gemma and LLaVA.
- D3: local serving route (container Ollama in WSL recommended).
- Limits: Phase A local (1,605 requests), Phase A Modal (321 requests, cap USD 2.00);
  Phase B is approved separately later.
