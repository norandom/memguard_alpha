# Phi-4 Mini disqualified from the Phase A model set, 2026-11-08

Phi-4 Mini (`phi4-mini:3.8b` on Ollama 0.40.11, iGPU) is removed from the
release-1.0 Phase A matched model set by user decision. Its run stopped before
calibration finished, and it will not be rerun. Gemma 3 4B and the Llama models
stay in the set.

## What happened

| Field | Value |
| --- | --- |
| Profile | `ollama-phi4-mini`, context 2,048, 32 output tokens, top-5 logprobs, temperature 0, 0 retries |
| Data | SEC EDGAR corpus (`data/edgar/`), the same inputs as every Phase A model |
| Requests | 119 of 321, all HTTP 200; 3 to 32 completion tokens each |
| Completed | probe 1; control baseline 100 of 100; memorized calibration 17 of 40 |
| Stop | call 119, a memorized calibration row: its response failed a per-response evidence check |
| Run status | `ineligible`, reason `invalid_evidence_identity_or_fit`; no calibrator, no held-out scores |

The harness then treated any per-response evidence defect as fatal for the whole
run, and discarded the defect's code. Commit 0c02b1e changed that: such a
response now fails one item and the code is reported. The exact defect for call
119 was therefore not recorded. Candidates from the code path are duplicate or
missing top-5 alternatives, an empty output, or a censored logprob.

## Decision and consequences

- Phi-4 Mini is not part of the Phase A results or of Phase B.
- The approved local rerun is withdrawn for this model.
- The profile stays available for adopters; this is a measurement-set decision,
  not a statement that the model cannot be scored.
- No calibration, discrimination or calibration-quality claim is made for it.

Private evidence: `runs/container/phase-a/ollama-phi4-mini/` (journal and report).
Register entry: [R23](../research-register.md).
