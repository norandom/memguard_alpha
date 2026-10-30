# Design Validation — Divergence Decoding

Date: 2026-10-30  
Decision: **NO-GO for implementation of the current design**  
Scope: static design review against approved requirements, current code and the user's concern about XL scope. This does not reject the divergence-decoding formula.

## Review Summary

The mathematical operation and optional-backend direction are sound. The current design has three repairable contract/evaluation gaps. All 45 criteria appear in its traceability table, but that mechanical coverage does not establish executable contracts. This review supersedes the earlier generation-time review's readiness conclusion.

## Critical Issues

### 1. Replay omits reduction inputs

**Concern:** The replay signature takes audited draws and a caller-supplied EnsembleSpec, but the artifact does not preserve wave membership or conservative_quantile.  
**Impact:** Reordered replay can change dependence diagnostics or cannot reproduce the original result.  
**Recommendation:** Define an ensemble artifact containing the original spec, reduction options, stable draw IDs, original wave IDs and all failed draws. Validate a single prompt/configuration/calibration identity; never reconstruct waves from input order.  
**Traceability:** 5.4, 5.5.  
**Evidence:** design.md:200, 208, 218; recall_guard/harness/scorer.py:404, 411, 463, 471.

### 2. Completion and audit evidence have no atomic handoff

**Concern:** The bound adapter returns CompletionResult while generate_evidence returns a separate type; the design does not connect these through scoring and failures.  
**Impact:** A shared last-result field would misassociate repeated/concurrent draws, and existing reduction drops failed completions.  
**Recommendation:** Let the divergence orchestrator retain one per-draw envelope containing completion-or-error, evidence and draw/wave IDs. Project only its completion into existing scoring helpers; retain the envelope for audit/replay. Calibration uses a projection adapter without mutable evidence side channels.  
**Traceability:** 5.1, 5.2, 5.3, 6.1, 6.2, 6.3.  
**Evidence:** design.md:160–171, 208; recall_guard/core/nvidia_lm.py:73–90; recall_guard/harness/scorer.py:375–404, 504.

### 3. Historical failures can produce a positive mitigation verdict

**Concern:** Failed responses count as incorrect, while the verdict rewards reduced historical excess accuracy. Merely reporting parse rates does not prevent a false positive.  
**Impact:** With baseline historical/OOS accuracy 0.8/0.6 and steered accuracy 0.5/0.6, the effect is +0.3 even if the historical drop comes entirely from new parse failures. Larger samples can make this spurious effect statistically precise.  
**Recommendation:** Retain all-row utility reporting, but prevent failures from counting as mitigation evidence. Predeclare cohort-specific completion/parse noninferiority gates and paired-success sensitivity analysis; if failure differences explain the effect, return inconclusive. Specify the target historical band explicitly.  
**Traceability:** 7.5, 7.6, 7.7.  
**Evidence:** design.md:232–239, Experiment runner.

## Strengths

- Clear opt-in isolation and preservation of the default NVIDIA path.
- Correct separation of pre-temperature likelihoods, calibration identity and empirical mitigation claims.

## Scope and Next Steps

The XL label covers local model execution, scoring integration, replay/audit and a research framework; it does not describe the formula. The design largely follows the broad approved requirements. Given the user's concern, recommend a smaller first milestone: caller-supplied compatible logits/models, mathematical combination, minimal decoding integration and alpha-zero/failure tests. Defer integrated calibration, ensemble replay and automated scientific verdicts to follow-up work. Such narrowing requires revising requirements 4–8 and renewed approval; this review does not silently remove them.

If retaining the full scope, repair the three issues and rerun design validation. If choosing the smaller milestone, revise requirements and design together before task generation. No additional abstraction or platform expansion is needed to fix these findings.

## Verification

Reviewed source contracts plus an independent read-only contract review. No runtime tests were needed or run for this document review. Requirements approval remains unchanged; design approval and implementation readiness remain false.

