# Implementation Validation — Divergence Decoding MVP

Date: 2026-10-30  
Scope: revision 2, mathematical layer and greedy callback decoder  
Decision: **NO-GO for final repository acceptance**. Feature code and documentation are implemented and task-local reviews approved. Final integration acceptance remains open for one upstream Windows test failure and unavailable mandatory Sentrux verification.

## Delivered

- recall_guard/core/divergence.py: finite float64 vocabulary-vector validation and the equation; zero-alpha independent copy; input nonmutation and overflow rejection.
- recall_guard/core/divergence_decode.py: frozen source/result values, preflight, sequential shared-prefix greedy generation, deterministic ties, EOS/budget stops and role/step/cause failures.
- docs/divergence-decoding.md and MkDocs navigation: exact runnable synthetic example and caller/runtime limitations.

No model runtime or dependency was added. Existing scorer, NVIDIA client, ensemble, MCS and manifest behavior are unchanged. This utility returns generated token IDs, not calibrated contamination scores.

## Task-local Review Evidence

| Task | Commit | Review | Verification |
|---|---|---|---|
| 1 | 9b44ce4 | Independent APPROVED | 46 math tests and scoped Ruff |
| 2.1 | a4b2121 | Independent APPROVED after remediation | 74 preflight tests, including NumPy integer-budget overflow regression |
| 2.2 | 68040c7 | Independent APPROVED | 147 combined mathematical/decoder tests and scoped Ruff |
| 3 | 7653491 | Independent APPROVED | Exact example outputs (1, 2) eos; strict MkDocs build; 20 documentation tests |
| 4 | Pending | BLOCKED on external gates | 245 targeted tests pass; no feature-local defects found |

Tests were written and run failing before behavioral implementation. Temporary development flags were removed. Documentation and validation-only tasks required no behavior flag.

## Mechanical Results

- PASS: 147 new focused tests.
- PASS: independent targeted compatibility run: 245 tests covering both new modules, scorer, NVIDIA client, ensembles, packaging and ensemble boundaries.
- PASS: uv run --no-sync ruff check .
- PASS: uv run --no-sync python scripts/check_architecture.py.
- PASS: uv run --no-sync mkdocs build --strict; exact fenced example execution.
- PASS: uv build produces wheel and source distribution.
- PASS: a fresh interpreter imports the decoder from the built wheel and generates an EOS result. The smoke asserts that the imported module path is inside the wheel.
- PASS: direct-import smoke leaves torch, transformers, matplotlib and vectorbt unloaded.
- PASS: changed production/example files have no residual placeholder markers or hardcoded credentials.
- FAIL: final full suite with PYTHONUTF8=1, uv run --no-sync pytest -q --tb=short: **597 passed, 1 failed**, 127.57 seconds.
- UNAVAILABLE: required Sentrux scan/check_rules or documented CLI. The import-boundary script does not establish complexity or cycle compliance.

The single final full-suite failure is tests/test_architecture.py::test_gate_flags_banned_dspy at line 60: expected harness/runner.py, actual harness\\runner.py on Windows. The checker formats a native pathlib path in its diagnostic. The same failure was established before decoder implementation; both that checker and test remain unchanged.

The first default-environment full run had 491 passes and six failures: two pacing-timing assertions, three cp1252 portfolio-report Unicode errors and the path assertion. In the final UTF-8 run, timing assertions passed and encoding errors were avoided by the environment setting; the path assertion remained. No underlying timing or portfolio code was altered, and a clean default-environment full run is not claimed.

## Requirements Coverage and Integration

| Criteria | Evidence |
|---|---|
| 1.1, 1.2, 1.3, 1.4, 1.5 | combine_logits and controlled-vector validation tests |
| 2.1, 2.2, 2.3, 2.4, 2.5 | Source/preflight and decoder tests, including zero-callback rejection, shared prefixes and structured failures |
| 3.1, 3.2 | Unchanged default-path/dependency files, compatibility tests and lean-import smoke |
| 3.3, 3.4, 3.5 | Documentation, token-only result and explicit absence of calibrated scoring |
| 4.1, 4.2, 4.3 | Focused tests and exact executed documentation example |
| 5.1, 5.2, 5.3 | Independent task reviews and final source/example inspection of idioms, types, semantic names and array axes |

All 21 criteria have implementation evidence. Existing dependency direction and the planned file layout are preserved. No new shared mutable execution state, persistence schema or downstream coupling was introduced.

## Ownership and Remaining Steps

- **UPSTREAM:** architecture/test portability, owned by the review-hardening architecture gate (6.1). Normalize diagnostic path formatting or its platform expectation in that scope, then rerun the full suite and dependent architecture tests.
- **ENVIRONMENT:** provide Sentrux and run the required scan/rule checks against the new core modules and package graph.
- **LOCAL:** no unresolved feature defect identified by task-local or final integration inspection.

Task 4 remains unchecked with its blockers recorded. The feature is implemented but is not represented as fully validated. No additional model downloads, calibration work, ensemble framework or empirical mitigation claims are needed for the narrowed milestone.

