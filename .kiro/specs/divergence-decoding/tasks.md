# Implementation Plan — Divergence Decoding MVP

Scope revision 2: mathematical layer and greedy caller-callback integration. Requirements and design are approved. The user approved tasks and implementation on 2026-10-30.

## Execution and Quality Rules

Use the current requirements.md and design.md; archive/full-scope/ is historical context. No model hosting, downloaded checkpoints, calibration, ensemble/replay, scientific verdicts or new dependencies belong to these tasks.

Every executable task includes review for idiomatic Python/NumPy, explicit types, focused functions, and names matching meaning and representation. Distinguish model sources from vocabulary-indexed logits, token IDs from counts, and indices from values. Document array shapes/axes at boundaries. If DataFrames are encountered, check domain-specific frame/column/index names, row grain, units and label alignment; this milestone introduces none. Review naming in code/examples rather than testing variable spelling.

Existing environment and pytest/ruff/MkDocs tooling suffice; use the repository's uv workflow. No setup package or new complexity checker is needed. Before production edits, confirm the required sentrux scan/rule tooling or documented sentrux check CLI is available. If unavailable, report the unverified gate explicitly; the existing scripts/check_architecture.py checks import boundaries but does not prove complexity/cycle compliance. Apply this project's steering limits (CC <=25, <=120 lines/function, no cycles and downward imports); the fallback template's other-project complexity inventory/caps do not apply.

Tasks are sequential: the decoder depends on the mathematical contract, and example/integration checks depend on both. No (P) markers or optional tests.

## Tasks

- [x] 1. Provide validated mathematical steering
  - Implement the standalone equation using real numeric one-dimensional vectors converted to float64, with equal nonempty shapes and finite nonnegative alpha.
  - Validate all three inputs even at alpha=0; return an independent main-logits copy without evaluating subtraction on that path. Preserve all input arrays.
  - Reject unsupported dtypes, malformed vectors, non-finite inputs and non-finite/overflowed results with actionable errors.
  - Add controlled-vector tests for the hand-calculated formula, zero-alpha identity/copy, input nonmutation, rejection cases and overflow; they pass when this task is complete.
  - Review direct NumPy arithmetic, explicit annotations, vocabulary-axis documentation and main_logits/older_logits/newer_logits/adjusted_logits naming against the quality contract.
  - Verify applicable project complexity limits for the mathematical module; do not introduce a generic validation framework.
  - _Boundary: Mathematical function — recall_guard/core/divergence.py; tests/core/test_divergence.py_
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 4.1, 5.1, 5.2, 5.3_

- [x] 2. Provide minimal greedy decoding over compatible callbacks
- [x] 2.1 Establish callback metadata and preflight validation
  - Define the frozen callback-source and token-result values plus role/step-bearing decoder error in accordance with the approved contract.
  - Require matching declared vocabulary identities/sizes and EOS sets; validate callables, positive integer context limits/budget, nonempty prompt and in-range non-boolean integer token IDs.
  - Reject an insufficient minimum context window and invalid alpha before any model callback executes; allow an empty EOS set for budget-only termination.
  - Add preflight tests that demonstrate zero callback invocations for invalid requests and acceptance at the exact context boundary.
  - Review model-source versus logit naming, prompt_token_ids/vocabulary_size semantics, explicit types and focused Python validation; keep metadata declarations clearly distinct from verified tokenizer compatibility.
  - Verify applicable project complexity limits for the decoder's values/preflight functions.
  - _Boundary: Greedy decoder — recall_guard/core/divergence_decode.py; tests/core/test_divergence_decode.py_
  - _Depends: 1_
  - _Requirements: 2.1, 2.3, 5.1, 5.2, 5.3_

- [x] 2.2 Execute the shared-prefix greedy loop and expose failures
  - Call the three sources sequentially with the same immutable full prefix for each generation step, including at alpha=0; validate each returned vector against its declared vocabulary size.
  - Apply the completed mathematical layer and choose the lowest token ID among tied maximum adjusted logits.
  - Return generated IDs excluding the prompt and including a selected terminal EOS, with eos or length finish reason.
  - Wrap callback/returned-value failures with model role and zero-based step; mark mathematical overflow as combination failure. Stop immediately without silent fallback or a successful partial result; let KeyboardInterrupt/SystemExit propagate.
  - Add tests for evolving shared prefixes, tie selection, first-token EOS, exact budget stop, each failing source, malformed callback vectors, combination overflow and no callbacks after failure. These behavioral tests pass when complete.
  - Review straightforward control flow, explicit types and names such as prefix_token_ids/generated_token_ids/next_token_id/generation_step; preserve vocabulary-axis meaning through transformations.
  - Verify applicable project complexity limits for the completed decoder.
  - _Boundary: Greedy decoder — recall_guard/core/divergence_decode.py; tests/core/test_divergence_decode.py_
  - _Requirements: 2.2, 2.4, 2.5, 4.2, 5.1, 5.2, 5.3_

- [ ] 3. Integrate an executable usage example into the documentation site
  - Add a runnable synthetic callback example using direct core imports, showing how callers provide compatible sources and interpret generated IDs/finish status.
  - Explain caller-controlled logits access, declared vocabulary compatibility, shared-prefix expectations, and why the existing NVIDIA completion client cannot supply this intervention.
  - Label the utility experimental; state that mathematical correctness establishes neither reduced memorization nor a calibrated contamination score. Do not imply existing calibration can score adjusted generation unchanged.
  - Register the page in MkDocs navigation and execute the exact documented example without credentials or downloaded models; confirm the expected deterministic output and successful strict documentation build.
  - Review example code for idiomatic NumPy/Python, meaningful source/logit/token names and documented vocabulary axes. Check descriptive labels and units wherever applicable without introducing DataFrames.
  - Production complexity limits do not apply to Markdown/YAML; keep the executable example focused and review it under the same naming/code-quality contract.
  - _Boundary: Explicit integration — docs/divergence-decoding.md; mkdocs.yml; existing core API_
  - _Depends: 2.2_
  - _Requirements: 3.3, 3.4, 3.5, 4.3, 5.1, 5.2, 5.3_

- [ ] 4. Verify package integration and implementation quality
  - Run focused mathematical/decoder tests and the existing packaging/import, architecture, NVIDIA-client, scorer and ensemble compatibility checks with uv; run lint and the repository-required architecture/complexity scan and rule checks for the two new modules.
  - Confirm default imports remain lean, the dependency metadata is unchanged, and existing scorer/client/ensemble behavior passes its established tests. Add a focused import regression assertion only if existing coverage does not exercise the new direct-import path.
  - Independently review the implementation and executable example against every acceptance criterion, with idiomatic code and semantic names an explicit completion condition. Inspect array shapes/axes and any frame/column/index meanings, rather than asserting names in tests.
  - Record actual commands/results and resolve blocking findings; completion requires the targeted checks, documented example, strict docs build and required architecture/complexity gates to pass.
  - Verify project complexity limits on maintained production code; tests and docs receive focused quality review without inventing production caps for them.
  - _Boundary: Explicit integration validation — new core modules, their tests, documentation; existing compatibility checks_
  - _Depends: 1, 2.2, 3_
  - _Requirements: 3.1, 3.2, 3.5, 4.1, 4.2, 4.3, 5.1, 5.2, 5.3_

## Task Plan Review

Five executable work units across four major tasks, each intended as approximately 1–3 hours of focused work. All 21 criteria are mapped. Tests accompany their implementations; final integration verifies existing contracts. No environment/runtime prerequisite beyond the existing repository toolchain is introduced. Task-plan review passed; independent task-graph sanity review: PASS. The review confirmed sequential dependencies, bounded deliverables and no additional model-runtime prerequisites. Executable documentation is an explicit integration deliverable required by the approved specification. Required sentrux tooling availability must be checked before production edits; missing tools must be reported rather than replaced with invented verification.




## Implementation Notes

Task 1: 46 focused tests and scoped Ruff pass; independent reviewer APPROVED. Sentrux unavailable; final gate remains open. Baseline Windows architecture test has slash/backslash assertion mismatch; actual architecture checker passes.


Task 2.1: 74 tests and scoped Ruff pass; independent review approved after fixing NumPy integer overflow in budget arithmetic. Preflight normalizes prompt IDs and uses Python ints for context arithmetic.


Task 2.2: 147 combined tests, scoped Ruff and architecture checker pass. Independent reviewer APPROVED; loop has explicit per-source role/step errors and shared-prefix tests.
