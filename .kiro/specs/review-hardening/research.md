# Research & Design Decisions

---
**Purpose**: Capture discovery findings, architectural investigations, and rationale that inform the technical design.

**Usage**:
- Log research activities and outcomes during the discovery phase.
- Document design decision trade-offs that are too detailed for `design.md`.
- Provide references and evidence for future audits or reuse.
---

## Summary
- **Feature**: `review-hardening`
- **Discovery Scope**: Extension
- **Key Findings**:
  - The confirmed review findings cluster around a small number of shared contract surfaces: harness orchestration, parser/LM-client behavior, manifest/input provenance, CMMD/backtest semantics, dataset ingest/update rules, and package/CI metadata.
  - Several defects are false-success or silent-corruption failures rather than missing features, so fail-fast behavior and typed error normalization should be repaired before domain-level metric fixes.
  - The repo already has strong canonical reuse points (`core.loader`, `core.bootstrap`, `core.manifest`, `core.nvidia_lm`, `harness.evaluator`, `portfolio.prices`, `portfolio.cmmd`, `portfolio.backtest`); the design should route fixes through those instead of adding parallel logic.

## Research Log

### Existing runtime and provenance contract surfaces
- **Context**: The review findings span harness runtime behavior, manifest integrity, and public façade behavior.
- **Sources Consulted**:
  - `recall_guard/core/loader.py`
  - `recall_guard/core/bootstrap.py`
  - `recall_guard/core/manifest.py`
  - `recall_guard/core/nvidia_lm.py`
  - `recall_guard/harness/evaluator.py`
  - `recall_guard/harness/runner.py`
  - `recall_guard/harness/scorer.py`
  - `recall_guard/mia/mcs.py`
  - `tests/core/test_loader.py`, `test_bootstrap.py`, `test_manifest.py`, `test_nvidia_lm.py`
  - `tests/harness/test_runner.py`, `test_cutoff_guard.py`, `test_evaluator.py`, `test_scorer.py`
  - `tests/mia/test_mcs.py`
- **Findings**:
  - `core.loader` is the canonical eval/cutoff boundary and already owns `assert_cutoff_safe`.
  - `core.bootstrap` is the shared CI engine used by evaluator metrics and majority baseline.
  - `core.manifest` already defines the strict manifest schema and optional `backtest` extension boundary.
  - `core.nvidia_lm` and `generate_many` are the only intended LM-call surfaces, including timeout/retry/pacing semantics.
  - `harness.evaluator` owns the authoritative parser contract for direction/confidence and the row failure model.
  - `harness.scorer` is intentionally a thin façade over existing primitives, not a separate scoring implementation.
- **Implications**:
  - Fixes should converge duplicated behavior onto the canonical modules above.
  - Parser, runtime-error, and provenance fixes can remove multiple findings at once if applied at those shared boundaries.

### Existing CMMD/backtest and post-harness flow
- **Context**: A large subset of findings live in `portfolio/*` and `scripts/run_cmmd_backtest.py`.
- **Sources Consulted**:
  - `recall_guard/portfolio/cmmd.py`
  - `recall_guard/portfolio/prices.py`
  - `recall_guard/portfolio/backtest.py`
  - `scripts/run_cmmd_backtest.py`
  - `scripts/analyze_is_oos_gap.py`
  - `scripts/build_etf_portfolio_eval.py`
  - `scripts/build_etf_multiyear_eval.py`
  - `tests/portfolio/test_cmmd.py`, `test_prices.py`, `test_backtest_engine.py`, `test_backtest_writers.py`, `test_run_cmmd_backtest_smoke.py`
  - `tests/scripts/test_build_etf_portfolio_eval.py`
- **Findings**:
  - `run_cmmd_backtest.py` is an orchestrator layered on top of the harness, then price fetch, then backtest, then manifest extension.
  - Prompt/date/ticker joins are not carried in `Record`; they are reconstructed from eval-row metadata via prompt-hash.
  - `portfolio.backtest` has one shared eligibility path (`parse_ok_records`) and one shared tradeability path (`_build_weight_matrix`); divergence there explains several of the confirmed defects.
  - `portfolio.cmmd` centralizes filter semantics; one fix there covers tie-handling and non-finite score handling.
- **Implications**:
  - Shared row-eligibility semantics should be centralized before the CMMD threshold is computed.
  - Manifest/backtest provenance work belongs in the orchestrator + manifest extension block, not in ad hoc script outputs.

### Dataset ingest and package/CI drift
- **Context**: The remaining findings concern FMP corpora, eval builders, package extras, and CI enforcement.
- **Sources Consulted**:
  - `recall_guard/dataset/fmp_corpora.py`
  - `tests/dataset/test_fmp_corpora.py`
  - `pyproject.toml`
  - `ci/src/ci/main.py`
  - `docs/index.md`
  - `tests/docs/test_docs_site.py`
- **Findings**:
  - `fmp_corpora.py` already centralizes article normalization, dedup, cutoff-windowing, and incremental OOS refresh, so its fixes are shared rather than scattered.
  - `build_etf_multiyear_eval.py` is a legacy-style builder with weaker fetch/error behavior than the bounded portfolio-eval builder.
  - Package metadata currently exports only `backtest` and `docs` as extras; `dev` and `pipeline` are uv dependency groups only.
  - CI `lint` currently runs Ruff only; there is no separate structural-architecture gate in the GitHub Actions lint path.
- **Implications**:
  - Dataset fixes should stay inside `fmp_corpora.py` where possible.
  - Packaging/CI fixes are contract cleanup work and can be isolated from runtime/backtest changes.

## Architecture Pattern Evaluation

| Option | Description | Strengths | Risks / Limitations | Notes |
|--------|-------------|-----------|---------------------|-------|
| Scattershot local fixes | Patch each finding where it appears | Fast for isolated defects | Repeats logic, preserves divergence, easy to miss sibling paths | Rejected; too many findings share root causes |
| Shared-boundary hardening | Fix canonical loader/parser/LM/manifest/backtest surfaces first, then leaf scripts | Removes duplicated behavior and aligns contracts | Requires discipline to sequence by dependency | Selected |
| Split into multiple unrelated specs | Create one spec per subsystem | Smaller documents | Cross-cutting fixes (parser, provenance, manifest) would duplicate planning and drift | Rejected; one coordinated hardening spec is cleaner |

## Design Decisions

### Decision: Sequence phases by failure severity and shared-contract leverage
- **Context**: The confirmed findings range from false-success runs to metric inconsistencies to package drift.
- **Alternatives Considered**:
  1. Fix by file ownership only
  2. Fix by confirmed-finding order
  3. Fix by shared contract and failure severity
- **Selected Approach**: Start with fail-fast/runtime integrity, then parser/LM correctness, then provenance, then CMMD/backtest semantics, then dataset drift, then package/CI cleanup.
- **Rationale**: This order prevents invalid runs from continuing while later phases are still being repaired and maximizes shared-fix reuse.
- **Trade-offs**: Lower-risk metadata/doc packaging work lands later even if simpler.
- **Follow-up**: Validate each phase with targeted tests before moving to the next one.

### Decision: Reuse canonical modules instead of adding parallel helpers
- **Context**: Several findings were caused by duplicated parsing or duplicated input handling.
- **Alternatives Considered**:
  1. Keep separate smoke/script helpers and align them manually
  2. Route callers through the existing canonical module boundaries
- **Selected Approach**: Consolidate onto `core.loader`, `core.bootstrap`, `core.manifest`, `core.nvidia_lm`, `harness.evaluator`, `portfolio.prices`, `portfolio.cmmd`, and `portfolio.backtest`.
- **Rationale**: The codebase already has canonical boundaries with tests; reusing them is cheaper and safer than managing multiple parallel contracts.
- **Trade-offs**: Some scripts may need modest reshaping to consume the canonical paths.
- **Follow-up**: Add regression tests at the shared boundary rather than only at the leaf script.

### Decision: Treat review-hardening as one brownfield spec, not a new feature family
- **Context**: The user asked to move the improvement plan into Kiro, and the repo has no existing `metric.md` pattern.
- **Alternatives Considered**:
  1. Create an ad hoc metric file outside the repo’s spec conventions
  2. Create multiple tiny specs by subsystem
  3. Create one repair program spec with measurable acceptance criteria
- **Selected Approach**: One new Kiro spec, `review-hardening`, with requirements/design/tasks tied to the confirmed review findings.
- **Rationale**: It matches the repo’s existing Kiro workflow and keeps the work executable under one approval chain.
- **Trade-offs**: The spec is broader than a normal feature spec and needs careful boundary writing.
- **Follow-up**: Keep tasks strictly phase-bounded to avoid a “fix everything at once” implementation attempt.

## Risks & Mitigations
- Broad repair scope could create merge-heavy implementation work — mitigate by phase-bounded tasks and explicit file boundaries.
- Some findings are linked by shared helpers, so partial fixes could create inconsistent intermediate states — mitigate by sequencing parser/runtime/provenance work before backtest/reporting work.
- Packaging/CI contract changes can alter maintainer workflows — mitigate by isolating them to the final phase and validating built metadata plus CI commands explicitly.

## References
- `recall_guard/core/loader.py` — canonical eval/cutoff boundary
- `recall_guard/core/bootstrap.py` — canonical CI helper
- `recall_guard/core/manifest.py` — canonical manifest boundary
- `recall_guard/core/nvidia_lm.py` — canonical LM client
- `recall_guard/harness/evaluator.py` — canonical parser + row-scoring contract
- `recall_guard/harness/scorer.py` — public façade contract
- `recall_guard/portfolio/backtest.py` — canonical backtest engine and artifact writer
- `recall_guard/portfolio/cmmd.py` — canonical CMMD filter
- `recall_guard/portfolio/prices.py` — canonical price alignment surface
- `.kiro/specs/cmmd-backtest/requirements.md` — CMMD/backtest behavior contract
- `.kiro/specs/honest-model-ranking/requirements.md` — harness/core/dataset behavior contract
- `.kiro/specs/recall-guard-package/requirements.md` — package/CI contract
