# recall-guard-package — Gap Analysis

_Generated 2026-06-25. Brownfield analysis mapping the 9 requirements against the existing
codebase. Information over decisions: options + trade-offs, not final choices._

## 1. Current state (grounded facts)

- **Layout**: `src/{core,mia,harness,portfolio,dataset}` + `harness.py` entry + `scripts/`;
  `tests/` mirror the package tree. Public surface already curated via `__all__` in
  `core/__init__.py`, `mia/__init__.py`, `harness/__init__.py`.
- **Not a distribution yet**: `pyproject.toml` has **no `[build-system]`**, name
  `recall-guard`, and the code is importable only as `src.*`. **140 `src.` import
  statements across 44 files** (src + tests + scripts + harness.py) — this is the concrete
  rename scope.
- **Python pin**: `requires-python = ">=3.14"`, `.python-version = 3.14`, and `uv.lock`
  pins `>=3.14` — so `uv` refuses a 3.12 environment today (the consumer is `>=3.12,<3.13`).
- **3.12 syntax safety (verified)**: no PEP 695 type-alias or generic syntax, no 3.13+
  stdlib (`@override`, `itertools.batched`, `TypeIs`, `warnings.deprecated` all absent —
  the "override" hits are comments). Lowest-floor construct is `datetime.UTC`
  (`harness/runner.py`, `scripts/run_cmmd_backtest.py`), which is **3.11+**. ⇒ the floor can
  drop to 3.11/3.12 with **zero code changes**; Req 2 is a pin + CI task, not a porting task.
- **Test wiring**: no `conftest.py`, no `[tool.pytest.ini_options]`. `from src.*` resolves
  because `src/__init__.py` makes `src` a package and pytest inserts the repo root on
  `sys.path`. After packaging, tests should import the final package name.
- **Architecture guard**: `.sentrux/rules.toml` enforces layer order
  (`core ← {dataset,mia,portfolio} ← harness`), 0 cycles, CC ≤ 25, ≤ 120 lines/fn, and the
  `dspy` ban. Its layer globs are `src/<layer>/*` — they must be retargeted on rename.
- **Heavy / optional deps (verified by import scan)**:
  - `matplotlib` → `portfolio/backtest.py` (Agg PNG) **and** `harness/plots.py`.
  - `vectorbt` → `portfolio/backtest.py` only.
  - `rich` → `harness/report.py` only.
  - `core` and `mia` import **none** of these — they are already lean.
  - ⚠️ **`src/harness/__init__.py` eagerly imports `plots`** (hence matplotlib). Any façade
    that re-exports through the `harness` package will drag matplotlib into a bare
    `import recall_guard` unless that `__init__` is made lazy or the façade avoids it.

## 2. Requirement → asset map (gaps tagged)

| Req | Existing asset | Gap |
|---|---|---|
| **1** Installable distro | name `recall-guard` set | **Missing** `[build-system]`; **Constraint** `src.*` layout (140 imports) blocks a clean import name |
| **2** Python 3.12 | code is 3.12-safe (verified) | **Missing** floor relax in pyproject + uv.lock + `.python-version`; **Missing** 3.12 CI run |
| **3** Façade | `NvidiaLM`, `build_baseline`/`ControlBaseline`, `mcs.train`/`MCSCalibrator` (+`is_weak`), `compute_mia_features`, evaluator parser + MemGuard formula; `NvidiaLM` already raises on empty key | **Missing** a single `MemoryGuardedScorer` façade object; **Constraint** building blocks span `core`+`mia`+`harness` |
| **4** Lean deps + extras | one flat dependency list; core+mia dep-free | **Missing** extras split (jupyter/pytest/pre-commit/ruff/nb*/matplotlib out of runtime); **Constraint** `harness/__init__`→plots→matplotlib eager import |
| **5** Preserve arch/tests | `.sentrux/rules.toml`, dspy ban, 241 green tests | **Constraint** rename must retarget sentrux globs + 140 imports + test imports |
| **6** Reproducible pipeline | none | **Missing** entirely (new Dagger module) |
| **7** GHA management | none (`.github/` absent) | **Missing** entirely (ci/release/docs workflows; OIDC; Pages) |
| **8** Docs site | rich NumPy-style docstrings + type hints + `__all__`; markdown-first repo | **Missing** mkdocs config + autodoc site + publish |
| **9** Integration recipe | README "Use as a package" stub → this spec | **Missing** runnable recipe + example |

## 3. Implementation approach options

### 3a. Packaging layout — the pivotal decision

- **Option A — rename `src/ → recall_guard/`** and rewrite the 140 `src.` imports across 44
  files (mechanical `sed`), retarget the sentrux globs, and switch test imports to
  `recall_guard.*`.
  - ✅ Installed name == imported name == test import; no namespace landmine; simplest mental
    model; the green suite catches any miss.
  - ❌ Large (but mechanical) diff; touches scripts' `sys.path` splices.
  - **Recommended.** The 140 absolute `src.` imports make the import name part of the
    package's identity, so the only clean fix is for that name to *be* `recall_guard`.
- **Option B — keep `src/`, convert internals to relative imports, remap `src → recall_guard`
  in the build backend.**
  - ✅ Fewer edited import lines.
  - ❌ Tests still `import src.*` (break against the installed wheel); editable-install
    semantics get ambiguous; long-term confusion. Net more fragile than A.
- **Option C — Option A + a transitional `src` shim** that re-exports `recall_guard` for the
  notebooks/scripts during migration.
  - ✅ Non-breaking for existing notebooks/`scripts/` while they migrate.
  - ❌ Extra surface to delete later; risk it lingers.

### 3b. Façade (Req 3)

- **Option A — extend `harness`** with the scorer. ❌ `harness/__init__` pulls matplotlib;
  conflicts with Req 4 AC3 unless that `__init__` is de-eagered.
- **Option B — new top-level façade** (`recall_guard/__init__.py` or `recall_guard/guard.py`)
  importing only from `core` + `mia` (both matplotlib/vectorbt-free), exposing a small
  `MemoryGuardedScorer`. ✅ Keeps `import recall_guard` lean. **Recommended**, paired with
  making `harness/__init__` lazy-import `plots`.
- **Option C — hybrid**: façade in a new module + refactor `harness/__init__` to lazy plots.

### 3c. Build backend

- **hatchling** (recommended — modern, minimal, `packages = ["recall_guard"]`, uv-native)
  vs **setuptools** (package discovery). Either works once the dir is `recall_guard/`.

### 3d. Pipeline / CI / docs (all net-new — "create new")

- New `dagger/` Python-SDK module (test/lint/build/docs/publish); thin `.github/workflows/`
  (`ci.yml`, `release.yml`, `docs.yml`) calling it; `mkdocs.yml` + `mkdocs-gen-files` /
  `mkdocs-literate-nav` for autogen nav.

## 4. Effort & risk (workstreams)

| # | Workstream (reqs) | Effort | Risk | Note |
|---|---|---|---|---|
| W1 | build-system + layout rename (R1, R5) | **M** | Low-Med | mechanical; tests are the safety net |
| W2 | Python floor relax + 3.12 verify (R2) | **S** | Low | no code change; verify 3.12 lock resolves |
| W3 | dependency extras + lazy heavy imports (R4) | **S-M** | Med | matplotlib eager import in `harness/__init__` is the snag |
| W4 | façade API + parity tests (R3) | **M** | Med | new public contract; assert `p_memorized` parity vs harness |
| W5 | Dagger pipeline (R6) | **M** | Med | new tech for the repo; mature SDK |
| W6 | GHA workflows + OIDC + Pages (R7) | **S-M** | Med | one-time PyPI trusted-publisher + Pages permission setup |
| W7 | docs site via mkdocstrings (R8) | **S-M** | Low | docstrings already present |
| W8 | integration recipe + example (R9) | **S** | Low | depends on W4 façade shape |

**Dependency order**: `W1 → {W2, W4} → W3 → W5 → {W6, W7} → W8` (W2 can run alongside W1;
build/publish/docs-autodoc all gate on W1).

## 5. Research Needed (carry to design)

- Confirm the pinned versions (numpy 1.26 / scikit-learn 1.4 / vectorbt 0.28 + numba /
  llvmlite) co-resolve on **Python 3.12** under uv (expected fine — they match the
  consumer's own stack — but produce a 3.12 lock to prove it).
- **Publish target** for Req 7 AC2: public **PyPI** (check `recall-guard` name availability)
  vs a private index vs git-only / GitHub-release artifact. Drives OIDC trusted-publisher
  config.
- **Docs hosting**: project GitHub Pages; deploy on every default-branch push vs only on
  release; strict (griffe) build for Req 8 AC5.
- **mkdocstrings handler**: `docstring_style: numpy`; confirm no docstrings need
  normalization for griffe strict mode.
- **Dagger pinning**: Dagger engine version + `dagger/dagger-for-github` action; runner
  image must carry manylinux wheels for vectorbt/numba.
- **Transitional `src` shim** (Option C): keep for the notebooks/`scripts/` or migrate them
  in the same pass?
- **Exact floor**: code permits **3.11**; spec sets **3.12** to match the consumer —
  confirm 3.12 (vs 3.11 for wider reuse).

## 6. Recommendation for design

- **Layout**: Option A (rename `src/ → recall_guard/`); reconsider Option C shim only if the
  notebooks/scripts can't migrate in the same pass.
- **Façade**: Option B (a `core`+`mia`-only `MemoryGuardedScorer`) + de-eager
  `harness/__init__`'s plots import, so `import recall_guard` stays matplotlib-free.
- **Backend**: hatchling. **Pipeline/CI/docs**: GHA-calls-Dagger + MkDocs/mkdocstrings
  (already locked).
- **Open decisions for design to commit**: publish target (PyPI vs git), docs deploy
  cadence, transitional shim yes/no, floor 3.12 vs 3.11.

---

# Design Discovery & Synthesis

_Appended 2026-06-25 during `/kiro-spec-design`._

## Summary
- **Discovery scope**: Extension (packaging/refactor of an existing layered codebase) plus
  net-new automation components. Light discovery; the codebase is already mapped above.
- **Key findings**: the only code change for a lean `import recall_guard` is de-eagering the
  `harness/__init__` plots import (reuse the PEP 562 `__getattr__` pattern already in
  `dataset/__init__`); the 3.12 floor relax needs no porting; the façade is pure
  composition of existing primitives, so parity (Req 3.3) is free.

## Architecture Pattern Evaluation
| Option | Description | Strengths | Risks | Verdict |
|--------|-------------|-----------|-------|---------|
| Layout A: rename `src→recall_guard` | move dir, rewrite 140 imports | name==import==test; no `src` landmine | large mechanical diff incl. notebooks | **Selected** |
| Layout B: keep `src`, build remap | relative imports + hatch sources remap | fewer edited lines | tests import `src.*` break vs wheel; editable ambiguity | Rejected |
| Façade in harness + lazy plots | order-0 façade; lazy `__init__` | reuses evaluator parser; lean import | `__init__` must stay lazy | **Selected** |
| Façade in root (core+mia only) | relocate parser | zero harness import | relocating parser touches evaluator+tests | Rejected (more churn) |

## Design Decisions

### Decision: Layout = rename `src/ → recall_guard/` (no shim)
- Alternatives: A rename; B build-time remap; C rename + transitional `src` shim.
- Selected: A, no shim — notebooks/scripts are in-repo and migrate in the same pass.
- Rationale: 140 absolute `src.` imports make the import name part of the package's identity;
  only a real rename gives name==import==test-import. A shim adds lingering surface.
- Trade-offs: large but mechanical diff; the 241-test suite + sentrux are the safety net.

### Decision: Lean import via lazy plots (not façade relocation)
- Selected: keep façade in `harness`; convert `harness/__init__` to PEP 562 lazy
  `__getattr__` for `plot_*`; matplotlib/vectorbt → `backtest` extra.
- Rationale: matches the existing `dataset/__init__` pattern; preserves the tested
  `from recall_guard.harness import plot_*` re-export; smallest change.

### Decision: Distribution channel = Git + GitHub Releases primary, PyPI optional
- Selected: tag → build → GitHub Release asset + git-installable; PyPI publish wired behind
  OIDC but opt-in (activated when a public project name exists).
- Rationale: the known consumer uses a uv git dependency; avoids PyPI name/maintenance
  overhead for a thesis dependency while still satisfying Req 7 (configurable index).
- Trade-offs: consumers pin a tag/commit; public discoverability deferred.

### Decision: Python floor = 3.12; Docs deploy on default-branch push
- Floor 3.12 == consumer pin; CI matrix 3.12 + dev 3.14. 3.11 is technically possible
  (only `datetime.UTC`) but not claimed because it is not in the matrix.
- `docs.yml` deploys on push to main (release also rebuilds docs) — simplest cadence for 8.4.

## Build vs Adopt
- **Adopt**: hatchling, Dagger SDK, MkDocs + mkdocstrings, `pypa/gh-action-pypi-publish` +
  OIDC, `actions/deploy-pages`.
- **Build (thin)**: `MemoryGuardedScorer` façade — domain glue over recall_guard primitives.
- **Reuse unchanged**: all `mia`/`core`/evaluator statistical code.

## Simplification
- One `MemoryGuardedScorer`; no detector-plugin abstraction (MCS is the only detector).
- Minimal mkdocs config; nav auto-generated; no hand-maintained API pages.
- Dagger module = 5 functions + 1 base-container helper.

## Risks & Mitigations
- 3.12 dependency co-resolution (numpy/sklearn/vectorbt/numba) — build + test a 3.12 lock in
  CI before release.
- Rename churn breaks committed notebooks / `test_notebook` — regenerate notebooks; run the
  full suite (incl. the notebook-execution test) on 3.12 + 3.14.
- Lazy-plots regression (eager import re-added) — add a `sys.modules` assertion test that
  `import recall_guard` pulls no `matplotlib`/`vectorbt`.
