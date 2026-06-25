# Implementation Plan

Tasks follow the design's dependency order: packaging foundation (the high-fan-out rename
gates everything) → public façade → documentation site → reproducible pipeline → GitHub
Actions control plane → cross-cutting validation. Requirement IDs are from
`requirements.md`; component boundaries are from `design.md`.

- [x] 1. Packaging foundation: a distributable recall_guard package
- [x] 1.1 Rename the source tree to the public package name
  - Move the source package to the `recall_guard` import name and rewrite every internal
    import from the `src.` namespace to `recall_guard.` across the package, tests, scripts,
    and the root CLI entry point
  - Retarget the structural-architecture layer definitions and the banned-dependency rule to
    the new path; update the notebook builders' embedded imports and the committed notebooks
  - Observable: the full test suite passes with no remaining `src` import and no top-level
    `src` package; the structural check still reports the layered order with zero cycles and
    no banned dependency; both existing CLIs still run unchanged
  - _Requirements: 1.3, 5.1, 5.2, 5.3, 5.4, 5.5, 5.6_

- [x] 1.2 Add the build backend and lower the Python floor
  - Add a build system and wheel target so the project builds a wheel and source
    distribution under the `recall_guard` name; lower the supported Python floor to 3.12 and
    regenerate the environment lock
  - Observable: a built wheel installs into a fresh environment and `import recall_guard`
    succeeds with no source tree on the path; building emits both a wheel and an sdist; an
    interpreter below the floor is refused at install resolution
  - _Requirements: 1.1, 1.2, 1.4, 2.1, 2.4_

- [x] 1.3 Defer the plotting import in the harness package initializer
  - Make the harness package initializer expose the plotting helpers lazily, reusing the
    on-demand attribute pattern already used by the dataset package, so importing the package
    does not pull the plotting stack
  - Observable: importing the package and the harness subpackage loads no plotting or
    backtest library, while accessing a plotting helper still resolves it on demand
  - _Requirements: 4.3_

- [x] 1.4 Split runtime dependencies from optional extras
  - Reduce the default dependency set to what inference and scoring require; move
    plotting/backtest, documentation, and development/test tooling into named optional
    extras; keep the backtest plotting dependency compatible with the consumer's pinned
    constraint
  - Observable: a default install excludes the plotting/backtest, notebook, and test tooling,
    and the core inference + scoring API imports and runs with no extra installed
  - _Requirements: 4.1, 4.2, 4.4_

- [x] 2. Inference-without-recall public façade
- [x] 2.1 Calibration entry point for the guarded scorer
  - Build the scorer's calibration path that runs the model over the control and IS/OOS
    corpora and trains the per-model contamination calibrator, reusing the existing baseline
    and calibrator components unchanged
  - Surface calibrator quality (held-out separation and weak flag) to the caller; raise a
    clear configuration error when the credential is missing or empty and a clear error when
    the endpoint rejects it; raise on an uncalibrated baseline
  - Observable: calibrating returns a scorer exposing its held-out separation and weak flag;
    an empty credential raises the configuration error; a single-class or uncalibrated corpus
    raises
  - _Requirements: 3.4, 3.5, 3.7_
  - _Boundary: MemoryGuardedScorer_

- [x] 2.2 Per-prompt guarded scoring
  - Implement single- and batch-prompt scoring returning the parsed direction, the raw
    memorization features, the calibrated memorization probability, and the discounted
    confidence, reusing the existing parser, feature, standardisation, and
    calibrator-prediction components
  - Include the reference-delta feature only when a reference model was configured; operate on
    any prompt without requiring the bundled eval set
  - Observable: scoring a prompt returns the documented result whose discounted confidence
    equals raw confidence times one-minus-probability; a parity test shows the memorization
    probability equals the existing harness's value for identical inputs
  - _Requirements: 3.1, 3.3, 3.5, 3.6_
  - _Boundary: MemoryGuardedScorer_
  - _Depends: 2.1_

- [x] 2.3 Curate the top-level public API
  - Define the package root's public surface so the façade and a curated set of core/scoring
    names are importable from the top level, explicitly excluding any plotting symbol and
    anything from the backtest layer
  - Observable: the façade imports from the top-level package; the package's declared public
    surface contains no plotting name; importing the package loads no plotting or backtest
    library, asserted against the loaded-modules set
  - _Requirements: 3.2, 4.1, 4.3_
  - _Boundary: recall_guard root API_
  - _Depends: 2.2_

- [x] 3. API documentation site
- [x] 3.1 Autodoc reference generation
  - Configure the documentation site to generate the API reference from the package's public
    docstrings and type hints, presenting the façade as the primary reference and failing the
    build on an unresolved reference
  - Observable: a strict documentation build succeeds and produces a façade reference page
    generated from docstrings; changing a public docstring or signature changes the rendered
    page with no manual page edit
  - _Requirements: 8.1, 8.2, 8.3, 8.5_
  - _Boundary: Docs site_
  - _Depends: 2.3_

- [x] 3.2 Integration recipe and runnable example
  - Author the landing documentation with a recipe for declaring the package as a
    version-controlled dependency of a uv-managed project, a minimal runnable example that
    scores a prompt and reads back the signal and memorization probability, and a statement
    of required inputs versus consumer responsibilities
  - Observable: the docs contain the dependency recipe and an example whose code path matches
    the façade signature; the input-versus-responsibility split is documented
  - _Requirements: 9.1, 9.2, 9.3_
  - _Boundary: Docs site_
  - _Depends: 2.3_

- [x] 4. Reproducible pipeline
- [x] 4.1 Pipeline module with test and lint operations
  - Scaffold the container-run pipeline module with a shared base-environment builder and
    discrete test and lint operations; the test operation runs the suite across the supported
    Python matrix and the lint operation runs the linter plus the structural-architecture
    check
  - Observable: invoking the pipeline test operation runs the suite in a container for each
    matrix version and the lint operation runs linter plus structural check, producing the
    same result locally and in CI
  - _Requirements: 6.1, 6.2, 6.3, 6.4_
  - _Boundary: Dagger pipeline module_
  - _Depends: 1.2_

- [x] 4.2 Build, docs, and publish operations
  - Add the build operation (wheel and sdist), the documentation-build operation, and the
    publish operation that consumes the build artifacts
  - Observable: the build operation emits the same wheel and sdist that the release path
    publishes; the docs operation builds the site; the publish operation yields the artifacts
    ready for upload
  - _Requirements: 6.1, 6.5_
  - _Boundary: Dagger pipeline module_
  - _Depends: 1.2, 3.1_

- [x] 5. GitHub Actions control plane
- [x] 5.1 (P) Continuous integration workflow
  - Add a workflow that, on pull request and push, calls the pipeline test operation across
    the matrix and the lint operation and reports a pass/fail status
  - Observable: opening or updating a pull request runs test and lint through the pipeline and
    reports status
  - _Requirements: 7.1_
  - _Boundary: ci workflow_
  - _Depends: 4.1_

- [x] 5.2 (P) Release workflow
  - Add a workflow that, on a version tag, calls the pipeline build, publishes the wheel to
    the configured index without storing long-lived credentials, and creates a release with
    the wheel and source distribution attached; any failed pipeline operation aborts before
    publish
  - Observable: pushing a version tag builds and creates a release with artifacts attached
    using ephemeral credentials; an injected operation failure blocks publish
  - _Requirements: 7.2, 7.3, 7.4, 7.5_
  - _Boundary: release workflow_
  - _Depends: 4.2_

- [x] 5.3 (P) Documentation publish workflow
  - Add a workflow that, on push to the default branch, calls the pipeline docs operation and
    publishes the generated site to the hosted documentation location
  - Observable: a push to the default branch publishes the regenerated site
  - _Requirements: 8.4_
  - _Boundary: docs workflow_
  - _Depends: 4.2_

- [x] 6. Cross-cutting validation
- [x] 6.1 Cross-version and structural validation
  - Verify the suite passes on both supported Python versions through the pipeline matrix, the
    structural-architecture check passes (layer order, zero cycles, no banned dependency), and
    the existing CLIs still run their parsers unchanged
  - Observable: the matrix test run is green on both versions, the structural check passes, and
    both CLIs start and parse arguments without error
  - _Requirements: 2.2, 2.3, 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 6.2_
  - _Depends: 4.1_

- [x] 6.2 End-to-end release and install dry-run
  - Exercise the build and publish path end to end: confirm the pipeline build output equals
    what the release workflow would publish, run the release workflow in dry-run to confirm
    credential-free publish and failure-blocks-publish behavior, run the strict documentation
    build, and install the built wheel into a fresh environment
  - Observable: the dry-run produces a release without stored credentials, a forced operation
    failure blocks publish, the strict documentation build succeeds, and the fresh-environment
    install imports the package
  - _Requirements: 1.1, 6.4, 6.5, 7.4, 7.5, 8.5_
  - _Depends: 4.2, 5.2, 5.3_

## Requirements coverage

All requirement IDs 1.1–9.3 are mapped: 1.x → 1.1/1.2/1.3/1.4; 2.x → 1.2/6.1; 3.x → 2.1/2.2/2.3;
4.x → 1.3/1.4/2.3; 5.x → 1.1/6.1; 6.x → 4.1/4.2/6.1/6.2; 7.x → 5.1/5.2; 8.x → 3.1/5.3/6.2;
9.x → 3.2. No requirements deferred.

## Implementation Notes

- **Major 1 done (2026-06-25)** on branch `feat/recall-guard-package`. Evidence: suite 241
  passed on Python 3.12 and 3.14; sentrux 11/11 rules pass (0 violations); `uv build`
  produces wheel+sdist; fresh-venv install of the wheel imports `recall_guard` + core + mia
  + harness with `matplotlib`/`vectorbt` absent from `sys.modules`; install into Python 3.11
  is refused by `requires-python`.
- **Rename gotcha**: the `src.<layer>` namespace also appears in *string literals* —
  `mocker.patch("src.portfolio.prices...")` targets, `caplog ... logger="src.core.loader"`
  names, the `python -m ...` CLI `prog`, and the dataset `AttributeError` message — not just
  `from src.` imports. A plain import sed left 5 failing mock-target strings; rewrite
  `src.(core|mia|harness|portfolio|dataset)` everywhere, including strings/comments.
- **Dependency model**: runtime deps are lean; `backtest` + `docs` are `[project.optional-dependencies]`
  extras; test/lint/notebook/docs/dagger tooling lives in `[dependency-groups]`. uv installs
  the `dev` group by default, so `uv run pytest` works out of the box while consumers inherit
  only the lean runtime set.
- **Lazy plots**: `harness/__init__` resolves `plot_*`/`configure_paper_style` via PEP 562
  `__getattr__` (mirrors `dataset/__init__`). Do not re-add an eager `from ...plots import`
  to `harness/__init__` or the package root — it would re-pull matplotlib onto the
  `import recall_guard` path (Req 4 regression). A `sys.modules` assertion guards this.
- **Major 2 done (2026-06-25)**: `recall_guard/harness/scorer.py`
  (`MemoryGuardedScorer`/`GuardedScore`/`ConfigurationError`) + curated root `__all__`.
  Evidence: 253 tests pass (12 new), sentrux 11/11, ruff clean. The façade reuses the
  evaluator's parser + the MIA/MCS functions verbatim, so a parity test asserts
  `score().p_memorized == evaluate_model(...).records[0].p_memorized` (Req 3.3). Auth
  failures (401/403) raise `ConfigurationError` at calibrate (0 usable rows) and at score;
  timeouts/parse failures return a `GuardedScore` failure record instead. A subprocess test
  asserts `import recall_guard` loads no matplotlib/vectorbt and `__all__` has no `plot_*`.
- **Major 3 done (2026-06-25)**: `mkdocs.yml` + `docs/index.md` + `docs/gen_ref_pages.py`
  (mkdocstrings/griffe, numpy docstrings, `strict: true`). Evidence: `mkdocs build --strict`
  succeeds; `site/reference/recall_guard/index.html` documents `MemoryGuardedScorer`
  (façade-first); 258 tests pass (5 docs tests); ruff + sentrux clean. Notes: griffe is
  static, so it documents the matplotlib/vectorbt-importing modules without importing them.
  `exclude_docs: gen_ref_pages.py` keeps the generator out of the published site.
  mkdocs-material prints a third-party "mkdocs 2.0 / ProperDocs" advisory banner to stderr —
  noise, not a build failure (the strict build returns 0).
- **Major 4 done (2026-06-25)**: Dagger Python-SDK module at `ci/` (name `ci`, v0.21).
  Functions: `base`, `test`, `test_matrix`, `lint`, `build`, `docs`, `publish`. **Verified
  with the real engine** (docker available): `dagger -m ci functions` loads + type-checks
  all seven; `dagger -m ci call build` runs in a container and exports the recall_guard
  wheel + sdist. `ci/sdk` is gitignored (regenerated by `dagger develop`); `ci` + `site`
  added to the repo ruff `extend-exclude` so the vendored SDK / build output don't pollute
  the lint gate (sentrux still 11/11). Scoping notes: the dagger `lint` runs ruff; the
  sentrux structural check stays a separate step (sentrux is an MCP plugin, not
  pip-installable into a container).
- **Distribution decision (2026-06-25, user): GitHub Release only — no PyPI.** Removed the
  dagger `publish` function and the release workflow's PyPI/OIDC step; the dagger module is
  now base/test/test_matrix/lint/build/docs (6 functions). Consumers install via the uv git
  dependency or the GitHub Release asset. The "publish" pipeline op (Req 6.1) is realized by
  the release workflow's GitHub Release step, not a dagger function.
- **Major 5 done (2026-06-25)**: `.github/workflows/{ci,release,docs}.yml` — thin shims that
  install the pinned Dagger CLI (v0.21.4) and call the `ci` module. ci: test (3.12+3.14
  matrix) + lint on PR/push. release (`v*` tag): test-matrix + build gates, then a GitHub
  Release with wheel+sdist attached (failure of either gate aborts before the release step —
  Req 7.5). docs (main push): strict docs build -> GitHub Pages. Verified locally: all YAML
  parses; every `dagger call` target resolves to a real module function; third-party action
  (`softprops/action-gh-release`) pinned to a commit SHA per the semgrep hook; GitHub Release
  uses the ephemeral `GITHUB_TOKEN` (no stored long-lived creds — Req 7.4).
- **Major 6 (2026-06-25)**: 6.1 verified — full suite **258 passing on Python 3.12 and
  3.14**, sentrux 11/11, ruff clean, both CLIs (`harness.py`, `run_cmmd_backtest.py`) still
  parse post-rename. 6.2 verified locally where possible: `dagger call build` -> wheel+sdist
  (the exact set the release workflow uploads), strict docs build, and fresh-venv lean
  install all pass.
- **Deferred live verification (no sandbox capability)**: the GitHub Actions *runtime* has
  not executed — PR-triggered CI, tag-triggered GitHub Release creation, runtime
  failure-blocks-release, and Pages deploy. The workflows are structurally validated (YAML +
  resolvable dagger targets + pinned actions) and the Dagger functions they call are
  engine-verified; a first push to GitHub is needed to confirm the triggers end-to-end.
