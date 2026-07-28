# recall-guard-package — Requirements

## Introduction

This spec turns `recall-guard` from a pair of CLIs into an installable Python library
(importable as `recall_guard`) so that **Global_Macro_AI_Factors**
([github.com/norandom/Global_Macro_AI_Factors](https://github.com/norandom/Global_Macro_AI_Factors),
the `macro_framework` package) can depend on it and consume **measured
inference-without-recall** as a factor input.

recall_guard already grades NIM-hosted models without crediting memorised data: per-token
logprobs → five MIA features → a per-model MCS calibrator → `p_memorized` → the MemGuard
confidence discount `raw_confidence * (1 - p_memorized)`. The suite is green and the NIM
endpoint is live. What is missing is the packaging, a stable public façade, and the
release/documentation machinery a downstream project needs.

The consumer's Track A (`llm_agent.py`) is a zero-temperature DSPy agent that reasons from
anonymised, z-scored macro state and "never sees a date, a year, or a real ticker" — it
enforces recall-avoidance qualitatively. recall_guard supplies the missing quantitative
half: a measured `p_memorized` per prompt, computed from per-token logprobs that DSPy
hides. Predictive alpha is explicitly **not** a goal; near-coin-flip directional accuracy
is the correct result, and the package's contract is honest inference plus a measurable
contamination score, not forecasting skill.

Packaging is the foundation; the reproducible pipeline, GitHub-Actions-managed
release/CI/docs, and the API-documentation site are layered on top and depend on the
package building and importing under its final name.

## Boundary Context

- **In scope**: an installable `recall_guard` distribution (build backend, wheel + sdist,
  import without `sys.path` edits); a relaxed Python floor (3.12) with the suite green on
  3.12 and 3.14; a stable public façade for inference-without-recall (signal,
  `p_memorized`, raw MIA features, discounted confidence); lean runtime dependencies with
  `backtest`/`docs` consumer extras and `dev`/`pipeline` uv dependency groups;
  preservation of the layered architecture, the
  DSPy ban, and all existing tests and CLI behaviour; a reproducible test/lint/build/docs/
  publish pipeline; GitHub-Actions-managed CI, release, and docs workflows; an
  autodoc API-documentation site; and a consumer integration recipe.
- **Out of scope**: any predictive-alpha objective; re-designing the bundled ETF eval
  prompts to chase accuracy (they are near-coin-flip by design); changing the NIM endpoint
  request/response contract; changing the behaviour of the existing CLIs (`start.sh`
  recall-guard check; `scripts/run_cmmd_backtest.py`); building `macro_framework` features
  (the consumer owns its DSPy agent, factor pipeline, and data sources).
- **Adjacent expectations**:
  - The consumer supplies its own prompt stream (e.g. anonymised macro state) and DSPy
    agent; recall_guard supplies inference + `p_memorized` only and must work on an
    arbitrary prompt stream, not just the bundled ETF eval set.
  - Live scoring requires the NVIDIA NIM endpoint and a valid `NVIDIA_API_KEY` at runtime;
    the package does not own key provisioning or rotation.
  - The consumer pins Python `>=3.12,<3.13` and already uses DSPy and
    `vectorbt>=0.28.2`; recall_guard must be compatible with that environment.
- **Locked tooling decisions (2026-06-25)** — recorded here as constraints the design
  phase must honour, not re-open:
  - **Pipeline engine**: a Python-SDK **Dagger** module is the execution engine; the
    acceptance criteria below describe pipeline operations tool-neutrally.
  - **Control plane**: **GitHub Actions** triggers and manages CI/release/docs and calls
    the Dagger module; publishing uses **OIDC trusted publishing** (no stored tokens);
    docs deploy to **GitHub Pages**.
  - **API docs**: **MkDocs + Material + `mkdocstrings[python]`** (griffe, NumPy-style
    docstrings).

## Requirements

### Requirement 1: Installable distribution

**Objective:** As a maintainer of a downstream project, I want recall_guard to be a
standard installable Python distribution, so that I can declare it as a dependency and
import it without source-tree hacks.

#### Acceptance Criteria

1. When a consumer installs the distribution into a fresh environment, the recall_guard
   package shall be importable as `import recall_guard` without adding the source tree to
   `sys.path`.
2. When the distribution is built, the recall_guard distribution shall produce a wheel and
   a source distribution from a declared build backend.
3. The recall_guard distribution shall not expose a top-level `src` import package to
   consumers.
4. Where the distribution is consumed as a version-control dependency, the recall_guard
   distribution shall be installable directly from its Git repository.

### Requirement 2: Python version compatibility

**Objective:** As a consumer pinned to Python 3.12, I want recall_guard to install and run
on my interpreter, so that it fits my environment without forcing an upgrade.

#### Acceptance Criteria

1. The recall_guard distribution shall declare a minimum supported Python version of 3.12.
2. When the test suite runs on Python 3.12, the recall_guard package shall pass all
   pre-existing tests.
3. When the test suite runs on Python 3.14, the recall_guard package shall pass all
   pre-existing tests.
4. If a consumer environment uses a Python version below the declared minimum, the
   installer shall refuse installation with a clear version-mismatch error.

### Requirement 3: Inference-without-recall public façade

**Objective:** As a factor researcher, I want a stable public API that runs a model on a
prompt and returns the signal together with a memorisation score, so that I can use
honest, recall-filtered inference as a factor input.

#### Acceptance Criteria

1. When a caller scores a prompt through the public façade, the recall_guard public API
   shall return the parsed directional signal, the raw MIA features, the calibrated
   `p_memorized` in `[0, 1]`, and the memguard-discounted confidence.
2. The recall_guard public API shall expose its supported public symbols from the
   top-level `recall_guard` namespace.
3. The recall_guard public API shall compute `p_memorized` using the same MIA-feature and
   calibration method already validated by the existing harness, producing equal values
   for equal inputs.
4. While a per-model calibrator is uncalibrated or below the weak-calibration threshold,
   the recall_guard public API shall surface that status to the caller rather than
   returning a silently invalid score.
5. Where the caller supplies a reference model, the recall_guard public API shall include
   the reference-delta feature; where no reference model is supplied, it shall omit that
   feature without error.
6. When the caller scores an arbitrary prompt stream, the recall_guard public API shall
   operate without requiring the bundled ETF eval set or any specific prompt schema.
7. If the NIM credential is absent or rejected at scoring time, the recall_guard public
   API shall raise a clear configuration error rather than returning a score.

### Requirement 4: Lean runtime dependencies and extras

**Objective:** As a consumer, I want recall_guard's default install to pull only the
dependencies needed for inference and scoring, so that I do not inherit notebook, test,
plotting, or pipeline tooling.

#### Acceptance Criteria

1. When a consumer installs the default distribution, the recall_guard distribution shall
   install only the dependencies required for inference and memorisation scoring.
2. The recall_guard distribution shall provide separate consumer-facing optional extras
   for documentation (`docs`) and plotting/backtest (`backtest`); development and
   pipeline tooling shall be kept out of the distribution metadata as uv dependency
   groups (`dev`, `pipeline`), installed from a checkout via `uv sync`, never by
   consumers. (Aligned with the as-built contract by the review-hardening spec:
   the published install surface documents exactly the `backtest` and `docs` extras.)
3. When the documentation, plotting, and development extras are not installed, the core
   inference and scoring API shall import and run without error.
4. The recall_guard distribution shall declare a `vectorbt` requirement compatible with a
   consumer constraint of `>=0.28.2`.

### Requirement 5: Architecture and behaviour preservation

**Objective:** As the recall_guard maintainer, I want packaging to preserve the existing
architecture, test coverage, and dependency bans, so that the library stays correct and
honest.

#### Acceptance Criteria

1. The recall_guard package shall preserve the existing layered import structure, with the
   core layer depended on by the feature layers and no upward imports into the core layer.
2. The recall_guard package shall contain no import cycles.
3. The recall_guard package shall not depend on DSPy.
4. When the full test suite runs after packaging changes, the recall_guard package shall
   keep all pre-existing tests passing.
5. The recall_guard package shall keep the existing CLI workflows (recall-guard check and
   the CMMD backtest) functional with unchanged observable behaviour.
6. The recall_guard package shall not alter the NIM endpoint request or response contract.

### Requirement 6: Reproducible release pipeline

**Objective:** As a maintainer, I want one pipeline that runs the same steps locally and in
CI, so that releases are deterministic and debuggable off-CI.

#### Acceptance Criteria

1. The release pipeline shall expose discrete operations for test, lint, build,
   documentation, and publish.
2. When the pipeline test operation runs, it shall execute the test suite across the
   supported Python matrix.
3. When the pipeline lint operation runs, it shall execute the configured linter and the
   structural-architecture check.
4. When a maintainer runs a pipeline operation locally, it shall produce the same outcome
   it produces in CI for the same inputs.
5. When the pipeline build operation runs, it shall produce the same wheel and source
   distribution that the release workflow publishes.

### Requirement 7: GitHub Actions release management

**Objective:** As a maintainer, I want GitHub Actions to trigger and manage CI, releases,
and docs, so that the project lifecycle is automated from pushes and tags.

#### Acceptance Criteria

1. When a pull request is opened or updated, the CI workflow shall run the pipeline test
   and lint operations and report a pass/fail status.
2. When a maintainer pushes a version tag matching the release pattern, the release
   workflow shall build the distribution and publish it to the configured package index.
3. When the release workflow publishes a version, it shall create a GitHub Release with
   the built wheel and source distribution attached.
4. The release workflow shall publish to the package index without storing long-lived
   credentials in the repository.
5. If any pipeline operation fails during a workflow run, the workflow shall fail and
   shall not publish artifacts.

### Requirement 8: API documentation site

**Objective:** As a consumer, I want browsable API documentation generated from the code's
docstrings and types, so that I can learn the façade without reading source.

#### Acceptance Criteria

1. When the documentation operation runs, the documentation site shall be generated from
   the package's public API docstrings and type hints.
2. The documentation site shall present the public `recall_guard` façade as the primary
   reference and shall not surface internal-only modules as the primary reference.
3. When a documented public symbol's signature or docstring changes, the regenerated site
   shall reflect the change without manual page edits.
4. When the docs workflow runs on the default branch, it shall publish the generated site
   to the project's hosted documentation location.
5. If the documentation build cannot resolve a referenced public symbol, the documentation
   operation shall fail rather than publish a broken reference.

### Requirement 9: Consumer integration recipe

**Objective:** As a Global_Macro_AI_Factors developer, I want a documented way to add
recall_guard and call it from `macro_framework`, so that I can wire measured
inference-without-recall into the factor pipeline.

#### Acceptance Criteria

1. The recall_guard documentation shall provide a recipe for declaring recall_guard as a
   dependency of a uv-managed project, including the version-control dependency form.
2. The recall_guard documentation shall provide a minimal runnable example that scores a
   prompt through the public façade and reads back the signal and `p_memorized`.
3. Where the consumer integrates the façade into its own agent or scoring step, the
   recall_guard documentation shall state which inputs the package requires and which
   responsibilities remain with the consumer.
