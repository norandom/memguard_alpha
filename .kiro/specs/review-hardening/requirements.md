# review-hardening — Requirements

## Introduction

This spec turns the confirmed code-review findings into a coordinated hardening pass for `recall_guard`. The goal is not to add new product features. The goal is to make the existing harness, scoring façade, CMMD backtest, dataset builders, manifest/reporting layer, and package automation fail clearly, preserve provenance, and produce outputs whose published metrics match the behavior the code actually executed.

The feature is successful when invalid inputs, malformed provider responses, calibration failures, and partial-write situations stop producing false-success runs; when shared parsing and provenance rules behave consistently across the harness and the post-processing scripts; and when the portfolio/backtest artifacts and dataset builders are trustworthy enough to use as research evidence.

## Boundary Context

- **In scope**: harness failure handling; shortlist gating; parser correctness for direction and confidence; LM-client payload validation and pacing; bootstrap-argument validation; manifest/input provenance; CMMD filtering semantics; backtest accounting and artifact integrity; eval-row metadata joins used by post-harness scripts; FMP corpus date/dedup/update behavior; CI/package contract fixes that affect operator-visible release and validation behavior.
- **Out of scope**: new MIA features; new portfolio strategies; prompt redesign; changes to the statistical objective of `p_memorized`; new external providers; live-trading behavior; consumer-side feature work in downstream projects.
- **Adjacent expectations**: this hardening work reuses the existing public surfaces in `core`, `mia`, `harness`, `portfolio`, and the current Kiro specs. It may tighten error behavior and artifact contracts, but it shall not introduce a second scoring path or a second manifest format.

## Requirements

### Requirement 1: Fail-fast run integrity
**Objective:** As an operator running the harness or CMMD pipeline, I want invalid or unusable runs to fail clearly instead of producing success-looking artifacts, so that I can trust a zero exit code and a completed run directory.

#### Acceptance Criteria
1. When the harness smoke gate yields an empty shortlist, the recall_guard run pipeline shall stop with a non-zero exit status instead of writing a success-looking ranking run.
2. If per-model evaluation encounters an unexpected internal error outside the explicitly supported calibration-failure cases, then the recall_guard run pipeline shall surface the run as failed instead of rewriting that model into an `uncalibrated` placeholder result.
3. When a CMMD orchestration run reads a harness summary that marks the signal model as `uncalibrated` or `weak-calibration`, the run_cmmd_backtest pipeline shall abort before any post-harness analytics or backtest stage begins.
4. If documented price-fetch failure conditions occur during CMMD orchestration, then the run_cmmd_backtest pipeline shall exit through its documented price-fetch failure path instead of emitting an uncaught traceback.
5. The recall_guard run pipeline shall reject invalid bootstrap-resample counts before model-evaluation work starts.

### Requirement 2: Shared parsing and provider-response correctness
**Objective:** As a maintainer or API consumer, I want the harness, smoke gate, and public scoring façade to interpret model outputs and provider responses consistently, so that accepted inputs, rejected inputs, and reported failures do not diverge across entry points.

#### Acceptance Criteria
1. When the harness accepts a response format for `Direction` and `Confidence`, the smoke gate shall apply the same acceptance contract for shortlist eligibility.
2. If a model response contains a non-integer direction value, then the recall_guard parser shall reject it as a parse failure instead of coercing it into `-1`, `0`, or `1`.
3. When a model expresses confidence as a percentage, the recall_guard parser shall interpret the percentage by its percentage meaning, including values at or below `1%`.
4. If the NVIDIA-compatible provider returns a malformed HTTP-200 success payload, then the LM client shall convert that payload into a typed runtime failure instead of leaking unrelated parser exceptions to callers.
5. If a provider response omits required token-logprob data, then the LM client shall reject the response instead of fabricating substitute logprob values.
6. While concurrent requests share one LM client instance with request pacing enabled, the client shall enforce the configured pacing contract across those requests.
7. If a reference-model call fails while the primary-model output was otherwise valid, then the harness and public façade shall degrade according to the documented optional-reference behavior instead of converting the row or score into an unrelated generic failure.

### Requirement 3: Reproducible inputs and strict manifest provenance
**Objective:** As an auditor or researcher, I want the manifest and loader layer to describe the exact inputs and row classifications used by a run, so that I can reproduce and verify a result after the fact.

#### Acceptance Criteria
1. When a run loads its eval set, calibration corpora, and cutoff registry, the recall_guard run pipeline shall bind the manifest provenance to those consumed bytes rather than to whatever bytes happen to be on disk later.
2. If an eval JSONL file declares `_cutoff_date` on the first non-empty line, then the loader shall recognize that header regardless of preceding blank lines.
3. When the post-harness scripts consume eval metadata dates, they shall apply the same accepted date normalization contract as the orchestrator and backtest pipeline.
4. If a manifest contains malformed `shortlist` or `backtest` values, then the manifest reader shall reject the manifest instead of silently coercing those values into different data.
5. The CMMD run manifest shall preserve the IS/OOS provenance required to audit how each historical run classified rows for the signal model.
6. The recall_guard run pipeline shall record the required input-path and input-hash provenance needed to identify the exact run inputs after the run completes.

### Requirement 4: CMMD and backtest output integrity
**Objective:** As a researcher reading the backtest outputs, I want the CMMD filter, signal counts, equity curves, and artifact-writing behavior to match the strategy that actually ran, so that the published numbers are internally consistent and reproducible.

#### Acceptance Criteria
1. When the CMMD filter removes the top memorization slice, the filter shall apply that slice deterministically even when many rows tie at the cutoff threshold.
2. If a row cannot become a tradable position because of missing or invalid date/ticker mapping, then the backtest pipeline shall exclude that row before CMMD-threshold computation, low-row warnings, and reported signal counts are finalized.
3. The backtest pipeline shall keep `equity_curves.csv`, `daily_returns.csv`, and `backtest_summary.csv` numerically consistent with the same executed fee treatment.
4. If a backtest-artifact write fails after earlier files have already been written into an existing run directory, then the writer shall preserve the directory's pre-write state instead of deleting or partially replacing prior artifacts.
5. While a backtest run uses vectorbt frequency settings for its own calculations, the backtest pipeline shall not leak that frequency choice into unrelated later vectorbt portfolios in the same process.
6. If a CMMD backtest run contains only IS rows or only OOS rows, then the orchestrator shall warn that CMMD has no meaningful cross-regime rows to remove before presenting the run as completed.
7. If CMMD receives a non-finite memorization score, then the filter shall reject or handle that row without silently collapsing the entire filtered stream.

### Requirement 5: Dataset and eval-builder trustworthiness
**Objective:** As a maintainer regenerating corpora or eval sets, I want valid source data to be accepted, invalid source data to fail clearly, and boundary-date rules to follow the approved spec, so that regenerated datasets are dependable.

#### Acceptance Criteria
1. When FMP article timestamps use valid ISO-8601 forms, the corpus builder shall accept them as valid publication timestamps.
2. If an article is published on the earliest model cutoff date, then the calibration builder shall classify it according to the approved strict pre-cutoff rule rather than treating the cutoff day as in-sample by default.
3. When the corpus builder ingests valid body-only articles, it shall not collapse distinct untitled articles into a single dedup bucket solely because their titles are empty.
4. If a newly visible OOS article is dated on the current maximum publication day, then incremental OOS refresh shall still have a way to ingest that article.
5. When the multiyear ETF eval builder requests a date window, the builder shall fetch data for that explicit window instead of silently relying on provider defaults.
6. If the multiyear ETF eval builder receives an empty or malformed upstream payload, then it shall fail clearly instead of writing a silent empty or partial eval file and exiting successfully.

### Requirement 6: Package and CI contract enforcement
**Objective:** As a maintainer publishing and validating the package, I want the CI and package metadata to enforce the documented validation and installation contract, so that operators and consumers receive the tooling and checks the project claims to provide.

#### Acceptance Criteria
1. When CI runs the lint path, the project shall enforce both style/lint checks and the required structural-architecture gate.
2. Where the package documentation or requirements promise installable development or pipeline extras, the built distribution shall expose matching installable extras rather than only local dependency groups.
3. The review-hardening work shall preserve the current lean runtime boundary: fixes for CI, docs, or development tooling shall not force plotting, notebook, or pipeline dependencies into the default runtime install.
