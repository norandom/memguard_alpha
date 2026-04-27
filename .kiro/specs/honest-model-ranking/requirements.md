# Requirements Document

## Project Description (Input)

**Who has the problem.** The quant researcher (sole user) running MemGuard-Alpha against NVIDIA-hosted foundation models. The end goal is to pick the **top 3 models** to carry forward into a downstream financial-reasoning pipeline that will eventually consume macroeconomic indicators (not news).

**Current situation.** The evaluation harness in `main.py` produces distorted, untrustworthy numbers:
- Devset is 1 example (default `lookahead_bench_sample.jsonl`, 5 rows × 80/20 split) or 5 examples (OOS file). Accuracy can only land on a coarse grid; "100%" is a 1-of-1 hit.
- `lookahead_bench_2026_oos.jsonl` replicates the same `target_direction` across 5 news items per ticker, so an "always bearish" classifier scores ~80%.
- MIA scoring is a hardcoded threshold (`Loss < 0.5`, `Min-K% > -0.5`) that the paper explicitly argues against (Section 5.7 — threshold-based filters are fragile and destroy signal).
- The pipeline implements only Loss and Min-K%. The paper's full feature set — Min-K%++, zlib ratio, reference-model baseline, Memorization Contamination Score (MCS) calibrated on IS/OOS labels — is missing.
- No control prompts per model, so the "Base Perplexity Paradox" (`Qualified_Models.md`) is unaddressed: large models look "memorized" simply because they are fluent.
- No bootstrap confidence intervals, no significance testing, no majority-class baseline. Parse failures are silently scored as wrong (`direction = 0`), inflating "calibrated 0%" rows.
- CLI output is a flat CSV of point estimates with no uncertainty, no per-model calibration trace, and no ranking — the user cannot defensibly pick top-3 from it.

**What should change.** Rebuild the evaluation harness so it produces statistically defensible, reproducible model rankings:
1. Reduce the candidate pool up front to ~10 models that pass a smoke test (parses output cleanly, returns logprobs, completes within timeout).
2. Implement the paper's full MIA feature set (Loss, Min-K%, Min-K%++, zlib ratio, reference-model delta) plus per-model MCS calibration trained on a labelled IS-vs-OOS prompt set.
3. Use **control prompts** (same template, OOS time window) to establish each model's baseline perplexity, so MIA scores are interpreted *relative to that model* — not against a global hardcoded threshold.
4. Replace the 80/20 split with a fixed evaluation set of sufficient size (target ≥ 100 rows with balanced labels) and report bootstrap 95% CIs on accuracy, MCS-AUC, and the Raw-vs-MemGuard delta.
5. Always report a majority-class baseline alongside model results, and distinguish parse failures from genuinely-wrong predictions.
6. Replace the current CSV with a structured CLI report: per-model row showing Raw Acc ± CI, MemGuard Acc ± CI, Parse-success rate, MCS-AUC vs control prompts, and a final composite rank score. Output a `top3.md` summary the user can act on.
7. Decouple input source from the harness: the harness must accept any `(prompt, target)` JSONL so it works for both the current news data and a future macroeconomic-indicators dataset, without code changes to the scoring path.

**Non-goals.**
- Live trading, portfolio backtesting, or Sharpe-based evaluation (the paper's CMMD is signal-flow specific; this spec stops at directional-accuracy + MCS calibration).
- Building or fine-tuning models.
- Macro data ingestion itself (will be a separate spec; this spec only guarantees the harness will accept its output).

## Boundary Context

- **In scope**: Model shortlisting via smoke test; per-model control-prompt calibration; full MIA feature set (loss, Min-K%, Min-K%++, zlib ratio, reference-model delta); MCS classifier trained on IS-vs-OOS labelled prompts; bootstrap-CI accuracy and MCS-AUC reporting; majority-class baseline; parse-failure accounting; structured CLI report and `top3.md` artifact; input-source-agnostic JSONL interface.
- **Out of scope**: Sharpe / portfolio / CMMD trading-signal debiasing; macro data ingestion (separate spec); model fine-tuning; live trading; UI/dashboard work; alternative MIA methods beyond the five named features.
- **Adjacent expectations**: A separate macro-ingest spec will produce JSONL conforming to the harness input contract defined here. The IS-vs-OOS labelled prompt corpus used to train MCS is built within this spec.

## Requirements

### Requirement 1: Model Shortlist via Smoke Test
**Objective:** As a quant researcher, I want the harness to start from a pre-filtered shortlist of viable models, so that wasted evaluation time on broken or unreachable endpoints does not pollute the final ranking.

#### Acceptance Criteria
1. The harness shall accept a candidate pool of model IDs and emit a shortlist of at most 10 models that pass smoke-test gates.
2. When a candidate model does not return a parseable `Direction:` value within the configured per-call timeout for any of N smoke-test prompts, the harness shall exclude that model from the shortlist and record the failure reason.
3. When a candidate model returns responses without per-token log-probabilities, the harness shall exclude that model from the shortlist and record the failure reason.
4. The harness shall write the smoke-test outcome (pass / fail-reason) for every candidate to a persisted artifact so the shortlist is reproducible.
5. Where the user passes an explicit shortlist override, the harness shall use that list verbatim and skip the smoke-test selection step.

### Requirement 2: Evaluation Dataset Contract
**Objective:** As a quant researcher, I want the harness to consume any `(prompt, target)` JSONL, so that the same harness works unchanged for the current news data and for a future macroeconomic-indicators dataset.

#### Acceptance Criteria
1. The harness shall accept a JSONL evaluation file where every row contains at least a `prompt` field and a `target_direction` field with value in `{-1, 0, 1}`.
2. If an evaluation file contains fewer than 100 rows, the harness shall emit a warning identifying low statistical power and continue.
3. If an evaluation file's majority-class share exceeds 60%, the harness shall emit a warning identifying class imbalance and continue.
4. The harness shall not perform a train/dev split on the evaluation file; the entire file is the evaluation set.
5. The harness shall require evaluation prompts to be timestamped after every shortlisted model's training cutoff, and shall fail fast if the input file declares a `cutoff_date` field whose value precedes any model's known cutoff.

### Requirement 3: Per-Model Control-Prompt Baseline
**Objective:** As a quant researcher, I want each model's MIA scores interpreted relative to that model's own baseline perplexity on out-of-sample prose, so that fluent large models are not mislabelled "memorizing" simply because they predict English well.

#### Acceptance Criteria
1. The harness shall maintain a fixed control-prompt corpus drawn from a time window after every shortlisted model's training cutoff.
2. Before scoring a model on the evaluation set, the harness shall compute that model's baseline distribution of every MIA feature on the control corpus.
3. When reporting an MIA feature for an evaluation prompt, the harness shall report both the raw value and the value standardised against that model's control distribution (z-score or percentile).
4. If a model's control corpus pass produces fewer than the configured minimum number of valid logprob-bearing responses, the harness shall mark that model's MIA features as `uncalibrated` and exclude the model from the final ranking.

### Requirement 4: Full MIA Feature Set
**Objective:** As a quant researcher, I want the five MIA features from the paper computed for every (model, prompt) pair, so that the calibrated memorization classifier has the inputs the paper validated.

#### Acceptance Criteria
1. The harness shall compute, per (model, prompt): average negative log-likelihood (Loss), Min-K% probability, Min-K%++, zlib compression ratio, and reference-model delta.
2. When the reference-model feature cannot be computed (no reference model configured or reference call fails), the harness shall record `null` for that feature and continue, rather than aborting the evaluation.
3. The harness shall emit a per-(model, prompt) record containing all five features alongside the model's directional prediction and raw confidence.

### Requirement 5: MCS Calibration Replacing Fixed Thresholds
**Objective:** As a quant researcher, I want a calibrated `p(memorized | features)` per (model, prompt) instead of a global hardcoded threshold, so that memorization is judged on a continuous, model-aware probability rather than the brittle `Loss < 0.5` rule.

#### Acceptance Criteria
1. The harness shall train a per-model MCS classifier on a labelled corpus of in-sample (pre-cutoff) and out-of-sample (post-cutoff) prompts.
2. The harness shall report MCS-AUC for each shortlisted model on a held-out portion of the IS/OOS labelled corpus.
3. If a model's MCS-AUC on the held-out IS/OOS set is below the configured minimum (default 0.6), the harness shall mark that model's calibration as `weak` and surface the warning in the final report.
4. The harness shall apply the per-prompt MCS probability as a continuous penalty to the model's raw confidence, with no fixed threshold step.
5. The harness shall not retain the legacy hardcoded Loss / Min-K% threshold logic in the active scoring path.

### Requirement 6: Statistical Reporting with Bootstrap CIs
**Objective:** As a quant researcher, I want every reported metric to carry a confidence interval and a baseline reference, so that I can tell signal from noise on small evaluation sets.

#### Acceptance Criteria
1. For every shortlisted model, the harness shall report Raw Accuracy and MemGuard Accuracy each with a bootstrap 95% confidence interval computed from at least 1000 resamples on the evaluation set.
2. The harness shall report a majority-class baseline accuracy with the same bootstrap CI on the same evaluation set.
3. For every shortlisted model, the harness shall report MCS-AUC with a bootstrap 95% CI.
4. The harness shall flag any model whose Raw-Accuracy lower CI bound does not exceed the majority-class upper CI bound as `not-better-than-baseline`.
5. The harness shall use a fixed random seed for all bootstrap resampling and persist that seed in the report.

### Requirement 7: Parse-Failure Accounting
**Objective:** As a quant researcher, I want parse failures separated from wrong answers, so that "0% accuracy / calibrated" rows do not silently include broken output formats.

#### Acceptance Criteria
1. When a model response cannot be parsed into a directional prediction, the harness shall record that response as a parse failure, not as `direction = 0`.
2. The harness shall report a per-model parse-success rate on the evaluation set.
3. The harness shall exclude parse failures from accuracy calculation and explicitly report the excluded count.
4. If a model's parse-success rate falls below the configured minimum (default 80%), the harness shall mark that model as `parse-unreliable` and surface the warning in the final report.

### Requirement 8: Top-3 Ranking and `top3.md` Artifact
**Objective:** As a quant researcher, I want a single, defensible top-3 ranking based on a documented composite score, so that I can pick downstream models without re-deriving the scoring rule each time.

#### Acceptance Criteria
1. The harness shall compute a composite rank score per model from at minimum: MemGuard Accuracy lower CI bound, MCS-AUC, and parse-success rate.
2. The harness shall write a `top3.md` artifact listing the three highest-scoring models, each with its composite score, the metrics that produced it, and any active warnings (`weak-calibration`, `parse-unreliable`, `not-better-than-baseline`, `uncalibrated`).
3. If fewer than three models survive all gates (parse-success ≥ 80%, MCS-AUC ≥ 0.6, accuracy lower CI > majority-class upper CI), the harness shall write a `top3.md` listing only the surviving models and an explicit note explaining why the list is short.
4. The harness shall persist the composite-score formula and weights in the report so a reader can reproduce the ranking from the per-model metrics.

### Requirement 9: Structured CLI Report
**Objective:** As a quant researcher, I want a readable terminal report and a structured CSV/JSON artifact replacing the current flat CSV, so that I can scan results at a glance and downstream tools can consume them programmatically.

#### Acceptance Criteria
1. When evaluation completes, the harness shall print to the terminal one row per shortlisted model showing model ID, Raw Accuracy with CI, MemGuard Accuracy with CI, MCS-AUC with CI, parse-success rate, and active warnings.
2. The harness shall print the majority-class baseline row alongside the model rows.
3. The harness shall write a structured artifact (CSV and/or JSON) containing every per-(model, prompt) record produced during the run, including all MIA features, the directional prediction, raw and penalized confidence, and parse-success flag.
4. The harness shall print the location of the `top3.md`, the structured artifact, and the smoke-test artifact at the end of the run.
5. The harness shall not produce the legacy `models_report.csv` schema in the active output path.

### Requirement 10: Reproducibility and Run Artifacts
**Objective:** As a quant researcher, I want a single run to be reproducible from its persisted artifacts, so that re-running with the same inputs produces the same ranking.

#### Acceptance Criteria
1. The harness shall write to a per-run directory a manifest containing: evaluation file path and hash, control-corpus path and hash, shortlist, random seed, MCS hyperparameters, composite-score weights, and the harness version.
2. When invoked with `--from-manifest <path>`, the harness shall reproduce the prior run bit-for-bit assuming model endpoints are deterministic at temperature 0.
3. The harness shall request temperature 0 (or the lowest available) for every model call used in scoring, and shall record any model that does not honour that setting.

### Requirement 11: Calibration Corpora via FMP API
**Objective:** As a quant researcher, I want both calibration corpora built once and updated incrementally from FMP's news endpoints, so that the IS/OOS labels rest on real, dated content I can cite, not on text fabricated by an AI subagent.

#### Acceptance Criteria
1. The harness shall ship a build mode that produces `data/calibration/is_memorized.jsonl` (label=1) and `data/calibration/oos_control.jsonl` (label=0) from articles fetched via the FMP news endpoints (`fmp-articles`, `news/general-latest`, and optionally `news/stock-latest`).
2. The build mode shall filter articles strictly by publication date: IS rows come from before the earliest training cutoff in the cutoff registry; OOS rows come from after the latest training cutoff in the cutoff registry.
3. The build mode shall deduplicate articles by URL and by title hash before writing.
4. If an article lacks a non-empty body or a parseable publication date, the build mode shall skip the article and log a WARNING.
5. The harness shall ship an update mode that appends new post-cutoff articles to `oos_control.jsonl`, deduplicating against existing rows, and shall never modify `is_memorized.jsonl`.

### Requirement 12: Public API and Notebook Walkthrough
**Objective:** As a quant researcher writing up the methodology for a paper, I want every harness step callable from a Jupyter notebook with paper-ready visualisations, so that I can author the results section as a reproducible notebook and lift figures directly into a two-column manuscript.

#### Acceptance Criteria
1. Each `src/{core,mia,harness}/__init__.py` shall re-export the consumer-facing public API (loader, smoke_test, build_baseline, MCS train, evaluate_model, composite_score, bootstrap_ci, runner.run, plotting helpers) so that `from src.harness import evaluate_model` succeeds without internal-path imports.
2. The harness shall ship a stepwise notebook at `notebooks/qualification.ipynb` that walks through every stage of the qualification pipeline (load → smoke shortlist → control baselines → MCS train → evaluate → rank → top-3) using only the public API.
3. Each notebook step shall produce at least one figure that visualises the underlying statistical process: MIA feature distributions (IS vs OOS), MCS calibration curve, bootstrap CI bars on Raw + MemGuard accuracy and on MCS-AUC, and the composite ranking.
4. Figures shall be paper-ready: vector format (PDF or SVG), default width fitted for a single column of a two-column manuscript (≈3.5 inches), font sizes large enough to remain legible at native size, colorblind-safe palette, with markers or hatching that survive black-and-white printing.
5. The harness shall expose a plotting helper module that consumes `ModelEvalResult`, `ControlBaseline`, `MCSCalibrator`, and `CompositeScore` dataclasses to produce the figures, so the notebook contains orchestration logic only and not plotting boilerplate.
6. Each statistical step in the notebook shall display its underlying mathematical formula via Markdown/LaTeX rendering immediately before the cell that computes it, with consistent notation across the notebook, so that a reader of the paper can verify the implementation matches the documented statistic. The formulas covered shall include at minimum: Loss, Min-K%, Min-K%++ (per-position z-score), zlib ratio, reference-model delta, control-baseline standardisation, MCS logistic-regression probability, MemGuard penalty `c_raw × (1 - p_memorized)`, bootstrap percentile CI, ROC-AUC, majority-class baseline accuracy, and the composite ranking score.

## Open Defaults Flagged for Review

The following defaults were chosen to make requirements concrete and testable. Override before approval if needed; they are referenced by acceptance criteria above.

- **Shortlist size cap (Req 1)**: 10.
- **Smoke-test prompt count `N` (Req 1)**: 5.
- **Per-call timeout (Req 1)**: 15 s (matches current `main.py`).
- **Minimum evaluation rows (Req 2)**: 100.
- **Majority-class warning threshold (Req 2)**: > 60%.
- **Minimum control-corpus valid responses (Req 3)**: 50.
- **Reference model for MIA delta (Req 4)**: `meta/llama-3.2-1b-instruct` (small, NVIDIA-hosted, well-known training data — strong "should-have-memorized" baseline).
- **Minimum MCS-AUC (Req 5, 8)**: 0.6.
- **Bootstrap resamples (Req 6)**: 1000.
- **Minimum parse-success (Req 7, 8)**: 80%.
- **Composite score (Req 8)**: `MemGuard_Acc_lowerCI × MCS_AUC × parse_success_rate` (multiplicative; any zero gate kills the score).
- **FMP endpoints (Req 11)**: `fmp-articles` + `news/general-latest` by default (broad financial commentary, low ticker-topic bias); `news/stock-latest` opt-in via `--include-stock-news` for diversity.
- **Per-corpus target row count (Req 11)**: 100.
- **Paper figure rcParams (Req 12)**: `figsize=(3.5, 2.5)`, `font.size=8`, `savefig.dpi=300`, `savefig.format="pdf"`; colorblind-safe palette `["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#F0E442"]`.
