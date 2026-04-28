# Requirements

## What this spec is for

You're a quant researcher running NVIDIA-hosted LLMs against a financial prediction task and you want to pick the top three. The old harness gave you junk numbers: 1-row dev set, hardcoded `Loss < 0.5` threshold, broken labels, no confidence intervals. This spec rebuilds the harness so its numbers are trustworthy and the top-three pick is defensible.

## What's wrong with the old harness

- One-row dev set. Accuracy is either 0% or 100%, no resolution.
- The "OOS" data file replicates the same target across articles for the same ticker, so always-bearish scores ~80%.
- MIA scoring is a hardcoded threshold the paper explicitly argues against (Section 5.7 — threshold filters destroy signal).
- Only Loss and Min-K% are computed. The paper uses five features plus a calibrated classifier.
- Big models look "memorized" just because they're fluent; there's no per-model baseline to control for that.
- No CIs, no significance, no majority-class baseline. Parse failures count as wrong answers.
- Output is a flat CSV of point estimates with no uncertainty.

## What this spec changes

1. Smoke-test the candidate pool down to ~10 working models.
2. Compute the full MIA feature set (Loss, Min-K%, Min-K%++, zlib ratio, ref-model delta).
3. Train a per-model MCS classifier on labelled IS/OOS prompts. Use it to discount confidence on memorized-looking rows. No hardcoded threshold.
4. Use the same control corpus to standardize each model's MIA scores against its own baseline. Fixes the "fluent = memorized" bug.
5. Bootstrap 95% CIs on every accuracy and AUC. Always print the majority-class baseline.
6. Distinguish parse failures from wrong answers.
7. Replace the flat CSV with a structured run directory: `top3.md`, `summary.csv`, `records.jsonl`, `manifest.json`.
8. Accept any `(prompt, target_direction)` JSONL so the same harness works for news now and macro indicators later.

## What's out of scope

- Live trading or portfolio backtesting.
- Sharpe-based debiasing (the paper's CMMD method).
- Training or fine-tuning models.
- Macro-data ingestion. That's a separate spec; this one only guarantees the harness will accept its output.

## Boundary

- **In:** smoke-test shortlist, control-prompt calibration, the five MIA features, the MCS classifier, bootstrap CIs, parse-failure accounting, the structured run output, the JSONL input contract.
- **Out:** Sharpe / portfolio / CMMD; macro-data ingestion; alternative MIA methods beyond the named five; UI work.
- **Adjacent:** the IS/OOS calibration corpus is built inside this spec. A future macro-ingest spec will produce JSONL conforming to the input contract this spec defines.

## Requirements

The Acceptance Criteria use EARS format. Each requirement starts with a one-line plain-English summary so you don't have to translate from "the system shall."

### Requirement 1: smoke-test shortlist

Skip broken models before they pollute the run.

**Objective:** As a quant researcher, I want the harness to start from a pre-filtered shortlist of viable models, so that wasted evaluation time on broken or unreachable endpoints does not pollute the final ranking.

#### Acceptance Criteria
1. The harness shall accept a candidate pool of model IDs and emit a shortlist of at most 10 models that pass smoke-test gates.
2. When a candidate model does not return a parseable `Direction:` value within the configured per-call timeout for any of N smoke-test prompts, the harness shall exclude that model from the shortlist and record the failure reason.
3. When a candidate model returns responses without per-token log-probabilities, the harness shall exclude that model from the shortlist and record the failure reason.
4. The harness shall write the smoke-test outcome (pass / fail-reason) for every candidate to a persisted artifact so the shortlist is reproducible.
5. Where the user passes an explicit shortlist override, the harness shall use that list verbatim and skip the smoke-test selection step.

### Requirement 2: eval set contract

The harness eats any (prompt, target) JSONL. No domain assumptions.

**Objective:** As a quant researcher, I want the harness to consume any `(prompt, target)` JSONL, so that the same harness works unchanged for the current news data and for a future macroeconomic-indicators dataset.

#### Acceptance Criteria
1. The harness shall accept a JSONL evaluation file where every row contains at least a `prompt` field and a `target_direction` field with value in `{-1, 0, 1}`.
2. If an evaluation file contains fewer than 100 rows, the harness shall emit a warning identifying low statistical power and continue.
3. If an evaluation file's majority-class share exceeds 60%, the harness shall emit a warning identifying class imbalance and continue.
4. The harness shall not perform a train/dev split on the evaluation file; the entire file is the evaluation set.
5. The harness shall require evaluation prompts to be timestamped after every shortlisted model's training cutoff, and shall fail fast if the input file declares a `cutoff_date` field whose value precedes any model's known cutoff.

### Requirement 3: per-model control baseline

Standardize each model's MIA scores against its own logprob fingerprint, not a global threshold.

**Objective:** As a quant researcher, I want each model's MIA scores interpreted relative to that model's own baseline perplexity on out-of-sample prose, so that fluent large models are not mislabelled "memorizing" simply because they predict English well.

#### Acceptance Criteria
1. The harness shall maintain a fixed control-prompt corpus drawn from a time window after every shortlisted model's training cutoff.
2. Before scoring a model on the evaluation set, the harness shall compute that model's baseline distribution of every MIA feature on the control corpus.
3. When reporting an MIA feature for an evaluation prompt, the harness shall report both the raw value and the value standardised against that model's control distribution (z-score or percentile).
4. If a model's control corpus pass produces fewer than the configured minimum number of valid logprob-bearing responses, the harness shall mark that model's MIA features as `uncalibrated` and exclude the model from the final ranking.

### Requirement 4: the five MIA features

Compute everything the paper validated, not just two.

**Objective:** As a quant researcher, I want the five MIA features from the paper computed for every (model, prompt) pair, so that the calibrated memorization classifier has the inputs the paper validated.

#### Acceptance Criteria
1. The harness shall compute, per (model, prompt): average negative log-likelihood (Loss), Min-K% probability, Min-K%++, zlib compression ratio, and reference-model delta.
2. When the reference-model feature cannot be computed (no reference model configured or reference call fails), the harness shall record `null` for that feature and continue, rather than aborting the evaluation.
3. The harness shall emit a per-(model, prompt) record containing all five features alongside the model's directional prediction and raw confidence.

### Requirement 5: MCS classifier replaces fixed thresholds

A per-model logistic regression on labelled IS/OOS data outputs `p(memorized)` per prompt.

**Objective:** As a quant researcher, I want a calibrated `p(memorized | features)` per (model, prompt) instead of a global hardcoded threshold, so that memorization is judged on a continuous, model-aware probability rather than the brittle `Loss < 0.5` rule.

#### Acceptance Criteria
1. The harness shall train a per-model MCS classifier on a labelled corpus of in-sample (pre-cutoff) and out-of-sample (post-cutoff) prompts.
2. The harness shall report MCS-AUC for each shortlisted model on a held-out portion of the IS/OOS labelled corpus.
3. If a model's MCS-AUC on the held-out IS/OOS set is below the configured minimum (default 0.6), the harness shall mark that model's calibration as `weak` and surface the warning in the final report.
4. The harness shall apply the per-prompt MCS probability as a continuous penalty to the model's raw confidence, with no fixed threshold step.
5. The harness shall not retain the legacy hardcoded Loss / Min-K% threshold logic in the active scoring path.

### Requirement 6: bootstrap CIs

Every accuracy carries a 95% interval. Majority-class is always reported.

**Objective:** As a quant researcher, I want every reported metric to carry a confidence interval and a baseline reference, so that I can tell signal from noise on small evaluation sets.

#### Acceptance Criteria
1. For every shortlisted model, the harness shall report Raw Accuracy and MemGuard Accuracy each with a bootstrap 95% confidence interval computed from at least 1000 resamples on the evaluation set.
2. The harness shall report a majority-class baseline accuracy with the same bootstrap CI on the same evaluation set.
3. For every shortlisted model, the harness shall report MCS-AUC with a bootstrap 95% CI.
4. The harness shall flag any model whose Raw-Accuracy lower CI bound does not exceed the majority-class upper CI bound as `not-better-than-baseline`.
5. The harness shall use a fixed random seed for all bootstrap resampling and persist that seed in the report.

### Requirement 7: parse-failure accounting

A model that doesn't emit `Direction: X` is unreliable, not just wrong.

**Objective:** As a quant researcher, I want parse failures separated from wrong answers, so that "0% accuracy / calibrated" rows do not silently include broken output formats.

#### Acceptance Criteria
1. When a model response cannot be parsed into a directional prediction, the harness shall record that response as a parse failure, not as `direction = 0`.
2. The harness shall report a per-model parse-success rate on the evaluation set.
3. The harness shall exclude parse failures from accuracy calculation and explicitly report the excluded count.
4. If a model's parse-success rate falls below the configured minimum (default 80%), the harness shall mark that model as `parse-unreliable` and surface the warning in the final report.

### Requirement 8: top-3 ranking

One composite score, one Markdown file, one definitive answer.

**Objective:** As a quant researcher, I want a single, defensible top-3 ranking based on a documented composite score, so that I can pick downstream models without re-deriving the scoring rule each time.

#### Acceptance Criteria
1. The harness shall compute a composite rank score per model from at minimum: MemGuard Accuracy lower CI bound, MCS-AUC, and parse-success rate.
2. The harness shall write a `top3.md` artifact listing the three highest-scoring models, each with its composite score, the metrics that produced it, and any active warnings (`weak-calibration`, `parse-unreliable`, `not-better-than-baseline`, `uncalibrated`).
3. If fewer than three models survive all gates (parse-success ≥ 80%, MCS-AUC ≥ 0.6, accuracy lower CI > majority-class upper CI), the harness shall write a `top3.md` listing only the surviving models and an explicit note explaining why the list is short.
4. The harness shall persist the composite-score formula and weights in the report so a reader can reproduce the ranking from the per-model metrics.

### Requirement 9: structured run output

Replace the flat CSV with a directory of artifacts a downstream tool can consume.

**Objective:** As a quant researcher, I want a readable terminal report and a structured CSV/JSON artifact replacing the current flat CSV, so that I can scan results at a glance and downstream tools can consume them programmatically.

#### Acceptance Criteria
1. When evaluation completes, the harness shall print to the terminal one row per shortlisted model showing model ID, Raw Accuracy with CI, MemGuard Accuracy with CI, MCS-AUC with CI, parse-success rate, and active warnings.
2. The harness shall print the majority-class baseline row alongside the model rows.
3. The harness shall write a structured artifact (CSV and/or JSON) containing every per-(model, prompt) record produced during the run, including all MIA features, the directional prediction, raw and penalized confidence, and parse-success flag.
4. The harness shall print the location of the `top3.md`, the structured artifact, and the smoke-test artifact at the end of the run.
5. The harness shall not produce the legacy `models_report.csv` schema in the active output path.

### Requirement 10: reproducibility

Every run is replayable from its manifest at temperature 0.

**Objective:** As a quant researcher, I want a single run to be reproducible from its persisted artifacts, so that re-running with the same inputs produces the same ranking.

#### Acceptance Criteria
1. The harness shall write to a per-run directory a manifest containing: evaluation file path and hash, control-corpus path and hash, shortlist, random seed, MCS hyperparameters, composite-score weights, and the harness version.
2. When invoked with `--from-manifest <path>`, the harness shall reproduce the prior run bit-for-bit assuming model endpoints are deterministic at temperature 0.
3. The harness shall request temperature 0 (or the lowest available) for every model call used in scoring, and shall record any model that does not honour that setting.

### Requirement 11: calibration corpora from FMP

Build IS/OOS calibration data from real, dated news articles. No fabricated content.

**Objective:** As a quant researcher, I want both calibration corpora built once and updated incrementally from FMP's news endpoints, so that the IS/OOS labels rest on real, dated content I can cite, not on text fabricated by an AI subagent.

#### Acceptance Criteria
1. The harness shall ship a build mode that produces `data/calibration/is_memorized.jsonl` (label=1) and `data/calibration/oos_control.jsonl` (label=0) from articles fetched via the FMP news endpoints (`fmp-articles`, `news/general-latest`, and optionally `news/stock-latest`).
2. The build mode shall filter articles strictly by publication date: IS rows come from before the earliest training cutoff in the cutoff registry; OOS rows come from after the latest training cutoff in the cutoff registry.
3. The build mode shall deduplicate articles by URL and by title hash before writing.
4. If an article lacks a non-empty body or a parseable publication date, the build mode shall skip the article and log a WARNING.
5. The harness shall ship an update mode that appends new post-cutoff articles to `oos_control.jsonl`, deduplicating against existing rows, and shall never modify `is_memorized.jsonl`.

### Requirement 12: public API and notebook walkthrough

Every harness step is callable from a Jupyter notebook with paper-ready figures and rendered LaTeX equations.

**Objective:** As a quant researcher writing up the methodology for a paper, I want every harness step callable from a Jupyter notebook with paper-ready visualisations, so that I can author the results section as a reproducible notebook and lift figures directly into a two-column manuscript.

#### Acceptance Criteria
1. Each `src/{core,mia,harness}/__init__.py` shall re-export the consumer-facing public API (loader, smoke_test, build_baseline, MCS train, evaluate_model, composite_score, bootstrap_ci, runner.run, plotting helpers) so that `from src.harness import evaluate_model` succeeds without internal-path imports.
2. The harness shall ship a stepwise notebook at `notebooks/qualification.ipynb` that walks through every stage of the qualification pipeline (load → smoke shortlist → control baselines → MCS train → evaluate → rank → top-3) using only the public API.
3. Each notebook step shall produce at least one figure that visualises the underlying statistical process: MIA feature distributions (IS vs OOS), MCS calibration curve, bootstrap CI bars on Raw + MemGuard accuracy and on MCS-AUC, and the composite ranking.
4. Figures shall be paper-ready: vector format (PDF or SVG), default width fitted for a single column of a two-column manuscript (≈3.5 inches), font sizes large enough to remain legible at native size, colorblind-safe palette, with markers or hatching that survive black-and-white printing.
5. The harness shall expose a plotting helper module that consumes `ModelEvalResult`, `ControlBaseline`, `MCSCalibrator`, and `CompositeScore` dataclasses to produce the figures, so the notebook contains orchestration logic only and not plotting boilerplate.
6. Each statistical step in the notebook shall display its underlying mathematical formula via Markdown/LaTeX rendering immediately before the cell that computes it, with consistent notation across the notebook, so that a reader of the paper can verify the implementation matches the documented statistic. The formulas covered shall include at minimum: Loss, Min-K%, Min-K%++ (per-position z-score), zlib ratio, reference-model delta, control-baseline standardisation, MCS logistic-regression probability, MemGuard penalty `c_raw × (1 - p_memorized)`, bootstrap percentile CI, ROC-AUC, majority-class baseline accuracy, and the composite ranking score.

## Default values

The acceptance criteria above reference these defaults. They're tunable; any change should re-validate the run.

| Setting | Default | Where it's referenced |
|---|---|---|
| Shortlist size cap | 10 | Req 1 |
| Smoke-test prompts per model (N) | 5 | Req 1 |
| Per-call timeout | 45 s (was 15 s; reasoning models need more) | Req 1 |
| Min eval rows before warning | 100 | Req 2 |
| Class-imbalance warning threshold | > 60% majority share | Req 2 |
| Min control-corpus valid responses | 50 | Req 3 |
| Reference model | `meta/llama-3.2-1b-instruct` (small, NVIDIA-hosted, well-known training data) | Req 4 |
| Min MCS-AUC | 0.6 | Req 5, 8 |
| Bootstrap resamples | 1000 | Req 6 |
| Min parse-success rate | 80% | Req 7, 8 |
| Composite score | `MemGuard_Acc_lowerCI × MCS_AUC × parse_success_rate` | Req 8 |
| FMP endpoints (default) | `fmp-articles` + `news/general-latest`; opt-in `news/stock-latest` | Req 11 |
| Per-corpus target rows | 100 | Req 11 |
| Paper figure rcParams | `figsize=(3.5, 2.5)`, `font.size=8`, `savefig.dpi=300`, `format=pdf`; Wong palette | Req 12 |
| Max parallel API calls per model (`--max-workers`) | 8 | runtime perf, post-spec addition |
