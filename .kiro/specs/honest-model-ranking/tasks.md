# Implementation Plan

## Tasks

- [ ] 1. Foundation: dependencies, calibration corpora, model registry, test layout

- [x] 1.1 Declare project dependencies and test layout
  - Add `numpy`, `scikit-learn`, `rich`, `pyyaml` to `pyproject.toml` `[project] dependencies`; declare existing `requests`, `python-dotenv`, `pytest`, `pytest-mock` explicitly.
  - Run `uv sync` so the lockfile reflects the new deps and a fresh checkout can install without manual steps.
  - Create empty `tests/core/`, `tests/mia/`, `tests/harness/` directories with `__init__.py` files so the new tests can land into a stable layout.
  - Observable: `uv sync` succeeds on a clean clone; `pytest` discovers zero new failures (no new tests yet).
  - _Requirements: 4.1, 5.1, 6.1, 9.1, 10.1_

- [x] 1.2 (P) Build the FMP calibration corpus builder
  - Create `src/dataset/__init__.py` and `src/dataset/fmp_corpora.py` exposing `ArticleRecord`, `build_calibration`, `update_oos`, and a small `fetch_articles(endpoint, api_key, from_date, to_date, page, limit)` helper.
  - `build_calibration(out_dir, cutoffs, target_per_corpus=100, api_key=None, endpoints=DEFAULT_ENDPOINTS)` paginates the FMP news endpoints (`fmp-articles`, `news/general-latest` by default; `news/stock-latest` opt-in via parameter), filters articles by publication date against `is_window = (epoch, earliest_cutoff)` and `oos_window = (latest_cutoff, today)` derived from the cutoff registry, deduplicates by URL and by sha256(title), and writes `is_memorized.jsonl` (label=1) + `oos_control.jsonl` (label=0) under `data/calibration/`.
  - `update_oos(out_dir, api_key=None, endpoints=...)` reads the existing `oos_control.jsonl`, fetches articles after its max `published_at`, dedups against existing rows, and appends. It must never modify `is_memorized.jsonl`.
  - Articles missing a non-empty body or a parseable `publishedDate` are skipped with a single WARNING per skip.
  - Provide a `__main__` so `python -m src.dataset.fmp_corpora build` and `... update [--since YYYY-MM-DD]` work as CLIs.
  - Add `tests/dataset/__init__.py` and `tests/dataset/test_fmp_corpora.py` with `pytest-mock` patching `requests.get` to cover: date-window filter buckets correctly, URL+title dedup removes both kinds of duplicates, missing-body articles are rejected with a WARNING, `update_oos` only appends rows newer than the existing max date and never touches `is_memorized.jsonl`, and `build_calibration` writes both files atomically.
  - Observable: `pytest tests/dataset/test_fmp_corpora.py` passes (≥ 6 tests); `python -m src.dataset.fmp_corpora --help` prints both subcommands; no real HTTP calls in the test run.
  - _Requirements: 11.1, 11.2, 11.3, 11.4, 11.5_
  - _Boundary: dataset.fmp_corpora_
  - _Depends: 1.4_

- [x] 1.3 Run build_calibration to produce both calibration JSONL files
  - With `FMP_API_KEY` set in `.env` and `data/cutoffs.yaml` present, run `python -m src.dataset.fmp_corpora build` to produce `data/calibration/is_memorized.jsonl` and `data/calibration/oos_control.jsonl`.
  - Confirm both files exist with row count ≥ `target_per_corpus` (default 100), each row well-formed against the corpus schema (`prompt`, `label`, `metadata.published_at`, `metadata.source`, `metadata.url`), label values strictly 1 and 0 respectively, and no duplicate URLs or title hashes within a file.
  - Spot-check 5 random rows from each corpus and confirm dates land cleanly on the expected side of the cutoff boundary.
  - Observable: both JSONL files exist; `jq -c '.label' data/calibration/is_memorized.jsonl | sort | uniq -c` shows only `1` ≥ 100; same check on the OOS file shows only `0` ≥ 100; published dates respect the IS / OOS windows.
  - _Requirements: 3.1, 5.1, 5.2, 11.1_
  - _Boundary: data/calibration_
  - _Depends: 1.2, 1.4_

- [x] 1.4 (P) Author the per-model training-cutoff registry
  - Create `data/cutoffs.yaml` mapping each candidate model ID to its training cutoff (ISO date) under a top-level `models:` key.
  - Include the initial candidate pool the smoke test will exercise: at minimum the models referenced in `Qualified_Models.md` plus the planned reference model `meta/llama-3.2-1b-instruct`.
  - Source dates from NVIDIA model cards / vendor docs; if a date is unknown, omit the model so the runner fails fast rather than silently mis-evaluating it.
  - Observable: `python -c "import yaml; print(len(yaml.safe_load(open('data/cutoffs.yaml'))['models']))"` returns ≥ 10.
  - _Requirements: 2.5, 3.1_
  - _Boundary: data/cutoffs.yaml_

- [x] 1.5 Stratify the IS/OOS sampling in build_calibration
  - The 1.3 run produced 100/100 rows but every IS row landed on a single day (2023-12-30, one day before earliest_cutoff=2023-12-31) and every OOS row on the most recent 3 days. The reviewer flagged this as a methodology weakness: edge-of-cutoff IS articles produce a weaker memorization signal than mid-history articles, and the MCS classifier in task 3.3 may not separate IS from OOS reliably enough to clear the design's `min_auc=0.6` gate.
  - Fix `src/dataset/fmp_corpora.py` so the IS sampling is chronologically stratified. Two viable approaches; pick whichever is simpler given FMP's actual API shape: (a) split `is_window = (epoch, earliest_cutoff)` into K equal sub-windows (default K=5 → ~3-year buckets across 2010-2024) and request `target_per_corpus // K` rows per sub-window; or (b) add a `sort_by` parameter to `fetch_articles`, default `"date_asc"` for IS and `"date_desc"` for OOS, so IS pulls from deep history rather than the cutoff edge.
  - Apply the same logic to OOS only if FMP's recent-history coverage is too dense at the latest pages (otherwise leave OOS as-is, since recent OOS articles are equally "unseen" by the model).
  - Update `tests/dataset/test_fmp_corpora.py` to assert the stratification: synthetic 5-year window with 100 articles per year → IS sample contains rows from at least K-1 distinct years.
  - After the code change passes its tests, delete the existing `data/calibration/*.jsonl` and re-run `python -m src.dataset.fmp_corpora build`. Re-verify the same observables as task 1.3 plus a new check: `python -c "import json; from datetime import date; ds=[date.fromisoformat(json.loads(l)['metadata']['published_at']).year for l in open('data/calibration/is_memorized.jsonl')]; from collections import Counter; print(Counter(ds))"` — should show IS rows spread across ≥ 3 distinct years.
  - Observable: `pytest tests/dataset/test_fmp_corpora.py` passes the new stratification test; the regenerated `is_memorized.jsonl` shows a year distribution covering ≥ 3 years; OOS distribution unchanged (still dense in recent months).
  - _Requirements: 11.2_
  - _Boundary: dataset.fmp_corpora_
  - _Depends: 1.2, 1.3_

- [ ] 2. Core layer: HTTP client, loader, bootstrap, manifest

- [x] 2.1 (P) Extend the NVIDIA LM client with temperature and reference-model support
  - Move the existing client to `src/core/nvidia_lm.py` and add a `temperature` parameter (default 0.0) plus a frozen `CompletionResult` dataclass exposing `content`, `logprobs` (list of `TokenLogprob` with `top_logprobs`), and `raw_temperature_observed`.
  - Continue to send `logprobs=true, top_logprobs=20` and a 15 s timeout; on timeout raise `TimeoutError`, on missing `top_logprobs` in the response raise a clear error.
  - Move and extend `tests/core/test_nvidia_lm.py` to assert the request body contains `temperature: 0.0` and `top_logprobs: 20`, and that the response is parsed into `CompletionResult`.
  - Observable: `pytest tests/core/test_nvidia_lm.py` passes including the new temperature assertion.
  - _Requirements: 4.1, 4.2, 10.3_
  - _Boundary: core.nvidia_lm_

- [x] 2.2 (P) Implement the generic JSONL loader with cutoff guard
  - Create `src/core/loader.py` exporting `EvalRow`, `EvalSet`, `load_eval_set(path)`, `load_cutoffs(path)`, and `assert_cutoff_safe(eval_set, models, cutoffs)`.
  - `load_eval_set` validates every row has `prompt: str` and `target_direction: int in {-1,0,1}`; emits `logging` warnings (not exceptions) when N < 100 or majority-class share > 60%.
  - `assert_cutoff_safe` raises `CutoffViolation` if any model's cutoff post-dates `eval_set.cutoff_date` or if a shortlisted model is missing from the cutoff registry.
  - Add `tests/core/test_loader.py` covering: row schema validation, small-N warning, imbalance warning, cutoff-safe rejection, and the no-split contract (return value is a single list).
  - Observable: `pytest tests/core/test_loader.py` passes; running on a 30-row 80/20 fixture logs both warnings and returns 30 rows (not 6 / 24).
  - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5_
  - _Boundary: core.loader_

- [x] 2.3 (P) Implement the bootstrap CI helper
  - Create `src/core/bootstrap.py` exporting `bootstrap_ci(samples, statistic, n_resamples=1000, confidence=0.95, seed=0)` returning `(point, lo, hi)`.
  - Use `numpy.random.default_rng(seed)`; resample with replacement; statistic is computed on each resample; CI bounds via percentile.
  - Handle degenerate cases: if `len(samples) == 1` return `(point, point, point)`; if statistic raises on a resample (e.g., AUC with one class), drop that resample with a warning rather than crashing.
  - Add `tests/core/test_bootstrap.py` asserting determinism for a fixed seed and that for `samples = [0]*50 + [1]*50` the bootstrap mean CI brackets 0.5.
  - Observable: `pytest tests/core/test_bootstrap.py` passes; same seed produces identical `(point, lo, hi)` across two runs.
  - _Requirements: 6.1, 6.3, 6.5_
  - _Boundary: core.bootstrap_

- [x] 2.4 (P) Implement the run manifest
  - Create `src/core/manifest.py` exporting a frozen `Manifest` dataclass and `write_manifest(out_dir, manifest)` / `read_manifest(path)` functions.
  - Manifest fields: `harness_version`, `seed`, hashes (`eval_set`, `control_corpus`, `is_memorized`, `cutoffs`), `shortlist`, `composite_score` (formula + weights), `mcs_hyperparams`, `bootstrap_n`, `artifacts` (name → path).
  - Use `hashlib.sha256` over file bytes for hashes; serialize as `manifest.json`.
  - Add `tests/core/test_manifest.py` for round-trip (`read(write(m)) == m`) and asserting the JSON is human-readable (`json.loads` succeeds).
  - Observable: round-trip test passes; the written `manifest.json` opens and reproduces the dataclass.
  - _Requirements: 6.5, 8.4, 10.1, 10.2_
  - _Boundary: core.manifest_

- [ ] 3. MIA layer: features, control baseline, MCS calibrator

- [x] 3.1 Implement the five MIA features
  - Create `src/mia/features.py` exporting a frozen `MiaFeatures` dataclass and `compute_mia_features(response, logprobs, ref_logprobs, k=0.2)`.
  - Compute Loss (mean negative logprob), Min-K% (mean of bottom-K logprobs), Min-K%++ (per-token z-score using each position's `top_logprobs` mean and std), zlib ratio (`-sum(logprobs) / len(zlib.compress(response.encode()))`), and reference-model delta (`loss_self - loss_ref`; `None` if `ref_logprobs is None`).
  - Clip individual logprobs to a finite floor (e.g., `-30.0`) before averaging to avoid `-inf` poisoning.
  - Add `tests/mia/test_features.py` with a fixed token-logprob fixture asserting exact values for each feature, plus `None` ref-delta when `ref_logprobs` is None.
  - Observable: `pytest tests/mia/test_features.py` passes; a row with five well-known logprobs produces the documented numerical values.
  - _Requirements: 4.1, 4.2, 4.3_
  - _Boundary: mia.features_

- [x] 3.2 Implement the per-model control-corpus baseline
  - Create `src/mia/control.py` exporting a frozen `ControlBaseline` dataclass plus `build_baseline(model_lm, control_rows, ref_lm, min_valid=50)` and `standardise(features, baseline)`.
  - `build_baseline` calls the model on every control row, computes MIA features, drops rows where logprobs are missing, and stores per-feature mean and std; sets `is_calibrated = (n_valid >= min_valid)`.
  - `standardise` returns a dict of `(value - mean) / max(std, 1e-6)` for each feature and passes through `None` for `ref_delta` when disabled.
  - Add `tests/mia/test_control.py` covering: the `is_calibrated` boundary at `min_valid`, the standardised mean ≈ 0 / std ≈ 1 invariant on the control set itself, and `None` passthrough for ref_delta.
  - Observable: `pytest tests/mia/test_control.py` passes; with `n_valid = min_valid - 1` the resulting baseline reports `is_calibrated == False`.
  - _Requirements: 3.1, 3.2, 3.3, 3.4_
  - _Boundary: mia.control_

- [x] 3.3 Implement the MCS calibrator
  - Create `src/mia/mcs.py` exporting a frozen `MCSCalibrator` dataclass and `train(model_lm, is_memorized, oos_control, baseline, ref_lm, min_auc=0.6, seed=0)`.
  - Train `sklearn.linear_model.LogisticRegression(class_weight="balanced", solver="liblinear")` on standardised features (using `mia.control.standardise`) labelled by corpus origin; split a held-out portion with the given seed; report `holdout_auc` via `sklearn.metrics.roc_auc_score`; set `is_weak = (holdout_auc < min_auc)`.
  - Provide `predict_proba(features, baseline) -> float` returning a probability in [0, 1].
  - Add `tests/mia/test_mcs.py`: synthetic separable features → `holdout_auc > 0.95` and `is_weak == False`; label-shuffled features → `holdout_auc ≈ 0.5` and `is_weak == True`.
  - Observable: `pytest tests/mia/test_mcs.py` passes both AUC arms.
  - _Requirements: 5.1, 5.2, 5.3, 5.4_
  - _Boundary: mia.mcs_

- [ ] 4. Harness layer: smoke gate, evaluator, ranker, report

- [x] 4.1 (P) Implement the smoke-test gate
  - Create `src/harness/smoke.py` exporting `SmokeOutcome`, `Shortlist`, and `smoke_test(candidates, api_key, smoke_prompts, max_size=10, timeout_s=15.0)`.
  - For each candidate, run N=5 fixed prompts; exclude on `TimeoutError`, missing `top_logprobs`, or unparseable `Direction:` value; record `fail_reason`.
  - Cap selected models at `max_size`; persist the `Shortlist` as JSON via the runner caller (smoke returns the dataclass).
  - Add `tests/harness/test_smoke.py` mocking the LM client so the three exclusion paths fire correctly.
  - Observable: `pytest tests/harness/test_smoke.py` passes; a candidate that times out is in `outcomes` with `fail_reason == "timeout"` and not in `selected`.
  - _Requirements: 1.1, 1.2, 1.3, 1.4_
  - _Boundary: harness.smoke_

- [x] 4.2 Implement the per-model evaluator
  - Create `src/harness/evaluator.py` exporting `Record`, `CIBound`, `ModelEvalResult`, `evaluate_model(...)`, and `compute_majority_baseline(eval_set, bootstrap_n=1000, seed=0)`.
  - For each eval row: parse `Direction:` strictly (only `-1, 0, 1` accepted); on parse failure set `parse_ok=False`, `predicted_direction=None`, `raw_confidence=None`, `penalized_confidence=None`. On parse success compute MIA features, standardise via `baseline`, run `mcs.predict_proba` to get `p_memorized`, set `penalized_confidence = raw_confidence * (1 - p_memorized)`.
  - Compute Raw Accuracy and MemGuard Accuracy CIs by bootstrapping over parse-OK records; compute MCS-AUC CI by bootstrapping over `(p_memorized, label)` pairs from the held-out IS/OOS rows; record `parse_success_rate`, `parse_failures`.
  - Surface `temperature-not-honoured` warning when `raw_temperature_observed` is non-zero on any call; surface no other warnings here (gating warnings come from the ranker).
  - Add `tests/harness/test_evaluator.py`: with a mocked LM returning parseable responses on rows 1–8 and garbage on rows 9–10 over a 10-row eval set, assert `parse_failures == 2`, `parse_success_rate == 0.8`, and that the Raw Accuracy denominator is 8.
  - Observable: `pytest tests/harness/test_evaluator.py` passes; the result dataclass shows `parse_success_rate == 0.8` and a `CIBound` with `lo <= point <= hi`.
  - _Requirements: 3.3, 4.3, 5.4, 6.1, 6.2, 6.3, 7.1, 7.2, 7.3, 10.3_
  - _Boundary: harness.evaluator_
  - _Depends: 2.1, 2.2, 2.3, 3.1, 3.2, 3.3_

- [x] 4.3 (P) Implement the composite ranker and top-3 writer
  - Create `src/harness/ranker.py` exporting `CompositeScore`, `composite_score(results, majority_baseline, formula=COMPOSITE_FORMULA, gates=GATES)`, and `write_top3(scores, path)`.
  - Composite formula: `memguard_acc.lo * mcs_auc.point * parse_success_rate`. Gates: `parse_min=0.8`, `mcs_auc_min=0.6`, plus accuracy lower CI > majority upper CI. Failing any gate sets `survives_gates=False` and `score=0.0`.
  - Apply warnings: `weak-calibration` when `mcs_auc.point < mcs_auc_min`, `parse-unreliable` when `parse_success_rate < parse_min`, `not-better-than-baseline` when accuracy lower CI ≤ majority upper CI, `uncalibrated` when the result was marked as such by the evaluator.
  - `write_top3` writes `top3.md` with the top three by score; if fewer than three survive, list only survivors and include an explanatory note.
  - Add `tests/harness/test_ranker.py`: three synthetic results (one passes, one weak MCS, one parse-unreliable) → `top3.md` lists one model with the explanatory note.
  - Observable: `pytest tests/harness/test_ranker.py` passes; the generated `top3.md` contains the `Why fewer than three models` section when only one survives.
  - _Requirements: 5.3, 6.4, 7.4, 8.1, 8.2, 8.3, 8.4_
  - _Boundary: harness.ranker_
  - _Depends: 4.2_

- [x] 4.4 (P) Implement the structured report writers
  - Create `src/harness/report.py` exporting `render_terminal(results, majority, scores)`, `write_records(results, path)`, `write_summary_csv(results, scores, path)`, and `print_artifact_paths(paths)`.
  - `render_terminal` uses `rich.table.Table` to print one row per shortlisted model with model ID, Raw Acc with CI, MemGuard Acc with CI, MCS-AUC with CI, parse-success rate, and warnings; include a separate row for the majority baseline.
  - `write_records` streams one JSON object per `Record` to `records.jsonl` so memory stays bounded for long runs.
  - `write_summary_csv` writes one row per model plus a `__majority_baseline__` row with all CI fields and the composite score.
  - Add `tests/harness/test_report.py` with two synthetic results to assert: `records.jsonl` is valid JSONL with the expected key set; `summary.csv` has the right columns and the majority row.
  - Observable: `pytest tests/harness/test_report.py` passes; `cat records.jsonl | wc -l` equals the number of records across both models.
  - _Requirements: 9.1, 9.2, 9.3, 9.4_
  - _Boundary: harness.report_
  - _Depends: 4.2_

- [ ] 5. Integration: runner, replay, legacy cleanup, plotting, notebook

- [x] 5.1 Implement the runner CLI orchestrator
  - Create `harness.py` (project root) and `src/harness/runner.py` exporting `run(args)` plus an argparse front-end accepting `--eval-set`, `--candidates | --shortlist`, `--is-memorized`, `--oos-control`, `--cutoffs`, `--out-dir`, `--seed`, `--bootstrap-n`, `--reference-model | --no-reference`.
  - Sequence: load eval set + corpora + cutoffs → smoke (or honour `--shortlist`) → for each model: build_baseline → train MCS → evaluate_model → collect result → ranker → render_terminal + write_records + write_summary_csv → write_top3 → write_manifest → print_artifact_paths.
  - On `assert_cutoff_safe` failure, exit non-zero before any HTTP call to a candidate model.
  - Skip evaluation for any model whose `ControlBaseline.is_calibrated == False`; surface the `uncalibrated` warning into the result so it propagates to the ranker.
  - Observable: `python harness.py --eval-set tests/fixtures/tiny_eval.jsonl --shortlist mockA,mockB --cutoffs tests/fixtures/cutoffs.yaml --out-dir /tmp/run1` (with mocked LM) writes all five expected artifacts under `/tmp/run1/`.
  - _Requirements: 1.5, 2.5, 9.4, 10.1, 10.3_
  - _Boundary: harness.runner_
  - _Depends: 4.1, 4.2, 4.3, 4.4_

- [x] 5.2 Implement replay-from-manifest
  - Add a `harness replay --from-manifest <path>` subcommand to `harness.py` that reads a manifest, re-loads the same inputs by hash check, re-runs the pipeline with the recorded seed, and writes a fresh artifact set to a new `--out-dir`.
  - On hash mismatch (input file changed since the manifest), abort with a clear error rather than running with stale inputs.
  - Compare the new ranking against the manifest's ranking; warn if they differ. Bit-for-bit identity is not required but the ranking must be stable within bootstrap CIs.
  - Observable: replay against a saved manifest produces the same `top3.md` ordering as the original run; mutating an input file makes replay abort with a hash-mismatch message.
  - _Requirements: 10.2_
  - _Boundary: harness.runner_
  - _Depends: 5.1, 2.4_

- [x] 5.3 Remove legacy code and stale artifacts
  - Delete `main.py`, the entire `src/pipeline/` package, the legacy `src/dataset/lookahead_loader.py` and `src/dataset/fmp_ingest.py` only (do NOT remove the `src/dataset/` package — it now hosts `fmp_corpora.py` from task 1.2), `src/evaluate/metrics.py`, `src/utils/config_manager.py`, `data/lookahead_bench_2026_oos.jsonl`, `models_report.csv`, `test_fmp.py`, `test_timeout.py`, and the legacy tests (`tests/test_lookahead_loader.py`, `tests/test_metrics.py`, `tests/test_mia_scorer.py`, `tests/test_predict_module.py`).
  - `data/lookahead_bench_sample.jsonl` may be kept for reference; the FMP builder produces fresh calibration corpora and does not depend on it.
  - Confirm the new code path imports nothing from the deleted modules; `python -c "import src.harness.runner"` must succeed and `git grep -nE 'src\\.pipeline|src\\.dataset\\.(lookahead_loader|fmp_ingest)|src\\.evaluate|src\\.utils\\.config_manager|main\\.py' src harness.py tests` must return zero matches.
  - Observable: `pytest` is green and `git status` shows the listed files removed without leaving import errors.
  - _Requirements: 5.5, 9.5_
  - _Boundary: legacy cleanup_
  - _Depends: 5.1_

- [x] 5.4 (P) Implement paper-ready plotting helpers
  - Create `src/harness/plots.py` exporting `configure_paper_style()` and the five figure functions: `plot_mia_feature_distributions`, `plot_mcs_calibration`, `plot_accuracy_with_ci`, `plot_mcs_auc_with_ci`, `plot_composite_ranking`. Each returns a `matplotlib.figure.Figure`.
  - `configure_paper_style()` sets matplotlib `rcParams` once: `figure.figsize=(3.5, 2.5)`, `font.size=8`, `savefig.dpi=300`, `savefig.format="pdf"`, `savefig.bbox="tight"`, colorblind-safe palette `["#0072B2","#D55E00","#009E73","#CC79A7","#F0E442"]`, marker cycle that survives B&W reproduction.
  - Add `matplotlib >= 3.8` to `pyproject.toml` and run `uv sync`.
  - Add `tests/harness/test_plots.py` with synthetic dataclass fixtures asserting each plot returns a `Figure`, the figure size matches the paper width (3.5 inches at native), axes have non-empty x/y labels, and `fig.savefig(tmp_path / "x.pdf")` round-trips without error. Pixel content is not asserted.
  - Observable: `pytest tests/harness/test_plots.py` passes (≥ 5 tests, one per figure type); generated PDFs open at ~3.5" wide.
  - _Requirements: 12.3, 12.4, 12.5_
  - _Boundary: harness.plots_
  - _Depends: 3.2, 3.3, 4.2, 4.3_

- [x] 5.5 Author the qualification notebook with public API and rendered equations
  - Update `src/{core,mia,harness}/__init__.py` so `from src.core import NvidiaLM, EvalRow, EvalSet, load_eval_set, load_cutoffs, assert_cutoff_safe, bootstrap_ci, Manifest, write_manifest, read_manifest`, `from src.mia import MiaFeatures, compute_mia_features, ControlBaseline, build_baseline, MCSCalibrator, train_mcs`, and `from src.harness import smoke_test, evaluate_model, compute_majority_baseline, composite_score, write_top3, configure_paper_style, plot_mia_feature_distributions, plot_mcs_calibration, plot_accuracy_with_ci, plot_mcs_auc_with_ci, plot_composite_ranking, run` all succeed without internal-path imports.
  - Add `jupyter >= 1.0` (or `ipykernel + nbclient`) to `pyproject.toml` and run `uv sync`.
  - Author `notebooks/qualification.ipynb`. Required cells, in order: introduction; load cutoffs + eval set (display the cutoffs registry as a Markdown table sourced from `Qualified_Models.md` § "Training Cutoff Registry" so the reader sees which models are active, deferred, and omitted, and the derived `is_window` / `oos_window` dates); smoke shortlist; (Markdown LaTeX cell + compute cell + figure cell) for each of: Loss, Min-K%, Min-K%++, zlib ratio, ref-delta, control-baseline standardisation, MCS logistic regression, MemGuard penalty, bootstrap percentile CI, ROC-AUC, majority-class baseline, composite ranking score. Final cell saves all figures to `notebooks/figures/*.pdf`. All imports come from the public API roots — no `from src.harness.runner import _internal` paths.
  - Add `notebooks/figures/` to `.gitignore`.
  - Add `tests/harness/test_notebook.py` that uses `nbclient.NotebookClient` to execute the notebook with a mocked LM fixture (the same fixture pattern as task 6.1 establishes) and asserts every cell finishes without exception.
  - Observable: `jupyter nbconvert --execute notebooks/qualification.ipynb --to notebook --output /tmp/exec.ipynb` completes without error; `from src.harness import evaluate_model` succeeds at the Python REPL; `pytest tests/harness/test_notebook.py` passes.
  - _Requirements: 12.1, 12.2, 12.3, 12.4, 12.5, 12.6_
  - _Boundary: src/{core,mia,harness}/__init__.py, notebooks/qualification.ipynb, tests/harness/test_notebook.py_
  - _Depends: 5.1, 5.4_

- [ ] 6. Validation: integration tests for the harness

- [x] 6.1 (P) End-to-end run + manifest replay integration test
  - In `tests/harness/test_e2e.py`, drive `runner.run` against a 10-row in-memory eval set with two mocked models; assert `manifest.json`, `shortlist.json`, `records.jsonl`, `summary.csv`, and `top3.md` are written under the chosen `--out-dir`.
  - Replay from the written manifest into a second `--out-dir` and assert the top-3 ordering matches the first run.
  - Observable: both runs complete without errors; `top3.md` ordering is identical across the two runs.
  - _Requirements: 9.3, 9.4, 10.1, 10.2_
  - _Boundary: harness end-to-end_
  - _Depends: 5.1, 5.2_

- [ ] 6.2 (P) Cutoff-guard rejection integration test
  - In `tests/harness/test_cutoff_guard.py`, set the eval-set `_cutoff_date` to a date that precedes a candidate model's cutoff in `cutoffs.yaml`; invoke `runner.run` and assert the runner exits non-zero before any HTTP call (the mocked LM's `generate` should not be called).
  - Observable: the test asserts `mock_lm.generate.call_count == 0` and that a `CutoffViolation` was raised or surfaced as a non-zero exit.
  - _Requirements: 2.5_
  - _Boundary: harness end-to-end_
  - _Depends: 5.1_

- [ ] 6.3 (P) Small-N and class-imbalance warning integration test
  - In `tests/harness/test_warnings.py`, load a 30-row JSONL with 80% majority class via `load_eval_set`; capture log records and assert one `low statistical power` warning and one `class imbalance` warning are emitted, while loading still returns 30 rows.
  - Observable: caplog confirms both warnings present and `len(eval_set.rows) == 30`.
  - _Requirements: 2.2, 2.3, 2.4_
  - _Boundary: harness end-to-end_
  - _Depends: 2.2_

- [ ] 6.4 (P) Majority-baseline gating integration test
  - In `tests/harness/test_majority_gate.py`, drive `runner.run` against an eval set with majority-class accuracy 0.8 and a mocked model whose Raw Accuracy lower CI is 0.78; assert the resulting `ModelEvalResult.warnings` includes `not-better-than-baseline` and `top3.md` reflects this.
  - Observable: the test asserts the warning string is present in both the in-memory result and the generated `top3.md`.
  - _Requirements: 6.2, 6.4_
  - _Boundary: harness end-to-end_
  - _Depends: 5.1, 4.3_

## Implementation Notes
- 4.1 → 5.1 carryover: `harness.smoke` distinguishes `no_logprobs` from a generic `error` via a substring check on `RuntimeError` messages. Today this matches the wording in `src/core/nvidia_lm.py` (the parser raises with `"logprobs"`/`"top_logprobs"` in the message). When 5.1 wires the runner, consider promoting that distinction to a sentinel exception subclass so a future wording change in `nvidia_lm` does not silently demote `no_logprobs` failures to `error`.
