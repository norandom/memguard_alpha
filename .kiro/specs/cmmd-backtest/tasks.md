# Implementation plan

This is the task list for `cmmd-backtest`. Every task ties back to
`requirements.md` via `_Requirements: X.Y_` and (where useful) to a
component in `design.md` via `_Boundary:_`. Tasks marked `(P)` can run
in parallel with their immediate peers because they touch separate
files and have no shared state.

The plan groups work into four phases: Foundation (dependencies,
sentrux rules, eval-set builder), Core (the four `src/portfolio/`
modules), Integration (manifest extension, orchestrator, README), and
Validation (a single end-to-end run that proves the whole thing works
on real data).

## Tasks

- [ ] 1. Foundation: dependencies, layer registration, eval-set builder

- [x] 1.1 Add backtest dependencies and verify they install on Python 3.14
  - Add `vectorbt = "0.28.*"` and `matplotlib >= 3.8` to `pyproject.toml` `[project] dependencies`.
  - Run `uv sync` and confirm the lockfile updates without errors.
  - Smoke-import: `uv run python -c "import vectorbt; import matplotlib"` returns exit code 0.
  - If `uv sync` fails because vectorbt has no Python 3.14 wheel, record the failure mode on the task and stop the spec; the design's pandas-only fallback path needs to be invoked instead of forcing a workaround.
  - Observable: `uv.lock` contains `vectorbt` and `matplotlib`, and the smoke import command prints nothing and exits 0.
  - _Requirements: 5.1_

- [x] 1.2 Register the `portfolio` layer with sentrux and create an empty package
  - Add `[[layers]] name = "portfolio", paths = ["src/portfolio/*"], order = 1` to `.sentrux/rules.toml`.
  - Add two `[[boundaries]]` entries forbidding `portfolio ↔ dataset` and `portfolio ↔ mia` cross-imports, mirroring the existing `dataset ↔ mia` boundary.
  - Create `src/portfolio/__init__.py` as an empty module so the layer exists for sentrux to scan.
  - This is the foundation work that makes Req 4.1 enforceable (signal generation can only reach the harness through `evaluate_model` because the layer rules forbid sideways imports) and Req 5.1 addressable (the backtest engine has somewhere to live without breaking the existing dependency direction). The architectural constraint itself is documented in `design.md` § Architecture Pattern & Boundary Map.
  - Observable: `mcp__plugin_sentrux_sentrux__rescan` followed by `check_rules` reports `pass: true` and lists `portfolio` in the layer set; `uv run python -c "import src.portfolio"` succeeds.
  - _Requirements: 4.1, 5.1_

- [x] 1.3 Build the three-asset eval-set builder
  - Create `scripts/build_etf_portfolio_eval.py` modelled on `scripts/build_etf_multiyear_eval.py`.
  - Fetch EOD price series from FMP for `SWDA.L`, `XLK`, `IAU` using the `historical-price-eod/light` endpoint; do not fetch `BIL` here (BIL is the cash leg, no signals required).
  - Sample at least 100 distinct trading days from 2020-01-01 to today, drawing from both pre-2024-07-01 and post-2024-07-01 windows so the resulting set straddles the gpt-oss-20b cutoff.
  - For each `(date, ticker)` pair emit a prompt that follows the same "commitment" template the multiyear builder uses (no refusal, no reasoning, two-line answer).
  - Use a fixed `random.Random(seed)` and print the seed in the stdout summary.
  - If FMP returns fewer than 100 valid trading days for any ticker in the window, exit non-zero with a clear message naming the ticker.
  - Persist `metadata.ticker` and `metadata.date` on every row so the backtest can group signals later.
  - Observable: `uv run python scripts/build_etf_portfolio_eval.py` writes `data/eval/etf_portfolio.jsonl` with ≥ 300 rows; `jq '.metadata.ticker' data/eval/etf_portfolio.jsonl | sort | uniq -c` shows three tickers each ≥ 100; `jq '.metadata.date' ...` shows both pre- and post-2024-07-01 dates.
  - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6_
  - _Boundary: scripts.build_etf_portfolio_eval_

- [ ] 2. Core: the four `src/portfolio/` modules

- [x] 2.1 (P) Implement `portfolio.prices` for the universe
  - Create `src/portfolio/prices.py` exposing `fetch_universe_prices(tickers, start, end, api_key=None) -> pandas.DataFrame` and a `PriceFetchError` exception class.
  - Fetch EOD close prices from FMP `historical-price-eod/light` per ticker; use the same retry / API-key resolution pattern as `dataset.fmp_corpora.fetch_articles`.
  - Inner-join all per-ticker series on `date` so rows where any ticker is missing (e.g. LSE holiday vs NYSE) are dropped uniformly; the returned frame has no NaN cells.
  - Raise `PriceFetchError` when any ticker has fewer than 30 aligned trading days in the window.
  - Add `tests/portfolio/__init__.py` and `tests/portfolio/test_prices.py` with `pytest-mock` patching `requests.get`: cover the happy-path 4-ticker fetch, the inner-join behaviour against asymmetric series, the under-30-days error, and the API-key-missing error.
  - Observable: `uv run pytest tests/portfolio/test_prices.py -q` passes ≥ 4 tests; no real HTTP calls in the test run.
  - _Requirements: 5.1, 5.7, 7.2, 9.2_
  - _Boundary: portfolio.prices_

- [x] 2.2 (P) Implement `portfolio.cohens_d` and write per-(model, feature) artifact
  - Create `src/portfolio/cohens_d.py` exposing `compute_cohens_d(run_dir, cutoffs_path) -> pandas.DataFrame` plus the writer that produces `cohens_d.csv` and `cohens_d.md` in the run directory.
  - Read `records.jsonl`, group rows by `(model, feature_name)`, split each group into IS / OOS by joining `metadata.date` against the model's cutoff in `data/cutoffs.yaml`.
  - Compute Cohen's d on the raw (non-standardised) feature value with `pooled_std = sqrt(((n_is-1)*var_is + (n_oos-1)*var_oos) / (n_is+n_oos-2))`.
  - When either subset has fewer than 2 valid samples or when `pooled_std == 0`, emit `cohens_d = NaN` with `note = "insufficient samples"` (do not fail the artifact).
  - Include each model's `mcs_auc_holdout` (read from the run's `summary.csv`, which is where the harness already records per-model AUC) on every row of that model so the artifact is self-contained.
  - This task builds the function and unit-tests it on fixture data only; it does NOT need a live harness run. Task 3.2 is what actually invokes `compute_cohens_d` against a real run dir.
  - Add `tests/portfolio/test_cohens_d.py` covering: a synthetic two-class distribution with a known d-value (assert match within 1e-6); identical-class distributions produce d ≈ 0; zero-std subset produces NaN with the expected note; missing model in cutoffs.yaml is reported, not crashed.
  - Observable: running `compute_cohens_d` on a fixture run dir writes `cohens_d.csv` with one row per `(model, feature)`; `pytest tests/portfolio/test_cohens_d.py -q` passes ≥ 4 tests.
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 9.1_
  - _Boundary: portfolio.cohens_d_

- [x] 2.3 (P) Implement `portfolio.cmmd` filter
  - Create `src/portfolio/cmmd.py` exposing `apply_cmmd_filter(records, quantile=0.80) -> tuple[list[Record], float]`.
  - Filter out rows where `parse_ok` is False or `p_memorized` is None first, then drop rows whose `p_memorized` falls in the top `(1 - quantile)` slice of the surviving distribution.
  - Return the surviving records (in the original input order) and the empirical threshold value used, so the orchestrator can record it in the manifest.
  - Never modify a row's `predicted_direction` — the filter only drops rows.
  - Add `tests/portfolio/test_cmmd.py` covering: 100 records with uniform `p_memorized` ∈ [0, 1] yields exactly 80 survivors at `quantile=0.80`; threshold returned matches the 80th percentile within bootstrap noise; `parse_ok=False` rows are dropped before percentile computation; original input order is preserved in the output.
  - Observable: `pytest tests/portfolio/test_cmmd.py -q` passes ≥ 4 tests.
  - _Requirements: 6.1, 6.2, 6.4_
  - _Boundary: portfolio.cmmd_

- [x] 2.4 Implement `portfolio.backtest` engine core (weights → metrics)
  - Create `src/portfolio/backtest.py` and define the `BacktestMetrics` and `BacktestResult` dataclasses described in `design.md` § Components and Interfaces.
  - Implement `run_backtest(records, prices, *, cmmd_quantile=0.80, fees_one_way=0.00075, init_cash=1.0, seed=0, bootstrap_n=1000) -> BacktestResult`.
  - Build the `(date × ticker)` weight matrix from the record stream where `weight[d, t] = direction[d, t] * confidence[d, t]`; cap leverage by row-scaling when `sum(|weights[d, :]|) > 1.0`; route the residual `1 - sum(|weights|)` into the `BIL` column so the row always sums to 1.
  - Convert weights into vectorbt `size` units (`size = init_cash * weight / price`) and pass `fees=0.00075` (the one-way half of the paper's 15 bps round-trip cost).
  - Run the engine twice on the same price matrix — once on the raw record stream (`raw_alpha`), once after `apply_cmmd_filter` (`cmmd`) — and bundle metrics + equity curves + daily returns into a single `BacktestResult`.
  - Use `core.bootstrap.bootstrap_ci` to attach 95% CIs to Sharpe and mean daily return for both variants.
  - When fewer than 30 parse-OK rows survive across the horizon, append `low-row-count` to `result.warnings` so the summary doesn't get over-interpreted.
  - Add `tests/portfolio/test_backtest_engine.py` with a deterministic 5-trading-day toy where Sharpe and total return are computable analytically; assert vectorbt's output matches within tolerance and the 15 bps cost on a `|Δw| = 1` trade deducts exactly 7.5 bps one-way. Cover the leverage cap and the residual-to-BIL routing as separate cases.
  - Observable: `pytest tests/portfolio/test_backtest_engine.py -q` passes ≥ 5 tests; running `run_backtest` on a fixture returns a `BacktestResult` with both `raw` and `cmmd` populated and `equity_curves` starting at 1.0.
  - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8, 6.3, 6.5, 9.4_
  - _Boundary: portfolio.backtest_
  - _Depends: 2.1, 2.3_

- [x] 2.5 Implement backtest artifact writers (atomic)
  - Add `write_backtest_artifacts(result, run_dir) -> dict[str, Path]` to `src/portfolio/backtest.py`. Build every artifact in memory (CSV bytes, MD string, PNG buffer) and only write to disk after every artifact has been successfully built; on any IO failure raise `BacktestArtifactError` with a clear path-and-reason message and leave the run directory untouched (no partial files).
  - Artifacts written: `backtest_summary.csv`, `backtest_summary.md`, `equity_curves.csv`, `equity_curves.png`, `daily_returns.csv`. The PNG plots `raw_alpha`, `cmmd`, and `buy_and_hold_swda` series with annotated max drawdowns and a legend.
  - The function returns the `{artifact_name: Path}` dict so `scripts.run_cmmd_backtest` can record those paths in the manifest's `backtest.artifacts` block.
  - Add `tests/portfolio/test_backtest_writers.py` covering: all five files appear after a successful call; tampering with one target path (e.g. permission-denied via `tmp_path / "ro"`) raises `BacktestArtifactError` and leaves the directory empty; the `backtest_summary.md` table contains both `raw_alpha` and `cmmd` rows.
  - Observable: `pytest tests/portfolio/test_backtest_writers.py -q` passes ≥ 3 tests; calling `write_backtest_artifacts` on a fixture `BacktestResult` produces exactly five files; the failure case leaves zero files behind.
  - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.6_
  - _Boundary: portfolio.backtest_
  - _Depends: 2.4_

- [ ] 3. Integration: manifest, orchestrator, README

- [x] 3.1 Extend the harness manifest writer with a backtest block
  - Modify `src/harness/runner.py::_build_manifest` (and the `Manifest` dataclass in `core.manifest` if needed) so callers can pass an optional `backtest` block — additive change; existing harness runs that do not pass it produce identical manifests.
  - The block records: `signal_model`, `universe` (list of tickers), `cash_ticker`, `cmmd_quantile`, `cmmd_threshold_value`, `fees_one_way`, `init_cash`, `seed`, `bootstrap_n`, `n_is_rows`, `n_oos_rows`, and an `artifacts` map for the new file names listed in the design.
  - Add or extend a unit test in `tests/harness/test_runner.py` (or `test_manifest.py`) that verifies: (a) when no backtest dict is passed, the serialised manifest matches the pre-existing schema; (b) when one is passed, every required key round-trips through JSON.
  - Observable: full test suite still green; the new test asserts both schema variants.
  - _Requirements: 7.5, 8.2_
  - _Boundary: harness.runner, core.manifest_

- [x] 3.2 Build `scripts/run_cmmd_backtest.py` orchestrator
  - Create `scripts/run_cmmd_backtest.py` that ties everything together. The explicit internal order is:
    1. ensure the eval set exists (build via Task 1.3 if `data/eval/etf_portfolio.jsonl` is missing)
    2. invoke the harness for `openai/gpt-oss-20b` against that eval set into a fresh run directory (this writes `records.jsonl`, `summary.csv`, and the base `manifest.json`)
    3. call `compute_cohens_d` on the run dir
    4. invoke `scripts/analyze_is_oos_gap.py` on the run dir as a subprocess
    5. call `fetch_universe_prices` for the date span of the eval set
    6. call `run_backtest` followed by `write_backtest_artifacts`
    7. extend the manifest in place with the `backtest` block recorded by Task 3.1
    8. print every artifact path on success (using the same pattern `harness.report.print_artifact_paths` already uses).
  - Before invoking `analyze_is_oos_gap.py`, validate its preconditions: `records.jsonl` and `data/cutoffs.yaml` exist, and at least one record has `metadata.date` parsing as ISO-8601. If those preconditions are not met, fail fast with a clear stderr message rather than letting the subprocess error.
  - Abort with a non-zero exit and a clear stderr message when MCS calibration fails (Req 9.3) or when `write_backtest_artifacts` raises `BacktestArtifactError` (Req 7.6). Do not write the `backtest` manifest block in either failure path.
  - Add `tests/portfolio/test_run_cmmd_backtest_smoke.py` that drives the orchestrator with a 10-row miniature eval set, a `MagicMock`-patched LM returning scripted `Direction:` / `Confidence:` lines + scripted `top_logprobs`, and a `MagicMock`-patched FMP price fetcher. Assert `records.jsonl`, `cohens_d.csv`, `is_oos_gap.csv`, `backtest_summary.csv`, `equity_curves.csv`, `daily_returns.csv`, and `manifest.json` all exist and that the manifest's `backtest` block matches the documented schema.
  - Observable: `pytest tests/portfolio/test_run_cmmd_backtest_smoke.py -q` passes; running the script against the real eval set + real gpt-oss-20b produces the seven artifacts in the run directory.
  - _Requirements: 1.1, 2.1, 2.2, 4.1, 4.2, 4.3, 4.4, 4.5, 7.1, 7.5, 7.6, 8.1, 8.3, 8.4, 9.1, 9.3_
  - _Boundary: scripts.run_cmmd_backtest_
  - _Depends: 1.3, 2.2, 2.4, 2.5, 3.1_

- [x] 3.3 Update README sample-run section with IS/OOS gap and backtest results
  - Add an "IS-vs-OOS gap" subsection immediately after the existing accuracy table that includes the per-model gap numbers from `is_oos_gap.md`, and quote MemGuard-Alpha Section 5.3's IS 40.8 → 52.5% / OOS 47 → 42% finding for direct comparison.
  - Add a "Backtest" subsection below it that surfaces the raw-vs-CMMD Sharpe / mean-daily-return / max-drawdown table from `backtest_summary.md`, the relative Sharpe improvement, and an inline reference to `equity_curves.png`.
  - Use the latest reference run for the numbers; flag in the text that re-running `scripts/run_cmmd_backtest.py` regenerates them.
  - Observable: a fresh reader can find both subsections in `README.md` immediately under the existing Sample-run heading; the gap table cites the paper's numbers; the backtest table includes both `raw_alpha` and `cmmd` rows with bootstrap CIs.
  - _Requirements: 2.4, 2.5, 7.1_
  - _Boundary: README.md_

- [ ] 4. Validation: end-to-end run on real signals

- [x] 4.1 Run the full pipeline end-to-end and verify artifacts
  - Run `scripts/run_cmmd_backtest.py` against the live eval set + live `openai/gpt-oss-20b` endpoint; capture the run directory.
  - Confirm every artifact exists: `records.jsonl`, `cohens_d.{csv,md}`, `is_oos_gap.{csv,md}`, `backtest_summary.{csv,md}`, `equity_curves.{csv,png}`, `daily_returns.csv`, `manifest.json`.
  - Verify the manifest's `backtest` block has every key listed in the design (Data Models § Manifest extension) and that `n_is_rows + n_oos_rows` equals the parse-OK row count in `records.jsonl`.
  - Verify Cohen's d has one row per `(model, feature)` for every feature the records contain, and that no row violates `n_is < 2 or n_oos < 2 ⇒ note == "insufficient samples"`.
  - Verify the backtest summary contains both `raw_alpha` and `cmmd` rows with non-degenerate Sharpe CIs; if `low-row-count` was triggered, confirm the warning is in `backtest_summary.md`.
  - Run `mcp__plugin_sentrux_sentrux__rescan` followed by `check_rules` and confirm `pass: true` with the new `portfolio` layer in the report.
  - Observable: every artifact exists and matches the documented schema; sentrux reports zero violations; the README sample-run section reflects the actual numbers from this run.
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 2.1, 2.2, 2.3, 2.4, 2.5, 3.1, 3.2, 3.3, 4.1, 4.2, 4.3, 4.4, 4.5, 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8, 6.1, 6.2, 6.3, 6.4, 6.5, 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 8.1, 8.2, 8.3, 8.4, 9.1, 9.2, 9.3, 9.4_
  - _Boundary: end-to-end_
  - _Depends: 2.5, 3.2, 3.3_

## Implementation Notes

This section collects cross-cutting insights and gotchas discovered during
implementation, so subsequent tasks (and re-runs of `/kiro-impl`) avoid
the same mistakes. Add one-line entries here as work proceeds.

- 2.2: `Record` schema does NOT carry `metadata.date`; recover via `prompt_hash` ↔ eval-set join (see `scripts/analyze_is_oos_gap.py`). Functions touching record dates need an `eval_path` argument.
- 2.2: actual `MiaFeatures` field name is `min_k` (not `min_k_pct` as design says); use the runtime field name. summary.csv column is `mcs_auc_point` (mapped to artifact column `mcs_auc_holdout`).
- 2.3: `portfolio` (order=1) cannot import from `harness` (order=0). Use `typing.Protocol` for structural typing of record-like inputs across the sentrux layer boundary.
- 2.4: `run_backtest()` signature extended with `prompt_metadata: dict[str, dict[str, str]]` (prompt_hash → ticker/date) for the same Record-schema reason as 2.2. vectorbt 0.28 works on Python 3.14 with `Portfolio.from_orders(size_type='targetpercent', cash_sharing=True, freq='1D')`; pass `fees=fees_one_way` (= 7.5 bps one-way half of the paper's 15 bps round-trip).
- 2.4: keep test fixture/builder helpers under sentrux's max_fn_lines=120 (the cohens_d test `_build_fixture_run` had to be split — split builders into per-purpose helpers proactively).
- 4.1: live run on 2026-04-29 (`runs/cmmd_20260429T064026Z/`). gpt-oss-20b parse_rate=95.5%, raw_acc=0.546, holdout_auc=0.704. IS−OOS gap = −0.106 (OOS *better* than IS); Cohen's d on `loss` = +1.84 / `zlib_ratio` = +1.83 (calibrator detects something MIA-like). Backtest: raw_alpha Sharpe = −2.03; cmmd Sharpe = −1.80; relative improvement −11.3% (cmmd is less-negative). Headline paper finding (CMMD lifts a positive Sharpe higher) does not reproduce on a single-model run against this 3-asset ETF universe.
- 4.1: `_count_is_oos` originally counted all eval rows including parse failures (209+121=330) which broke Req 4.1's "n_is + n_oos == parse-OK count" check. Fixed to join records.jsonl with eval set on prompt_hash and filter to parse_ok=True (now 194+121=315 = parse-OK count).
