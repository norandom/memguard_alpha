# cmmd-backtest — Research Log

## Discovery scope

Light discovery, since this is an extension to a system we already know
well. Two tracks ran in parallel: codebase analysis for the extension
points in `src/harness/`, `src/mia/`, `src/dataset/`, and external
research on the vectorbt API plus its Python 3.14 story.

## Codebase findings

### Extension points

- The layered import graph is already sentrux-enforced
  (`.sentrux/rules.toml`): `harness` (order 0) → `dataset` / `mia`
  (order 1) → `core` (order 2). We add `src/portfolio/` as a third
  sibling at order 1. It gets imported by `harness/` and only depends
  on `core/`, so the no-cycles and no-upward-imports rules keep
  working.
- A cross-sibling boundary between `dataset/` and `mia/` already exists
  in `.sentrux/rules.toml`. We add the same kind for
  `portfolio ↔ dataset` and `portfolio ↔ mia`, so signal generation
  stays in the harness and the new layer can't reach sideways.

### Reusable surface (no modifications)

- `src.harness.evaluator.evaluate_model` produces `Record` rows with
  `predicted_direction`, `raw_confidence`, `p_memorized`, `parse_ok`,
  and the recently-added `raw_response_excerpt`. The pipeline consumes
  these as-is per Req 4.1.
- `src.mia.mcs.MCSCalibrator` already exposes `holdout_auc` and
  `predict_proba`. Cohen's d reads from the same IS/OOS calibration
  corpora the calibrator was trained on.
- `src.harness.report.write_records` writes `records.jsonl`, which is
  the only input both Cohen's d and the backtest need.
- `src.dataset.fmp_corpora.fetch_articles` shows how we already do FMP
  HTTP. The new `src/portfolio/prices.py` follows the same shape against
  the `historical-price-eod/light` endpoint that
  `scripts/build_etf_multiyear_eval.py` already hits.
- `data/cutoffs.yaml` stays as-is. Cutoffs are per-model, not
  per-ticker, and the eval-set builder is supposed to straddle cutoff
  dates so CMMD has IS rows to filter.

### Modifications required (small)

- `src/harness/runner.py`: add a `backtest` block to the manifest dict
  (Req 7.5, 8.2). Additive, no behaviour change when the backtest
  isn't invoked.
- `pyproject.toml`: add `vectorbt` and `matplotlib`.
- `.sentrux/rules.toml`: add `[[layers]] portfolio` at order 1 plus the
  two cross-sibling boundaries above.
- `README.md`: Sample-run section gets an IS/OOS-gap subsection and a
  backtest subsection (Req 2.4, 7).

### Existing IS/OOS gap script

`scripts/analyze_is_oos_gap.py` already does the work: it joins
`records.jsonl` against `data/cutoffs.yaml` and computes per-model
`IS_acc - OOS_acc`. Req 2 is mostly plumbing: run it on every finished
run and pull its output into the README.

## External research findings

### vectorbt API contract

| Concern | Finding | Source |
|---------|---------|--------|
| Factory | `Portfolio.from_orders(close, size, fees, slippage, init_cash, cash_sharing)` | https://vectorbt.dev/api/portfolio/base/ |
| `size` units | **Shares**, not weights. Convert via `init_cash * weight / close`. | https://vectorbt.dev/api/portfolio/orders/ |
| Fees | Percentage of order value, **one-way** (not round-trip). 15 bps round-trip ⇒ pass `fees=0.0015 / 2 = 0.00075`. | vectorbt API docs |
| Cash leg | NOT supported natively. Model BIL as the 4th column in the universe. | vectorbt issue tracker |
| Stats | `pf.sharpe_ratio()`, `pf.total_return()`, `pf.max_drawdown()`, `pf.value()` (equity), `pf.returns()` | vectorbt portfolio docs |
| Annualisation | 252 trading days assumed; override via `freq='D'` if needed. | vectorbt portfolio docs |

### Python 3.14 compatibility risk

`pyproject.toml` declares `requires-python = ">=3.14"`. vectorbt's
PyPI metadata caps at Python 3.13 today, so installing on 3.14 falls
back to a source build with numpy/numba bindings, which may fail.

Mitigation: pin `vectorbt = "0.28.*"` (the last-known-good with the
current Portfolio API) and validate `uv sync` on 3.14 as part of the
first task that lands the dep.

Fallback: a slim pandas-only backtest in `src/portfolio/backtest.py`
covers the same Sharpe / mean-return / drawdown / equity-curve outputs
in roughly 150 LOC. The component interfaces in `design.md` are scoped
so this swap is internal: callers never see `vectorbt.Portfolio`.

## Synthesis decisions

- Build vs adopt: adopt vectorbt at the pinned version. Keep the call
  site narrow so a pandas fallback is a one-file swap, not a refactor.
- Generalisation: the Cohen's d helper is reusable beyond this spec,
  but it lives in `src/portfolio/` for now to keep the layer boundary
  clean. Promote it to `mia/` later if anything else needs it.
- Simplification: BIL is cash-only. Three risk assets only (`SWDA.L`,
  `XLK`, `IAU`). This avoids the same refusal pattern we saw with phi
  and llama when asked to predict T-bills, and it matches the paper's
  handling of the risk-free leg.
- Single model: gpt-oss-20b. The paper doesn't ensemble, so one model
  per backtest is the apples-to-apples comparison. Multi-model is a
  later spec if anyone wants it.
- Output formats: every artifact lands in the run dir using the
  existing convention (csv + md, optional png). The existing notebooks
  and the README consume them without inventing new schemas.

## Open risks

1. vectorbt on Python 3.14. Recorded above. The first task that lands
   the dep also runs `uv sync` plus `python -c "import vectorbt"` as
   a smoke check.
2. FMP calendar mismatch. SWDA.L is on the LSE; XLK, IAU, BIL are on
   the NYSE. They observe different holidays. The backtest engine
   inner-joins on `date` so non-overlapping days drop out uniformly.
3. Eval-set size. R3.2 wants at least 100 trading days × 3 tickers,
   so 300 prompts minimum. With 1.5 s pacing and 8 workers that's
   roughly 5 minutes per phase × 3 phases (control baseline, MCS
   training, eval) per model. Fine for a single-model run.
4. Signal density. BIL is excluded from prediction, so low-confidence
   days end up mostly in cash. Same shape the paper reports. Our
   Sharpe will sit below their 4.11 because we have 3 cross-sectional
   positions vs their 100 stocks. The design flags this as a known
   trade-off so a low number doesn't get read as a regression.
