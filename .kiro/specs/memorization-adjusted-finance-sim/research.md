# Gap analysis: memorization-adjusted-finance-sim

Date: 2026-11-09. Requirements: [requirements.md](requirements.md) (11 requirements,
65 acceptance criteria; ensemble by default after the user's direction "do not
accept the first answer; the ensemble reveals the statistical properties").
Requirements are not yet approved; findings below may feed back into them.

## Summary

- About 60 % of the building blocks exist. The ensemble reduction
  (`generate_ensemble`, `reduce_draws`, `score_ensemble`), calibration identity,
  paired bootstrap, Brier/ECE/reliability helpers, the identified-versus-anonymous
  S&P 500 precedent and both full-logit scorers are in the repository.
- The critical gap is coupling: per-draw scoring is hard-wired to the
  `Direction:`/`Confidence:` parser, so a JSON regime-loading answer fails every
  scored draw today. The feature-based estimate (`feature_p_memorized`) does not
  depend on parsing and is the seam to use.
- Three decisions conflict with current invariants or data: the ensemble is
  opt-in by package invariant (R1.1 needs a steering amendment), the HY OAS
  series is no longer available as history on FRED and its reproduction is
  prohibited by ICE, and no free price source clearly permits publishing derived
  figures.
- Recommended shape: hybrid. Extend the scorer, identity and ensemble reduction;
  add new components for decision definitions, the macro data loader, the study
  statistics and the figures.
- Effort L (1 to 2 weeks of focused work); risk Medium, driven by data licensing
  and the full-distribution generation route.

## 1. Requirement-to-asset map

Tags: **Exists**, **Partial**, **Missing**, **Constraint**, **Research needed**.

| Req | Exists or partial | Missing | Constraint / research |
| --- | --- | --- | --- |
| 1 Guarded decision | **Partial.** `GuardedScore` (`harness/scorer.py:98-129`), `conf * (1 - p)` (`scorer.py:862`), `FAIL_PARSE` (`scorer.py:844`); `RecallWorkflow.run` labels `original_single` / `original_ensemble` and records provenance (`harness/workflow.py:461-570`) | Decision vector with per-component clipped, sign-preserving scaling; ensemble as default; skip adjustment when the calibrator is weak (today the discount is applied even when `is_weak`, `scorer.py:851-865`); "estimate unavailable" pass-through (today `FAIL_ERROR` or `RequestError`); model/source/settings/calibrator identity on the result | **Constraint:** the ensemble is "opt-in, per-call, no implicit default" (`.kiro/steering/recall-guard-package.md:97-99`, `core/ensemble.py:10-14`): needs a steering amendment. The single-draw path must stay byte-identical (parity test). |
| 2 Decision definitions | **Partial.** `decide` / `components` callbacks (`core/ensemble.py:16-21, 442-452`), finite-value checks (`workflow.py:177-216`), strict JSON parser precedent (`scripts/sp500_scoring.py:10`) | A decision-definition type with validation before any request, summary mapping, the shipped macro definition (5 loadings to 4 tilts and the defensive spread) | **Critical coupling:** per-draw scoring calls `_parse_direction` / `_parse_confidence` (`scorer.py:842-845`, `evaluator.py:296-347`). |
| 3 Estimate sources | **Partial.** Top-k features `compute_mia_features` (`mia/features.py:155-218`); full-vocabulary log-softmax `full_vocabulary_logprobs` (`core/local_evidence.py:220-237`, tests only); full-logit routes in `scripts/score_documents_llamacpp.py` and `scripts/score_documents_oga.py` | Full-distribution capability, client and profile for generation; estimate-source label; one feature set across sources (document features are not `MiaFeatures`) | **Research needed:** generating with full logits means generate then teacher-force prompt plus response, or a host that returns full distributions during generation. The host scripts import nothing from Recall Guard and run natively. |
| 4 Calibration identity | **Exists, mostly.** `CalibrationIdentity` v2 (`harness/scorer_client.py:47-130`), refusal on mismatch (`scorer_client.py:180-223`), fingerprint (`scorer.py:291-315`), weak flag (`mia/mcs.py:381-383`), AUC interval helper (`harness/local_reporting.py:89-114`) | Explicit estimate-source field, calibration-corpus hash, AUC interval on the calibrator; identity and weak flag for the document calibrator | **Constraint:** `to_dict` stays compatible with v1 payloads. |
| 5 Ensemble and distribution | **Partial.** `score_ensemble` / `_reduce_scores` (`scorer.py:636-801`): mean `p_memorized_point`, order-statistic interval, Wilson agreement, lag dependence, `min_parsed` failure with `None`; `reduce_draws` (`core/ensemble.py:442-516`): component location, multimodal verdicts, `draws_sha256`; pure replay (`scorer.py:568-608`) | Per-component quantiles, spread, sign agreement; quantiles of the estimate; number of distinct answers; first-draw position; the independence statement | Two different representative draws today (scorer: median by p; core: modal decision); component location defaults to the median while p uses the mean; `draws_sha256` hashes text only; `retain_draws` defaults to False; `consensus.py` stays stdlib-only. |
| 6 Positive control | **Partial.** S&P 500 precedent: identified / anonymous prompts, cutoff cohorts, matched dates (`scripts/sp500_inputs.py:46-155`), cutoff contrast without interval (`scripts/sp500_scoring.py:83-140`); `load_cutoffs` (`core/loader.py:213-240`) | FRED / ALFRED client with vintages, macro-row loader, macro prompt renderer, realized returns for 4 assets | **Constraint:** pandas is not a runtime dependency; `portfolio` may not import `dataset`. **Research needed:** HY OAS (see section 3). |
| 7 Negative control | **Exists (data).** `data/edgar/` (40/100/40/40) and the R26, R38, R40, R41 results | A decision task over EDGAR rows, "change versus no adjustment" with interval, tolerance check | Defining a finance decision on press releases that keeps the negative control meaningful is a design decision. |
| 8 Tests | **Partial.** `bootstrap_ci` (`core/bootstrap.py:47-152`), paired effect (`harness/intervention_metrics.py:152-193`), Brier / ECE / reliability (`local_reporting.py:121-164`), `wilson_interval` (`core/consensus.py:78`), frozen-plan and approval patterns (`scripts/measure_modal_model.py:123-175`, `harness/workflow_plan.py`) | Holm adjustment, random-answerer null, log loss, power, half-ensemble variance test, first-draw-versus-consensus rate | **Constraint:** no scipy (deliberately, `core/bootstrap.py:21-22`); exact binomial and permutation tests need to be written with the standard library and numpy. Serial correlation across months calls for a block bootstrap (Bertrand, Duflo and Mullainathan 2004). |
| 9 Convergence | **Partial.** Full-distribution math (`token_statistics`, `full_vocabulary_logprobs`); pinned artifacts | Store top-K alternatives per position from the full vector so k can be truncated; mass coverage; per-feature agreement; downstream decision delta | Only within one model and one set of weights. |
| 10 Worked example and figures | **Partial.** `harness/plots.py` (Wong palette, marker cycle for black and white, `configure_paper_style`); MkDocs gen-files hook (`mkdocs.yml`); strict build | Any published image (docs has none); a reliability-diagram helper (the docstring still says it is "not constructible", which is no longer true); offline replay of a full worked example | **Constraint:** matplotlib only in the `backtest` extra and `dev` group; `import recall_guard` must not import matplotlib. **Research needed:** pre-rendered committed figures versus generation at docs build time. |
| 11 Execution and reporting | **Exists, mostly.** Host lifecycle with verification (`scripts/serve_local_*.ps1`), `CallBudget`, `LocalAttemptBudget`, Modal cost and approval records, report template and rules | One study runner that combines frozen limits, time, cost and approval for this study | Next register entry is R42. |

## 2. Implementation approaches

### Option A: extend existing components

Extend `MemoryGuardedScorer`, `GuardedScore` and `RecallWorkflow` with decision
vectors, sources and the distribution summary; add the macro task to the S&P 500
scripts.

- Pros: fewest new files; reuses the parity-tested path and provenance.
- Cons: `scorer.py` is already large and parse-coupled; adding decision vectors
  there risks breaking the byte-identical single-draw parity; study code would
  spread across scripts.

### Option B: new components

A new decision-and-guard layer (decision definitions, guarded result, adjustment)
in `harness`, a new macro dataset module in `dataset`, a new study package in
`scripts/` for the statistics and figures.

- Pros: clean seams, the public scorer stays untouched, easy to test in isolation.
- Cons: duplicates provenance and ensemble wiring that `RecallWorkflow` already
  has; two parallel "guarded" paths for adopters to choose between.

### Option C: hybrid (recommended for evaluation in design)

- **Extend:** `CalibrationIdentity` (estimate source, corpus hash), `score_ensemble`
  / `reduce_draws` (quantiles, spread, sign agreement, distinct answers, first-draw
  position, independence note), the scorer's weak-calibrator and
  no-evidence behaviour, `plots.py` (reliability diagram).
- **New:** decision definitions and a guarded-decision result that use
  `feature_p_memorized` instead of the direction parser (removing the coupling);
  a FRED/ALFRED macro loader in `dataset` (stdlib plus numpy); a study module for
  Holm, the random-answerer null, log loss, power, the half-ensemble variance test
  and the first-draw rate; a full-distribution source for generation.
- **Phasing:** (1) decision definitions plus ensemble-default API with top-k
  estimates on the existing local hosts; (2) full-distribution source and
  convergence; (3) controls, statistics, figures and the report.

Trade-off: more planning, but the public parity-tested path stays intact and the
new API has a clean, importable surface.

## 3. Research needed (carry into design)

1. **HY OAS replacement.** FRED now keeps only 3 years of BAMLH0A0HYM2
   ("Starting in April 2026, this series will only include 3 years of
   observations"), and ICE prohibits reproduction without written permission.
   Options: Moody's BAA10Y or another credit spread with clear terms (check its
   copyright notes), or obtain ICE permission. This changes the worked example's
   credit-stress input.
2. **Point-in-time macro data.** FRED API `series/observations` with
   `realtime_start = realtime_end = decision date`, or `vintage_dates`; free API key
   (one more variable in `.env`); 120 requests per minute. CPIAUCSL has vintages
   from 1972; T10Y2Y is rarely revised.
3. **Realized returns and licensing.** Tiingo free is internal use only; Yahoo is
   personal use; FMP Basic may not reach 2016 (test with the existing key);
   Alpha Vantage monthly adjusted is likely free (25 requests per day). Whether a
   published figure may show derived returns needs a decision; a cash leg can use
   FRED DTB3. The existing `portfolio/prices.py` uses FMP for SWDA.L, XLK, IAU and
   BIL.
4. **Full distribution for generated answers.** Generate with the host, then
   teacher-force prompt plus response through the verified full-logit routes
   (exact, but two passes and native hosts), or extend a host to return full
   distributions while generating.
5. **One feature set across sources.** The single-draw `MiaFeatures` and the
   document-likelihood features differ; convergence (R9) needs the same features
   from top-k and from full distributions.
6. **Negative-control decision task.** A finance decision over SEC EDGAR press
   releases that keeps "nothing to detect" true.
7. **Statistics without scipy.** Exact binomial and permutation nulls, Holm, a
   block bootstrap over months, paired loss differentials (Diebold and Mariano
   1995), correlated-variance comparison for the half-ensembles (Pitman 1939 or
   Morgan 1939), paired AUC (DeLong et al. 1988), power by simulation.
8. **Figure publishing.** Pre-render figures from recorded results and commit
   them, or render at docs build time (matplotlib is not in the `docs` extra).
9. **Steering amendment.** Ensemble by default changes a package invariant; the
   single-draw parity test must keep passing for the labelled single-draw path.

## 4. Effort and risk

- **Effort: L (1 to 2 weeks).** Several new components (decision layer, macro
  loader, statistics, figures) on top of substantial reusable code, plus two
  measured studies.
- **Risk: Medium.** Known patterns and verified hosts; the risk sits in data
  licensing (HY OAS, prices), the full-distribution generation route, and keeping
  the parity-tested single-draw path unchanged.

## 5. Recommendations for design

- Take option C; keep the existing scorer as the parity-tested single-draw
  baseline and build the guarded-decision API on `feature_p_memorized`.
- Decide early: HY OAS replacement, price source and publication rights, and the
  full-distribution route for generated answers.
- Make the distribution summary a pure function of recorded draws, so figures and
  statistics replay offline (R5.8, R10.7).
- Use a block bootstrap over decision months for every interval in the positive
  control.

## Sources

- Codebase map: files and lines cited above (read-only exploration, 2026-11-09).
- Adopter: `D:\Source\Global_Macro_AI_Factors`, `notebooks/appendix_i_factor_dispersion.ipynb`,
  `docs/ensemble_prompt_consensus.md`, `macro_framework/factor_scoring.py`.
- FRED API: [series_observations](https://fred.stlouisfed.org/docs/api/fred/series_observations.html),
  [realtime_period](https://fred.stlouisfed.org/docs/api/fred/realtime_period.html),
  [series_vintagedates](https://fred.stlouisfed.org/docs/api/fred/series_vintagedates.html),
  [errors](https://fred.stlouisfed.org/docs/api/fred/errors.html),
  [terms_of_use](https://fred.stlouisfed.org/docs/api/terms_of_use.html),
  [BAMLH0A0HYM2](https://fred.stlouisfed.org/series/BAMLH0A0HYM2).
- Prices: [Tiingo EOD](https://www.tiingo.com/documentation/end-of-day),
  [Tiingo pricing](https://www.tiingo.com/about/pricing),
  [Alpha Vantage](https://www.alphavantage.co/documentation/),
  [Yahoo API terms](https://legal.yahoo.com/us/en/yahoo/terms/product-atos/apiforydn/index.html).
- Statistics: Efron and Tibshirani (1993); Berg-Kirkpatrick et al. (2012);
  Bertrand, Duflo and Mullainathan (2004); Holm (1979); DeLong, DeLong and
  Clarke-Pearson (1988); Obuchowski and McClish (1997); Diebold and Mariano
  (1995); Pitman (1939); Morgan (1939).

---

# Design discovery and decisions (2026-11-09)

## Summary
- **Feature**: `memorization-adjusted-finance-sim`
- **Discovery Scope**: Complex integration (extension of the scorer and ensemble, new data and statistics components, native host)
- **Key Findings**:
  - BAA10Y (the user-chosen HY OAS substitute) has full history on FRED from 1986 but carries a Moody's copyright that prohibits reproduction ("Copyrighted: Citation Required"). It is used internally; only derived statistics are published, with citation. CPIAUCSL and DTB3 are public domain (citation requested); T10Y2Y is "Citation Required" and built from US Treasury data.
  - Realized returns use the existing FMP key (`portfolio/prices.py`), publishing aggregates only; coverage back to 2016 is verified in the first implementation task.
  - The ensemble-default requirement conflicts with the package invariant only for the existing scorer; a separate guarded-decision API resolves it without touching scorer parity.

## Consulted skills and guidance
- **dataviz**: form by job (interval dot plot, reliability diagram, strip plot, lines), one axis per chart with small multiples instead of dual axes, fixed-order categorical palette validated with `validate_palette.js` for light and dark surfaces, identity never by color alone (marker shape and fill), a data table beside every figure, dark mode as its own validated render.

## Design Decisions

### Decision: Separate ensemble-first API next to the unchanged scorer
- **Context**: R1.1 requires the ensemble by default; the package invariant says the ensemble is opt-in and the single-draw scorer must stay byte-identical (parity test).
- **Alternatives Considered**: 1. Flip the scorer's default; 2. New `guarded_decide` API, scorer unchanged.
- **Selected Approach**: 2. Steering amendment: the guarded-decision API is ensemble-first; `MemoryGuardedScorer` keeps opt-in ensembles.
- **Rationale**: keeps parity and existing adopters unaffected; one clear entry point for finance simulations.
- **Trade-offs**: two entry points; the docs must say which to use.

### Decision: Per-draw estimates from features, not the direction parser
- **Context**: the scorer's per-draw path requires `Direction:` / `Confidence:`; JSON decisions fail.
- **Selected Approach**: estimates from `MiaFeatures` computed per source (top-k or full distribution), independent of parsing; decisions parsed by the `DecisionDefinition`.
- **Rationale**: decouples decision format from memorization scoring (generalisation: any decision definition).

### Decision: One feature set for both sources
- **Context**: R9 compares top-k and full-distribution features; the document-likelihood features differ from `MiaFeatures`.
- **Selected Approach**: compute `MiaFeatures` from full-distribution evidence by deriving top-k alternatives from the full vector; keep the top 100 per position for truncation.
- **Rationale**: identical features make convergence measurable and allow one calibrator type.

### Decision: Full distribution of generated answers by teacher forcing on the same weights
- **Context**: no host returns full distributions while generating through an OpenAI-compatible endpoint.
- **Alternatives Considered**: 1. Extend llama-server; 2. A native host that teacher-forces prompt plus response with the llama.cpp library (verified exact in R40); 3. OGA only.
- **Selected Approach**: 2, behind the existing gateway pattern; generation through llama-server on the same GGUF digest.
- **Trade-offs**: two passes per draw; exactness already established.

### Decision: Statistics without scipy
- **Selected Approach**: `core/significance.py` with Holm, block bootstrap over months, exact binomial, Monte Carlo random-answerer null, paired loss difference, half-ensemble variance comparison and simulated power, numpy and stdlib only.
- **Rationale**: runtime dependency set unchanged; methods per Efron and Tibshirani 1993, Bertrand, Duflo and Mullainathan 2004, Holm 1979, Diebold and Mariano 1995, Pitman 1939.

### Decision: Negative control as a sentiment-tilt decision on SEC EDGAR releases
- **Context**: R7 needs a decision task where nothing should be detectable.
- **Selected Approach**: one component `outlook` in [-1, 1] per release; measure the change against no adjustment and flag variants above the frozen tolerance.

### Decision: Figures rendered offline and committed
- **Context**: matplotlib is not in the `docs` extra; strict builds should not depend on it.
- **Selected Approach**: render SVG light and dark variants from recorded results; commit them with CSV tables; docs build only links them.

## Simplifications
- No new model runtime; no generic workflow engine; the study runner reuses the frozen-plan and approval-record pattern.
- One calibrator type for both sources; one distribution summary function shared by API, study and figures.

## Risks and Mitigations
- Study duration (about 13 to 15 hours per model locally): resumable runner, one model loaded at a time, idle-stop hosts.
- Licensing of BAA10Y and FMP data: private raw values, published aggregates, citations.
- First-draw position depends on draw order: record `draw_index` from generation time; never reorder.
- Serial correlation across months: block bootstrap with a frozen block length.

## Design validation amendments (2026-11-09, user decisions)

- **Ensemble arms**: local llama.cpp is bitwise deterministic at temperature 0, so
  local ensembles use a frozen sampling temperature (default 0.7) with recorded
  per-draw seeds (sampling distribution); a Modal vLLM arm at temperature 0 measures
  serving nondeterminism. Claims name their arm.
- **Calibration set**: pre-cutoff macro months disjoint from evaluation, identifying
  rendering as memorized class and anonymised rendering as control (the adopter's
  `_build_corpus`), so all scarce post-cutoff months stay in evaluation.
- **Power gate**: pilot of about 12 months times 8 draws; full run only at simulated
  power of at least 0.8; pilot items excluded from confirmatory analysis.
