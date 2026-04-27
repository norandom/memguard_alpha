# Gap Analysis: honest-model-ranking

## 1. Current State Investigation

### Layout and conventions
```
src/
├── dataset/
│   ├── lookahead_loader.py      # JSONL → dspy.Example, fixed 80/20 split, seed=42
│   └── fmp_ingest.py            # OOS news ingestion (replicates label per ticker — broken signal)
├── models/nvidia_lm.py          # Thin requests wrapper, asks for logprobs/top_logprobs=20
├── pipeline/
│   ├── signature.py             # FinancialPrediction dspy.Signature (unused by predictor)
│   ├── predict_module.py        # Orchestrates masker → NvidiaLM → MIAScorer
│   ├── math_reasoning.py        # InputMasker via dspy.ChainOfThought
│   └── mia_scorer.py            # Loss + Min-K% only, hardcoded thresholds (Loss<0.5, Min-K%>-0.5)
├── evaluate/metrics.py          # Returns (acc, raw_conf, acc, mem_conf) on flat dict list
└── utils/config_manager.py      # Pulls model list from NVIDIA /v1/models, keyword-filtered
main.py                          # Per-model loop, ThreadPoolExecutor for 15s hard timeout
tests/                           # pytest + pytest-mock; one test per module, all unit-level
data/
├── lookahead_bench_sample.jsonl       # 5 rows, historical
└── lookahead_bench_2026_oos.jsonl     # 25 rows, 5 per ticker, label replicated → broken
```

### Conventions in use
- Plain Python classes, no abstract base / Protocol.
- DSPy used for `Example`, `Module`, `ChainOfThought`, `LM` configuration; NVIDIA logprobs path bypasses DSPy entirely (raw `requests.post` in `NvidiaLM.generate_with_logprobs`).
- Tests are pure unit-level with `mocker` for HTTP; no integration suite, no fixtures dir.
- `pyproject.toml` has an empty `dependencies = []` array — deps are resolved by `uv` from the `.venv` lockfile but not declared here. Any new library has to be added properly.
- No steering files (`.kiro/steering/` does not exist), so steering context is unavailable for this analysis.

### Integration surfaces
- NVIDIA OpenAI-compatible chat completions endpoint with `logprobs=true, top_logprobs=20`.
- DSPy `dspy.LM` is configured globally per-model in `main.py:54-61` for the masker.
- `config.json` is the model registry, mutated by `config_manager.sync_models`.

## 2. Requirement-to-Asset Map

Tag legend: **EXTEND** (existing asset is reusable with edits), **NEW** (no existing asset), **REPLACE** (existing asset must be removed/rewritten), **GAP** (research needed).

| Req | Subject | Existing asset | Tag | Notes |
|-----|---------|----------------|-----|-------|
| 1 | Smoke test → ≤10 model shortlist | `config_manager.sync_models` (lists models), `main.py` per-model loop | NEW | No smoke-test step exists; `main.py` runs every config'd model. Need a `smoke_test.py` that calls each model with N fixed prompts and gates on parse + logprobs + timeout. |
| 2 | `(prompt, target)` JSONL contract, no split, cutoff guard | `lookahead_loader.py` reads JSONL but force-splits 80/20 and assumes `ticker/date/context/target_direction` schema | REPLACE | Loader must accept generic `prompt` field, drop split, add `cutoff_date` enforcement. Existing schema is overspecific (news-shaped). |
| 3 | Per-model control-prompt baseline | none | NEW | No control corpus, no baseline distribution storage, no z-score path. Need a new `control_corpus.py` and a per-model `baseline.json` artifact. |
| 4 | 5-feature MIA set | `mia_scorer.py` has Loss + Min-K% | EXTEND | Add Min-K%++ (per-token calibration: needs `top_logprobs` distribution per position), zlib ratio (needs full response text), reference-model delta (needs second LM call). |
| 5 | MCS classifier replacing thresholds | `apply_penalty` is hardcoded threshold logic | REPLACE | Need a sklearn-style logistic regression per model, trained on labelled IS/OOS prompts, predicting `p(memorized)`. Drop `apply_penalty` thresholds entirely. |
| 6 | Bootstrap CIs + majority-class baseline | `evaluate/metrics.py` returns point estimates only | REPLACE | Rewrite Evaluator to return per-model dict with bootstrapped CIs and a baseline row. |
| 7 | Parse-failure accounting | `predict_module.py:23-35` silently sets `direction=0` on failure | EXTEND | Mark failures explicitly (e.g., `direction=None`), surface in evaluator, add per-model success-rate metric. |
| 8 | Top-3 ranking + `top3.md` | none | NEW | New `ranker.py` module + Markdown writer. |
| 9 | Structured CLI report + per-(model, prompt) artifact | `main.py:148-153` writes flat `models_report.csv`, `print` statements for terminal | REPLACE | Replace flat CSV. `rich.table` is the natural fit but plain string formatting works. JSONL artifact for raw records. |
| 10 | Run manifest + `--from-manifest` reproducibility | none | NEW | New `manifest.py`. Hash inputs, persist seed/weights/version. |

### Cross-cutting gaps
- **No labelled IS/OOS corpus** for MCS training (Req 5). The current data is unlabelled along this axis. Must build: take known pre-cutoff prompts (from sample.jsonl + new collection) as IS=1, post-cutoff prompts as IS=0.
- **Reference model selection** (Req 4) — `meta/llama-3.2-1b-instruct` was the proposed default but its training cutoff vs the eval window is unverified. **Research Needed**.
- **Min-K%++ implementation specifics** — needs per-token calibration: `(logprob_token - mean_top_k_logprobs_at_position) / std_top_k_logprobs_at_position`. The NVIDIA API returns `top_logprobs=20`, which is enough. Confirmed feasible.
- **zlib ratio** — needs the full response string and the prompt; both are available. Trivial.
- **Bootstrap implementation** — straightforward with `numpy.random.choice`. No new heavy dependency.
- **Empty `pyproject.toml` dependencies** — adding `numpy`, `scikit-learn`, `scipy` (or just `numpy` + a hand-rolled logistic regression) is a packaging concern, not a research one.

## 3. Implementation Approach Options

### Option A: Extend existing modules in place
Keep `src/{dataset,models,pipeline,evaluate}/` layout. Add features to existing files (extend `MIAScorer` with three new methods, extend `Evaluator` with bootstrap, add a class to `predict_module.py` for parse-failure tracking). Add new tiny modules only where there is no natural home (`smoke_test.py`, `ranker.py`, `manifest.py`).

- ✅ Minimal churn, existing tests still pass with small edits.
- ✅ Reuses NVIDIA LM client and DSPy `Example` plumbing.
- ❌ `MIAScorer.apply_penalty` becomes a misnomer once thresholds are dropped — has to be removed or repurposed.
- ❌ `evaluate/metrics.py` is a 23-line file; the new Evaluator is materially different (per-model CIs, baseline, parse-failures). Editing in place is functionally a rewrite of that file.

### Option B: New `src/harness/` package, leave old code alone
Build the new pipeline beside the old one: `src/harness/{shortlist.py, control.py, mia.py, mcs.py, bootstrap.py, ranker.py, manifest.py, runner.py}`. New entry point `harness.py` (or `main.py --harness`). Existing `src/pipeline/` stays as legacy reference.

- ✅ Clean separation; no risk of breaking existing tests during build-out.
- ✅ Easier to delete legacy code in one PR after parity is reached.
- ❌ Two-week period of duplicated logprob-handling code.
- ❌ More files; the project is small enough that this might over-structure.

### Option C: Hybrid — rewrite scoring + evaluation paths in place, isolate genuinely new concerns into new modules
- Rewrite in place: `dataset/lookahead_loader.py` → generic loader; `pipeline/mia_scorer.py` → 5-feature scorer (no penalty method); `evaluate/metrics.py` → bootstrap evaluator; `predict_module.py` → parse-failure-aware.
- New modules: `src/harness/{smoke_test.py, control_corpus.py, mcs.py, bootstrap.py, ranker.py, manifest.py}` and a new `harness.py` entry point that replaces `main.py`'s loop.
- Delete: `src/pipeline/math_reasoning.py` (input masker is not in scope per requirements; the harness should not silently mutate prompts).
- Delete: `src/dataset/fmp_ingest.py` (news ingest is out of scope; macro ingest is a separate spec).

- ✅ The pieces that are conceptually unchanged stay in place (logprob fetcher, JSONL reader skeleton).
- ✅ Genuinely new statistical machinery lives in its own package — easy to test in isolation.
- ✅ Parse-failure fix and threshold removal happen at the same time as MCS comes in, avoiding an interim broken state.
- ❌ Coordinated change set is larger than Option A's drip-feed.
- ❌ Touches `main.py` and `tests/test_metrics.py`, `tests/test_mia_scorer.py`, `tests/test_predict_module.py` simultaneously.

**Recommended: Option C.** The legacy threshold logic must be removed (Req 5.5) — there is no honest extension path that keeps it. Bootstrap CIs and parse-failure semantics are also rewrites of `evaluate/metrics.py`. The new statistical concerns (MCS, control corpus, ranker, manifest) are genuinely separate from per-call scoring and benefit from their own package. Leaving the input-masker and FMP-news ingest behind aligns the codebase with the new requirements scope (out-of-scope per Req 2 boundary).

## 4. Effort & Risk

| Component | Effort | Risk | One-line justification |
|-----------|--------|------|------------------------|
| Generic JSONL loader + cutoff guard (Req 2) | S | Low | Minor edits; existing parser stays. |
| Smoke-test runner (Req 1) | S | Low | Loop with timeouts; ≤80 LOC. |
| MIA feature set: Min-K%++, zlib, ref-model (Req 4) | M | Medium | Min-K%++ needs per-token top-k handling; ref-model adds latency budget per eval row. |
| Control-prompt corpus + baseline calibration (Req 3) | M | Medium | Need to build a corpus and decide a window; standardisation logic itself is mechanical. |
| MCS classifier (Req 5) | M | Medium | Requires a labelled IS/OOS dataset that does not exist yet; classifier itself is logistic regression. |
| Bootstrap CIs + majority-class baseline (Req 6) | S | Low | Pure numpy; ~100 LOC. |
| Parse-failure accounting (Req 7) | S | Low | Local change in predictor + evaluator. |
| Composite ranker + `top3.md` (Req 8) | S | Low | Trivial once metrics exist. |
| Structured CLI report + per-record JSONL (Req 9) | S | Low | `rich.table` or stdlib formatting. |
| Manifest + `--from-manifest` (Req 10) | S | Medium | Bit-for-bit reproduction depends on temperature-0 honour and stable ordering — **research item**. |
| Total | **L (1–2 weeks)** | **Medium** | Three M-effort items dominate; the rest are small. |

## 5. Research Needed (carry to design phase)

1. **Reference-model choice for MIA delta (Req 4).** Verify training cutoff of `meta/llama-3.2-1b-instruct` (or alternative) is *before* the evaluation window so its perplexity reflects a "should-have-memorized" baseline. Confirm it's reachable on NVIDIA's endpoint.
2. **IS/OOS labelled corpus for MCS (Req 5).** What is the source of in-sample (pre-cutoff) prompts? Options: paraphrases of the existing `lookahead_bench_sample.jsonl` content; deliberately-memorized facts (Wikipedia paragraphs predating model cutoffs); the paper's own reference dataset if released. Decide before design.
3. **Per-model training-cutoff registry (Req 2.5, Req 3.1).** No structured registry exists in the project. Either hardcode a `cutoffs.yaml` for the shortlisted 10 models or accept user input via a CLI flag. Sources: NVIDIA model cards, vendor documentation.
4. **Determinism on NVIDIA endpoints (Req 10.2, 10.3).** Whether `temperature=0` produces bit-stable token sequences across calls on every shortlisted model. If not, manifest reproducibility must be relaxed to "metric-stable within bootstrap noise."
5. **Adding numpy/scipy/sklearn to `pyproject.toml`.** The current empty `dependencies = []` is unsafe for collaborators. Decide whether the design adds: `numpy` (mandatory), `scikit-learn` (for logistic regression) vs hand-rolled (one fewer dep), `rich` (nice CLI) vs plain string formatting.

## 6. Recommendations for Design Phase

- **Adopt Option C** as the implementation strategy.
- **Land the MCS labelled corpus decision first** in design — it is the single biggest source of unknowns.
- Define explicit module contracts (input/output schemas) for `harness/{shortlist, control, mia, mcs, bootstrap, ranker, manifest}` so each can be tested in isolation against the existing `pytest-mock` style.
- Document the composite-score formula and the gating thresholds (parse ≥ 0.8, MCS-AUC ≥ 0.6, accuracy lower CI > majority-class upper CI) in `design.md` so the user can review weights once before approval rather than re-litigating per task.
- Plan for the input-masker deletion (Req 2 boundary) and FMP news ingest deletion (out-of-scope) as part of the migration; do not leave dead code paths.

---

## Synthesis (Design Phase)

### Generalizations Adopted
- All five MIA features are "scalar functions of (prompt, response, logprobs)". Compute them in a single `compute_mia_features(...)` function returning a `dict[str, float | None]` rather than a Protocol per feature — keeps imports simple and matches the small feature count.
- Smoke test, control-corpus baseline pass, IS/OOS calibration pass, and evaluation pass all share the primitive "run model over a JSONL of prompts and collect Records." Implemented once as `run_corpus(model, prompts) -> list[Record]` and reused by the four callers.
- Bootstrap CI for accuracy and bootstrap CI for AUC share `bootstrap_ci(samples, statistic_fn, n=1000, seed)` — one helper, two callers.

### Build vs Adopt Decisions
- **Bootstrap**: build with `numpy.random.default_rng`. scipy.stats.bootstrap exists but pulling scipy in just for this is heavy.
- **Logistic regression (MCS) + AUC**: adopt `scikit-learn` (`LogisticRegression`, `roc_auc_score`). Battle-tested, gives `predict_proba` directly.
- **CLI tables**: adopt `rich` for the terminal report. Single dependency, dramatically better UX than `print`.
- **JSON validation for input contract**: build (a 10-line dict check is enough; jsonschema is overkill).
- **Hashing for manifest**: build with stdlib `hashlib.sha256`.
- **NVIDIA HTTP client**: extend the existing `NvidiaLM` to accept a `temperature` parameter (default 0) and to support a separate reference-model instance. Do not introduce a second API client class.

### Simplifications Applied
- No `MiaFeature` Protocol or registry. Five features, one function, return a dict.
- No separate `predict_module.py` glue layer in the new path. The runner calls `run_corpus()` directly and feeds records into `evaluator.evaluate_model()`.
- Drop `src/pipeline/math_reasoning.py` (input masker is out of scope per Req 2 boundary; the harness must not silently mutate prompts).
- Drop `src/dataset/fmp_ingest.py` (news ingest is out of scope; macro ingest is a separate spec).
- Drop `src/pipeline/signature.py` (the FinancialPrediction `dspy.Signature` is unused by the existing predictor and not referenced in the new design).
- Drop DSPy from new code paths. Existing DSPy usage (`dspy.Example`, `dspy.configure`, `dspy.ChainOfThought`) lived in the masker and loader, both of which are replaced. Removing DSPy as a declared project dependency is out of scope for this spec; the new code simply does not import it.

### Resolved Research Items
- **Reference-model choice (Research item 1)**: Default to `meta/llama-3.2-1b-instruct`. Cutoff verification is delegated to the per-model `cutoffs.yaml` registry maintained as part of Req 2.5; if the registry says its cutoff is after the evaluation window, the runner falls back to disabling the ref-delta feature for that run rather than picking a different reference model dynamically.
- **IS/OOS labelled corpus (Research item 2)**: Build two small JSONL corpora under `data/calibration/`. `is_memorized.jsonl` (label=1) draws from well-known pre-2023 content (existing `lookahead_bench_sample.jsonl` rows + a handful of canonical Wikipedia paragraphs about pre-2023 events). `oos_control.jsonl` (label=0) draws from post-cutoff content authored after every shortlisted model's training window. Same corpus serves the control-prompt baseline (Req 3) and the MCS classifier (Req 5) — its OOS half is the per-model perplexity baseline.
- **Per-model cutoff registry (Research item 3)**: Hard-coded `data/cutoffs.yaml` mapping model ID → ISO date. Manual maintenance; the runner fails fast if a shortlisted model is missing from this file.
- **Determinism on NVIDIA endpoints (Research item 4)**: Manifest reproducibility is relaxed from "bit-for-bit" to "metric-stable within bootstrap CI." Req 10.2 acceptance criterion still verifiable: the manifest plus a re-run produces the same ranking and overlapping CIs. The runner records non-temperature-0 honour as a per-model warning.
- **Dependency additions (Research item 5)**: `numpy`, `scikit-learn`, `rich`, `requests`, `pyyaml` added to `pyproject.toml`. `python-dotenv` and `dspy` remain only because tests depend on them; the new harness does not import dspy.

---

## Spec Revision: FMP Calibration Corpora + Public API + Paper-Ready Notebook (2026-04-27)

### Trigger
User in-flight request after task 4.1 complete: (a) source the IS/OOS calibration corpora from FMP news endpoints rather than hand-curated content (avoids subagent fabrication risk; gives auditable, dated, citeable sources), (b) make every harness function importable from a Jupyter notebook with paper-ready visualisations, (c) embed LaTeX-rendered formulas in the notebook so a paper reviewer can verify each statistic by inspection.

### Decisions

- **Calibration data source = FMP news API**, not Wikipedia or hand-authored prose. Endpoints in scope: `fmp-articles`, `news/general-latest`, and optionally `news/stock-latest`. The build script lives at `src/dataset/fmp_corpora.py` (this revives `src/dataset/` as a kept package — task 5.3's deletion list now removes only the legacy `lookahead_loader.py` and `fmp_ingest.py` rather than the whole package).
- **Build-once, update-incrementally**. `build_calibration` produces both files atomically; `update_oos` appends new post-cutoff articles to `oos_control.jsonl` only, never touching `is_memorized.jsonl`. The IS half is frozen by construction (its date window is closed in the past).
- **Notebook + plotting are part of this spec**, not a follow-up. Public API discipline becomes a first-class requirement (Req 12.1). `src/harness/plots.py` consumes the existing dataclasses (`ModelEvalResult`, `ControlBaseline`, `MCSCalibrator`, `CompositeScore`) so the notebook is orchestration-only.
- **Paper-ready figure spec**: vector PDF, single-column width 3.5 inches, font.size 8 pt, 300 DPI, colorblind-safe palette (Wong 2011: `#0072B2 #D55E00 #009E73 #CC79A7 #F0E442`), markers/hatching for B&W reproducibility. These rcParams ship in a `configure_paper_style()` helper.
- **Equation rendering** (Req 12.6) — Markdown/LaTeX cells immediately precede each statistical computation. Twelve formulas listed in the requirement. Author them in the notebook itself, not in `design.md`.

### New dependencies
- `matplotlib >= 3.8` (mandatory for plots).
- `jupyter >= 1.0` (or `ipykernel + nbclient` if a leaner footprint is preferred — pick one in task 5.5; `jupyter` is included for now since the user will run notebooks interactively).
- No new HTTP client — `requests` already declared in task 1.1 covers FMP.

### Boundary impact
- Add to "This Spec Owns": FMP-backed corpora builder, paper-ready plotting helpers, the qualification notebook, and the public API re-export discipline.
- Out of Boundary remains the same (Sharpe / CMMD / portfolio backtesting / live trading).
- Revalidation trigger added: any change to the `harness.plots` figure signatures or to the corpus row schema (`ArticleRecord`) → downstream notebook breakage; bump and re-execute.

### Why FMP is the right call here
- Auditable: every row can be linked back to a real FMP article URL for paper-time citation.
- Date-truthful: FMP's `publishedDate` is the actual publication timestamp, so the IS/OOS partition is real, not asserted.
- Single-vendor: the existing project already uses FMP for `historical-price-eod/light` (eval-set price labels). Reusing the same vendor keeps the credentials and rate-limit story simple.
- Reusable for the eventual macro-indicators eval set: FMP also exposes economic calendar / indicator endpoints, so the harness's input-source-agnostic JSONL contract (Req 2) plus the same FMP fetch utilities will let the macro spec build its eval set with the same plumbing pattern.
