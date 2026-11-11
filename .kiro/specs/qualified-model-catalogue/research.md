# Research & Design Decisions

## Summary
- **Feature**: `qualified-model-catalogue`
- **Discovery Scope**: Extension (light discovery, codebase only, no new dependency)
- **Key Findings**:
  - A provider profile already is "a model on a serving route" and already says
    which evidence modes it offers. 13 profiles ship; all request logprobs; only
    `llamacpp-gemma3-4b` has a full-distribution host. The catalogue can key on
    the profile id.
  - The package persists no calibrator and has no qualification record. The
    only persisted calibrators are the finance simulation's private files. The
    quality rule in code is `holdout_auc >= 0.6` with at least 50 valid draws.
    The certification screen named in `docs/pit-architecture.md` (recall AUC
    with a permutation p-value) has no code in this repository.
  - Both modes already share one feature function and one calibrator type; they
    differ in the number of alternatives kept (about 5 against 100) and in
    where the logprobs come from (sampler against teacher forcing). R43 is the
    only run where the same weights and prompts were scored in both modes.
  - Statistics helpers exist for bootstrap intervals, Holm, paired block
    bootstrap and simulated power. Missing: a label permutation test and a
    paired AUC difference.
  - No test pins the report template, the register layout or the results
    index, so the record can be restructured without breaking a gate. The
    figure tests pin the light and dark file pairs and must change with the
    figure style.

## Research Log

### Where a catalogue entry comes from
- **Context**: the owner keyed the catalogue by model plus serving route.
- **Sources Consulted**: `recall_guard/core/profiles.py` (L55-56, L162-196,
  L276-302, L357-380), `recall_guard/core/provider_profiles/*.json`.
- **Findings**: `ProviderProfile.estimate_sources` returns `top_k` when the
  profile requests logprobs and `full_distribution` when it declares a host;
  `_check_full_distribution` forces the host's digest to equal the declared
  artifact digest, so both modes of one profile score the same weights.
  Declared digests exist for four profiles only. `load_profile` and
  `available_profiles` are not exported at the package root.
- **Implications**: an entry refers to a profile id and derives mode
  availability from the profile, so the two cannot disagree. Routes that were
  measured without a profile (Ryzen AI OGA in R41) need an entry with a route
  description and no profile.

### What a qualification can rest on today
- **Context**: Requirement 1.4 asks for one rule; the blueprint left it open.
- **Sources Consulted**: `recall_guard/mia/mcs.py` L314-383,
  `recall_guard/mia/control.py` L64-77 and L193,
  `recall_guard/harness/guarded_decision.py` L421-481 and L799-819,
  `scripts/finance_sim/calibrate.py` L163-200, L1201-1211, L1440-1477,
  `docs/pit-architecture.md` L37 and L76.
- **Findings**: `is_weak = holdout_auc < min_auc`; the interval is reported
  and not used. Calibrators live in memory on the scorer; the finance
  simulation writes `finance-sim-calibrator/1` records with a fingerprint, a
  prompt-clustered interval and the held-out labels and probabilities.
- **Implications**: the catalogue records evidence, it does not hold
  calibrators. The rule is stated in the design (see Decisions) and computed
  from the held-out labels and probabilities the records already keep.

### Evidence per model already in the register
- **Context**: Requirement 1.7 fills the first catalogue from recorded results.
- **Sources Consulted**: `docs/research-register.md` (R01, R26, R30, R38, R40,
  R41, R43) and the R43 report.
- **Findings**: single-draw logprobs on the EDGAR document corpus: Gemma 3 4B
  0.574, Llama 3.2 3B 0.503, Llama 3.1 8B 0.462 (Ollama, R26); GPT-OSS-20B
  0.574 (Modal, R30); Gemma 3 4B 0.599 and Gemma 4 E4B 0.438 (FastFlowLM, R38).
  Document likelihood from full logits: 0.562 (llama.cpp, R40), 0.470 (OGA,
  R41). Finance simulation, llama.cpp: macro 0.722 (logprobs) and 1.000
  (logits), EDGAR 0.592 and 0.532; Modal vLLM: macro 0.583. R26 to R41 use
  row-level intervals, R43 prompt-clustered ones.
- **Implications**: the catalogue records the interval method beside each
  value. Document likelihood uses another feature set and scores the document,
  not the reply, so it is recorded as evidence of a different method and does
  not count toward the aligned logits mode.

### Statistics that exist and that are missing
- **Sources Consulted**: `recall_guard/core/significance.py` (L144-826),
  `recall_guard/core/bootstrap.py`, `recall_guard/harness/local_reporting.py`
  L89, `scripts/finance_sim/calibrate.py` L1201-1211.
- **Findings**: present: `Interval`, `TestResult`, `holm`, `block_bootstrap`,
  `paired_loss_difference`, `simulated_power`, `bootstrap_ci`,
  `held_out_discrimination`, a prompt-clustered AUC interval (script level).
  Absent: label permutation test, paired AUC difference, agreement statistics.
- **Implications**: three small additions to `core/significance.py`; the
  clustered AUC interval moves there so the catalogue and the simulation use
  one implementation.

### Figures and publication
- **Sources Consulted**: `scripts/finance_sim/figures.py` (L127-188, L294-325,
  L1289-1316), `figures_draws.py`, `figures_convergence.py`, `replay.py`
  (L469, L585-686), `tests/scripts/finance_sim/test_figures*.py`,
  `test_replay.py` L529-551.
- **Findings**: each figure is rendered twice through `_theme_rc(theme)`; file
  names and captions are pinned by tests; the published set is built
  procedurally by the replay, which scans it with the licensing guard.
- **Implications**: one style module replaces the theme pair; the published
  set shrinks by one SVG per figure and gains the new tables and figures; the
  figure and replay tests change with it.

### Skills consulted
- COBESY composition back-end: `blueprint.md` (structure before prose).
- Humanizer: the output conditions become a documentation test and a manual
  pass on each rewritten passage.
- Dataviz: its interactive and colour guidance does not apply to static
  journal figures; its rules on one axis, direct labelling and a table view
  are kept.

## Architecture Pattern Evaluation

| Option | Description | Strengths | Risks / Limitations | Notes |
|--------|-------------|-----------|---------------------|-------|
| Catalogue as a shipped data file plus a typed loader | One JSON in the package, read by `core`, rendered into the docs | One source for code and docs; importable by consumers; reviewable diff | Curated by hand from the register | Selected |
| Catalogue computed at import from calibrator files | Read private calibrator records at runtime | Always current | Private files are not in the wheel; consumers have none | Rejected |
| Catalogue only as a docs page | A Markdown table | Smallest | No selection interface; docs and code drift | Rejected |
| Extend `ProviderProfile` with qualification fields | Put status into each profile JSON | No new file | Mixes how to reach a model with what was shown about it; profile schema is exact-key | Rejected |

## Design Decisions

### Decision: entry key is the profile id
- **Context**: Requirement 1.1; owner's answer "model plus serving route".
- **Alternatives Considered**: weights digest; model family.
- **Selected Approach**: `entry_id` equals the provider profile id when a
  profile exists; entries measured without a profile carry a route description.
  Entries with the same declared digest, or the same curated weights group,
  are shown as related.
- **Trade-offs**: one model appears several times; the related group restores
  the per-model view.

### Decision: one qualification rule with three checks
- **Context**: Requirement 1.4, 5.3, 5.4; the certification screen in
  `docs/pit-architecture.md`.
- **Alternatives Considered**: (1) the current code rule only (AUC at least
  0.6, 50 valid draws); (2) AUC plus permutation p; (3) AUC plus permutation p
  plus the inside-and-after-cutoff check.
- **Selected Approach**: (3). An entry and mode is qualified when its
  calibrator was fitted on a financial time-series recall boundary with at
  least 50 valid draws per class, its held-out AUC is at least 0.6 with a
  prompt-clustered permutation p below 0.05, and its recall estimate is higher
  for analysis dates inside the training range than after the cutoff (one-sided
  test, p below 0.05). A missing check gives "not qualified" with the reason.
- **Rationale**: the first two are the screen the architecture already names;
  the third is what separates recall of the series from recognising the proxy.
- **Trade-offs**: on the recorded data no entry passes the third check, so the
  first catalogue has no qualified entry.
- **Follow-up**: owner confirms the rule at design approval.

### Decision: selection wraps the guard, it does not change it
- **Context**: Requirements 2.3 to 2.6; steering invariant that `p_memorized`
  stays bit for bit identical to the batch harness.
- **Selected Approach**: a new harness module returns a `SelectedDecision`
  around the unchanged `GuardedDecision`. The correction is exposed only when
  the entry and mode is qualified and the supplied calibrator's fingerprint
  matches the catalogue's.
- **Trade-offs**: `macro_framework` keeps calling the guard as before; adopting
  the selection interface there is out of boundary.

### Decision: statistics once, in one methods page
- **Context**: Requirement 6 and 7.7 (each fact once).
- **Selected Approach**: `docs/research-methods.md` defines every statistic
  with its formula and one worked example; reports name the statistic and link
  to it, and each table carries the counts needed to recompute its values.

### Decision: analyses and figures run inside the guarded replay
- **Context**: Requirement 10.7.
- **Selected Approach**: the mode comparison, the cutoff check and the repeat
  baseline are new analysis modules of `scripts/finance_sim`, written and
  published by `replay()` like the existing ones.

### Synthesis outcomes
- **Generalization**: catalogue table, "routes without logprobs" list and
  "related entries" are three views of one record; one loader with filters.
- **Build vs adopt**: permutation and paired bootstrap are a few lines on top
  of existing helpers; no new dependency (scikit-learn's `roc_auc_score` and
  numpy are already used).
- **Simplification**: no calibrator store, no change to `GuardedDecision`, no
  change to profiles, no new docs generator beyond one render script.

## Risks & Mitigations
- The first catalogue shows no qualified entry: stated in the summary of the
  synthesis as the result, with what would change it.
- Small samples (5 results per cutoff cell on the probe arm): every test
  reports its detectable effect; "cannot decide" is a permitted answer (4.6).
- Restyled figures change published assets: done through the replay with a
  byte comparison, and the old pairs are removed in the same commit.
- Curated catalogue drifts from the register: a test checks that every
  evidence id exists in the register and every value appears there.

## References
- `docs/pit-architecture.md`: the point-in-time stack and the certification
  screen.
- `.kiro/steering/recall-guard-package.md`: public API invariants and layering.
- `.kiro/steering/research-reports.md`: reporting rules to be updated.
- `.kiro/specs/qualified-model-catalogue/blueprint.md`: composition blueprint.
