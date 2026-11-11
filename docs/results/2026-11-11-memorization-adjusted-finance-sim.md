# Memorization-adjusted finance simulation, probe arm on Gemma 3 4B (draft, serving arm pending), 2026-11-11

> Register: R43 · Status: measured · Track: Local

> **Draft, serving arm pending.** This page reports the local probe arm only.
> The Modal serving arm carries the pre-registered tests and runs only after the
> user's approval record (decision 2026-11-10); its results, the achieved power
> and the final register state are added when it has run.

On the probe arm the gap check is empty: identifying and anonymised macro
prompts were equally accurate before and after the model's cutoff on the 9
usable months (gap 0, p 1.0), while every adjusted variant lowered Brier and
log loss against no adjustment (Holm p 0.004 each), which is what shrinking an
over-confident forecast toward 0.5 does whenever raw accuracy is near chance
(consensus accuracy 0.375, random-answerer p 0.90). The ensemble reduced the
run-to-run variance of the top-k adjusted summary (ratio 0.72, Holm p 0.003);
the macro full-distribution calibrator separated the two renderings outright
(held-out AUC 1.00), so it detects the rendering rather than memorization as
such, and both EDGAR calibrators are weak (0.59 and 0.53), so the negative
control reads "no material change" by construction. The estimate from the
host's top-k evidence converges to the full-distribution value as k grows (mean
absolute difference 0.53 at k 1, 0.11 at k 5, 0.03 at k 50), the power gate
stopped for decision at expected power 0.11 as pre-registered, and the arm
proceeded under the frozen inform policy.

## Question

Does adjusting a Gemma 3 4B monthly macro-regime decision by a memorization
estimate (from the host's top-k logprobs or from the full next-token
distribution) remove the pre-cutoff advantage of identifying prompts over
anonymised ones without discounting honest answers on held-out releases; does an
8-draw ensemble reduce the run-to-run variance of the decision and of the
estimate; and how fast does the top-k estimate converge to the full-distribution
estimate as k grows? The probe arm answers the paired top-k versus
full-distribution questions and gives a small paired check of the gap; the
pre-registered tests belong to the serving arm.

## Setup

| Field | Value |
| --- | --- |
| Model | `google/gemma-3-4b-it`, GGUF q4_K_M, digest `sha256:f06e91e18fc74295d0c781bf91622508c75655e2c18ed869d52447ffe1e667e1` (llama-server name `gemma3:4b`); the serving arm pins HF revision `093f9f388b31de276ce2de164bdc2081324b9767` |
| Runtime | llama.cpp b12199, Vulkan, Radeon 890M iGPU, Windows 11; llama-server for generation (top-5 logprobs), the project's full-distribution host (`scripts/serve_full_distribution.py`, same GGUF) for teacher-forced evidence; one host loaded at a time behind bearer gateways on 127.0.0.1:11571 and 11581 |
| Frozen plan | `plan_sha256` `f7603eeb8efdbf74645d658f2bd733a35b3834ba7c90b74d00c6ac2e6b5332fe`, seed 20261110, created 2026-11-10T21:35:42+01:00, status frozen; every recording, calibrator, gate record and analysis carries this hash |
| Arms and roles | sampling arm, probe role (this report): six variants, top-k versus full-distribution comparison, convergence, a small paired check of the gap; serving arm, primary role (Modal vLLM, pending approval): none and top-k variants, the pre-registered gap, Brier and log-loss tests and the negative control, 68 pre-cutoff and 22 post-cutoff months, frozen limit USD 32.86 |
| Generation | temperature 0.7, 8 draws per rendering (at least 6 parsed), seed 1000 plus draw index, 512 output tokens, 300 s timeout, 2048-token context |
| Evidence | one `evidence_batch` per prompt and rendering over its parsed replies, 100 kept alternatives of the 262,145-token vocabulary plus the exact actual-token logprob, 600 s timeout, `host_busy` retried 5 times 60 s apart |
| Calibrators | one per estimate source and corpus; holdout 25 percent by prompt; `min_auc` 0.6, `min_valid` 50; held-out AUC with a prompt-clustered stratified percentile bootstrap, 1000 resamples, 95 percent |
| Statistics | non-circular moving-block percentile bootstrap, 2000 resamples, seed 20261110, 95 percent; block length 6 months (2 filing quarters for the negative control); Holm per metric family; pre-registered effect 0.10; negative-control tolerance 0.10; power threshold 0.8 |
| Route | native `uv run --no-sync` on Windows (Compose deferred: the dev image bakes the code in and the container route to the native gateways is unverified); API keys loaded by name for the plan build only; no paid request |

Probe-arm sizing (user decision 2026-11-10): calibration on 24 months with 8
draws per rendering plus 140 EDGAR rows with 4 draws (944 requests); pilot 12
months (192); evaluation on a seeded subset of 20 months, 15 on or before the
cutoff and 5 after (320); negative control 40 releases (320). Every stage ran
exactly its frozen request count.

| Stage | Requests | Output tokens | Generation | Evidence batches | Evidence | Host time | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Calibration, attempt 1 (2026-11-10 22:45) | 944 | 33,368 | 71 min (4,255 s) | refused | n/a | 71 min | failed: every draw `parse_error` (fenced JSON) |
| Calibration, attempt 2 (2026-11-11 00:12) | 944 | 33,368 | 56 min | 181 prompts (183 posts) | 108 min | 2 h 44 min | complete |
| Pilot (02:58) | 192 | 12,731 | 15 min | 23 | 19 min | 34 min | complete |
| Full run (03:34) | 320 | 21,309 | 24 min | 33 | 27 min | 51 min | complete |
| Negative control (04:26) | 320 | 4,576 | 9 min | 40 | 21 min | 30 min | complete |

Host time about 4 h 40 min in all plus the 71-minute failed attempt; no stage
needed a restart; host teardown was verified after every stage (no process, no
listener on 11570, 11571, 11580 or 11581). Attempt 1 failed because Gemma 3 4B
wraps its JSON object in a markdown code fence and the strict parser refused
every reply; both attempts used the same per-draw seeds and recorded the same
token count. Fix 9.2a lets `parse_json_components` unwrap exactly one
fence and leaves the prompts and every other strictness rule unchanged. The
failed attempt's records are kept privately; the stage was re-run from scratch.

## Data

- FRED (Federal Reserve Bank of St. Louis), point in time: CPIAUCSL, T10Y2Y
  and BAA10Y as known on each decision date (the last calendar day of each
  month from 2016-01 to 2026-06, 126 months), z-scores over the trailing 60
  months; DTB3 (current vintage) is the cash leg of the outcome and never an
  input. BAA10Y is Moody's Seasoned Baa Corporate Bond Yield Relative to Yield
  on 10-Year Treasury Constant Maturity, copyright Moody's, cited and not
  reproduced. Month split: calibration 24, pilot 12, evaluation 90 (68 before
  the model's cutoff, 22 after), the probe arm's seeded evaluation subset 20
  (15, 5); 0 months excluded for a missing realized return.
- Financial Modeling Prep (FMP): dividend-adjusted end-of-day closes for SWDA.L,
  XLK and IAU; the outcome is whether the defensive pair beats the risk pair over
  the next month. Licensed; no close, return or level is published.
- SEC EDGAR 8-K press releases (EX-99.1, public): calibration 40 memorized rows
  (filed before the cutoff) and 100 control rows (after); negative control 40
  held-out control releases over 9 filing quarters (2024Q3 to 2026Q3); the 40
  held-out memorized rows are unused.
- Publication rule (decision 2026-11-11): this page and the published tables
  under `docs/assets/finance-sim/` carry derived statistics. The draws table
  (`draws.md`, `draws.csv`) publishes, per probe-month item id and rendering,
  the derived first-draw statistics (mid-rank, band, category, off flag), the
  missing-item rows with their reasons, and the example item's per-draw summary
  values and estimates, exactly as the design's F3 worked example specifies;
  the serving-arm pre-registered markdown (`pre-registered-serving-full_run.md`)
  lists each planned month id and rendering with its missing reason and no
  value; every other table carries aggregates. No published file carries a reply text,
  an outcome, a licensed value (a BAA10Y level, an FMP close, a DTB3 rate or a
  realized return) or an EDGAR per-release row.

## Results

### Parse rate per stage and corpus

| Stage | Corpus | Draws | Parsed | Failed share |
| --- | --- | ---: | ---: | ---: |
| Calibration | macro (24 months x 2 renderings x 8) | 384 | 277 (132 identifying, 145 anonymised) | 28 percent |
| Calibration | EDGAR (140 x 4) | 560 | 560 | 0 |
| Pilot | macro (12 x 2 x 8) | 192 | 138 | 28 percent |
| Full run | macro (20 x 2 x 8) | 320 | 186 | 42 percent |
| Negative control | EDGAR (40 x 8) | 320 | 320 | 0 |

After the fence fix the remaining failures are replies outside the strict
schema (a loading out of bounds or an extra key) and they fall on the macro
prompts only. Pilot: 7 of 12 months unusable (6 with a rendering below the
minimum of 6 parsed draws, 1 whose anonymised rendering had an unresolved
component), 0 lost to interruptions. Full run: 7 of 40 renderings had no parsed
draw; the single-draw variants lost 11 of 20 months (draw 0 failed in at least
one rendering) and the ensemble variants 15 of 20 (fewer than 6 parsed draws in
at least one rendering).

### Calibrator quality

| Corpus | Source | Held-out AUC | 95 percent interval, prompt-clustered | Draw-level interval (diagnostic) | ECE | Held-out draws (memorized, control) | Held-out prompts | Weak |
| --- | --- | ---: | --- | --- | ---: | --- | ---: | --- |
| macro | top_k | 0.722 | 0.586 to 0.855 | 0.595 to 0.830 | 0.070 | 74 (37, 37) | 11 (6, 5) | no |
| macro | full_distribution | 1.000 | 1.000 to 1.000 | 1.000 to 1.000 | 0.048 | 74 (37, 37) | 11 (6, 5) | no |
| EDGAR | top_k | 0.592 | 0.367 to 0.815 | 0.479 to 0.703 | 0.200 | 140 (40, 100) | 35 (10, 25) | yes |
| EDGAR | full_distribution | 0.532 | 0.319 to 0.754 | 0.419 to 0.645 | 0.206 | 140 (40, 100) | 35 (10, 25) | yes |

Valid draws entering each fit (holdout included): macro 132 identifying and 145
anonymised of 192 each; EDGAR 160 memorized and 400 control of 160 and 400.
Labels: for the macro calibrator the identifying rendering is the "memorized"
class and the anonymised rendering the control, so an AUC of 1.00 means the
full-distribution features (loss, Min-K, Min-K%++, zlib ratio over the kept
top-100 alternatives) tell the two renderings apart on every held-out pair. ECE
is the count-weighted mean absolute difference between mean predicted
p_memorized and the observed memorized share over 10 equal-width bins. A weak
calibrator (AUC below 0.6) is not applied by the guard.

### Pilot and power gate (as pre-registered)

| Field | Value |
| --- | --- |
| Pre-registered effect, alpha, threshold | 0.10, 0.05, 0.8 |
| Gated arm's planned split | 15 pre-cutoff and 5 post-cutoff months |
| Noise variant and statistic | `ensemble_top_k`, identifying minus anonymised correctness per month |
| Pilot months used | 5 of 12 (7 missing, see above; 0 lost to interruption) |
| Noise sd | 0.447 (sample sd, ddof 1, over 5 months) |
| Expected power | 0.111 (0.1105; one-sided pooled two-sample t test simulated 2000 times, Monte Carlo standard error about 0.01) |
| Decision | `stop_for_decision`: 0.111 is below 0.8 at the frozen sizes |
| Policy | `pilot.gate_policy = "inform"` (frozen): the probe arm proceeded after the stop |

At 15 and 5 months (effect 0.10, one-sided alpha 0.05, noncentral t) power 0.8
needs a noise sd of at most about 0.075. The smallest non-zero sd of
month-level accuracy differences in {-1, 0, 1} is 0.289 over 12 months and
0.447 over the 5 usable months (the gate record's 0.1105 is reproduced at sd
0.447), so the gate can pass only if every difference is equal, which is why
the plan pre-registered it as informative. The plan's `pilot.rule` text still
says the full run starts only at power at or above the threshold; the frozen
`gate_policy` governs and that rule text is stale.

### Pre-registered analysis on the probe arm (a small paired check)

Planned months 20 (15 pre-cutoff, 5 post); pilot items excluded 0; a month
enters a statistic only when both renderings carry the value. Intervals are the
95 percent moving-block bootstrap over months (block length 6, 2000 resamples,
seed 20261110), 0 resamples dropped in every computed cell.

| Variant | DiD accuracy gap | 95 percent interval | p raw | p Holm | Months (pre, post) |
| --- | ---: | --- | ---: | ---: | --- |
| single_none | 0 | 0 to 0.333 | 1 | 1 | 9 (4, 5) |
| single_top_k | 0 | 0 to 0.333 | 1 | 1 | 9 (4, 5) |
| single_full_distribution | 0 | 0 to 0.333 | 1 | 1 | 9 (4, 5) |
| ensemble_none | missing | | | | 5 (4, 1): below the block length of 6 |
| ensemble_top_k | missing | | | | 5 (4, 1): below the block length of 6 |
| ensemble_full_distribution | missing | | | | 5 (4, 1): below the block length of 6 |

The identifying minus anonymised accuracy difference is 0 in both periods for
all three single variants (pre 0 to 0; post -0.333 to 0).

| Variant (against the same draw mode's no adjustment) | Brier difference | 95 percent interval | Log-loss difference | 95 percent interval | p raw | p Holm | Months |
| --- | ---: | --- | ---: | --- | ---: | ---: | ---: |
| single_top_k | -0.257 | -0.267 to -0.141 | -2.039 | -2.073 to -1.249 | 0.001 | 0.004 | 9 |
| single_full_distribution | -0.205 | -0.260 to -0.117 | -1.481 | -1.739 to -0.912 | 0.001 | 0.004 | 9 |
| ensemble_top_k | -0.206 | -0.226 to -0.115 | -1.558 | -1.729 to -0.946 | 0.001 | 0.004 | 9 |
| ensemble_full_distribution | -0.192 | -0.196 to -0.080 | -1.367 | -1.377 to -0.735 | 0.001 | 0.004 | 9 |

| Variant | Estimate difference pre-cutoff | 95 percent interval | Estimate difference post-cutoff | 95 percent interval |
| --- | ---: | --- | ---: | --- |
| single_top_k | 0.222 | 0.118 to 0.275 | 0.053 | -0.048 to 0.053 |
| single_full_distribution | 0.734 | 0.517 to 0.765 | 0.618 | 0.509 to 0.783 |
| ensemble_top_k | 0.068 | 0.053 to 0.098 | 0.094 | 0.076 to 0.105 |
| ensemble_full_distribution | 0.624 | 0.495 to 0.639 | 0.554 | 0.554 to 0.769 |

- The gap check carries no evidence either way: accuracy was identical across
  renderings in both periods on 9 months, and the ensemble cells are undefined
  below the block length.
- All four adjusted variants lower Brier and log loss by about 0.2 and 1.4 to
  2.0. The scored probability is the share of draws with a positive defensive
  spread (0 or 1 for a single draw), clipped to [0.01, 0.99], and the
  adjustment shrinks it toward 0.5 by 1 - p_memorized. With raw accuracy near
  chance, any shrinkage toward 0.5 lowers these losses; the result says the raw
  forecasts are over-confident, not that the adjustment adds skill.
- The estimate difference (identifying minus anonymised p_memorized) is 0.55 to
  0.73 for the full-distribution variants in both periods and 0.05 to 0.22 for
  top-k: the full-distribution calibrator assigns a high p_memorized to the
  identifying rendering whenever it sees it, before and after the cutoff.
- Achieved power is missing: fewer than two months carry the noise statistic
  after the cutoff (15 months without it), so no pooled sd exists.

### Ensemble metrics (40 items analysed per stage)

| Stage | Metric | Cell | n | Statistic | 95 percent interval | p | Holm p |
| --- | --- | --- | ---: | ---: | --- | ---: | ---: |
| full run | half-ensemble over single-draw variance, summary | none | 20 | 0.994 | 0.784 to 4.037 | 0.441 | 0.441 |
| full run | | top_k | 20 | 0.715 | 0.317 to 0.830 | 0.001 | 0.003 |
| full run | | full_distribution | 20 | 0.781 | 0.276 to 2.222 | 0.169 | 0.338 |
| full run | half-ensemble over single-draw variance, p_memorized | top_k | 20 | 0.511 | 0.242 to 1.124 | 0.048 | 0.088 |
| full run | | full_distribution | 20 | 0.299 | 0.163 to 1.840 | 0.044 | 0.088 |
| full run | first-draw rate (off the consensus or outside the band 0.25 to 0.75) | all items | 17 | 0.412 | 0.216 to 0.640 (Wilson) | | |
| full run | | identifying | 9 | 0.222 | 0.063 to 0.547 | | |
| full run | | anonymised | 8 | 0.625 | 0.306 to 0.863 | | |
| full run | random-answerer null | consensus accuracy | 16 | 0.375 | null central range 0.25 to 0.75 | 0.902 | |
| negative control | half-ensemble over single-draw variance, summary | none, top_k, full_distribution | 40 | undefined | | | |
| negative control | half-ensemble over single-draw variance, p_memorized | top_k | 40 | 0.124 | 0.073 to 0.207 | 0.0005 | 0.001 |
| negative control | | full_distribution | 40 | 0.409 | 0.064 to 1.046 | 0.032 | 0.032 |
| negative control | first-draw rate | all items (outlook) | 40 | 0 | 0 to 0.088 (Wilson) | | |
| negative control | random-answerer null | consensus accuracy | 0 | not applicable | | | |

Halves are the first and second 4 draws of each item's one recorded ensemble
(3 parsed needed), singles are draws 0 and 4; Holm over m = 3 summary cells and
m = 2 p_memorized cells on the full run (on the negative control no summary cell
is computed and m = 2 p_memorized cells);
0 resamples dropped or unbounded in every cell; the probability of a zero
single-draw variance per resample is at most 9.1e-13 on the full run. On the
negative control the two single draws agree on every one of the 40 releases, so
the single-draw variance of the summary is zero and its ratios are undefined;
an EDGAR outlook has no realized outcome, so the random-answerer null is not
applicable there. Full run: all 7 off first draws are outside the band, none
differs in category; the exchangeability baseline (the share expected off if
the first draw were exchangeable with the other draws) is 0.411 against the
observed 0.412; 1 of the 17 rated items has a consensus without a category and
enters because its first draw lies outside the band; 23 of 40 items have no
decided first draw: 19 whose first draw did not parse, 1 without a consensus (5
of 8 draws parsed) and 3 whose consensus has no category with the first draw
inside the band; 0 items have halves listed as unresolved.

### Negative control (40 held-out EDGAR releases, 9 filing quarters)

| Variant | Outcome | Releases | Mean absolute change | 95 percent interval | Signed mean | Sign flips | Guard statuses |
| --- | --- | ---: | ---: | --- | ---: | ---: | --- |
| single_top_k | no material change | 40 | 0 | 0 to 0 | 0 | 0 | adjustment_skipped_weak_calibrator 40 |
| single_full_distribution | no material change | 40 | 0 | 0 to 0 | 0 | 0 | adjustment_skipped_weak_calibrator 40 |
| ensemble_top_k | no material change | 40 | 0 | 0 to 0 | 0 | 0 | adjustment_skipped_weak_calibrator 40 |
| ensemble_full_distribution | no material change | 40 | 0 | 0 to 0 | 0 | 0 | adjustment_skipped_weak_calibrator 40 |

Tolerance 0.1; interval over filing quarters with block length 2. Both EDGAR
calibrators are weak, so the guard skipped the adjustment on 40 of 40 releases
and the change is 0 by construction. The negative control says nothing about
whether a working calibrator would discount honest answers.

### Top-k versus full-distribution convergence

Full run: 40 items, 186 parsed draws; negative control: 40 items, 320 draws.
The kept 100 alternatives cover a mean mass of 0.99998 (minimum 0.990) and
0.99999 (minimum 0.9992). Rows truncate each draw's full-distribution evidence
to its first k alternatives and keep the full-distribution calibrator; the
served row is the generation host's own top-5 logprobs through the top-k
calibrator.

| k | Estimate, mean abs diff (full run) | max | Mass mean (min), full run | Estimate, mean abs diff (negative control) | Mass mean (min), negative control | Min-K%++, mean abs diff (full run) |
| ---: | ---: | ---: | --- | ---: | --- | ---: |
| 1 | 0.531 | 1.000 | 0.946 (0.123) | 0.499 | 0.978 (0.395) | 2.2e6 |
| 5 | 0.106 | 0.780 | 0.992 (0.505) | 0.262 | 0.9994 (0.933) | 5.68 |
| 10 | 0.088 | 0.647 | 0.9992 (0.895) | 0.229 | 0.9999 (0.994) | 3.78 |
| 20 | 0.053 | 0.188 | 0.99992 (0.982) | 0.211 | 0.99993 (0.997) | 2.08 |
| 50 | 0.026 | 0.092 | 0.99997 (0.987) | 0.121 | 0.99998 (0.9986) | 1.01 |
| 100 (kept) | 0 | 0 | 0.99998 (0.990) | 0 | 0.99999 (0.9992) | 0 |
| served top-5, top-k calibrator | 0.361 | 0.826 | 0.9993 (0.698) | 0.063 | 0.99999 (0.9955) | 2.00 |

- Loss, Min-K and the zlib ratio are exact at every k in both stages (mean and
  maximum absolute difference 0, every draw within 1e-6): they use only the
  actual token's logprob, which the evidence carries exactly at any k.
- Min-K%++ never agrees below the kept count: it needs the moments of the kept
  alternatives, so truncation changes it at every k below 100.
- The estimate converges as k grows: 0.53 at k 1, 0.11 at k 5, 0.03 at k 50,
  identical at the kept 100 by construction. At k 1 the adjusted decision
  category changed for 0.875 of ensemble items and 0.762 of single items in
  the full run; from k 5 upward no category changed.
- The served row is different in kind: its loss differs from the
  full-distribution loss by a mean of 0.46 (maximum 0.79) in the full run and
  1.28 in the negative control while every truncation row is exact, so the
  host's own top-5 logprobs are not a truncation of the teacher-forced
  distribution.
- Values compare only within this model and these weights.

### Figures

The four figures were rendered offline by the replay
(`python -m scripts.finance_sim.replay --require-guard`, with fix 9.3a, commit
6d9927d) from the stored analyses into `docs/assets/finance-sim/` as light and
dark SVGs, each with a CSV and a markdown table (`<name>.csv`, `<name>.md`)
carrying the full caption and every plotted number. The run passed with the
guard required: 26 files scanned, 12,836 probes (BAA10Y, CPI and T10Y2Y raw
levels, DTB3 rates, FMP adjusted closes, realized returns, reply text), every
required body probed (items, 3 FMP bodies for 3 priced symbols, 1 DTB3 body), 0
licensed leaks, and 10 advisory DTB3 matches in 6 files (gap light and dark 1
each, reliability light and dark 2 each, reliability.csv 2, reliability.md 2),
which are public FRED rates coinciding with numbers inside SVG style attributes,
glyph transform scale factors and the reliability bin edges. After a fix to the
F4 direct labels (commit 724c966) a second guarded run re-rendered the figures
with the same guard result (26 files, 0 leaks, the same 10 advisory matches);
only the two convergence SVGs changed, and the `--compare` run against the copy
in the repository passed byte for byte. Identity is carried by marker shape (circle no adjustment, square
top-k, triangle full distribution) and fill (open single draw, filled ensemble),
not by colour alone.

| File | sha256 |
| --- | --- |
| `gap-light.svg` | `2055e92354752bac91ab4d29906068aa43f9847b79feff761eb37338b936acbc` |
| `gap-dark.svg` | `1d3a1fa2fa7df642c76038f7eae8fe924ea5436273a5b245b94921b1d0dfc80c` |
| `reliability-light.svg` | `dbfb4ec9d03541ad13eccb67a1281724085f4d3a9abcf4a2a0df6769395ce725` |
| `reliability-dark.svg` | `a90df13efdb6f5f87eae4386c7467b6b6f99d4d876297cd2c80bf797f8a0f570` |
| `draws-light.svg` | `5a5f00281d2fe8170b3bfac67d250035ce537ae9dca9ed946319944b16ceb7cf` |
| `draws-dark.svg` | `9e49fa3f9c1099fece4b9c86c860778b94e3e9ef314fe8ea751f04bec2061156` |
| `convergence-light.svg` | `b48d0732bf253a43d185c8954d2e32ec7d09f4dc04e42caa59833cf3b17b7b58` |
| `convergence-dark.svg` | `80dc95bb9bbbfe9f1d1f998a02e5ea2b014145db9d3ddc05eb9ce4ee75c704d7` |
| `replay.json` (manifest; private copy `runs/finance-sim/replay/replay-20261111d.json`) | `987ad8f9b0c0d36e84163b6d2978cfaedf87b8f0fdcdb32ff4d95c6a9d532430` |

### F1 Accuracy gap before and after adjustment

![Figure F1. Pre-cutoff minus post-cutoff difference of identifying minus anonymised accuracy per variant, one panel per arm (light theme)](../assets/finance-sim/gap-light.svg#only-light)
![Figure F1. Pre-cutoff minus post-cutoff difference of identifying minus anonymised accuracy per variant, one panel per arm (dark theme)](../assets/finance-sim/gap-dark.svg#only-dark)

Data: the stored pre-registered analysis of the full_run stage (read, not
recomputed), pilot items excluded; per variant the DiD accuracy gap with the
no-adjustment variants beside the top-k and full-distribution adjustments; the
sampling arm (probe role) with 20 planned months (15, 5). The serving-arm panel
is empty because that arm has not run: its rows read "no planned month carries
the value in both renderings" and its full-distribution variants are missing by
design (different weights, Requirement 9.4). Sample size: 9 months for the
single variants, the ensemble variants not drawn (5 months, below the block
length). Interval: 95 percent non-circular moving-block percentile bootstrap
over months, block length 6, 2000 resamples, seed 20261110; p raw two-sided
from the kept resamples, Holm-adjusted across the variants of each arm;
achieved power missing. Forward item: `gap.md` gives the serving arm the
achieved-power reason "fewer than two months carry the noise statistic", which
is misleading for an arm that has not run. Sources: macro rows from FRED
including BAA10Y (Moody's, cited and not reproduced); realized returns from FMP
adjusted closes (licensed, not published) with DTB3 as the cash leg. Derived
statistics only.

### F2 Reliability of the memorization estimate

![Figure F2. Reliability diagram of p_memorized on held-out calibration draws, one panel per corpus and estimate source (light theme)](../assets/finance-sim/reliability-light.svg#only-light)
![Figure F2. Reliability diagram of p_memorized on held-out calibration draws, one panel per corpus and estimate source (dark theme)](../assets/finance-sim/reliability-dark.svg#only-dark)

Data: the held-out draws each calibrator retained when it was fitted (a
stratified 25 percent split by prompt) against their labels, 1 memorized or
identifying and 0 control or anonymised; one calibrator per corpus and source
serves both draw modes. Sample size: macro 74 held-out draws (37, 37) from 11
prompts, EDGAR 140 (40, 100) from 35 prompts; the draws of one prompt are not
independent. Bins: 10 equal-width bins, empty bins omitted, marker area by
count; ECE 0.070 (macro top-k), 0.048 (macro full distribution), 0.200 (EDGAR
top-k) and 0.206 (EDGAR full distribution). Interval: none on the bins; the
held-out AUC carries the prompt-clustered 95 percent interval of the calibrator
table; the two EDGAR calibrators are marked weak. The serving arm's panels are
not drawn: its calibrators are not fitted. Sources: FRED macro rows including
BAA10Y (Moody's, cited and not reproduced); SEC EDGAR outlook prompts; the
study's outcomes come from FMP adjusted closes (licensed, not published) and no
FMP value enters this figure. Derived statistics only.

### F3 Per-draw distribution against the ensemble result

![Figure F3. Per-draw summary value and p_memorized of one example item with the ensemble decisions, the first draw marked, and every item's first-draw mid-rank against the band (light theme)](../assets/finance-sim/draws-light.svg#only-light)
![Figure F3. Per-draw summary value and p_memorized of one example item with the ensemble decisions, the first draw marked, and every item's first-draw mid-rank against the band (dark theme)](../assets/finance-sim/draws-dark.svg#only-dark)

Data: the stored item results of the probe arm's full_run stage and the stored
ensemble metrics (read, not recomputed); panels 1 and 2 show the example item
macro-2016-05 (identifying rendering; the first item in plan order with a
decided first draw and a consensus summary value): the defensive spread of each
of its 8 draws (8 parsed, 0 failed) and its p_memorized per estimate source,
with the stored ensemble decisions; panels 3 and 4 show every analysed item's
first-draw mid-rank against the frozen band 0.25 to 0.75 and the stored
first-draw rate 0.412 (17 items; 9 identifying, 8 anonymised) against the
exchangeability baseline 0.411. Items without a consensus category: 1 of 17,
included because its first draw lies outside the band; 23 items are missing with
their reasons in the table; 0 items have halves listed as unresolved. Interval:
none on the per-draw values; the shaded band is the interquartile range of the
parsed draws with the median marked; the rate carries its Wilson interval at
95 percent. Halves of 4 draws never run the consensus split check, so an
unmarked item is not shown as resolved. Sources: FRED macro rows including
BAA10Y (Moody's, cited and not reproduced); the study's outcomes come from FMP
adjusted closes (licensed, not published) and no FMP value enters this figure.
Derived statistics only.

### F4 Top-k versus full-distribution agreement as k grows

![Figure F4. Mean and maximum absolute difference of each feature and of the estimate between top-k truncated and full-distribution evidence, and the mass the first k alternatives cover, by stage (light theme)](../assets/finance-sim/convergence-light.svg#only-light)
![Figure F4. Mean and maximum absolute difference of each feature and of the estimate between top-k truncated and full-distribution evidence, and the mass the first k alternatives cover, by stage (dark theme)](../assets/finance-sim/convergence-dark.svg#only-dark)

Data: the stored convergence tables of the probe arm (read, not recomputed),
k in {1, 5, 10, 20, 50, 100}; the served row (the host's top-5 logprobs through
the top-k calibrator, a different calibrator) is a separate open marker at k 5,
not joined to the lines. Sample size: full run 186 draws of 40 items, negative
control 320 draws of 40 items, 100 kept alternatives of a 262,145-token
vocabulary covering a mean mass of 0.99998 and 0.99999. Interval: none; the
tables give the mean and maximum absolute difference and the share within 1e-6.
Values compare only within one model and one set of weights. Sources: FRED
macro rows including BAA10Y (Moody's, cited and not reproduced); SEC EDGAR
outlook prompts; no FMP value enters this figure. Derived statistics only.

## Limits

- Parse failures: 28 percent of macro draws in calibration and the pilot and 42
  percent in the full run did not parse after the fence fix (bounds or extra
  keys), 0 percent on the EDGAR prompts. They cost 7 of 12 pilot months, 11 of
  20 full-run months for the single variants and 15 of 20 for the ensemble
  variants, and they leave 23 of 40 items without a decided first draw. The
  usable sample is conditional on the draws that parsed.
- The probe arm's gap check rests on 9 months (single variants) and the ensemble
  cells are undefined; it is a small paired check, not the pre-registered test,
  and its block bootstrap with 9 months and block length 6 is coarse.
- The gate behaved as pre-registered: at 15 and 5 months and effect 0.10 it
  passes only if every month-level difference is equal (a noise sd of at most
  about 0.075 is needed and the smallest non-zero sd is 0.289 over 12 months),
  otherwise it can only stop; the frozen inform policy let the probe arm
  proceed at expected power 0.11; the plan's `pilot.rule` text is stale.
- The serving arm is pending. The pre-registered tests (none and top-k variants,
  68 and 22 months), the achieved power and the paid negative control come with
  it; nothing here is confirmatory.
- The macro full-distribution calibrator detects the rendering: its held-out AUC
  of 1.00 and estimate differences of 0.55 to 0.73 in both periods show that the
  identifying prompt is recognised as such, not that the month was memorized.
  The loss improvements of every adjusted variant follow from shrinking
  over-confident forecasts toward 0.5 at near-chance accuracy.
- The negative control is uninformative: both EDGAR calibrators are weak and
  the adjustment was skipped 40 of 40 times.
- Pooled items are not independent: the two renderings of a month share one
  realized outcome, so the pooled first-draw rate and the random-answerer null
  overstate the evidence; the per-rendering rates stand beside the pooled one
  (0.22 identifying, 0.63 anonymised on 9 and 8 items). Ensemble intervals
  assume independent items and draws, and the unadjusted summary ratio 0.99
  (0.78 to 4.04) is no reduction in the raw decision; the negative-control
  interval over 9 quarters with block length 2 may undercover.
- The served row compares a different calibrator (top-k, fitted on the host's
  top-5 logprobs) with the full-distribution one, so its disagreement mixes the
  calibrator difference with the evidence difference: its loss differs from the
  full-distribution loss by a mean of 0.46 while every truncation row is exact.
  Whether llama-server's top logprobs are pre-temperature was not confirmed in
  this run.
- The first replay was refused by the licensing guard on 10 DTB3 probe matches
  in the gap and reliability outputs: three distinct runs of short public DTB3
  rates coinciding with numbers inside SVG style attributes, with glyph
  transform scale factors (font sizes) and with the reliability diagram's bin
  edges in the CSV and markdown. No BAA10Y, realized-return, FMP close or reply
  probe matched. Fix 9.3a (commit 6d9927d) records DTB3 matches as advisory
  warnings in replay.json and keeps every licensed refusal; the re-run passed
  with the guard required, 0 leaks and the 10 advisory matches recorded, and
  the `--compare` run confirmed the repository copy byte for byte.
- The runner ran natively on Windows rather than through Compose; the statistics
  do not depend on where the runner process lives, and the container route to
  the native gateways remains to be verified. The attempt 1 receipt says
  complete although its evidence pass was refused (a runner follow-up); the
  successful receipts are corroborated by the calibrator files, the stored
  results and the analyses. The receipts' `evidence_batches` counter stays 0 on
  the batch path (cosmetic; the batches are in the gateway log and the evidence
  files).

## Evidence

- Plan: `runs/finance-sim/plan/receipt.json` (status frozen, `plan_sha256`
  `f7603eeb8efdbf74645d658f2bd733a35b3834ba7c90b74d00c6ac2e6b5332fe`, created
  2026-11-10T20:35:42+00:00); build log
  `runs/finance-sim/validation/live-9.2-build-plan.log` (redacted).
- Stage receipts and ledgers, all private:
  `runs/finance-sim/records/sampling/{calibration,pilot,full_run,negative_control}/receipt.json`
  (status complete, exit code 0, two sessions each, both host checks ok against
  the pinned digest, `teardown_verified` true) and `budget.json` (status
  complete, requests at the frozen maximum, 0 unknown outcomes). Attempt 1:
  `runs/finance-sim/records/sampling/calibration.attempt1-parse-error-20261111-0001/`
  (`budget.json`: 944 requests, 33,368 output tokens, 4,254.8 s elapsed).
- Gate record:
  `runs/finance-sim/gates/f7603eeb8efdbf74645d658f2bd733a35b3834ba7c90b74d00c6ac2e6b5332fe.json`
  (schema `finance-sim-power-gate/1`, decision `stop_for_decision`).
- Calibrators: `runs/finance-sim/calibrators/sampling/{top_k,full_distribution}/{macro,edgar}.json`
  (schema `finance-sim-calibrator/1`, no reply text).
- Analyses (12 files, written once, replay byte-identical offline):
  `runs/finance-sim/analysis/pre-registered-sampling-full_run.{json,md}`,
  `ensemble-sampling-{full_run,negative_control}.{json,md}`,
  `negative-control-sampling.{json,md}`,
  `convergence-sampling-{full_run,negative_control}.{json,md}`; log
  `runs/finance-sim/validation/live-9.3-analyses.log`.
- Replay: the refused first run `runs/finance-sim/validation/live-9.3-replay.log`;
  the first guarded run, which published the 26 files,
  `runs/finance-sim/validation/live-9.3-replay-2.log` (its manifest
  `runs/finance-sim/replay/replay-20261111b.json`) with its byte comparison
  `runs/finance-sim/validation/live-9.5-replay-compare.log`; the second guarded
  run after the F4 label fix `runs/finance-sim/validation/live-9.3-replay-3.log`
  (26 files, 0 leaks, 6 advisory DTB3 warning entries with 10 matches, the same
  as before; only the two convergence files changed) with its byte comparison
  against the repository copy
  `runs/finance-sim/validation/live-9.5-replay-compare-2.log` (passed byte for
  byte); the durable private manifest copy
  `runs/finance-sim/replay/replay-20261111d.json` (schema `finance-sim-replay/1`,
  26 files scanned, 12,836 probes, `required` true, items, closes and DTB3
  probed, 0 leaks, the 6 advisory DTB3 warnings, matplotlib 3.10.9, the SHA-256
  of every published and private file).
  Published copies: `docs/assets/finance-sim/` (26 files: 8 SVGs, 4 CSVs, 4
  figure tables and 10 analysis markdown copies under a sources header).
- Code: `scripts/finance_sim/` (`run_study.py`, `calibrate.py`, `pilot.py`,
  `execute.py`, `negative_control.py`, `analysis.py`, `ensemble_metrics.py`,
  `convergence.py`, `figures.py`, `figures_draws.py`, `figures_convergence.py`,
  `replay.py`); the parser fix in `recall_guard/core/decision.py` (commit
  73cb8e4, logs `runs/finance-sim/validation/task-9.2a-*.log`); the guard fix
  in `scripts/finance_sim/replay.py` (commit 6d9927d); the F4 direct-label fix
  (commit 724c966).
- Docs validation for this page: `runs/finance-sim/validation/task-9.5-*.log`.
- Register entry [R43](../research-register.md); related entries R35 (full
  logprobs via llama.cpp) and R40 (Gemma 3 4B document likelihood).
