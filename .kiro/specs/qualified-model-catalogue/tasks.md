# Implementation Plan

Working rules for every task: commit and push before and after the change;
tests first (RED, then GREEN); `uv run --no-sync pytest -q <paths>`,
`uv run --no-sync ruff check .` and `uv run --no-sync python scripts/check_architecture.py`
from PowerShell; no new dependency; no measured number changes. The byte
comparison of the guarded replay against the published assets reads private
records and is not part of CI; it is run locally where a task names it.

- [ ] 1. Foundation: statistics, figure style and catalogue types
- [x] 1.1 Statistics with their calculation exposed
  - Add to the core significance module a record for one reported statistic that carries its estimand, formula, inputs with counts, unit of analysis, sample size, independence flag, value, interval, test, effect size, detectable effect, a by-construction flag and the name of a missing input
  - Add a prompt-clustered AUC interval, a label permutation test for AUC that permutes at cluster level, a paired AUC difference on shared resamples with its interval and test, agreement statistics for two estimates (mean difference with limits of agreement, mean absolute difference, rank correlation), and a one-sided permutation test of a mean difference that permutes at block level
  - The detectable effect at power 0.8 and alpha 0.05 comes from the bootstrap standard error
  - Done when tests show known answers on constructed data, cluster-level and block-level resampling, seed stability, a missing input giving no value, and the architecture check passes
  - _Boundary: Significance additions_
  - _Requirements: 4.3, 5.3, 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.9, 6.10, 6.11_
- [x] 1.2 (P) One academic figure style
  - Provide one style for every figure: white background, black axes and text, a serif typeface, no grid, series told apart by marker, line style or grey level
  - Provide the numbered caption builder that states the message, the sample size and what the interval is
  - Done when tests show the style values, distinct series styles that differ without colour, and a caption that starts with "Figure N."
  - _Boundary: Figure style_
  - _Requirements: 8.2, 8.4, 8.5, 8.6, 8.8_
- [x] 1.3 (P) Catalogue types and the qualification rule
  - Define the entry (model on a serving route, weights identity, weights group), the per-mode record (availability with reason and requirement, features and features a mode cannot supply, recall boundary, held-out quality with interval method and sample sizes, cutoff check, calibrator fingerprint, evidence) and the three statuses
  - Implement the single qualification rule with its four checks in order (time-series boundary with reply features, at least 50 valid draws per class, held-out AUC at least 0.6 with permutation p below 0.05, estimate higher inside the training range than after the cutoff): a recorded check that fails gives not qualified with that reason; if none fails and an input is not recorded, the status is not tested with the first missing input as the reason; all four true gives qualified
  - Done when tests show each check failing alone, a record without quality as not tested, a time-series record with quality but no permutation p or no cutoff check as not tested with that reason, a record whose permutation p is missing and whose cutoff check fails as not qualified, a document-text boundary as not qualified, and the same rule applied to both modes
  - _Boundary: Catalogue_
  - _Requirements: 1.2, 1.3, 1.4, 1.5, 1.6, 1.8, 4.1, 4.2_
- [ ] 1.4 Catalogue loader, validation, availability and filters
  - Load and validate a catalogue data file (exact keys, evidence ids, no value without evidence, profile ids that exist); derive mode availability from the provider profile when the entry has one, with the reason and the requirement taken from the data file
  - Provide listing, filtering by mode and status, the entries usable without logprobs, related entries by weights group and the not-usable flag when neither mode is available
  - Done when tests on a fixture data file show each validation error, availability equal to the profile's estimate sources, and every filter
  - _Boundary: Catalogue_
  - _Requirements: 1.1, 1.9, 2.1, 2.2, 3.1, 3.2, 3.3, 3.4_

- [ ] 2. Core: catalogue content, selection, shared interval, page renderer
- [ ] 2.1 First catalogue content from the register
  - Write the shipped catalogue data: one entry per shipped provider profile plus the route measured without a profile; values from the register entries R26, R30, R38, R40, R41 and R43 only, each with its interval method, sample sizes, recall boundary and evidence ids; document-likelihood results recorded as evidence of a different method; profiles without a recorded calibrator carry no quality; no entry carries a calibrator fingerprint, a permutation p or a cutoff check yet (the R43 entries get them in 4.2)
  - Make the data file part of the installed package; build the wheel into `D:\BuildArtifacts\memguard_alpha\wheel-check` and load the catalogue from it
  - Done when the catalogue loads from that build, and a test finds every evidence id and every recorded value in the register text
  - _Depends: 1.4_
  - _Requirements: 1.7, 1.8, 10.1_
- [ ] 2.2 (P) Selection interface
  - Select an entry and a mode; refuse a mode the entry does not offer and name the offered ones; without a mode, return the single qualified mode or ask for one
  - Run the unchanged guard in the selected mode and return the answer with entry, mode, status and reason; expose the correction only when the status is qualified, the guard status is ok and the supplied calibrator's fingerprint equals the catalogue's; a fingerprint mismatch, or an entry without a recorded fingerprint, downgrades the run to not tested
  - Export the catalogue and selection names at the package root without importing matplotlib
  - Done when tests with a fake client, a fake calibrator and a fixture catalogue show the qualified, not qualified, not tested and mismatch cases, the refusal text, and a recall estimate identical to the guard's
  - _Boundary: Selection_
  - _Depends: 1.4_
  - _Requirements: 2.3, 2.4, 2.5, 2.6, 2.7_
- [ ] 2.3 (P) The simulation uses the shared clustered interval
  - Replace the simulation's own prompt-clustered AUC interval with the core one
  - Done when the calibration tests pass and, before any figure change of group 3 lands, the guarded replay still compares byte for byte with the published assets
  - _Boundary: calibrate_
  - _Depends: 1.1_
  - _Requirements: 6.4, 10.1_
- [ ] 2.4 (P) Catalogue page renderer
  - Render a Markdown page from a catalogue: the table (entry, route, per mode availability, quality with interval and sample sizes, status with reason, evidence), the list of entries usable without logprobs with what the logits mode needs, and related entries
  - Done when a test shows the rendered text for a fixture catalogue and that rendering twice gives the same bytes
  - _Boundary: render_catalogue_
  - _Depends: 1.4_
  - _Requirements: 1.4, 1.5, 3.1, 3.2, 3.3_

- [ ] 3. Analyses and figures of the finance simulation
- [ ] 3.1 (P) Logprobs against logits on the same weights and prompts
  - From the calibrator records of one arm in both sources and the stored item results: held-out quality per mode with its cluster-level permutation p, the paired difference with interval and test, agreement of the two recall estimates, the difference of the correction factor the consumer uses, and the convergence by number of alternatives kept read from the existing convergence analysis
  - State one of three answers with its statistics: improve, match (difference within 0.05 and agreement limits within 0.10), or undecided with the detectable effect; refuse the comparison when the two modes carry different weights identities
  - Write a private analysis file and its Markdown table, each value with its statistic record
  - Done when tests on fixtures show the per-mode permutation p, each of the three answers, the refusal, and a missing input reported without a value
  - _Boundary: Mode comparison_
  - _Depends: 1.1_
  - _Requirements: 4.3, 4.4, 4.5, 4.6, 4.7, 6.1, 6.2, 6.3, 6.6, 6.9, 6.11_
- [ ] 3.2 (P) Recall estimate inside and after the training cutoff
  - Per mode and prompt kind: mean estimate for analysis dates inside the training range and after the cutoff, the difference with a month-block interval and the one-sided block permutation test, sample sizes and detectable effect
  - Write a private analysis file and its Markdown table
  - Done when tests on fixtures show a passing case, a failing case and a case with too few months reported as missing
  - _Boundary: Cutoff check_
  - _Depends: 1.1_
  - _Requirements: 5.3, 5.4, 6.1, 6.2, 6.3, 6.5, 6.9, 6.11_
- [ ] 3.3 (P) Repeat-the-prompt baseline
  - From the recorded draws of both arms, fully recorded ensembles only: per serving route and temperature, the distribution of the number of different answers per prompt and the share of prompts with one answer with a Wilson interval
  - Write a private analysis file and its Markdown table
  - Done when tests on fixtures show the counts, the exclusion of incomplete ensembles and the interval
  - _Boundary: Repeat baseline_
  - _Depends: 1.1_
  - _Requirements: 5.6, 6.1, 6.2, 6.3, 6.4_
- [ ] 3.4 Figures for the three analyses
  - One figure each for the mode comparison, the cutoff check and the repeat baseline, in the academic style, with both axes labelled with quantity and unit, interval bars where the value has an interval, a table view and a numbered caption with the sample size
  - Done when tests show labelled axes, no title inside the plot, one SVG with its table and caption file per figure, and a caption that names the sample size
  - _Boundary: figures_catalogue_
  - _Depends: 1.2, 3.1, 3.2, 3.3_
  - _Requirements: 5.6, 8.1, 8.2, 8.3, 8.4, 8.5, 8.6_
- [ ] 3.5 (P) The four existing figures in the academic style
  - Render the accuracy gap, reliability, per-draw and convergence figures through the one style: one SVG per figure, both axes labelled with units, no title inside the plot, numbered captions, interval bars kept
  - The accuracy-gap caption states that its variants cannot differ in accuracy by construction
  - Update the replay's figure calls and the file names its tests expect, so that the replay publishes one SVG per figure; the new analyses are not added here (4.1)
  - The published assets and the report's references to them stay as they are until 4.2; until then the local replay comparison against the published copy is expected to differ: in the figure files from this task on, and after 4.1 also in the new analysis and figure files
  - Done when the figure tests and the replay tests pass with the new file sets and captions and no light or dark variant is produced
  - _Boundary: figures, figures_draws, figures_convergence, replay_
  - _Depends: 1.2, 2.3_
  - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7, 8.8_

- [ ] 4. Integration: replay, published assets, catalogue page
- [ ] 4.1 The replay writes and publishes the new analyses and figures
  - The guarded replay writes the mode comparison, the cutoff check and the repeat baseline, renders their figures beside the four restyled ones, publishes one SVG per figure with its table and caption, and scans every published file with the licensing guard
  - Done when the replay tests show the new private and published sets, no light or dark file, and a leak still refused
  - _Depends: 3.1, 3.2, 3.3, 3.4, 3.5_
  - _Requirements: 10.5, 10.7_
- [ ] 4.2 Republish from the recorded data and complete the catalogue
  - Run the guarded replay on the private records, replace the published assets, and in the same commit point the R43 report at the new file names so the strict build never sees a missing figure
  - Enter into the catalogue data, for the R43 entries only: the per-mode permutation p-values from the mode comparison, the cutoff checks from the cutoff analysis, and the calibrator fingerprints from the private calibrator records; mark the p-values and cutoff checks as computed in this restructure with the date
  - Done when the replay compares byte for byte with the published copy, the strict documentation build passes, and the catalogue shows a status with a reason for every entry and mode
  - _Depends: 2.1, 4.1_
  - _Requirements: 1.2, 1.3, 5.3, 5.4, 10.2, 10.7_
- [ ] 4.3 Catalogue page published
  - Render the catalogue page from the completed data file with the renderer, add it to the navigation, and add a test that the committed page equals a fresh render
  - Done when the page is generated by one command, the equality test passes, and the strict build passes
  - _Depends: 2.4, 4.2_
  - _Requirements: 1.4, 1.5, 3.1, 3.2, 3.3, 7.2_

- [ ] 5. Research record
- [ ] 5.1 Report template, reporting rules and the record test
  - Update the report template: four-line summary block (result, catalogue effect, not shown, class), sections Question, Methods, Data, Results, Limits, References; specs linked in References only; no narration of spec execution
  - Update the research-reports steering: a run updates the catalogue entry it informs or says it informs none; a new model or route is a catalogue entry, not a phase; the statistical statements are required for every value; quant content only; academic figures
  - Add the documentation test with two scopes. Every dated report: the summary block and a class from the fixed list. The synthesis, the R43 report, the register's new passages, the methods page, the catalogue page and the summary blocks themselves: a fixed list of banned infrastructure terms, the absence of em dashes, no section head that contains a model, provider or route name taken from the shipped catalogue entries and provider profiles, numbered figure captions and no themed figure variants. The findings of other dated reports are not checked for content. Documents not yet brought into form are listed in the test and removed from that list by the tasks that do so
  - Done when the test passes for the template and steering and lists the documents still to be brought into form
  - _Depends: 2.1_
  - _Requirements: 6.8, 7.11, 9.1, 9.2, 9.3, 9.4, 9.5_
- [ ] 5.2 Methods page
  - One page that defines every statistic the record reports, in the order of the synthesis: estimand, formula, unit of analysis, interval method and level, test with its null, family adjustment, effect size, detectable effect; one worked example per kind of statistic, from inputs in a published table to the result
  - Done when every statistic named in the published analysis tables has an entry and a worked example whose result equals the published value, and the page is in the navigation
  - _Depends: 4.2_
  - _Requirements: 6.1, 6.2, 6.4, 6.5, 6.7, 6.8_
- [ ] 5.3 Synthesis restructured from the blueprint
  - Structure first: the ten heads of the accepted blueprint, the summary, the catalogue as the first table; then prose; then the humanizer pass on every passage
  - Content: recall defined for point-in-time analysis of financial time series; the prompt contrast named as the proxy; chance-level accuracy as the design condition; the answer to whether logits help with its statistics; routes without logprobs; inside and after the cutoff; the repeat baseline; what is not shown; each fact once; model and API names only in tables
  - Done when the record test passes for the synthesis, every number in it is found in the register or in a published analysis table, the humanizer pass is recorded as done, and the strict build passes
  - _Depends: 4.3, 5.1, 5.2_
  - _Requirements: 4.6, 5.1, 5.2, 5.5, 5.7, 7.1, 7.2, 7.3, 7.7, 7.8, 7.11, 7.13, 7.14, 10.1_
- [ ] 5.4 The finance-simulation report in the new form
  - Rewrite R43 to the template: summary block, Question, Methods (linking the methods page), Data, Results with a calculation a reader can recompute from each table, Limits, References with the spec and the private receipts; results and their meaning only; values equal by construction stated as such; the stopped serving run in one sentence saying what data is missing and what that does to the result
  - Done when the record test passes for R43, every measured number of the previous version that is still reported is unchanged, the humanizer pass is recorded as done, and the strict build passes
  - _Depends: 4.2, 5.1, 5.2_
  - _Requirements: 6.3, 6.6, 6.10, 7.4, 7.5, 7.6, 7.7, 7.8, 7.11, 7.12, 7.13, 10.1, 10.4_
- [ ] 5.5 Register restructured
  - Purpose from the governing idea; state of evidence replaced by the catalogue and the four analyses, with the earlier items kept as dated entries; the index gains Class and Catalogue entry; every entry classed from the fixed list, the model catalog update as maintenance and the NPU route as convenience; a dated note where the restructure changes an earlier conclusion
  - Done when the record test passes for the register, every entry has exactly one class, no earlier finding text is edited, and the humanizer pass on the new passages is recorded as done
  - _Depends: 5.1, 5.3_
  - _Requirements: 7.9, 7.10, 7.11, 7.13, 10.3, 10.4_
- [ ] 5.6 Summary blocks for the other dated reports
  - Add the four-line summary block and the class to every other dated report and the results index, with a dated note where needed; their findings stay as written
  - Done when every dated report passes the summary-block and class checks, the new blocks pass the content checks, the list of documents still to be brought into form is empty, no finding text of those reports is changed, and the humanizer pass on the new blocks is recorded as done
  - _Depends: 5.1, 5.5_
  - _Requirements: 7.4, 7.9, 7.11, 7.13, 10.3_

- [ ] 6. Validation
- [ ] 6.1 Whole-feature validation
  - Run the full test suite, the lint and architecture checks, the strict documentation build with the documentation tests, and the guarded replay with the byte comparison; check the CI run of the final push
  - Done when all of them pass and their logs are stored with the private validation logs
  - _Requirements: 10.5, 10.6, 10.7_

## Implementation Notes

- 1.3: a record without any quality value is not tested even when another recorded field would fail a check (design invariant, Requirement 1.8); the checks mapping still shows the failing check as False. The cutoff check compares the inside and after means and ignores the recorded difference field; the loader (1.4) should validate that the two agree.
- 1.1: clustered_auc_interval reproduces the calibration script's interval exactly (tested with ==) under two caller rules that 2.3 must keep: pass clusters in the script's group order, and keep the caller's guard that returns no bounds when a class is absent (the core function raises ValueError there). permutation_auc_test is one-sided (greater) and records that in its parameters; the reviewer suggests an alternative keyword later. Only agreement builds a missing-input Statistic; the other functions raise ValueError and the analyses assemble the missing record.
