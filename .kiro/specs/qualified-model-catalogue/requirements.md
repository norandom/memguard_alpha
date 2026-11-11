# Requirements Document

## Project Description (Input)
**Who has the problem.** The project owner and anyone who adopts Recall Guard.
They need to pick a model that is qualified to give recall-aware answers, and
they cannot tell from the current record which models are.

**Current situation.** Recall Guard's objective is recall-aware answers: an
answer that carries a calibrated recall estimate and is adjusted by it. The
research strategy was to take the existing guard built on logprobs and produce
its counterpart built on logits. Both exist, but nothing brings them together:

- The package has provider profiles (`recall_guard/core/provider_profiles/`,
  `available_profiles()`), and a profile says which estimate sources it offers
  (top-k logprobs, full distribution). It does not say whether the model is
  qualified: whether a calibrator was fitted for it, in which mode, and whether
  that calibrator passed.
- When a model or API returns no logprobs, the record does not answer whether
  logits can be used instead, or from which models.
- The two modes are not presented as one aligned method. Their features,
  calibration and interface are described in different reports under different
  names.
- The research record (`docs/research-report.md`, `docs/research-register.md`,
  the dated reports under `docs/results/`) is a set of data dumps. The reports
  read as summaries of the specs that produced them: tasks, arms, gates,
  receipts and fixes. A spec is one technical way to get reportable data, not
  the thing to report. No report has a summary a reader can use, and the same
  facts are repeated across summary, tables, limits and evidence. Which model
  or API was used is data, not a result. The model catalog update was
  maintenance and the NPU route a computation convenience. Figures are complex,
  some axes carry no label, and the tables show single against ensemble and
  adjusted against unadjusted as equal because the accuracy metric cannot see
  the difference. A forecast at 50:50 is reported as a finding although it is
  the intended design condition.
- There is no foundational truth: no cases where recall is known, and no plain
  baseline of how much answers vary when a prompt is repeated.

**What should change.**

1. A qualification catalogue. Each entry is a model with, per evidence mode
   (logprobs, logits), whether the mode is available, whether a calibrator was
   fitted, its held-out quality, whether it passed, and the evidence behind
   that. Runs on particular models and APIs become rows of data in it.
2. A selection interface. A caller lists the catalogue, filters it by mode and
   qualification, and selects an entry; the guard then runs in that entry's
   mode and refuses or flags an entry that is not qualified.
3. An answer for models without logprobs. The catalogue states, per model,
   whether the logits mode can stand in and what that needs.
4. Aligned modes. Logprobs and logits use the same features, the same
   calibration procedure, the same qualification rule and the same interface,
   and the record shows how closely their estimates agree on the same weights.
5. A readable research record organised by this objective. Reports state
   results, not how a spec was executed; specs are linked in the references
   only. The synthesis and every dated report open with a plain summary.
   Redundant information is trimmed. Each register entry is classified.
   Earlier findings are not rewritten; dated notes are added, as
   `.kiro/steering/research-reports.md` requires. The structure comes from the
   COBESY composition back-end (`blueprint.md` in this spec) before any prose,
   and every rewritten passage goes through the humanizer skill.
6. The missing foundation, stated: what would make a qualification trustworthy
   (cases of known recall; a repeat-the-prompt baseline per runtime and
   temperature), with the plain baseline figure from the recorded draws.
7. Figures a reader can interpret alone.
8. The report template and the research-reports steering updated so that future
   runs are recorded as catalogue evidence with a summary and a class.

**Out of scope.** New paid runs, new model runs, and any change to a measured
number. The first catalogue is filled from results already recorded.

**Constraints.** Licensed data and reply text stay private. Published assets
pass the guarded replay. The strict MkDocs build and `tests/docs` pass. Plain
prose without em dashes.

## Introduction

Recall Guard exists to give recall-aware answers: an answer, a calibrated
estimate of how likely it was recalled, and an adjustment by that estimate. Only
a model whose estimate has been calibrated can be trusted to do that. This
feature makes that visible and selectable. It adds a catalogue that says, per
model and per evidence mode (logprobs or logits), whether the model is
qualified, and an interface to select from it. It aligns the two modes as one
method, answers what to do when a model returns no logprobs, and restructures
the research record so that it reports results as evidence for catalogue
entries, with a summary first and without repetition.

## Boundary Context

- **In scope**: the qualification catalogue and its first content from results
  already recorded; the selection interface; the statement of which models can
  use logits in place of logprobs; the alignment of the two modes; the
  restructured synthesis, register and dated report R43; the plain
  repeat-the-prompt figure; labelled figures; the report template and the
  research-reports steering.
- **Out of scope**: new model runs, paid runs, new calibrations, any change to
  a measured number, new known-recall data, and the rewriting of dated reports
  other than R43 beyond adding a summary block, a class and dated notes.
- **Adjacent expectations**: operational detail (hosts, memory, antivirus,
  process failures, serving-stack fixes) belongs in the specs' implementation
  notes and the handover, never in the research record. The existing provider
  profiles stay the source of
  how a model is reached; the catalogue refers to them and does not replace
  them. The existing calibrators and their quality rule stay as they are; the
  catalogue reads their outcome. The guarded replay stays the only route by
  which finance-simulation assets are published. Specs stay the place where
  execution detail lives; reports link to them.

## Requirements

### Requirement 1: Qualification catalogue
**Objective:** As an adopter, I want one catalogue that says which models are
qualified for recall-aware answers and in which mode, so that I can choose a
model without reading the research reports.

#### Acceptance Criteria
1. The Catalogue shall hold one entry per model, identified by the model and
   its weights, with the ways it can be reached (APIs, runtimes, hardware)
   recorded as attributes of the entry.
2. The Catalogue shall record, for each entry and for each of the two evidence
   modes (logprobs, logits), whether the mode is available, whether a
   calibrator was fitted, the held-out quality with its interval and
   denominators, and whether the calibrator passed the quality rule.
3. The Catalogue shall give each entry and mode exactly one status: qualified,
   not qualified or not tested.
4. The Catalogue shall name, for each recorded quality value, the register
   entries that are its evidence.
5. The Catalogue shall record, for each calibrator, what its labels mean
   (real exposure of the text, or another distinction such as prompt wording).
6. If a calibrator's labels do not mean real exposure, then the Catalogue shall
   not give that entry and mode the status qualified, and shall state the
   reason.
7. When the first catalogue is produced, the Catalogue shall be filled only
   from results already recorded in the research register.
8. If a result needed for an entry and mode does not exist, then the Catalogue
   shall show the status not tested and shall not infer a value from another
   model or mode.

### Requirement 2: Selection interface
**Objective:** As a developer using Recall Guard, I want to list the catalogue
and select an entry, so that the guard runs with a model and mode whose
qualification I know.

#### Acceptance Criteria
1. When a caller asks for the catalogue, the Selection Interface shall return
   every entry with its modes, statuses and evidence references.
2. When a caller filters by mode, by status or by both, the Selection Interface
   shall return only the entries that match.
3. When a caller selects an entry and a mode, the Guard shall produce its
   recall-aware answer in that mode for that model.
4. If a caller selects a mode that the entry does not offer, then the Selection
   Interface shall refuse the selection and name the modes the entry offers.
5. If a caller selects an entry and mode whose status is not qualified or not
   tested, then the Guard shall return the answer unadjusted and marked with
   that status, and shall not present an adjusted answer as recall-aware.
6. The Guard shall state, with every answer, the catalogue entry, the mode and
   the status it ran under.
7. When a caller does not name a mode, the Selection Interface shall choose the
   entry's qualified mode if exactly one is qualified, and otherwise shall ask
   the caller to name one.

### Requirement 3: Logits in place of logprobs
**Objective:** As an adopter whose model returns no logprobs, I want to know
whether the logits mode can be used and for which models, so that I am not
left without a recall estimate.

#### Acceptance Criteria
1. The Catalogue shall state, for each entry, whether the logprobs mode is
   unavailable and why.
2. Where the logprobs mode is unavailable for an entry, the Catalogue shall
   state whether the logits mode is available for it and what that requires of
   the adopter.
3. If neither mode is available for an entry, then the Catalogue shall say so
   and shall show the entry as not usable for recall-aware answers.
4. When a caller asks which models can be used without logprobs, the Selection
   Interface shall return the entries whose logits mode is available, with
   their statuses.

### Requirement 4: Aligned modes
**Objective:** As the project owner, I want the logprobs mode and the logits
mode to be one method with two sources of evidence, so that their results can
be compared and a calibrated model means the same thing in both.

#### Acceptance Criteria
1. The Guard shall derive the recall estimate from the same set of features in
   both modes, and shall name any feature that one mode cannot supply.
2. The Guard shall apply the same calibration procedure and the same quality
   rule in both modes.
3. The Guard shall return answers of the same form in both modes: the answer,
   the recall estimate, the adjustment and the status.
4. Where both modes are available for an entry on the same weights, the
   Research Record shall report how closely the two estimates agree and the
   held-out quality of each, side by side.
5. If the two modes of an entry were measured on different weights, then the
   Research Record shall not compare their values and shall state why.

### Requirement 5: Foundation for qualification
**Objective:** As the project owner, I want the record to state what a
trustworthy qualification rests on and what is missing, so that the catalogue
does not claim more than was shown.

#### Acceptance Criteria
1. The Research Record shall state whether cases of known recall exist for each
   catalogue entry and mode.
2. While no cases of known recall exist for an entry and mode, the Research
   Record shall say that its recall estimate has not been checked against
   recall.
3. The Research Record shall report, per runtime and temperature for which
   repeated draws were recorded, how many different answers the same prompt
   returned.
4. The Research Record shall present that repeat baseline as a figure and a
   table computed from the recorded draws.
5. The Research Record shall state that chance-level forecast accuracy is the
   intended design condition and shall not report it as a finding about the
   guard.
6. The Research Record shall name what would supply the missing foundation,
   without defining new data or runs in this feature.

### Requirement 6: Readable research record
**Objective:** As a reader, I want each report and the synthesis to tell me the
result first and nothing twice, so that I can use them without reading the
specs.

#### Acceptance Criteria
1. The Synthesis shall open with a summary that states which models are
   qualified, in which mode, and what is not yet shown, before any other
   content.
2. The Synthesis shall present the catalogue as its first table.
3. The Synthesis shall be organised by the heads of the accepted composition
   blueprint, and its heads read in order shall state the argument.
4. Each Dated Report shall open with a summary block of four lines: the result,
   its effect on the catalogue, what it does not show, and its class.
5. A Dated Report shall state results and their meaning, and shall not narrate
   how a spec was executed (tasks, gates, sessions, fixes, attempts).
6. Where execution detail is needed to audit a result, the Dated Report shall
   link to the spec and the private receipts in its references section only.
7. The Research Record shall state each fact once in a report, and shall refer
   to it elsewhere in that report instead of repeating it.
8. The Research Record shall use model and API names as data in tables and
   shall not use them as section heads or phase names.
9. The Register shall give every entry one class from a fixed list:
   qualification evidence, mode alignment, foundation, maintenance,
   convenience or not aligned.
10. The Register shall classify the model catalog update as maintenance and
    the NPU route as convenience.
11. When a rewritten passage is prose, the Research Record shall carry it only
    after the humanizer pass, with no em dashes and no marketing language.
12. The Research Record shall be restructured from the composition blueprint
    accepted by the owner, and no prose shall be written before that
    acceptance.
13. The Research Record shall be written as quantitative research: every
    section shall serve a result, its uncertainty or its meaning for the
    catalogue.
14. The Research Record shall not discuss the operating system, host memory,
    antivirus, process exit codes, ports, driver scripts, serving-stack
    troubleshooting or tooling incidents.
15. If an infrastructure incident removed data from a result, then the Dated
    Report shall state only what data is missing and its effect on the result,
    in one sentence, and shall link to the operational notes in its references
    section.

### Requirement 7: Figures
**Objective:** As a reader, I want every figure to be readable without the
surrounding text, so that I do not draw the wrong conclusion from it.

#### Acceptance Criteria
1. Each Figure shall label both axes with the quantity and its unit.
2. Each Figure shall carry one message, stated in its title.
3. Each Figure shall have a table view of the plotted values.
4. If the series of a figure cannot differ by construction, then the Figure
   shall say so in its caption.
5. The Research Record shall not present a figure from which a reader would
   conclude that repeated answers never vary, without the repeat baseline
   beside it.

### Requirement 8: Future reports
**Objective:** As the project owner, I want the rules for new reports to follow
the same structure, so that the record does not turn back into dumps.

#### Acceptance Criteria
1. The Report Template shall require the four-line summary block, the class and
   the catalogue entry that a run informs.
2. The Report Template shall place links to specs in the references section and
   shall forbid narrating spec execution in the body.
3. The Reporting Rules shall require that a new run updates the catalogue entry
   it informs, or states that it informs none.
4. When a new model or API is added, the Reporting Rules shall treat it as a
   catalogue entry or attribute and not as a research phase.

### Requirement 9: Integrity of the record
**Objective:** As an auditor, I want the restructure to leave the measured
results intact, so that the new record can be trusted as much as the old one.

#### Acceptance Criteria
1. The Research Record shall keep every measured number, interval and
   denominator exactly as recorded.
2. When a restructured passage changes what an earlier report concluded, the
   Research Record shall add a dated note to the earlier report and shall not
   edit its findings.
3. The Research Record shall keep failed, stopped and not aligned runs, each
   with its cause.
4. The Research Record shall publish no licensed value and no reply text.
5. When the restructured record is complete, the documentation build in strict
   mode and the documentation tests shall pass.
6. Where a figure or table of the finance simulation is published, it shall
   come from the guarded replay and compare byte for byte with the published
   copy.
