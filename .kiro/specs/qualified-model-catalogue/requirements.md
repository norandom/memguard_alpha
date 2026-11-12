# Requirements Document

## Project Description (Input)
**Who has the problem.** The project owner and quant researchers who run
point-in-time analysis of financial time series with an LLM. They need to pick
a model and serving route that is qualified to give recall-aware answers, and
they cannot tell from the current record which are.

**Current situation.** A point-in-time simulation is fenced against lookahead.
An LLM breaks that fence when the analysis date lies inside its training range:
the model can be certain what comes tomorrow because it has read it. Recall
Guard measures that risk per prompt and prices it. In the consumer pipeline
(`macro_framework`, see `docs/pit-architecture.md`) a Black-Litterman tilt is
softened when the model's answer is recall-poisoned: the tilt is scaled by
`1 - p_memorized`, and `p_memorized` is a heuristic correction computed from
the answer's logprobs through a per-model calibrator.

Recall here means recall of financial time series inside the model's training
range. The contrast between an identifying prompt (date, tickers, raw levels)
and the fenced point-in-time prompt is a proxy used to elicit recall and to
extract logits; prompt wording is not what recall means.

The open research question is whether logits help at all: whether the full
next-token distribution improves on the logprobs correction, or whether a
similar system can be built from logits for routes that return no logprobs.

What is wrong today:

- The package has provider profiles (`recall_guard/core/provider_profiles/`,
  `available_profiles()`), and a profile says which estimate sources it offers
  (top-k logprobs, full distribution). It does not say whether the model on
  that route is qualified: whether a calibrator was fitted, in which mode, and
  whether it passed.
- When a route returns no logprobs, the record does not answer whether logits
  can be used instead, or from which models.
- The two modes are not presented as one aligned method.
- The research record (`docs/research-report.md`, `docs/research-register.md`,
  the dated reports under `docs/results/`) is a set of data dumps. The reports
  read as summaries of the specs that produced them. A spec is one technical
  way to get reportable data, not the thing to report. No report has a usable
  summary, and the same facts are repeated. Which model or API was used is
  data, not a result. The model catalog update was maintenance and the NPU
  route a computation convenience. The record discusses operating-system and
  tooling incidents, which have no place in quant research.
- The statistical analysis is missing or opaque. Tables show values, but how a
  value was calculated is not clear: the estimand, the unit of analysis, the
  sample, the test and the interval are not stated where the value appears.
- Figures are complex, some axes carry no label, and the style is not that of
  a research paper.
- A forecast at 50:50 is reported as a finding although it is the intended
  design condition: a question the model cannot answer by skill is the right
  question to derive recall and calibrate it back.

**What should change.**

1. A qualification catalogue, one entry per model on a serving route. Per
   evidence mode (logprobs, logits) it says whether the mode is available,
   whether a calibrator was fitted, its held-out quality, whether it passed,
   and the evidence. Runs on particular models and APIs are rows of data in it.
2. A selection interface: list, filter by mode and qualification, select; the
   guard then runs in that entry's mode and marks an unqualified entry.
3. An answer for routes without logprobs: whether logits can stand in, for
   which models, and what that needs.
4. An answer to the research question, from numbers already measured: on the
   same weights and the same prompts, do logits improve the correction that
   logprobs give, and how far do the two agree.
5. A statistical analysis a reader can follow: every reported value with its
   estimand, calculation, sample and uncertainty.
6. A readable research record organised by this objective: summary first,
   results not spec narration, each fact once, no infrastructure discussion.
   The structure comes from the COBESY composition back-end (`blueprint.md` in
   this spec) before any prose; every rewritten passage goes through the
   humanizer skill.
7. Figures in an academic style that a reader can interpret alone.
8. The report template and the research-reports steering updated so that
   future runs are recorded the same way.

9. A front (owner's amendment, 2026-11-11): "objective for us is to provide a
   front, where people run these prompt, and we make PiT inference answers
   possible as good as we can. this should be our basis for integration (tests
   and demos)". The front is the existing Python interface, improved and
   extended; it takes any prompt with a decision definition.

**Out of scope.** New paid runs, new model runs, and any change to a measured
number. The first catalogue is filled from results already recorded.

**Constraints.** Licensed data and reply text stay private. Published assets
pass the guarded replay. The strict MkDocs build and `tests/docs` pass. Plain
prose without em dashes. Commit and push before and after each change.

## Introduction

Recall Guard exists so that an LLM can be used in point-in-time analysis of
financial time series without silent lookahead. For an analysis date inside the
model's training range it estimates how far the answer is recalled and softens
the position taken on it. That correction is a heuristic built on the answer's
logprobs. This feature makes three things explicit. First, which model and
serving route combinations are qualified for that use, in a catalogue with a
selection interface. Second, whether logits can do the same job, where a route
returns no logprobs, and whether they improve on logprobs where both exist.
Third, a research record that reports this as quantitative research: results
with their statistics, a summary first, no retelling of specs.

## Boundary Context

- **In scope**: the qualification catalogue and its first content from results
  already recorded; the selection interface; the statement of which entries can
  use logits in place of logprobs; the paired comparison of the two modes; the
  statistical methods and worked calculations for every reported value; the
  restructured synthesis, register and dated report R43; the repeat-the-prompt
  baseline; figures in academic style; the report template and the
  research-reports steering.
- **In scope (amendment 2026-11-11)**: the front, as an extension of the
  existing Python interface: one call that runs a caller's prompt for a
  catalogue entry and returns the point-in-time answer with its recall
  estimate, correction, status and limits; its offline integration tests; its
  demos; one documentation page.
- **Out of scope**: a web page, an HTTP service, a command-line product, prompt
  construction for callers (the fenced macro prompt ships as the worked
  example only), new model runs, paid runs, new calibrations, any change to
  a measured number, changes to `macro_framework` or to the tilt correction
  itself, and the rewriting of dated reports other than R43 beyond adding a
  summary block, a class and dated notes.
- **Adjacent expectations**: `macro_framework` stays the consumer that applies
  the softened tilt; this feature tells it which entries are qualified and
  does not change how it uses them. The existing provider profiles stay the
  source of how a model is reached; a catalogue entry refers to one. The
  existing calibrators and the certification screen stay as they are; the
  catalogue reads their outcome. Operational detail (hosts, memory, antivirus,
  process failures, serving-stack fixes) belongs in the specs' implementation
  notes and the handover, never in the research record. The guarded replay
  stays the only route by which finance-simulation assets are published.

## Requirements

### Requirement 1: Qualification catalogue
**Objective:** As a quant researcher, I want one catalogue that says which
model and serving route combinations are qualified for recall-aware
point-in-time answers and in which mode, so that I can choose one without
reading the research reports.

#### Acceptance Criteria
1. The Catalogue shall hold one entry per model on a serving route, and shall
   record the model, its weights identity where known, and the route.
2. The Catalogue shall record, for each entry and for each of the two evidence
   modes (logprobs, logits), whether the mode is available, whether a
   calibrator was fitted, the held-out quality with its interval and sample
   sizes, and whether the calibrator passed the qualification rule.
3. The Catalogue shall give each entry and mode exactly one status: qualified,
   not qualified or not tested.
4. The Catalogue shall state the qualification rule once, with the statistics
   it uses, and shall apply the same rule to every entry and mode.
5. The Catalogue shall name, for each recorded quality value, the register
   entries that are its evidence.
6. The Catalogue shall record, for each calibrator, the recall boundary it was
   fitted on (which prompts stand for recall-exposed answers and which for
   fenced ones) and the date range of the series relative to the model's
   training cutoff.
7. When the first catalogue is produced, the Catalogue shall be filled only
   from results already recorded in the research register.
8. If a result needed for an entry and mode does not exist, then the Catalogue
   shall show the status not tested and shall not infer a value from another
   entry or mode.
9. Where two entries share the same weights, the Catalogue shall show them as
   related, so that a reader can compare routes for one model.

### Requirement 2: Selection interface
**Objective:** As a developer using Recall Guard, I want to list the catalogue
and select an entry, so that the guard runs with a model, route and mode whose
qualification I know.

#### Acceptance Criteria
1. When a caller asks for the catalogue, the Selection Interface shall return
   every entry with its modes, statuses and evidence references.
2. When a caller filters by mode, by status or by both, the Selection Interface
   shall return only the entries that match.
3. When a caller selects an entry and a mode, the Guard shall produce its
   recall-aware answer in that mode for that entry.
4. If a caller selects a mode that the entry does not offer, then the Selection
   Interface shall refuse the selection and name the modes the entry offers.
5. If a caller selects an entry and mode whose status is not qualified or not
   tested, then the Guard shall return the answer without the correction and
   marked with that status.
6. The Guard shall state, with every answer, the catalogue entry, the mode and
   the status it ran under.
7. When a caller does not name a mode, the Selection Interface shall choose the
   entry's qualified mode if exactly one is qualified, and otherwise shall ask
   the caller to name one.

### Requirement 3: Logits where a route returns no logprobs
**Objective:** As a researcher whose serving route returns no logprobs, I want
to know whether logits can be used and for which models, so that I am not left
without a correction.

#### Acceptance Criteria
1. The Catalogue shall state, for each entry, whether the logprobs mode is
   unavailable and why.
2. Where the logprobs mode is unavailable for an entry, the Catalogue shall
   state whether the logits mode is available for it and what that requires.
3. If neither mode is available for an entry, then the Catalogue shall say so
   and shall show the entry as not usable for recall-aware answers.
4. When a caller asks which entries can be used without logprobs, the Selection
   Interface shall return the entries whose logits mode is available, with
   their statuses.

### Requirement 4: Do logits help
**Objective:** As the project owner, I want the record to answer whether logits
improve on the logprobs correction or can replace it, so that the logits mode
is kept, dropped or built out on evidence.

#### Acceptance Criteria
1. The Guard shall derive the recall estimate from the same set of features in
   both modes, and shall name any feature that one mode cannot supply.
2. The Guard shall apply the same calibration procedure and the same
   qualification rule in both modes, and shall return answers of the same form.
3. Where both modes were measured on the same weights and the same prompts,
   the Research Record shall report a paired comparison: the held-out quality
   of each mode, their difference with an interval, and the agreement of the
   two recall estimates prompt by prompt.
4. The Research Record shall report how the correction differs between the
   modes in the quantity the consumer uses: the size of the softened tilt.
5. The Research Record shall report how the logprobs estimate approaches the
   logits estimate as more alternatives per token are kept.
6. The Research Record shall state one of three answers to the question, with
   the statistics behind it: logits improve on logprobs, logits match logprobs
   and can stand in, or the recorded data cannot decide.
7. If the two modes of a model were measured on different weights, then the
   Research Record shall not compare their values and shall state why.

### Requirement 5: What recall means and what the evidence rests on
**Objective:** As a reader, I want recall defined for this domain and the
evidence base stated, so that I know what a qualification does and does not
show.

#### Acceptance Criteria
1. The Research Record shall define recall as the model's knowledge of a
   financial time series beyond the analysis date, for dates inside its
   training range, and shall define a recall-aware answer as one corrected for
   it.
2. The Research Record shall state that the contrast between the identifying
   prompt and the fenced point-in-time prompt is a proxy to elicit recall, and
   shall not present prompt wording as the object of study.
3. The Research Record shall report the recall estimate separately for
   analysis dates inside and after the model's training cutoff, with a test of
   the difference, wherever both were recorded.
4. If the recall estimate does not fall after the cutoff, then the Research
   Record shall say so beside the entry's quality value and shall state what
   that means for the entry's status.
5. The Research Record shall state that chance-level forecast accuracy is the
   intended design condition and shall not report it as a finding about the
   guard.
6. The Research Record shall report, per serving route and temperature for
   which repeated draws were recorded, how many different answers the same
   prompt returned, as a figure and a table.
7. The Research Record shall name what evidence is missing for a trustworthy
   qualification, without defining new data or runs in this feature.

### Requirement 6: Statistical analysis
**Objective:** As a quant reader, I want to see how every reported value was
calculated and how certain it is, so that I can check it and judge it.

#### Acceptance Criteria
1. Each reported statistic shall state its estimand in words: what quantity,
   over which units, under which condition.
2. Each reported statistic shall state its calculation: the formula, the inputs
   with their counts, and the result, so that a reader can recompute it from
   the table it sits in.
3. Each reported statistic shall state its sample size and its unit of
   analysis (prompt, draw, month, filing), and shall say when units are not
   independent.
4. Each estimate shall carry an interval, with the method and the confidence
   level named.
5. Each test shall state its null hypothesis, the test used, the test
   statistic, the p-value and, where several tests form a family, the
   adjustment applied.
6. Each comparison shall report an effect size with its interval, and shall
   not report a p-value alone.
7. The Research Record shall give one worked example per kind of statistic it
   reports, from inputs to result.
8. Each Dated Report shall have a methods section that lists the statistics
   used, in the order they appear in the results.
9. If a statistic cannot be computed from the recorded data, then the Dated
   Report shall say which input is missing and shall show no value.
10. If compared values are equal by construction, then the Dated Report shall
    say so and shall not present them as a test result.
11. The Research Record shall state the statistical power or the detectable
    effect for each main comparison at its sample size.

### Requirement 7: Readable research record
**Objective:** As a reader, I want each report and the synthesis to tell me the
result first and nothing twice, so that I can use them without reading the
specs.

#### Acceptance Criteria
1. The Synthesis shall open with a summary that states which entries are
   qualified, in which mode, the answer to whether logits help, and what is
   not yet shown, before any other content.
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
   qualification evidence, mode comparison, foundation, maintenance,
   convenience or not aligned.
10. The Register shall classify the model catalog update as maintenance and
    the NPU route as convenience.
11. The Research Record shall not discuss the operating system, host memory,
    antivirus, process exit codes, ports, driver scripts, serving-stack
    troubleshooting or tooling incidents.
12. If an infrastructure incident removed data from a result, then the Dated
    Report shall state only what data is missing and its effect on the result,
    in one sentence, and shall link to the operational notes in its references
    section.
13. When a rewritten passage is prose, the Research Record shall carry it only
    after the humanizer pass, with no em dashes and no marketing language.
14. The Research Record shall be restructured from the composition blueprint
    accepted by the owner, and no prose shall be written before that
    acceptance.

### Requirement 8: Figures
**Objective:** As a reader, I want every figure to read like a figure in a
research paper and to stand alone, so that I do not draw the wrong conclusion
from it.

#### Acceptance Criteria
1. Each Figure shall label both axes with the quantity and its unit.
2. Each Figure shall carry one message, stated in a numbered caption below the
   figure ("Figure N."), and shall carry no title or annotation text inside
   the plot area beyond axis labels, tick labels and a legend.
3. Each Figure shall have a table view of the plotted values.
4. Each Figure shall follow the conventions of a quantitative journal article:
   a plain white background, black axes and text, a serif or standard
   scientific typeface, and series told apart by marker shape, line style or
   grey level so that the figure reads in black and white print.
5. Each Figure shall show uncertainty where an estimate has one (interval bars
   or bands), and its caption shall state the sample size and what the
   interval is.
6. The Figure shall not use decorative colour, rounded bars, drop shadows,
   value labels on every mark, dashboard-style tiles or separate light and
   dark themed variants.
7. If the series of a figure cannot differ by construction, then the Figure
   shall say so in its caption.
8. The Research Record shall use one figure style throughout.

### Requirement 9: Future reports
**Objective:** As the project owner, I want the rules for new reports to follow
the same structure, so that the record does not turn back into dumps.

#### Acceptance Criteria
1. The Report Template shall require the four-line summary block, the class,
   the catalogue entry that a run informs and a methods section.
2. The Report Template shall place links to specs in the references section and
   shall forbid narrating spec execution in the body.
3. The Reporting Rules shall require that a new run updates the catalogue entry
   it informs, or states that it informs none.
4. When a new model or serving route is added, the Reporting Rules shall treat
   it as a catalogue entry and not as a research phase.
5. The Reporting Rules shall require the statistical statements of
   Requirement 6 for every reported value.

### Requirement 10: Integrity of the record
**Objective:** As an auditor, I want the restructure to leave the measured
results intact, so that the new record can be trusted as much as the old one.

#### Acceptance Criteria
1. The Research Record shall keep every measured number, interval and
   denominator exactly as recorded.
2. Where a statistic required by Requirement 6 is added, the Research Record
   shall compute it from the recorded data only and shall mark it as computed
   in this restructure, with its date.
3. When a restructured passage changes what an earlier report concluded, the
   Research Record shall add a dated note to the earlier report and shall not
   edit its findings.
4. The Research Record shall keep failed, stopped and not aligned runs, each
   with its cause in one sentence.
5. The Research Record shall publish no licensed value and no reply text.
6. When the restructured record is complete, the documentation build in strict
   mode and the documentation tests shall pass.
7. Where a figure or table of the finance simulation is published, it shall
   come from the guarded replay and compare byte for byte with the published
   copy.
### Requirement 11: The front for point-in-time answers
**Objective:** As a person with a prompt, I want one place to run it and get
the best point-in-time answer Recall Guard can give, so that integration tests
and demos all enter the same way I do.

#### Acceptance Criteria
1. The Front shall be the existing Python interface of Recall Guard, extended;
   every existing public name shall keep its behaviour.
2. When a caller gives the Front a prompt, a decision definition and a
   catalogue entry, the Front shall return the answer, the recall estimate,
   the corrected answer where one applies, and the entry, mode, status and
   reason it ran under.
3. The Front shall accept any prompt with a decision definition and shall not
   require the prompt to be built by Recall Guard.
4. When the caller names no mode, the Front shall choose one by a stated order:
   a qualified mode first; otherwise an available mode for which the caller
   supplied a calibrator; otherwise the logprobs mode if available, else the
   logits mode; and shall state in the result which mode it chose and why.
5. If the entry and mode is not qualified or not tested, then the Front shall
   still return the answer, uncorrected and marked with the status and reason,
   and shall not refuse.
6. Where the caller supplies no calibrator, the Front shall return the answer
   and the status without a recall estimate and shall state that no calibrator
   was supplied.
7. Where the caller supplies no client, the Front shall reach the model through
   the entry's provider profile.
8. The Front shall state with each result what limited it: no qualified mode,
   no calibrator supplied, a calibrator other than the recorded one, an
   estimate that could not be computed, or nothing.
9. If the Front cannot produce an answer (the entry is unknown, neither mode is
   available, the model cannot be reached, or the reply cannot be read), then
   the Front shall say which step failed and what the caller can change, and
   shall return no partial result as an answer.
10. The Front shall ship one worked example: the fenced point-in-time macro
    prompt with its decision definition.
11. The integration tests of this feature shall enter through the Front and
    shall run offline with fixed replies, without a key and without a network,
    covering every status, both modes, the mode choice, the missing calibrator
    and each failure of criterion 9.
12. Where a local model host is declared available, an integration test shall
    run the worked example through the Front against it and record the entry,
    mode and status of the result; it shall be skipped otherwise.
13. The demos of this feature shall enter through the Front: a script and a
    notebook that list the catalogue, select an entry, run the worked example
    and show the result, both runnable offline.
14. The documentation shall describe the Front on one page: how to list the
    catalogue, select an entry, run a prompt and read the result, with the
    meaning of each status and a link to the catalogue.

### Requirement 12: The two evidence modes, defined and shown side by side

> Amendment 2026-11-12, on the owner's instruction ("amend current spec"),
> approved by the owner on 2026-11-12 ("approved, go"). It follows the research guardrails in
> `.kiro/steering/research-guardrails.md`.

**Objective:** As a quant reader, I want the logprobs mode and the logits mode
defined mathematically and run on the same prompt side by side, so that I can
see what each recall estimate is, where the two are the same quantity, where
they differ, and what the recorded data supports.

#### Acceptance Criteria
1. The Mode Note shall define, for each mode: the evidence it reads (the
   alternatives per generated token that the route returns, or the next-token
   vector), each feature as a formula, the calibrator, the recall estimate and
   the correction of the answer.
2. The Mode Note shall state for each feature whether the two modes give the
   same value by construction, the same value in a stated limit, or different
   values, each with its derivation.
3. The Mode Note shall state the estimand of the recall estimate under the
   labels the calibrators were fitted on, what that estimand is not (exposure
   of an analysis month to training), and which analysis tests exposure.
4. Where the Mode Note cites a statistic of the recorded data, it shall give
   the estimand, the sample size, the unit of independence and one label:
   established, exploratory or not interpretable.
5. The owner shall approve the Mode Note before the Figure Specification, the
   Notebook or any published text is built on it.
6. The Figure Specification shall state for each published figure: the
   question it answers about recall or about the modes, the estimand, the
   comparison that is drawn, the one statement the caption leads with, and the
   table columns in plain words.
7. If a figure answers no question about recall or about the modes, then the
   Figure Specification shall remove it from the published set or state the
   owner's reason to keep it.
8. The owner shall approve the Figure Specification before a figure is
   reworked.
9. The Notebook shall be a usage example under `docs/`, not a research
   document (owner, 2026-11-12: "simple and clear ... just how to use the
   tool"). When a person runs it, it shall send one point-in-time prompt
   through the Front in the logprobs mode and in the logits mode and shall show
   for each mode the status, the recall estimate, the answer and the corrected
   answer.
10. The Notebook shall define each term it uses once, and shall state in one
    table what differs between the two modes for a person who uses them: the
    evidence read, the extra host, when the mode applies and the calibrator.
    The derivations and the recorded statistics belong to the Mode Note.
11. The Notebook shall state how the call changes on a person's own route, and
    shall show the status of every entry of the shipped catalogue, so that a
    mode that is not qualified or not tested is not read as qualified.
12. The Notebook shall run offline on an invented setup that the package
    provides, and shall state that the entry, the calibrators and the replies
    are invented and that its numbers say nothing about any model.
13. The documentation shall show the Notebook as a page with its stored
    outputs.
14. A test shall execute the Notebook offline, shall check that both modes
    return a recall estimate and a corrected answer, and shall check that the
    stored outputs equal those of a fresh run.
16. The Notebook shall state that the recall estimate is the result and that
    its use is the analyst's decision, the multiplication of `corrected` being
    one rule among others (owner, 2026-11-12).
17. The Notebook shall state that the estimate applies to periods on or before
    the model's training cutoff, where an answer needs correction, and that
    for periods after the cutoff an estimate is noise and no correction is
    applied; and that the caller compares the period of the data with the
    cutoff, because the Front does not know it (owner, 2026-11-12).
18. Where a person turns an online group on, the Notebook shall run the same
    call against a running local model: the logprobs mode, the logits mode on
    the replies the logprobs group drew (one model host at a time), and the
    logprobs mode on the NPU. The groups are off by default, their cells are
    tagged, and their stored outputs are exempt from the comparison of 12.14.
19. One start script per online group shall start its model in the foreground
    and stop it on Ctrl+C.
20. The Front shall take the sampling temperature of the draws as an optional
    argument, because a calibrator is used only at the temperature it was
    fitted at.
15. Every text of this requirement shall follow the research guardrails, and a
    text published under `docs/` shall pass the structure pass and the wording
    pass named there before publication.
