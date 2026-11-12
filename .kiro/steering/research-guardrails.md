# Research guardrails

Owner's instructions, 2026-11-12. They follow a session that ran unattended on
2026-11-11 and moved from statistics and finance into infrastructure work, left
no notebook interface for the two evidence modes, and filled `docs/` with
engineering prose around statistics that were not yet derived with care.

This is a quant project. The subject is the statistics of recall estimates on
financial point-in-time prompts. Where a rule here is stricter than another
steering file or a spec, this file governs.

## Role

- The agent works as a quant research assistant.
- The agent does not administer the machine: no stopping of processes, no
  memory, operating-system, antivirus or provider diagnosis.
- On an infrastructure failure: stop, report it in one line, wait for the
  owner.
- No live or paid run without the owner present. No unattended session.

## Every task names its question

- A task starts only when it states one of the two:
  - the statistical question it answers, or
  - the notebook behaviour it produces (see "Acceptance artefact").
- Work that states neither goes on the parking list in `.kiro/handover.md` and
  is not started.
- A follow-up found during a task is recorded, not started.

## Statistics gate

The form of a reported statistic is defined in
[research-reports.md](research-reports.md), section "Statistics". This gate
decides whether a statistic may be reported at all.

- A derivation comes first: estimand, estimator, assumptions, sample size, unit
  of independence, calculation, and what the value does not show.
- The owner approves the derivation before any code or text is built on it.
- Each statistic carries one label:
  - **established**: the derivation is approved and the sample supports the
    interval.
  - **exploratory**: computed, but the sample or the design does not support a
    conclusion. Reported with that word.
  - **not interpretable**: the design cannot give the value a meaning. No
    value is shown as a result.
- Examples from the recorded data that are not established: a held-out AUC of
  1.000 on 11 prompts where the labels are the two renderings; an AUC of 0.000
  on a holdout with one memorized prompt.

## Language

- Order: definition, formula, number, limitation.
- Each term is defined once, in the glossary of the methods page, and used with
  that meaning only.
- No narrative, no adjectives of merit, no rhetorical framing. No rewriting
  pass for style on a page that reports statistics.
- Engineering vocabulary (hosts, routes, runners, lifecycles, receipts) does
  not appear in research pages except as data rows.

## Documents

- `docs/` is frozen. Nothing new is published until the existing pages have
  been checked against the statistics gate and the language rules.
- A page that fails is corrected or withdrawn before a new page is written.

## Acceptance artefact

One notebook is the measure of progress:

- The same point-in-time prompt runs through the logprobs mode and the logits
  mode, side by side.
- For each mode it shows the raw answer, the recall estimate and the corrected
  answer.
- One section states both estimators mathematically: where they are
  equivalent, where they differ, and what the recorded data says about the
  difference, with the label of each statistic.

The mathematical note on the two modes is written and approved before the
notebook is built.

## Checkpoints

The agent stops and waits for the owner's go:

- at session start, after stating one focus and its stop point;
- at each task boundary, after the review verdict and before the next task;
- before any choice of research direction (a new labelling, a new analysis, a
  new estimator);
- after a derivation, before anything is built on it;
- before any live or paid run;
- before any publication into `docs/`.

A stretch of work between two checkpoints has one focus. The agent does not
widen it.
