# Research reports

User instructions, 2026-11-09 and 2026-11-11.

The record serves one objective: recall-aware answers for point-in-time analysis
of financial time series. The catalogue (`docs/catalogue.md`, data in
`recall_guard/core/catalogue_data.json`) says which combinations of model and
serving route are qualified, per evidence mode (logprobs, logits).

## Form

- Write every dated report in `docs/results/` with the structure in
  [docs/results/template.md](../../docs/results/template.md): title
  "<what was measured>, YYYY-MM-DD", the register line (ID and status), the
  four-line summary block (Result, Catalogue effect, Not shown, Class), then
  Question, Methods, Data, Results, Limits, References. Do not invent other
  layouts.
- Give each report a numbered entry in `docs/research-register.md`, in date
  order, with its class and its catalogue entry in the index.
- Classes: qualification evidence, mode comparison, foundation, maintenance,
  convenience, not aligned. Each register entry carries one. The model catalog
  update is maintenance. The NPU route is convenience.
- Link every report from MkDocs: add it under **Research > Dated reports** in
  `mkdocs.yml`, in date order, and add a row to `docs/results/index.md`. Keep
  `docs/research-report.md` (the synthesis) current when a result changes what
  it says.

## Catalogue

- A new run updates the catalogue entry it informs (the data file and the
  rendered page), or states that it informs none.
- A new model or serving route is a catalogue entry, not a research phase.
- The route a run used is recorded in the catalogue entry it informs, not in a
  track.
- Model, provider and route names are data in tables. They are never section
  heads or phase names.

## Statistics

- Every reported value carries: the estimand in words, the calculation with its
  inputs and their counts, the unit of analysis, the sample size and any
  dependence between units, and an interval with its method and level.
- Every test states its null hypothesis, the test, the statistic, the p-value
  and the adjustment for its family.
- Every comparison carries an effect size with its interval, never a p-value
  alone. A report states the detectable effect for each main comparison at its
  sample size.
- A report meets this by naming each statistic and linking its definition on
  the methods page (`docs/research-methods.md`). The Methods section lists the
  statistics in the order they appear in the results.
- Values equal by construction are stated as such and are not tested. A
  statistic that cannot be computed names the missing input and shows no value.
- Chance-level forecast accuracy is the intended design condition of the macro
  question. Never report it as a finding about the guard.

## Content

- Reports are quant results. State the result and what it means, each fact once
  per report.
- Not in the record: the operating system, host memory, antivirus, process exit
  codes, ports, driver scripts, serving-stack troubleshooting and tooling
  incidents.
- Operational detail goes to the operational notes: the spec's implementation
  notes and `.kiro/handover.md`. A report links the spec in References and
  never retells it: no tasks, gates, sessions, attempts or fixes.
- If an incident cost data, one sentence says what is missing and what that
  does to the result, and References links the operational notes.
- Figures use the one academic style: numbered caption below ("Figure N."),
  both axes labelled with units, interval bars, a table view, no light and dark
  variants.
- A figure's caption states the sample size and what the interval is. Series
  that cannot differ by construction are noted in the caption.
- Private receipts and per-token scores stay under `runs/` and are never
  published. Reports cite their paths.

## Integrity

- Keep numbers exactly as measured, with intervals and denominators.
- Never edit an earlier report's findings. Add a dated note under its summary
  block: `> **Later evidence (YYYY-MM-DD):** ...`.
- Record failed, stopped and disqualified runs too, each with its cause in one
  sentence.

## Writing

- Structure first, prose second. A new synthesis or a substantially rewritten
  report starts from a COBESY composition blueprint (a skeleton of claim
  headlines). Every rewritten prose passage goes through the humanizer pass.
- Plain prose: no em dashes, no marketing language.

## Before and after a change

- Commit and push before and after a change to the record.
- Before committing docs, run the strict MkDocs build and `tests/docs`.
