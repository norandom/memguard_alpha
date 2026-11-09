# Research reports

User instruction, 2026-11-09. Every research result is reported the same way, so
results can be compared, cited and audited.

- Write every dated report in `docs/results/` with the structure in
  [docs/results/template.md](../../docs/results/template.md): title
  "<what was measured>, YYYY-MM-DD", a register line (ID, status, track), a
  summary that leads with the result, then Question, Setup, Data, Results,
  Limits, Evidence. Do not invent other section layouts.
- Give each report a numbered entry in `docs/research-register.md`, in date
  order, with the same six fields (question, setup, data, result, limits,
  evidence). Update the register's state of evidence when a result changes it.
- Link every report from MkDocs: add it under **Research > Dated reports** in
  `mkdocs.yml`, in date order, and add a row to `docs/results/index.md`. Keep
  `docs/research-report.md` (the synthesis) current when a result changes what
  it says.
- Keep numbers exactly as measured, with intervals and denominators. Never edit
  an earlier report's findings; add a dated note under its summary:
  `> **Later evidence (YYYY-MM-DD):** ...`.
- Record failed, stopped and disqualified runs too, with their cause.
- Plain prose: no em dashes, no marketing language. Use COBESY's composition
  back-end when structuring a new synthesis or a substantially rewritten report.
- Before committing docs, run the strict MkDocs build and `tests/docs`.
- Private receipts and per-token scores stay under `runs/` and are never
  published; reports cite their paths.
