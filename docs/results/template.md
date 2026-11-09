# Report template

Every dated report in `docs/results/` uses this structure, so results can be
compared and cited the same way. The [research register](../research-register.md)
holds the one-line index; the report holds the detail.

```markdown
# <What was measured>, <YYYY-MM-DD>

> Register: R<nn> · Status: measured | feasibility | diagnostic | failed | decision ·
> Track: <Local | NPU | Modal | NIM | Completion | Data | Method | Platform>

<Two to four sentences: the result first, then what it means. State the
expected outcome when the design has one.>

## Question

<The single question the run set out to answer.>

## Setup

<Model and revision or digest, runtime and version, hardware, generation
settings, request counts, cost. A table when there are more than four fields.>

## Data

<Corpus, splits and sizes, date windows, filters.>

## Results

<Tables with intervals; then two to five bullets on what the numbers show.>

## Limits

<What the run cannot establish; small samples; known defects.>

## Evidence

<Receipts, companion JSON, code paths, commits, related register entries.>
```

Rules:

- Lead with the result. Keep numbers exactly as measured; give intervals and
  denominators.
- A later finding never rewrites an earlier report. Add a dated note under the
  summary instead: `> **Later evidence (YYYY-MM-DD):** ...`.
- Plain prose: no em dashes, no marketing language.
