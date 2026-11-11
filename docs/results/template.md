# Report template

Every dated report in `docs/results/` uses this structure. A report states a
result and how certain it is. The [research register](../research-register.md)
holds the one-line index, and the catalogue holds the status of each model and
serving route.

```markdown
# <What was measured>, <YYYY-MM-DD>

> Register: R<nn> · Status: measured | feasibility | diagnostic | failed | decision

| Summary | |
|---|---|
| Result | <What was measured and what came out, with the one number that matters, its interval and its sample size. One or two sentences.> |
| Catalogue effect | <The catalogue entry and mode this run informs, and whether the entry's status changes. Write "none" when it informs no entry.> |
| Not shown | <What the run cannot tell. One sentence.> |
| Class | <one class from the fixed list> |

## Question

<The single question the run set out to answer.>

## Methods

<The statistics used, in the order they appear in the results. Name each one
and link its definition: [<statistic>](../research-methods.md#<anchor>). State
the resamples and the seed. For each statistic state:

- the estimand in words: what quantity, over which units, under which condition
- the calculation, with its inputs and their counts
- the unit of analysis, the sample size and any dependence between units
- the interval, with its method and level
- for each test: the null hypothesis, the test, the statistic, the p-value and
  the adjustment for its family
- for each comparison: an effect size with its interval, never a p-value alone>

## Data

<Corpus, splits and sizes, date ranges, filters.>

## Results

<Tables from which a reader can recompute each value: the value, its interval,
the sample size, and the inputs with their counts. Then at most five bullets on
what the numbers show.>

## Limits

<What the run cannot establish. State the detectable effect for each main
comparison at its sample size. For missing data, one sentence on what is
missing and what that does to the result.>

## References

<The register entry, the catalogue entry, the spec that produced the data (link
only), the operational notes when an incident removed data (link only), private
receipts by path, code paths, commits.>
```

The summary block is a two-column table with the header row `| Summary | |`
and exactly these four rows in this order. Each row is one line. The class is
written in lower case, as listed here.

Classes: qualification evidence, mode comparison, foundation, maintenance,
convenience, not aligned.

Each statistic is defined once, on the methods page
(`docs/research-methods.md`). A report names the statistic and links there. It
does not restate the formula.

Rules:

- Lead with the result. Write the summary block last, from the finished tables.
- Keep numbers exactly as measured, with intervals and denominators.
- Each fact once per report. The summary, the tables, the limits and the
  references do not repeat one another.
- Values that are equal by construction are stated as such and are not tested.
- A statistic that cannot be computed names the missing input and shows no value.
- Model, provider and route names are data in tables, never section heads or
  phase names.
- A report states results and what they mean. It does not narrate how a spec was
  executed: no tasks, gates, sessions, attempts or fixes. Link the spec in
  References and nowhere else.
- Not in the record: the operating system, host memory, antivirus, process exit
  codes, ports, driver scripts, serving-stack troubleshooting and tooling
  incidents.
- Figures use the one academic style: a numbered caption below the figure
  ("Figure N."), both axes labelled with units, interval bars where an estimate
  has an interval, a table view of the plotted values, no light and dark variants.
- A figure's caption states the sample size and what the interval is. Series
  that cannot differ by construction are noted in the caption.
- A later finding never rewrites an earlier report. Add a dated note under the
  summary block: `> **Later evidence (YYYY-MM-DD):** ...`.
- Plain prose: no em dashes, no marketing language.
