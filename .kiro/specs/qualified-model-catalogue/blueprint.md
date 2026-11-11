# Composition blueprint: the Recall Guard research record

Second pass from the COBESY composition back-end, after the owner's answers of
2026-11-11. This is a skeleton: governing idea, grid of claim headlines, bullets
and cues. It holds no report prose. Prose is written only after the owner
accepts the grid, and every rewritten passage then goes through the humanizer
pass.

Numbers in the bullets are the ones already measured (register IDs in brackets).
Nothing here changes a number.

## 0. Format

Academic article, long form: one synthesis, a register and dated reports that
carry the same idea. Quant research; the reader expects methods and statistics.

## 1. The governing idea

- One sentence: for point-in-time analysis of financial time series inside a
  model's training range, Recall Guard corrects an answer for recall with a
  heuristic built on logprobs, and a catalogue says which model and serving
  route combinations are qualified for that; the question this record answers
  is whether logits improve that correction or can replace it where a route
  returns no logprobs.
- Where it must be visible: the synthesis title and first paragraph, the
  register's purpose, the summary of every dated report, each section head.
- The claim, not the catalogue of runs: a run on a particular model or API is a
  row of data. A spec is how the data was obtained and is linked in the
  references only.

## 2. The four keys

- Audience: the project owner and quant researchers. They know point-in-time
  simulation, lookahead bias, Black-Litterman tilts and hypothesis tests. They
  reject the current reports as dumps.
- Purpose: after one screen the reader can say which entries are qualified, in
  which mode, whether logits help, and what is still unproven. After the
  methods section the reader can recompute any reported value.
- Message: the catalogue is the result; logprobs give the working correction;
  the evidence on logits is stated with its statistics.
- Style and tone: a research paper. Plain, short sentences, numbers with sample
  sizes and intervals, methods named where they are used, no em dashes.
- Action wanted: accept or correct the grid in section 4, then approve the
  requirements.

## 3. The opening

- Protagonist: a quant who runs a point-in-time simulation with an LLM in the
  loop.
- What is happening: the simulation is fenced against lookahead, but for dates
  inside its training range the model can be certain what comes tomorrow.
- What is at stake: a tilt taken on a recalled answer is lookahead bias with a
  position size attached.
- Opening guidance: the catalogue status and the answer on logits first, in two
  or three sentences; then the catalogue table; then what is missing.
- Words to avoid in the opening: phase names, arm names, gate, pilot, variant
  ids, runtime names, anything about hosts or tooling.

## 4. Skeleton grid for the synthesis

Each head is a claim. Read top to bottom, the heads are the argument.

| # | Head | Bullets to cover | Emphasis and reader cue | Keywords | Serves the governing idea |
|---|---|---|---|---|---|
| 1 | An LLM inside its training range breaks the point-in-time fence | Lookahead in a point-in-time simulation. Why a model can be certain what comes tomorrow. What a recall-aware answer is: answer, recall estimate, correction, status. Why a 50:50 question is the right test bed. | Two short paragraphs. Recall is about the time series, not the prompt wording. | point-in-time, lookahead, training range, recall-aware answer | yes |
| 2 | The correction is a heuristic on logprobs | Features from the answer's logprobs, a calibrator per model, `p_memorized`, the tilt scaled by `1 - p_memorized`. The identifying prompt as the proxy that elicits recall. | One equation, one small diagram. Link `docs/pit-architecture.md`; do not repeat it. | logprobs, calibrator, p_memorized, softened tilt, proxy | yes |
| 3 | The catalogue: which entries are qualified | One row per model on a serving route. Per mode: available, calibrator fitted, held-out quality with interval and sample sizes, passed or not, recall boundary, evidence IDs. Status: qualified, not qualified, not tested. The qualification rule stated once. | First table of the record. Model and API names appear here as data. | catalogue, entry, serving route, qualification rule, status | yes |
| 4 | Do logits help | Same features and calibration in both modes. Paired comparison on the same weights and prompts: quality of each mode, difference with interval, agreement of the two estimates, difference in the softened tilt. Convergence as more alternatives are kept: 0.53 at k 1, 0.11 at k 5, 0.03 at k 50 [R43]. One of three answers: improve, match and can stand in, or undecided. | The research question. State the answer in the head's first sentence, then the statistics. | logits, paired comparison, agreement, convergence | yes |
| 5 | Routes without logprobs | Which entries have a logits mode and which cannot. What an entry needs to gain it. | A short list derived from the catalogue. | no logprobs, logits mode | yes |
| 6 | Inside and after the cutoff | The recall estimate for analysis dates inside the training range against dates after it, with a test of the difference [R43]. What it means for an entry's status when the estimate does not fall. | The check that the estimate tracks the time series and not the proxy. | training cutoff, natural experiment | yes |
| 7 | How much answers vary on a repeated prompt | Different answers per prompt, by serving route and temperature: at 0.7 answers vary on 26 of 29 prompts; at 0 on a served stack 31 of 35 prompts return one answer [R43 draws]. | One figure in journal style, one table. | repeat baseline, nondeterminism | yes |
| 8 | Methods | Each statistic in the order it appears: estimand, formula, unit of analysis, sample, interval method, test, adjustment, power. One worked example per kind. | A reader can recompute a table value from this section. | estimand, interval, test, effect size, power | yes |
| 9 | What is not shown | Evidence still missing for a trustworthy qualification. Limits of the recorded samples. | Short. No new runs defined here. | limits | yes |
| 10 | Register classes | Every entry classed: qualification evidence, mode comparison, foundation, maintenance (model catalog update), convenience (NPU route), not aligned. | A table, one line per entry. | maintenance, convenience, not aligned | yes |

- MECE check: heads 1 and 2 state the problem and the method, 3 is the result,
  4 to 7 are the analyses behind it, 8 the methods, 9 the limits, 10 the
  bookkeeping.
- Post-outline: to be done on the prose, one topic sentence per head.

### Summary block for every dated report

Placed under the register line, before Question. Four lines, in this order:

| Line | Content | Rule |
|---|---|---|
| Result | What was measured and what came out, with the one number that matters, its interval and sample size. | One or two sentences. |
| Catalogue effect | Which entry and mode this run informs, and whether it changes the entry's status. | "None" is a valid answer. |
| Not shown | What the run cannot tell. | One sentence. |
| Class | qualification evidence, mode comparison, foundation, maintenance, convenience or not aligned. | From the fixed list. |

Report sections after the block: Question, Methods, Data, Results, Limits,
References. Setup detail that a reader does not need to judge the result moves
to References as a link.

### Statistics

- Every value: estimand in words, formula, inputs with counts, unit of
  analysis, interval with method and level.
- Every test: null, test, statistic, p-value, family adjustment.
- Every comparison: effect size with interval; never a p-value alone.
- Values equal by construction are stated as such, not tested.
- Power or detectable effect for each main comparison.
- One worked example per kind of statistic.

### Register

- Purpose paragraph: restated from the governing idea.
- State of evidence: replaced by the catalogue table and heads 4 to 7. Old
  items stay as dated entries.
- Index table: two new columns, Class and Catalogue entry.

### Figures

- Academic style, as in a quantitative journal: white background, black axes
  and text, serif or standard scientific typeface, series told apart by marker,
  line style or grey level, interval bars where an estimate has one, numbered
  caption below ("Figure N.") with sample size. No title inside the plot, no
  decorative colour, no rounded bars, no value label on every mark, no light
  and dark variants.
- Both axes labelled with units. One message per figure. A table view.

## 5. Paragraph moves

- The "they say": a point-in-time prompt that hides dates and tickers is enough
  to prevent lookahead.
- Strongest objection: the recall estimate separates the identifying prompt
  from the fenced one, so it works.
- Answer: that contrast is the proxy. The estimate has to follow the time
  series, which is what the inside-and-after-cutoff test checks.
- So what: a qualified entry lets a quant size the correction; an unqualified
  one leaves the lookahead unpriced.
- Emphasis: the catalogue table, the answer on logits, the methods.
- Style guidance: subject first, verb second, number third. One idea per
  sentence. Sample sizes and intervals with every number.
- Writing advice: write the summary last, from the finished tables.
- Reports are not summaries of specs. Report the result; link the spec in the
  references only.
- Trim: state each fact once per report.
- This is quant research. No operating system, host memory, antivirus, exit
  codes, ports or tooling incidents. If an incident cost data, one sentence
  says what is missing and what that does to the result.
- No marketing language, no significance claims, no upbeat ending.

## 6. Linter pass on the current record

| Malady | Grain | Hit | Fix in the restructure |
|---|---|---|---|
| First-paragraph drop-off | Opening | yes: R43 opens with a 13-line sentence of statistics | Four-line summary block |
| Information overload | Macro | yes: every table and every caveat at one level | Catalogue table first, analyses below, references last |
| Mismatched TOC | Macro | yes: heads are phases, arms and runtimes | Heads from the grid |
| Scope creep | Macro | yes: gates, incidents and single models read as results | Classed, or moved to references |
| Flat exposition, no stakes | Meso | yes | Opening states the lookahead problem |
| Expert fog | Micro | yes: variant ids in running text, statistics without their calculation | Plain names in text, methods section, worked examples |
| Sentence sprawl | Micro | yes | One idea per sentence |
| Reader mismatch | Opening | yes: written as an audit trail, read by a quant | Results and methods for the reader, audit trail linked |
| AI slop, generic voice | Output | to check on the prose | Humanizer pass on every rewritten passage |

## 7. Format checks (academic article)

- [ ] One idea, stated as the contribution in title, summary and first paragraph
- [ ] The summary is the argument in miniature: context, question, answer, evidence, limits
- [ ] The strongest objection is named and answered in the text
- [ ] Evidence selected for the claim, not dumped
- [ ] Section heads carry claims; topic sentences post-outlined to match
- [ ] Every statistic can be recomputed from the methods section

## 8. Points settled by the owner (2026-11-11)

- Governing idea: catalogue first.
- Catalogue entry: model plus serving route.
- Recall: recall of financial time series inside the model's training range,
  seen in the model's certainty (logprobs) about what comes next; the prompt
  contrast is a proxy to elicit it and to extract logits.
- Research question: whether logits help at all, or whether a similar system
  can be built from them.
- Reports need a statistical analysis whose calculation is clear.

## 9. Open point

The qualification rule. The existing certification screen checks a model on a
controlled recall boundary (recall AUC with a permutation p-value). The design
phase states that rule once and says how the inside-and-after-cutoff result
enters it.
