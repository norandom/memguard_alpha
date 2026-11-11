# Composition blueprint: the Recall Guard research record

First pass from the COBESY composition back-end. This is a skeleton: governing
idea, grid of claim headlines, bullets and cues. It holds no report prose. Prose
is written only after the owner accepts the grid, and every rewritten passage
then goes through the humanizer pass.

Numbers in the bullets are the ones already measured (register IDs in brackets).
Nothing here changes a number.

## 0. Format

Thesis / long form: one synthesis, a register and dated reports that must all
carry the same idea.

## 1. The governing idea

- One sentence: Recall Guard qualifies models from a catalogue to give
  recall-aware answers, in one of two aligned evidence modes: logprobs where the
  model returns them, logits where it does not and its weights can be run.
- Where it must be visible: the synthesis title and first paragraph, the
  register's purpose, the summary of every dated report, each section head.
- The claim, not the catalog of runs: the record argues which models are
  qualified, in which mode, and on what evidence. A run on a particular model or
  API is a row of data under that claim, not a phase of the research.

## 2. The four keys

- Audience: the project owner, then an adopter choosing a model. Knows the
  method. Has read the current reports and rejects them as dumps.
- Purpose: after one screen the reader can say which models are qualified, in
  which mode, what to use when a model has no logprobs, and what is still
  unproven.
- Message: the catalogue is the result. Two modes, one method. Today no entry is
  qualified on exposure-labelled data, and the foundation that would make a
  qualification trustworthy is missing.
- Style and tone: plain, short sentences, numbers with denominators, method
  names only where the reader needs them, no em dashes.
- Action wanted: accept or correct the grid in section 4, then approve the
  requirements.

## 3. The opening

- Protagonist: someone who wants an LLM answer about a dated fact and has to
  choose a model for it.
- What is happening: some models return logprobs and some do not; the guard
  needs evidence from one of the two modes; only a calibrated model can be
  trusted to flag recall.
- What is at stake: choosing an unqualified model gives an adjustment that
  follows the prompt's wording, discounts honest answers and passes recalled
  ones.
- Opening guidance: the catalogue status first, in two or three sentences; then
  the catalogue table; then what is missing.
- Words to avoid in the opening: phase names, arm names, gate, pilot, DiD, Holm,
  bootstrap, variant ids, runtime names.

## 4. Skeleton grid for the synthesis

Each head is a claim. Read top to bottom, the heads are the argument.

| # | Head | Bullets to cover | Emphasis and reader cue | Keywords | Serves the governing idea |
|---|---|---|---|---|---|
| 1 | Recall Guard qualifies models for recall-aware answers | What a recall-aware answer is: answer, recall estimate, adjustment, flag when the estimate is not valid. What qualified means: a calibrator fitted for the model in a mode, passing the quality rule on held-out data. Why a 50:50 question is the right test bed. | Define once, in the reader's words. Forecast accuracy is not the goal. | recall-aware answer, qualified, calibrated, catalogue | yes |
| 2 | Two evidence modes, one method | Logprobs mode: top-k logprobs from the serving stack. Logits mode: the full next-token distribution from weights run locally. Same features, same calibration, same qualification rule, same interface. | Two-row table: what each mode needs, where it works. | logprobs mode, logits mode, aligned | yes |
| 3 | The modes agree on the same weights | Top-k estimate against full distribution as k grows: mean absolute difference 0.53 at k 1, 0.11 at k 5, 0.03 at k 50 [R43]. Held-out quality side by side: Phase A documents 0.57 against 0.56 [R26, R40]; EDGAR filings 0.59 against 0.53 [R43]. Where they part: macro prompts 0.72 against 1.00, where the label is the prompt wording [R43]. | One table. State what the 1.00 does and does not mean. | convergence, like for like, same weights | yes |
| 4 | The catalogue today | One row per model. Columns per mode: available, calibrator fitted, held-out AUC with interval, passed or weak, evidence IDs. Filled from R26, R30, R38, R40, R41, R43. Status column: qualified, not qualified, not tested. | This is the core table of the record and comes first after the opening. Model and API names appear here as data and nowhere else as headings. | catalogue, entry, status | yes |
| 5 | Without logprobs, logits can stand in for these models | Which entries have a logits mode today (weights run locally with a full-distribution host) and which cannot (hosted APIs without weights). What an entry needs to gain the logits mode. | A short list derived from the catalogue. Answer the question in the first sentence. | no logprobs, logits mode, local weights | yes |
| 6 | No entry is qualified on exposure-labelled data yet | Every held-out AUC on text labelled by real exposure includes 0.5, in both modes. The macro calibrators pass the quality rule, but their label is the prompt wording, and the estimate does not fall after the cutoff. | Say it plainly. This is the honest status of the catalogue. | at chance, exposure label, prompt wording | yes |
| 7 | The foundation a qualification needs is missing | No cases of known recall, so neither mode has been checked against what it estimates. No repeat-the-prompt baseline per runtime and temperature: at 0.7 locally answers vary on 26 of 29 prompts; at 0 on a served stack 31 of 35 prompts return one answer [R43 draws]. | Plain figure: x "different answers among the draws of one prompt", y "number of prompts". Name what would supply the truth. | known recall, ground truth, repeat baseline | yes |
| 8 | How the rest of the work counts | Classification table of all register entries: qualification evidence, mode alignment, foundation, maintenance (model catalog update), convenience (NPU route), not aligned (gates, accuracy tests, negative control, mitigation track). | A table, one line per entry. Not aligned entries stay in the record with their cause. | maintenance, convenience, not aligned | yes |
| 9 | Next step | Known-recall cases, run in both modes on the same weights, to qualify a first entry. Nothing paid. | Two items at most. | next step | yes |

- MECE check: head 2 is the method, 3 its alignment, 4 the result, 5 and 6 two
  readings of the result, 7 what the result still lacks, 8 everything else.
- Post-outline: to be done on the prose, one topic sentence per head.

### Summary block for every dated report

Placed under the register line, before Question. Four lines, in this order:

| Line | Content | Rule |
|---|---|---|
| Result | What was measured and what came out, with the one number that matters. | No method names. One or two sentences. |
| Catalogue effect | Which entry and mode this run informs, and whether it changes the entry's status. | "None" is a valid answer. |
| Not shown | What the run cannot tell. | One sentence. |
| Class | qualification evidence, mode alignment, foundation, maintenance, convenience or not aligned. | From the fixed list. |

### Register

- Purpose paragraph: restated from the governing idea. The sentence that serving
  stacks are nondeterministic at temperature 0 gets a dated note with the R43
  count (31 of 35 prompts identical).
- State of evidence: replaced by the catalogue table and heads 5 to 7. Old items
  stay as dated entries.
- Index table: two new columns, Class and Catalogue entry.

### Figures

- One message per figure. Both axes labelled with units. A table view beside it.
- F1 (accuracy gap) and F3 (per-draw distribution) have no y-axis label today.
- New plain figure for head 7, from the recorded draws.
- The accuracy-gap figure states that its variants cannot differ in accuracy,
  because the adjustment never changes the direction of the call.

## 5. Paragraph moves

- The "they say": the current record says the design behaves as expected, phase
  by phase, model by model.
- Strongest objection: an AUC of 1.00 in logits mode on macro prompts looks like
  a qualified entry.
- Answer: the label there is the prompt wording, and the estimate stays high
  after the cutoff. It shows that two wordings are told apart, not that a month
  was recalled.
- So what: until an entry is qualified against known recall, the catalogue can
  say which modes are available, not which models can be trusted.
- Emphasis: the catalogue table and the missing foundation. Everything else is
  support.
- Style guidance: subject first, verb second, number third. One idea per
  sentence. Units and denominators in tables.
- Writing advice: write the summary last, from the finished tables. Cut any
  sentence that names a method the summary does not need.
- Reports are not summaries of specs. A spec is one technical way to get
  reportable data. Report the result; link the spec in the references only.
- Trim: state each fact once per report. Summary, tables, limits and evidence
  must not repeat one another.
- No marketing language, no significance claims, no upbeat ending.
- This is quant research. No operating system, host memory, antivirus, exit
  codes, ports or tooling incidents in the record. If an incident cost data,
  one sentence says what is missing and what that does to the result.

## 6. Linter pass on the current record

| Malady | Grain | Hit | Fix in the restructure |
|---|---|---|---|
| First-paragraph drop-off | Opening | yes: R43 opens with a 13-line sentence of statistics | Four-line summary block |
| Information overload | Macro | yes: every table and every caveat at one level | Catalogue table first, detail below, receipts last |
| Mismatched TOC | Macro | yes: heads are phases, arms and runtimes; the argument is a catalogue in two modes | Heads from the grid |
| Scope creep | Macro | yes: gates, power, negative control and single models read as results | Classified, moved under head 8 |
| Flat exposition, no stakes | Meso | yes | Opening states what an unqualified model costs |
| Expert fog | Micro | yes: variant ids and method names in running text | Plain names in text, ids in tables |
| Sentence sprawl | Micro | yes | One idea per sentence |
| Reader mismatch | Opening | yes: written for an auditor, read by the owner and adopters | Summary for the reader, audit trail in Evidence |
| AI slop, generic voice | Output | to check on the prose | Humanizer pass on every rewritten passage |

## 7. Format checks (long form)

- [ ] The heads read as the argument, top to bottom
- [ ] The governing idea is traceable through title, opening, each head and the next step
- [ ] Post-outlined per section
- [ ] Why before what; reference detail in Evidence
- [ ] Scope fixed; not aligned material sits under head 8

## 8. Open points for the owner

1. Is the governing idea in section 1 the right one?
2. Is the fixed list of classes right: qualification evidence, mode alignment,
   foundation, maintenance, convenience, not aligned?
3. What counts as a case of known recall? The blueprint names the gap and does
   not define the cases.
4. Is the catalogue keyed by model (weights), with APIs and runtimes as
   attributes of an entry?
