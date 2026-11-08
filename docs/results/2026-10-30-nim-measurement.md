# Fresh calibration worked; divergence mitigation remains unmeasured

> **Later evidence (2026-11-08):** this report's calibration corpus separates its classes by publication date and source as well as by memorization, so the AUC reflects corpus separation. On a clean SEC EDGAR corpus, held-out results show no single-draw skill for GPT-OSS-20B or three local models. See [Phase A](2026-11-08-phase-a-single-draw.md) and the [research register](../research-register.md) (R24, R26, R30).


Measurement date: 2026-10-30. Package: Recall Guard v0.5.0.

This run gives adopters a measured starting point for hosted scoring with
`openai/gpt-oss-20b`. The fresh classifier separated the supplied calibration
labels with ROC-AUC 0.876 and discounted response confidence. Directional
accuracy stayed near the majority baseline. Divergence decoding was inactive,
so this run provides no estimate of its effect on memorization.

## What the hosted run measured

The harness built a control baseline, fitted a model-specific MCS classifier
and scored 330 ETF evaluation prompts through NVIDIA NIM. It used no reference
model. One control-baseline response was unusable; all calibration responses
were usable.

| Measurement | Result |
| --- | --- |
| Control baseline | 99 of 100 usable responses |
| Calibration data | 40 in-sample and 100 out-of-sample responses |
| Held-out calibration ROC-AUC | 0.876; `is_weak=false` |
| Parsed evaluation responses | 312 of 330 (94.55%) |
| Failures | 18 `parse_failure` records |
| Correct directions | 170 of 312 parsed responses |
| Directional accuracy | 54.49%; 95% bootstrap interval 48.72% to 60.26% |
| Majority baseline | 53.94% on all 330 targets; interval 48.48% to 59.70% |
| Ranking outcome | `not-better-than-baseline`; composite score 0 |
| Elapsed time | 2,272.6 seconds, approximately 38 minutes |

The accuracy gate requires the model's lower confidence bound to exceed the
majority baseline's upper bound. Here 48.72% is below 59.70%, so the model did
not qualify. Model accuracy uses parsed responses; the majority baseline uses
all evaluation targets. Failed responses are visible in the downloadable
records rather than counted as correct or silently dropped from the parse rate.

Calibration AUC measures separation of corpus labels. Publication dates do
not prove training-set membership, and differences in source, format or period
can explain separation. The AUC does not establish an individual response's
probability of memorization. The summary repeats 0.876 at both AUC interval
endpoints because the reporting path receives no holdout records to bootstrap.
Those endpoints represent a fallback point estimate, not measured uncertainty.

## The scorer reduced confidence without changing predictions

On the 312 parsed responses, the mean raw confidence was 0.5992 and the mean
discounted confidence was 0.2219. The scorer applies this formula to each row:

```text
discounted_confidence = raw_confidence * (1 - p_memorized)
```

The mean MCS output was 0.6283. This is a classifier output, not an observed
contamination rate. Lower confidence is the direct arithmetic effect of the
discount; this run does not establish improved confidence calibration. The
scorer preserves each predicted direction, so raw and MemGuard directional
accuracy coincide by construction.

The baseline and classifier were fitted afresh and used for this run. The
harness exports results and provenance, not fitted parameters that an
application can reload. A subsequent run recalibrates.

## Divergence decoding can change the next token

The optional decoder implements the equation discussed by James Somers in
Jane Street's [Mitigating memorization in LLMs](https://blog.janestreet.com/mitigating-memorization-in-llms/)
(September 30, 2026):

```text
adjusted_logits = main_logits + alpha * (older_logits - newer_logits)
```

For a shared token prefix, the auxiliary difference raises scores that the
older auxiliary favors relative to the newer one, and lowers scores with the
opposite relationship. `alpha` controls the adjustment; zero preserves the
main vector. Greedy selection can therefore choose a different next token.
For example, main logits `[2, 0, 0]`, older logits `[0, 3, 0]` and newer
logits `[0, 0, 0]` produce `[2, 3, 0]` at `alpha=1`: the selected token
changes from ID 0 to ID 1. The [executable example](../divergence-decoding.md#runnable-example)
checks this behavior.

Jane Street reported reduced pre-cutoff cricket performance with preserved
post-cutoff performance in its experiment. Our equation and callback-loop
tests establish implementation behavior. They do not reproduce that empirical
result for our models or ETF task. No model weights change, and the operation
does not establish that particular facts have been erased.

### Completion-only APIs need a different integration or method

An endpoint can return text without exposing logprobs and still apply this
equation inside its server. It needs the three compatible next-token vectors
and control of generation there. Recall Guard does not currently provide that
server integration.

Ordinary completed text from OpenAI, Anthropic or another provider does not
contain the information or controls needed to apply the equation on the client.
The hosted NIM run used completed responses and their logprobs for scoring;
it could not steer the provider's next token. Its measured confidence discount
therefore says nothing about divergence decoding's effectiveness.

Support for arbitrary completion APIs is an unmet requirement. Prompt
rewriting is one API-compatible candidate discussed in the Jane Street blog,
but it is a separate experimental method and is not implemented in this release.

## What would establish a mitigation benefit

An evaluation needs an unsteered baseline and a steered run on the same
prompts, using compatible tokenizers and verified older/newer auxiliary
training histories. It must examine matched pre-cutoff and post-cutoff cohorts
and report useful task performance, recall evidence, parsing failures and
latency together. Any MCS comparison also needs calibration matched to the
generation settings; this unsteered calibrator cannot simply be reused.

The hypothesis fails if apparent recall reduction comes from lost task
ability, changed factual inputs or more failed responses. A reduction in
historical accuracy alone is insufficient. A completion-only rewriting
experiment would additionally need checks that the rewritten prompt preserves
the information available at the decision time.

## Download and reproduce the measurement

The following snapshot contains all evaluation records and uses portable
relative artifact paths:

- [Summary CSV](2026-10-30-gpt-oss-20b/summary.csv)
- [All 330 evaluation records, JSONL](2026-10-30-gpt-oss-20b/records.jsonl)
- [Manifest with input hashes, JSON](2026-10-30-gpt-oss-20b/manifest.json)
- [Run settings and verified metrics, JSON](2026-10-30-gpt-oss-20b/measurement-setup.json)

The run used source commit `36b3b1914ba5bca4dad41a2a5366edb2990fb6b3` on native
Windows, with temperature 0, a 512-token limit, four workers, request starts
spaced by at least 1.5 seconds, seed 0 and 1,000 bootstrap resamples. The
registry cutoff for this model was `2024-06-30`. It ran from approximately
18:00:20 to 18:38:12 Europe/Berlin and exited with code 0.

From that revision with `NVIDIA_API_KEY` configured locally, this command works
in PowerShell and bash:

```text
uv run python harness.py --eval-set data/eval/etf_portfolio.jsonl --is-memorized data/calibration/is_memorized.jsonl --oos-control data/calibration/oos_control.jsonl --cutoffs data/cutoffs.yaml --shortlist openai/gpt-oss-20b --no-reference --max-workers 4 --min-call-interval 1.5 --out-dir runs/nim-measurement
```

A rerun makes fresh provider calls and may differ even with identical settings.
The saved snapshot preserves this run. Independent record checks reproduced
its parse rate, accuracy, failure counts, majority baseline and per-response
confidence formula; input hashes matched the manifest.
