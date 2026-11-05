# Anthropic fail-early prototype

User authorization: 2026-11-01, explicitly requested a quick prototype before
building the feature and authorized use of Anthropic. This is an isolated
discovery spike, not approval of the broader design/tasks.

Maximum 17 requests, sequential, no retries: eligibility plus eight draws for
each of two prompts. Existing configured model claude-opus-5-5, max_tokens=256,
60-second request timeout, five-minute admission deadline (one in-flight request
may finish after it), two-second request-start spacing. No NIM or auxiliary
provider calls. Usage recorded; monetary cost unknown.

Question: can completion samples be reweighted using an explicit categorical
auxiliary contrast without target logprobs, with usable support and weights?

Use named historical and anonymous hypothetical forecasts with the same 80/20
supplied probabilities. This is an instruction-following and sample-support probe,
not a dated pre/post-cutoff benchmark. Synthetic older counts A80/B20 and newer
counts A20/B80 define a known contrast. They are not evidence of temporal training
exposure and cannot establish that real auxiliary sourcing works.

Freeze alpha values 0, 0.5, 1. Report weighted categorical probabilities and
effective sample size. Stop after an item if either category has fewer than two
samples, or alpha=1 ESS <4 or probability change <0.10. Those are feasibility
heuristics, not statistical significance criteria. There is no alpha fit or
held-out efficacy claim. Eight draws cannot establish stable probabilities.

Offline checks: zero-alpha identity, known analytic reweighting, sham cancellation
through unit weights, and inability to create an unobserved category.
Live failures stop the spike; no repair retry consumes extra budget automatically.

A pass only supports building a more rigorous source-validation experiment.
A support failure means this small request budget cannot support this approach
for these prompts/settings. It does not prove general impossibility. No outcome
authorizes the full build.

## Observed prototype outcome

The authorized isolated Anthropic spike used 9/17 requests and stopped after
eight identical A answers failed the predeclared sample-support gate. The anonymous
control was unperformed. Reweighting changed no probabilities. Synthetic
auxiliary counts tested mechanics only; actual temporal sourcing and mitigation
remain untested. Do not expand implementation on this result or automatically
consume the unused allowance. See the
[dated report](../../../docs/results/2026-11-01-anthropic-reweighting-prototype.md).
