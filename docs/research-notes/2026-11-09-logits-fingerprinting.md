# Logits identify a model as much as they measure it, 2026-11-09

Research note. Register entry R42, track: calibration identity and the
full-distribution host. Literature only; no new measurement.

## Summary

An adopter who runs the guarded decision against a hosted endpoint has one
exposure that no calibrator can absorb: the weights behind the endpoint change
and nobody tells them. The cause can be a quantised replacement, a silent
update or a different serving kernel. The calibrator keeps producing numbers,
but they describe a model that is no longer served.

The fingerprinting literature shows that the same evidence Recall Guard uses
for `p_memorized`, top-k log-probabilities, already carries the model's
identity. On OpenAI's production endpoints, researchers recovered the hidden
dimension and the final projection layer from top-k logprobs alone. For
Anthropic's models, which return no logprobs, only behavioural fingerprinting
exists. Two consequences for this project: identity must be bound explicitly,
by digest and calibration identity, never inferred from the scores; and the
full-distribution host returns more than those papers had, so a hidden-dimension
check of the served weights costs a few hundred positions.

## Question

Are logits and logprobs used to fingerprint models, has it been done on OpenAI
or Anthropic models, and what follows for a guard that calibrates on top-k
logprobs?

## What was shown on OpenAI models

The route to full logit vectors through a top-k API came first. Morris et al.
(2023) showed that an endpoint returning the top k log-probabilities can be made
to reveal the whole vector: a logit bias lifts any chosen token into the top k,
and the shift it needs gives that token's logprob relative to the others. A
vocabulary-sized vector costs a few hundred requests. They used it to invert
prompts from next-token distributions; the later fingerprinting work used it to
read the model.

Finlayson, Swayamdipta and Ren (2024) made the identity argument. A language
model's output distribution passes through a softmax over a linear projection
from the hidden state, so every logit vector the model can produce lies in a
subspace of the vocabulary space whose dimension is the hidden size. Enough full
vectors from one endpoint pin that subspace down. Its dimension gives the hidden
size, which they estimated at about 4096 for gpt-3.5-turbo, and the subspace
itself is a signature: two endpoints serving the same weights share it, two
different models do not, and a silent update moves it. They proposed the
subspace as an audit tool against undisclosed model changes.

Carlini et al. (2024) went one step further on the same access. Within the
subspace, the full logit vectors are images of hidden states under the final
projection, and enough of them recover that projection up to a symmetry. They
recovered the hidden dimension and the output layer of several OpenAI production
models, ada, babbage and gpt-3.5-turbo among them, for a few thousand dollars of
API calls, with the affected vendors' agreement. OpenAI then restricted the use
of logit bias together with logprobs, which closed the cheap route to full
vectors on that API.

Gao, Liang and Guestrin (2024) asked the adopter's question directly: is this
API serving the model it claims? Their method needs no logprobs. It samples
completions from the endpoint and from a reference, and runs a two-sample test
on the outputs. It detected quantised and otherwise altered variants of open
models behind commercial endpoints. This is the output-only counterpart of the
logit subspace, and the closest published method to the serving-arm question
in this project.

## What exists for Anthropic models

The Claude API has not exposed token log-probabilities, so the subspace and
projection methods have had nothing to work on. What has been done is
behavioural. LLMmap (Pasquini, Kornaropoulos and Ateniese, 2024) sends a fixed
set of crafted queries and classifies the responses; it identified closed
models, Claude 3 variants among them, from text alone. Classifiers trained on
output style (Sun et al., 2025, on model idiosyncrasies) separate Claude from
other families from ordinary completions. These methods say which family and
roughly which version is answering. They cannot read a hidden dimension, and
they cannot tell a quantised copy from the original the way a logit subspace
can.

## What follows for Recall Guard

The methods above read identity out of the same evidence this guard reads
memorization out of. Three design choices follow, and the design already makes
them.

Identity is bound, not inferred. The full-distribution host refuses a request
whose `model_digest` differs from the pinned GGUF, and the calibration identity
binds model, protocol, endpoint, response mapping, estimate source, generation
settings and calibration corpus. A calibrator is refused on evidence from a
different source or different settings, naming the field. The fingerprinting
work is the argument for why this has to be explicit: an endpoint's identity is
recoverable from its logits, so a guard that does not pin identity can score a
different model after one silent update without noticing.

Values compare only within one set of weights. The subspace is per model.
Features and estimates from top-k and from the full distribution are compared
on the same responses and the same weights (requirement 9.4); the serving arm
on Modal and the local sampling arm are reported as two arms, because the bf16
checkpoint and the Q4_K_M GGUF are different weights even though both are
Gemma 3 4B.

The full-distribution host can check the weights it serves. Those papers worked
from top-k vectors reconstructed at hundreds of requests each. The host returns
whole-vocabulary vectors at every position, 262,144 entries for Gemma 3 4B, so
the rank of a few hundred stacked log-probability rows should come out at the
hidden size, 2560 for this model. That is a cheap, digest-independent check
that the resident weights are the architecture they claim to be. It is not in
the current task plan; it would fit the exactness receipt of task 6.1 if
wanted.

Serving nondeterminism is the noise floor. The subspace methods need many
positions because batch-dependent kernels move logits by small amounts between
requests. That same movement is what the serving arm measures at temperature 0.
A finding that identical prompts give identical draws on one stack, and not on
another, is informative in both directions: for the ensemble's purpose, and for
how many positions a fingerprint would need on that stack.

## Limits

This note is a literature summary. The citations below are given from memory,
including the arXiv identifiers and venues; verify them before citing
externally. The hidden-dimension check is proposed, not run; nothing here
changes the frozen plan or any requirement. Behavioural fingerprinting results
for Claude are reported as the papers state them and have not been reproduced.

## References

- Morris, J. X., Zhao, W., Chiu, J. T., Shmatikov, V. and Rush, A. M. (2023).
  Language Model Inversion. arXiv:2311.13647. ICLR 2024. Recovers full
  next-token distributions from a top-k API through logit bias; inverts prompts
  from them.
- Finlayson, M., Swayamdipta, S. and Ren, X. (2024). Logits of API-Protected LLMs
  Leak Proprietary Information. arXiv:2403.09539. The softmax bottleneck makes
  the output subspace a model signature; hidden size of gpt-3.5-turbo estimated
  at about 4096; proposed as an audit for silent model changes.
- Carlini, N., Paleka, D., Dvijotham, K. D., Steinke, T., Hayase, J., Cooper,
  A. F., Lee, K., Jagielski, M., Nasr, M., Conmy, A., Wallace, E., Rolnick, D.
  and Tramer, F. (2024). Stealing Part of a Production Language Model.
  arXiv:2403.06634. ICML 2024. Recovers the hidden dimension and the final
  projection layer of OpenAI production models from top-k logprobs.
- Gao, I., Liang, P. and Guestrin, C. (2024). Model Equality Testing: Which
  Model Is This API Serving? arXiv:2410.20247. Two-sample tests on sampled
  outputs detect altered or quantised models behind an endpoint.
- Pasquini, D., Kornaropoulos, E. M. and Ateniese, G. (2024). LLMmap:
  Fingerprinting For Large Language Models. arXiv:2407.15847. Behavioural
  fingerprinting of closed models, including Claude 3 variants, from crafted
  queries.
- Sun, M., Yin, Y., Xu, Z., Kolter, J. Z. and Liu, Z. (2025). Idiosyncrasies in
  Large Language Models. arXiv:2502.12150. Text classifiers separate model
  families, Claude among them, from ordinary outputs.

Related in this documentation: [Logits and logprobs on local
hardware](../research-report.md#3-logits-and-logprobs-on-local-hardware) in the
research report; register entries R35, R36 and R40 for the local full-logit
routes.
