# Provider contract audit and offline fixtures

Reviewed: 2026-10-30. Scope: task 1.1, requirements 1.4, 1.5, 7.1, 7.2,
7.4 and 7.5; design section **Contracts**. This audit supplies inputs for later
adapters. No provider adapter or live inference was added by this task.

## Explicit protocol and parameter mappings

| Provider and audited variant | Explicit protocol and endpoint | Request controls | Response evidence |
| --- | --- | --- | --- |
| OpenAI Chat Completions | `openai_chat`; `https://api.openai.com/v1/chat/completions` | `model`, `messages`, `max_completion_tokens`; optional `temperature`, `top_p`, `seed`, `logprobs`, `top_logprobs` where supported by the selected model | `choices[].message.content`; optional `choices[].logprobs.content`; `finish_reason`, `usage` |
| Anthropic first-party Messages, API version `2026-10-21` | `anthropic_messages`; `https://api.anthropic.com/v1/messages` | `model`, `messages`, `max_tokens`; separate top-level `system`; explicit `anthropic-version` header | Typed `content` blocks, `stop_reason`, `usage.input_tokens` and `usage.output_tokens` |
| Mistral hosted Chat Completion | `openai_chat`; `https://api.mistral.ai/v1/chat/completions` | `model`, `messages`, `max_tokens`; schema lists `temperature`, `top_p`, `random_seed`, `logprobs`, `top_logprobs` | `choices[].message.content`, `finish_reason`, `usage`; model-specific token evidence still requires a probe |
| NVIDIA hosted chat / NIM container chat | `openai_chat`; hosted `https://integrate.api.nvidia.com/v1/chat/completions`, or caller-configured container `/v1/chat/completions` | Common chat fields include `model`, `messages`, `max_tokens`; other controls depend on the particular model, recipe and server version | Chat-shaped content, optional token evidence and usage; a documented NIM example has `logprobs: null` |

OpenAI documents the Chat endpoint's boolean `logprobs` and `top_logprobs`
request with at most 20 alternatives. `max_tokens` is deprecated in favor of
`max_completion_tokens` and is incompatible with o-series models. Parameter
availability varies by model. An adapter must preserve the caller's explicit
choice rather than replace an incompatible control. The fixtures do not select
a frontier model for live measurement. [OpenAI Chat Completions reference](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create).

Rechecked on 2026-10-31 for the task 1.2 evidence-contract correction: OpenAI's
response schema distinguishes measured token logprobs from the reserved
`-9999.0` marker. It reports the actual value for tokens within the top 20 and
uses that marker otherwise, for either a realized token or a returned
`top_logprobs` alternative. The marker is censored evidence, not a measured
probability. Retain its exact raw value for audit and text-only generation;
mark MIA/MCS unavailable and reject legacy scoring conversion when it occurs.
This is an exact marker check, not a cutoff on ordinary negative logprobs.
[OpenAI response logprob contract](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create).

The fetched Anthropic Messages schema does not list `temperature`, `top_p`,
`seed` or `logprobs`. Treat them as unavailable in this audited contract,
without asserting that every historical API version or hosting platform rejects
them. Native typed text blocks must not be parsed as OpenAI choices. Reaching
`max_tokens` is a stop reason on a successful response, not an authentication
error. [Anthropic Messages reference](https://platform.claude.com/docs/en/api/messages/create).

The current Anthropic authentication guide uses bearer authentication and still
allows legacy `x-api-key`. Credentials must be resolved at runtime; neither
header belongs in stored fixtures or results. Version selection is explicit.
[Authentication](https://platform.claude.com/docs/en/manage-claude/authentication),
[API versioning](https://platform.claude.com/docs/en/api/versioning?version=2026-10-21).

Mistral's wire name for its sampling seed is `random_seed`. Its current request
schema includes logprob fields; that alone does not establish that a chosen
model returns the complete evidence needed by existing MIA features. Do not
assume that accepting OpenAI-shaped messages makes every OpenAI parameter valid.
The documentation uses `mistral-small-latest` in examples; an alias is not a
frozen model identity. [Mistral Chat reference](https://docs.mistral.ai/api).

NVIDIA's hosted `openai/gpt-oss-20b` model documentation establishes the model
identity and text modality, not universal hosted logprob/control guarantees.
The container reference delegates detailed schemas to the running server's
OpenAPI explorer. Hosted and container deployments require separate probes.
[Hosted model documentation](https://docs.api.nvidia.com/nim/reference/openai-gpt-oss-20b),
[NIM API reference](https://docs.nvidia.com/nim/large-language-models/latest/api-reference.html),
[versioned NIM 2.0 response example](https://docs.nvidia.com/nim/large-language-models/2.0.0/get-started/quickstart.html).

## Capability and error decisions for subsequent tasks

- Protocol, endpoint and model are separate configuration fields. Do not infer
  protocol from a hostname. Responses APIs, legacy text completions, streaming,
  Bedrock, Vertex and other deployment variants are outside these fixtures.
- Mark a feature supported only with relevant evidence for the configured
  endpoint/model. Schema support is documentation evidence; account access,
  effective controls and complete token evidence remain unverified until an
  actual generation probe. A model listing is insufficient.
- A successful text response with omitted or null logprobs is distinct from a
  rejected call. Text-only generation can proceed when explicitly chosen;
  existing MIA/MCS must stay unavailable if their token inputs are absent.
- HTTP 401 is an authentication failure; HTTP 403 is an access/permission failure
  whose cause must be retained. Neither means logprobs are missing. Parameter
  rejection uses its status and provider error, retains the explicit field,
  and must not trigger a retry that silently removes it. Error body details
  vary across gateways and versions. [OpenAI errors](https://developers.openai.com/api/docs/guides/error-codes),
  [Anthropic errors](https://platform.claude.com/docs/en/api/errors),
  [Mistral hosted examples and error statuses](https://docs.mistral.ai/resources/cookbooks/mistral-connectors-05-connectors-in-completions).
- Record requested parameters and omitted fields separately from confirmed
  effective values. None of the success fixtures echoes temperature or seed;
  a 200 response does not establish that those controls took effect.
- Direct JSON HTTP calls can use existing `requests>=2.31`; no new SDK or runtime
  dependency is needed for these audited protocols. Caller-supplied completion
  functions remain SDK-independent; their adapter is task 2.3.

## Fixture inventory and provenance

`tests/fixtures/providers/{openai,anthropic,mistral,nim}.json` are handwritten,
sanitized examples inspired by the sources above, not captured traffic or live
compatibility evidence. Model/request IDs, texts, token counts and probabilities
are synthetic. Model IDs beginning `fixture-` are deliberately non-deployable.
No environment file, credential or authorization header was read or copied.

Each file contains an explicit protocol/endpoint, a minimal request, text-only
success, 401 and 403 failures, and a rejected explicit parameter. OpenAI and
NIM additionally contain chat-shaped token evidence. Anthropic includes a
successful truncated response. Mistral omits the logprobs key while OpenAI/NIM
use null, freezing both absence forms. No fixture invents zero-valued token
evidence to permit scoring.

`unsupported_fixture_parameter` is deliberately fictional. Its rejection
fixtures exercise error preservation, not a claim about a real model's rejection
of a particular generation control. OpenAI/Anthropic error envelopes follow
their documented conventions. Mistral's flat error and NIM's problem-detail /
validation-detail variants are representative adversarial examples; the cited
pages do not establish one universal error envelope for these services.
Adapters must preserve unknown error details instead of requiring those exact
synthetic bodies. These fixture tests validate the inventory and distinctions;
task 2 adapter tests must replay the fixtures through production code.

## Compatibility baseline and validation

Before production changes on native Windows / Python 3.12.10:

- Focused scorer/ensemble baseline:
  `uv run --no-sync pytest tests/core/test_ensemble.py tests/harness/test_scorer.py tests/test_ensemble_boundary.py -q`
  — **73 passed in 14.95 s** (2026-10-30).
- Parent-run full offline baseline: `uv run --no-sync pytest -q` — **602 passed
  in 122.53 s**, with one existing notebook/zmq runtime warning. This result was
  supplied by the parent controller, independently of the focused run above.

Task-local validation uses
`uv run --no-sync pytest tests/core/test_provider_fixtures.py -q` and
`uv run --no-sync ruff check tests/core/test_provider_fixtures.py`.
Fresh results: **24 passed in 0.05 s** and **All checks passed**, respectively.
Offline fixture validation establishes no live provider support or mitigation
effectiveness; live access remains a later task.
