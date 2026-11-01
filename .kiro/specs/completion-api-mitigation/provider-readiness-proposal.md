# Proposal: adaptive provider readiness

Status: requirements approved by the user on 2026-11-01. The canonical copy is
[adaptive-provider-readiness/requirements.md](../adaptive-provider-readiness/requirements.md).
This addition
does not change the approved 25-task plan or any frozen measurement protocol.

## Problem and intended outcome

Recall Guard users configure hosted models whose availability, rate limits and
response times vary. The user reports that this is particularly relevant to
NIM's free service and asks for sensing rather than a single hard timeout.
The proposed optional behavior observes the selected endpoint and model,
reports its current state, and waits or defers according to actual evidence
and explicit caller limits.

The existing implementation already distinguishes authentication rejection
from transient failures. Its Requests timeout is a connection/read inactivity
limit, not a total request deadline. The missing behavior is visible progress,
server-directed waiting, pending-request handling and readiness decisions
that survive temporary unavailability. Increasing one timeout cannot repair a
gateway that has already returned HTTP 504.

## Scope

Include optional readiness sensing for configured completion clients, model-
specific waiting and retry policies, observable waiting/deferred states,
supported response progress, strict physical-call accounting and notebook
parity. Start with the selected NVIDIA-hosted OpenAI-chat configuration.
Keep other providers' capabilities explicit and unverified until established.

Exclude automatic model switching, changes to generation parameters or token
limits, fabricated logprobs, inferred account balances, indefinite background
polling, global rate-limit tables and changes to existing study outcomes.
Provider progress and status facilities must actually be supported by the
selected endpoint; a catalog listing or a local heartbeat cannot prove that
inference is advancing.

## Requirement 1: Explicit model-specific control

1. Where adaptive readiness is enabled, Recall Guard shall use the caller's
   selected endpoint, model, generation settings and required capabilities.
2. The feature shall allow callers to configure request, retry and waiting
   limits independently for their selected models.
3. When adaptive readiness is disabled, Recall Guard shall preserve existing
   configured-client behavior.
4. If a requested observation capability is unavailable, the feature shall
   expose that limitation before relying on it for a readiness decision.

## Requirement 2: Evidence-based states

1. When provider evidence is received, the feature shall distinguish accepted
   pending work, observed progress, completed usable inference, temporary
   unavailability, explicit permanent rejection and unknown outcome.
2. If a request times out or receives a gateway failure without an explicit
   credential rejection, the feature shall leave credential eligibility
   unverified and retain the actual failure evidence.
3. When a valid inference response satisfies the selected capability checks,
   the feature shall report readiness for that exact configuration and the
   observation time, without guaranteeing future availability.
4. While waiting without new provider evidence, the feature shall report
   elapsed waiting separately from observed provider progress.

## Requirement 3: Wait and retry according to observations

1. When a provider supplies a valid retry delay, the feature shall wait at
   least that long before a permitted follow-up, or defer if the caller's
   remaining waiting allowance cannot accommodate it.
2. If retry timing is absent or invalid after a retryable failure, the feature
   shall use a bounded configurable backoff and report its chosen delay.
3. Where the selected provider supports observing an already accepted request,
   the feature shall observe that same request without submitting duplicate
   generation work.
4. Where response progress is available, the feature shall use it to distinguish
   an active response from an idle connection and preserve partial data if
   completion is interrupted.
5. If a request outcome is ambiguous, the feature shall retain that ambiguity
   and require an explicit configured replay policy before resubmitting it.
6. If authentication, access, parameter or hard-quota rejection is explicit,
   the feature shall stop automatic retries and report the actionable reason.

## Requirement 4: Bounded and inspectable operation

1. When a request budget, retry limit, waiting allowance or caller cancellation
   is reached, the feature shall stop further request admission and preserve
   completed observations with the stopping reason.
2. The feature shall count every physical inference, probe, retry and status
   request against the shared request budget and distinguish their purposes.
3. When reporting a run, the feature shall retain per-model waiting decisions,
   provider status, timestamps, request identities and reported usage without
   persisting credentials or authorization headers.
4. If a partial response lacks required completion or token evidence, the
   feature shall not count it as a usable answer or available MCS score.

## Requirement 5: Portable use and measured behavior

1. When used in an application or an existing Jupyter kernel, the feature shall
   expose the same configuration, observation and cancellation behavior on the
   supported Windows and Linux Python versions.
2. The validation shall demonstrate delayed success, progress followed by
   completion, temporary throttling, provider-directed delay, permanent
   rejection, ambiguous disconnect, cancellation and exhausted budgets using
   deterministic offline scenarios.
3. Before a live readiness study, its protocol shall declare the configuration,
   observation method, request/waiting limits and comparison metrics.
4. When reporting live behavior, the report shall distinguish operational
   availability and time to usable response from mitigation effectiveness,
   preserve failed attempts, and identify missing capability evidence.

## Existing interfaces and design questions

`ModelConfig` already carries per-client transport settings. `RequestControl`
already owns admission, fixed exponential backoff and sanitized physical-attempt
records. `OpenAICompatibleClient` currently expects one complete JSON response;
it does not assemble event streams or poll accepted work. Its `probe()` checks
actual inference, which remains the right readiness criterion. Experiment
journals already refuse ambiguous automatic restarts.

Design should extend these boundaries and preserve old serialized artifacts.
It must define separate connection, idle and overall-wait limits; validate
server timing hints; specify observable states and cancellation; and establish
endpoint-specific request-ID and status-poll contracts before enabling them.
Any polling endpoint must be explicitly trusted; response-provided locations
must not redirect credentials to another origin. Interrupted streams cannot
silently become successful completions or lose token alignment.

The exact gpt-oss-20b reference documents streaming and pending responses, but
the exact polling route/request-ID mapping for that configuration has not yet
been established in this investigation. A polling route documented for another
model is supporting context, not proof of support for this one.

## Measurement boundary

The October 31 eligibility budget remains exhausted at four attempts. Its last
NIM attempt returned HTTP 504 after 302.453 seconds with a 600-second client
timeout. This proposal does not relabel that failure or establish its cause.
A future live readiness study needs its own explicit frozen limits and dated
artifacts; it must not borrow silently from calibration or evaluation phases.
No additional provider requests have been made for this proposal.

## Sources checked on 2026-11-01

- [NVIDIA NIM FAQ](https://docs.api.nvidia.com/nim/docs/product): developer-program
  access includes free API endpoints for prototyping; this does not establish
  current availability or account-specific limits.
- [NVIDIA gpt-oss-20b API](https://docs.api.nvidia.com/nim/reference/openai-gpt-oss-20b-infer):
  documents SSE deltas and completion marker, model-specific output-token range,
  and HTTP 202 pending work with request-ID polling.
- [NVIDIA hosted status polling example](https://docs.api.nvidia.com/nim/reference/mistralai-mistral-medium-3-5-128b-statuspolling):
  documents a status route for that particular model; do not generalize silently.
- [Requests timeout semantics](https://requests.readthedocs.io/en/latest/user/quickstart/#timeouts):
  timeout concerns socket inactivity rather than an overall response deadline.
- [RFC 9110 Retry-After](https://www.rfc-editor.org/rfc/rfc9110.html#section-10.2.3):
  delay may be expressed as seconds or an HTTP date.

## Requirements review

The draft covers explicit configuration, observed state, waiting/retry behavior,
budgets, cancellation, portable use and measurement. Numeric requirement IDs
and EARS criteria were checked; implementation choices are separated above.
The repository requirements template is absent, so this proposal uses the
existing specification's document structure. Requirements and design were approved on 2026-11-01. The implementation
task plan passed independent review, and the user approved implementation
and the separately bounded pilot. No new live measurement has started.
