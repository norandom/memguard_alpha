# Reassessment update — 2026-11-01

The user requested a smaller portable API/calculation delivery and a Docker
approach. This update supersedes broader delivery assumptions in the historical
research below. Applied kiro-spec-design synthesis and kiro-spec-tasks graph
review: reuse current clients, ledger and calculations; integrate the actual
adopter flow before adding infrastructure. The local design template is absent;
use the skill's boundary/flow/file-plan/validation/traceability fallback structure.

The [infrastructure assessment](infrastructure-assessment.md) records concrete
callers and keep/consolidate/defer/prune decisions. In particular, active polling
has only a loopback fixture route, not a verified hosted NIM route. Preserve
accepted-request identity and safe deferral. Keep existing optional SSE integrity,
study tools and Dagger; defer new paired-runner/resume/profile expansion.

The separate [Docker research](../docker-execution/research.md) covers the shared
uv/Dagger environment. No Docker builds, application tests, deletions or provider
calls were made while writing these proposals. Prior implementation receipts
remain historical; current task3.2 repair is partial and unvalidated.

---

# Research: adaptive provider readiness

Date: 2026-11-01. Discovery: extension with focused provider-contract and
workflow-integration investigations. No inference requests or credentials were
used. Requirements were approved, then extended by the user's explicit request
for missing-logprob fallback and reporting of actual execution.

## Existing behavior

- `ModelConfig` owns explicit endpoint/model/settings and per-client timeout,
  retry and pacing values. `RequestControl` owns physical admission and a shared
  `CallBudget`. It already treats HTTP 504/timeouts separately from credential
  rejection, but loses retry-delay headers and uses fixed exponential backoff.
- `OpenAICompatibleClient` expects complete JSON; `probe()` observes actual
  inference and returns capabilities, but not the response as a portable probe
  artifact. Missing MIA token evidence raises an error with usage, discarding
  the response before the workflow can record it.
- `ModelConfig.inspect()` includes every explicit parameter. Turning MIA off
  alone does not remove a declared/requested logprob capability. Constructor
  validation runs before workflow preflight, so fallback cannot be bolted onto
  per-item execution or implemented only inside workflow preflight.
- `RecallWorkflow._selection()` already checks selected rewrite policy, draw
  count and matching scorer. Reuse its completion-only path and the existing
  ensemble. Calibration identity binds endpoint/model/settings/intervention.
- `ExperimentRunner` requires zero provider work in its workflow factory and
  exact agreement with the frozen protocol. Resolve optional fallback before
  freezing the final protocol. Preparation requests remain explicitly counted.

## Official provider evidence

The [exact gpt-oss-20b reference](https://docs.api.nvidia.com/nim/reference/openai-gpt-oss-20b-infer)
and its embedded OpenAPI (`script#ssr-props`, `document.api.schema`) establish
the hosted chat POST, SSE deltas, a terminal `[DONE]` message, and HTTP 202 with
UUID `NVCF-REQID` and string `NVCF-STATUS` headers. Its body example is empty;
there is no verified body request-ID mapping. The embedded specification has
only `/chat/completions`, without a polling route. Samples treat optional
`reasoning_content` separately from answer content. Streaming usage and
logprobs are not established by the schema; absence remains unavailable.

The [Mistral hosted polling reference](https://docs.api.nvidia.com/nim/reference/mistralai-mistral-medium-3-5-128b-statuspolling)
documents `/v1/status/{requestId}` for that model. It does not prove that route
for gpt-oss-20b. The initial gpt-oss profile must therefore retain a pending
identity and defer when it cannot observe that accepted request safely.

[NVIDIA's FAQ](https://docs.api.nvidia.com/nim/docs/product) describes free
developer prototyping access. No universal RPM or exact-model guarantee of
retry-delay headers was established. Limits remain caller configuration and
observed provider evidence rather than a static global table.

[Requests timeout documentation](https://requests.readthedocs.io/en/latest/user/quickstart/#timeouts)
defines socket inactivity, not a total response deadline. Use its existing
public transport API, explicit bounded I/O and cooperative cancellation; do
not promise that local cancellation terminates remote generation.

[RFC 9110 section 10.2.3](https://www.rfc-editor.org/rfc/rfc9110.html#section-10.2.3)
defines retry delay as seconds or an HTTP date. A valid delay beyond the
caller's remaining allowance leads to deferral, never an earlier retry.

[NVIDIA's long-request documentation](https://docs.nvidia.com/nemo/gym/main/infrastructure/nvcf-h2-ping-sidecar/)
describes intermediary idle limits. This is context for why increasing a local
timeout may not suffice; it does not establish the cause of the October 31
HTTP 504, and does not justify adding a sidecar to this Python package.

## Synthesis and decisions

- Generalize a small observation policy and trace around existing physical
  admission, not a provider-routing or availability-prediction service.
- Use existing Requests, stdlib monotonic time, thread events and HTTP-date
  parsing. No new SDK, event loop, background daemon or global client cache.
- Distinguish transport decisions from feature decisions. Transport may wait
  according to observations; a harness preparation step resolves an explicitly
  permitted completion-only fallback before evaluation. Neither changes models.
- Use fixed caller bounds plus observed hints and progress. Do not fit latency
  percentiles from a handful of successes or interpret missing headers as a
  rate-limit estimate.
- Keep pending polling capability-gated. Supporting an explicit verified
  profile is useful; guessing the selected model's route is not.
- Streamed answers remain subject to framing, parsing and token-evidence
  validation. Reasoning/progress is not answer content or a memorization score.
- Version new persisted execution plans/traces while preserving exact old
  artifact serialization and metric replay. Show requested, selected and
  provider-confirmed settings as three distinct concepts.

## Risks and validation consequences

Independent design review identified one handoff ambiguity: the existing paired
runner requires all three roles and original/rewritten arms, whereas library
preparation allows arbitrary existing feature choices. The design now keeps the
runner's shape explicit and rejects incompatible plans before network work.
Fresh evaluation/resume clients intentionally re-probe, consuming the remaining
allowance; preparation observations do not populate an identity-scoped cache.

The main risks are duplicate generation after an ambiguous outcome, falsely
counting partial streams as successful answers, missed physical poll calls,
and changing a paired study's method after freezing it. The design makes each
an explicit tested transition. Cooperative cancellation can prevent further
admission immediately when observed; an already blocked synchronous transport
returns under its configured I/O bounds. It cannot prove remote cancellation.

The October 31 eligibility budget remains exhausted. Live operational testing
needs a new frozen dated protocol and separately declared limits; it cannot
rewrite previous eligibility or mitigation outcomes.

Skills applied: Kiro requirements generation/review and design discovery,
synthesis and review gates. Repository templates are absent; the design uses
the installed skill's documented basic structure and existing project format.
