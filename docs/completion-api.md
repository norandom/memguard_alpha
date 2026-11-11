# Configure a completion workflow

Choose a target model and protocol, select the features you need, then inspect
readiness before scheduling a batch. `RecallWorkflow` supports a single answer
or the existing n-draw ensemble, with optional evidence rewriting. Text-only
responses keep MIA/MCS unavailable.

This interface is unreleased development-checkout code, outside v0.5.0. Start
with the offline example below. Its fixed callbacks demonstrate mechanics;
they do not validate a provider or demonstrate reduced memorization.

## Run the offline application or notebook

From a development checkout containing `examples/completion_workflow.py`, run it
in the development container (see the [Docker guide](development.md)):

```text
docker compose build dev
docker compose run --rm dev uv run --no-sync python examples/completion_workflow.py
```

Read `examples/completion_workflow.py` for the complete public-module imports
and fixture definitions. To work in Jupyter, start
`docker compose --profile notebook up notebook` and open
`notebooks/completion_workflow.ipynb`. On Linux and macOS you can instead select
the checkout's installed Python environment as its kernel. The notebook loads the companion
application's fixture functions with `%run -n ../examples/completion_workflow.py`
from `notebooks/`. It does not execute `main()` or require a shell command inside
the kernel.

Both examples explicitly select:

```python
features = FeatureChoices(mia_scoring=False, ensemble_draws=3, rewriting=True)
ensemble = EnsembleSpec(
    draws=features.ensemble_draws, min_parsed=3,
    max_workers=1, retain_draws=True,
)
```

These names come from `recall_guard.core.completion` and
`recall_guard.core.ensemble`. The complete imports and fixture definitions are
in the application. Choose `rewriting=False` to use original evidence without
rewrite or verifier calls. For a single answer choose `ensemble_draws=1` and
pass `ensemble=None`. `FeatureChoices.ensemble_draws` must match the explicit
`EnsembleSpec`; it does not create one automatically.

The accepted fixture produces three usable `up` answers, agreement of 1.0 and
`p_memorized=None`. Its verifier returns a fixed positive JSON verdict. The
second fixture runs only when rewriting is selected. It changes `2%` to `3%`;
fidelity rejection leaves
`accepted_evidence=None`, zero target attempts and an explicit error. The
workflow does not substitute an original-evidence answer after a rewrite fails.
Neither outcome measures model behavior.

## Inspect configuration before inference

`ModelConfig` holds protocol, endpoint, model, credential reference, generation
parameters and declared capability evidence. Its `inspect(features)` method
makes no calls. Read `unsupported` and `requires_probe`; a supported declaration
without observed evidence still requires a probe.

`workflow.validate(rewrite=..., ensemble=...)` checks the selected configuration
locally. `workflow.preflight(rewrite=..., ensemble=...)` calls each
active role. Inspect `ready_for_batch`, `affected_role`, `failure_kind`, `error`
and `diagnostics` before proceeding. `run()` also preflights and reuses cached
probes while the configuration remains identical. A successful probe establishes
access and acceptance for its request, not the effective value of a control the
provider did not echo.

Keep generation settings on `ModelConfig`. If you also supply `max_tokens` or
`temperature` on `EnsembleSpec`, they must match the target configuration.
Omitted parameter keys remain omitted, while explicit `None` stays explicit.
Inspect `response.requested_settings` separately from
`response.confirmed_settings`. Unreported usage and monetary cost remain unknown.

## Configure a native client explicitly

The following configuration and validation are local. They do not read a key
or make a request:

```python
from recall_guard.core.completion import FeatureChoices, ModelConfig
from recall_guard.core.provider_anthropic import validate_anthropic_config

features = FeatureChoices(mia_scoring=False, ensemble_draws=3, rewriting=False)
config = ModelConfig(
    protocol="anthropic_messages",
    endpoint="https://api.anthropic.com/v1/messages",
    model="claude-opus-5-5",
    credential_ref="ANTHROPIC_API_KEY",
    parameters={"max_tokens": 2048},
    timeout_s=90, max_retries=0,
)
validate_anthropic_config(config, features)
print(config.inspect(features).requires_probe)
```

The audited native Messages adapter pins API version `2026-10-21`. It accepts
native `max_tokens` and rejects controls such as `temperature`, `top_p`, `seed`,
`logprobs` and `top_logprobs`. These are adapter/protocol limits, not Jupyter
errors. `validate_anthropic_config` rejects incompatible parameters or selected
MIA/MCS before credential lookup. The offline examples exercise both failures.
Change the configuration explicitly; the adapter never removes rejected
parameters and retries silently.

Only run this next block if you intend to make live calls and have supplied
the referenced environment variable to the Python process. It uses the same
`config` and `features` above, without loading an environment file:

```python
from recall_guard.core.ensemble import EnsembleSpec
from recall_guard.core.provider_anthropic import AnthropicClient
from recall_guard.core.request_control import CallBudget
from recall_guard.harness.rewriting import RewriteInput
from recall_guard.harness.workflow import RecallWorkflow

budget = CallBudget(4)  # One target probe and up to three target draws; no retries.
target = AnthropicClient(config, budget, features=features)
workflow = RecallWorkflow(target, features=features, budget=budget, strict_budget=True)
ensemble = EnsembleSpec(draws=3, min_parsed=2, max_workers=1, retain_draws=True)
item = RewriteInput(
    item_id="application-example", instructions="Return only up or down.",
    evidence="Acme rose 2% on 2024-01-02; estimates remain uncertain.",
    as_of="2024-01-02",
)
preflight = workflow.preflight(rewrite=False, ensemble=ensemble)
if not preflight.ready_for_batch:
    print(preflight.affected_role, preflight.failure_kind, preflight.error, preflight.diagnostics)
else:
    result = workflow.run(item, rewrite=False, ensemble=ensemble)
    print(result.status, result.error, result.n_requested, result.n_attempted, result.n_usable)
    print(result.answer, result.p_memorized, result.unavailable_reasons)
```

This is a standalone integration example with synthetic evidence, not a new
run of the dated study. Native clients resolve the credential reference at
construction; recreate a client after rotating its credential.

To enable real rewriting, configure admitted rewriter and verifier clients
explicitly and share the target's budget. Compose
`RewritePolicy(EvidenceRewriter(rewriter_client), verifier_client)` from
`recall_guard.harness.rewriting`, supply it as `rewrite_policy` to
`RecallWorkflow`, and select `rewriting=True`. Their models and parameters can
differ from the target. The fixture application shows this composition without
making live calls.

For OpenAI-compatible chat, use `CompletionProtocol.OPENAI_CHAT` (or
`"openai_chat"`) and `OpenAICompatibleClient` from
`recall_guard.core.providers`, with an explicit endpoint and model. Mistral,
NIM and other compatible URLs require validation of that exact configuration;
a URL or a successful schema fixture is not evidence of live capability.
Native Responses, Azure and other endpoint variants have not been audited by
this workflow. Adapt them through the callback contract rather than assume
universal endpoint support.

## Use a callback in an application or Jupyter

`CallbackClient` lives in `recall_guard.core.provider_callback`. Supply
`ModelConfig(protocol="callback", endpoint=None, ...)`, a budget and a
synchronous callable accepting `prompt` plus the configured keyword parameters.
Return a string or a `ModelResponse` containing only genuine provider-reported
evidence. The caller owns SDK setup and credentials; the adapter does not
resolve or forward its credential reference.

For strict accounting, set `physical_request_hook="request"` and
`strict_budget=True`. Inside the callback, call `request(operation)` for each
physical request. The zero-argument operation performs exactly one synchronous
request with hidden retries and redirects disabled. The fixture's
`fixture_client` shows the complete hook wiring. A real operation calls your
SDK instead of returning its fixed string. Register known credential strings
with the callback adapter's `secrets` argument for diagnostic redaction.

The hook relies on caller cooperation. It cannot attest to provider activity or
detect requests made outside the hook. Opaque callbacks remain useful for
generation, but their internal retries, physical count and cost are unverified;
strict-budget live experiments reject them.

Call synchronous clients directly in an active notebook kernel. They may block
the current cell while waiting for a response, but do not start another event
loop. An async callback returns an awaitable and raises
`RequestError(code="callback_runtime_error")`; supply a synchronous callback
instead of wrapping it with `asyncio.run`.

| Failure | Where to inspect | Next action |
| --- | --- | --- |
| Unsupported declared feature or native control | Local `ValueError`; no batch starts | Change the explicit feature/parameter choice |
| Provider rejects an explicit control | `RequestError.parameter` and `.code`, or preflight/result diagnostics | Inspect the named control; do not strip it automatically |
| Callback or notebook runtime constraint | `callback_runtime_error`, often with `parameter=None` | Fix the synchronous integration or its dependency |
| Failed rewrite or fidelity check | `result.stage`, `.rewrite`, `.error` | Inspect retained source/candidate/checks; no fallback ran |
| Failed draw or insufficient usable draws | `.draws`, `.n_attempted`, `.n_usable`, `.failure_kind` | Inspect failures and the chosen minimum before accepting consensus |

## Keep answer statistics and MCS separate

The workflow reuses `EnsembleSpec` and the existing ensemble reducer. You may
supply `decide(response)` for an answer label and `components(response)` for
named numeric quantities. Agreement and dispersion describe those projections;
neither is a memorization probability. Self-reported confidence also cannot
replace missing token evidence.

Read `p_memorized`, `mia_features`, `guarded_score` and `unavailable_reasons`
explicitly. Text-only generation still supports parsing, answer ensembles and
optional rewriting. It does not create zeros or a new estimator for MCS.
Existing `MemoryGuardedScorer.calibrate`, `score`, `score_many`, `score_ensemble`
and `generate_ensemble` entry points retain their contracts. See
[scorer integration](index.md#integrate-the-scorer) and
[ensemble scoring](ensemble.md), including the distinction between the
`p_memorized_point` exposure multiplier and the representative draw's score.

Real configured-client scoring requires complete genuine realized-token and
top-alternative log probabilities, without censored evidence, plus matching
calibration. A logprob-shaped response alone is insufficient. Use separate
original and rewritten fits; `workflow.calibrate(..., rewrite=True)` applies
the same rewrite pipeline to calibration inputs and retains preparation
failures. Calibration and evaluation items must remain separate. Inspect actual
valid counts, held-out quality and weak-fit status before using a fit. The
dated completion pilot did not produce fresh matching calibration.

A configured fit binds protocol, endpoint, model, generation settings and
intervention/template identity. The workflow checks identity before probes or
inference and refuses mismatches. Supply the matching scorer explicitly for
each arm. Score genuine responses per draw before the existing scored reduction;
do not score averaged features or reconstruct logprobs from answer confidence.

## Bound calls and retain paired identity

One `CallBudget` must cover every active role. Count role probes, calibration
probes, rewrite generation, semantic verification, target requests, auxiliary
calls and retries. An n-draw ensemble does not mean n total physical requests.
`max_workers=1` controls concurrency in these examples, not the total request
budget. Exhaustion preserves partial results and prevents further admission.

`RewriteInput` separates instructions, evidence and the as-of date. Only evidence
presentation changes; instructions and output-format requirements stay outside
the rewrite. One accepted rewrite is shared by all target draws for an item,
with `rewrite_scope="shared_per_item"`. Fidelity acceptance combines factual-token
and semantic checks; it is not proof of equivalence or absence of future knowledge.

When exporting an experiment, the draw-record grain is one item/arm/draw.
`draw_index` is zero-based logical attempt order, not the reducer's canonical
usable order. Keep item IDs, evidence hashes and draw associations. The four
conditions are original single, original ensemble, rewritten single and
rewritten ensemble. Single response is predesignated draw 0 from that arm's
ensemble, so the observations are dependent. Pair by item; bootstrap whole items
with all their arms and draws together. Report all-attempted and common-valid
denominators separately. Failed interventions are unsuccessful task outcomes,
not evidence of reduced recall.

Experiment journals retain operation intent. An ambiguous in-flight journal
cannot auto-resume: an unrecorded outcome might already have consumed a paid
request. Resolve that state explicitly; a safe resume uses the saved remaining
budget and performs a fresh access probe.

## Replay the saved study before interpreting an effect

From the repository root, this command runs offline with the checked-in bundle:

```text
docker compose run --rm dev uv run --no-sync python scripts/run_intervention.py --replay docs/results/2026-10-31-completion-api/anthropic-evaluation-live
```

`--replay` verifies the portable bundle and recomputes saved paired metrics.
It does not query a model or refit calibration. Read the
[dated report and exact artifacts](results/2026-10-31-completion-api-pilot.md)
for the study's protocol, denominators and uncertainty. All 16 rewrites were
rejected, leaving zero valid rewrite pairs. On the 14 common-valid original
single/ensemble items, both conditions were correct on 11. The NIM attempt
ended in HTTP 504 after 302.453 seconds under a 600-second client timeout; all
four eligibility requests were consumed. This does not prove invalid credentials.
Fresh calibration and full two-mode live coverage remain incomplete. No
mitigation or aggregation benefit was established.

Do not rerun the dated paid protocol with a fresh full budget. Its phase
allocations have already been partly consumed and its eligibility budget is
exhausted. A new live experiment needs its own declared protocol, budget and
inputs.

## Relation to Jane Street's methods

Jane Street's [Mitigating memorization in LLMs](https://blog.janestreet.com/mitigating-memorization-in-llms/)
describes next-token-logit divergence decoding and, separately, a podcast-form
rewriting experiment. This workflow adapts the presentation idea while retaining
facts and verbatim quotations. It is not an equivalent implementation of the
logit equation; ordinary completion text cannot supply that equation's inputs.
See [divergence decoding](divergence-decoding.md) for the separate mathematical
interface. Recall Guard's own measured effects are limited to the dated report.

The application and notebook need Python 3.12 or newer and no GPU. On Windows
they run only in Linux containers (Docker Desktop or Docker Engine in WSL); native
Windows Python is no longer supported. On Linux and macOS they also run directly.
