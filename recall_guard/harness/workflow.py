"""Application inference composed from configured completions and rewrites.

The existing core executor owns waves, failure limits and pure reduction. This
layer records all attempted outcomes and stops calling a client after a fatal
configuration failure, without converting surviving draws into a successful run.
"""

import json
from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from threading import Lock

from recall_guard.core.completion import (
    CapabilityEvidence,
    CapabilityReport,
    CapabilityStatus,
    CompletionClient,
    FeatureChoices,
    ModelConfig,
    ModelResponse,
    validate_features,
)
from recall_guard.core.ensemble import EnsembleSpec, generate_ensemble
from recall_guard.core.provider_anthropic import AnthropicClient
from recall_guard.core.provider_callback import CallbackClient
from recall_guard.core.providers import OpenAICompatibleClient
from recall_guard.core.request_control import CallBudget, RequestError
from recall_guard.harness.intervention_types import EvidenceArm, RecordStatus, text_hash
from recall_guard.harness.request_failures import FATAL_FAILURES as _FATAL_FAILURES
from recall_guard.harness.request_failures import request_failure as _failure
from recall_guard.harness.request_failures import rewrite_failure_kind
from recall_guard.harness.rewriting import RewriteInput, RewritePolicy, render_target_prompt
from recall_guard.harness.scorer import MemoryGuardedScorer, ScoredDraw
from recall_guard.harness.workflow_calibration import WorkflowCalibration, calibrate_workflow
from recall_guard.harness.workflow_types import WorkflowDraw, WorkflowPreflight, WorkflowResult

_STOCK_CLIENTS = (OpenAICompatibleClient, AnthropicClient, CallbackClient)
_PROBE_PROMPT = "Reply with the word OK."
_DEFAULT_SCORER = object()


def _generate(client, prompt, *, stage, parameters):
    stage_generate = getattr(client, "generate_for_stage", None)
    response = (stage_generate(prompt, stage=stage, **parameters) if callable(stage_generate)
                else client.generate(prompt, **parameters))
    if not isinstance(response, ModelResponse):
        raise RequestError("Completion client must return ModelResponse.", code="invalid_response")
    return response


def _accounting(client) -> CapabilityEvidence:
    if isinstance(client, CallbackClient):
        return client.request_accounting
    if isinstance(client, (OpenAICompatibleClient, AnthropicClient)):
        return CapabilityEvidence("supported", "Stock transport physical-request admission",
                                  "Every transport attempt uses the client's shared CallBudget.")
    return CapabilityEvidence("unverified", "Generic completion client",
                              "Physical requests, hidden retries and shared-budget enforcement are unknown.")


def _check_settings(client, config, response=None):
    """Reject changed configuration and contradictory metadata, leaving omissions unknown."""
    if client.config != config:
        raise RequestError("Completion client configuration changed during workflow execution.",
                           code="configuration_changed")
    if response is None:
        return
    expected = config.to_dict()["parameters"]
    for field in ("requested_settings", "confirmed_settings"):
        for name, value in response.to_dict()[field].items():
            if name not in expected and field == "confirmed_settings":
                continue
            if name not in expected or value != expected[name]:
                raise RequestError("Response settings contradict the configured workflow.",
                                   parameter=name, code="response_settings_mismatch")


@dataclass(frozen=True)
class _DrawResponse:
    index: int
    response: ModelResponse

    @property
    def content(self):
        return self.response.content


class _RecordingClient:
    """Assign identities before outcomes, including responses later rejected.

    A fatal configuration latch refuses subsequent client invocations. The core
    executor may finish its already scheduled wave; these refusals are not new
    attempts and cannot produce a successful survivor result at the boundary.
    """

    def __init__(self, client, item, prompt, intervention_id, evidence, config, scorer=None):
        self.client, self.item, self.prompt = client, item, prompt
        self.config = config
        self.intervention_id, self.evidence_hash = intervention_id, text_hash(evidence)
        self.arm = EvidenceArm.REWRITTEN if intervention_id else EvidenceArm.ORIGINAL
        self.draws = []
        self.lock = Lock()
        self.fatal = None
        self.diagnostics = None
        self.projected = set()
        self.component_values = {}
        self.scorer = scorer

    def update(self, index, **changes):
        with self.lock:
            self.draws[index] = replace(self.draws[index], **changes)

    def generate(self, prompt, **parameters):
        with self.lock:
            if self.fatal is not None:
                raise self.fatal
            index = len(self.draws)
            self.draws.append(WorkflowDraw(
                self.item.item_id, self.arm, index, prompt=prompt, intervention_id=self.intervention_id,
                evidence_hash=self.evidence_hash,
            ))
        try:
            _check_settings(self.client, self.config)
            response = _generate(self.client, prompt, stage="target", parameters=parameters)
            self.update(index, response=response)
            _check_settings(self.client, self.config, response)
            if self.scorer is not None:
                try:
                    score = self.scorer.score_response(prompt, response, client=self.client,
                                                       intervention_id=self.intervention_id or "original")
                except ValueError:
                    # The response remains available on the draw, including any
                    # raw censoring markers. Selected scoring cannot downgrade.
                    raise RequestError("Configured scoring evidence is unavailable or differs from calibration.",
                                       code="missing_token_evidence" if response.mia_unavailable_reason
                                       else "response_settings_mismatch") from None
                self.update(index, guarded_score=score)
            return _DrawResponse(index, response)
        except Exception as error:
            kind, message, diagnostics = _failure(error)
            self.update(index, status=RecordStatus.FAILED, error=message, failure_kind=kind, diagnostics=diagnostics)
            if kind in _FATAL_FAILURES:
                with self.lock:
                    if self.fatal is None:
                        self.fatal, self.diagnostics = error, diagnostics
            raise

    def decide(self, draw, project):
        if draw.index in self.projected:
            return self.draws[draw.index].answer
        if self.fatal is not None:
            self.update(draw.index, status=RecordStatus.FAILED, error="Projection aborted after a fatal workflow failure.",
                        failure_kind="aborted")
            raise self.fatal
        try:
            answer = project(draw.response)
            hash(answer)
        except Exception as error:
            kind, message, diagnostics = _failure(error)
            self.update(draw.index, status=RecordStatus.FAILED, error=message,
                        failure_kind=kind if kind in _FATAL_FAILURES else "projection", diagnostics=diagnostics)
            if kind in _FATAL_FAILURES:
                self.fatal, self.diagnostics = error, diagnostics
                raise
            raise ValueError("Answer projection failed; exception text was not retained.") from None
        self.projected.add(draw.index)
        self.update(draw.index, status=RecordStatus.COMPLETED, answer=answer)
        return answer

    def components(self, draw, project):
        if draw.index not in self.component_values:
            try:
                values = {name: float(value) for name, value in project(draw.response).items()}
            except Exception as error:
                kind, message, diagnostics = _failure(error)
                self.update(draw.index, status=RecordStatus.FAILED, error=message,
                            failure_kind=kind if kind != "runtime_error" else "components", diagnostics=diagnostics)
                if kind in _FATAL_FAILURES:
                    self.fatal, self.diagnostics = error, diagnostics
                    raise
                raise ValueError("Component projection failed; exception text was not retained.") from None
            self.component_values[draw.index] = values
            self.update(draw.index, components=values)
        return self.component_values[draw.index]

    def outcomes(self):
        """Finalize admitted responses that a fatal wave stopped before parsing."""
        return tuple(replace(draw, status=RecordStatus.FAILED, failure_kind="aborted",
                             error="Projection aborted after the workflow stopped.")
                     if draw.status == RecordStatus.PENDING else draw for draw in self.draws)


class RecallWorkflow:
    """One configured target with explicit single/ensemble and rewrite choices.

    ``run`` automatically preflights active roles; ``preflight`` exposes and
    caches those real probes before a caller schedules a batch. Disabled rewrite
    never validates, probes or invokes auxiliary clients. Strict accounting is
    supported by stock admitted transports and cooperative callback hooks only.
    Generic clients remain usable with explicitly unknown physical accounting.
    Set generation controls on ModelConfig; explicit EnsembleSpec controls must
    match that configuration so the actual probe covers the chosen controls.
    An optional default ``scorer`` selects configured scoring. Per-run/preflight
    ``scorer=...`` overrides that fit; explicit ``scorer=None`` disables it.
    Rewritten scoring requires a separate fit bound to the policy identity.
    """

    def __init__(
        self, target: CompletionClient, *, rewrite_policy: RewritePolicy | None = None,
        budget: CallBudget | None = None, strict_budget: bool = False,
        features: FeatureChoices | None = None, rewrite_scope: str = "shared_per_item",
        scorer: MemoryGuardedScorer | None = None,
    ) -> None:
        if scorer is not None and not isinstance(scorer, MemoryGuardedScorer):
            raise TypeError("scorer must be MemoryGuardedScorer or None")
        if not isinstance(target, CompletionClient) or not isinstance(target.config, ModelConfig):
            raise TypeError("target must implement CompletionClient with a ModelConfig")
        if rewrite_policy is not None and not isinstance(rewrite_policy, RewritePolicy):
            raise TypeError("rewrite_policy must be RewritePolicy")
        if budget is not None and not isinstance(budget, CallBudget):
            raise TypeError("budget must be CallBudget")
        if not isinstance(strict_budget, bool):
            raise TypeError("strict_budget must be boolean")
        if rewrite_scope != "shared_per_item":
            raise ValueError("Only shared_per_item rewrite scope is supported")
        if features is not None and not isinstance(features, FeatureChoices):
            raise TypeError("features must be FeatureChoices")
        self.target, self.rewrite_policy = target, rewrite_policy
        self.scorer = scorer
        self.features, self.strict_budget = features, strict_budget
        self.budget = budget if budget is not None else target.budget if isinstance(target, _STOCK_CLIENTS) else None
        self._probes = {}
        self._selection(None, None, validate_draws=False, validate_scoring=False)

    def _selection(self, rewrite, ensemble, *, scorer=_DEFAULT_SCORER, validate_draws=True, validate_scoring=True):
        if rewrite is None:
            rewrite = self.features.rewriting if self.features else False
        if not isinstance(rewrite, bool):
            raise TypeError("rewrite must be boolean")
        if ensemble is not None and not isinstance(ensemble, EnsembleSpec):
            raise TypeError("ensemble must be EnsembleSpec or None")
        selected = self.scorer if scorer is _DEFAULT_SCORER else scorer
        if selected is not None and not isinstance(selected, MemoryGuardedScorer):
            raise TypeError("scorer must be MemoryGuardedScorer or None")
        if validate_scoring and self.features and self.features.mia_scoring and selected is None:
            raise ValueError("MIA scoring requires a separately configured matching scorer; none was supplied")
        draws = ensemble.draws if ensemble else 1
        if validate_draws and self.features and self.features.ensemble_draws != draws:
            raise ValueError("FeatureChoices.ensemble_draws must match the explicit EnsembleSpec")
        overrides = {name: getattr(ensemble, name) for name in ("max_tokens", "temperature")
                     if ensemble is not None and getattr(ensemble, name) is not None}
        if any(name not in self.target.config.parameters or value != self.target.config.parameters[name]
               for name, value in overrides.items()):
            raise ValueError("EnsembleSpec controls must match ModelConfig; reconfigure the target before preflight")
        roles = {"target": self.target}
        if rewrite:
            if self.rewrite_policy is None:
                raise ValueError("Rewriting requires an explicit rewrite policy")
            self.rewrite_policy.validate_configuration()
            roles |= {"rewriter": self.rewrite_policy.rewriter.client, "verifier": self.rewrite_policy.verifier}
        if validate_scoring and selected is not None:
            # Validate before probes, rewrites, verification or target calls.
            selected.validate_client(self.target, intervention_id=self.rewrite_policy.intervention_id if rewrite
                                     else "original", **overrides)
        for role, client in roles.items():
            config = replace(client.config, parameters=dict(client.config.parameters) | (overrides if role == "target" else {}))
            validate_features(config, FeatureChoices(mia_scoring=validate_scoring and selected is not None)
                              if role == "target" else None)
            if self.budget is not None and isinstance(client, _STOCK_CLIENTS) and client.budget is not self.budget:
                raise ValueError("All active clients must share the workflow budget")
            if self.strict_budget and (self.budget is None or not isinstance(client, _STOCK_CLIENTS)
                                       or _accounting(client).status != CapabilityStatus.SUPPORTED):
                raise ValueError("Strict budget requires supported physical request accounting for every active role")
        return rewrite, roles, overrides, selected

    def preflight(self, *, rewrite: bool | None = None, ensemble: EnsembleSpec | None = None,
                  refresh: bool = False, scorer=_DEFAULT_SCORER) -> WorkflowPreflight:
        """Probe only selected roles; changed parameters/configurations invalidate cache.

        Accepted controls are capability evidence. Effective values remain
        unknown unless echoed on actual ModelResponses. Probe failures preserve
        completed role reports and never admit an inference batch.
        """
        _, roles, overrides, selected = self._selection(rewrite, ensemble, scorer=scorer)
        return self._preflight(roles, overrides, mia_scoring=selected is not None, refresh=refresh)

    def _preflight(self, roles, overrides, *, mia_scoring=False, refresh=False):
        """Probe selected roles; calibration also uses this before preparation."""
        reports, responses = {}, {}
        for role, client in roles.items():
            original_config = client.config
            parameters = overrides if role == "target" else {}
            config = replace(client.config, parameters=dict(client.config.parameters) | parameters)
            features = FeatureChoices(mia_scoring=mia_scoring and role == "target")
            key = (id(client), json.dumps(config.to_dict(), sort_keys=True), features.mia_scoring)
            try:
                if refresh:
                    self._probes.pop(key, None)
                if refresh or key not in self._probes:
                    probe = getattr(client, "probe", None)
                    if callable(probe) and config.parameters == client.config.parameters:
                        report, response = probe(), None
                        if not isinstance(report, CapabilityReport):
                            raise TypeError("probe must return CapabilityReport")
                    else:
                        response = _generate(client, _PROBE_PROMPT, stage="probe", parameters=parameters)
                        _check_settings(client, original_config, response)
                        report = self._observed_report(config, response, features)
                    _check_settings(client, original_config)
                    unknown = CapabilityEvidence("unverified", "Required capability absent from probe report")
                    report = CapabilityReport({name: report.capabilities.get(name, unknown)
                                               for name in config.inspect(features).capabilities}, features)
                    if report.ready_for_batch:
                        self._probes[key] = report, response
                else:
                    report, response = self._probes[key]
                reports[role] = report
                if response is not None:
                    responses[role] = response
                if not report.ready_for_batch:
                    return WorkflowPreflight(reports, responses, role, "Required generation capabilities remain unavailable.",
                                             "capability_unavailable")
            except Exception as error:
                kind, message, diagnostics = _failure(error)
                return WorkflowPreflight(reports, responses, role, message, kind, diagnostics)
        return WorkflowPreflight(reports, responses)

    @staticmethod
    def _observed_report(config, response, features=None):
        capabilities = {}
        for name in config.inspect(features).capabilities:
            status, observed = CapabilityStatus.SUPPORTED, True
            detail = "Actual generation accepted the requested control; effective value is unconfirmed unless echoed."
            if name == "text_generation" and not response.content.strip():
                status, observed, detail = CapabilityStatus.UNVERIFIED, False, "No non-whitespace text observed."
            elif (name == "logprobs" and (config.parameters.get(name) is True or features and features.mia_scoring)
                  and response.logprobs_unavailable_reason):
                status, detail = CapabilityStatus.UNSUPPORTED, response.logprobs_unavailable_reason
            elif (name == "top_logprobs" and (config.parameters.get(name) or features and features.mia_scoring)
                  and response.mia_unavailable_reason):
                status, detail = CapabilityStatus.UNSUPPORTED, response.mia_unavailable_reason
            capabilities[name] = CapabilityEvidence(status, "Actual configured generation probe", detail, observed)
        return CapabilityReport(capabilities, features or FeatureChoices())

    def calibrate(
        self, *, is_memorized: Sequence[RewriteInput], oos_control: Sequence[RewriteInput],
        rewrite: bool = False, min_auc: float = 0.6,
        min_valid: int = 50, seed: int = 0, max_workers: int = 8,
    ) -> WorkflowCalibration:
        """Fit an original or policy-matched scorer from label-free inputs.

        Retains preparation failures and actual valid fitting counts. Selection
        of the returned scorer remains explicit; this does not change defaults.
        """
        return calibrate_workflow(self, is_memorized=is_memorized, oos_control=oos_control, rewrite=rewrite,
                                  min_auc=min_auc, min_valid=min_valid, seed=seed, max_workers=max_workers)

    def run(
        self, item: RewriteInput, *, rewrite: bool | None = None, ensemble: EnsembleSpec | None = None,
        decide: Callable[[ModelResponse], Hashable] | None = None,
        components: Callable[[ModelResponse], Mapping[str, float]] | None = None,
        scorer=_DEFAULT_SCORER,
    ) -> WorkflowResult:
        """Infer once or execute the existing ensemble over one shared prompt.

        Single means predesignated draw zero, never the first successful draw.
        Configuration mistakes raise before calls; runtime failures return
        partial outcomes. No inference failure changes the selected method.
        """
        if not isinstance(item, RewriteInput):
            raise TypeError("item must be RewriteInput")
        if decide is not None and not callable(decide) or components is not None and not callable(components):
            raise TypeError("decide and components must be callable when supplied")
        rewrite, roles, _, selected = self._selection(rewrite, ensemble, scorer=scorer)
        configurations = {role: client.config for role, client in roles.items()}
        requested = ensemble.draws if ensemble else 1
        result = WorkflowResult(item.item_id, item.evidence, None if rewrite else item.evidence, RecordStatus.PENDING,
                                "preflight", ("rewritten" if rewrite else "original") + ("_ensemble" if ensemble else "_single"),
                                requested, rewrite_scope="shared_per_item" if rewrite else None,
                                provenance={"target": configurations["target"].to_dict(),
                                            "ensemble": None if ensemble is None else asdict(ensemble)})
        if selected is not None:
            result = replace(result, calibration_identity=selected.calibration_identity,
                             provenance=dict(result.provenance) | {"scoring": {
                                 "identity": asdict(selected.calibration_identity),
                                 "holdout_auc": selected.holdout_auc, "is_weak": selected.is_weak,
                                 "valid_counts": selected.calibration_valid_counts}})
        preflight = self.preflight(rewrite=rewrite, ensemble=ensemble, scorer=selected)
        result = replace(result, preflight=preflight)
        if not preflight.ready_for_batch:
            return self._finish(replace(result, status=RecordStatus.FAILED, error=preflight.error,
                                        failure_kind=preflight.failure_kind,
                                        provenance=dict(result.provenance) | {
                                            "failure_diagnostics": preflight.diagnostics}), roles)
        try:
            for role, client in roles.items():
                _check_settings(client, configurations[role])
        except Exception as error:
            kind, message, diagnostics = _failure(error)
            return self._finish(replace(result, status=RecordStatus.FAILED, error=message, failure_kind=kind,
                                        provenance=dict(result.provenance) | {
                                            "failure_diagnostics": diagnostics}), roles)
        if rewrite:
            try:
                record = self.rewrite_policy.rewrite(item)
            except Exception as error:
                kind, message, _ = _failure(error)
                return self._finish(replace(result, status=RecordStatus.FAILED, stage="rewrite", error=message,
                                            failure_kind=kind), roles)
            result = replace(result, rewrite=record, stage=record.stage)
            if record.status != RecordStatus.COMPLETED:
                metadata = record.fidelity_checks.get("verification", record.fidelity_checks.get("generation", {}))
                return self._finish(replace(result, status=RecordStatus.FAILED, error=record.error,
                                            failure_kind=rewrite_failure_kind(metadata.get("failure_kind"))), roles)
            result = replace(result, accepted_evidence=record.rewritten_evidence)
        prompt = render_target_prompt(item, evidence=result.accepted_evidence)
        recorder = _RecordingClient(self.target, item, prompt,
                                    result.rewrite.fidelity_checks["intervention_id"] if rewrite else None,
                                    result.accepted_evidence, configurations["target"], selected)
        project = decide if decide is not None else lambda response: response.content
        choose = lambda draw: recorder.decide(draw, project)  # noqa: E731
        project_components = None if components is None else lambda draw: recorder.components(draw, components)
        result = replace(result, stage="target", sampled_at=datetime.now(UTC).isoformat())
        try:
            for role, client in roles.items():
                _check_settings(client, configurations[role])
            if ensemble is None:
                draw = recorder.generate(prompt)
                answer = choose(draw)
                if project_components is not None:
                    project_components(draw)
                result = replace(result, response=draw.response, answer=answer)
            else:
                reduced = generate_ensemble(recorder, prompt, ensemble, decide=choose, components=project_components)
                if recorder.fatal is None:
                    consensus_index = reduced.consensus.index
                    response = reduced.consensus.response
                    reduced = replace(reduced, consensus=response, draws=tuple(draw.response for draw in reduced.draws))
                    result = replace(result, ensemble=reduced, response=response,
                                     answer=recorder.draws[consensus_index].answer, sampled_at=reduced.sampled_at)
            if recorder.fatal is not None:
                raise recorder.fatal
            result = replace(result, status=RecordStatus.COMPLETED)
        except Exception as error:
            kind, message, diagnostics = _failure(recorder.fatal if recorder.fatal is not None else error)
            result = replace(result, status=RecordStatus.FAILED, response=None, answer=None, ensemble=None,
                             error=message, failure_kind=kind,
                             provenance=dict(result.provenance) | {
                                 "failure_diagnostics": recorder.diagnostics or diagnostics})
        result = replace(result, draws=recorder.outcomes())
        if selected is not None and recorder.fatal is None:
            result = self._score_result(result, selected, prompt, ensemble)
        return self._finish(result, roles)

    @staticmethod
    def _score_result(result, scorer, prompt, spec):
        if spec is None:
            score = result.draws[0].guarded_score if result.draws else None
            if score is None or not score.parse_ok:
                if result.status == RecordStatus.FAILED:
                    return replace(result, guarded_score=score)
                return replace(result, status=RecordStatus.FAILED, stage="scoring",
                               error="Selected single-response scoring did not produce a usable score.",
                               failure_kind="scoring_unavailable", guarded_score=score)
            return replace(result, guarded_score=score, mia_features=score.features, p_memorized=score.p_memorized)
        draws, failures = [], {}
        for draw in result.draws:
            if draw.guarded_score is not None:
                draws.append(ScoredDraw(draw.guarded_score, draw.response.content, draw.draw_index // spec.max_workers))
            else:
                reason = draw.failure_kind or "error"
                failures[reason] = failures.get(reason, 0) + 1
        scored = scorer.reduce_scored_draws(prompt, draws, spec=spec, failures=failures,
                                           sampled_at=result.sampled_at)
        if scored.p_memorized_point is None:
            if result.status == RecordStatus.FAILED:
                return replace(result, scored_ensemble=scored)
            return replace(result, status=RecordStatus.FAILED, stage="scoring", scored_ensemble=scored,
                           error="Selected ensemble scoring has too few usable draws.", failure_kind="scoring_unavailable")
        return replace(result, scored_ensemble=scored, p_memorized=scored.p_memorized_point)

    def _finish(self, result, roles):
        response = result.response or next((draw.response for draw in result.draws if draw.response is not None), None)
        mia_reason = response.mia_unavailable_reason if response is not None else "No completed target response."
        mia_reason = mia_reason or "MIA feature scoring was not selected."
        reasons = {"mia_features": mia_reason, "guarded_score": "No matching fitted calibration/scorer was supplied.",
                   "p_memorized": "No matching fitted calibration/scorer was supplied."}
        if result.calibration_identity is not None:
            detail = "Selected scoring is unavailable: " + (result.error or "No completed target response.")
            reasons = {name: detail for name in ("mia_features", "guarded_score", "p_memorized")}
            if result.scored_ensemble is not None:
                reasons.update({name: "Ensemble evidence is per draw; inspect draws or scored_ensemble.consensus."
                                for name in ("mia_features", "guarded_score")})
            reasons = {name: reason for name, reason in reasons.items() if getattr(result, name) is None}
        return replace(result, unavailable_reasons=reasons,
                       provenance=dict(result.provenance) | self._request_provenance(roles))

    def _request_provenance(self, roles):
        """Physical accounting shared by inference and calibration results."""
        accounting = {role: _accounting(client).to_dict() for role, client in roles.items()}
        known_requests = self.budget is not None and all(
            evidence["status"] == CapabilityStatus.SUPPORTED for evidence in accounting.values()
        )
        return {
            "request_accounting": accounting,
            "physical_requests_used": self.budget.used if known_requests else None,
            "physical_budget_limit": self.budget.limit if known_requests else None,
            "observed_budget_requests": None if self.budget is None else self.budget.used,
            "strict_budget": self.strict_budget,
        }


__all__ = ["RecallWorkflow", "WorkflowResult", "WorkflowDraw", "WorkflowPreflight"]
