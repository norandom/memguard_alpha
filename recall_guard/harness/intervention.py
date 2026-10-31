"""Paired experiment execution over the existing RecallWorkflow executor.

An arm generates n draws exactly once; its predesignated draw zero also serves
as the single condition. Safe segmented resumes intentionally re-probe access
under the remaining physical budget. Terminal arms are never regenerated.
"""

from collections.abc import Callable, Sequence
from dataclasses import asdict, replace
from pathlib import Path

from recall_guard.core.ensemble import EnsembleSpec
from recall_guard.core.request_control import CallBudget
from recall_guard.harness.intervention_journal import (
    ExperimentArm,
    ExperimentJournal,
    ExperimentRun,
    arm_snapshot,
    json_value,
    preflight_snapshot,
)
from recall_guard.harness.intervention_types import (
    DrawRecord,
    EvidenceArm,
    ExperimentCheckpoint,
    ExperimentItem,
    ExperimentProtocol,
    RecordStatus,
    RequestRecord,
    RewriteRecord,
    text_hash,
)
from recall_guard.harness.request_failures import FATAL_FAILURES
from recall_guard.harness.rewriting import REWRITE_METHOD_VERSION, RewriteInput
from recall_guard.harness.workflow import RecallWorkflow
from recall_guard.harness.workflow_types import WorkflowResult

_PROBE_PROMPT = "Reply with the word OK."


class ExperimentRunner:
    """Execute frozen paired items with a durable journal and a shared call cap.

    ``workflow_factory`` must only construct clients, using its fresh supplied
    CallBudget. It must not probe or perform other provider work. Both optional
    scorers must be supplied together and their numeric fingerprints must match
    protocol.data_hashes original_calibration/rewritten_calibration. The factory
    remains responsible for runtime credentials; the journal never stores them.
    Cooperative callbacks must use the supplied logical prompt for every hooked
    physical operation, without hidden retries/prompt variations. This relies
    on the caller's hook contract and does not attest hidden wire payloads.
    """

    def __init__(self, protocol: ExperimentProtocol, items: Sequence[ExperimentItem],
                 workflow_factory: Callable[[CallBudget], RecallWorkflow], checkpoint_path: str | Path, *,
                 original_scorer=None, rewritten_scorer=None):
        if not isinstance(protocol, ExperimentProtocol) or not callable(workflow_factory):
            raise TypeError("ExperimentRunner requires ExperimentProtocol and a workflow factory")
        self.protocol, self.items = protocol, tuple(items)
        if not self.items:
            raise ValueError("Experiment requires at least one item")
        self.factory = workflow_factory
        self.journal = ExperimentJournal(checkpoint_path)
        settings = dict(protocol.ensemble_settings)
        if "draws" in settings and settings.pop("draws") != protocol.draws:
            raise ValueError("Ensemble draw count differs from the protocol")
        self.spec = EnsembleSpec(draws=protocol.draws, **settings)
        self.scorers = {EvidenceArm.ORIGINAL: original_scorer, EvidenceArm.REWRITTEN: rewritten_scorer}
        if (original_scorer is None) != (rewritten_scorer is None):
            raise ValueError("Scored experiment requires both original and rewritten calibration")
        if original_scorer is None and any(arm.value + "_calibration" in protocol.data_hashes for arm in EvidenceArm):
            raise ValueError("Protocol binds calibration fits; both scorers must be supplied")
        self._initial_checkpoint()  # validate item/cohort/identity contracts without calls

    def _initial_checkpoint(self):
        return ExperimentCheckpoint(
            self.protocol, self.items,
            draws=tuple(DrawRecord(item.item_id, arm, index, metadata={"attempted": False})
                        for item in self.items for arm in EvidenceArm for index in range(self.protocol.draws)),
            rewrites=tuple(RewriteRecord(item.item_id, item.evidence) for item in self.items),
        )

    def load(self) -> ExperimentRun:
        """Read and validate saved state without constructing provider clients."""
        result = self.journal.load()
        result.checkpoint.validate_restart(self.protocol, self.items)
        identities = [(arm.item_id, arm.arm) for arm in result.arms]
        if len(set(identities)) != len(identities):
            raise ValueError("Duplicate saved arm identity")
        for item in self.items:
            for arm in EvidenceArm:
                draws = [draw for draw in result.checkpoint.draws if draw.item_id == item.item_id and draw.arm == arm]
                terminal = (item.item_id, arm) in identities
                expected = self.protocol.draws if terminal else 0
                if len(draws) != self.protocol.draws or sum(draw.status != RecordStatus.PENDING for draw in draws) != expected:
                    raise ValueError("Ambiguous saved arm/draw completion markers prevent automatic restart")
                if arm == EvidenceArm.REWRITTEN:
                    rewrite = next(record for record in result.checkpoint.rewrites if record.item_id == item.item_id)
                    if (rewrite.status != RecordStatus.PENDING) != terminal:
                        raise ValueError("Ambiguous saved rewrite/arm completion markers prevent automatic restart")
        if any(identity not in {
            (item.item_id, arm) for item in self.items for arm in EvidenceArm} for identity in identities):
            raise ValueError("Unknown saved arm identity")
        return result

    def _validate_workflow(self, workflow, budget):
        if not isinstance(workflow, RecallWorkflow) or workflow.budget is not budget or not workflow.strict_budget:
            raise ValueError("Experiment requires a strict workflow using the supplied remaining budget")
        policy = workflow.rewrite_policy
        if policy is None or self.protocol.rewrite_method_version != REWRITE_METHOD_VERSION:
            raise ValueError("Experiment rewrite policy/method version differs from protocol")
        clients = {"target": workflow.target, "rewriter": policy.rewriter.client, "verifier": policy.verifier}
        for role, client in clients.items():
            if client.config.to_dict() != getattr(self.protocol, role + "_model").to_dict():
                raise ValueError("Workflow model configuration differs from frozen protocol")
        for arm, scorer in self.scorers.items():
            if scorer is not None:
                key = arm.value + "_calibration"
                if self.protocol.data_hashes.get(key) != scorer.calibration_fingerprint:
                    raise ValueError("Calibration fit fingerprint differs from frozen protocol data_hashes")
            workflow.validate(rewrite=arm == EvidenceArm.REWRITTEN, ensemble=self.spec, scorer=scorer)

    def run(self, *, decide=None, components=None, max_arms: int | None = None) -> ExperimentRun:
        """Run pending whole arms, optionally stopping at a deliberate safe boundary.

        ``max_arms`` only segments scheduling; it does not alter draws/settings.
        A fatal stopped run returns its partial state without issuing more calls.
        A leftover write-ahead intent refuses automatic restart before clients.
        """
        if max_arms is not None and (type(max_arms) is not int or max_arms < 1):
            raise ValueError("max_arms must be a positive integer")
        if decide is not None and not callable(decide) or components is not None and not callable(components):
            raise TypeError("decide and components must be callable")
        project = decide or (lambda response: response.content)

        def safe_decide(response):
            answer = project(response)
            json_value(answer)  # invalid projections become explicit workflow failures
            return answer

        with self.journal.writer():
            result = self.load() if self.journal.path.exists() else ExperimentRun(self._initial_checkpoint())
            resume = result.checkpoint.validate_restart(self.protocol, self.items)
            if result.checkpoint.status == RecordStatus.COMPLETED or result.stop_reason:
                return result
            budget = CallBudget(resume.remaining_requests)
            # A construction-only factory is the public contract. Preserve an
            # intent even if it accidentally probes before we can reject it.
            self.journal.save(result, {"kind": "construction", "execution_id": resume.execution_id})
            workflow = self.factory(budget)
            if budget.used:
                raise ValueError("Workflow factory must not perform provider calls")
            try:
                self._validate_workflow(workflow, budget)
            except (ValueError, TypeError):
                self.journal.save(result)  # construction finished without physical activity
                raise
            result = self._preflight(result, workflow, budget, resume.execution_id)
            if result.stop_reason:
                return result
            terminal = {(arm.item_id, arm.arm) for arm in result.arms}
            processed = 0
            for item_index, item in enumerate(self.items):
                order = tuple(EvidenceArm) if item_index % 2 == 0 else tuple(reversed(EvidenceArm))
                for arm in order:
                    if (item.item_id, arm) in terminal:
                        continue
                    result = self._run_arm(result, workflow, budget, resume.execution_id, item_index, item, arm,
                                           safe_decide, components)
                    processed += 1
                    if result.stop_reason or max_arms is not None and processed >= max_arms:
                        return result
            return result

    def _preflight(self, result, workflow, budget, execution_id):
        intent = {"kind": "preflight", "execution_id": execution_id}
        self.journal.save(result, intent)
        preflight = workflow.preflight(rewrite=True, ensemble=self.spec,
                                      scorer=self.scorers[EvidenceArm.REWRITTEN], request_namespace=execution_id)
        requests = self._requests(budget.events, execution_id)
        stop = None if preflight.ready_for_batch else preflight.failure_kind or "capability_unavailable"
        checkpoint = replace(result.checkpoint, requests=result.checkpoint.requests + requests,
                             status=RecordStatus.PENDING if stop is None else RecordStatus.FAILED)
        result = replace(result, checkpoint=checkpoint, preflight=json_value(preflight_snapshot(preflight)), stop_reason=stop)
        self.journal.finish(result, intent)
        return result

    def _run_arm(self, result, workflow, budget, execution_id, item_index, item, arm, decide, components):
        intent = {"kind": "arm", "execution_id": execution_id, "item_id": item.item_id, "arm": arm.value}
        self.journal.save(result, intent)
        offset = budget.used
        namespace = f"{execution_id}/arm/{item_index}/{arm.value}"
        label_free = RewriteInput(item.item_id, item.instructions, item.evidence, item.as_of)
        try:
            self._validate_workflow(workflow, budget)
            outcome = workflow.run(label_free, rewrite=arm == EvidenceArm.REWRITTEN, ensemble=self.spec,
                                   decide=decide, components=components, scorer=self.scorers[arm],
                                   request_namespace=namespace)
        except ValueError:
            if budget.used != offset:
                raise  # provider activity without a complete outcome remains ambiguous
            message = "Workflow configuration or calibration differs from the frozen experiment."
            outcome = WorkflowResult(item.item_id, item.evidence, None, RecordStatus.FAILED, "configuration",
                                     arm.value + "_ensemble", self.protocol.draws,
                                     rewrite=RewriteRecord(item.item_id, item.evidence, RecordStatus.FAILED,
                                                           error=message, stage="configuration")
                                     if arm == EvidenceArm.REWRITTEN else None,
                                     error=message, failure_kind="configuration_error")
        try:
            self._validate_workflow(workflow, budget)
        except ValueError:
            message = "Workflow configuration or calibration fit changed during the arm."
            outcome = replace(outcome, status=RecordStatus.FAILED, failure_kind="configuration_error",
                              error=message, answer=None, response=None, ensemble=None,
                              scored_ensemble=None, p_memorized=None,
                              draws=tuple(replace(draw, status=RecordStatus.FAILED, error=message,
                                                  failure_kind="configuration_error") for draw in outcome.draws))
        requests = self._requests(budget.events[offset:], execution_id, namespace, item, arm, outcome)
        draws = self._draws(item, arm, outcome, requests)
        replacements = {draw.identity: draw for draw in draws}
        rewrites = tuple(outcome.rewrite if rewrite.item_id == item.item_id and outcome.rewrite is not None else rewrite
                         for rewrite in result.checkpoint.rewrites)
        arms = result.arms + (ExperimentArm(item.item_id, arm, arm_snapshot(outcome)),)
        complete = len(arms) == len(self.items) * len(EvidenceArm)
        fatal = outcome.failure_kind if outcome.failure_kind in FATAL_FAILURES else None
        checkpoint = replace(result.checkpoint,
                             draws=tuple(replacements.get(draw.identity, draw) for draw in result.checkpoint.draws),
                             rewrites=rewrites, requests=result.checkpoint.requests + requests,
                             status=RecordStatus.FAILED if fatal else RecordStatus.COMPLETED if complete else RecordStatus.PENDING)
        result = replace(result, checkpoint=checkpoint, arms=arms, stop_reason=fatal)
        self.journal.finish(result, intent)
        return result

    def _draws(self, item, arm, outcome, requests):
        attempted = {draw.draw_index: draw for draw in outcome.draws}
        records = []
        for index in range(self.protocol.draws):
            draw = attempted.get(index)
            if draw is None:
                records.append(DrawRecord(item.item_id, arm, index, RecordStatus.FAILED, stage=outcome.stage,
                                          error=outcome.error or "Draw not attempted after the arm stopped.",
                                          metadata={"attempted": False, "physical_attempts": 0,
                                                    "failure_kind": outcome.failure_kind,
                                                    "intervention_id": None if outcome.rewrite is None else
                                                        outcome.rewrite.fidelity_checks.get("intervention_id"),
                                                    "evidence_hash": None if outcome.accepted_evidence is None else
                                                        text_hash(outcome.accepted_evidence),
                                                    "answer": None, "guarded_score": None}))
            else:
                records.append(DrawRecord(
                    item.item_id, arm, index, draw.status, draw.prompt, draw.response, "target", draw.error,
                    metadata=json_value({"attempted": True,
                                         "physical_attempts": sum(request.arm == arm and request.draw_index == index
                                                                  for request in requests),
                                         "failure_kind": draw.failure_kind,
                                         "intervention_id": draw.intervention_id, "evidence_hash": draw.evidence_hash,
                                         "answer": draw.answer, "components": draw.components,
                                         "diagnostics": draw.diagnostics,
                                         "guarded_score": None if draw.guarded_score is None else asdict(draw.guarded_score)}),
                ))
        return tuple(records)

    @staticmethod
    def _requests(events, execution_id, namespace=None, item=None, arm=None, outcome=None):
        requests = []
        for event in events:
            item_id, draw_index, request_arm = None, None, None
            if "/probe/" in event.stage:
                role = event.stage.rsplit("/probe/", 1)[1]
                prompt = _PROBE_PROMPT
            elif namespace is not None and event.stage.startswith(namespace + "/target/"):
                role, item_id, request_arm = "target", item.item_id, arm
                draw_index = int(event.stage.removeprefix(namespace + "/target/"))
                prompt = next(draw.prompt for draw in outcome.draws if draw.draw_index == draw_index)
            elif event.stage in {"rewrite", "verify"} and outcome is not None and outcome.rewrite is not None:
                role, item_id = ("rewriter" if event.stage == "rewrite" else "verifier"), item.item_id
                prompt = outcome.rewrite.rewrite_prompt if role == "rewriter" else outcome.rewrite.verification_prompt
            else:
                raise ValueError("Physical request stage has no exact experiment identity")
            requests.append(RequestRecord(execution_id, event, role, text_hash(prompt), item_id, request_arm, draw_index))
        return tuple(requests)
