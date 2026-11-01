"""Bounded original-input NIM refresh; planning never resolves credentials.

The future --run command requires a newly approved measurement budget. Every
single result is predesignated draw zero of its item's three serial draws.
This exploratory baseline cannot complete the four-condition mitigation pilot.
"""

import argparse
import hashlib
import json
import os
import time
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path

from recall_guard.core.completion import FeatureChoices, ModelConfig, ModelResponse
from recall_guard.core.ensemble import EnsembleSpec
from recall_guard.core.providers import OpenAICompatibleClient
from recall_guard.core.readiness_types import CancellationToken, ReadinessPolicy
from recall_guard.core.request_control import CallBudget
from recall_guard.harness.evaluator import _parse_direction
from recall_guard.harness.intervention_types import ExperimentItem, RecordStatus
from recall_guard.harness.request_failures import FATAL_FAILURES
from recall_guard.harness.workflow import RecallWorkflow
from scripts.prepare_completion_inputs import label_free_input

INPUT_FILES = ("evaluation.json", "calibration-is.json", "calibration-oos.json")
PHASE_CAPS = {"eligibility": 2, "calibration": 80, "evaluation": 64}
DEFAULT_INPUTS = Path("data/experiments/2026-10-31-completion-api")


def target_config() -> ModelConfig:
    return ModelConfig(
        protocol="openai_chat", provider="nvidia_nim",
        endpoint="https://integrate.api.nvidia.com/v1/chat/completions",
        model="openai/gpt-oss-20b", credential_ref="NVIDIA_API_KEY",
        parameters={"max_tokens": 2048, "temperature": 0, "logprobs": True, "top_logprobs": 20},
        timeout_s=600, max_retries=0, min_call_interval_s=1.5,
        readiness=ReadinessPolicy(connect_timeout_s=10, socket_idle_timeout_s=600,
                                  max_wait_s=1200, max_poll_requests=0, allow_ambiguous_replay=False),
    )


def ensemble_spec() -> EnsembleSpec:
    return EnsembleSpec(draws=3, max_workers=1, min_parsed=2, retain_draws=True,
                        max_tokens=2048, temperature=0)


def load_inputs(directory: Path) -> dict[str, tuple[ExperimentItem, ...]]:
    groups = {}
    seen_ids, seen_prompts = set(), set()
    for name, expected in zip(INPUT_FILES, (16, 20, 20), strict=True):
        data = json.loads((directory / name).read_text(encoding="utf-8"))
        rows = tuple(ExperimentItem.from_dict(row) for row in data["items"])
        if len(rows) != expected:
            raise ValueError("Refresh requires the frozen 16/20/20 input sizes")
        for item in rows:
            prompt_key = (item.instructions, item.evidence, item.as_of)
            if item.item_id in seen_ids or prompt_key in seen_prompts:
                raise ValueError("Calibration and evaluation must be disjoint, including duplicate prompts")
            if type(item.label) is not int or item.label not in {-1, 0, 1}:
                raise ValueError("Direction labels must be integers in {-1, 0, 1}")
            seen_ids.add(item.item_id)
            seen_prompts.add(prompt_key)
        groups[name] = rows
    return groups


def build_plan(directory: Path, *, eligibility_request_cap: int = 2) -> dict:
    if type(eligibility_request_cap) is not int or eligibility_request_cap not in {1, 2}:
        raise ValueError("Eligibility request cap must be 1 or 2")
    groups = load_inputs(directory)
    phase_caps = dict(PHASE_CAPS, eligibility=eligibility_request_cap)
    return {
        "schema": "recall_guard.nim_refresh.v1", "status": "offline_plan",
        "target": target_config().to_dict(), "credential_reference": "NVIDIA_API_KEY",
        "ensemble": asdict(ensemble_spec()), "rewrite": False, "divergence": False,
        "input_hashes": {name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
                         for name in INPUT_FILES},
        "input_counts": {name: len(rows) for name, rows in groups.items()},
        "phase_caps": phase_caps, "maximum_total_requests": sum(phase_caps.values()),
        "nominal_requests": {"eligibility": 1, "calibration": 62, "evaluation": 49},
        "calibration": {"min_valid": 10, "min_auc": 0.6, "seed": 0, "max_workers": 1},
        "single_view": "draw_index=0, shared with ensemble; no additional requests",
        "scope": "Exploratory baseline and aggregation only; no mitigation efficacy claim",
        "restart": "Refuse any nonempty output directory; never replay ambiguous work",
        "budget_authorization": "New measurement approval required; no transfer from readiness allowance",
        "run_admission_window_s": 3600,
        "deadline_semantics": "Stop new calls after window; in-flight bounded I/O may finish later",
        "cost": None, "cost_unavailable_reason": "Provider pricing and monetary cost unavailable",
    }


def write_json(path: Path, value: dict) -> None:
    """Replace a local checkpoint atomically and flush it before subsequent work."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def response_snapshot(response) -> dict | None:
    """Explicit owning contracts only; never serialize arbitrary runtime objects."""
    if response is None:
        return None
    if isinstance(response, ModelResponse):
        return response.to_dict()
    # Readiness partial responses have a distinct, bounded public contract.
    from recall_guard.core.readiness_evidence import PartialResponse
    if isinstance(response, PartialResponse):
        return response.to_dict()
    raise TypeError("Unsupported response evidence type")


def ledger(budget: CallBudget) -> dict:
    # Provider free-text errors are intentionally excluded from this public view.
    fields = ("call_id", "stage", "model", "attempt", "started_at", "timestamp", "status",
              "latency_s", "usage", "cost", "status_code", "purpose")
    return {"limit": budget.limit, "used": budget.used,
            "events": [{name: event.to_dict().get(name) for name in fields} for event in budget.events]}


class RunDeadlineToken(CancellationToken):
    """Expose the run deadline to the stock admission and pacing guards."""

    def __init__(self, deadline: float):
        super().__init__()
        self.deadline = deadline

    @property
    def cancelled(self) -> bool:
        return super().cancelled or time.monotonic() >= self.deadline

    def wait(self, timeout_s: float) -> bool:
        super().wait(min(timeout_s, max(0.0, self.deadline - time.monotonic())))
        return self.cancelled


class RecordedClient(OpenAICompatibleClient):
    """Thin stock-client receipt hook; admission and provider handling stay upstream."""

    def __init__(self, config: ModelConfig, budget: CallBudget, *, directory: Path, deadline: float,
                 environment: dict[str, str] | None = None):
        self.directory = directory
        self.deadline = deadline
        self.deadline_token = RunDeadlineToken(deadline)
        self.sequence = 0
        super().__init__(config, budget, features=FeatureChoices(mia_scoring=True), environment=environment)

    def _observe(self, prompt, parameters, *, stage, **kwargs):
        if time.monotonic() >= self.deadline:
            raise TimeoutError("Run admission window exhausted")
        self.sequence += 1
        prefix = self.directory / f"request-{self.sequence:03d}"
        write_json(prefix.with_suffix(".intent.json"), {
            "stage": stage, "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "status": "intent", "timestamp": datetime.now(UTC).isoformat(),
        })
        kwargs["cancellation"] = self.deadline_token
        result = super()._observe(prompt, parameters, stage=stage, **kwargs)
        # Apply the stock registered-secret sanitizer to complete/partial data.
        failure = self.sanitize_execution_failure(
            result.error or RuntimeError("Evidence receipt"), response=result.response)
        write_json(prefix.with_suffix(".receipt.json"), {
            "stage": stage, "ready": result.ready,
            "response": response_snapshot(failure.response_evidence),
            "observations": [entry.to_dict() for entry in result.observations],
            "budget": ledger(self.budget),
        })
        return replace(result, response=failure.response_evidence)


def decide(response: ModelResponse) -> int:
    direction = _parse_direction(response.content)
    if direction is None:
        raise ValueError("Direction parse failed")
    return direction


def item_snapshot(item: ExperimentItem, result) -> dict:
    """One JSON row per item; draws retain admission order and zero identity."""
    draws = [{"draw_index": draw.draw_index, "status": draw.status.value,
              "answer": draw.answer, "failure_kind": draw.failure_kind,
              "response": response_snapshot(draw.response),
              "guarded_score": None if draw.guarded_score is None else asdict(draw.guarded_score)}
             for draw in result.draws]
    single = next((draw for draw in draws if draw["draw_index"] == 0), None)
    score = result.scored_ensemble
    return {"item_id": item.item_id, "label": item.label, "cohort": item.cohort,
            "status": result.status.value, "failure_kind": result.failure_kind,
            "single": single, "single_correct": single is not None and single["answer"] == item.label,
            "ensemble_answer": result.answer,
            "ensemble_correct": result.status == RecordStatus.COMPLETED and result.answer == item.label,
            "n_attempted": result.n_attempted, "n_usable": result.n_usable,
            "agreement": None if result.ensemble is None else result.ensemble.agreement,
            "scored_ensemble": None if score is None else asdict(score), "draws": draws}


def credential_environment(path: Path) -> dict[str, str]:
    """Read only an explicitly selected runtime secret file; never interpolate."""
    from dotenv import dotenv_values

    value = dotenv_values(path, interpolate=False).get("NVIDIA_API_KEY")
    if not value:
        raise ValueError("Credential file lacks the selected NVIDIA_API_KEY")
    return {"NVIDIA_API_KEY": value}


def run(inputs: Path, output: Path, *, plan_file: Path, credential_file: Path | None = None,
        eligibility_request_cap: int = 2) -> dict:
    """Execute only after explicit external approval of this separate budget."""
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Nonempty output: ambiguous restart is forbidden")
    plan = build_plan(inputs, eligibility_request_cap=eligibility_request_cap)
    frozen_plan = json.loads(plan_file.read_text(encoding="utf-8"))
    if json.dumps(frozen_plan, sort_keys=True, allow_nan=False) != json.dumps(plan, sort_keys=True, allow_nan=False):
        raise ValueError("Frozen plan differs from current inputs or configuration")
    groups = load_inputs(inputs)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    deadline = started + plan["run_admission_window_s"]
    write_json(output / "plan.json", plan)
    budgets = {}
    receipt = {"status": "incomplete", "phase": "eligibility", "processed_items": 0,
               "failed_items": 0,
               "cost": None, "cost_unavailable_reason": plan["cost_unavailable_reason"]}

    def checkpoint():
        receipt["budgets"] = {name: ledger(budget) for name, budget in budgets.items()}
        receipt["physical_requests"] = sum(budget.used for budget in budgets.values())
        receipt["elapsed_s"] = time.monotonic() - started
        receipt["admission_window_exhausted"] = time.monotonic() >= deadline
        write_json(output / "receipt.json", receipt)

    def workflow(phase: str) -> RecallWorkflow:
        receipt["phase"] = phase
        directory = output / phase
        directory.mkdir()
        budgets[phase] = CallBudget(plan["phase_caps"][phase])
        checkpoint()  # Client construction may resolve credentials; intent precedes it.
        client = RecordedClient(target_config(), budgets[phase], directory=directory, deadline=deadline,
                                environment=environment)
        return RecallWorkflow(client, budget=budgets[phase], strict_budget=True)

    try:
        environment = None if credential_file is None else credential_environment(credential_file)
        eligibility = workflow("eligibility").preflight(rewrite=False)
        checkpoint()
        if not eligibility.ready_for_batch:
            receipt["failure_kind"] = eligibility.failure_kind
            return receipt
        calibration_workflow = workflow("calibration")
        fit = calibration_workflow.calibrate(
            is_memorized=[label_free_input(item) for item in groups["calibration-is.json"]],
            oos_control=[label_free_input(item) for item in groups["calibration-oos.json"]],
            rewrite=False, **plan["calibration"])
        write_json(output / "calibration.json", {
            "status": fit.status.value, "failure_kind": fit.failure_kind,
            "counts": {name: dict(value) for name, value in fit.counts.items()},
            "identity": asdict(fit.identity), "holdout_auc": fit.holdout_auc, "is_weak": fit.is_weak,
            "holdout_labels": fit.holdout_labels, "holdout_probabilities": fit.holdout_probabilities,
            "fingerprint": None if fit.scorer is None else fit.scorer.calibration_fingerprint,
        })
        checkpoint()
        if fit.scorer is None or fit.status != RecordStatus.COMPLETED:
            receipt["failure_kind"] = fit.failure_kind
            return receipt
        evaluation = workflow("evaluation")
        preflight = evaluation.preflight(rewrite=False, ensemble=ensemble_spec(), scorer=fit.scorer)
        checkpoint()
        if not preflight.ready_for_batch:
            receipt["failure_kind"] = preflight.failure_kind
            return receipt
        for item in groups["evaluation.json"]:
            write_json(output / "item-intent.json", {"item_id": item.item_id, "status": "intent"})
            event_start = len(budgets["evaluation"].events)
            result = evaluation.run(label_free_input(item), rewrite=False, ensemble=ensemble_spec(),
                                    scorer=fit.scorer, decide=decide, request_namespace=item.item_id)
            write_json(output / f"item-{receipt['processed_items']:02d}.json", item_snapshot(item, result))
            receipt["processed_items"] += 1
            failed = result.status != RecordStatus.COMPLETED
            receipt["failed_items"] += int(failed)
            checkpoint()
            events = budgets["evaluation"].events[event_start:]
            local_quality_failure = (bool(events) and all(event.status == "succeeded" for event in events)
                                     and result.failure_kind not in FATAL_FAILURES
                                     and all(draw.failure_kind in {None, "projection", "components"}
                                             for draw in result.draws)
                                     and time.monotonic() < deadline)
            if failed and not local_quality_failure:
                receipt["failure_kind"] = result.failure_kind
                return receipt
        receipt["status"] = "completed"
        return receipt
    except Exception:
        # Arbitrary exception text can carry secrets; request receipts retain typed evidence.
        receipt["failure_kind"] = "execution_failed"
        return receipt
    finally:
        checkpoint()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--plan", action="store_true")
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--run", action="store_true", help="Requires newly approved measurement budget")
    parser.add_argument("--inputs", type=Path, default=DEFAULT_INPUTS)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--credential-file", type=Path, help="Run only: explicit dotenv secret path")
    parser.add_argument("--plan-file", type=Path, help="Run only: previously frozen offline plan")
    parser.add_argument("--eligibility-request-cap", type=int, choices=(1, 2), default=2)
    args = parser.parse_args(argv)
    if args.run:
        if args.output is None or args.plan_file is None:
            parser.error("--run requires explicit --output and --plan-file")
        result = run(args.inputs, args.output, plan_file=args.plan_file, credential_file=args.credential_file,
                     eligibility_request_cap=args.eligibility_request_cap)
    else:
        if args.credential_file:
            parser.error("--credential-file requires --run")
        result = build_plan(args.inputs, eligibility_request_cap=args.eligibility_request_cap)
        if args.output:
            if args.output.exists():
                raise FileExistsError("Plan destination already exists")
            args.output.parent.mkdir(parents=True, exist_ok=True)
            write_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0 if result.get("status") in {"offline_plan", "completed"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
