"""Run, dry-run or replay an explicitly configured paired experiment.

Portable protocol JSON never contains credentials. Live native runs explicitly
name each role's credential environment variable. This command never loads an
environment file. Dry-run/replay do not resolve credentials or create clients.
"""

import argparse
import json
import math
import os
import sys
from dataclasses import replace
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from recall_guard.core.completion import CompletionProtocol, FeatureChoices
from recall_guard.core.provider_anthropic import AnthropicClient
from recall_guard.core.providers import OpenAICompatibleClient
from recall_guard.harness.intervention import ExperimentRunner
from recall_guard.harness.intervention_config import experiment_plan, load_experiment_config, read_experiment_json
from recall_guard.harness.intervention_journal import ExperimentJournal
from recall_guard.harness.intervention_report import export_experiment, load_experiment_report
from recall_guard.harness.rewriting import EvidenceRewriter, RewritePolicy
from recall_guard.harness.workflow import RecallWorkflow

_ROLES = ("target", "rewriter", "verifier")
_FIT_KEYS = {"original_calibration", "rewritten_calibration"}


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true", help="Validate and print an offline plan; zero provider calls")
    mode.add_argument("--run", action="store_true", help="Probe all roles, then run under the protocol's hard cap")
    mode.add_argument("--replay", type=Path, metavar="REPORT_DIR", help="Verify and replay offline using the report's saved analysis options")
    parser.add_argument("--protocol", type=Path, help="ExperimentProtocol.to_dict() JSON, schema_version=1")
    parser.add_argument("--items", type=Path, help="UTF-8 {schema_version:1,items:[...]} JSON; data_hashes.evaluation is its byte hash")
    parser.add_argument("--data", action="append", default=[], metavar="NAME=PATH", help="File binding for each other dataset hash; repeatable")
    parser.add_argument("--cutoff-registry", type=Path, help="Optional versioned source-bearing exact-model registry JSON")
    parser.add_argument("--checkpoint", type=Path, help="Runtime journal path; existing safe state resumes with remaining budget")
    parser.add_argument("--output", type=Path, help="NEW portable-report directory, separate from the runtime journal")
    for role in _ROLES:
        parser.add_argument(f"--{role}-credential-env", help=f"Explicit environment variable NAME for {role}; never a literal key")
    parser.add_argument("--answer-format", choices=("text", "json"), default="text",
                        help="text: stripped content; json: object containing --answer-field and optional confidence")
    parser.add_argument("--answer-field", default="answer", help="Single JSON key for the answer (e.g. direction); no nested field syntax")
    parser.add_argument("--confidence-level", type=float, default=0.95, help="Report bootstrap interval confidence (default .95)")
    component = parser.add_mutually_exclusive_group()
    component.add_argument("--confidence-component", help="Explicit JSON confidence field and report component (JSON default: confidence)")
    component.add_argument("--no-confidence-component", action="store_true", help="Do not project a response confidence component")
    parser.add_argument("--max-arms", type=int, help="Stop at a safe whole-arm boundary, export partial report, then resume to a NEW output")
    return parser


def _bindings(values):
    bindings = {}
    for value in values:
        name, separator, path = value.partition("=")
        if not separator or not name.strip() or not path.strip() or name in bindings:
            raise ValueError("--data requires unique NAME=PATH bindings")
        bindings[name] = Path(path)
    return bindings


def _runtime_models(protocol, args):
    models, environment = {}, {}
    for role in _ROLES:
        config = getattr(protocol, role + "_model")
        if config.protocol == CompletionProtocol.CALLBACK:
            raise ValueError(f"{role}: CLI cannot construct a callback; use Python ExperimentRunner with a physical-admission hook")
        reference = getattr(args, role + "_credential_env")
        if reference is None:
            raise ValueError(f"{role}: --{role}-credential-env must explicitly name the credential environment variable")
        config = replace(config, credential_ref=reference)
        name = reference.removeprefix("env:")
        value = os.environ.get(name)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{role}: referenced credential is missing or empty; batch was not scheduled")
        environment[name] = value
        models[role + "_model"] = config
    return replace(protocol, **models), environment


def _native_factory(protocol, environment):
    def factory(budget):
        clients = {}
        for role in _ROLES:
            model = getattr(protocol, role + "_model")
            client_type = AnthropicClient if model.protocol == CompletionProtocol.ANTHROPIC_MESSAGES else OpenAICompatibleClient
            clients[role] = client_type(model, budget, environment=environment)
        policy = RewritePolicy(EvidenceRewriter(clients["rewriter"]), clients["verifier"])
        return RecallWorkflow(clients["target"], rewrite_policy=policy, budget=budget, strict_budget=True,
                              features=FeatureChoices(ensemble_draws=protocol.draws, rewriting=True))
    return factory


def _projections(args):
    if args.answer_format == "text":
        def text_answer(response):
            answer = response.content.strip()
            if not answer:
                raise ValueError("Text response answer must be nonempty")
            return answer
        return text_answer, None

    def document(response):
        value = json.loads(response.content)
        if not isinstance(value, dict) or args.answer_field not in value or value[args.answer_field] is None:
            raise ValueError("JSON response must contain the selected non-null answer field")
        return value

    def decide(response):
        return document(response)[args.answer_field]

    def components(response):
        value = document(response).get(args.confidence_component)
        if value is None:
            return {}
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("Explicit response confidence must be a finite number in [0,1]")
        return {args.confidence_component: value}

    return decide, None if args.no_confidence_component else components


def _print(value):
    """Emit lossless JSON on consoles whose encoding cannot represent metadata."""
    print(json.dumps(value, ensure_ascii=True, allow_nan=False, sort_keys=True, indent=2))


def _projection_config(args):
    return {"format": args.answer_format, "answer_field": args.answer_field if args.answer_format == "json" else None,
            "confidence_component": args.confidence_component
            if args.answer_format == "json" and not args.no_confidence_component else None}


def _check_cli_state(checkpoint, protocol, args, *, create=False):
    """Bind command projection semantics without changing the journal schema.

    Exclusive creation never replaces conflicting/partial state. A crash during
    publication leaves a sidecar requiring explicit repair, before any calls.
    Existing Python-only journals must resume through their Python API owner.
    """
    path = checkpoint.with_name(checkpoint.name + ".cli.json")
    expected = {"schema_version": 1, "protocol_id": protocol.protocol_id, "projection": _projection_config(args)}
    if path.exists():
        try:
            saved, _ = read_experiment_json(path)
        except (ValueError, OSError):
            raise ValueError("CLI projection sidecar is unreadable or partial; reconcile it before resuming") from None
        if (not isinstance(saved, dict) or type(saved.get("schema_version")) is not int
                or saved != expected):
            raise ValueError("CLI projection sidecar differs from this protocol/projection; restore the original command or use the Python API")
        return
    if checkpoint.exists():
        raise ValueError("Existing journal has no CLI projection sidecar; resume Python-only journals through the Python ExperimentRunner API")
    if create:
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with path.open("x", encoding="utf-8", newline="\n") as handle:
                json.dump(expected, handle, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
        except FileExistsError:
            raise ValueError("CLI projection sidecar appeared concurrently; inspect it and retry without overwriting") from None


def _execute(args, workflow_factory):
    if args.replay is not None:
        report = load_experiment_report(args.replay)
        _print(report.recompute_metrics())  # saved analysis options own replay
        return 0
    if args.protocol is None or args.items is None:
        raise ValueError("--protocol and --items are required for --dry-run and --run")
    if not math.isfinite(args.confidence_level) or not 0 < args.confidence_level < 1:
        raise ValueError("--confidence-level must be in (0,1)")
    if not args.answer_field.strip() or args.confidence_component is not None and not args.confidence_component.strip():
        raise ValueError("Answer and confidence field names must be nonempty")
    if args.answer_format == "text" and args.confidence_component is not None:
        raise ValueError("--confidence-component requires --answer-format json; text answers provide no confidence component")
    args.confidence_component = args.confidence_component or "confidence"
    if args.max_arms is not None and args.max_arms < 1:
        raise ValueError("--max-arms must be positive")
    config = load_experiment_config(args.protocol, args.items, data_files=_bindings(args.data),
                                    cutoff_registry=args.cutoff_registry)
    protocol = config.protocol
    for role in _ROLES:
        reference = getattr(args, role + "_credential_env")
        if reference is not None:
            try:
                replace(getattr(protocol, role + "_model"), credential_ref=reference)
            except (TypeError, ValueError) as error:
                raise ValueError(f"{role}: {error}") from None
    if args.dry_run:
        prior = None
        if args.checkpoint is not None:
            _check_cli_state(args.checkpoint, protocol, args)
        if args.checkpoint is not None and args.checkpoint.exists():
            if _FIT_KEYS & protocol.data_hashes.keys():
                prior = ExperimentJournal(args.checkpoint).load()
            else:
                prior = ExperimentRunner(protocol, config.items, lambda budget: None, args.checkpoint).load()
        plan = experiment_plan(config, prior_run=prior)
        plan["answer_projection"] = {"format": args.answer_format, "answer_field": args.answer_field,
                                     "confidence_component": None if args.no_confidence_component or args.answer_format == "text" else args.confidence_component,
                                     "confidence_availability": "unavailable in text mode" if args.answer_format == "text"
                                     else "available only when an explicit valid numeric field is returned"}
        _print(plan)
        return 0
    if args.checkpoint is None or args.output is None:
        raise ValueError("--run requires --checkpoint and --output (a new report directory)")
    if args.output.exists():
        raise ValueError("--output must be a NEW directory; prior report bundles are never overwritten")
    if args.checkpoint.resolve().is_relative_to(args.output.resolve()):
        raise ValueError("Runtime checkpoint must be separate from the new report directory")
    if _FIT_KEYS & protocol.data_hashes.keys():
        raise ValueError("Protocol requires actual matching fitted scorers; CLI has no fit loader. Use Python ExperimentRunner.")
    _check_cli_state(args.checkpoint, protocol, args)
    if workflow_factory is None:
        protocol, environment = _runtime_models(protocol, args)
        workflow_factory = _native_factory(protocol, environment)
    _check_cli_state(args.checkpoint, protocol, args, create=True)
    runner = ExperimentRunner(protocol, config.items, workflow_factory, args.checkpoint)
    decide, components = _projections(args)
    run = runner.run(decide=decide, components=components, max_arms=args.max_arms)
    manifest = export_experiment(run, args.output, confidence_level=args.confidence_level,
                                 confidence_component=None if args.no_confidence_component else args.confidence_component)
    _print({"status": run.checkpoint.status.value, "stop_reason": run.stop_reason, "manifest": str(manifest),
            "physical_requests": len(run.checkpoint.requests), "request_budget": protocol.request_budget,
            "historical": config.historical, "validation_execution": "unverified"})
    return 0 if run.checkpoint.status == "completed" and run.stop_reason is None else 1


def main(argv=None, *, workflow_factory=None):
    """Command entry point; optional construction-only factory aids offline fixtures."""
    args = _parser().parse_args(argv)
    try:
        return _execute(args, workflow_factory)
    except (ValueError, TypeError, KeyError, OSError) as error:
        # Keep Unicode paths identifiable even on a strict legacy Windows console.
        message = f"Experiment configuration/execution error: {error}"
        print(message.encode("ascii", errors="backslashreplace").decode("ascii"), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
