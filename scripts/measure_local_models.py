"""Validate or explicitly execute a frozen local measurement plan.

PowerShell: uv run --no-sync python scripts/measure_local_models.py --help
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from recall_guard.harness.local_evaluation import load_plan


def ollama_observer(plan):
    """Read fresh Ollama metadata only; never load/unload or generate here."""
    from types import SimpleNamespace

    from recall_guard.core.local_evidence import LocalModelProvenance
    from recall_guard.harness.local_execution import HostObservation, HostResources
    from scripts.probe_local_model import observe_host

    config = SimpleNamespace(endpoint=plan.config.endpoint, model=plan.config.model)
    def observe():
        data = observe_host(config)
        return HostObservation(LocalModelProvenance(
            runtime='ollama', runtime_version=data['runtime_version'],
            artifact_digest=data['digest'], backend=data['observed_backend'],
            context_length=data['effective_context_tokens']), tuple(data['loaded_models']),
            HostResources(model_allocation_bytes=data['model_allocation_bytes'],
                          accelerator_allocation_bytes=data['accelerator_allocation_bytes']))
    return observe


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True, type=Path)
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--corpus', required=True, type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--dry-run', action='store_true')
    mode.add_argument('--execute', action='store_true')
    parser.add_argument('--journal', type=Path)
    parser.add_argument('--observer', choices=['ollama'])
    args = parser.parse_args(argv)
    if args.execute and (args.journal is None or args.observer is None):
        parser.error('--execute requires a new --journal and explicit --observer ollama')
    try:
        plan = load_plan(args.plan, args.manifest, args.corpus)
    except (ValueError, TypeError, KeyError, OSError):
        # Parsing failures may contain private paths or corpus material.
        print('Local measurement refused: invalid plan, manifest or corpus.', file=sys.stderr)
        return 2
    if args.dry_run:
        print(json.dumps({'status': 'dry_run', 'plan': plan.safe_export()}, indent=2))
        return 0
    from recall_guard.harness.local_evaluation import LocalAttemptBudget
    from recall_guard.harness.local_execution import execute_plan
    from recall_guard.harness.local_reporting import build_report, report_paths, write_report

    try:
        if any(path.exists() for path in report_paths(args.journal)):
            raise ValueError('report_already_exists')
        budget = LocalAttemptBudget(plan, args.journal)
        result = execute_plan(budget, ollama_observer(plan))
        report = build_report(budget, result)
        write_report(args.journal, report)
    except (ValueError, OSError):
        print('Local measurement refused: execution or journal unavailable.', file=sys.stderr)
        return 2
    print(json.dumps({'status': result.status,
                     'calibrated_claims_allowed': result.calibrated_claims_allowed,
                     'observed_execution_mode': result.observed_execution_mode,
                     'npu_verified': result.npu_verified, 'attempts': budget.used}, indent=2))
    return 0 if result.status == 'completed' else 2


if __name__ == '__main__':
    raise SystemExit(main())
