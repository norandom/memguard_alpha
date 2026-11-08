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
    from recall_guard.core.local_evidence import LocalModelProvenance
    from recall_guard.harness.local_execution import HostObservation, HostResources
    from scripts.probe_local_model import observe_host

    # The plan's credential reference (if any) also authenticates metadata reads,
    # for example through the authenticating Windows gateway.
    config = _probe_config(plan)
    def observe():
        data = observe_host(config)
        return HostObservation(LocalModelProvenance(
            runtime='ollama', runtime_version=data['runtime_version'],
            artifact_digest=data['digest'], backend=data['observed_backend'],
            context_length=data['effective_context_tokens']), tuple(data['loaded_models']),
            HostResources(model_allocation_bytes=data['model_allocation_bytes'],
                          accelerator_allocation_bytes=data['accelerator_allocation_bytes']))
    return observe


def fastflowlm_observer(plan, profile_id):
    """Read fresh FastFlowLM gateway metadata only; FLM has no unload (stop the process host-side).

    Backend ``NPU`` comes from FLM's own /api/npu/status (FastFlowLM executes only
    on the NPU); context is the serving process's ``--ctx-len`` reported by the gateway.
    """
    from recall_guard.core.local_evidence import LocalModelProvenance
    from recall_guard.core.profiles import observe_fastflowlm_host
    from recall_guard.harness.local_execution import HostObservation, HostResources

    profile = _plan_profile(plan, profile_id)
    headers = _metadata_headers(plan, profile)
    def observe():
        data = observe_fastflowlm_host(plan.config.endpoint, plan.config.model, headers=headers)
        return HostObservation(LocalModelProvenance(
            runtime='fastflowlm', runtime_version=data['runtime_version'],
            artifact_digest=data['digest'], backend=data['observed_backend'],
            context_length=data['effective_context_tokens']), tuple(data['loaded_models']), HostResources())
    return observe


def _plan_profile(plan, profile_id):
    """The curated profile named by the manifest, carrying the plan's endpoint and credential."""
    from dataclasses import replace

    from recall_guard.core.profiles import load_profile

    profile = load_profile(profile_id)
    if profile.model_config.model != plan.config.model:
        raise ValueError('manifest_profile_model_mismatch')
    config = replace(profile.model_config, endpoint=plan.config.endpoint, credential_ref=plan.config.credential_ref)
    return replace(profile, model_config=config, allow_unauthenticated=plan.config.credential_ref is None)


def _metadata_headers(plan, profile):
    from recall_guard.core.profiles import metadata_headers
    from scripts.probe_local_model import ProbeConfig

    environment = ProbeConfig(plan.config.model, plan.config.model, plan.config.endpoint, '', '',
                              plan.config.credential_ref).environment()
    return metadata_headers(profile, environment)


def _probe_config(plan):
    from scripts.probe_local_model import ProbeConfig

    return ProbeConfig(plan.config.model, plan.config.model, plan.config.endpoint, '', '',
                       plan.config.credential_ref)


def host_gate(plan, snapshot_path, profile_id=None):
    """Shared RAM/metadata preflight with a host snapshot; returns a refusal code or None.

    A FastFlowLM manifest profile runs the fastflowlm preflight branch of that
    profile; Ollama plans keep the frozen probe configuration as before.
    """
    from recall_guard.core.profiles import FLM_RUNTIME, HostMemorySnapshot, load_profile, preflight_host
    from scripts.probe_local_model import host_preflight

    snapshot = HostMemorySnapshot.from_file(snapshot_path)
    if profile_id is not None and load_profile(profile_id).model_config.provider == FLM_RUNTIME:
        config = _probe_config(plan)
        report = preflight_host(_plan_profile(plan, profile_id), environment=config.environment(),
                                host_snapshot=snapshot)
    else:
        report = host_preflight(_probe_config(plan), snapshot)
    return None if report.passed else (report.reason or 'host_preflight_failed')


def manifest_profile_id(manifest_path):
    """``profile_id`` recorded by build_phase_a_inputs.py, or None (Ollama plans then behave as before).

    load_plan already validated the manifest bytes; an unreadable one only loses the
    FastFlowLM routing, and --observer fastflowlm refuses a plan without its profile.
    """
    try:
        value = json.loads(manifest_path.read_text(encoding='utf-8-sig')).get('profile_id')
    except (OSError, ValueError, AttributeError):
        return None
    return value if isinstance(value, str) else None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True, type=Path)
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--corpus', required=True, type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--dry-run', action='store_true')
    mode.add_argument('--execute', action='store_true')
    parser.add_argument('--journal', type=Path)
    parser.add_argument('--observer', choices=['ollama', 'fastflowlm'])
    parser.add_argument('--host-snapshot', type=Path,
                        help='fresh free-RAM snapshot from the model host; runs the host preflight before execution')
    args = parser.parse_args(argv)
    if args.execute and (args.journal is None or args.observer is None):
        parser.error('--execute requires a new --journal and explicit --observer ollama|fastflowlm')
    try:
        plan = load_plan(args.plan, args.manifest, args.corpus)
        profile_id = manifest_profile_id(args.manifest)
        if args.observer == 'fastflowlm':
            from recall_guard.core.profiles import FLM_RUNTIME, load_profile

            if profile_id is None or load_profile(profile_id).model_config.provider != FLM_RUNTIME:
                raise ValueError('fastflowlm_observer_requires_a_fastflowlm_manifest_profile')
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

    if args.host_snapshot is not None:
        try:
            refusal = host_gate(plan, args.host_snapshot, profile_id)
        except (ValueError, TypeError, KeyError, OSError):
            refusal = 'host_snapshot_unreadable'
        except Exception as error:  # ProfileError and similar: name only the class.
            refusal = type(error).__name__
        if refusal is not None:
            print(f'Local measurement refused before any journal: host preflight {refusal}.', file=sys.stderr)
            return 2
    try:
        if any(path.exists() for path in report_paths(args.journal)):
            raise ValueError('report_already_exists')
        budget = LocalAttemptBudget(plan, args.journal)
        observer = (fastflowlm_observer(plan, profile_id) if args.observer == 'fastflowlm'
                    else ollama_observer(plan))
        result = execute_plan(budget, observer)
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
