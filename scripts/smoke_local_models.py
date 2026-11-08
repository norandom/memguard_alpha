"""Bounded serial Ollama smoke. Default: metadata only, no inference or unloading.

Execute with --execute --output NEW.json --host-observation host-observation.json.
No downloads, retries, calibration, raw generated text or host configuration changes.
Token rows are generated positions; log probabilities are natural logs, partial top-k.
"""
from __future__ import annotations

import argparse
import math
import subprocess
import sys
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from time import monotonic
from urllib.parse import urlsplit

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from recall_guard.core.local_evidence import LocalEvidenceError, validate_local_evidence
from recall_guard.core.profiles import (
    HostMemorySnapshot,
    PreflightError,
    ProfileError,
    load_profile,
    preflight_host,
    running_in_container,
)
from recall_guard.harness.request_failures import print_failure
from scripts.probe_local_model import (
    PROFILE_MODELS,
    PROMPT,
    SETTINGS,
    CallBudget,
    ProbeConfig,
    RequestError,
    connect_probe,
    observe_host,
    persist,
    preflight,
    read_json,
    safe_events,
)

MODELS = ('phi4-mini:3.8b', 'llama3.1:8b', 'llama3.2:3b', 'gemma3:4b',
          'llava-llama3:latest', 'gpt-oss:20b')
GIB = 1024 ** 3
HOST_ERRORS = (OSError, requests.RequestException, ValueError, KeyError, TypeError)
DEFAULT_ENDPOINT = 'http://127.0.0.1:11434/v1/chat/completions'


def profile_selection(profile_ids, endpoint: str | None) -> tuple[list[str], str]:
    """Map curated local profile IDs to model tags; an explicit endpoint wins."""
    models, endpoints = [], set()
    for profile_id in profile_ids:
        if profile_id not in PROFILE_MODELS:
            raise ProfileError('smoke_profiles_must_be_curated_local_ollama_profiles')
        config = load_profile(profile_id).model_config
        models.append(config.model)
        endpoints.add(config.endpoint)
    if endpoint is None:
        if len(endpoints) > 1:
            raise ProfileError('selected_profiles_disagree_on_endpoint_pass_endpoint')
        endpoint = endpoints.pop() if endpoints else DEFAULT_ENDPOINT
    return models, endpoint


def profile_preflight(profile_ids, endpoint: str, *, check=preflight_host,
                      host_snapshot: HostMemorySnapshot | None = None) -> None:
    """Run the shared host preflight for each selected profile at the chosen endpoint.

    ``host_snapshot`` is free RAM measured on the model host (needed inside a
    container). Raises :class:`PreflightError` on the first failure; nothing is loaded.
    """
    extra = {} if host_snapshot is None else {'host_snapshot': host_snapshot}
    for profile_id in profile_ids:
        profile = load_profile(profile_id)
        profile = replace(profile, model_config=replace(profile.model_config, endpoint=endpoint))
        report = check(profile, environment={}, **extra)
        if not report.passed:
            raise PreflightError(report)


def available_memory_bytes() -> int | None:
    """Actual local available RAM, never installed RAM or dedicated VRAM."""
    try:
        if sys.platform == 'win32':
            result = subprocess.run(
                ['powershell.exe', '-NoProfile', '-NonInteractive', '-Command',
                 '(Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory'],
                capture_output=True, text=True, timeout=10, check=True)
            return int(result.stdout.strip()) * 1024
        for line in Path('/proc/meminfo').read_text().splitlines():
            if line.startswith('MemAvailable:'):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    return None


def memory_gate(endpoint: str, threshold: int, reader=None, *, host_snapshot: HostMemorySnapshot | None = None,
                in_container: bool | None = None) -> dict:
    # A caller-supplied reader explicitly supplies verified SERVER memory; the CLI
    # supplies one only from a fresh --host-snapshot taken on the model host.
    # Remote endpoints, and any endpoint seen from inside a container (C20), never use client RAM.
    if host_snapshot is not None:
        problem = host_snapshot.freshness_problem()
        sample = None if problem else host_snapshot.available_bytes
        return {'available_bytes': sample, 'threshold_bytes': threshold, 'sampled_at': host_snapshot.observed_at,
                'source': host_snapshot.source if problem is None else problem[0],
                'policy': 'conservative_available_ram_admission_not_proven_minimum',
                'admitted': sample is not None and sample >= threshold}
    container = running_in_container() if in_container is None else in_container
    local = urlsplit(endpoint).hostname in {'localhost', '127.0.0.1', '::1'} and not container
    sample = None
    if local or reader is not None:
        try:
            sample = (reader or available_memory_bytes)()
        except (OSError, ValueError):
            pass
    if type(sample) is not int or sample < 0:
        sample = None
    return {'available_bytes': sample, 'threshold_bytes': threshold,
            'sampled_at': datetime.now(UTC).isoformat(),
            'source': ('verified_host_callback' if reader else 'local_host_available_ram' if local
                       else 'unknown_container_host' if container else 'unknown_remote_host'),
            'policy': 'conservative_available_ram_admission_not_proven_minimum',
            'admitted': sample is not None and sample >= threshold}


def unload_model(config: ProbeConfig) -> None:
    """Ollama control POST, deliberately separate from inference accounting."""
    parsed = urlsplit(config.endpoint)
    with requests.Session() as session:
        session.trust_env = False
        result = session.post(f'{parsed.scheme}://{parsed.netloc}/api/generate',
                              json={'model': config.model, 'keep_alive': 0},
                              timeout=30, allow_redirects=False)
        if result.status_code != 200:
            raise ValueError('unload_failed')


def probe(config: ProbeConfig, row: dict, journal: dict, output: Path) -> None:
    budget = CallBudget(1)
    client = connect_probe(config, budget)
    row['status'] = 'intent'
    persist(output, journal)
    started = monotonic()
    try:
        response = client.generate_for_stage(PROMPT, stage='local-smoke')
        row['generated_token_count'] = len(response.token_evidence or ())
        row['tokens'] = [{'position': i, 'logprob': token.logprob,
                          'alternative_count': len(token.top_logprobs),
                          'alternative_logprobs': [alt.logprob for alt in token.top_logprobs]}
                         for i, token in enumerate(response.token_evidence or ())]
        validate_local_evidence(response, required_alternatives=5)
        row.update(evidence_status='valid', evidence_reason='aligned_token_evidence_observed')
    except LocalEvidenceError as error:
        row.update(evidence_status='ineligible', evidence_reason=str(error))
    except RequestError as error:
        known = {'timeout', 'connection_error', 'transport_error', 'invalid_response',
                 'missing_token_evidence', 'unsupported_parameter', 'invalid_request_error', 'model_not_found'}
        code = error.code if error.code in known else 'provider_rejected_request'
        row.update(evidence_status='ineligible' if code in {'invalid_response', 'missing_token_evidence'}
                   else 'unavailable', evidence_reason=code, http_status=error.status_code,
                   unknown_outcome=code in {'timeout', 'connection_error', 'transport_error'})
    except KeyboardInterrupt:
        row.update(evidence_status='unavailable', evidence_reason='cancelled', unknown_outcome=True)
    except Exception:
        # Unexpected adapter failure has no safely established transport outcome.
        row.update(evidence_status='unavailable', evidence_reason='unknown_outcome', unknown_outcome=True)
    finally:
        row.update(elapsed_s=monotonic() - started, physical_attempts=budget.used, events=safe_events(budget))
        if row['elapsed_s'] > 300:
            row.update(unknown_outcome=True, reason='attempt_elapsed_bound_exceeded')
        row['status'] = 'unavailable' if row['unknown_outcome'] else 'ineligible'
        persist(output, journal)


def run_smoke(endpoint: str, host: dict, output: Path | None = None, *, models=MODELS,
              execute: bool = False, minimum_available_gib: float = 16, memory_reader=None,
              host_snapshot: HostMemorySnapshot | None = None) -> dict:
    """One invocation, one new receipt, at most one inference per selected model.

    ``memory_reader`` is an optional verified server-memory callback (bytes);
    ``host_snapshot`` a fresh snapshot taken on the model host. Only loopback
    endpoints outside a container use the built-in reader. No receipt may be replayed.
    """
    models = tuple(models)
    if not models or len(set(models)) != len(models) or any(m not in MODELS for m in models):
        raise ValueError('select_unique_documented_models_only')
    if not math.isfinite(minimum_available_gib) or minimum_available_gib <= 0:
        raise ValueError('memory_threshold_must_be_positive')
    threshold = math.ceil(minimum_available_gib * GIB)
    configs = [ProbeConfig(m, m, endpoint, '', '') for m in models]
    for config in configs:
        config.client_config()
    if urlsplit(endpoint).path != '/v1/chat/completions':
        raise ValueError('explicit_full_chat_endpoint_required')
    journal = {'schema_version': 1, 'status': 'planning', 'stop_reason': None,
               'plan': {'models': list(models), 'endpoint': endpoint, 'settings': dict(SETTINGS),
                        'context_tokens': 2048, 'inference_limit': len(models), 'attempts_per_model': 1,
                        'max_seconds_per_request': 300, 'automatic_retries': 0,
                        'concurrent_requests': 1, 'max_loaded_models': 1,
                        'gpt_oss_minimum_available_bytes': threshold},
               'metadata_operations': [], 'control_operations': [], 'attempts': []}
    if execute:
        if output is None:
            raise ValueError('execution_requires_new_output')
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open('x', encoding='utf-8') as stream:
            stream.write('{"status":"reserved_do_not_replay"}\n')

    def save():
        if execute:
            persist(output, journal)

    def observe(config, phase):
        operation = {'model': config.model, 'phase': phase,
                     'routes': ['/api/version', '/api/tags', '/api/ps'], 'status': 'intent'}
        journal['metadata_operations'].append(operation)
        save()
        try:
            result = observe_host(config)
        except HOST_ERRORS:
            operation['status'] = 'failed'
            raise
        operation['status'] = 'completed'
        return result

    def unload(config):
        operation = {'model': config.model, 'route': '/api/generate', 'keep_alive': 0, 'status': 'intent'}
        journal['control_operations'].append(operation)
        save()
        try:
            unload_model(config)
            empty = not observe(config, 'verify_unload')['loaded_models']
        except HOST_ERRORS:
            empty = False
        operation['status'] = 'verified_empty' if empty else 'unverified'
        if not empty:
            journal['stop_reason'] = 'unload_not_verified'
        save()
        return empty

    try:
        # Freeze ALL selected installed identities before the first inference.
        frozen = [(config, observe(config, 'freeze')) for config in configs]
        journal['plan']['installed'] = {config.model: obs for config, obs in frozen}
        save()
        if not execute:
            journal['status'] = 'dry_run'
            return journal
        for initial, frozen_observation in frozen:
            row = {'model': initial.model, 'digest': frozen_observation['digest'], 'status': 'skipped',
                   'reason': None, 'evidence_status': 'not_tested', 'evidence_reason': None,
                   'environment_status': 'unverified', 'physical_attempts': 0, 'unknown_outcome': False,
                   'generated_token_count': None, 'tokens': [], 'score_semantics': 'unknown',
                   'alternative_distribution': 'partial_top_k', 'memory_gate': None,
                   'load_time_s': None, 'generation_time_s': None, 'dedicated_gpu_memory_bytes': None,
                   'shared_gpu_memory_bytes': None, 'system_ram_usage_bytes': None,
                   'host_execution_stopped': None}
            journal['attempts'].append(row)
            config = ProbeConfig(initial.model, initial.model, endpoint, row['digest'], '')
            before = observe(config, 'before_admission')
            row['observed_before'] = before
            if before['loaded_models']:
                # Never unload somebody else's work. A selected model already loaded
                # is unloaded only when it is the sole currently selected candidate.
                if before['loaded_models'] != [config.model]:
                    journal['stop_reason'] = 'other_model_loaded'
                    break
                if not unload(config):
                    break
                before = observe(config, 'after_initial_unload')
            if not row['digest']:
                row['reason'] = 'model_missing_or_digest_unknown'
                continue
            reason = preflight(config, host, before)
            if before['runtime_version'] != frozen_observation['runtime_version'] or not before['runtime_version']:
                reason = 'runtime_identity_changed_or_unknown'
            if reason:
                row.update(reason=reason, environment_status='mismatched')
                continue
            if config.model == 'gpt-oss:20b':
                row['memory_gate'] = memory_gate(endpoint, threshold, memory_reader, host_snapshot=host_snapshot)
                if not row['memory_gate']['admitted']:
                    row['reason'] = 'available_host_memory_unknown_or_below_policy'
                    continue
            probe(config, row, journal, output)
            if row['unknown_outcome']:
                journal['stop_reason'] = 'unknown_inference_outcome_no_retry'
                break
            try:
                after = observe(config, 'after_inference')
                row['observed_after'] = after
                matched = (after['digest'] == config.digest
                           and after['runtime_version'] == frozen_observation['runtime_version']
                           and after['effective_context_tokens'] == 2048
                           and after['loaded_models'] == [config.model])
                row['environment_status'] = 'matched' if matched else 'mismatched'
            except HOST_ERRORS:
                row['environment_status'] = 'unverified'
                journal['stop_reason'] = 'post_inference_metadata_unavailable'
                break
            row['status'] = ('eligible' if row['evidence_status'] == 'valid' and matched
                             else 'ineligible')
            # Do not send even a control request if an unrelated model appeared.
            if after['loaded_models'] not in ([], [config.model]):
                journal['stop_reason'] = 'other_model_loaded'
                break
            if not unload(config):
                row.update(environment_status='unverified', status='ineligible')
                break
    except KeyboardInterrupt:
        journal['stop_reason'] = 'cancelled'
    except HOST_ERRORS:
        journal['stop_reason'] = 'host_metadata_unavailable'
    finally:
        if execute:
            journal['status'] = 'stopped' if journal['stop_reason'] else 'completed'
            save()
    return journal


def main(argv=None) -> int:
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--endpoint', default=None, help=f'Full chat-completions URL (default {DEFAULT_ENDPOINT}).')
    parser.add_argument('--host-observation', type=Path, default=Path('runs/local-provider/host-observation.json'))
    parser.add_argument('--model', action='append', choices=MODELS)
    parser.add_argument('--profile', action='append', choices=sorted(PROFILE_MODELS),
                        help='Curated local provider profile ID; may be combined with --model.')
    parser.add_argument('--minimum-available-gib', type=float, default=16)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--host-snapshot', type=Path, default=None,
                        help='Host memory snapshot JSON written on the model host by scripts/host_memory_snapshot.ps1 '
                             '(inside the container: /outputs/host-memory.json).')
    args = parser.parse_args(argv)
    try:
        snapshot = None if args.host_snapshot is None else HostMemorySnapshot.from_file(args.host_snapshot)
        profiled, endpoint = profile_selection(args.profile or (), args.endpoint)
        models = list(dict.fromkeys([*profiled, *(args.model or ())])) or MODELS
        profile_preflight(args.profile or (), endpoint, host_snapshot=snapshot)
        host = read_json(args.host_observation) if args.execute else {}
        result = run_smoke(endpoint, host, args.output, models=models, execute=args.execute,
                           minimum_available_gib=args.minimum_available_gib, host_snapshot=snapshot)
    except PreflightError as error:
        print_failure(error, stage='host preflight')
        return 2
    except ProfileError as error:
        print(f'Smoke refused: {error}', file=sys.stderr)
        return 2
    except (OSError, ValueError):
        print('Smoke refused: invalid configuration, unavailable host file or existing receipt.', file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0 if result['status'] in {'dry_run', 'completed'} else 1


if __name__ == '__main__':
    raise SystemExit(main())
