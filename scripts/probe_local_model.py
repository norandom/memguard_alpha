"""Frozen, one-at-a-time Ollama feasibility probes through OpenAICompatibleClient.

PowerShell: uv run --no-sync python scripts/probe_local_model.py --help
No model hosting, downloads, private prompt input, retries or calibration occur here.
Use the same journal for the entire approved feasibility plan. An interrupted intent
is never replayed: its physical outcome must be reconciled externally first.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from time import monotonic
from urllib.parse import urlsplit

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from recall_guard.core.completion import ModelConfig, ModelResponse
from recall_guard.core.local_evidence import validate_local_evidence
from recall_guard.core.profiles import (
    HostingMode,
    HostPreflightReport,
    PreflightError,
    ProviderProfile,
    connect,
    load_profile,
    observe_ollama_host,
    preflight_host,
)
from recall_guard.core.providers import OpenAICompatibleClient  # noqa: F401 - re-exported to smoke/tests
from recall_guard.core.request_control import CallBudget, RequestError
from recall_guard.harness.request_failures import print_failure, retained_failure

PROMPT = 'Reply with the word OK.'
MODEL_IDS = {'ollama-gemma3-4b': 'gemma3:4b', 'ollama-gpt-oss-20b': 'gpt-oss:20b'}
#: Curated local profiles (recall_guard/core/provider_profiles) by model tag; probe settings stay frozen.
PROFILE_MODELS = MODEL_IDS | {'ollama-phi4-mini': 'phi4-mini:3.8b', 'ollama-llama3.2-3b': 'llama3.2:3b',
                              'ollama-llama3.1-8b': 'llama3.1:8b'}
SETTINGS = {'temperature': 0, 'top_p': 1, 'top_logprobs': 5, 'logprobs': True, 'max_tokens': 32}
BOUNDS = {'total_generation_attempts': 6, 'attempts_per_configuration': 2,
          'context_tokens': 2048, 'max_output_tokens': 32, 'automatic_retries': 0,
          'max_seconds_per_attempt': 300, 'concurrent_requests': 1, 'loaded_test_models': 1,
          'blocked_candidate_budget_transfer': False}


@dataclass(frozen=True)
class ProbeConfig:
    candidate_id: str
    model: str
    endpoint: str
    digest: str
    manifest_hash: str

    def client_config(self) -> ModelConfig:
        return ModelConfig(protocol='openai_chat', model=self.model, endpoint=self.endpoint,
                           provider='ollama', parameters=SETTINGS, timeout_s=300, max_retries=0)

    def profile(self) -> ProviderProfile:
        """The curated local profile for this model, carrying the frozen probe config."""
        profile_id = next((pid for pid, tag in PROFILE_MODELS.items() if tag == self.model), None)
        if profile_id is None:
            return ProviderProfile('ollama-adhoc', f'Ollama {self.model}', HostingMode.LOCAL,
                                   self.client_config(), allow_unauthenticated=True)
        return replace(load_profile(profile_id), model_config=self.client_config())


def connect_probe(config: ProbeConfig, budget: CallBudget):
    """Build the unauthenticated local client through the single profile factory.

    The host gate already ran (:func:`host_preflight` plus the frozen-plan
    :func:`preflight`), so the factory does not repeat it.
    """
    return connect(config.profile(), budget, environment={}, enforce_preflight=False)


def host_preflight(config: ProbeConfig) -> HostPreflightReport:
    """Free-RAM gate and read-only Ollama metadata before any journal entry or load."""
    return preflight_host(config.profile(), environment={})


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8-sig'))


def load_config(manifest_path: Path, candidate_id: str) -> ProbeConfig:
    manifest = read_json(manifest_path)
    if candidate_id not in MODEL_IDS:
        raise ValueError('candidate_not_in_frozen_ollama_plan')
    if any(manifest['budget'].get(key) != value for key, value in BOUNDS.items()):
        raise ValueError('manifest_bounds_changed')
    entries = [item for item in manifest['candidates'] if item['id'] == candidate_id]
    if len(entries) != 1:
        raise ValueError('candidate_missing_or_duplicated')
    candidate = entries[0]
    if (candidate['model'] != MODEL_IDS[candidate_id] or candidate['authentication'] != 'none'
            or candidate['generation_settings'] != SETTINGS or candidate['requested_context_tokens'] != 2048):
        raise ValueError('candidate_configuration_changed')
    config = ProbeConfig(candidate_id, candidate['model'], candidate['endpoint'], candidate['manifest_sha256'],
                         hashlib.sha256(manifest_path.read_bytes()).hexdigest())
    config.client_config()  # Credential-free URL validation before exporting any value.
    if urlsplit(config.endpoint).path != '/v1/chat/completions':
        raise ValueError('explicit_full_chat_endpoint_required')
    return config


def validate_evidence(response: ModelResponse, required_alternatives: int = 5) -> None:
    """Retain the frozen probe's five-alternative gate and evidence reason codes."""
    validate_local_evidence(response, required_alternatives=required_alternatives)


def observe_host(config: ProbeConfig) -> dict:
    """Read-only metadata, no generation, loading, unloading or inference transport.

    Delegates to the shared core observer (also used by host preflight).
    """
    observed = observe_ollama_host(config.endpoint, config.model)
    observed.pop('model_installed')
    return observed


def preflight(config: ProbeConfig, host: dict, observed: dict) -> str | None:
    if (host.get('endpoint') != config.endpoint or host.get('configured_context_tokens') != 2048
            or host.get('max_loaded_models') != 1 or host.get('num_parallel') != 1
            or host.get('context_source') != 'process-start-environment'
            or not isinstance(host.get('process_id'), int)):
        return 'host_configuration_not_established'
    if observed['digest'] != config.digest:
        return 'model_digest_missing_or_mismatched'
    if observed['effective_context_tokens'] not in (None, 2048):
        return 'context_bound_not_established'
    if observed['loaded_models'] not in ([], [config.model]):
        return 'other_model_loaded'
    return None


def persist(path: Path, data: dict) -> None:
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w', encoding='utf-8') as stream:
        json.dump(data, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def new_receipt(config: ProbeConfig) -> dict:
    return {'candidate_id': config.candidate_id, 'model': config.model, 'endpoint': config.endpoint,
            'authentication': 'none', 'status': 'unverified', 'reason': None,
            'physical_attempts': 0, 'unknown_outcome': False, 'events': [], 'timestamp': datetime.now(UTC).isoformat(),
            'configured_context_tokens': 2048, 'elapsed_s': None, 'host_execution_stopped': None,
            'load_time_s': None, 'generation_time_s': None, 'dedicated_gpu_memory_bytes': None,
            'shared_gpu_memory_bytes': None, 'system_ram_usage_bytes': None,
            'score_semantics': 'unknown', 'alternative_distribution': 'partial_top_k',
            'tokenizer_identity': None, 'observed_before': None, 'observed_after': None}


def safe_events(budget: CallBudget) -> list[dict]:
    # Do not export provider-controlled messages, response bodies, request IDs,
    # parameters or arbitrary usage keys: they may echo secrets/private inputs.
    fields = ('call_id', 'attempt', 'started_at', 'timestamp', 'status', 'latency_s', 'status_code')
    return [{key: event.to_dict()[key] for key in fields} | {
        'usage': None if event.usage is None else {k: event.usage[k] for k in
            ('prompt_tokens', 'completion_tokens', 'total_tokens') if k in event.usage}}
        for event in budget.events]


def execute(config: ProbeConfig, receipt: dict, journal: dict, path: Path) -> None:
    budget = CallBudget(1)
    client = connect_probe(config, budget)
    receipt['status'] = 'intent'
    persist(path, journal)  # Durable intent BEFORE RequestControl admits its physical POST.
    started = monotonic()
    try:
        response = client.generate_for_stage(PROMPT, stage='local-feasibility')
        validate_evidence(response)
        receipt.update(status='eligible', reason='aligned_token_evidence_observed')
    except ValueError as error:
        receipt.update(status='ineligible', reason=str(error))  # Only probe-owned evidence codes.
    except RequestError as error:
        known = {'timeout', 'connection_error', 'transport_error', 'invalid_response',
                 'missing_token_evidence', 'unsupported_parameter', 'invalid_request_error', 'model_not_found'}
        code = error.code if error.code in known else 'provider_rejected_request'
        receipt.update(status='ineligible' if code in {'invalid_response', 'missing_token_evidence'} else 'unavailable',
                       reason=code, http_status=error.status_code,
                       rejected_parameter=error.parameter if error.parameter in SETTINGS else None)
        # R2.7: keep the bounded, sanitized provider body of an HTTP failure (for
        # example an Ollama 500 on load); transport failures retain nothing.
        receipt['provider_failure'] = retained_failure(error, secrets=(PROMPT,))
    except KeyboardInterrupt:
        receipt.update(status='cancelled', reason='unknown_outcome')
    finally:
        receipt.update(elapsed_s=monotonic() - started, physical_attempts=budget.used, events=safe_events(budget))
        if receipt['status'] == 'intent':
            receipt.update(status='unavailable', reason='unknown_outcome')
        if receipt['elapsed_s'] > 300:
            receipt.update(status='unavailable', reason='attempt_elapsed_bound_exceeded')
        receipt['unknown_outcome'] = receipt['reason'] in {
            'timeout', 'connection_error', 'transport_error', 'unknown_outcome', 'attempt_elapsed_bound_exceeded'}
        persist(path, journal)
    try:
        receipt['observed_after'] = observe_host(config)
        after = receipt['observed_after']
        if (after['digest'] != config.digest or after['effective_context_tokens'] not in (None, 2048)
                or after['loaded_models'] not in ([], [config.model])):
            receipt.update(status='ineligible', reason='post_probe_identity_or_context_mismatch')
    except (requests.RequestException, ValueError, KeyError, TypeError):
        receipt['post_observation_reason'] = 'host_metadata_unavailable'
        if receipt['status'] == 'eligible':
            receipt.update(status='unavailable', reason='post_probe_identity_unverified')
    persist(path, journal)


def run_probe(manifest_path: Path, candidate_id: str, host_path: Path, journal_path: Path,
              *, dry_run: bool = False) -> dict:
    """One explicitly requested probe. A separate invocation is never an automatic retry."""
    config = load_config(manifest_path, candidate_id)
    receipt = new_receipt(config)
    if dry_run:
        return receipt | {'status': 'dry_run', 'bounds': BOUNDS, 'ollama_attempt_limit': 4,
                          'reserved_blocked_npu_attempts': 2}
    report = host_preflight(config)  # Zero inference; no journal entry, no load.
    if not report.passed:
        raise PreflightError(report)
    journal_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = journal_path.with_suffix(journal_path.suffix + '.lock')
    with lock_path.open('x', encoding='utf-8') as lock:
        lock.write('A running or interrupted probe owns this journal. Never silently replay.\n')
    try:
        plan = {'manifest_sha256': config.manifest_hash, 'prompt_sha256': hashlib.sha256(PROMPT.encode()).hexdigest(),
                'bounds': BOUNDS, 'ollama_attempt_limit': 4, 'reserved_blocked_npu_attempts': 2,
                'models': MODEL_IDS, 'settings': SETTINGS}
        journal = read_json(journal_path) if journal_path.exists() else {'schema_version': 1, 'plan': plan, 'attempts': []}
        if journal['plan'] != plan:
            raise ValueError('frozen_plan_changed')
        previous = journal['attempts']
        uncertain = {'timeout', 'connection_error', 'transport_error', 'unknown_outcome', 'attempt_elapsed_bound_exceeded'}
        if any(r['status'] == 'intent' or r.get('unknown_outcome') or r['reason'] in uncertain for r in previous):
            raise ValueError('unknown_outcome_requires_external_reconciliation_no_replay')
        if sum(r['physical_attempts'] for r in previous if r['candidate_id'] == candidate_id) >= 2:
            raise ValueError('candidate_budget_exhausted')
        if sum(r['physical_attempts'] for r in previous) >= 4:
            raise ValueError('ollama_budget_exhausted_npu_reservation_not_transferable')
        if sum(r['elapsed_s'] or 0 for r in previous) >= 1200:
            raise ValueError('elapsed_budget_exhausted')
        previous.append(receipt)
        persist(journal_path, journal)
        try:
            host = read_json(host_path)
            observed = observe_host(config)
            receipt['observed_before'] = observed
            receipt['host_process_id'] = host.get('process_id')
            receipt['context_source'] = 'process-start-environment'
            reason = preflight(config, host, observed)
        except (OSError, requests.RequestException, ValueError, KeyError, TypeError):
            reason = 'host_metadata_unavailable'
        if reason:
            receipt.update(status='unavailable', reason=reason)
            persist(journal_path, journal)
        else:
            execute(config, receipt, journal, journal_path)
        return receipt
    finally:
        lock_path.unlink()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=Path('.kiro/specs/local-ai-provider/candidate-manifest.json'))
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument('--candidate', choices=MODEL_IDS)
    selection.add_argument('--profile', choices=MODEL_IDS,
                           help='Curated provider profile ID (recall_guard/core/provider_profiles); same IDs as --candidate.')
    parser.add_argument('--host-observation', type=Path, default=Path('runs/local-provider/host-observation.json'))
    parser.add_argument('--journal', type=Path, default=Path('runs/local-provider/ollama-probe-journal.json'))
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    candidate = args.profile or args.candidate
    try:
        result = run_probe(args.manifest, candidate, args.host_observation, args.journal, dry_run=args.dry_run)
    except PreflightError as error:
        print_failure(error, profile_id=candidate, stage='host preflight')
        return 2
    except ValueError as error:
        print(f'Probe refused: {error}', file=sys.stderr)
        return 2
    except FileExistsError:
        print('Probe refused: journal_lock_requires_review', file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result['status'] in {'eligible', 'dry_run'} else 1


if __name__ == '__main__':
    raise SystemExit(main())

