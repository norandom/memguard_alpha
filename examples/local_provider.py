"""Offline local-provider walkthrough; never connects to the configured endpoint.

PowerShell: uv run --no-sync python examples/local_provider.py --fixture
"""
import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from recall_guard.core.completion import ModelConfig
from recall_guard.core.local_evidence import LocalModelIdentity, LocalModelProvenance
from recall_guard.core.providers import OpenAICompatibleClient
from recall_guard.harness.local_evaluation import (
    LocalAttemptBudget,
    MeasurementLimits,
    load_plan,
)
from recall_guard.harness.local_execution import HostObservation, execute_plan


def write_inputs(directory):
    """Write synthetic inputs in a new directory; these are not a research corpus."""
    directory.mkdir(parents=True, exist_ok=False)
    config = ModelConfig('openai_chat', 'http://localhost:11434/v1/chat/completions',
        'gemma3:4b', parameters={'max_tokens': 32, 'temperature': 0, 'top_p': 1,
        'logprobs': True, 'top_logprobs': 5}, timeout_s=2, max_retries=0,
        local_identity=LocalModelIdentity())
    manifest = {'config': config.to_dict(), 'authentication': 'none',
                'requested_execution_mode': 'CPU'}
    corpus = {split: [{'item_id': f'{split}-{i}', 'prompt': f'{split}-{i}',
                      'input_tokens': 20} for i in range(count)]
              for split, count in [('is_memorized', 8), ('oos_control', 8), ('held_out', 1)]}
    hashes = {}
    for name, value in [('manifest', manifest), ('corpus', corpus)]:
        data = json.dumps(value, indent=2).encode('utf-8')
        (directory / f'{name}.json').write_bytes(data)
        hashes[f'{name}_sha256'] = hashlib.sha256(data).hexdigest()
    plan = {'schema_version': 1, **hashes, 'prompt_template': '{prompt}',
            'selected_features': ['mia', 'mcs'], 'ensemble_draws': 3,
            'min_valid': 4, 'seed': 0, 'min_auc': 0.6, 'ensemble_min_parsed': 1,
            'limits': asdict(MeasurementLimits(29, 2048, 100, 32, 3828, 120))}
    (directory / 'plan.json').write_text(json.dumps(plan, indent=2), encoding='utf-8')
    return load_plan(directory / 'plan.json', directory / 'manifest.json', directory / 'corpus.json')


def fixture_post(client, payload):
    """Substitute only HTTP: real provider parsing and scoring stay active.

    Entries follow generated text positions. Log probabilities use natural logs;
    alternatives are a partial set, never a normalized full vocabulary.
    """
    prompt = payload['messages'][-1]['content']
    base = -0.2 if prompt.startswith('is_memorized') else -5 if prompt.startswith('oos_control') else -1
    base += sum(map(ord, prompt)) % 10 / 1000
    content = 'Direction: 1\nConfidence: 0.8'
    entries = [{'token': char, 'logprob': base + i / 1000,
                'top_logprobs': [{'token': 'a', 'logprob': base},
                                {'token': 'b', 'logprob': base - 1}]}
               for i, char in enumerate(content)]
    body = {'choices': [{'message': {'content': content},
                        'logprobs': {'content': entries}, 'finish_reason': 'stop'}]}
    return SimpleNamespace(status_code=200, headers={}, json=lambda: body)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--dry-run', action='store_true', help='Validate synthetic plan (default)')
    mode.add_argument('--fixture', action='store_true', help='Run real scorer against fixed HTTP fixture')
    mode.add_argument('--write-inputs', type=Path, help='Write synthetic plan/manifest/corpus to NEW directory')
    args = parser.parse_args(argv)
    with TemporaryDirectory() as temporary:
        directory = args.write_inputs or Path(temporary) / 'inputs'
        plan = write_inputs(directory)
        output = {'status': 'dry_run', 'network_requests': 0, 'plan': plan.safe_export()}
        if args.fixture:
            budget = LocalAttemptBudget(plan, Path(temporary) / 'journal.json')
            def observer():
                return HostObservation(LocalModelProvenance(runtime='fixture',
                    runtime_version='1', artifact_digest='synthetic', backend='CPU',
                    context_length=2048), ('gemma3:4b',))
            with patch.object(OpenAICompatibleClient, '_post', fixture_post):
                result = execute_plan(budget, observer)
            output = {'status': result.status, 'network_requests': 0,
                      'fixture_attempts': budget.used,
                      'scored_held_out_rows': sum(row.arm in {'single', 'ensemble'}
                                                 and row.status == 'scored' for row in result.rows),
                      'evidence': 'synthetic mechanics only; no model efficacy measurement'}
        print(json.dumps(output, indent=2))
    return 0 if output['status'] in {'dry_run', 'completed'} else 2


if __name__ == '__main__':
    raise SystemExit(main())
