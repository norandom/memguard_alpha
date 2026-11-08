"""Frozen plans and durable accounting for serial local measurements.

No inference is started by loading a plan. Input token counts describe the fully
rendered prompts, measured with the selected model's tokenizer by the caller.
They are declarations, not tokenizer or host-context verification.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from threading import Event, Lock
from time import monotonic

from recall_guard.core.completion import ModelConfig
from recall_guard.core.providers import OpenAICompatibleClient
from recall_guard.core.request_control import BudgetExhaustedError, CallBudget


def _hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical(data) -> bytes:
    return json.dumps(data, sort_keys=True, ensure_ascii=False, allow_nan=False).encode('utf-8')


def _positive(value, name, *, integer=True):
    if isinstance(value, bool) or not isinstance(value, int if integer else (int, float)):
        raise ValueError(f'{name} must be positive and finite')
    if value <= 0 or not math.isfinite(value):
        raise ValueError(f'{name} must be positive and finite')


@dataclass(frozen=True)
class MeasurementLimits:
    max_requests: int
    context_tokens: int
    max_input_tokens: int
    max_output_tokens: int
    max_total_tokens: int
    max_elapsed_s: float

    def __post_init__(self):
        for name, value in asdict(self).items():
            _positive(value, name, integer=name != 'max_elapsed_s')
        if self.max_input_tokens + self.max_output_tokens > self.context_tokens:
            raise ValueError('input_and_output_exceed_context')


@dataclass(frozen=True)
class MeasurementItem:
    """Stable item key, split, rendered private prompt and declared token count.

    ``label`` (1 memorized, 0 out-of-sample, None unlabeled) lets a report score
    labeled held-out items; it never changes requests. Unlabeled plans load as before.
    """
    item_id: str
    split: str
    prompt: str
    input_tokens: int
    label: int | None = None


@dataclass(frozen=True)
class LocalMeasurementPlan:
    config: ModelConfig
    authentication: str
    requested_execution_mode: str
    manifest_sha256: str
    corpus_sha256: str
    prompt_template: str
    selected_features: tuple[str, ...]
    ensemble_draws: int
    limits: MeasurementLimits
    items: tuple[MeasurementItem, ...]
    seed: int = 0
    min_valid: int = 50
    min_auc: float = 0.6
    ensemble_min_parsed: int = 1

    def __post_init__(self):
        object.__setattr__(self, 'items', tuple(self.items))
        object.__setattr__(self, 'selected_features', tuple(self.selected_features))
        config, limits = self.config, self.limits
        if not isinstance(config, ModelConfig) or not isinstance(limits, MeasurementLimits):
            raise ValueError('typed_config_and_limits_required')
        if config.local_identity is None:
            raise ValueError('explicit_local_identity_required_including_unknowns')
        if self.authentication not in {'none', 'environment'} or (
            (self.authentication == 'none') != (config.credential_ref is None)
        ):
            raise ValueError('explicit_authentication_choice_mismatch')
        if config.protocol != 'openai_chat' or config.readiness is not None:
            raise ValueError('local_measurement_requires_synchronous_openai_chat')
        if self.requested_execution_mode not in {'CPU', 'GPU', 'CPU/GPU offload', 'NPU', 'mixed NPU', 'unknown'}:
            raise ValueError('invalid_requested_execution_mode')
        for digest in (self.manifest_sha256, self.corpus_sha256):
            if not isinstance(digest, str) or not re.fullmatch('[0-9a-f]{64}', digest):
                raise ValueError('invalid_frozen_input_hash')
        if config.parameters.get('max_tokens') != limits.max_output_tokens or 'max_completion_tokens' in config.parameters:
            raise ValueError('explicit_output_token_bound_required')
        # The reservation covers one generated completion per physical request.
        # Candidate selection can generate unseen outputs, so it is unsupported.
        completion_count = config.parameters.get('n', 1)
        if (type(completion_count) is not int or completion_count != 1
                or any(name in config.parameters for name in (
                    'best_of', 'num_return_sequences', 'num_beams', 'beam_width', 'use_beam_search'))):
            raise ValueError('single_completion_required_no_hidden_candidates')
        if config.parameters.get('logprobs') is not True:
            raise ValueError('local_logprobs_required')
        if config.timeout_s > limits.max_elapsed_s:
            raise ValueError('request_timeout_exceeds_elapsed_bound')
        features = self.selected_features
        if not features or len(set(features)) != len(features) or any(f not in {'mia', 'mcs'} for f in features):
            raise ValueError('unsupported_selected_features')
        if config.parameters.get('top_logprobs', 0) < 1:
            raise ValueError('selected_features_require_alternatives')
        # Zero draws is a single-answer-only plan (measurement Phase A); the
        # ensemble arm is then neither planned, reserved nor executed.
        if type(self.ensemble_draws) is not int or self.ensemble_draws < 0:
            raise ValueError('ensemble_draws must be a nonnegative integer')
        _positive(self.min_valid, 'min_valid')
        _positive(self.ensemble_min_parsed, 'ensemble_min_parsed')
        if self.ensemble_draws and self.ensemble_min_parsed > self.ensemble_draws:
            raise ValueError('ensemble_min_parsed_exceeds_draws')
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError('invalid_calibration_seed')
        if isinstance(self.min_auc, bool) or not isinstance(self.min_auc, (int, float)) or not 0 <= self.min_auc <= 1:
            raise ValueError('invalid_min_auc')
        if not isinstance(self.prompt_template, str) or self.prompt_template.count('{prompt}') != 1:
            raise ValueError('template_requires_one_prompt_marker')
        seen = set()
        splits = set()
        for item in self.items:
            if not isinstance(item, MeasurementItem):
                raise ValueError('typed_measurement_items_required')
            if not isinstance(item.item_id, str) or not item.item_id or item.item_id in seen:
                raise ValueError('duplicate_or_invalid_item_key')
            if not isinstance(item.prompt, str) or not item.prompt:
                raise ValueError('empty_prompt')
            _positive(item.input_tokens, 'input_tokens')
            if item.input_tokens > limits.max_input_tokens:
                raise ValueError('prompt_exceeds_input_token_bound')
            if item.label is not None and (type(item.label) is not int or item.label not in (0, 1) or (
                    item.split != 'held_out' and item.label != (item.split == 'is_memorized'))):
                raise ValueError('invalid_item_label')
            seen.add(item.item_id)
            splits.add(item.split)
        if splits != {'is_memorized', 'oos_control', 'held_out'}:
            raise ValueError('explicit_development_and_held_out_splits_required')
        if self.physical_requests > limits.max_requests:
            raise ValueError('request_bound_excludes_planned_probes_or_retries')
        if limits.max_requests * (limits.max_input_tokens + limits.max_output_tokens) > limits.max_total_tokens:
            raise ValueError('total_token_bound_excludes_reserved_attempts')

    @property
    def logical_requests(self):
        # calibrate_client uses OOS for BOTH baseline and training.
        counts = {split: sum(item.split == split for item in self.items)
                  for split in ('is_memorized', 'oos_control', 'held_out')}
        return (1 + counts['is_memorized'] + 2 * counts['oos_control']
                + counts['held_out'] * (1 + self.ensemble_draws))

    @property
    def physical_requests(self):
        return self.logical_requests * (1 + self.config.max_retries)

    def safe_export(self):
        """Detached commitment: no prompts, arbitrary metadata or credentials."""
        return {'schema_version': 1, 'manifest_sha256': self.manifest_sha256,
                'corpus_sha256': self.corpus_sha256,
                'config_sha256': _hash(_canonical(self.config.to_dict())),
                'credential_reference_sha256': None if self.config.credential_ref is None else
                    _hash(self.config.credential_ref.encode('utf-8')),
                'authentication': self.authentication,
                'requested_execution_mode': self.requested_execution_mode,
                'prompt_template_sha256': _hash(self.prompt_template.encode('utf-8')),
                'selected_features': list(self.selected_features), 'ensemble_draws': self.ensemble_draws,
                'seed': self.seed, 'min_valid': self.min_valid, 'min_auc': self.min_auc,
                'ensemble_min_parsed': self.ensemble_min_parsed,
                'limits': asdict(self.limits), 'logical_requests': self.logical_requests,
                'physical_requests': self.physical_requests,
                # A label key appears only on labeled items, so unlabeled exports are unchanged.
                'items': [{'item_sha256': _hash(item.item_id.encode('utf-8')), 'split': item.split,
                           'prompt_sha256': _hash(item.prompt.encode('utf-8')), 'input_tokens': item.input_tokens,
                           **({} if item.label is None else {'label': item.label})}
                          for item in self.items]}


def load_plan(plan_path: Path, manifest_path: Path, corpus_path: Path) -> LocalMeasurementPlan:
    """Validate immutable inputs before creating a client or journal."""
    data = json.loads(plan_path.read_text(encoding='utf-8-sig'))
    manifest_bytes, corpus_bytes = manifest_path.read_bytes(), corpus_path.read_bytes()
    if data['schema_version'] != 1:
        raise ValueError('unsupported_plan_schema')
    if (data['manifest_sha256'], data['corpus_sha256']) != (_hash(manifest_bytes), _hash(corpus_bytes)):
        raise ValueError('frozen_input_hash_mismatch')
    manifest, corpus = json.loads(manifest_bytes), json.loads(corpus_bytes)
    config = ModelConfig.from_dict(manifest['config'])
    authentication = manifest['authentication']
    config = replace(config, credential_ref=manifest.get('credential_ref'))
    mode = manifest['requested_execution_mode']
    limits = MeasurementLimits(**data['limits'])
    features = tuple(data['selected_features'])
    template = data['prompt_template']
    if not isinstance(template, str) or template.count('{prompt}') != 1:
        raise ValueError('template_requires_one_prompt_marker')
    items, seen = [], set()
    if set(corpus) != {'is_memorized', 'oos_control', 'held_out'}:
        raise ValueError('explicit_development_and_held_out_splits_required')
    for split, rows in corpus.items():
        if not isinstance(rows, list) or not rows:
            raise ValueError('empty_corpus_split')
        for row in rows:
            item_id, prompt = row['item_id'], row['prompt']
            if not isinstance(item_id, str) or not item_id or item_id in seen:
                raise ValueError('duplicate_or_invalid_item_key')
            if not isinstance(prompt, str) or not prompt:
                raise ValueError('empty_prompt')
            _positive(row['input_tokens'], 'input_tokens')
            if row['input_tokens'] > limits.max_input_tokens:
                raise ValueError('prompt_exceeds_input_token_bound')
            seen.add(item_id)
            items.append(MeasurementItem(item_id, split, template.replace('{prompt}', prompt), row['input_tokens'],
                                         row.get('label')))
    plan = LocalMeasurementPlan(config, authentication, mode, _hash(manifest_bytes), _hash(corpus_bytes),
                                template, features, data['ensemble_draws'], limits, tuple(items),
                                **{name: data[name] for name in ('seed', 'min_valid', 'min_auc',
                                                               'ensemble_min_parsed') if name in data})
    return plan


def _persist(path: Path, data):
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w', encoding='utf-8') as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


class LocalAttemptBudget(CallBudget):
    """One-use journal sharing RequestControl's physical admission boundary.

    Every attempt reserves the full configured input/output token allowance.
    Existing journals are deliberately refused, including completed runs: replay
    or continuation needs explicit external reconciliation and a new run plan.
    A synchronous timeout cannot establish that host execution has stopped.
    """

    def __init__(self, plan: LocalMeasurementPlan, path: Path, *, cancellation=None, clock=monotonic):
        super().__init__(plan.limits.max_requests)
        self.plan, self.path = plan, path
        self.cancellation = Event() if cancellation is None else cancellation
        self._clock = clock
        self._started = clock()
        self._journal_lock = Lock()
        self._active = False
        self._stopped = False
        self._report_secrets = ()
        self._journal = {'plan': plan.safe_export(), 'attempts': [], 'status': 'open'}
        # Exclusive creation also protects independent processes from replay.
        with path.open('x', encoding='utf-8') as stream:
            json.dump(self._journal, stream, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())

    def client(self, *, environment=None):
        environment = os.environ if environment is None else environment
        reference = self.plan.config.credential_ref
        credential = None if reference is None else environment.get(reference.removeprefix('env:'))
        self._report_secrets = () if not credential else (credential,)
        return OpenAICompatibleClient(self.plan.config, self, environment=environment,
                                      allow_unauthenticated=self.plan.authentication == 'none')

    def redact_report_text(self, value):
        """Redact the same resolved credential registered for this run's client."""
        for secret in self._report_secrets:
            value = value.replace(secret, '[REDACTED]')
        return value

    @property
    def stopped(self) -> bool:
        """Whether cancellation or an uncertain/bounded outcome forbids more work."""
        with self._journal_lock:
            return self._stopped or self.cancellation.is_set()

    def safe_receipt(self) -> dict:
        """Detached journal snapshot; response bodies and diagnostics stay private."""
        with self._journal_lock:
            return json.loads(json.dumps(self._journal))

    def begin_execution(self):
        """Claim this fresh journal once; a partially used budget cannot replay."""
        with self._journal_lock:
            if self._journal['status'] != 'open' or self.used:
                raise ValueError('measurement_execution_already_started')
            self._journal['status'] = 'executing'
            _persist(self.path, self._journal)

    def finish_execution(self, status):
        """Persist a non-content terminal status; reports are a separate layer."""
        with self._journal_lock:
            self._journal['status'] = status
            self._stopped = True
            _persist(self.path, self._journal)

    def _admit(self, **kwargs):
        with self._journal_lock:
            if self.cancellation.is_set() or self._stopped:
                raise BudgetExhaustedError('local_measurement_cancelled_or_stopped')
            if self._active:
                raise BudgetExhaustedError('serial_local_request_already_active')
            if self._clock() - self._started + self.plan.config.timeout_s > self.plan.limits.max_elapsed_s:
                raise BudgetExhaustedError('local_elapsed_budget_exhausted')
            event = super()._admit(**kwargs)
            self._active = True
            self._journal['attempts'].append({'call_id': event.call_id, 'attempt': event.attempt,
                'stage_sha256': _hash(event.stage.encode('utf-8')), 'purpose': event.purpose,
                'status': 'unknown_outcome', 'latency_s': None, 'usage': None, 'cost': None,
                'http_status': None, 'reason': 'intent_persisted', 'host_execution_stopped': None})
            # A write failure blocks the POST; the in-memory run remains stopped.
            try:
                _persist(self.path, self._journal)
            except BaseException:
                self._stopped = True
                raise
            return event

    def _finish(self, event, **changes):
        with self._journal_lock:
            super()._finish(event, **changes)
            finished = self.events[event.call_id - 1]
            unknown = (finished.status_code in {408, 504} or (finished.status != 'succeeded'
                       and finished.status_code is None
                       and finished.code not in {'missing_token_evidence', 'invalid_response'}))
            exceeded = (self._clock() - self._started > self.plan.limits.max_elapsed_s
                        or finished.latency_s > self.plan.config.timeout_s)
            self._stopped = unknown or exceeded
            usage = None if finished.usage is None else {key: finished.usage[key] for key in
                ('prompt_tokens', 'completion_tokens', 'total_tokens') if key in finished.usage}
            token_exceeded = usage and (
                usage.get('prompt_tokens', 0) > self.plan.limits.max_input_tokens
                or usage.get('completion_tokens', 0) > self.plan.limits.max_output_tokens
                or usage.get('total_tokens', 0) > self.plan.limits.max_input_tokens + self.plan.limits.max_output_tokens)
            if token_exceeded:
                self._stopped = True
            known_codes = {'model_not_found', 'unsupported_parameter', 'invalid_request_error',
                           'missing_token_evidence', 'invalid_response', 'timeout', 'connection_error',
                           'transport_error'}
            record = self._journal['attempts'][event.call_id - 1]
            record.update(status='unknown_outcome' if unknown else finished.status,
                          latency_s=finished.latency_s, usage=usage, http_status=finished.status_code,
                          code=finished.code if finished.code in known_codes else None,
                          rejected_parameter=finished.parameter if finished.parameter in {
                              'max_tokens', 'temperature', 'top_p', 'logprobs', 'top_logprobs'} else None,
                          reason='token_bound_exceeded' if token_exceeded else 'elapsed_bound_exceeded' if exceeded else
                          'unknown_outcome' if unknown else 'completed' if finished.status == 'succeeded'
                          else 'provider_rejected_request')
            self._active = False
            try:
                _persist(self.path, self._journal)
            except BaseException:
                self._stopped = True
                raise


