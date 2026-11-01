"""Immutable requested/selected workflow contracts, without executable clients.

Role configurations preserve caller credential *references* in memory; exports
omit them. A rewrite policy identity binds the caller's runtime policy, never
the policy's clients. Preparation/execution are separate consumers of these
contracts; constructing or loading a plan performs no provider work.
"""

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from recall_guard.core.completion import FeatureChoices, ModelConfig, _integer
from recall_guard.core.consensus import Tail
from recall_guard.core.ensemble import EnsembleSpec, LocationMode, MultimodalAction, ReferenceMode
from recall_guard.core.readiness_types import _fields, _text, _timestamp

_FEATURE_FIELDS = {"mia_scoring", "ensemble_draws", "rewriting"}
_ENSEMBLE_FIELDS = {
    "draws", "max_workers", "min_parsed", "grid", "confidence", "tail", "agreement_target", "location_mode",
    "trim", "multimodal_action", "mass_min", "trough_steps", "density_ratio", "min_cluster_draws",
    "min_cluster_density", "max_total_requests", "max_transport_failure_ratio", "retain_draws",
    "reference_mode", "max_tokens", "temperature",
}
_ENSEMBLE_ENUMS = {"tail": Tail, "location_mode": LocationMode, "multimodal_action": MultimodalAction,
                   "reference_mode": ReferenceMode}
_ENSEMBLE_INTEGERS = {"draws", "max_workers", "min_parsed", "trough_steps", "min_cluster_draws",
                      "max_total_requests", "max_tokens"}


def canonical_json(value):
    """JSON identity preserves bool/number, arrays and absent/null distinctions."""
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def contract_hash(value):
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _features(value):
    return {"mia_scoring": value.mia_scoring, "ensemble_draws": value.ensemble_draws, "rewriting": value.rewriting}


def _ensemble_dict(value):
    if value is None:
        return None
    return {name: getattr(value, name).value if name in _ENSEMBLE_ENUMS else getattr(value, name)
            for name in sorted(_ENSEMBLE_FIELDS)}


def _ensemble_from_dict(data):
    if data is None:
        return None
    values = _fields(data, _ENSEMBLE_FIELDS)
    for name, value in values.items():
        if name in _ENSEMBLE_ENUMS:
            values[name] = _ENSEMBLE_ENUMS[name](value)
        elif name == "retain_draws":
            if type(value) is not bool:
                raise ValueError("retain_draws must be boolean")
        elif value is not None and name in _ENSEMBLE_INTEGERS:
            _integer(value, name)
        elif value is not None and (isinstance(value, bool) or not isinstance(value, (float, int))):
            raise ValueError("Ensemble settings must have their declared numeric types")
    canonical_json(values)  # Reject nonfinite numbers even where old EnsembleSpec permits them.
    return EnsembleSpec(**values)


@dataclass(frozen=True)
class WorkflowRequest:
    """One declared feature selection and all caller-supplied role configs.

    Inactive auxiliary roles may be retained for an explicitly chosen fallback.
    Active rewriting requires their configurations and a runtime policy identity.
    Ensemble generation overrides must already match the target configuration.
    """

    roles: Mapping[str, ModelConfig]
    features: FeatureChoices
    rewrite_policy_id: str | None = None
    ensemble: EnsembleSpec | None = None

    def __post_init__(self):
        if not isinstance(self.roles, Mapping) or "target" not in self.roles or self.roles.keys() - {"target", "rewriter", "verifier"}:
            raise ValueError("roles require target and permit only rewriter/verifier")
        if not all(isinstance(config, ModelConfig) for config in self.roles.values()):
            raise TypeError("roles must contain ModelConfig values")
        object.__setattr__(self, "roles", MappingProxyType(dict(self.roles)))
        if not isinstance(self.features, FeatureChoices):
            raise TypeError("features must be FeatureChoices")
        if self.rewrite_policy_id is not None:
            _text(self.rewrite_policy_id, "rewrite_policy_id")
        if self.features.rewriting and (not {"rewriter", "verifier"} <= self.roles.keys() or self.rewrite_policy_id is None):
            raise ValueError("Rewriting requires caller role configurations and rewrite_policy_id")
        if self.ensemble is not None:
            if not isinstance(self.ensemble, EnsembleSpec):
                raise TypeError("ensemble must be EnsembleSpec or None")
            _ensemble_from_dict(_ensemble_dict(self.ensemble))
            if self.ensemble.draws != self.features.ensemble_draws:
                raise ValueError("Ensemble draw count differs from features")
            for name in ("max_tokens", "temperature"):
                value = getattr(self.ensemble, name)
                if value is not None and (name not in self.roles["target"].parameters
                                          or canonical_json(value) != canonical_json(self.roles["target"].parameters[name])):
                    raise ValueError("Ensemble generation parameters must match target configuration")
        elif self.features.ensemble_draws != 1:
            raise ValueError("Multiple draws require an explicit EnsembleSpec")

    @property
    def method(self):
        return ("rewritten" if self.features.rewriting else "original") + ("_ensemble" if self.ensemble else "_single")

    def to_dict(self):
        return {"roles": {role: config.to_dict() for role, config in self.roles.items()},
                "features": _features(self.features), "rewrite_policy_id": self.rewrite_policy_id,
                "ensemble": _ensemble_dict(self.ensemble), "method": self.method}

    @classmethod
    def from_dict(cls, data):
        values = _fields(data, {"roles", "features", "rewrite_policy_id", "ensemble", "method"}, required={"method"})
        method = values.pop("method")
        values["roles"] = {role: ModelConfig.from_dict(config) for role, config in values["roles"].items()}
        values["features"] = FeatureChoices(**_fields(values["features"], _FEATURE_FIELDS))
        values["ensemble"] = _ensemble_from_dict(values.get("ensemble"))
        result = cls(**values)
        if result.method != method:
            raise ValueError("Persisted method differs from selected features")
        return result


@dataclass(frozen=True)
class CompletionFallbackPolicy:
    """Explicit completion-only choice; omission authority is absent by default."""

    features: FeatureChoices
    omit_logprob_controls: bool = False

    def __post_init__(self):
        if not isinstance(self.features, FeatureChoices) or self.features.mia_scoring:
            raise ValueError("Fallback requires completion-only FeatureChoices")
        if type(self.omit_logprob_controls) is not bool:
            raise TypeError("omit_logprob_controls must be boolean")

    @property
    def policy_id(self):
        return contract_hash(self.to_dict())

    def to_dict(self):
        return {"features": _features(self.features), "omit_logprob_controls": self.omit_logprob_controls}

    @classmethod
    def from_dict(cls, data):
        values = _fields(data, {"features", "omit_logprob_controls"})
        values["features"] = FeatureChoices(**_fields(values["features"], _FEATURE_FIELDS))
        return cls(**values)


@dataclass(frozen=True)
class WorkflowPlan:
    """Frozen requested/selected decisions; capabilities live in probe evidence.

    This validates a declared decision, not readiness or successful preparation.
    A fallback reason must be a missing-token trigger. Service failure and missing
    calibration cannot stand in for evidence of unsupported token probabilities.
    """

    requested: WorkflowRequest
    selected: WorkflowRequest
    decided_at: str
    fallback_policy: CompletionFallbackPolicy | None = None
    omitted_controls: tuple[str, ...] = ()
    reason: str | None = None

    def __post_init__(self):
        if not isinstance(self.requested, WorkflowRequest) or not isinstance(self.selected, WorkflowRequest):
            raise TypeError("requested and selected must be WorkflowRequest")
        if self.fallback_policy is not None and not isinstance(self.fallback_policy, CompletionFallbackPolicy):
            raise TypeError("fallback_policy must be CompletionFallbackPolicy or None")
        _timestamp(self.decided_at)
        omitted = tuple(self.omitted_controls)
        if len(set(omitted)) != len(omitted) or set(omitted) - {"logprobs", "top_logprobs"}:
            raise ValueError("Only explicit logprob controls may be omitted")
        object.__setattr__(self, "omitted_controls", tuple(sorted(omitted)))
        if self.reason is None:
            if omitted or canonical_json(self.requested.to_dict()) != canonical_json(self.selected.to_dict()):
                raise ValueError("Unchanged plan must preserve the requested configuration")
            return
        if self.reason not in {"missing_token_evidence", "unsupported_token_evidence", "logprob_controls_rejected"}:
            raise ValueError("Fallback reason must identify unavailable token evidence")
        if self.fallback_policy is None or not self.requested.features.mia_scoring:
            raise ValueError("Fallback requires an explicit policy and requested MIA")
        if self.selected.features != self.fallback_policy.features:
            raise ValueError("Selected features differ from the fallback policy")
        if omitted and not self.fallback_policy.omit_logprob_controls:
            raise ValueError("Logprob control omission was not authorized")
        target = self.requested.roles["target"].parameters
        if target.get("logprobs", True) is not True or ("top_logprobs" in target and (target["top_logprobs"] is None or target["top_logprobs"] == 0)):
            raise ValueError("Explicitly disabled logprob controls conflict with requested MIA")
        expected = self.requested.to_dict()
        for name in omitted:
            if name not in target:
                raise ValueError("Omitted control must exist in requested parameters")
            del expected["roles"]["target"]["parameters"][name]
        actual = self.selected.to_dict()
        # Features/method and ensemble choice are caller-selected; config and
        # rewrite policy identity are immutable across capability resolution.
        for key in ("features", "method", "ensemble"):
            expected.pop(key)
            actual.pop(key)
        if canonical_json(expected) != canonical_json(actual):
            raise ValueError("Fallback changed configuration beyond authorized parameters")

    @property
    def policy_id(self):
        return None if self.fallback_policy is None else self.fallback_policy.policy_id

    @property
    def mia_unavailable_reason(self):
        if self.selected.features.mia_scoring:
            return None
        return ("MIA/MCS unavailable: " + self.reason) if self.reason else "MIA/MCS scoring was not selected."

    @property
    def plan_id(self):
        return contract_hash(self.to_dict())

    def to_dict(self):
        return {"requested": self.requested.to_dict(), "selected": self.selected.to_dict(), "decided_at": self.decided_at,
                "fallback_policy": None if self.fallback_policy is None else self.fallback_policy.to_dict(),
                "policy_id": self.policy_id, "omitted_controls": list(self.omitted_controls), "reason": self.reason,
                "mia_unavailable_reason": self.mia_unavailable_reason}

    @classmethod
    def from_dict(cls, data):
        values = _fields(data, {"requested", "selected", "decided_at", "fallback_policy", "policy_id",
                               "omitted_controls", "reason", "mia_unavailable_reason"}, required={"policy_id", "mia_unavailable_reason"})
        policy_id, unavailable = values.pop("policy_id"), values.pop("mia_unavailable_reason")
        for name in ("requested", "selected"):
            values[name] = WorkflowRequest.from_dict(values[name])
        if values.get("fallback_policy") is not None:
            values["fallback_policy"] = CompletionFallbackPolicy.from_dict(values["fallback_policy"])
        result = cls(**values)
        if result.policy_id != policy_id or result.mia_unavailable_reason != unavailable:
            raise ValueError("Persisted fallback identity or MIA availability differs from plan")
        return result
