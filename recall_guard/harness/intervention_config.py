"""Offline validation and call planning for the versioned experiment command.

Items files are objects ``{schema_version: 1, items: [...]}``. The protocol's
``data_hashes.evaluation`` hashes their exact UTF-8 file bytes. Other dataset
keys need explicit file bindings; original/rewritten_calibration instead bind
semantic fitted-scorer fingerprints and are never interpreted as file hashes.

Optional per-cohort ``historical`` metadata has period (pre/post), matching_group,
matching (source/task/format/selection strings), cutoff (model/date/source_url/
retrieved_at/alias_uncertainty), and outcome_dates keyed by that cohort's item
IDs. Outcome dates are never inferred from as_of. A separate source-bearing
registry ``{schema_version: 1, models: {exact_model: {date, source_url,
retrieved_at, alias_uncertainty}}}`` must agree before historical availability
is reported. Unknown/ambiguous or absent bases permit general comparisons.
These declarations are provenance, not independent evidence of good matching.
"""

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from urllib.parse import urlsplit

from recall_guard.core.completion import CompletionProtocol, FeatureChoices, validate_features
from recall_guard.core.provider_anthropic import validate_anthropic_config
from recall_guard.harness._execution_metadata import check_execution_metadata
from recall_guard.harness.intervention_metrics import CONDITION_ORDER
from recall_guard.harness.intervention_types import ExperimentCheckpoint, ExperimentItem, ExperimentProtocol
from recall_guard.harness.rewriting import REWRITE_METHOD_VERSION

_FIT_KEYS = frozenset({"original_calibration", "rewritten_calibration"})
_ROLES = ("target", "rewriter", "verifier")
_BASIS_KEYS = {"date", "source_url", "retrieved_at", "alias_uncertainty"}


def _unique_fields(pairs):
    result = {}
    for name, value in pairs:
        if name in result:
            raise ValueError(f"Duplicate JSON field: {name}")
        result[name] = value
    return result


def _invalid_number(value):
    raise ValueError("Non-finite JSON number is not supported")


def read_experiment_json(path):
    """Read strict UTF-8 JSON once, retaining the exact bytes for hash checks."""
    content = Path(path).read_bytes()
    return json.loads(content.decode("utf-8"), object_pairs_hook=_unique_fields, parse_constant=_invalid_number), content


def _object(value, keys, name):
    if not isinstance(value, Mapping) or set(value) != keys:
        raise ValueError(f"{name} requires exactly: {', '.join(sorted(keys))}")


def _iso_date(value, name):
    if not isinstance(value, str) or date.fromisoformat(value).isoformat() != value:
        raise ValueError(f"{name} must be an ISO YYYY-MM-DD date")
    return value


def _nonempty(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")


def _cutoff_basis(basis):
    _object(basis, _BASIS_KEYS, "cutoff basis")
    if basis["alias_uncertainty"] is not None:
        _nonempty(basis["alias_uncertainty"], "alias_uncertainty")
    if basis["date"] is not None:
        _iso_date(basis["date"], "cutoff date")
        source = basis["source_url"]
        _source_url(source)
        _iso_date(basis["retrieved_at"], "cutoff retrieved_at")
    elif basis["source_url"] is not None or basis["retrieved_at"] is not None:
        if basis["source_url"] is not None:
            _source_url(basis["source_url"])
        if basis["retrieved_at"] is not None:
            _iso_date(basis["retrieved_at"], "cutoff retrieved_at")


def _source_url(value):
    try:
        parsed = urlsplit(value) if isinstance(value, str) else None
        valid = (parsed is not None and parsed.scheme in {"http", "https"} and parsed.hostname
                 and parsed.username is None and parsed.password is None and not parsed.query and not parsed.fragment)
        if valid:
            _ = parsed.port
    except ValueError:
        valid = False
    if not valid:
        raise ValueError("cutoff source_url must be a credential-free HTTP(S) URL without query or fragment")


def _registry(path):
    if path is None:
        return {}
    value, _ = read_experiment_json(path)
    _object(value, {"schema_version", "models"}, "cutoff registry")
    if type(value["schema_version"]) is not int or value["schema_version"] != 1:
        raise ValueError("Unsupported cutoff registry schema_version")
    if not isinstance(value["models"], dict):
        raise ValueError("cutoff registry models must be an object")
    for model, basis in value["models"].items():
        _nonempty(model, "registry exact model")
        _cutoff_basis(basis)
    return value["models"]


def _history(protocol, items, registry):
    cohorts, groups, reasons = {}, {}, []
    for name, definition in protocol.to_dict()["cohort_definitions"].items():
        if not isinstance(definition, dict):
            raise ValueError("cohort definitions must be objects")
        metadata = definition.get("historical")
        if metadata is None:
            reasons.append(f"{name}: no historical metadata; general comparison only")
            continue
        _object(metadata, {"period", "matching_group", "matching", "cutoff", "outcome_dates"}, "historical metadata")
        if metadata["period"] not in {"pre", "post"}:
            raise ValueError("historical period must be pre or post")
        _nonempty(metadata["matching_group"], "matching_group")
        _object(metadata["matching"], {"source", "task", "format", "selection"}, "historical matching")
        for key, value in metadata["matching"].items():
            _nonempty(value, "matching " + key)
        cutoff = metadata["cutoff"]
        _object(cutoff, _BASIS_KEYS | {"model"}, "historical cutoff")
        if cutoff["model"] != protocol.target_model.model:
            raise ValueError("Historical cutoff model must match the exact target model")
        basis = {key: cutoff[key] for key in _BASIS_KEYS}
        _cutoff_basis(basis)
        members = [item for item in items if item.cohort == name]
        outcomes = metadata["outcome_dates"]
        if not isinstance(outcomes, dict) or set(outcomes) != {item.item_id for item in members}:
            raise ValueError("Historical outcome_dates must explicitly cover exactly the cohort item IDs")
        for item in members:
            outcome = _iso_date(outcomes[item.item_id], "outcome_date")
            if outcome < item.as_of:
                raise ValueError("Historical outcome_date must not precede as_of")
            if cutoff["date"] is not None and not cutoff["alias_uncertainty"]:
                if metadata["period"] == "pre" and outcome > cutoff["date"]:
                    raise ValueError("Historical pre outcome lies after the documented cutoff")
                if metadata["period"] == "post" and (item.as_of <= cutoff["date"] or outcome <= cutoff["date"]):
                    raise ValueError("Historical post as_of and outcome must both follow the documented cutoff")
        entry = registry.get(cutoff["model"])
        if entry is not None and entry != basis:
            raise ValueError("Historical cutoff metadata contradicts the exact-model registry entry")
        if entry is None:
            reasons.append(f"{name}: no exact-model source-bearing registry entry")
        if cutoff["date"] is None or cutoff["alias_uncertainty"]:
            reasons.append(f"{name}: unknown cutoff or ambiguous alias; general comparison only")
        if not members:
            reasons.append(f"{name}: no eligible items")
        groups.setdefault(metadata["matching_group"], []).append(metadata)
        cohorts[name] = metadata
    for group, members in groups.items():
        if {member["period"] for member in members} != {"pre", "post"}:
            raise ValueError(f"Historical matching group {group} needs both pre and post cohorts")
        if any(member["matching"] != members[0]["matching"] or member["cutoff"] != members[0]["cutoff"]
               for member in members):
            raise ValueError(f"Historical matching group {group} has inconsistent matching/cutoff declarations")
    return {"cutoff_comparison_eligible": bool(cohorts) and not reasons, "reasons": reasons, "cohorts": cohorts,
            "interpretation": "historical gaps remain proxies; matching declarations require audit"}


@dataclass(frozen=True)
class InterventionConfig:
    """Validated offline inputs; no client or resolved credential is retained."""

    protocol: ExperimentProtocol
    items: tuple[ExperimentItem, ...]
    historical: dict


def load_experiment_config(protocol_path, items_path, *, data_files=None, cutoff_registry=None):
    """Validate bytes, schemas, local provider policies and declared comparisons."""
    raw_protocol, _ = read_experiment_json(protocol_path)
    protocol = ExperimentProtocol.from_dict(raw_protocol)
    raw_items, item_bytes = read_experiment_json(items_path)
    _object(raw_items, {"schema_version", "items"}, "items file")
    if type(raw_items["schema_version"]) is not int or raw_items["schema_version"] != 1:
        raise ValueError("Unsupported items schema_version")
    if not isinstance(raw_items["items"], list) or not raw_items["items"]:
        raise ValueError("items must be a nonempty array")
    items = tuple(ExperimentItem.from_dict(value) for value in raw_items["items"])
    ExperimentCheckpoint(protocol, items)
    if protocol.task_metric != "accuracy":
        raise ValueError("The experiment analyzer currently supports task_metric=accuracy only")
    if protocol.rewrite_method_version != REWRITE_METHOD_VERSION:
        raise ValueError("Unsupported rewrite_method_version")
    protocol.build_ensemble_spec()
    fits = _FIT_KEYS & protocol.data_hashes.keys()
    if fits and fits != _FIT_KEYS:
        raise ValueError("Scored protocol requires both original_calibration and rewritten_calibration fitted scorer fingerprints")
    _nonempty(protocol.audit_selection.get("method"), "audit_selection.method")
    for role in _ROLES:
        model = getattr(protocol, role + "_model")
        try:
            if model.protocol == CompletionProtocol.ANTHROPIC_MESSAGES:
                validate_anthropic_config(model, FeatureChoices(mia_scoring=bool(fits) and role == "target"))
            else:
                validate_features(model, FeatureChoices(mia_scoring=bool(fits) and role == "target"))
        except (ValueError, TypeError) as error:
            raise ValueError(f"{role}: {error}") from None
    bindings = dict(data_files or {})
    if "evaluation" in bindings or _FIT_KEYS & bindings.keys():
        raise ValueError("evaluation is bound to --items; calibration keys bind scorers, not data files")
    expected = set(protocol.data_hashes) - _FIT_KEYS - {"evaluation"}
    if set(bindings) != expected:
        raise ValueError(f"Dataset hash bindings differ: required {sorted(expected)}, supplied {sorted(bindings)}")
    hashes = {"evaluation": hashlib.sha256(item_bytes).hexdigest()}
    hashes.update({name: hashlib.sha256(Path(path).read_bytes()).hexdigest() for name, path in bindings.items()})
    for name, digest in hashes.items():
        if protocol.data_hashes.get(name) != digest:
            raise ValueError(f"Dataset hash mismatch for {name}")
    history = _history(protocol, items, _registry(cutoff_registry))
    return InterventionConfig(protocol, items, history)


def experiment_plan(config, *, prior_run=None):
    """Return a credential-free offline plan of physical request attempts.

    Nominal counts assume accepted rewrites and first-attempt successes. Retry
    upper bounds are subordinate to the shared remaining cap. A safe resume
    probes all roles again and schedules only pending whole arms.
    """
    protocol, items = config.protocol, config.items
    used, pending = 0, {(item.item_id, arm) for item in items for arm in ("original", "rewritten")}
    stopped = False
    if prior_run is not None:
        resume = prior_run.checkpoint.validate_restart(protocol, items)
        used = resume.prior_used_requests
        pending -= {(arm.item_id, arm.arm.value) for arm in prior_run.arms}
        stopped = bool(prior_run.stop_reason) or prior_run.checkpoint.status == "completed"
    nominal = {role: 0 if stopped else 1 for role in _ROLES}
    if not stopped:
        nominal["target"] += len(pending) * protocol.draws
        rewritten = sum(arm == "rewritten" for _, arm in pending)
        nominal["rewriter"] += rewritten
        nominal["verifier"] += rewritten
    upper = {role: count * (1 + getattr(protocol, role + "_model").max_retries) for role, count in nominal.items()}
    roles = {}
    for role in _ROLES:
        model = getattr(protocol, role + "_model")
        features = FeatureChoices(ensemble_draws=protocol.draws, rewriting=True,
                                  mia_scoring=bool(_FIT_KEYS & protocol.data_hashes.keys())) if role == "target" else FeatureChoices()
        declared = model.inspect(features)
        roles[role] = {"config": model.to_dict(), "selected_features": asdict(features),
                       "declared_capabilities": {name: evidence.to_dict() for name, evidence in declared.capabilities.items()},
                       "actual_capabilities": {name: {"status": "unverified", "observed": False,
                                                      "source": "No actual probe in this offline plan"}
                                               for name in declared.capabilities}}
    fits = sorted(_FIT_KEYS & protocol.data_hashes.keys())
    plan = {"protocol_id": protocol.protocol_id, "conditions": list(CONDITION_ORDER), "single_draw_index": 0,
            "draws_per_arm": protocol.draws, "rewrite_scope": protocol.rewrite_scope,
            "shared_target_settings": True, "roles": roles, "ready_for_batch": False,
            "calibration": {"required_fingerprints": {key: protocol.data_hashes[key] for key in fits},
                            "available": False, "reason": "CLI has no fitted scorers; use the Python ExperimentRunner API"
                            if fits else "MIA/MCS not selected; raw confidence is a separate optional response field"},
            "audit_selection": protocol.to_dict()["audit_selection"], "task_metric": protocol.task_metric,
            "utility_tolerance": protocol.utility_tolerance, "data_hashes": dict(protocol.data_hashes),
            "historical": config.historical,
            "physical_calls": {"nominal_by_role": nominal, "nominal_total": sum(nominal.values()),
                               "retry_upper_bound_by_role": upper, "retry_upper_bound_total": sum(upper.values()),
                               "hard_request_cap": protocol.request_budget, "prior_used": used,
                               "hard_remaining_cap": protocol.request_budget - used,
                               "nominal_fits_remaining_cap": sum(nominal.values()) <= protocol.request_budget - used,
                               "basis": "Fresh execution has one probe per role; resume intentionally re-probes. "
                               "Fidelity rejection, failures and hard cap can reduce verification/target calls. "
                               "Single conditions reuse ensemble draw 0; n is not total request cost."}}
    check_execution_metadata(plan)
    return plan
