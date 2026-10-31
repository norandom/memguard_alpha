"""Versioned, credential-free experiment bundles and offline metric replay.

``records.json`` is the explicit ExperimentRun snapshot, not runtime restart
state. ``metrics.json`` contains the existing analyzer's output. The manifest
binds exact UTF-8 file bytes (including final LF) with SHA-256; copying must
preserve bytes. These hashes establish integrity, never provider origin.

The destination must not exist. A manifest is published last, so an interrupted
export cannot be mistaken for a complete bundle. Paths inside the bundle are
fixed and relative. Evidence/prompt/response text is preserved verbatim; factual
paths in that text are data, never artifact references. Callers remain
responsible for sensitive material in source text. Typed configuration excludes
credentials; request controls sanitize diagnostics. Structured credential or
header fields in execution metadata are rejected before publication.
"""

import hashlib
import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath, PureWindowsPath
from urllib.parse import urlsplit

from recall_guard.harness.intervention_journal import ExperimentRun
from recall_guard.harness.intervention_metrics import summarize_experiment

_VERSION = 1
_FILES = {"records": "records.json", "metrics": "metrics.json"}
_ROLES = ("target", "rewriter", "verifier")
_INTEGRITY = "sha256_exact_utf8_file_bytes_including_lf; not_provider_origin_evidence"
_CREDENTIAL_FIELDS = frozenset({
    "apikey", "authorization", "accesstoken", "password", "credential", "credentials", "credentialref",
    "xapikey", "bearertoken", "secret", "clientsecret", "headers",
})


def _object(value, allowed, name, *, required=()):
    if not isinstance(value, Mapping) or set(value) - set(allowed) or set(required) - set(value):
        raise ValueError(f"Invalid {name} fields")
    return value


def _version(value):
    if type(value) is not int or value != _VERSION:
        raise ValueError("Unsupported report version")


def _relative_path(value):
    if (not isinstance(value, str) or not value or "\\" in value
            or PurePosixPath(value).is_absolute() or PureWindowsPath(value).drive
            or any(part in {"", ".", ".."} for part in value.split("/"))):
        raise ValueError("Invalid relative artifact path")
    return value


def _evidence(values):
    if not isinstance(values, (list, tuple)):
        raise ValueError("Validation evidence must contain portable references")
    result = []
    for value in values:
        if isinstance(value, str) and value.startswith(("https://", "http://")):
            url = urlsplit(value)
            if not url.hostname or url.username is not None or url.password is not None or url.query or url.fragment:
                raise ValueError("Validation evidence URLs must exclude credentials, query and fragment")
        else:
            _relative_path(value)
        result.append(value)
    return result


def _validation(protocol, provenance):
    """Small caller-asserted schema; never infer live coverage from outcomes."""
    provenance = {} if provenance is None else provenance
    _object(provenance, ("execution", "evidence", "providers"), "provenance")
    execution = provenance.get("execution", "unverified")
    if execution not in {"unverified", "offline_fixture", "live_provider"}:
        raise ValueError("Unknown execution validation level")
    evidence = _evidence(provenance.get("evidence", []))
    if execution != "unverified" and not evidence:
        raise ValueError("Explicit execution validation requires evidence")
    providers = provenance.get("providers", {})
    _object(providers, _ROLES, "provider validation")
    result = {}
    for role in _ROLES:
        source = providers.get(role, {})
        _object(source, ("implemented", "offline_tested", "live_measured", "evidence"), "provider evidence")
        coverage = {key: source.get(key, False if key == "live_measured" else None)
                    for key in ("implemented", "offline_tested", "live_measured")}
        if any(type(value) is not bool and value is not None for value in coverage.values()):
            raise ValueError("Provider coverage must be explicit booleans or unknown")
        refs = _evidence(source.get("evidence", []))
        if any(value is True for value in coverage.values()) and not refs:
            raise ValueError("Provider validation claims require evidence")
        if coverage["live_measured"] and execution != "live_provider":
            raise ValueError("Live provider coverage requires live execution provenance")
        config = getattr(protocol, role + "_model")
        result[role] = {"provider": config.provider, "protocol": config.protocol.value, "model": config.model,
                        **coverage, "evidence": refs}
    return {"execution": execution, "evidence": evidence, "providers": result,
            "basis": "caller_asserted_evidence; hashes_do_not_verify_live_origin"}


def _response_controls(response, source):
    return {"source": source, "requested": response["requested_settings"],
            "confirmed": response["confirmed_settings"],
            "requested_unconfirmed": sorted(set(response["requested_settings"]) - set(response["confirmed_settings"]))}


def _check_metadata_fields(value):
    """Inspect JSON metadata containers only, never arbitrary objects or text."""
    if isinstance(value, Mapping):
        for key, child in value.items():
            if key.lower().replace("_", "").replace("-", "") in _CREDENTIAL_FIELDS:
                raise ValueError("Execution metadata must exclude credential and authorization fields")
            _check_metadata_fields(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _check_metadata_fields(child)


def _check_snapshot_metadata(snapshot):
    _object(snapshot, ("checkpoint", "arms", "preflight", "stop_reason"), "run snapshot",
            required=("checkpoint", "arms", "preflight", "stop_reason"))
    # Inspect every saved container, excluding only schema-defined task values.
    # Raw content, tokens and prompts are strings, which the key check ignores.
    checkpoint = snapshot["checkpoint"]
    metadata = dict(snapshot) | {
        "checkpoint": dict(checkpoint) | {
            "items": [{key: value for key, value in item.items() if key != "label"}
                      for item in checkpoint["items"]],
            "draws": [dict(draw) | {
                "metadata": {key: value for key, value in draw["metadata"].items() if key != "answer"},
            } for draw in checkpoint["draws"]],
        },
        "arms": [dict(arm) | {
            "outcome": {key: value for key, value in arm["outcome"].items() if key != "answer"},
        } for arm in snapshot["arms"]],
    }
    _check_metadata_fields(metadata)


def _settings(snapshot):
    """Keep observed controls response-specific; never promote accepted defaults."""
    checkpoint = snapshot["checkpoint"]
    settings = {role: {"requested": checkpoint["protocol"][role + "_model"]["parameters"], "responses": [],
                       "omitted_defaults": "unknown; omitted parameter names and provider defaults are not inferred"}
                for role in _ROLES}
    for index, draw in enumerate(checkpoint["draws"]):
        if draw["response"] is not None:
            settings["target"]["responses"].append(_response_controls(draw["response"], f"checkpoint.draws[{index}].response"))
    for index, rewrite in enumerate(checkpoint["rewrites"]):
        for role, field in (("rewriter", "rewrite_response"), ("verifier", "verification_response")):
            if rewrite[field] is not None:
                settings[role]["responses"].append(_response_controls(rewrite[field], f"checkpoint.rewrites[{index}].{field}"))
    for role, response in snapshot["preflight"].get("responses", {}).items():
        if role not in _ROLES:
            raise ValueError("Unknown preflight model role")
        settings[role]["responses"].append(_response_controls(response, f"preflight.responses.{role}"))
    return settings


def _derived_metadata(run, metrics):
    """All manifest claims derived from saved records share one definition."""
    return {
        "status": run.checkpoint.status.value, "settings": _settings(run.to_dict()),
        "cost": {"estimate": None, "estimate_status": "not_estimated", "currency": None,
                 "recorded_cost": metrics["pooled"]["usage"]["cost"],
                 "recorded_unit": metrics["pooled"]["usage"]["cost_unit"]},
        "interpretation": {
            "rewriting": "paired original versus rewritten outcomes; see metrics effects.rewriting_*",
            "aggregation": "paired single versus ensemble outcomes; single reuses draw zero",
            "scoring": "existing per-draw confidence discounts and calibration quality are separate from rewriting",
            "limits": metrics["pooled"]["interpretation"]["limits"],
            "historical": metrics["historical"],
        },
    }


def _json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2) + "\n").encode("utf-8")


def _write_bytes(path, content):
    with path.open("xb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())


@dataclass(frozen=True)
class ExperimentReport:
    """Loaded records and verified metrics; per-draw scores are retained, not refit.

    Analyzer rows remain item/arm/draw with the single condition sharing draw
    zero. Metric units and eligible denominators belong to the existing analyzer.
    Calibration quality is replayed from its whitelist; no scorer is loaded.
    """

    run: ExperimentRun
    metrics: dict
    manifest: dict

    def recompute_metrics(self) -> dict:
        analysis = self.manifest["analysis"]
        return summarize_experiment(self.run, confidence_level=analysis["confidence_level"],
                                    confidence_component=analysis["confidence_component"])


def export_experiment(run: ExperimentRun, directory, *, confidence_level=0.95,
                      confidence_component="confidence", provenance=None) -> Path:
    """Publish a new bundle, refusing to overwrite any existing directory.

    ``provenance`` allows execution (unverified/offline_fixture/live_provider),
    evidence references and per-role implemented/offline_tested/live_measured
    coverage with its evidence. Missing coverage is unknown or unmeasured.
    These are caller assertions, not inferred from synthetic successful records.
    Inputs must already satisfy the sanitized ExperimentRun contracts.
    """
    metrics = summarize_experiment(run, confidence_level=confidence_level, confidence_component=confidence_component)
    protocol = run.checkpoint.protocol
    validation = _validation(protocol, provenance)
    snapshot = run.to_dict()  # Explicit whitelist; never vars/asdict on live workflow/calibration/client objects.
    _check_snapshot_metadata(snapshot)
    records_bytes = _json_bytes({"schema_version": _VERSION, "run": snapshot})
    metrics_bytes = _json_bytes({"schema_version": _VERSION, "metrics": metrics})
    manifest = {
        "schema_version": _VERSION, "protocol_id": protocol.protocol_id,
        "exported_at": datetime.now(UTC).isoformat(),
        "analysis": {"version": _VERSION, "confidence_level": confidence_level, "confidence_component": confidence_component},
        "integrity": _INTEGRITY,
        "artifacts": {name: {"path": _FILES[name], "sha256": hashlib.sha256(content).hexdigest()}
                      for name, content in (("records", records_bytes), ("metrics", metrics_bytes))},
        "validation": validation, **_derived_metadata(run, metrics),
    }
    manifest_bytes = _json_bytes(manifest)
    destination = Path(directory)
    destination.mkdir(parents=True, exist_ok=False)
    _write_bytes(destination / _FILES["records"], records_bytes)
    _write_bytes(destination / _FILES["metrics"], metrics_bytes)
    pending_manifest = destination / ".manifest.pending"
    _write_bytes(pending_manifest, manifest_bytes)
    pending_manifest.rename(destination / "manifest.json")
    return destination / "manifest.json"


def _decode_json(content):
    try:
        return json.loads(content.decode("utf-8"), parse_constant=_invalid_json, object_pairs_hook=_unique_fields)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("Cannot decode report JSON artifact") from error


def _unique_fields(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate report JSON field")
        result[key] = value
    return result


def _invalid_json(value):
    raise ValueError("Nonfinite report JSON value")


def _artifact(directory, name, entry):
    _object(entry, ("path", "sha256"), "artifact", required=("path", "sha256"))
    reference = _relative_path(entry["path"])
    path = directory / reference
    if reference != _FILES[name] or path.is_symlink() or not path.resolve().is_relative_to(directory.resolve()):
        raise ValueError("Invalid or escaping artifact path")
    try:
        content = path.read_bytes()
    except OSError as error:
        raise ValueError(f"Missing report artifact {name}") from error
    if hashlib.sha256(content).hexdigest() != entry["sha256"]:
        raise ValueError(f"Report artifact integrity mismatch: {name}")
    value = _decode_json(content)
    _object(value, ("schema_version", "run" if name == "records" else "metrics"), name,
            required=("schema_version", "run" if name == "records" else "metrics"))
    _version(value["schema_version"])
    return value["run" if name == "records" else "metrics"]


def load_experiment_report(directory) -> ExperimentReport:
    """Validate versions, relative paths, byte integrity and exact metric replay.

    No clients, credential resolution, requests, fitting or response reduction
    occur here. Partial and failed runs are valid portable research artifacts.
    A checksum is not authentication of a provider or caller validation claim.
    """
    directory = Path(directory)
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise ValueError("Missing or invalid report manifest")
    try:
        manifest = _decode_json(manifest_path.read_bytes())
    except OSError as error:
        raise ValueError("Cannot read report manifest") from error
    fields = ("schema_version", "protocol_id", "exported_at", "status", "analysis", "integrity", "artifacts",
              "settings", "validation", "cost", "interpretation")
    _object(manifest, fields, "manifest", required=fields)
    _version(manifest["schema_version"])
    if datetime.fromisoformat(manifest["exported_at"]).tzinfo is None:
        raise ValueError("Report export timestamp must include a timezone")
    _object(manifest["analysis"], ("version", "confidence_level", "confidence_component"), "analysis",
            required=("version", "confidence_level", "confidence_component"))
    _version(manifest["analysis"]["version"])
    if manifest["integrity"] != _INTEGRITY:
        raise ValueError("Unsupported report integrity contract")
    _object(manifest["artifacts"], _FILES, "artifacts", required=_FILES)
    snapshot = _artifact(directory, "records", manifest["artifacts"]["records"])
    _check_snapshot_metadata(snapshot)
    run = ExperimentRun.from_dict(snapshot)
    if manifest["protocol_id"] != run.checkpoint.protocol.protocol_id:
        raise ValueError("Report protocol identity mismatch")
    metrics = _artifact(directory, "metrics", manifest["artifacts"]["metrics"])
    report = ExperimentReport(run, metrics, manifest)
    if _json_bytes(metrics) != _json_bytes(report.recompute_metrics()):
        raise ValueError("Saved report metrics do not match offline replay")
    for key, expected in _derived_metadata(run, metrics).items():
        if _json_bytes(manifest[key]) != _json_bytes(expected):
            raise ValueError(f"Report {key} does not match records and replayed metrics")
    validation = manifest["validation"]
    _object(validation, ("execution", "evidence", "providers", "basis"), "validation",
            required=("execution", "evidence", "providers", "basis"))
    _object(validation["providers"], _ROLES, "provider validation", required=_ROLES)
    for source in validation["providers"].values():
        _object(source, ("provider", "protocol", "model", "implemented", "offline_tested", "live_measured", "evidence"),
                "provider validation")
    providers = {role: {key: value for key, value in source.items()
                        if key not in {"provider", "protocol", "model"}}
                 for role, source in validation["providers"].items()}
    if validation != _validation(run.checkpoint.protocol, {
        "execution": validation["execution"], "evidence": validation["evidence"], "providers": providers,
    }):
        raise ValueError("Invalid report provider validation")
    return report
