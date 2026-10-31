"""Pure protocol validation, byte hashes, cohorts and honest call planning."""

import hashlib
import json
from dataclasses import replace

import pytest

from recall_guard.core.completion import CapabilityEvidence, ModelConfig
from recall_guard.harness.intervention_config import experiment_plan, load_experiment_config
from tests.harness.test_intervention import items, protocol


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def config_files(tmp_path, *, count=2, models="openai_chat", **changes):
    values = items(count)
    item_path = tmp_path / "items.json"
    write_json(item_path, {"schema_version": 1, "items": [item.to_dict() for item in values]})
    configs = {role + "_model": ModelConfig(models, "https://fixture.invalid/v1/completions", role,
               parameters={"max_tokens": 64}, max_retries=0) for role in ("target", "rewriter", "verifier")}
    p = protocol(**configs, data_hashes={"evaluation": hashlib.sha256(item_path.read_bytes()).hexdigest()}, **changes)
    protocol_path = tmp_path / "protocol.json"
    write_json(protocol_path, p.to_dict())
    return protocol_path, item_path, p, values


def mutate(path, change):
    data = json.loads(path.read_text(encoding="utf-8"))
    change(data)
    write_json(path, data)


def test_offline_plan_counts_physical_roles_and_never_promotes_declarations(tmp_path):
    pp, ip, p, values = config_files(tmp_path)
    p = replace(p, target_model=replace(p.target_model, max_retries=2,
                capabilities={"text_generation": CapabilityEvidence("supported", "prior probe", observed=True)}),
                rewriter_model=replace(p.rewriter_model, max_retries=1))
    write_json(pp, p.to_dict())
    config = load_experiment_config(pp, ip)
    plan = experiment_plan(config)
    assert config.protocol.to_dict() == p.to_dict() and config.items == values
    assert plan["physical_calls"]["nominal_by_role"] == {"target": 13, "rewriter": 3, "verifier": 3}
    assert plan["physical_calls"]["retry_upper_bound_by_role"] == {"target": 39, "rewriter": 6, "verifier": 3}
    assert plan["physical_calls"]["nominal_total"] == 19
    assert plan["physical_calls"]["hard_remaining_cap"] == 100
    assert plan["roles"]["target"]["actual_capabilities"]["text_generation"]["status"] == "unverified"
    assert plan["roles"]["target"]["declared_capabilities"]["text_generation"]["status"] == "supported"
    assert plan["historical"]["cutoff_comparison_eligible"] is False
    assert len(plan["conditions"]) == 4 and plan["single_draw_index"] == 0


@pytest.mark.parametrize("location,change,match", [
    ("protocol", lambda d: d.update(schema_version=99), "schema_version"),
    ("items", lambda d: d.update(schema_version=99), "schema_version"),
    ("protocol", lambda d: d.update(task_metric="f1"), "accuracy"),
    ("protocol", lambda d: d.update(rewrite_method_version="other"), "rewrite_method"),
    ("protocol", lambda d: d["ensemble_settings"].update(draws=4), "draw"),
    ("protocol", lambda d: d["ensemble_settings"].update(temperature=0.7), "target"),
    ("protocol", lambda d: d["ensemble_settings"].update(min_parsed=4), "min_parsed"),
    ("protocol", lambda d: d["ensemble_settings"].update(max_workers=1.5), "integer"),
    ("protocol", lambda d: d["ensemble_settings"].update(location_mode="invented"), "location_mode"),
    ("protocol", lambda d: d["ensemble_settings"].update(retain_draws="yes"), "boolean"),
    ("protocol", lambda d: d["audit_selection"].pop("method"), "predeclared"),
    ("protocol", lambda d: d["target_model"]["parameters"].update(max_tokens=False), "max_tokens"),
    ("protocol", lambda d: d["target_model"].update(credential_ref="KEY"), "credential_ref"),
    ("protocol", lambda d: d["data_hashes"].update(evaluation="b" * 64), "hash"),
    ("items", lambda d: d["items"].append(d["items"][0]), "Duplicate"),
    ("items", lambda d: d["items"][0].update(cohort="absent"), "cohort"),
])
def test_invalid_inputs_fail_before_calls(tmp_path, location, change, match):
    pp, ip, _, _ = config_files(tmp_path)
    mutate(pp if location == "protocol" else ip, change)
    if location == "items":
        mutate(pp, lambda d: d["data_hashes"].update(evaluation=hashlib.sha256(ip.read_bytes()).hexdigest()))
    with pytest.raises((ValueError, TypeError), match=match):
        load_experiment_config(pp, ip)


def test_all_declared_files_are_hashed_and_missing_or_extra_bindings_fail(tmp_path):
    pp, ip, _, _ = config_files(tmp_path)
    source = tmp_path / "source.txt"
    source.write_bytes("東京\r\n".encode())
    mutate(pp, lambda d: d["data_hashes"].update(source=hashlib.sha256(source.read_bytes()).hexdigest()))
    with pytest.raises(ValueError, match="source"):
        load_experiment_config(pp, ip)
    assert load_experiment_config(pp, ip, data_files={"source": source})
    source.write_text("東京\n", encoding="utf-8", newline="\n")
    with pytest.raises(ValueError, match="hash"):
        load_experiment_config(pp, ip, data_files={"source": source})


def test_calibration_fingerprints_are_not_file_hashes_and_plan_requires_actual_fits(tmp_path):
    pp, ip, _, _ = config_files(tmp_path)
    mutate(pp, lambda d: d["data_hashes"].update(original_calibration="b" * 64, rewritten_calibration="c" * 64))
    plan = experiment_plan(load_experiment_config(pp, ip))
    assert plan["roles"]["target"]["selected_features"]["mia_scoring"] is True
    assert plan["calibration"]["available"] is False
    assert len(plan["calibration"]["required_fingerprints"]) == 2
    with pytest.raises(ValueError, match="scorers, not data"):
        load_experiment_config(pp, ip, data_files={"original_calibration": ip})
    mutate(pp, lambda d: d["data_hashes"].pop("rewritten_calibration"))
    with pytest.raises(ValueError, match="both.*fingerprints"):
        load_experiment_config(pp, ip)


def test_anthropic_scored_protocol_is_locally_incompatible(tmp_path):
    pp, ip, _, _ = config_files(tmp_path, models="anthropic_messages")
    mutate(pp, lambda d: d["data_hashes"].update(original_calibration="b" * 64, rewritten_calibration="c" * 64))
    with pytest.raises(ValueError, match="target.*no logprobs"):
        load_experiment_config(pp, ip)


def historical_definitions(values, *, cutoff="2024-01-02", uncertainty=None):
    matching = {"source": "fixture", "task": "direction", "format": "same", "selection": "balanced"}
    basis = {"model": "target", "date": cutoff, "source_url": "https://example.org/model-card",
             "retrieved_at": "2026-10-31", "alias_uncertainty": uncertainty}
    return {"before": {"historical": {"period": "pre", "matching_group": "paired", "matching": matching,
                "cutoff": basis, "outcome_dates": {values[0].item_id: "2024-01-02"}}},
            "after": {"historical": {"period": "post", "matching_group": "paired", "matching": matching,
                "cutoff": basis, "outcome_dates": {values[1].item_id: "2024-01-04"}}}}


def historical_files(tmp_path, **options):
    pp, ip, p, values = config_files(tmp_path)
    values = (replace(values[0], cohort="before"), replace(values[1], cohort="after", as_of="2024-01-03"))
    write_json(ip, {"schema_version": 1, "items": [item.to_dict() for item in values]})
    p = replace(p, cohort_definitions=historical_definitions(values, **options),
                data_hashes={"evaluation": hashlib.sha256(ip.read_bytes()).hexdigest()})
    write_json(pp, p.to_dict())
    registry_path = tmp_path / "cutoffs.json"
    basis = dict(p.cohort_definitions["before"]["historical"]["cutoff"])
    basis.pop("model")
    write_json(registry_path, {"schema_version": 1, "models": {"target": basis}})
    return pp, ip, registry_path


def test_sourced_exact_history_validates_actual_dates_and_matching(tmp_path):
    pp, ip, registry = historical_files(tmp_path)
    history = experiment_plan(load_experiment_config(pp, ip, cutoff_registry=registry))["historical"]
    assert history["cutoff_comparison_eligible"] is True
    assert history["cohorts"]["before"]["cutoff"]["source_url"] == "https://example.org/model-card"


@pytest.mark.parametrize("cutoff,uncertainty", [(None, None), ("2024-01-02", "moving alias")])
def test_unknown_or_ambiguous_cutoff_allows_only_general_comparison(tmp_path, cutoff, uncertainty):
    pp, ip, registry = historical_files(tmp_path, cutoff=cutoff, uncertainty=uncertainty)
    assert experiment_plan(load_experiment_config(pp, ip, cutoff_registry=registry))["historical"]["cutoff_comparison_eligible"] is False


@pytest.mark.parametrize("field,value,match", [
    ("model", "another-model", "exact target"),
    ("date", "2023-12-31", "pre.*cutoff"),
    ("source_url", None, "source_url"),
])
def test_invalid_explicit_historical_claims_fail(tmp_path, field, value, match):
    pp, ip, _ = historical_files(tmp_path)
    mutate(pp, lambda d: [c["historical"]["cutoff"].update({field: value}) for c in d["cohort_definitions"].values()])
    with pytest.raises(ValueError, match=match):
        load_experiment_config(pp, ip)


def test_no_outcome_date_is_inferred_from_as_of(tmp_path):
    pp, ip, _ = historical_files(tmp_path)
    mutate(pp, lambda d: d["cohort_definitions"]["after"]["historical"].update(outcome_dates={}))
    with pytest.raises(ValueError, match="outcome_dates"):
        load_experiment_config(pp, ip)


def test_missing_exact_registry_basis_withholds_historical_eligibility(tmp_path):
    pp, ip, registry = historical_files(tmp_path)
    assert experiment_plan(load_experiment_config(pp, ip))["historical"]["cutoff_comparison_eligible"] is False
    mutate(registry, lambda d: d["models"].update(other=d["models"].pop("target")))
    plan = experiment_plan(load_experiment_config(pp, ip, cutoff_registry=registry))
    assert plan["historical"]["cutoff_comparison_eligible"] is False
    assert any("exact-model" in reason for reason in plan["historical"]["reasons"])


@pytest.mark.parametrize("change,match", [
    (lambda d: d.update(schema_version=2), "schema_version"),
    (lambda d: d["models"]["target"].update(date="2024-01-01"), "contradicts"),
    (lambda d: d["models"]["target"].update(source_url="https://user:secret@example.org/source"), "credential-free"),
    (lambda d: d["models"]["target"].update(source_url="https://example.org/source?token=secret"), "credential-free"),
])
def test_invalid_cutoff_registry_fails_clearly(tmp_path, change, match):
    pp, ip, registry = historical_files(tmp_path)
    mutate(registry, change)
    with pytest.raises(ValueError, match=match):
        load_experiment_config(pp, ip, cutoff_registry=registry)


def test_historical_mismatched_format_and_post_dates_are_rejected(tmp_path):
    pp, ip, _ = historical_files(tmp_path)
    mutate(pp, lambda d: d["cohort_definitions"]["after"]["historical"]["matching"].update(format="different"))
    with pytest.raises(ValueError, match="inconsistent matching"):
        load_experiment_config(pp, ip)
    pp, ip, _ = historical_files(tmp_path)
    mutate(ip, lambda d: d["items"][1].update(as_of="2024-01-02"))
    mutate(pp, lambda d: d["data_hashes"].update(evaluation=hashlib.sha256(ip.read_bytes()).hexdigest()))
    with pytest.raises(ValueError, match="post as_of"):
        load_experiment_config(pp, ip)


def test_anthropic_native_parameter_validation_is_pure_and_role_specific(tmp_path):
    pp, ip, _, _ = config_files(tmp_path, models="anthropic_messages")
    mutate(pp, lambda d: d["rewriter_model"]["parameters"].update(temperature=0))
    with pytest.raises(ValueError, match="rewriter.*temperature"):
        load_experiment_config(pp, ip)


@pytest.mark.parametrize("content", ['{"schema_version":1,"schema_version":1}', '{"schema_version":NaN}', '\ufeff{}'])
def test_invalid_json_is_rejected(tmp_path, content):
    pp, ip, _, _ = config_files(tmp_path)
    pp.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError):
        load_experiment_config(pp, ip)
