"""Stage limits of the Modal serving arm: pure validation, no SDK, network or credentials."""

import json
import os
import socket
from pathlib import Path

import pytest

from deploy.modal.manifest import load_plan
from deploy.modal.stage_limits import (
    STAGES,
    StageLimits,
    load_stage_limits,
    parse_stage_limits,
    require_serving_limits,
    require_staging_limits,
    staging_worst_case_usd,
)

ROOT = Path(__file__).resolve().parents[2]
GEMMA = ROOT / "configs/modal/gemma-3-4b-measurement.json"
PILOT = ROOT / "configs/modal/gpt-oss-20b-pilot.json"
PLAN_SHA = "0123456789abcdef" * 4


def limits(**overrides):
    return {"plan_sha256": PLAN_SHA, "stage": "weight_staging", "max_requests": 0, "max_output_tokens": 0,
            "max_seconds": 3600, "max_usd": 1.0} | overrides


def parse(raw, manifest=GEMMA):
    return parse_stage_limits(json.dumps(raw), load_plan(manifest))


def test_limits_load_from_a_file_without_network_or_environment(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Stage limits attempted network or environment access")

    path = tmp_path / "full-run.json"
    path.write_text(json.dumps(limits(stage="full_run", max_requests=8064, max_output_tokens=8064 * 256,
                                      max_seconds=50000, max_usd=20)), encoding="utf-8")
    plan = load_plan(GEMMA)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(os, "getenv", forbidden)
    loaded = load_stage_limits(path, plan)
    assert loaded == StageLimits(PLAN_SHA, "full_run", 8064, 2064384, 50000, 20.0)
    assert type(loaded.max_usd) is float


def test_the_stage_set_is_fixed_in_execution_order():
    assert STAGES == ("weight_staging", "deployment", "smoke_run", "calibration", "full_run", "negative_control")
    assert [parse(limits(stage=stage)).stage for stage in STAGES] == list(STAGES)
    for stage in ("pilot", "macro_calibration", "WEIGHT_STAGING", "", None):
        with pytest.raises(ValueError, match="Unknown stage"):
            parse(limits(stage=stage))


@pytest.mark.parametrize("field,ceiling", [
    ("max_requests", 13857), ("max_output_tokens", 3547392), ("max_seconds", 86142), ("max_usd", 34.34),
])
def test_each_limit_must_fit_the_gemma_manifest_ceiling(field, ceiling):
    assert getattr(parse(limits(**{field: ceiling})), field) == ceiling
    above = ceiling + (0.01 if field == "max_usd" else 1)
    with pytest.raises(ValueError, match=f"{field} exceeds the manifest's reviewed ceiling"):
        parse(limits(**{field: above}))


def test_the_ceiling_is_the_selected_manifest_own():
    assert parse(limits(max_seconds=1200, max_usd=5), PILOT).max_seconds == 1200
    with pytest.raises(ValueError, match="max_seconds exceeds"):
        parse(limits(max_seconds=1201, max_usd=1), PILOT)
    with pytest.raises(ValueError, match="max_requests exceeds"):
        parse(limits(max_requests=9), PILOT)


@pytest.mark.parametrize("value", ["A" * 64, "0" * 63, "0" * 65, "g" * 64, 64, None, " " + "0" * 63])
def test_plan_hash_must_be_64_lowercase_hex(value):
    with pytest.raises(ValueError, match="plan_sha256"):
        parse(limits(plan_sha256=value))


@pytest.mark.parametrize("field,value", [
    ("max_requests", -1), ("max_requests", True), ("max_requests", 1.0), ("max_output_tokens", "256"),
    ("max_seconds", 0), ("max_seconds", 300.0), ("max_seconds", False), ("max_usd", 0), ("max_usd", -1.0),
    ("max_usd", True), ("max_usd", "1.0"), ("max_usd", None),
])
def test_limits_must_be_typed_and_positive(field, value):
    with pytest.raises(ValueError, match=field):
        parse(limits(**{field: value}))


@pytest.mark.parametrize("text", [
    json.dumps(limits()).replace('"max_usd": 1.0', '"max_usd": NaN'),
    json.dumps(limits()).replace('"max_usd": 1.0', '"max_usd": Infinity'),
    json.dumps(limits()).replace('"max_usd": 1.0', '"max_usd": 1e400'),
    json.dumps(limits()).replace('"stage": "weight_staging"', '"stage": "full_run", "stage": "weight_staging"'),
    json.dumps(limits() | {"approval": True}),
    json.dumps({key: value for key, value in limits().items() if key != "max_usd"}),
    json.dumps([limits()]),
])
def test_malformed_files_are_refused(text):
    with pytest.raises(ValueError):
        parse_stage_limits(text, load_plan(GEMMA))


def test_staging_refuses_when_running_to_its_timeout_could_cost_more_than_its_limit():
    plan = load_plan(GEMMA)
    cost = json.loads(GEMMA.read_text(encoding="utf-8"))["cost"]
    assert (plan.resources.gpu_count, plan.resources.cpu_physical_cores, plan.resources.memory_gib) == (1, 4, 16)
    # CPU only, no GPU: 4 cores x USD 0.0000131 + 16 GiB x USD 0.00000222 per second, over 3,600 s.
    worst = 4 * 0.0000131 * 3600 + 16 * 0.00000222 * 3600
    assert worst == pytest.approx(0.316512)
    assert staging_worst_case_usd(parse(limits()), plan, cost) == pytest.approx(worst)
    assert require_staging_limits(parse(limits(max_usd=0.32)), plan, cost) == pytest.approx(worst)
    with pytest.raises(ValueError, match="Staging refused"):
        require_staging_limits(parse(limits(max_usd=0.31)), plan, cost)
    with pytest.raises(ValueError, match="max_seconds exceeds"):
        parse(limits(max_seconds=86143, max_usd=34))


def test_each_mode_takes_only_its_own_stage():
    plan = load_plan(GEMMA)
    cost = json.loads(GEMMA.read_text(encoding="utf-8"))["cost"]
    with pytest.raises(ValueError, match="weight_staging stage limits"):
        require_staging_limits(parse(limits(stage="deployment")), plan, cost)
    with pytest.raises(ValueError, match="not weight_staging"):
        require_serving_limits(parse(limits()))
    for stage in STAGES[1:]:
        require_serving_limits(parse(limits(stage=stage)))
