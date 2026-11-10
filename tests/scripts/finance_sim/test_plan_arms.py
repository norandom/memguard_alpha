"""Study plan arms, variant sets, limits, pilot settings and scored probability (task 7.3; Requirements 8.1, 8.4, 8.10).

Plans are frozen offline from the synthetic month and EDGAR inputs of test_plan.py,
with the committed Gemma 3 4B serving manifest; nothing is fetched or written
outside ``tmp_path``.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import fields, replace
from decimal import Decimal
from pathlib import Path

import pytest

from deploy.modal.manifest import load_plan as load_modal_plan
from deploy.modal.stage_limits import STAGES, parse_stage_limits, require_serving_limits, require_staging_limits
from recall_guard.dataset.macro_regime import RealizedReturns, is_correct
from scripts.finance_sim import limits as study_limits
from scripts.finance_sim import plan as plan_module
from scripts.finance_sim.plan import (
    VARIANTS,
    ArmPlan,
    PilotSettings,
    PlanChangedError,
    PlanLimits,
    ScoredProbability,
    StudyArms,
    StudyPlan,
    ensure_unchanged,
    freeze_plan,
    load_plan,
    require_complete,
    serving_limits_record,
    size_limits,
    stage_limits_files,
    study_arms,
    variants_for,
    write_plan,
)
from scripts.measure_modal_model import CONTROLLER_CORES, CONTROLLER_EXTRA_SECONDS, CONTROLLER_GIB, compute_rate
from tests.scripts.finance_sim.test_plan import (
    CREATED_AT,
    EDGAR_FILES,
    GEMMA,
    GEMMA_CUTOFF,
    ITEMS,
    SEED,
    _at,
    _dict_paths,
    _leaf_paths,
    canonical,
    classes,
    digest,
    edgar_rows,
    rehash,
)

ROOT = Path(plan_module.__file__).resolve().parents[2]
MANIFEST_PATH = ROOT / "configs" / "modal" / "gemma-3-4b-measurement.json"
MANIFEST_BYTES = MANIFEST_PATH.read_bytes()
MANIFEST_JSON = json.loads(MANIFEST_BYTES)
LATER = "vendor/later-model-4b"
SIX = ("single_none", "single_top_k", "single_full_distribution",
       "ensemble_none", "ensemble_top_k", "ensemble_full_distribution")
SERVING_FOUR = ("single_none", "single_top_k", "ensemble_none", "ensemble_top_k")


def full_plan(cutoffs: dict[str, str] | None = None, **overrides) -> StudyPlan:
    arguments = {"seed": SEED, "created_at": CREATED_AT, "items": ITEMS, "edgar_files": EDGAR_FILES,
                 "arms": study_arms(), "serving_manifest": plan_module.load_serving_manifest()} | overrides
    return freeze_plan(classes(cutoffs or {GEMMA: GEMMA_CUTOFF}), edgar_rows(), **arguments)


@pytest.fixture(scope="module")
def plan() -> StudyPlan:
    return full_plan()


def stages(plan: StudyPlan, arm: str) -> dict:
    return {stage.stage: stage for stage in plan.limits.stages[arm]}


def usd_up(value: float) -> float:
    return math.ceil(round(value * 100, 6)) / 100


# ---------- arms and variant sets (8.1, 8.10) ----------

def test_the_sampling_arm_freezes_the_six_variants_its_temperature_and_its_seed_base(plan):
    sampling = plan.arms.arms["sampling"]
    assert VARIANTS == SIX
    assert sampling.variants == SIX and dict(sampling.missing_variants) == {}
    assert sampling.temperature == 0.7
    assert sampling.seed_base == plan_module.SAMPLING_SEED_BASE and type(sampling.seed_base) is int
    # The probe arm (user decision 2026-11-10): 8 draws per rendering with at least 6 parsed.
    assert (sampling.draws, sampling.min_parsed, sampling.gated) == (8, 6, True)
    assert dict(sampling.calibration_draws) == {"macro": 8, "edgar": 4}
    assert sampling.estimate_sources == ("top_k", "full_distribution")
    assert (sampling.profile_id, sampling.served_model) == ("llamacpp-gemma3-4b", "gemma3:4b")
    assert "sampling distribution" in sampling.measures


def test_the_plan_records_each_arms_role_with_the_users_reason_and_what_it_supplies(plan):
    sampling, serving = plan.arms.arms["sampling"], plan.arms.arms["serving"]
    assert (serving.role, sampling.role) == ("primary", "probe")
    for arm in (sampling, serving):
        assert arm.role_reason == plan_module.ROLE_REASON
        assert "we don't have 44h of time. so modal it is" in arm.role_reason
        assert "1-2h local is fine" in arm.role_reason and "2026-11-10" in arm.role_reason
    assert "pre-registered tests" in serving.supplies and "achieved power" in serving.supplies
    assert "full evaluation set" in serving.supplies
    assert "convergence" in sampling.supplies and "full_distribution" in sampling.supplies
    assert "probe evaluation months" in sampling.supplies and "pilot and power gate" in sampling.supplies
    payload = plan.to_dict()["arms"]["arms"]
    assert (payload["serving"]["role"], payload["sampling"]["role"]) == ("primary", "probe")


def test_each_arm_freezes_its_gguf_digest_and_its_fit_and_evidence_settings(plan):
    sampling, serving = plan.arms.arms["sampling"], plan.arms.arms["serving"]
    assert sampling.artifact_digest == "sha256:f06e91e18fc74295d0c781bf91622508c75655e2c18ed869d52447ffe1e667e1"
    assert serving.artifact_digest is None  # the serving weights are pinned by the manifest digest
    for arm in (sampling, serving):
        assert (arm.fit.min_auc, arm.fit.min_valid) == (0.6, 50)
    assert (sampling.evidence.timeout_s, sampling.evidence.top_k_keep, sampling.evidence.host_busy_retries,
            sampling.evidence.host_busy_wait_s) == (600.0, 100, 5, 60.0)
    assert serving.evidence is None  # no full-distribution source on the serving arm
    payload = plan.to_dict()["arms"]["arms"]
    assert payload["sampling"]["fit"] == {"min_auc": 0.6, "min_valid": 50}
    assert payload["sampling"]["evidence"] == {"timeout_s": 600.0, "top_k_keep": 100, "host_busy_retries": 5,
                                               "host_busy_wait_s": 60.0}
    assert payload["sampling"]["artifact_digest"] == sampling.artifact_digest and payload["serving"]["evidence"] is None


@pytest.mark.parametrize(("arm", "changes", "message"), [
    ("sampling", {"artifact_digest": None}, "artifact_digest"),
    ("sampling", {"artifact_digest": "f06e91e18fc74295d0c781bf91622508c75655e2c18ed869d52447ffe1e667e1"},
     "artifact_digest"),
    ("sampling", {"evidence": None}, "evidence"),
    ("serving", {"evidence": plan_module.EvidencePlan()}, "evidence"),
    ("sampling", {"calibration_draws": {"macro": 8}}, "calibration_draws"),
    ("sampling", {"calibration_draws": {"macro": 8, "edgar": 0}}, "calibration_draws"),
    ("sampling", {"role": "confirmatory"}, "role"),
    ("sampling", {"supplies": ""}, "supplies"),
    ("sampling", {"fit": {"min_auc": 0.6, "min_valid": 50}}, "fit"),
])
def test_the_new_arm_fields_are_validated(plan, arm, changes, message):
    with pytest.raises((TypeError, ValueError), match=message):
        replace(plan.arms.arms[arm], **changes)


@pytest.mark.parametrize("changes", [
    {"min_auc": 1.5}, {"min_auc": float("nan")}, {"min_valid": 0}, {"min_valid": True},
])
def test_fit_settings_validate_their_values(changes):
    with pytest.raises((TypeError, ValueError)):
        plan_module.FitPlan(**changes)


@pytest.mark.parametrize("changes", [
    {"timeout_s": 0}, {"timeout_s": float("inf")}, {"top_k_keep": 0}, {"host_busy_retries": -1},
    {"host_busy_wait_s": -1.0}, {"host_busy_retries": True},
])
def test_evidence_settings_validate_their_values(changes):
    with pytest.raises((TypeError, ValueError)):
        plan_module.EvidencePlan(**changes)


def test_the_roles_are_fixed_the_serving_arm_primary_and_the_sampling_arm_probe(plan):
    with pytest.raises(ValueError, match="probe"):
        replace(plan.arms, arms={**plan.arms.arms, "sampling": replace(plan.arms.arms["sampling"], role="primary")})
    with pytest.raises(ValueError, match="primary"):
        replace(plan.arms, arms={**plan.arms.arms, "serving": replace(plan.arms.arms["serving"], role="probe")})


def test_the_serving_arm_freezes_no_adjustment_and_top_k_and_lists_its_full_distribution_variants_as_missing(plan):
    serving = plan.arms.arms["serving"]
    assert serving.variants == SERVING_FOUR
    assert set(serving.missing_variants) == {"single_full_distribution", "ensemble_full_distribution"}
    for reason in serving.missing_variants.values():
        assert "different weights" in reason and "9.4" in reason
    assert (serving.temperature, serving.seed_base, serving.gated) == (0.0, None, False)
    assert serving.estimate_sources == ("top_k",)
    assert (serving.profile_id, serving.served_model) == ("modal-gemma3-4b", "google/gemma-3-4b-it")
    assert (serving.draws, serving.min_parsed, serving.top_logprobs, serving.max_tokens) == (32, 24, 5, 256)


def test_the_plan_freezes_an_explicit_arm_to_registry_id_map_and_the_single_draw_index(plan):
    payload = plan.to_dict()["arms"]
    assert dict(plan.arms.registry_ids) == payload["registry_ids"] == {"sampling": GEMMA, "serving": GEMMA}
    assert payload["variants"] == list(SIX) and payload["single_draw_index"] == 0
    assert "draw index 0" in payload["rule"] and "one recorded ensemble per item and rendering" in payload["rule"]
    assert payload["arms"]["serving"]["seed_base"] is None


def test_variants_follow_the_offered_estimate_sources():
    assert variants_for(("top_k", "full_distribution")) == SIX
    assert variants_for(("top_k",)) == SERVING_FOUR
    assert plan_module.variant_parts("single_full_distribution") == ("single", "full_distribution")
    assert plan_module.variant_parts("ensemble_none") == ("ensemble", "none")


def test_an_arm_whose_model_is_excluded_or_not_in_the_split_is_refused():
    with pytest.raises(ValueError, match="arm 'serving' has no included model"):
        full_plan(arms=study_arms(registry_ids={"sampling": GEMMA, "serving": "vendor/no-cutoff"}),
                  excluded_models={"vendor/no-cutoff": "no sourced cutoff"})
    with pytest.raises(ValueError, match="not an included model"):
        full_plan(arms=study_arms(registry_ids={"sampling": GEMMA, "serving": "vendor/unknown"}))


@pytest.mark.parametrize(("arm", "changes", "message"), [
    ("sampling", {"estimate_sources": ("top_k",)}, "variants"),
    ("serving", {"variants": SIX, "missing_variants": {}}, "variants"),
    ("sampling", {"seed_base": None}, "seed_base"),
    ("serving", {"seed_base": 7}, "seed_base"),
    ("sampling", {"seed_base": 2**31 - 5}, "seed_base"),
    ("sampling", {"seed_base": -1}, "seed_base"),
    ("sampling", {"min_parsed": 33}, "min_parsed"),
    ("sampling", {"draws": 1, "min_parsed": 1}, "draws"),
    ("sampling", {"temperature": -0.1}, "temperature"),
    ("serving", {"missing_variants": {"single_full_distribution": "different weights"}}, "missing_variants"),
    ("serving", {"missing_variants": {"single_full_distribution": "x", "ensemble_full_distribution": ""}},
     "missing_variants"),
    ("sampling", {"estimate_sources": ("full_distribution", "top_k")}, "order"),
    ("sampling", {"gated": 1}, "gated"),
])
def test_an_arm_plan_validates_its_settings_and_variant_set(plan, arm, changes, message):
    with pytest.raises((TypeError, ValueError), match=message):
        replace(plan.arms.arms[arm], **changes)


@pytest.mark.parametrize(("arm", "changes"), [
    ("sampling", {"temperature": 0.0, "seed_base": None}),
    ("sampling", {"gated": False}),
    ("serving", {"gated": True}),
    ("serving", {"temperature": 0.7, "seed_base": 5}),
])
def test_the_sampling_arm_must_sample_and_be_gated_and_the_serving_arm_must_not(plan, arm, changes):
    with pytest.raises(ValueError, match="arm"):
        replace(plan.arms, arms={**plan.arms.arms, arm: replace(plan.arms.arms[arm], **changes)})


# ---------- request and token limits: one ensemble per item and rendering ----------

def test_sampling_request_limits_match_one_recorded_ensemble_per_item_and_rendering_never_one_per_variant(plan):
    split, edgar = plan.months, plan.edgar
    sampling = stages(plan, "sampling")
    assert tuple(sampling) == ("calibration", "pilot", "full_run", "negative_control")
    ensembles = {
        "calibration": {"macro": len(split.calibration) * 2,
                        "edgar": len(edgar.calibration_memorized) + len(edgar.calibration_control)},
        "pilot": {"macro": len(split.pilot) * 2},
        "full_run": {"macro": len(split.probe_evaluation) * 2},
        "negative_control": {"edgar": len(edgar.negative_control)},
    }
    # The probe arm: 8 draws per rendering everywhere except the EDGAR calibration rows, which take 4.
    group_draws = {"calibration": {"macro": 8, "edgar": 4}, "pilot": {"macro": 8}, "full_run": {"macro": 8},
                   "negative_control": {"edgar": 8}}
    for name, stage in sampling.items():
        assert dict(stage.ensembles) == ensembles[name] and stage.draws == 8
        assert stage.group_draws() == group_draws[name]
        assert stage.max_requests == sum(count * group_draws[name][group] for group, count in ensembles[name].items())
        assert stage.max_output_tokens == stage.max_requests * 512
        assert stage.extra_requests == 0 and stage.max_usd is None
    assert dict(sampling["calibration"].draws_by_group) == {"edgar": 4}
    assert [stage.max_requests for stage in sampling.values()] == [48 * 8 + 140 * 4, 24 * 8, 40 * 8, 40 * 8]
    assert [stage.max_requests for stage in sampling.values()] == [944, 192, 320, 320]
    assert [stage.max_output_tokens for stage in sampling.values()] == [483328, 98304, 163840, 163840]
    # One ensemble feeds all six variants, so six variants never multiply the requests.
    assert sum(stage.max_requests for stage in sampling.values()) == 1776


def test_the_probe_arm_runs_its_frozen_probe_months_and_the_primary_arm_every_evaluation_month(plan):
    assert plan_module.full_run_months(plan, "sampling") == plan.months.probe_evaluation
    assert plan_module.full_run_months(plan, "serving") == plan.months.evaluation
    assert len(plan.months.probe_evaluation) == 20 and set(plan.months.probe_evaluation) < set(plan.months.evaluation)
    assert dict(stages(plan, "serving")["full_run"].ensembles) == {"macro": 180}
    assert dict(stages(plan, "sampling")["full_run"].ensembles) == {"macro": 40}


def test_every_probe_stage_fits_the_bound_of_two_and_a_half_hours(plan):
    assert plan.limits.probe_stage_max_seconds == 9000
    sampling = stages(plan, "sampling")
    # ceil(requests x 6 x 1.25) + 300: calibration 7,080 + 300, pilot 1,440 + 300, the others 2,400 + 300.
    assert [stage.max_seconds for stage in sampling.values()] == [7380, 1740, 2700, 2700]
    assert all(stage.max_seconds <= 9000 for stage in sampling.values())
    assert sum(stage.max_seconds for stage in sampling.values()) == 14520
    rule = plan.limits.rule
    assert "probe_stage_max_seconds" in rule and "probe_time_headroom" in rule and "probe_request_timeout" in rule


def test_a_probe_stage_above_its_bound_is_refused(plan):
    with pytest.raises(ValueError, match="probe_stage_max_seconds"):
        size_limits(plan.months, plan.edgar, plan.arms, plan.pilot, plan.limits.manifest,
                    sampling_seconds_per_generation=8)  # calibration: ceil(944 x 8 x 1.25) + 300 = 9740 s
    with pytest.raises(ValueError, match="probe_stage_max_seconds"):
        size_limits(plan.months, plan.edgar, plan.arms, plan.pilot, plan.limits.manifest,
                    probe_stage_max_seconds=5000)


def test_a_stage_budget_validates_its_per_group_draws(plan):
    calibration = stages(plan, "sampling")["calibration"]
    with pytest.raises(ValueError, match="draws_by_group"):
        replace(calibration, draws_by_group={"edgar": 8})  # the same as draws: not an override
    with pytest.raises(ValueError, match="draws_by_group"):
        replace(calibration, draws_by_group={"smoke": 4})
    with pytest.raises(ValueError, match="max_requests"):
        replace(calibration, draws_by_group={})


def test_serving_request_and_token_limits_follow_the_same_ensembles_plus_one_eligibility_post(plan):
    serving = stages(plan, "serving")
    assert tuple(serving) == STAGES
    assert [stage.max_requests for stage in serving.values()] == [0, 1, 32, 6016, 5760, 1280]
    assert serving["deployment"].extra_requests == 1 and dict(serving["deployment"].ensembles) == {}
    assert dict(serving["calibration"].ensembles) == {"edgar": 140, "macro": 48}
    assert dict(serving["smoke_run"].ensembles) == {"smoke": 1}
    for stage in serving.values():
        assert stage.max_output_tokens == stage.max_requests * 256
    assert (serving["weight_staging"].max_requests, serving["weight_staging"].max_output_tokens) == (0, 0)


def test_full_distribution_evidence_calls_do_not_count_against_max_requests_and_the_plan_says_so(plan):
    rule = plan.to_dict()["limits"]["evidence_requests"]
    assert "do not count against max_requests" in rule and "own timeout" in rule
    sampling = stages(plan, "sampling")
    assert sampling["full_run"].max_requests == len(plan.months.probe_evaluation) * 2 * 8


def test_sampling_time_limits_use_the_frozen_seconds_per_generation_with_headroom_and_one_timeout(plan):
    limits = plan.limits
    for stage in limits.stages["sampling"]:
        expected = math.ceil(stage.max_requests * limits.sampling_seconds_per_generation * limits.probe_time_headroom)
        assert stage.max_seconds == expected + limits.probe_request_timeout_seconds
    # The design's planning rate of 6 s per generation on the iGPU (user decision 2026-11-10), a headroom of 1.25
    # and the arm's 300 s client timeout, so the last admission (elapsed plus timeout) still fits.
    assert (limits.sampling_seconds_per_generation, limits.probe_time_headroom,
            limits.probe_request_timeout_seconds) == (6, 1.25, 300)
    payload = plan.to_dict()["limits"]
    assert (payload["probe_time_headroom"], payload["probe_request_timeout_seconds"]) == (1.25, 300)


@pytest.mark.parametrize(("changes", "message"), [
    ({"probe_time_headroom": 0.9}, "probe_time_headroom"), ({"probe_time_headroom": float("nan")}, "probe_time_headroom"),
    ({"probe_request_timeout_seconds": 0}, "probe_request_timeout_seconds"),
])
def test_the_probe_time_headroom_and_timeout_are_validated(plan, changes, message):
    with pytest.raises((TypeError, ValueError), match=message):
        size_limits(plan.months, plan.edgar, plan.arms, plan.pilot, plan.limits.manifest, **changes)


def test_every_sampling_stage_converts_to_the_ledger_limits_of_task_7_5(plan):
    for stage in plan.limits.stages["sampling"]:
        converted = study_limits.StudyLimits(plan.plan_sha256, "sampling", stage.stage, stage.max_requests,
                                             stage.max_output_tokens, stage.max_seconds, stage.max_usd)
        assert converted.max_usd is None


# ---------- serving-arm stage limits (time and cost), in execution order with the manifest hash ----------

def test_every_serving_stage_limit_is_present_in_execution_order_with_the_manifest_hash(plan):
    limits = plan.limits
    assert tuple(stage.stage for stage in limits.stages["serving"]) == STAGES == plan_module.SERVING_STAGES
    assert limits.manifest.path == "configs/modal/gemma-3-4b-measurement.json"
    assert limits.manifest.sha256 == hashlib.sha256(MANIFEST_BYTES).hexdigest()
    assert plan.to_dict()["limits"]["manifest"]["sha256"] == limits.manifest.sha256
    for stage in limits.stages["serving"]:
        assert stage.max_seconds >= 1 and stage.max_usd > 0


def test_serving_time_and_cost_limits_are_derived_from_the_manifest_cost_rates(plan):
    modal = load_modal_plan(MANIFEST_PATH)
    cost, manifest_limits = MANIFEST_JSON["cost"], MANIFEST_JSON["limits"]
    rate = compute_rate(cost, modal.resources)
    controller = CONTROLLER_CORES * cost["physical_core_usd_per_second"] + CONTROLLER_GIB * cost[
        "memory_gib_usd_per_second"]
    seconds_per_post = 6  # (86,142 - 5 x 600) / 13,857, the manifest's worst-case post
    assert plan.limits.serving_seconds_per_post == seconds_per_post
    serving = stages(plan, "serving")
    for name in STAGES[1:]:
        stage = serving[name]
        assert stage.max_seconds == manifest_limits["startup_seconds"] + stage.max_requests * seconds_per_post
        floor = (rate * (stage.max_seconds + manifest_limits["cleanup_seconds"])
                 + controller * (stage.max_seconds + CONTROLLER_EXTRA_SECONDS))
        assert stage.max_usd >= floor and stage.max_usd == usd_up(floor)
    staging = serving["weight_staging"]
    cpu_rate = (modal.resources.cpu_physical_cores * cost["physical_core_usd_per_second"]
                + modal.resources.memory_gib * cost["memory_gib_usd_per_second"])
    assert staging.max_seconds == 3600 and staging.max_usd == usd_up(cpu_rate * 3600)
    assert [serving[name].max_seconds for name in STAGES] == [3600, 606, 792, 36696, 35160, 8280]
    assert [serving[name].max_usd for name in STAGES] == [0.32, 0.29, 0.37, 14.58, 13.97, 3.33]


def test_the_plan_controller_constants_are_those_of_measure_modal_model():
    assert (plan_module.CONTROLLER_CORES, plan_module.CONTROLLER_MEMORY_GIB, plan_module.CONTROLLER_ALLOWANCE_SECONDS) \
        == (CONTROLLER_CORES, CONTROLLER_GIB, CONTROLLER_EXTRA_SECONDS)


def test_the_serving_stages_fit_five_gpu_lifecycles_within_the_manifest_ceiling(plan):
    serving = plan.limits.stages["serving"]
    gpu = serving[1:]
    assert len(gpu) == 5
    assert sum(stage.max_requests for stage in serving) == 13089 <= 13857
    assert sum(stage.max_output_tokens for stage in serving) == 13089 * 256 <= 3547392
    assert sum(stage.max_seconds for stage in gpu) == 81534 <= 86142
    assert float(sum((Decimal(str(stage.max_usd)) for stage in gpu), Decimal(0))) == 32.54 <= 34.34


def test_the_stage_limits_files_carry_the_plan_hash_and_pass_the_deploy_validation(plan):
    modal = load_modal_plan(MANIFEST_PATH)
    files = stage_limits_files(plan)
    assert tuple(stage for stage, _ in files) == STAGES
    for (stage, data), expected in zip(files, plan.limits.stages["serving"], strict=True):
        document = json.loads(data)
        assert document == {"plan_sha256": plan.plan_sha256, "stage": stage, "max_requests": expected.max_requests,
                            "max_output_tokens": expected.max_output_tokens, "max_seconds": expected.max_seconds,
                            "max_usd": expected.max_usd}
        limits = parse_stage_limits(data, modal)
        if stage == "weight_staging":
            assert require_staging_limits(limits, modal, MANIFEST_JSON["cost"]) <= limits.max_usd
        else:
            require_serving_limits(limits)


def test_the_frozen_serving_limits_pass_the_task_7_5_presentation_and_its_ceiling_check(plan, tmp_path):
    paths = []
    for stage, data in stage_limits_files(plan):
        path = tmp_path / f"{stage}.json"
        path.write_bytes(data)
        paths.append(path)
    presentation = study_limits.present_serving_arm(plan, MANIFEST_PATH, paths)
    record = serving_limits_record(plan)
    assert presentation.plan_sha256 == record["plan_sha256"] == plan.plan_sha256
    assert presentation.manifest_sha256 == record["manifest_sha256"] == plan.limits.manifest.sha256
    assert presentation.stage_limits_sha256 == record["stage_limits_sha256"]
    assert presentation.max_usd_total == record["max_usd_total"] == 32.86
    assert [stage.sha256 for stage in presentation.stages] == [stage["sha256"] for stage in record["stages"]]
    assert [stage["stage"] for stage in record["stages"]] == list(STAGES)
    study_limits._check_arm_ceiling(presentation.stages, load_modal_plan(MANIFEST_PATH))
    # The ledger's cost horizon (elapsed plus timeout, at the configured rate) never exceeds a GPU stage's max_usd.
    for stage in presentation.stages[1:]:
        assert presentation.usd_per_second * stage.max_seconds <= stage.max_usd


def test_serving_limits_above_the_manifest_ceiling_are_refused(plan):
    small = replace(plan.limits.manifest, inference_posts=1000)
    with pytest.raises(ValueError, match="ceiling"):
        size_limits(plan.months, plan.edgar, plan.arms, plan.pilot, small)
    cheap = replace(plan.limits.manifest, target_usd=10.0)
    with pytest.raises(ValueError, match="ceiling"):
        size_limits(plan.months, plan.edgar, plan.arms, plan.pilot, cheap)


def test_a_serving_arm_that_asks_for_more_tokens_or_logprobs_than_the_manifest_serves_is_refused(plan):
    for changes in ({"max_tokens": 512}, {"top_logprobs": 10}):
        arms = replace(plan.arms, arms={**plan.arms.arms, "serving": replace(plan.arms.arms["serving"], **changes)})
        with pytest.raises(ValueError, match="manifest"):
            size_limits(plan.months, plan.edgar, arms, plan.pilot, plan.limits.manifest)


def test_the_serving_manifest_record_is_read_from_the_manifest_file():
    manifest = plan_module.load_serving_manifest()
    resources, limits, cost = MANIFEST_JSON["resources"], MANIFEST_JSON["limits"], MANIFEST_JSON["cost"]
    assert (manifest.gpu_count, manifest.cpu_physical_cores, manifest.memory_gib) == (
        resources["gpu_count"], resources["cpu_physical_cores"], resources["memory_gib"])
    assert (manifest.inference_posts, manifest.total_output_tokens, manifest.gpu_phase_seconds) == (
        limits["inference_posts"], limits["total_output_tokens"], limits["gpu_phase_seconds"])
    assert (manifest.gpu_usd_per_second, manifest.target_usd) == (cost["gpu_usd_per_second"], cost["target_usd"])
    assert manifest.pricing_checked_at == cost["pricing_checked_at"]


# ---------- pilot settings ----------

def test_the_pilot_freezes_12_months_times_8_draws_per_rendering_with_its_own_minimum(plan):
    pilot = plan.pilot
    assert (pilot.arm, pilot.months, pilot.draws, pilot.min_parsed) == ("sampling", 12, 8, 6)
    assert pilot.min_parsed <= pilot.draws
    assert pilot.months == len(plan.months.pilot)
    assert pilot.renderings == ("identifying", "anonymised") and pilot.noise_variant == "ensemble_top_k"
    # The gate simulates the power of the gated (probe) arm's own full run: its 15 and 5 probe evaluation months.
    assert (pilot.pre_cutoff_evaluation_months, pilot.post_cutoff_evaluation_months) == (15, 5)
    assert "excluded from the confirmatory analysis" in pilot.rule and "probe evaluation months" in pilot.rule


def test_the_pilot_freezes_its_gate_policy_inform_so_the_probe_arm_proceeds_with_its_expected_power(plan):
    assert plan.pilot.gate_policy == plan_module.PILOT_GATE_POLICY == "inform"
    assert plan_module.PILOT_GATE_POLICIES == ("inform", "block")
    assert plan.to_dict()["pilot"]["gate_policy"] == "inform"
    assert replace(plan.pilot, gate_policy="block").gate_policy == "block"
    with pytest.raises(ValueError, match="gate_policy"):
        replace(plan.pilot, gate_policy="warn")


def test_the_evaluation_month_counts_of_each_arm_follow_its_full_run_months(plan):
    assert plan_module.evaluation_month_counts(plan, "sampling") == (15, 5) == (
        plan.pilot.pre_cutoff_evaluation_months, plan.pilot.post_cutoff_evaluation_months)
    # The primary arm's achieved-power report uses every evaluation month.
    assert plan_module.evaluation_month_counts(plan, "serving") == (68, 22)
    with pytest.raises(ValueError, match="arm"):
        plan_module.evaluation_month_counts(plan, "local")


@pytest.mark.parametrize("changes", [
    {"min_parsed": 9}, {"min_parsed": 0}, {"draws": 0}, {"months": 0}, {"renderings": ("identifying",)},
    {"noise_variant": "ensemble_unknown"}, {"arm": "other"}, {"pre_cutoff_evaluation_months": 0},
])
def test_pilot_settings_validate_their_values(plan, changes):
    with pytest.raises((TypeError, ValueError)):
        replace(plan.pilot, **changes)


@pytest.mark.parametrize("changes", [
    {"months": 11}, {"draws": 33, "min_parsed": 6}, {"pre_cutoff_evaluation_months": 67},
    {"post_cutoff_evaluation_months": 23}, {"arm": "serving"},
])
def test_a_pilot_that_does_not_match_the_split_or_its_arm_is_refused(plan, changes):
    with pytest.raises(ValueError, match="pilot"):
        replace(plan, pilot=replace(plan.pilot, **changes))


# ---------- scored probability (8.4) ----------

def test_the_scored_probability_mapping_is_frozen_with_its_outcome_clip_and_shrink(plan):
    payload = plan.to_dict()["scored_probability"]
    assert payload["clip"] == [0.01, 0.99]
    assert set(payload["mapping"]) == {"ensemble", "single"}
    assert "share of the parsed draws" in payload["mapping"]["ensemble"]
    assert "draw index 0" in payload["mapping"]["single"]
    assert "defensive pair beats the risk pair" in payload["outcome"] and "DTB3" in payload["outcome"]
    assert "0.5 + (1 - clip(p_memorized, 0, 1)) * (p_raw - 0.5)" in payload["adjustment"]
    assert "clip" in payload["adjustment"].split(";")[0]
    assert set(payload["losses"]) == {"brier", "log_loss"} and "natural" in payload["losses"]["log_loss"]
    assert payload["sign_decimals"] == 9


def test_the_scored_probability_maps_each_draw_mode_and_clips(plan):
    scored = plan.scored_probability
    assert scored.probability([0.4] * 24 + [-0.3] * 8, mode="ensemble") == 0.75
    assert scored.probability([0.4] * 32, mode="ensemble") == 0.99
    assert scored.probability([-0.2] * 32, mode="ensemble") == 0.01
    # A spread that is zero after rounding to nine decimals counts as risk-on, as in is_correct.
    assert scored.probability([1e-12] * 16 + [0.2] * 16, mode="ensemble") == 0.5
    assert scored.probability([0.3], mode="single") == 0.99
    assert scored.probability([-0.3], mode="single") == 0.01
    assert scored.probability([0.0], mode="single") == 0.01


def test_the_scored_probability_shrinks_toward_one_half_after_clipping(plan):
    scored = plan.scored_probability
    assert scored.adjusted(0.75, 0.4) == pytest.approx(0.5 + 0.6 * 0.25)
    assert scored.adjusted(0.99, 1.0) == 0.5 and scored.adjusted(0.99, 1.7) == 0.5
    assert scored.adjusted(0.01, 0.0) == 0.01 and scored.adjusted(0.01, -0.2) == 0.01
    assert scored.adjusted(0.75, None) == 0.75
    # Clip first, then shrink: a unanimous ensemble with p_memorized 0.01 gives 0.5 + 0.99 * 0.49.
    assert scored.adjusted(scored.probability([0.4] * 32, mode="ensemble"), 0.01) == pytest.approx(0.9851)


def test_the_scored_outcome_is_the_defensive_pair_beating_the_risk_pair(plan):
    scored = plan.scored_probability
    defensive = RealizedReturns(world_equity=0.01, tech_sector=0.02, gold_commodity=0.03, short_treasury_cash=0.001)
    risk_on = RealizedReturns(world_equity=0.03, tech_sector=0.02, gold_commodity=0.01, short_treasury_cash=0.001)
    tie = RealizedReturns(world_equity=0.01, tech_sector=0.01, gold_commodity=0.019, short_treasury_cash=0.001)
    assert [scored.outcome_of(realized) for realized in (defensive, risk_on, tie)] == [1, 0, 0]
    for realized in (defensive, risk_on, tie):
        assert scored.outcome_of(realized) == int(is_correct(0.5, realized))


@pytest.mark.parametrize(("call", "error"), [
    (lambda s: s.probability([0.1, 0.2], mode="single"), ValueError),
    (lambda s: s.probability([], mode="ensemble"), ValueError),
    (lambda s: s.probability([0.1], mode="half"), ValueError),
    (lambda s: s.probability([float("nan")], mode="ensemble"), ValueError),
    (lambda s: s.probability([True], mode="single"), TypeError),
    (lambda s: s.adjusted(1.0, 0.2), ValueError),
    (lambda s: s.adjusted(0.5, float("inf")), ValueError),
    (lambda s: s.outcome_of({"world_equity": 0.1}), TypeError),
])
def test_the_scored_probability_validates_its_inputs(plan, call, error):
    with pytest.raises(error):
        call(plan.scored_probability)


@pytest.mark.parametrize("changes", [
    {"clip": (0.6, 0.9)}, {"clip": (0.0, 0.99)}, {"clip": (0.01,)}, {"mapping": {"ensemble": "x"}},
    {"losses": {"brier": "x"}}, {"sign_decimals": -1}, {"outcome": ""},
])
def test_scored_probability_settings_validate_their_values(changes):
    with pytest.raises((TypeError, ValueError)):
        ScoredProbability(**changes)


# ---------- hashes, completeness and changed-plan refusal ----------

def test_the_full_plan_gives_each_7_3_section_a_content_hash(plan):
    payload = plan.to_dict()
    for name in ("arms", "pilot", "limits", "scored_probability"):
        assert payload["content_sha256"][name] == digest(canonical(payload[name]))
    body = {key: value for key, value in payload.items() if key != "plan_sha256"}
    assert payload["plan_sha256"] == plan.plan_sha256 == digest(canonical(body))
    json.dumps(payload, allow_nan=False)


def test_a_plan_without_the_7_3_sections_keeps_its_7_2_hash_and_is_not_complete(plan):
    partial = freeze_plan(classes({GEMMA: GEMMA_CUTOFF}), edgar_rows(), seed=SEED, created_at=CREATED_AT,
                          items=ITEMS, edgar_files=EDGAR_FILES)
    assert set(partial.to_dict()) == set(plan.to_dict()) - {"arms", "pilot", "limits", "scored_probability"}
    assert partial.plan_sha256 != plan.plan_sha256
    assert replace(plan, arms=None, pilot=None, limits=None, scored_probability=None) == partial
    with pytest.raises(ValueError, match="lacks"):
        require_complete(partial)
    assert require_complete(plan) is plan


@pytest.mark.parametrize("sections", [{"pilot": None}, {"arms": None, "pilot": None}, {"arms": None}])
def test_a_section_without_the_sections_it_depends_on_is_refused(plan, sections):
    with pytest.raises(ValueError, match="needs"):
        replace(plan, **sections)


def test_freeze_plan_needs_the_arms_and_the_serving_manifest_together():
    with pytest.raises(ValueError, match="serving manifest"):
        full_plan(serving_manifest=None)
    with pytest.raises(ValueError, match="arms"):
        full_plan(arms=None)


def test_a_written_full_plan_loads_back_identical(plan, tmp_path):
    path = write_plan(plan, tmp_path / "plan.json")
    loaded = load_plan(path, expected_sha256=plan.plan_sha256)
    assert loaded == plan and loaded.plan_sha256 == plan.plan_sha256
    assert require_complete(loaded).limits.stages["serving"] == plan.limits.stages["serving"]


def _changed_leaf(value):
    if value is None:
        return "x"
    if isinstance(value, bool):
        return not value
    if isinstance(value, int):
        return value + 1
    if isinstance(value, float):
        return value + 0.01
    if isinstance(value, str):
        return value + "x"
    if isinstance(value, dict):
        return {"extra": "x"}
    if isinstance(value, list):
        return ["extra"]
    raise AssertionError(f"unexpected leaf {value!r}")


def _refused(path: Path, payload: dict) -> bool:
    path.write_text(json.dumps(payload), encoding="utf-8")
    try:
        load_plan(path)
    except PlanChangedError:
        return True
    return False


def test_load_refuses_a_change_to_any_field_of_a_full_plan_file(plan, tmp_path):
    original = plan.to_dict()
    path = tmp_path / "plan.json"
    leaves = list(_leaf_paths(original))
    assert ("arms", "arms", "sampling", "temperature") in leaves and ("arms", "arms", "serving", "seed_base") in leaves
    assert ("limits", "manifest", "sha256") in leaves and ("scored_probability", "clip", 0) in leaves
    assert ("pilot", "min_parsed") in leaves and ("limits", "stages", "serving", 5, "max_usd") in leaves
    accepted = []
    for leaf in leaves:
        mutated = copy.deepcopy(original)
        parent = _at(mutated, leaf[:-1])
        parent[leaf[-1]] = _changed_leaf(parent[leaf[-1]])
        if not _refused(path, mutated):
            accepted.append(leaf)
    assert accepted == []


def test_load_refuses_a_removed_or_an_added_field_of_a_full_plan(plan, tmp_path):
    original = plan.to_dict()
    path = tmp_path / "plan.json"
    accepted = []
    for dict_path in _dict_paths(original):
        for key in _at(original, dict_path):
            mutated = copy.deepcopy(original)
            del _at(mutated, dict_path)[key]
            if not _refused(path, mutated):
                accepted.append(("removed", *dict_path, key))
        mutated = copy.deepcopy(original)
        _at(mutated, dict_path)["added_field"] = 1
        if not _refused(path, mutated):
            accepted.append(("added", *dict_path))
    assert accepted == []


def _forge_stage(payload):
    payload["limits"]["stages"]["serving"][4]["max_seconds"] += 1


def _forge_ensembles(payload):
    stage = payload["limits"]["stages"]["sampling"][2]
    stage["ensembles"]["macro"] += 1
    stage["max_requests"] += stage["draws"]


def _forge_group_draws(payload):
    stage = payload["limits"]["stages"]["sampling"][0]
    stage["draws_by_group"]["edgar"] = 3
    stage["max_requests"] -= 140


def _forge_digest(payload):
    payload["arms"]["arms"]["sampling"]["artifact_digest"] = "sha256:" + "0" * 63


def _forge_seconds(payload):
    payload["limits"]["serving_seconds_per_post"] = 5


def _forge_registry(payload):
    payload["arms"]["registry_ids"]["serving"] = "vendor/unknown"


def _forge_pilot(payload):
    payload["pilot"]["min_parsed"] = 9


def _forge_variants(payload):
    payload["arms"]["arms"]["serving"]["variants"] = list(SIX)
    payload["arms"]["arms"]["serving"]["missing_variants"] = {}


@pytest.mark.parametrize(("forge", "message"), [
    (_forge_stage, "sizing rule"), (_forge_ensembles, "sizing rule"), (_forge_seconds, "sizing rule"),
    (_forge_registry, "included model"), (_forge_pilot, "min_parsed"), (_forge_variants, "variants"),
    (_forge_group_draws, "sizing rule"), (_forge_digest, "artifact_digest"),
])
def test_load_validates_a_consistently_rehashed_full_plan(plan, tmp_path, forge, message):
    payload = plan.to_dict()
    forge(payload)
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(rehash(payload)), encoding="utf-8")
    with pytest.raises(ValueError, match=message) as raised:
        load_plan(path)
    assert not isinstance(raised.value, PlanChangedError)


# A valid change of every 7.3 field: each gives another plan, which ensure_unchanged refuses.
@pytest.fixture(scope="module")
def two_model_plan() -> StudyPlan:
    """Both arms on Gemma, with a second included model whose later cutoff leaves the split unchanged."""
    return full_plan(cutoffs={GEMMA: GEMMA_CUTOFF, LATER: "2025-06"})


def _resize(p: StudyPlan, *, arms: StudyArms | None = None, pilot: PilotSettings | None = None,
            manifest=None, **settings) -> StudyPlan:
    arms = p.arms if arms is None else arms
    pilot = p.pilot if pilot is None else pilot
    limits = size_limits(p.months, p.edgar, arms, pilot, p.limits.manifest if manifest is None else manifest,
                         **(p.limits.settings() | settings))
    return replace(p, arms=arms, pilot=pilot, limits=limits)


def _rearm(p: StudyPlan, arm: str, **changes) -> StudyPlan:
    return _resize(p, arms=replace(p.arms, arms={**p.arms.arms, arm: replace(p.arms.arms[arm], **changes)}))


def _remanifest(p: StudyPlan) -> StudyPlan:
    data = json.dumps(MANIFEST_JSON, indent=1).encode("utf-8")
    return _resize(p, manifest=plan_module.ServingManifest.from_manifest(p.limits.manifest.path, data))


SECTION_CHANGES = {
    "arms.registry_ids": lambda p: _resize(p, arms=replace(p.arms, registry_ids={"sampling": GEMMA, "serving": LATER})),
    "arms.rule": lambda p: replace(p, arms=replace(p.arms, rule=p.arms.rule + " (restated)")),
    "arms.arms.profile_id": lambda p: _rearm(p, "sampling", profile_id="llamacpp-gemma3-4b-q8"),
    "arms.arms.served_model": lambda p: _rearm(p, "sampling", served_model="gemma3:4b-q8"),
    "arms.arms.measures": lambda p: _rearm(p, "serving", measures="serving nondeterminism, restated"),
    "arms.arms.temperature": lambda p: _rearm(p, "sampling", temperature=0.8),
    "arms.arms.seed_base": lambda p: _rearm(p, "sampling", seed_base=2000),
    # Re-sized with the probe arm (8 draws, stages of at most two hours): draws 10, min_parsed 7, pilot draws 7.
    "arms.arms.draws": lambda p: _rearm(p, "sampling", draws=10),
    "arms.arms.min_parsed": lambda p: _rearm(p, "sampling", min_parsed=7),
    "arms.arms.top_logprobs": lambda p: _rearm(p, "sampling", top_logprobs=10),
    "arms.arms.max_tokens": lambda p: _rearm(p, "sampling", max_tokens=256),
    "arms.arms.missing_variants": lambda p: _rearm(p, "serving", missing_variants={
        name: reason + " (restated)" for name, reason in p.arms.arms["serving"].missing_variants.items()}),
    "arms.arms.role_reason": lambda p: _rearm(p, "serving", role_reason=p.arms.arms["serving"].role_reason + " (restated)"),
    "arms.arms.supplies": lambda p: _rearm(p, "sampling", supplies=p.arms.arms["sampling"].supplies + " (restated)"),
    "arms.arms.artifact_digest": lambda p: _rearm(p, "sampling", artifact_digest="sha256:" + "1" * 64),
    "arms.arms.calibration_draws": lambda p: _rearm(p, "sampling", calibration_draws={"macro": 8, "edgar": 3}),
    "arms.arms.fit.min_auc": lambda p: _rearm(p, "sampling", fit=plan_module.FitPlan(min_auc=0.65)),
    "arms.arms.fit.min_valid": lambda p: _rearm(p, "serving", fit=plan_module.FitPlan(min_valid=60)),
    "arms.arms.evidence.timeout_s": lambda p: _rearm(p, "sampling", evidence=plan_module.EvidencePlan(timeout_s=900.0)),
    "arms.arms.evidence.top_k_keep": lambda p: _rearm(p, "sampling", evidence=plan_module.EvidencePlan(top_k_keep=50)),
    "arms.arms.evidence.host_busy_retries": lambda p: _rearm(
        p, "sampling", evidence=plan_module.EvidencePlan(host_busy_retries=3)),
    "arms.arms.evidence.host_busy_wait_s": lambda p: _rearm(
        p, "sampling", evidence=plan_module.EvidencePlan(host_busy_wait_s=30.0)),
    "pilot.draws": lambda p: _resize(p, pilot=replace(p.pilot, draws=7)),
    "pilot.min_parsed": lambda p: replace(p, pilot=replace(p.pilot, min_parsed=7)),
    "pilot.noise_variant": lambda p: replace(p, pilot=replace(p.pilot, noise_variant="ensemble_full_distribution")),
    "pilot.rule": lambda p: replace(p, pilot=replace(p.pilot, rule=p.pilot.rule + " (restated)")),
    "pilot.gate_policy": lambda p: replace(p, pilot=replace(p.pilot, gate_policy="block")),
    "limits.manifest": _remanifest,
    "limits.sampling_seconds_per_generation": lambda p: _resize(p, sampling_seconds_per_generation=7),
    "limits.probe_stage_max_seconds": lambda p: _resize(p, probe_stage_max_seconds=8000),
    "limits.probe_time_headroom": lambda p: _resize(p, probe_time_headroom=1.2),
    "limits.probe_request_timeout_seconds": lambda p: _resize(p, probe_request_timeout_seconds=400),
    "limits.weight_staging_seconds": lambda p: _resize(p, weight_staging_seconds=1800),
    "limits.controller_cores": lambda p: _resize(p, controller_cores=0.25),
    "limits.controller_memory_gib": lambda p: _resize(p, controller_memory_gib=0.25),
    "limits.controller_allowance_seconds": lambda p: _resize(p, controller_allowance_seconds=200),
    "limits.evidence_requests": lambda p: replace(p, limits=replace(p.limits, evidence_requests="restated")),
    "limits.rule": lambda p: replace(p, limits=replace(p.limits, rule=p.limits.rule + " (restated)")),
    "scored_probability.outcome": lambda p: replace(p, scored_probability=replace(p.scored_probability,
                                                                                    outcome="restated")),
    "scored_probability.mapping": lambda p: replace(p, scored_probability=replace(
        p.scored_probability, mapping={"ensemble": "restated", "single": "restated"})),
    "scored_probability.clip": lambda p: replace(p, scored_probability=replace(p.scored_probability,
                                                                                 clip=(0.02, 0.98))),
    "scored_probability.adjustment": lambda p: replace(p, scored_probability=replace(p.scored_probability,
                                                                                       adjustment="restated")),
    "scored_probability.losses": lambda p: replace(p, scored_probability=replace(
        p.scored_probability, losses={"brier": "restated", "log_loss": "restated"})),
    "scored_probability.missing": lambda p: replace(p, scored_probability=replace(p.scored_probability,
                                                                                    missing="restated")),
    "scored_probability.sign_decimals": lambda p: replace(p, scored_probability=replace(p.scored_probability,
                                                                                          sign_decimals=8)),
}

# Fields fixed by the design or derived from other fields: any change is refused when the plan is built.
FIXED_CHANGES = {
    "arms.variants": lambda p: replace(p.arms, variants=SIX[::-1]),
    "arms.single_draw_index": lambda p: replace(p.arms, single_draw_index=1),
    "arms.arms.estimate_sources": lambda p: replace(p.arms, arms={**p.arms.arms, "sampling": replace(
        p.arms.arms["sampling"], estimate_sources=("top_k",), variants=SERVING_FOUR,
        missing_variants={"single_full_distribution": "x", "ensemble_full_distribution": "x"})}),
    "arms.arms.variants": lambda p: replace(p.arms.arms["serving"], variants=SERVING_FOUR[:3]),
    "arms.arms.gated": lambda p: replace(p.arms, arms={**p.arms.arms, "serving": replace(
        p.arms.arms["serving"], gated=True)}),
    "arms.arms.role": lambda p: replace(p.arms, arms={**p.arms.arms, "sampling": replace(
        p.arms.arms["sampling"], role="primary")}),
    "pilot.pre_cutoff_evaluation_months": lambda p: replace(p, pilot=replace(p.pilot, pre_cutoff_evaluation_months=1)),
    "pilot.post_cutoff_evaluation_months": lambda p: replace(p, pilot=replace(p.pilot,
                                                                              post_cutoff_evaluation_months=1)),
    "pilot.arm": lambda p: replace(p, pilot=replace(p.pilot, arm="serving")),
    "pilot.months": lambda p: replace(p, pilot=replace(p.pilot, months=13)),
    "pilot.renderings": lambda p: replace(p.pilot, renderings=("anonymised", "identifying")),
    "limits.stages": lambda p: replace(p, limits=replace(p.limits, stages={
        **p.limits.stages, "sampling": p.limits.stages["sampling"][::-1]})),
    "limits.serving_seconds_per_post": lambda p: replace(p, limits=replace(p.limits, serving_seconds_per_post=7)),
}


def test_the_7_3_change_tables_cover_every_field_of_every_7_3_section():
    expected = {f"arms.{f.name}" for f in fields(StudyArms) if f.name != "arms"}
    expected |= {f"arms.arms.{f.name}" for f in fields(ArmPlan) if f.name not in ("fit", "evidence")}
    expected |= {f"arms.arms.fit.{f.name}" for f in fields(plan_module.FitPlan)}
    expected |= {f"arms.arms.evidence.{f.name}" for f in fields(plan_module.EvidencePlan)}
    expected |= {f"pilot.{f.name}" for f in fields(PilotSettings)}
    expected |= {f"limits.{f.name}" for f in fields(PlanLimits)}
    expected |= {f"scored_probability.{f.name}" for f in fields(ScoredProbability)}
    assert not set(SECTION_CHANGES) & set(FIXED_CHANGES)
    assert set(SECTION_CHANGES) | set(FIXED_CHANGES) == expected


@pytest.mark.parametrize("name", sorted(SECTION_CHANGES))
def test_ensure_unchanged_refuses_a_plan_after_any_7_3_field_changes(two_model_plan, name):
    ensure_unchanged(two_model_plan, two_model_plan.plan_sha256)
    changed = SECTION_CHANGES[name](two_model_plan)
    assert isinstance(changed, StudyPlan) and changed.plan_sha256 != two_model_plan.plan_sha256
    with pytest.raises(PlanChangedError):
        ensure_unchanged(changed, two_model_plan.plan_sha256)


@pytest.mark.parametrize("name", sorted(FIXED_CHANGES))
def test_a_change_to_a_fixed_or_derived_7_3_field_is_refused_when_built(two_model_plan, name):
    with pytest.raises(ValueError):
        FIXED_CHANGES[name](two_model_plan)


def test_serving_limits_record_needs_a_complete_plan():
    partial = freeze_plan(classes({GEMMA: GEMMA_CUTOFF}), edgar_rows(), seed=SEED, created_at=CREATED_AT,
                          items=ITEMS, edgar_files=EDGAR_FILES)
    with pytest.raises(ValueError, match="lacks"):
        stage_limits_files(partial)
    with pytest.raises(ValueError, match="lacks"):
        serving_limits_record(partial)
