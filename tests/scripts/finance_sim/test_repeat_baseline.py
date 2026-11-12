"""Repeat-the-prompt baseline (qualified-model-catalogue task 3.3; Requirements 5.6 and 6.1 to 6.4).

Recorded draws of both arms are built by hand under the small frozen plan of the
convergence tests (4 draws per ensemble). Files under ``tmp_path`` only; no
network and no model.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import socket
from statistics import median

import pytest

import scripts.finance_sim.repeat_baseline as repeat_baseline_module
from recall_guard.core.completion import ModelResponse
from recall_guard.core.consensus import wilson_interval
from recall_guard.core.distribution import RecordedDraw
from recall_guard.core.significance import Statistic
from scripts.finance_sim.execute import INTERRUPTED, ExecutionDraw, planned_keys
from scripts.finance_sim.plan import PlanChangedError, write_plan
from scripts.finance_sim.recording import DrawLog, draw_log_path
from scripts.finance_sim.repeat_baseline import (
    BASELINE_SCHEMA,
    NO_DRAWS,
    NO_PROMPTS,
    STAGE,
    RepeatBaselineError,
    baseline_path,
    main,
    repeat_baseline,
    repeat_baseline_from_records,
    write_baseline,
)
from tests.scripts.finance_sim.test_convergence import DRAWS, PLAN, frozen_plan
from tests.scripts.finance_sim.test_recording_draws import CUT, aged, unchanged, writing_calls

SAMPLING, SERVING = "sampling", "serving"
PROVIDERS = {SAMPLING: "llama.cpp", SERVING: "modal"}
MARKER = "PRIVATE-REPLY"
A = {"equity": 0.3172, "gold": 0.5}
B = {"equity": 0.4172, "gold": 0.5}
C = {"equity": 0.5172, "gold": 0.25}
PROSE = "prose"       # a reply that did not parse
LOST = "interrupted"  # a request whose outcome was lost: no reply


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("the repeat baseline attempted a network connection")

    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def draw(arm: str, key: tuple[str, str], index: int, answer) -> ExecutionDraw:
    """One recorded draw of ``key``: a parsed answer, an unparsed reply (PROSE) or a lost request (LOST)."""
    arm_plan = PLAN.arms.arms[arm]
    seed = None if arm_plan.seed_base is None else arm_plan.seed_base + index
    item_id, rendering = key
    if answer == LOST:
        text, parsed, failure, response = "", None, INTERRUPTED, None
    elif answer == PROSE:
        text, parsed, failure = f"{MARKER} no numbers here", None, "parse_error"
        response = ModelResponse(content=text)
    else:
        text, parsed, failure = f"{MARKER} {json.dumps(answer)}", answer, None
        response = ModelResponse(content=text)
    recorded = RecordedDraw(draw_index=index, temperature=arm_plan.temperature, seed=seed,
                            response_sha256=digest(text), parsed=parsed, failure_code=failure, estimate=None,
                            evidence_ref=None)
    return ExecutionDraw(item_id, rendering, digest(f"prompt {item_id} {rendering}"), recorded, response, "fingerprint")


def ensemble(arm: str, key: tuple[str, str], answers) -> list[ExecutionDraw]:
    return [draw(arm, key, index, answer) for index, answer in enumerate(answers)]


SAMPLING_KEYS = planned_keys(PLAN, SAMPLING, STAGE)
SERVING_KEYS = planned_keys(PLAN, SERVING, STAGE)
#: Sampling arm: four prompts analysed (1, 1, 2 and 3 different answers), one with too few parsed draws, one
#: ensemble recorded only in part. The first two prompts are the two renderings of one month.
SAMPLING_ANSWERS = (
    (A, {"gold": 0.5, "equity": 0.3172}, A, A),  # one answer; the key order does not make a new answer
    (A, A, A, LOST),                              # one answer among three parsed draws
    (A, A, B, B),                                 # two
    (A, B, C, PROSE),                             # three
    (A, PROSE, PROSE, PROSE),                     # fully recorded, one parsed draw: cannot show variation
    (A, B, C),                                    # three of four planned draws recorded: excluded
)
SERVING_ANSWERS = ((A, A, A, A), (B, B, B, B), (C, C, PROSE, C))


def sampling_draws() -> list[ExecutionDraw]:
    return [made for key, answers in zip(SAMPLING_KEYS, SAMPLING_ANSWERS, strict=False)
            for made in ensemble(SAMPLING, key, answers)]


def serving_draws() -> list[ExecutionDraw]:
    return [made for key, answers in zip(SERVING_KEYS, SERVING_ANSWERS, strict=False)
            for made in ensemble(SERVING, key, answers)]


def build(**draws):
    return repeat_baseline(PLAN, draws, providers=PROVIDERS)


def plan_labelled():
    """The baseline of both arms' draws with the routes labelled by the frozen plan alone, as the records give it."""
    return repeat_baseline(PLAN, {SAMPLING: sampling_draws(), SERVING: serving_draws()})


@pytest.fixture(scope="module")
def baseline():
    return build(sampling=sampling_draws(), serving=serving_draws())


def values(baseline, arm: str) -> dict[str, float | None]:
    return {statistic.name: statistic.value for statistic in baseline.statistics[arm]}


def test_the_fixture_plan_draws_four_times_per_ensemble():
    assert DRAWS == 4 and PLAN.arms.arms[SAMPLING].draws == 4 and PLAN.arms.arms[SERVING].draws == 4
    assert SAMPLING_KEYS[0][0] == SAMPLING_KEYS[1][0] and SAMPLING_KEYS[0][1] != SAMPLING_KEYS[1][1]


def test_the_distribution_counts_the_different_parsed_answers_per_prompt(baseline):
    found = values(baseline, SAMPLING)
    assert found["prompts_analysed"] == 4
    assert found["prompts_with_1_different_answers"] == 2
    assert found["prompts_with_2_different_answers"] == 1
    assert found["prompts_with_3_different_answers"] == 1
    assert "prompts_with_4_different_answers" not in found
    assert found["median_different_answers"] == median([1, 1, 2, 3]) == 1.5
    assert found["share_of_prompts_with_one_answer"] == 0.5
    serving = values(baseline, SERVING)
    assert serving["prompts_analysed"] == 3 and serving["prompts_with_1_different_answers"] == 3
    assert serving["median_different_answers"] == 1 and serving["share_of_prompts_with_one_answer"] == 1.0


def test_ensembles_not_fully_recorded_and_prompts_with_one_parsed_draw_are_excluded_and_counted(baseline):
    found = values(baseline, SAMPLING)
    assert found["ensembles_not_fully_recorded"] == 1
    assert found["prompts_with_fewer_than_2_parsed_draws"] == 1
    incomplete = baseline.statistic(SAMPLING, "ensembles_not_fully_recorded")
    assert incomplete.inputs["prompts_with_recorded_draws"] == 6 and incomplete.inputs["fully_recorded_ensembles"] == 5
    assert incomplete.inputs["planned_draws_per_ensemble"] == 4
    assert incomplete.inputs["planned_prompts"] == len(SAMPLING_KEYS)
    # Completing the partial ensemble brings it into the analysis.
    completed = build(sampling=[*sampling_draws(), draw(SAMPLING, SAMPLING_KEYS[5], 3, C)])
    assert values(completed, SAMPLING)["ensembles_not_fully_recorded"] == 0
    assert values(completed, SAMPLING)["prompts_analysed"] == 5
    assert values(completed, SAMPLING)["prompts_with_3_different_answers"] == 2


def test_the_share_with_one_answer_carries_a_95_percent_wilson_interval(baseline):
    share = baseline.statistic(SAMPLING, "share_of_prompts_with_one_answer")
    assert (share.interval.lo, share.interval.hi) == wilson_interval(2, 4, confidence=0.95)
    assert share.interval.confidence == 0.95 and "Wilson" in share.interval.method
    full = baseline.statistic(SERVING, "share_of_prompts_with_one_answer")
    lo, hi = wilson_interval(3, 3, confidence=0.95)
    assert full.interval.lo == lo < 1.0  # not degenerate at 3 of 3
    # The upper bound at n of n is exactly 1 (and the lower bound at 0 of n exactly 0): no floating-point residue.
    assert full.interval.hi == 1.0 and abs(hi - 1.0) < 1e-12
    none = build(serving=[made for key, answers in zip(SERVING_KEYS, ((A, B, A, B), (A, B, C, C)), strict=False)
                          for made in ensemble(SERVING, key, answers)])
    empty = none.statistic(SERVING, "share_of_prompts_with_one_answer")
    upper = wilson_interval(0, 2, confidence=0.95)[1]
    assert empty.value == 0.0 and empty.interval.lo == 0.0 and empty.interval.hi == upper > 0.0
    assert baseline.statistic(SAMPLING, "prompts_analysed").interval is None


def test_every_value_is_a_statistic_with_its_calculation_exposed(baseline):
    for arm in (SAMPLING, SERVING):
        for statistic in baseline.statistics[arm]:
            assert isinstance(statistic, Statistic)
            assert statistic.unit == "prompt" and statistic.estimand and statistic.formula and statistic.inputs
            assert statistic.test is None and not statistic.by_construction
    share = baseline.statistic(SAMPLING, "share_of_prompts_with_one_answer")
    assert share.n == 4
    assert dict(share.inputs) == {"prompts_with_1_different_answers": 2, "prompts_analysed": 4}
    assert share.value == share.inputs["prompts_with_1_different_answers"] / share.inputs["prompts_analysed"]
    assert "0.7" in share.estimand and "llamacpp-gemma3-4b" in share.estimand and "llama.cpp" in share.estimand
    # Two renderings of one month share the month, so the prompts are not independent.
    assert share.independent_units is False
    single = build(sampling=ensemble(SAMPLING, SAMPLING_KEYS[0], (A, A, B, B))
                   + ensemble(SAMPLING, SAMPLING_KEYS[2], (A, A, A, A)))
    assert single.statistic(SAMPLING, "share_of_prompts_with_one_answer").independent_units is True


def test_an_arm_without_recorded_draws_gives_missing_records_never_a_value(baseline):
    alone = build(sampling=sampling_draws())
    assert values(alone, SAMPLING) == values(baseline, SAMPLING)
    records = alone.statistics[SERVING]
    assert {"prompts_analysed", "median_different_answers", "share_of_prompts_with_one_answer"} <= {
        statistic.name for statistic in records}
    for statistic in records:
        assert statistic.value is None and statistic.interval is None and statistic.missing == NO_DRAWS
    document = alone.to_dict()
    assert document["arms"][SERVING]["missing"] == NO_DRAWS and document["arms"][SAMPLING]["missing"] is None
    assert "missing" in alone.to_markdown()


def test_an_arm_whose_ensembles_are_all_excluded_reports_the_counts_and_no_share():
    partial = build(serving=ensemble(SERVING, SERVING_KEYS[0], (A, A, A)))
    found = values(partial, SERVING)
    assert found["ensembles_not_fully_recorded"] == 1 and found["prompts_analysed"] == 0
    for name in ("median_different_answers", "share_of_prompts_with_one_answer", "prompts_with_1_different_answers"):
        assert found[name] is None and partial.statistic(SERVING, name).missing == NO_PROMPTS


def test_the_document_names_route_and_temperature_of_each_arm(baseline):
    document = baseline.to_dict()
    assert document["schema"] == BASELINE_SCHEMA and document["plan_sha256"] == PLAN.plan_sha256
    assert document["stage"] == STAGE == "full_run"
    sampling, serving = document["arms"][SAMPLING], document["arms"][SERVING]
    assert sampling["route"] == {"profile_id": "llamacpp-gemma3-4b", "provider": "llama.cpp",
                                 "served_model": "gemma3:4b"}
    assert serving["route"]["profile_id"] == "modal-gemma3-4b" and serving["route"]["provider"] == "modal"
    assert sampling["temperature"] == 0.7 and serving["temperature"] == 0.0
    assert sampling["planned_draws"] == 4 and sampling["recorded_draws"] == 23 and serving["recorded_draws"] == 12
    share = next(row for row in sampling["statistics"] if row["name"] == "share_of_prompts_with_one_answer")
    lo, hi = wilson_interval(2, 4, confidence=0.95)
    assert share["value"] == 0.5 and share["n"] == 4 and share["unit"] == "prompt"
    assert share["interval"]["lo"] == lo and share["interval"]["hi"] == hi and share["interval"]["confidence"] == 0.95
    assert share["inputs"] == {"prompts_analysed": 4, "prompts_with_1_different_answers": 2}
    assert share["independent_units"] is False and share["missing"] is None


def test_the_route_label_comes_from_the_frozen_plan_alone_when_no_provider_is_given(monkeypatch):
    """The replay compares bytes, so the label of a route must not follow a later edit of a provider profile: without
    ``providers`` the route is the plan's profile id and served model, and no profile file is read."""
    import inspect

    from recall_guard.core import profiles
    from scripts.finance_sim import repeat_baseline as module

    def forbidden(*args, **kwargs):
        raise AssertionError("the repeat baseline read a provider profile")

    monkeypatch.setattr(profiles, "load_profile", forbidden)
    assert "load_profile" not in inspect.getsource(module) and "core.profiles" not in inspect.getsource(module)
    found = repeat_baseline(PLAN, {SAMPLING: sampling_draws()})
    document = found.to_dict()
    for arm in (SAMPLING, SERVING):
        arm_plan = PLAN.arms.arms[arm]
        assert document["arms"][arm]["route"] == {"profile_id": arm_plan.profile_id, "provider": None,
                                                  "served_model": arm_plan.served_model}
    assert "the serving route llamacpp-gemma3-4b (model gemma3:4b) at temperature 0.7" in (
        found.statistic(SAMPLING, "prompts_analysed").estimand)
    markdown = found.to_markdown()
    assert "serving route llamacpp-gemma3-4b (model gemma3:4b), temperature 0.7" in markdown
    assert "| llamacpp-gemma3-4b |" in markdown and "None" not in markdown and "provider" not in markdown
    # A caller that knows the provider can still name it, as before.
    named = repeat_baseline(PLAN, {SAMPLING: sampling_draws()}, providers=PROVIDERS).to_dict()
    assert named["arms"][SAMPLING]["route"]["provider"] == "llama.cpp"
    assert named["arms"][SERVING]["route"]["provider"] == "modal"


def test_each_table_row_shows_value_interval_n_and_the_calculation(baseline):
    text = baseline.to_markdown()
    lines = text.splitlines()
    header = next(line for line in lines if line.startswith("| Arm"))
    for column in ("Serving route", "Temperature", "Statistic", "Value", "Interval", "n", "Calculation"):
        assert column in header
    row = next(line for line in lines if line.startswith(f"| {SAMPLING} ") and "share_of_prompts_with_one_answer" in line)
    lo, hi = wilson_interval(2, 4, confidence=0.95)
    assert "| 0.5 |" in row and f"{lo:.4g} to {hi:.4g} (95 percent)" in row and "| 4 |" in row
    assert "prompts_with_1_different_answers = 2" in row and "prompts_analysed = 4" in row
    assert "llamacpp-gemma3-4b" in row and "llama.cpp" in row and "| 0.7 |" in row
    assert "Wilson" in text and "not independent" in text and PLAN.plan_sha256 in text


def test_the_outputs_carry_derived_statistics_only(baseline):
    for text in (baseline.document, baseline.to_markdown()):
        assert MARKER not in text
        assert "0.3172" not in text and "0.4172" not in text and "0.5172" not in text
        assert "macro-" not in text and "equity" not in text
    assert "items" not in baseline.to_dict()["arms"][SAMPLING]


def record(records_dir, arm: str, draws) -> None:
    log = DrawLog(draw_log_path(arm, STAGE, records_dir), plan=PLAN, loads=ExecutionDraw.from_dict)
    for made in draws:
        log(made)


def test_the_baseline_is_read_from_the_recorded_draws_and_written_once(baseline, tmp_path):
    records_dir, out_dir = tmp_path / "records", tmp_path / "analysis"
    record(records_dir, SAMPLING, sampling_draws())
    record(records_dir, SERVING, serving_draws())
    read = repeat_baseline_from_records(PLAN, records_dir=records_dir)
    # Read from the records, the routes are labelled by the frozen plan alone; the statistics are the same.
    assert read.document == plan_labelled().document != baseline.document
    assert {arm: [row["value"] for row in entry["statistics"]] for arm, entry in read.to_dict()["arms"].items()} == {
        arm: [row["value"] for row in entry["statistics"]] for arm, entry in baseline.to_dict()["arms"].items()}
    assert read.document.endswith("\n") and "\r" not in read.document
    assert read.document == json.dumps(read.to_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n"
    json_path, markdown_path = write_baseline(read, out_dir)
    assert json_path == baseline_path(out_dir) == out_dir / "repeat-baseline.json"
    assert markdown_path == out_dir / "repeat-baseline.md"
    assert json_path.read_bytes() == read.document.encode("utf-8")
    assert markdown_path.read_bytes() == read.to_markdown().encode("utf-8") and b"\r" not in markdown_path.read_bytes()
    assert write_baseline(repeat_baseline_from_records(PLAN, records_dir=records_dir), out_dir) == (json_path,
                                                                                                    markdown_path)
    other = build(sampling=sampling_draws())
    with pytest.raises(RepeatBaselineError, match="refusing to replace"):
        write_baseline(other, out_dir)
    assert json_path.read_bytes() == read.document.encode("utf-8")


@pytest.mark.parametrize(("ending", "ignored"), [("no_newline", 0), ("cut_short", 1)])
def test_reading_the_records_never_modifies_a_draw_log(baseline, tmp_path, ending, ignored):
    """A last line without its newline is used, one cut short is ignored and counted; the log keeps its bytes."""
    record(tmp_path, SAMPLING, sampling_draws())
    record(tmp_path, SERVING, serving_draws())
    before = {}
    for arm in (SAMPLING, SERVING):
        path = draw_log_path(arm, STAGE, tmp_path)
        data = path.read_bytes()
        path.write_bytes(data.rstrip(b"\n") if ending == "no_newline" else data + CUT)
        before[path] = aged(path)
        assert not before[path][0].endswith(b"\n")

    read = repeat_baseline_from_records(PLAN, records_dir=tmp_path)

    assert all(unchanged(path, state) for path, state in before.items())
    assert sorted(path.name for path in tmp_path.rglob("*") if path.is_file()) == ["draws.jsonl", "draws.jsonl"]
    document = read.to_dict()
    for arm in (SAMPLING, SERVING):
        assert values(read, arm) == values(baseline, arm)
        assert document["arms"][arm]["incomplete_last_lines_ignored"] == ignored
        assert document["arms"][arm]["recorded_draws"] == baseline.to_dict()["arms"][arm]["recorded_draws"]
    assert ("incomplete last line" in read.to_markdown()) == bool(ignored)
    assert (read.document == plan_labelled().document) == (not ignored)


def test_the_analysis_holds_no_write_other_than_its_own_outputs():
    source = inspect.getsource(repeat_baseline_module)
    assert "DrawLog" not in source and ".latest(" not in source and "_repaired_lines" not in source
    assert not hasattr(repeat_baseline_module, "DrawLog")
    assert writing_calls(repeat_baseline_module) == {
        ("_write_text", "mkdir"), ("_write_text", "open"), ("_write_text", "os.replace"),
        ("write_baseline", "_write_text")}


def test_the_latest_version_of_a_draw_is_the_one_counted(tmp_path):
    first = ensemble(SERVING, SERVING_KEYS[0], (A, A, A, B))
    record(tmp_path, SERVING, [*first, draw(SERVING, SERVING_KEYS[0], 3, A)])
    read = repeat_baseline_from_records(PLAN, records_dir=tmp_path)
    assert values(read, SERVING)["prompts_with_1_different_answers"] == 1
    assert read.to_dict()["arms"][SERVING]["recorded_draws"] == 4
    assert read.to_dict()["arms"][SAMPLING]["missing"] == NO_DRAWS


def test_draws_the_plan_does_not_size_are_refused():
    with pytest.raises(RepeatBaselineError, match="not a planned"):
        build(sampling=[draw(SAMPLING, ("macro-1999-01", "identifying"), 0, A)])
    with pytest.raises(RepeatBaselineError, match="draw index"):
        build(sampling=[draw(SAMPLING, SAMPLING_KEYS[0], DRAWS, A)])
    with pytest.raises(RepeatBaselineError, match="twice"):
        build(sampling=[draw(SAMPLING, SAMPLING_KEYS[0], 0, A), draw(SAMPLING, SAMPLING_KEYS[0], 0, B)])
    with pytest.raises(RepeatBaselineError, match="temperature"):
        build(serving=[draw(SAMPLING, SERVING_KEYS[0], 0, A)])
    with pytest.raises(RepeatBaselineError, match="arm"):
        build(other=[])
    with pytest.raises(TypeError):
        build(sampling=[object()])


def test_draws_recorded_under_another_plan_are_refused(tmp_path):
    record(tmp_path, SAMPLING, sampling_draws())
    with pytest.raises(PlanChangedError):
        repeat_baseline_from_records(frozen_plan(seed=7), records_dir=tmp_path)


def test_the_command_writes_the_baseline_of_the_frozen_plan(tmp_path, capsys):
    plan_path = write_plan(PLAN, tmp_path / "plan.json")
    records_dir, out_dir = tmp_path / "records", tmp_path / "analysis"
    record(records_dir, SERVING, serving_draws())
    arguments = ["--plan", str(plan_path), "--records-dir", str(records_dir), "--out-dir", str(out_dir)]
    assert main([*arguments, "--plan-sha256", PLAN.plan_sha256]) == 0
    assert capsys.readouterr().out.splitlines() == [str(out_dir / "repeat-baseline.json"),
                                                    str(out_dir / "repeat-baseline.md")]
    assert main([*arguments, "--plan-sha256", "0" * 64]) == 2
    assert "refused" in capsys.readouterr().err
