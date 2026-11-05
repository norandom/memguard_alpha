"""Exercise the pilot's request envelope without a network or credentials."""

import hashlib
import json
from datetime import date

import pytest

from recall_guard.core.completion import ModelResponse
from recall_guard.core.request_control import CallBudget, RequestControl, RequestError
from scripts.prepare_completion_inputs import sessions
from scripts.prepare_sp500_pilot import build
from scripts.run_sp500_pilot import execute, make_client


@pytest.fixture
def bundle(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    days = sessions("2023-12-01", "2026-11-03")
    raw = ("observation_date,SP500\n" + "".join(
        f"{day},{4000 + index + (index % 7) * 0.3:.2f}\n" for index, day in enumerate(days))).encode()
    (source / "SP500.csv").write_bytes(raw)
    (source / "source.json").write_text(json.dumps({"series": "SP500", "sha256": hashlib.sha256(raw).hexdigest()}))
    return build(source, date(2026, 11, 4))


class FakeClock:
    value = 0

    def __call__(self):
        return self.value

    def sleep(self, duration):
        self.value += duration


def run_fake(bundle, output, *, fail_at=None, probe='{"value":1}', finish="stop", duration=0, only_model=None):
    clock = FakeClock()
    calls = []

    def factory(model, settings, environment):
        # Construct the real adapters/configurations, but replace only generation.
        client = make_client(model, settings, {settings["credential_ref"]: "offline-placeholder"})
        control = RequestControl(client.config, CallBudget(25), clock=clock, sleep=clock.sleep)

        class FakeClient:
            budget = control.budget

            def generate_for_stage(self, prompt, *, stage):
                def operation():
                    calls.append((model, prompt, clock()))
                    clock.sleep(duration)
                    if len(calls) == fail_at:
                        raise RequestError("synthetic transport failure", code="timeout")
                    eligibility = stage.endswith(":eligibility")
                    return ModelResponse(content=probe if eligibility else '{"value":100}',
                                         finish_reason="stop" if eligibility else finish)
                return control.run(operation, stage=stage)

        return FakeClient()

    summary = execute(*bundle, output, {}, client_factory=factory, clock=clock, sleep=clock.sleep,
                      only_model=only_model)
    return summary, calls


def test_exact_cap_pacing_and_frozen_prompts(bundle, tmp_path):
    output = tmp_path / "run"
    summary, calls = run_fake(bundle, output)
    assert summary["physical_requests"] == 50
    assert all(result["physical_requests"] == 25 for result in summary["providers"].values())
    assert all(right[2] - left[2] >= 2 for left, right in zip(calls, calls[1:], strict=False))
    plan, _, requests = bundle
    for model in plan["models"]:
        assert [prompt for selected, prompt, _ in calls if selected == model] == [
            plan["eligibility_prompt"], *[row["prompt"] for row in requests if row["model"] == model]]
    events = [json.loads(line) for line in (output / "events.jsonl").read_text().splitlines()]
    assert [event["state"] for event in events] == ["intent", "response"] * 50
    with pytest.raises(FileExistsError):
        run_fake(bundle, output)


@pytest.mark.parametrize("probe", ['{"value":null}', '{"value":2}', "not json"])
def test_bad_eligibility_stops_before_measurements(bundle, tmp_path, probe):
    summary, calls = run_fake(bundle, tmp_path / "run", probe=probe)
    assert len(calls) == 2
    assert all(result["status"] == "eligibility_rejected" for result in summary["providers"].values())
    assert all(result["cutoff_contrast"] is None for result in summary["models"].values())


def test_ambiguous_failure_stops_only_affected_provider(bundle, tmp_path):
    summary, calls = run_fake(bundle, tmp_path / "run", fail_at=2)
    assert len(calls) == summary["physical_requests"] == 27
    first, second = summary["providers"].values()
    assert first == {"status": "provider_failure", "physical_requests": 2}
    assert second == {"status": "completed", "physical_requests": 25}
    scores = json.loads((tmp_path / "run" / "scores.json").read_text())
    assert sum(score["status"] == "failed" for score in scores) == 1


def test_truncated_json_is_not_scored(bundle, tmp_path):
    summary, _ = run_fake(bundle, tmp_path / "run", finish="length")
    assert all(result["cutoff_contrast"] is None for result in summary["models"].values())
    scores = json.loads((tmp_path / "run" / "scores.json").read_text())
    assert len(scores) == 48 and all(score["status"] == "invalid" for score in scores)


def test_admission_deadline_preserves_unperformed_rows(bundle, tmp_path):
    summary, calls = run_fake(bundle, tmp_path / "run", duration=601)
    assert len(calls) == 6
    assert all(result["status"] == "admission_deadline" for result in summary["providers"].values())
    assert all(result["cutoff_contrast"] is None for result in summary["models"].values())


def test_missing_credentials_make_no_calls(bundle, tmp_path):
    summary = execute(*bundle, tmp_path / "run", {})
    assert summary["physical_requests"] == 0
    assert all(result["status"] == "configuration_or_credential_missing" for result in summary["providers"].values())


def test_selected_model_never_repeats_other_provider(bundle, tmp_path):
    summary, calls = run_fake(bundle, tmp_path / "run", only_model="openai/gpt-oss-20b")
    assert summary["physical_requests"] == len(calls) == 25
    assert {model for model, _, _ in calls} == {"openai/gpt-oss-20b"}
    assert summary["providers"]["claude-opus-5-5"] == {"status": "not_selected", "physical_requests": 0}
    assert summary["models"]["claude-opus-5-5"]["cutoff_contrast"] is None
    receipt = json.loads((tmp_path / "run" / "execution.json").read_text())
    assert receipt["approved_max_requests"] == 25


def test_unknown_model_fails_before_output_or_calls(bundle, tmp_path):
    with pytest.raises(ValueError, match="absent"):
        run_fake(bundle, tmp_path / "run", only_model="unknown")
    assert not (tmp_path / "run").exists()
