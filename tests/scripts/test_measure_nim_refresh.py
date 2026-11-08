"""Offline contract checks for the explicitly gated NIM refresh."""
import json
from pathlib import Path

import pytest

from scripts import measure_nim_refresh as refresh

INPUTS = Path("data/experiments/2026-10-31-completion-api")


def test_plan_is_environment_free(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("offline planning constructed a client")
    monkeypatch.setattr(refresh, "RecordedClient", forbidden)
    plan = refresh.build_plan(INPUTS)
    assert plan["phase_caps"] == {"eligibility": 2, "calibration": 80, "evaluation": 64}
    assert plan["nominal_requests"] == {"eligibility": 1, "calibration": 62, "evaluation": 49}
    assert plan["maximum_total_requests"] == 146
    assert plan["target"]["min_call_interval_s"] == 2.0
    assert plan["target"]["max_retries"] == 0
    assert plan["target"]["parameters"] == {
        "max_tokens": 2048, "temperature": 0, "logprobs": True, "top_logprobs": 20}
    assert plan["target"]["readiness"]["socket_idle_timeout_s"] == 600
    assert plan["target"]["readiness"]["max_wait_s"] == 1200
    assert not {"max_poll_requests", "pending_status"} & plan["target"]["readiness"].keys()
    assert len(plan["input_hashes"]) == 3


def test_label_free_inputs_and_separation():
    groups = refresh.load_inputs(INPUTS)
    assert [len(group) for group in groups.values()] == [16, 20, 20]
    for group in groups.values():
        for item in group:
            model_input = refresh.label_free_input(item)
            assert not hasattr(model_input, "label")
            assert not hasattr(model_input, "cohort")


def test_overlap_refused(tmp_path):
    for name in refresh.INPUT_FILES:
        (tmp_path / name).write_bytes((INPUTS / name).read_bytes())
    data = json.loads((tmp_path / "calibration-is.json").read_text())
    data["items"][0] = json.loads((tmp_path / "evaluation.json").read_text())["items"][0]
    (tmp_path / "calibration-is.json").write_text(json.dumps(data))
    with pytest.raises(ValueError, match="disjoint"):
        refresh.load_inputs(tmp_path)


def test_nonempty_destination_refused_before_client(tmp_path, monkeypatch):
    (tmp_path / "intent.json").write_text("{}")
    monkeypatch.setattr(refresh, "RecordedClient", lambda *a, **k: pytest.fail("constructed client"))
    with pytest.raises(FileExistsError):
        refresh.run(INPUTS, tmp_path, plan_file=tmp_path / "unused.json")


def test_cli_plan_and_dry_run_are_identical(capsys):
    assert refresh.main(["--plan", "--inputs", str(INPUTS)]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert refresh.main(["--dry-run", "--inputs", str(INPUTS)]) == 0
    assert json.loads(capsys.readouterr().out) == plan


def fixture_transport(monkeypatch, *, missing_logprobs=False, reject=False):
    """Replace HTTP exchange only; retain stock physical admission and MIA fit."""
    import hashlib

    from recall_guard.core import providers
    from recall_guard.core.completion import ModelResponse, TokenAlternative, TokenEvidence
    from recall_guard.core.request_control import RequestError

    original_init = providers.OpenAICompatibleClient.__init__

    def initialize(self, config, budget, **kwargs):
        from dataclasses import replace

        kwargs.pop("environment", None)
        # Disable pacing only at the fake transport boundary; frozen policy stays intact.
        config = replace(config, min_call_interval_s=0)
        return original_init(self, config, budget, environment={"NVIDIA_API_KEY": "fixture-only-key"}, **kwargs)

    def exchange(config, credential, payload, parameters, context, **kwargs):
        if reject:
            raise RequestError("fixture-only-key must never escape", status_code=401)
        prompt = payload["messages"][0]["content"]
        jitter = int(hashlib.sha256(prompt.encode()).hexdigest(), 16) % 1000 / 20000
        base = -0.2 if "2022-" in prompt else -2
        tokens = None if missing_logprobs else tuple(
            TokenEvidence("x", base + jitter + index * 0.01,
                          (TokenAlternative("x", base + jitter), TokenAlternative("y", base - 1)))
            for index in range(5))
        return ModelResponse('{"direction": 1, "confidence": 0.8}', tokens,
                             usage={"prompt_tokens": 5, "completion_tokens": 7},
                             requested_settings=parameters)

    monkeypatch.setattr(providers.OpenAICompatibleClient, "__init__", initialize)
    monkeypatch.setattr(providers, "exchange", exchange)


def frozen_plan(tmp_path, inputs=INPUTS, *, eligibility_request_cap=2):
    path = tmp_path / "frozen-plan.json"
    path.write_text(json.dumps(refresh.build_plan(inputs, eligibility_request_cap=eligibility_request_cap)),
                    encoding="utf-8")
    return path


@pytest.mark.parametrize("eligibility_request_cap", [1, 2])
def test_full_fixture_run_reuses_draw_zero_and_real_budget(tmp_path, monkeypatch, eligibility_request_cap):
    fixture_transport(monkeypatch)
    output = tmp_path / "run"
    receipt = refresh.run(INPUTS, output,
                          plan_file=frozen_plan(tmp_path, eligibility_request_cap=eligibility_request_cap),
                          eligibility_request_cap=eligibility_request_cap)
    assert receipt["status"] == "completed", receipt
    assert receipt["physical_requests"] == 112
    assert receipt["budgets"]["eligibility"]["limit"] == eligibility_request_cap
    assert {name: value["used"] for name, value in receipt["budgets"].items()} == {
        "eligibility": 1, "calibration": 62, "evaluation": 49}
    rows = [json.loads(path.read_text()) for path in sorted(output.glob("item-??.json"))]
    assert len(rows) == 16
    assert all(len(row["draws"]) == 3 for row in rows)
    assert all(row["single"] == row["draws"][0] for row in rows)
    assert all(row["scored_ensemble"]["p_memorized_point"] is not None for row in rows)
    fit = json.loads((output / "calibration.json").read_text())
    assert len(fit["fingerprint"]) == 64
    assert fit["counts"]["is_memorized"]["training_valid"] == 20


@pytest.mark.parametrize("missing_logprobs,reject", [(True, False), (False, True)])
def test_failed_eligibility_stops_before_calibration(tmp_path, monkeypatch, missing_logprobs, reject):
    fixture_transport(monkeypatch, missing_logprobs=missing_logprobs, reject=reject)
    output = tmp_path / "run"
    receipt = refresh.run(INPUTS, output, plan_file=frozen_plan(tmp_path))
    assert receipt["status"] == "incomplete"
    assert receipt["physical_requests"] == 1
    assert set(receipt["budgets"]) == {"eligibility"}
    assert not (output / "calibration").exists()
    for path in output.rglob("*.json"):
        assert "fixture-only-key" not in path.read_text()
    evidence = json.loads(next((output / "eligibility").glob("*.receipt.json")).read_text())
    terminal = evidence["observations"][-1]
    assert terminal["credential_eligibility"] == ("rejected" if reject else "unverified")


def test_admission_deadline_refuses_call_without_intent_or_event(tmp_path, monkeypatch):
    from recall_guard.core.request_control import CallBudget

    fixture_transport(monkeypatch)
    budget = CallBudget(2)
    client = refresh.RecordedClient(refresh.target_config(), budget, directory=tmp_path, deadline=100)
    monkeypatch.setattr(refresh.time, "monotonic", lambda: 100)
    with pytest.raises(TimeoutError, match="admission window"):
        client.probe()
    assert budget.used == 0
    assert not list(tmp_path.iterdir())


def test_explicit_credential_file_unquotes_and_excludes_unselected_keys(tmp_path):
    path = tmp_path / "fixture.env"
    path.write_text('NVIDIA_API_KEY="fixture-only-key"\nOTHER_PROVIDER_KEY="unused"\n', encoding="utf-8")
    assert refresh.credential_environment(path) == {"NVIDIA_API_KEY": "fixture-only-key"}


def test_plan_refuses_credential_file_without_reading_it():
    with pytest.raises(SystemExit):
        refresh.main(["--plan", "--credential-file", "nonexistent.env"])


def test_changed_frozen_plan_refused_before_credentials(tmp_path, monkeypatch):
    plan_file = frozen_plan(tmp_path)
    plan = json.loads(plan_file.read_text())
    plan["input_hashes"]["evaluation.json"] = "0" * 64
    plan_file.write_text(json.dumps(plan))
    monkeypatch.setattr(refresh, "credential_environment", lambda *a: pytest.fail("read secret"))
    monkeypatch.setattr(refresh, "RecordedClient", lambda *a, **k: pytest.fail("constructed client"))
    with pytest.raises(ValueError, match="Frozen plan differs"):
        refresh.run(INPUTS, tmp_path / "run", plan_file=plan_file, credential_file=Path("never-read.env"))
    assert not (tmp_path / "run").exists()


def test_run_requires_explicit_output_and_frozen_plan():
    with pytest.raises(SystemExit):
        refresh.main(["--run"])
    with pytest.raises(SystemExit):
        refresh.main(["--run", "--output", "unused-output"])


def test_one_unusable_item_does_not_drop_other_fifteen(tmp_path, monkeypatch):
    from dataclasses import replace

    from recall_guard.core import providers

    fixture_transport(monkeypatch)
    original_exchange = providers.exchange
    calls = 0

    def exchange(*args, **kwargs):
        nonlocal calls
        calls += 1
        response = original_exchange(*args, **kwargs)
        return replace(response, content="Unparseable fixture") if 65 <= calls <= 67 else response

    monkeypatch.setattr(providers, "exchange", exchange)
    output = tmp_path / "run"
    receipt = refresh.run(INPUTS, output, plan_file=frozen_plan(tmp_path))
    assert receipt["status"] == "completed", receipt
    assert receipt["processed_items"] == 16
    assert receipt["failed_items"] == 1
    assert receipt["physical_requests"] == 112
    rows = [json.loads(path.read_text()) for path in sorted(output.glob("item-??.json"))]
    assert rows[0]["status"] == "failed"
    assert all(row["status"] == "completed" for row in rows[1:])
    assert len(rows[0]["draws"]) == 3


def test_pacing_cannot_admit_a_second_request_after_run_deadline(tmp_path, monkeypatch):
    from dataclasses import replace

    from recall_guard.core.request_control import CallBudget, RequestError

    fixture_transport(monkeypatch)
    now = [99.0]
    monkeypatch.setattr(refresh.time, "monotonic", lambda: now[0])
    budget = CallBudget(2)
    client = refresh.RecordedClient(refresh.target_config(), budget, directory=tmp_path, deadline=100)
    client.config = replace(client.config, min_call_interval_s=1.5)
    client._control.config = client.config
    client._control._clock = lambda: now[0]
    client._control._sleep = lambda seconds: now.__setitem__(0, now[0] + seconds)
    assert client.probe().ready_for_batch
    now[0] = 99.6
    with pytest.raises(RequestError):
        client.probe()
    assert budget.used == 1
    assert [event.started_at for event in budget.events] == [99.0]
    receipt = json.loads((tmp_path / "request-002.receipt.json").read_text())
    assert receipt["observations"][-1]["state"] == "cancelled"


def test_lower_eligibility_cap_plan(capsys):
    assert refresh.main(["--plan", "--eligibility-request-cap", "1"]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["phase_caps"] == {"eligibility": 1, "calibration": 80, "evaluation": 64}
    assert plan["maximum_total_requests"] == 145
    assert sum(plan["nominal_requests"].values()) == 112
    assert plan["target"] == refresh.target_config().to_dict()


def test_lower_cap_requires_matching_frozen_plan_before_credentials(tmp_path, monkeypatch):
    plan_file = frozen_plan(tmp_path)
    monkeypatch.setattr(refresh, "credential_environment", lambda *a: pytest.fail("read secret"))
    monkeypatch.setattr(refresh, "RecordedClient", lambda *a, **k: pytest.fail("constructed client"))
    with pytest.raises(ValueError, match="Frozen plan differs"):
        refresh.run(INPUTS, tmp_path / "run", plan_file=plan_file, eligibility_request_cap=1,
                    credential_file=Path("never-read.env"))
    assert not (tmp_path / "run").exists()
