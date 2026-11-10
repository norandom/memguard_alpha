"""Versioned draw records: the latest version of a draw wins (task 7.4, the seam task 7.6 reports through).

A calibration draw is reported once when it is made, with its full-distribution
evidence pending, and again when the evidence pass settles it with its evidence
reference. The draw log keeps both versions on disk and returns the latest per
draw, which is what a resume passes back as ``recorded``. Files under
``tmp_path`` only; no network.
"""

from __future__ import annotations

import hashlib
import json
import socket

import pytest

from scripts.finance_sim.calibrate import FEATURE_ORDER, CalibrationDraw
from scripts.finance_sim.plan import PlanChangedError
from scripts.finance_sim.recording import DrawLog, RecordingError, draw_log_path
from tests.scripts.finance_sim.test_recording import OTHER_PLAN, PLAN, STAGE

REPLY = '{"equity": 0.2, "gold": 0.5}'


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("the draw log attempted a network connection")

    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def pending(draw_index: int = 0, item_id: str = "macro-2016-03") -> CalibrationDraw:
    """A parsed draw as the generation pass reports it: top-k settled, full distribution pending."""
    return CalibrationDraw(kind="macro", item_id=item_id, rendering="identifying", label=1,
                           prompt_sha256=sha256(f"prompt {item_id}"), draw_index=draw_index, seed=1000 + draw_index,
                           temperature=0.7, response_sha256=sha256(REPLY), response_text=REPLY, failure_code=None,
                           features={"top_k": dict.fromkeys(FEATURE_ORDER, -0.5)}, evidence_ref={"top_k": "a" * 64})


def settled(draw: CalibrationDraw) -> CalibrationDraw:
    return draw.settled(dict.fromkeys(FEATURE_ORDER, -0.25), "b" * 64)


def draw_log(tmp_path, plan=PLAN) -> DrawLog:
    return DrawLog(draw_log_path("sampling", STAGE, tmp_path), plan=plan, loads=CalibrationDraw.from_dict)


def test_a_settled_version_replaces_its_pending_version_and_both_stay_on_disk(tmp_path):
    log = draw_log(tmp_path)
    first, second = pending(0), pending(1)
    for draw in (first, second, settled(first)):
        log(draw)
    latest = draw_log(tmp_path).latest()
    assert latest == [settled(first), second]
    assert latest[0].evidence_ref == {"full_distribution": "b" * 64, "top_k": "a" * 64}
    assert latest[1].evidence_pending
    lines = [json.loads(line) for line in log.path.read_text(encoding="utf-8").splitlines()]
    assert [line["key"] for line in lines] == [list(first.key), list(second.key), list(first.key)]
    assert {line["plan_sha256"] for line in lines} == {PLAN.plan_sha256}
    assert log.path == tmp_path / "sampling" / STAGE / "draws.jsonl"


def test_a_resume_before_the_evidence_settled_gets_the_pending_version_then_the_settled_one(tmp_path):
    draw_log(tmp_path)(pending(0))
    recorded = draw_log(tmp_path).latest()
    assert recorded == [pending(0)] and recorded[0].evidence_pending
    resumed = draw_log(tmp_path)
    resumed(settled(recorded[0]))
    assert resumed.latest() == [settled(pending(0))]


def test_a_line_cut_short_by_a_crash_is_dropped_and_the_log_stays_line_aligned(tmp_path):
    log = draw_log(tmp_path)
    log(pending(0))
    with log.path.open("ab") as stream:
        stream.write(b'{"schema": "finance-sim-draw-version/1", "plan_sha')
    assert log.latest() == [pending(0)]
    log(pending(1))
    assert draw_log(tmp_path).latest() == [pending(0), pending(1)]


def test_a_draw_log_under_another_plan_is_refused_never_merged(tmp_path):
    draw_log(tmp_path)(pending(0))
    with pytest.raises(PlanChangedError, match="never merged"):
        draw_log(tmp_path, OTHER_PLAN).latest()


def test_a_tampered_key_or_payload_is_refused(tmp_path):
    log = draw_log(tmp_path)
    log(pending(0))
    line = json.loads(log.path.read_text(encoding="utf-8"))
    line["key"][3] = 7
    log.path.write_text(json.dumps(line) + "\n", encoding="utf-8")
    with pytest.raises(RecordingError, match="key"):
        log.latest()
    line["key"][3] = 0
    line["draw"]["response_text"] = "other reply"
    log.path.write_text(json.dumps(line) + "\n", encoding="utf-8")
    with pytest.raises(RecordingError, match="cannot be read"):
        log.latest()


def test_a_record_without_a_key_is_refused_before_anything_is_written(tmp_path):
    class Keyless:
        key = None

        def to_dict(self):
            return {}

    log = DrawLog(draw_log_path("sampling", STAGE, tmp_path), plan=PLAN, loads=lambda payload: Keyless())
    with pytest.raises(TypeError, match="key"):
        log(Keyless())
    assert not log.path.exists()
