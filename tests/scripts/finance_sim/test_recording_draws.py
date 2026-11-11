"""Versioned draw records: the latest version of a draw wins (task 7.4, the seam task 7.6 reports through).

A calibration draw is reported once when it is made, with its full-distribution
evidence pending, and again when the evidence pass settles it with its evidence
reference. The draw log keeps both versions on disk and returns the latest per
draw, which is what a resume passes back as ``recorded``. Files under
``tmp_path`` only; no network.
"""

from __future__ import annotations

import ast
import hashlib
import inspect
import json
import os
import socket

import pytest

from scripts.finance_sim.calibrate import FEATURE_ORDER, CalibrationDraw
from scripts.finance_sim.plan import PlanChangedError
from scripts.finance_sim.recording import DrawLog, DrawLogReading, RecordingError, draw_log_path, read_draw_log
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


# ---------- the read-only reader (qualified-model-catalogue 3.1: an analysis never modifies a private record) ----------

CUT = b'{"schema": "finance-sim-draw-version/1", "plan_sha'
PAST_NS = 1_600_000_000_000_000_000


def read_only(tmp_path, plan=PLAN) -> DrawLogReading:
    return read_draw_log(draw_log_path("sampling", STAGE, tmp_path), plan=plan, loads=CalibrationDraw.from_dict)


def aged(path) -> tuple[bytes, int]:
    """The bytes of ``path`` and its modification time, set to a past moment so that any write would show."""
    os.utime(path, ns=(PAST_NS, PAST_NS))
    return path.read_bytes(), path.stat().st_mtime_ns


def unchanged(path, before: tuple[bytes, int]) -> bool:
    return (path.read_bytes(), path.stat().st_mtime_ns) == before


def test_the_read_only_reader_returns_the_latest_version_of_each_draw_like_the_log(tmp_path):
    log = draw_log(tmp_path)
    first, second = pending(0), pending(1)
    for draw in (first, second, settled(first)):
        log(draw)
    before = aged(log.path)
    reading = read_only(tmp_path)
    assert reading.draws == [settled(first), second] and reading.incomplete_last_lines == 0
    assert unchanged(log.path, before)
    assert reading.draws == log.latest()


def test_the_read_only_reader_gives_nothing_for_a_log_that_does_not_exist(tmp_path):
    reading = read_only(tmp_path)
    assert reading.draws == [] and reading.incomplete_last_lines == 0
    assert not draw_log_path("sampling", STAGE, tmp_path).exists()


def test_the_read_only_reader_uses_a_complete_last_line_without_its_newline_and_writes_nothing(tmp_path):
    log = draw_log(tmp_path)
    log(pending(0))
    log(pending(1))
    log.path.write_bytes(log.path.read_bytes().rstrip(b"\n"))
    before = aged(log.path)
    assert not before[0].endswith(b"\n")
    reading = read_only(tmp_path)
    assert reading.draws == [pending(0), pending(1)] and reading.incomplete_last_lines == 0
    assert unchanged(log.path, before)


def test_the_read_only_reader_ignores_and_counts_a_last_line_cut_short_and_writes_nothing(tmp_path):
    log = draw_log(tmp_path)
    log(pending(0))
    with log.path.open("ab") as stream:
        stream.write(CUT)
    before = aged(log.path)
    reading = read_only(tmp_path)
    assert reading.draws == [pending(0)] and reading.incomplete_last_lines == 1
    assert unchanged(log.path, before) and before[0].endswith(CUT)
    # The recording reader, by contrast, repairs the log in place: that is why an analysis does not use it.
    assert log.latest() == [pending(0)] and not log.path.read_bytes().endswith(CUT)


def test_the_read_only_reader_refuses_what_the_log_refuses(tmp_path):
    log = draw_log(tmp_path)
    log(pending(0))
    with pytest.raises(PlanChangedError, match="never merged"):
        read_only(tmp_path, OTHER_PLAN)
    line = json.loads(log.path.read_text(encoding="utf-8"))
    line["key"][3] = 7
    log.path.write_text(json.dumps(line) + "\n", encoding="utf-8")
    with pytest.raises(RecordingError, match="key"):
        read_only(tmp_path)
    line["key"][3] = 0
    line["draw"]["response_text"] = "other reply"
    log.path.write_text(json.dumps(line) + "\n", encoding="utf-8")
    with pytest.raises(RecordingError, match="cannot be read"):
        read_only(tmp_path)
    log.path.write_text(json.dumps(line | {"schema": "other/1"}) + "\n", encoding="utf-8")
    with pytest.raises(RecordingError, match="finance-sim-draw-version/1"):
        read_only(tmp_path)


def test_the_read_only_reader_refuses_an_invalid_line_that_is_not_the_last(tmp_path):
    log = draw_log(tmp_path)
    log(pending(0))
    good = log.path.read_bytes()
    log.path.write_bytes(CUT + b"\n" + good)
    before = aged(log.path)
    with pytest.raises(RecordingError, match="not valid JSON"):
        read_only(tmp_path)
    assert unchanged(log.path, before)


def writing_calls(module) -> set[tuple[str, str]]:
    """Every call in ``module``'s source that writes, moves or removes a file, as (enclosing function, call).

    The analyses of the private records use it to show that the only writes they
    hold are those of their own output files.
    """
    writers = {"write_text", "write_bytes", "truncate", "mkdir", "unlink", "rmdir", "touch", "rename",
               "_write_text", "_write_json", "_append_line", "_repaired_lines"}
    found: set[tuple[str, str]] = set()

    def visit(node: ast.AST, function: str) -> None:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            function = node.name
        if isinstance(node, ast.Call):
            called = node.func
            name = called.attr if isinstance(called, ast.Attribute) else getattr(called, "id", "")
            of_os = isinstance(called, ast.Attribute) and getattr(called.value, "id", "") in ("os", "shutil")
            modes = [value.value for value in [*node.args, *(keyword.value for keyword in node.keywords)]
                     if isinstance(value, ast.Constant) and isinstance(value.value, str)]
            if name == "open" and any(set(mode) & set("wax+") and len(mode) <= 3 for mode in modes):
                found.add((function, "open"))
            elif name in writers or (of_os and name != "fsync"):
                found.add((function, f"os.{name}" if of_os else name))
        for child in ast.iter_child_nodes(node):
            visit(child, function)

    visit(ast.parse(inspect.getsource(module)), "<module>")
    return found


def test_the_read_only_reader_checks_its_arguments(tmp_path):
    with pytest.raises(TypeError, match="plan"):
        read_draw_log(tmp_path / "draws.jsonl", plan=None, loads=CalibrationDraw.from_dict)
    with pytest.raises(TypeError, match="loads"):
        read_draw_log(tmp_path / "draws.jsonl", plan=PLAN, loads=None)
