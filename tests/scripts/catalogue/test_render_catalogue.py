"""Catalogue page renderer (qualified-model-catalogue task 2.4).

Requirements 1.4, 1.5, 3.1, 3.2 and 3.3. The page is rendered from the fixture
catalogue only; the command-line entry writes under ``tmp_path``.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from recall_guard.core.catalogue import ALPHA, MIN_AUC, MIN_VALID, Catalogue, Mode, load_catalogue
from scripts.catalogue.render_catalogue import _number, main, render_catalogue

FIXTURE = Path(__file__).resolve().parents[2] / "fixtures" / "catalogue" / "catalogue_fixture.json"
EM_DASH = "—"


@pytest.fixture(scope="module")
def catalogue() -> Catalogue:
    return load_catalogue(FIXTURE)


@pytest.fixture(scope="module")
def page(catalogue: Catalogue) -> str:
    return render_catalogue(catalogue)


def _section(page: str, heading: str) -> str:
    """The text of one second-level section, without the following sections."""
    marker = f"\n## {heading}\n"
    assert marker in page, f"missing section {heading!r}"
    body = page.split(marker, 1)[1]
    return body.split("\n## ", 1)[0]


def _rows(text: str, *cells: str) -> list[str]:
    """Table rows of the text that contain every given cell text."""
    return [line for line in text.splitlines() if line.startswith("|") and all(cell in line for cell in cells)]


def test_page_opens_with_a_summary_of_entries_and_statuses(page: str) -> None:
    assert page.startswith("# Model catalogue\n\n")
    summary = page.split("\n## ", 1)[0]
    assert "4 entries" in summary
    assert "8 entry and mode combinations" in summary
    assert "2 are qualified, 1 is not qualified and 5 are not tested" in summary


def test_sections_appear_in_order_with_sentence_case_headings(page: str) -> None:
    headings = [line for line in page.splitlines() if line.startswith("#")]
    assert headings == [
        "# Model catalogue",
        "## Qualification rule",
        "## Catalogue table",
        "### Other recorded evidence",
        "## Entries without logprobs",
        "### Logprobs mode unavailable",
        "### Not usable for recall-aware answers",
        "### Entries with a logits mode",
        "## Related entries",
    ]


def test_rule_is_stated_once_with_the_thresholds_of_the_module(page: str) -> None:
    rule = _section(page, "Qualification rule")
    assert page.count("## Qualification rule") == 1
    for name in ("boundary", "sample", "discrimination", "cutoff"):
        assert f"**{name}**" in rule
    assert f"at least {MIN_VALID} valid draws in each class" in rule
    assert f"held-out AUC is at least {MIN_AUC:g}" in rule
    assert rule.count(f"below {ALPHA:g}") == 2
    assert "financial_time_series" in rule and "reply_features" in rule
    assert "not qualified" in rule and "not tested" in rule and "qualified" in rule
    assert "same rule" in rule


def test_rule_text_follows_the_order_in_which_the_status_is_decided(page: str) -> None:
    rule = _section(page, "Qualification rule")
    steps = [
        "A mode with no held-out quality record is **not tested**, whatever else is recorded.",
        "Otherwise, when a quality record exists and a check fails on the recorded values, "
        "the mode is **not qualified**.",
        "Otherwise, when an input of a check is not recorded, the mode is **not tested**.",
        "Otherwise all 4 checks pass and the mode is **qualified**.",
    ]
    positions = [rule.find(step) for step in steps]
    assert all(position >= 0 for position in positions), positions
    assert positions == sorted(positions)
    assert "The reason names the first failing check, in the order of the checks above." in rule
    assert "The reason names the first missing input, in the same order." in rule


def test_rule_text_agrees_with_the_status_of_a_failing_record_without_quality(catalogue: Catalogue) -> None:
    """A record that would fail the boundary check but has no quality record is not tested."""
    entry = catalogue.entry("llamacpp-gemma3-4b")
    record = entry.modes[Mode.LOGPROBS]
    boundary = replace(record.boundary, domain="document_text")
    modes = {**entry.modes, Mode.LOGPROBS: replace(record, boundary=boundary, quality=None)}
    page = render_catalogue(Catalogue(entries=(replace(entry, modes=modes),)))
    table = _section(page, "Catalogue table").split("\n### ", 1)[0]
    (row,) = _rows(table, "| llamacpp-gemma3-4b |", "| logprobs |")
    assert "| not tested: no quality record |" in row


def test_numbers_are_printed_as_recorded_without_scientific_notation(catalogue: Catalogue) -> None:
    entry = catalogue.entry("llamacpp-gemma3-4b")
    record = entry.modes[Mode.LOGPROBS]
    quality = replace(record.quality, holdout_auc=0.5833333333333333, interval_high=1.0, permutation_p=0.00001)
    modes = {**entry.modes, Mode.LOGPROBS: replace(record, quality=quality)}
    page = render_catalogue(Catalogue(entries=(replace(entry, modes=modes),)))
    table = _section(page, "Catalogue table").split("\n### ", 1)[0]
    (row,) = _rows(table, "| llamacpp-gemma3-4b |", "| logprobs |")
    cells = [cell.strip() for cell in row.strip("|").split("|")]
    assert cells[6] == "0.5833333333333333 [0.64, 1.0]"
    assert cells[8] == "exposed 120, fenced 110, prompts 40"
    assert cells[9] == "0.00001"
    assert float(cells[9]) == 0.00001
    assert "e-0" not in row


@pytest.mark.parametrize(
    ("value", "text"),
    [
        (0.722, "0.722"),
        (0.5833333333333333, "0.5833333333333333"),
        (1.0, "1.0"),
        (0.00001, "0.00001"),
        (1.5e-7, "0.00000015"),
        (1e22, "10000000000000000000000.0"),
        (-0.25, "-0.25"),
        (0.0, "0.0"),
        (120, "120"),
        (0, "0"),
    ],
)
def test_number_text_round_trips_to_the_recorded_value(value: float | int, text: str) -> None:
    printed = _number(value)
    assert printed == text
    assert type(value)(printed) == value


def test_table_has_one_row_per_entry_and_mode(page: str, catalogue: Catalogue) -> None:
    table = _section(page, "Catalogue table").split("\n### ", 1)[0]
    for entry in catalogue.entries:
        for mode in Mode:
            assert len(_rows(table, f"| {entry.entry_id} |", f"| {mode.value} |")) == 1
    assert len([line for line in table.splitlines() if line.startswith("|")]) == 2 + 8


def test_table_row_shows_quality_interval_samples_cutoff_status_and_evidence(page: str) -> None:
    table = _section(page, "Catalogue table").split("\n### ", 1)[0]
    (row,) = _rows(table, "| llamacpp-gemma3-4b |", "| logprobs |")
    cells = [cell.strip() for cell in row.strip("|").split("|")]
    assert cells == [
        "llamacpp-gemma3-4b",
        "gemma3:4b",
        "llama.cpp server, local",
        "logprobs",
        "yes",
        "financial_time_series; exposed: months inside the training range; "
        "fenced: months after the training cutoff; range: 24 months before to 12 months after the cutoff",
        "0.71 [0.64, 0.78]",
        "prompt_clustered_bootstrap, 0.95",
        "exposed 120, fenced 110, prompts 40",
        "0.001",
        "inside 0.62, after 0.41, difference 0.21 [0.09, 0.33], p 0.004, n 120 and 110",
        "qualified: all four checks passed",
        "[R43](research-register.md), [R40](research-register.md)",
    ]


def test_table_shows_each_status_with_its_reason(page: str) -> None:
    table = _section(page, "Catalogue table").split("\n### ", 1)[0]
    (failed,) = _rows(table, "| llamacpp-gemma3-4b |", "| logits |")
    assert "| not qualified: discrimination: held-out AUC 0.55 is below 0.6 |" in failed
    assert "| 0.55 [0.47, 0.63] |" in failed
    assert "| 0.2 |" in failed
    (untested,) = _rows(table, "| ollama-gemma3-4b |", "| logprobs |")
    assert "| not tested: no quality record |" in untested
    (passed,) = _rows(table, "| oga-phi4-mini |", "| logits |")
    assert "| qualified: all four checks passed |" in passed
    assert "[R41](research-register.md)" in passed


def test_unavailable_mode_shows_its_reason_and_missing_values_stay_empty(page: str) -> None:
    table = _section(page, "Catalogue table").split("\n### ", 1)[0]
    (row,) = _rows(table, "| ollama-gemma3-4b |", "| logits |")
    cells = [cell.strip() for cell in row.strip("|").split("|")]
    assert cells[4] == "no: the route returns at most the top 20 tokens"
    assert cells[5:11] == [""] * 6
    assert cells[11] == "not tested: no quality record"
    assert cells[12] == ""
    assert "An empty cell means that the value is not recorded." in page


def test_other_evidence_is_listed_apart_with_its_method(page: str) -> None:
    other = _section(page, "Catalogue table").split("\n### Other recorded evidence\n", 1)[1]
    rows = [line for line in other.splitlines() if line.startswith("|")]
    assert len(rows) == 2 + 1
    assert "| Method |" in rows[0]
    record = rows[2]
    assert "| llamacpp-gemma3-4b |" in record and "| logprobs |" in record
    assert "| document_likelihood |" in record
    assert "document_text; exposed: documents published before the cutoff" in record
    assert "| 0.9 [0.85, 0.95] |" in record
    assert "| row_bootstrap, 0.95 |" in record
    assert "| exposed 200, fenced 200 |" in record
    assert "not qualified: boundary: recall boundary domain is document_text" in record
    assert "[R26](research-register.md)" in record
    assert "do not set the status" in other
    # The document-text result is not in the qualifying table.
    assert "document_likelihood" not in _section(page, "Catalogue table").split("\n### ", 1)[0]


def test_every_evidence_id_of_the_fixture_appears_as_a_register_link(page: str) -> None:
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    ids = {
        register_id
        for entry in data["entries"]
        for mode in entry["modes"].values()
        for calibration in mode["calibrations"]
        for register_id in calibration["evidence"]["register_ids"]
    }
    assert ids == {"R26", "R40", "R41", "R43"}
    for register_id in ids:
        assert f"[{register_id}](research-register.md)" in page


def test_entries_without_logprobs_state_reason_logits_availability_and_requirement(page: str) -> None:
    section = _section(page, "Entries without logprobs")
    unavailable = section.split("\n### Logprobs mode unavailable\n", 1)[1].split("\n### ", 1)[0]
    rows = [line for line in unavailable.splitlines() if line.startswith("|")]
    assert len(rows) == 2 + 2
    (oga,) = _rows(unavailable, "| oga-phi4-mini |")
    assert "| the runtime has no logprobs field in its replies |" in oga
    assert "| yes |" in oga
    assert "| qualified: all four checks passed |" in oga
    assert "| in-process access to the logits of each generated token |" in oga
    (closed,) = _rows(unavailable, "| closed-route |")
    assert "| the API returns text only |" in closed
    assert "| no: the weights are not published |" in closed
    assert "| published weights and a runtime that exposes logits |" in closed
    assert not _rows(unavailable, "| ollama-gemma3-4b |")


def test_entry_with_neither_mode_is_shown_as_not_usable(page: str) -> None:
    section = _section(page, "Entries without logprobs")
    unusable = section.split("\n### Not usable for recall-aware answers\n", 1)[1].split("\n### ", 1)[0]
    rows = [line for line in unusable.splitlines() if line.startswith("|")]
    assert len(rows) == 2 + 1
    assert "| closed-route |" in rows[2]
    assert "Neither mode is available" in unusable
    assert "not usable for recall-aware answers" in unusable
    assert "The reasons are in the table above." in unusable


def test_each_unavailability_reason_is_stated_once_in_the_section(page: str) -> None:
    section = _section(page, "Entries without logprobs")
    assert section.count("the API returns text only") == 1
    assert section.count("the weights are not published") == 1
    assert section.count("published weights and a runtime that exposes logits") == 1


def test_entries_with_a_logits_mode_are_listed_under_a_heading_that_names_the_list(
    page: str, catalogue: Catalogue
) -> None:
    section = _section(page, "Entries without logprobs")
    assert "Usable without logprobs" not in page
    usable = section.split("\n### Entries with a logits mode\n", 1)[1]
    assert "can be used when logprobs are not wanted or not available" in usable
    (listed,) = _rows(usable, "| llamacpp-gemma3-4b |")
    assert "| yes |" in listed
    rows = [line for line in usable.splitlines() if line.startswith("|")][2:]
    assert [row.split("|")[1].strip() for row in rows] == [entry.entry_id for entry in catalogue.without_logprobs()]
    assert len(rows) == 2
    (first,) = _rows(usable, "| llamacpp-gemma3-4b |")
    assert "| not qualified: discrimination: held-out AUC 0.55 is below 0.6 |" in first
    assert "| a llama.cpp server started with the same GGUF file |" in first
    (second,) = _rows(usable, "| oga-phi4-mini |")
    assert "| qualified: all four checks passed |" in second


def test_related_entries_are_grouped_by_shared_weights(page: str) -> None:
    related = _section(page, "Related entries")
    rows = [line for line in related.splitlines() if line.startswith("|")][2:]
    assert len(rows) == 2
    assert all("| gemma3-4b |" in row for row in rows)
    assert "| llamacpp-gemma3-4b |" in rows[0] and "| sha256:fixture-gemma3-4b |" in rows[0]
    assert "| ollama-gemma3-4b |" in rows[1]
    assert "oga-phi4-mini" not in related and "closed-route" not in related


def test_rendering_twice_gives_identical_bytes(catalogue: Catalogue) -> None:
    first = render_catalogue(catalogue).encode("utf-8")
    second = render_catalogue(load_catalogue(FIXTURE)).encode("utf-8")
    assert first == second


def test_output_has_no_em_dash_and_only_lf_newlines(page: str) -> None:
    assert EM_DASH not in page
    assert "\r" not in page
    assert page.endswith("\n") and not page.endswith("\n\n")
    assert all(line == line.rstrip() for line in page.splitlines())


def test_empty_lists_are_stated_in_a_plain_sentence(catalogue: Catalogue) -> None:
    only = Catalogue(entries=(catalogue.entry("ollama-gemma3-4b"),))
    page = render_catalogue(only)
    assert "1 entry" in page.split("\n## ", 1)[0]
    section = _section(page, "Entries without logprobs")
    assert "No entry has an unavailable logprobs mode." in section
    assert "Every entry has at least one available mode." in section
    assert "No entry has an available logits mode." in section
    assert "No two entries share weights." in _section(page, "Related entries")
    assert "No further evidence is recorded." in _section(page, "Catalogue table")


def test_missing_optional_values_do_not_crash_the_renderer(catalogue: Catalogue) -> None:
    entry = catalogue.entry("closed-route")
    bare = {mode: replace(record, unavailable_reason=None, requires=None) for mode, record in entry.modes.items()}
    page = render_catalogue(Catalogue(entries=(replace(entry, modes=bare),)))
    (row,) = _rows(_section(page, "Catalogue table"), "| closed-route |", "| logits |")
    assert "| no (reason not recorded) |" in row
    unusable = _section(page, "Entries without logprobs")
    assert "| not recorded |" in unusable
    assert "| 0 |" not in page


def test_text_with_a_pipe_does_not_break_a_table_row(catalogue: Catalogue) -> None:
    entry = replace(catalogue.entry("oga-phi4-mini"), route="in process | no server")
    page = render_catalogue(Catalogue(entries=(entry,)))
    (row,) = _rows(_section(page, "Catalogue table").split("\n### ", 1)[0], "| oga-phi4-mini |", "| logits |")
    assert "in process \\| no server" in row
    assert len(row.replace("\\|", "").strip("|").split("|")) == 13


def test_command_line_entry_writes_the_page_with_lf_and_utf8(tmp_path: Path, page: str) -> None:
    out = tmp_path / "nested" / "catalogue.md"
    assert main(["--catalogue", str(FIXTURE), "--out", str(out)]) == 0
    written = out.read_bytes()
    assert written == page.encode("utf-8")
    assert b"\r" not in written


def test_command_line_entry_reports_an_unreadable_catalogue(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    out = tmp_path / "catalogue.md"
    assert main(["--catalogue", str(tmp_path / "absent.json"), "--out", str(out)]) == 2
    assert not out.exists()
    assert "absent.json" in capsys.readouterr().err
