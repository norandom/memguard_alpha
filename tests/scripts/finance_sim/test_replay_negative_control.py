"""The negative-control report writer of task 8.4 (Requirements 7.2, 7.3 and 10.1) on the 7.9 fixed results.

The published markdown table and the private JSON of
:mod:`scripts.finance_sim.negative_control`, from the fixed item results and
frozen plan of the 7.9 tests: derived statistics with the sources cited, no
release row; written once; rebuilt from stored results alone. Everything is
under ``tmp_path`` and no request is sent.
"""

from __future__ import annotations

import json
import socket

import pytest

from scripts.finance_sim.execute import write_result
from scripts.finance_sim.negative_control import (
    ADJUSTED_VARIANTS,
    DATA_STATEMENT,
    DERIVED_ONLY,
    DISCOUNTING_HONEST_ANSWERS,
    NO_CHANGE,
    NO_RESULT,
    REPORT_COLUMNS,
    NegativeControlError,
    negative_control_change,
    negative_control_from_records,
    report_path,
    write_report,
)
from scripts.finance_sim.plan import SERVING_FULL_DISTRIBUTION_REASON
from tests.scripts.finance_sim.test_negative_control import PLAN, QUARTER_SIZES, RELEASES, scenario, varied


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("the negative-control writer attempted a network connection")

    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)


@pytest.fixture(scope="module")
def report():
    return negative_control_change(PLAN, scenario(), RELEASES, arm="sampling")


def table_rows(markdown: str) -> list[dict[str, str]]:
    lines = [line for line in markdown.splitlines() if line.startswith("|")]
    header = [cell.strip() for cell in lines[0].strip("|").split("|")]
    rows = []
    for line in lines[2:]:
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        rows.append(dict(zip(header, cells, strict=True)))
    return rows


def test_the_table_lists_each_adjusted_variant_with_its_statistics_outcome_flag_and_status_counts(report):
    markdown = report.to_markdown()
    rows = table_rows(markdown)
    assert list(rows[0]) == list(REPORT_COLUMNS)
    assert [row["variant"] for row in rows] == list(ADJUSTED_VARIANTS)
    by_name = {row["variant"]: row for row in rows}
    flagged = report.variant("ensemble_top_k")
    row = by_name["ensemble_top_k"]
    assert row["outcome"] == DISCOUNTING_HONEST_ANSWERS and row["flagged"] == "true"
    assert (row["draw_mode"], row["adjustment"], row["reference"]) == ("ensemble", "top_k", "ensemble_none")
    assert row["releases"] == "40" and row["mean_absolute_change"] == f"{flagged.mean_absolute_change:.6g}"
    assert (row["interval_low"], row["interval_high"]) == (f"{flagged.interval.lo:.6g}", f"{flagged.interval.hi:.6g}")
    assert row["signed_mean_change"] == f"{flagged.signed_mean_change:.6g}"
    assert (row["signed_interval_low"], row["signed_interval_high"]) == (f"{flagged.signed_interval.lo:.6g}",
                                                                        f"{flagged.signed_interval.hi:.6g}")
    assert (row["sign_flips"], row["sign_flip_share"]) == (str(flagged.sign_flips), f"{flagged.sign_flip_share:.6g}")
    assert row["dropped_resamples"] == str(flagged.dropped_resamples) and row["statuses"] == "ok 40"
    assert row["missing"] == ""
    for name in ("single_full_distribution", "ensemble_full_distribution"):
        assert by_name[name]["outcome"] == "no material change" and by_name[name]["flagged"] == "false"
    assert f"Tolerance {PLAN.analysis.tolerance:g}." in markdown
    assert f"block length {PLAN.analysis.negative_control_block_length} quarters" in markdown
    assert f"{PLAN.analysis.n_resamples} resamples, seed {PLAN.seed}" in markdown
    assert "Variants flagged as discounting honest answers: ensemble_top_k." in markdown
    assert f"Releases: 40 frozen held-out control releases over {len(QUARTER_SIZES)} filing quarters" in markdown
    assert "results stored: 40; planned results missing: 0; stage complete." in markdown
    assert report.outcome_rule in markdown and report.method in markdown and report.anchor_note in markdown


def test_the_table_carries_the_citations_and_no_release_row_text_or_id(report):
    markdown = report.to_markdown()
    assert DATA_STATEMENT in markdown and DERIVED_ONLY in markdown
    assert "Moody's" in markdown and "BAA10Y" in markdown and "Financial Modeling Prep (FMP)" in markdown
    assert "SEC EDGAR" in markdown
    for release in RELEASES:
        assert release.item_id not in markdown and release.accession not in markdown
        assert release.release_text not in markdown and release.company not in markdown
    assert "Company" not in markdown
    payload = report.to_dict()
    assert len(payload["variants"][0]["items"]) == 40, "the per-release rows stay in the JSON"


def test_a_missing_variant_and_an_incomplete_stage_show_in_the_table():
    results = scenario(arm="serving")
    report = negative_control_change(PLAN, results[:30], RELEASES, arm="serving")
    assert not report.complete and len(report.missing_results) == 10
    markdown = report.to_markdown()
    rows = {row["variant"]: row for row in table_rows(markdown)}
    assert rows["ensemble_full_distribution"]["missing"] == SERVING_FULL_DISTRIBUTION_REASON
    assert rows["ensemble_full_distribution"]["outcome"] == "" and rows["ensemble_full_distribution"]["flagged"] == ""
    assert rows["ensemble_full_distribution"]["mean_absolute_change"] == ""
    assert rows["ensemble_top_k"]["releases"] == "30"
    assert "results stored: 30; planned results missing: 10; stage incomplete." in markdown
    for release in RELEASES[30:]:
        assert release.item_id not in markdown
        assert report.missing_results[release.item_id] == NO_RESULT
    empty = negative_control_change(PLAN, [], RELEASES, arm="sampling")
    rows = {row["variant"]: row for row in table_rows(empty.to_markdown())}
    assert rows["ensemble_top_k"]["missing"] == NO_CHANGE and rows["ensemble_top_k"]["releases"] == "0"


def test_the_report_is_written_once_as_private_json_and_published_markdown(report, tmp_path):
    json_path, markdown_path = write_report(report, tmp_path)
    assert (json_path, markdown_path) == (tmp_path / "negative-control-sampling.json",
                                          tmp_path / "negative-control-sampling.md")
    assert json_path == report_path("sampling", tmp_path)
    assert json.loads(json_path.read_text(encoding="utf-8")) == json.loads(json.dumps(report.to_dict()))
    assert markdown_path.read_text(encoding="utf-8") == report.to_markdown()
    assert write_report(report, tmp_path) == (json_path, markdown_path)  # the same report again is a no-op
    other = negative_control_change(PLAN, varied(), RELEASES, arm="sampling")
    with pytest.raises(NegativeControlError, match="refusing to replace"):
        write_report(other, tmp_path)
    assert markdown_path.read_text(encoding="utf-8") == report.to_markdown()
    with pytest.raises(TypeError):
        write_report(report.to_dict(), tmp_path)
    with pytest.raises(NegativeControlError, match="unknown arm"):
        report_path("other", tmp_path)


def test_the_report_rebuilds_from_the_stored_results_alone(report, tmp_path):
    results = scenario()
    for result in results:
        write_result(result, tmp_path)
    rebuilt = negative_control_from_records(PLAN, RELEASES, arm="sampling", results_dir=tmp_path)
    assert rebuilt.to_dict() == report.to_dict()
    partial = tmp_path / "partial"
    for result in results[:39]:
        write_result(result, partial)
    incomplete = negative_control_from_records(PLAN, RELEASES, arm="sampling", results_dir=partial)
    assert not incomplete.complete and incomplete.missing_results == {RELEASES[-1].item_id: NO_RESULT}
    assert incomplete.variant("ensemble_top_k").n_items == 39
    with pytest.raises(NegativeControlError, match="missing"):
        negative_control_from_records(PLAN, RELEASES[:-1], arm="sampling", results_dir=tmp_path)
    with pytest.raises(NegativeControlError, match="arm must be"):
        negative_control_from_records(PLAN, RELEASES, arm="other", results_dir=tmp_path)
