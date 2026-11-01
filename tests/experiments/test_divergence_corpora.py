"""Local corpus admission preserves provenance failures and human audit limits."""

import hashlib
from dataclasses import replace

import pytest

from experiments.divergence_evaluation.contracts import Cohort, CorpusDocument, Split
from experiments.divergence_evaluation.corpora import (
    CorpusEntry,
    CorpusManifest,
    FactExposure,
    SemanticAudit,
    matched_size_control,
    validate_corpora,
)


def entry(tmp_path, name, *, source="wire", tokens=10, published="2019-01-01T00:00:00Z",
          revised=None, split=Split.FITTING, facts=()):
    content = f"Unique document {name}".encode()
    (tmp_path / name).write_bytes(content)
    document = CorpusDocument(name, source, hashlib.sha256(content).hexdigest(),
                              "licenses/wire-permission", published, revised or published,
                              Cohort.HISTORICAL, split, facts)
    return CorpusEntry(document, name, tokens)


def manifest(entries, older=None, newer=None, exposures=()):
    ids = tuple(item.document.document_id for item in entries)
    return CorpusManifest(tuple(entries), older or ids, newer or ids,
                          SemanticAudit("reviewer", "audit/decision-1", "2026-11-01T00:00:00Z",
                                        ids, True, True), exposures)


def test_valid_manifest_replays_source_period_counts_and_json(tmp_path):
    old = entry(tmp_path, "old", tokens=8)
    later = entry(tmp_path, "later", tokens=12, published="2024-06-30T23:59:59Z")
    data = manifest([old, later], ("old",), ("old", "later"))
    assert CorpusManifest.from_json(data.to_json()) == data
    report = validate_corpora(data, tmp_path)
    assert report.admitted
    assert report.older_counts == (("wire", "older", 1, 8),)
    assert report.newer_counts == (("wire", "later", 1, 12), ("wire", "older", 1, 8))


@pytest.mark.parametrize("changes,reason", [
    ({"rights_reference": " "}, "rights"),
    ({"published_at": "2019-01-01"}, "timestamp"),
    ({"revised_at": "2018-01-01T00:00:00Z"}, "revision_before_publication"),
    ({"revised_at": "2020-01-01T00:00:00Z"}, "older_window"),
    ({"published_at": "2024-07-01T00:00:00Z", "revised_at": "2024-07-01T00:00:00Z"}, "newer_window"),
    ({"content_hash": "a" * 64}, "content_hash"),
])
def test_provenance_rejections_are_recorded(tmp_path, changes, reason):
    item = entry(tmp_path, "old")
    item = replace(item, document=replace(item.document, **changes))
    report = validate_corpora(manifest([item]), tmp_path)
    assert not report.admitted
    assert any(reason in issue.code for issue in report.issues)
    assert type(report).from_json(report.to_json()) == report


def test_all_failures_retained_and_local_paths_cannot_escape(tmp_path):
    item = entry(tmp_path, "old")
    broken = replace(item, content_path="../outside", token_count=0,
                     document=replace(item.document, rights_reference="", revised_at="missing"))
    report = validate_corpora(manifest([broken]), tmp_path)
    assert {issue.code for issue in report.issues} >= {"rights", "timestamp", "local_path", "token_count"}


def test_newer_must_include_older_and_fit_ids_must_be_fitting(tmp_path):
    old = entry(tmp_path, "old")
    heldout = entry(tmp_path, "answer", split=Split.EVALUATION)
    report = validate_corpora(manifest([old, heldout], ("old",), ("answer",)), tmp_path)
    assert {issue.code for issue in report.issues} >= {"older_not_in_newer", "fit_partition"}


def test_duplicate_text_and_ids_rejected_even_with_intentional_facts(tmp_path):
    old = entry(tmp_path, "old", facts=("event",))
    heldout = entry(tmp_path, "answer", split=Split.EVALUATION, facts=("event",))
    heldout = replace(heldout, document=replace(heldout.document, content_hash=old.document.content_hash),
                      content_path="old")
    exposure = FactExposure("event", ("old", "answer"), "Historical recall contrast", "audit/fact-1")
    report = validate_corpora(manifest([old, heldout], ("old",), ("old",), (exposure,)), tmp_path)
    assert "split_text_overlap" in {issue.code for issue in report.issues}
    duplicated = validate_corpora(manifest([old, old]), tmp_path)
    assert "duplicate_document_id" in {issue.code for issue in duplicated.issues}


def test_semantic_audit_and_shared_fact_exposure_are_separate_requirements(tmp_path):
    old = entry(tmp_path, "old", facts=("event",))
    answer = entry(tmp_path, "answer", split=Split.EVALUATION, facts=("event",))
    data = manifest([old, answer], ("old",), ("old",))
    report = validate_corpora(replace(data, semantic_audit=None), tmp_path)
    assert {issue.code for issue in report.issues} >= {"semantic_audit", "unattested_fact_exposure"}
    exposure = FactExposure("event", ("old", "answer"), "Intentional recall evidence", "audit/fact-1")
    report = validate_corpora(replace(data, intentional_fact_exposures=(exposure,)), tmp_path)
    assert report.admitted
    assert report.intentional_fact_exposures == (exposure,)
    incomplete = replace(data.semantic_audit, document_ids=("old",), near_duplicates_clear=False)
    assert not validate_corpora(replace(data, semantic_audit=incomplete), tmp_path).admitted


def test_control_is_deterministic_source_period_stratified_and_whole_document(tmp_path):
    reference = (entry(tmp_path, "ref-a", tokens=10), entry(tmp_path, "ref-b", source="filings", tokens=9))
    candidates = (entry(tmp_path, "a1", tokens=6), entry(tmp_path, "a2", tokens=6),
                  entry(tmp_path, "b1", source="filings", tokens=8),
                  entry(tmp_path, "later", tokens=10, published="2020-01-01T00:00:00Z"))
    result = matched_size_control(reference, candidates, seed=7)
    assert result == matched_size_control(tuple(reversed(reference)), tuple(reversed(candidates)), seed=7)
    assert "later" in result.document_ids
    assert result.target_tokens == 19
    assert result.actual_tokens == 24
    assert result.discrepancy_tokens == 5
    assert result.counts_by_source_window == (("filings", "older", 1, 8), ("wire", "later", 1, 10), ("wire", "older", 1, 6))
    assert type(result).from_json(result.to_json()) == result


def test_empty_control_candidates_report_size_shortfall(tmp_path):
    reference = (entry(tmp_path, "ref", source="missing", tokens=10),)
    result = matched_size_control(reference, (), seed=0)
    assert result.actual_tokens == 0
    assert result.discrepancy_tokens == -10
    assert result.stratum_discrepancies == ()


def test_control_rejects_heldout_and_post_window_candidates(tmp_path):
    reference = (entry(tmp_path, "ref"),)
    for candidate in (entry(tmp_path, "heldout", split=Split.EVALUATION),
                      entry(tmp_path, "late", published="2024-07-01T00:00:00Z")):
        with pytest.raises(ValueError, match="admitted fitting"):
            matched_size_control(reference, (candidate,), seed=0)


def test_timestamp_boundaries_are_utc_and_missing_bytes_are_retained(tmp_path):
    item = entry(tmp_path, "old", published="2020-01-01T00:30:00+01:00")
    data = manifest([item])
    assert validate_corpora(data, tmp_path).admitted
    (tmp_path / "old").unlink()
    assert "local_content" in {issue.code for issue in validate_corpora(data, tmp_path).issues}

