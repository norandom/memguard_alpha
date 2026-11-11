"""Catalogue types and the single qualification rule: pure, offline."""

from dataclasses import FrozenInstanceError, replace

import pytest

from recall_guard.core.catalogue import (
    ALPHA,
    CHECK_NAMES,
    MIN_AUC,
    MIN_VALID,
    CatalogueEntry,
    CutoffCheck,
    Evidence,
    Mode,
    ModeRecord,
    Qualification,
    Quality,
    RecallBoundary,
    Status,
    qualify,
)

BOUNDARY = RecallBoundary(
    domain="financial_time_series",
    exposed="months inside the training range",
    fenced="months after the training cutoff",
    date_range="24 months before to 12 months after the cutoff",
)
QUALITY = Quality(
    holdout_auc=0.71,
    interval_low=0.64,
    interval_high=0.78,
    interval_method="prompt_clustered_bootstrap",
    confidence=0.95,
    n_exposed=120,
    n_fenced=110,
    n_prompts=40,
    permutation_p=0.001,
)
CUTOFF = CutoffCheck(
    inside_mean=0.62,
    after_mean=0.41,
    difference=0.21,
    interval_low=0.09,
    interval_high=0.33,
    p_value=0.004,
    n_inside=120,
    n_after=110,
)
EVIDENCE = Evidence(register_ids=("R43",), method="reply_features")


def _record(mode: Mode = Mode.LOGPROBS, **changes: object) -> ModeRecord:
    record = ModeRecord(
        mode=mode,
        available=True,
        unavailable_reason=None,
        requires=None,
        features=("entropy", "top1_probability"),
        features_unavailable=(),
        boundary=BOUNDARY,
        quality=QUALITY,
        cutoff=CUTOFF,
        calibrator_fingerprint="sha256:abc",
        evidence=EVIDENCE,
    )
    return replace(record, **changes)


def test_modes_and_statuses_have_the_design_values() -> None:
    assert [m.value for m in Mode] == ["logprobs", "logits"]
    assert [s.value for s in Status] == ["qualified", "not_qualified", "not_tested"]


def test_thresholds_are_the_rule_of_the_design() -> None:
    assert (MIN_AUC, MIN_VALID, ALPHA) == (0.6, 50, 0.05)
    assert CHECK_NAMES == ("boundary", "sample", "discrimination", "cutoff")


def test_types_are_frozen() -> None:
    with pytest.raises(FrozenInstanceError):
        QUALITY.holdout_auc = 0.9  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        _record().available = False  # type: ignore[misc]


def test_entry_holds_one_record_per_mode() -> None:
    entry = CatalogueEntry(
        entry_id="route-a",
        profile_id=None,
        model="model-a",
        route="route a",
        weights_identity="sha256:weights",
        weights_group="model-a",
        modes={mode: _record(mode) for mode in Mode},
    )
    assert set(entry.modes) == {Mode.LOGPROBS, Mode.LOGITS}
    assert entry.modes[Mode.LOGITS].mode is Mode.LOGITS
    with pytest.raises(FrozenInstanceError):
        entry.model = "other"  # type: ignore[misc]


def test_all_four_checks_true_is_qualified() -> None:
    result = qualify(_record())
    assert isinstance(result, Qualification)
    assert result.status is Status.QUALIFIED
    assert dict(result.checks) == dict.fromkeys(CHECK_NAMES, True)
    assert result.reason


def test_thresholds_are_inclusive_for_auc_and_sample_and_strict_for_p() -> None:
    at_limits = replace(QUALITY, holdout_auc=MIN_AUC, n_exposed=MIN_VALID, n_fenced=MIN_VALID)
    assert qualify(_record(quality=at_limits)).status is Status.QUALIFIED
    p_at_alpha = replace(QUALITY, permutation_p=ALPHA)
    assert qualify(_record(quality=p_at_alpha)).status is Status.NOT_QUALIFIED
    cutoff_at_alpha = replace(CUTOFF, p_value=ALPHA)
    assert qualify(_record(cutoff=cutoff_at_alpha)).status is Status.NOT_QUALIFIED


@pytest.mark.parametrize(
    ("changes", "failed"),
    [
        ({"boundary": replace(BOUNDARY, domain="document_text")}, "boundary"),
        ({"evidence": replace(EVIDENCE, method="document_likelihood")}, "boundary"),
        ({"quality": replace(QUALITY, n_exposed=49)}, "sample"),
        ({"quality": replace(QUALITY, n_fenced=49)}, "sample"),
        ({"quality": replace(QUALITY, holdout_auc=0.59)}, "discrimination"),
        ({"quality": replace(QUALITY, permutation_p=0.2)}, "discrimination"),
        ({"cutoff": replace(CUTOFF, inside_mean=0.40, difference=-0.01)}, "cutoff"),
        ({"cutoff": replace(CUTOFF, p_value=0.3)}, "cutoff"),
    ],
)
def test_each_check_failing_alone_is_not_qualified_with_that_reason(changes: dict[str, object], failed: str) -> None:
    result = qualify(_record(**changes))
    assert result.status is Status.NOT_QUALIFIED
    assert result.reason.startswith(f"{failed}:")
    expected = dict.fromkeys(CHECK_NAMES, True)
    expected[failed] = False
    assert dict(result.checks) == expected


def test_first_failing_check_in_table_order_is_the_reason() -> None:
    result = qualify(
        _record(
            quality=replace(QUALITY, n_exposed=10, holdout_auc=0.5),
            cutoff=replace(CUTOFF, p_value=0.9),
        )
    )
    assert result.status is Status.NOT_QUALIFIED
    assert result.reason.startswith("sample:")
    assert result.checks["discrimination"] is False
    assert result.checks["cutoff"] is False


def test_record_without_quality_is_not_tested() -> None:
    result = qualify(_record(quality=None))
    assert result.status is Status.NOT_TESTED
    assert "quality" in result.reason
    assert result.checks["sample"] is None
    assert result.checks["discrimination"] is None


def test_record_without_quality_is_not_tested_whatever_else_is_recorded() -> None:
    # Design invariant: a record without quality is not_tested (Requirement 1.8).
    result = qualify(_record(quality=None, boundary=replace(BOUNDARY, domain="document_text")))
    assert result.status is Status.NOT_TESTED
    assert result.checks["boundary"] is False


def test_empty_record_is_not_tested() -> None:
    result = qualify(
        _record(
            boundary=None,
            quality=None,
            cutoff=None,
            calibrator_fingerprint=None,
            evidence=None,
        )
    )
    assert result.status is Status.NOT_TESTED
    assert dict(result.checks) == dict.fromkeys(CHECK_NAMES, None)


def test_quality_without_permutation_p_is_not_tested_with_that_reason() -> None:
    result = qualify(_record(quality=replace(QUALITY, permutation_p=None)))
    assert result.status is Status.NOT_TESTED
    assert "permutation p" in result.reason
    assert result.checks["discrimination"] is None
    assert result.checks["cutoff"] is True


def test_quality_without_cutoff_check_is_not_tested_with_that_reason() -> None:
    result = qualify(_record(cutoff=None))
    assert result.status is Status.NOT_TESTED
    assert "cutoff check" in result.reason
    assert result.checks["cutoff"] is None
    assert result.checks["discrimination"] is True


def test_first_missing_input_is_the_reason() -> None:
    result = qualify(_record(quality=replace(QUALITY, permutation_p=None), cutoff=None))
    assert result.status is Status.NOT_TESTED
    assert "permutation p" in result.reason


def test_missing_permutation_p_with_failing_cutoff_is_not_qualified() -> None:
    result = qualify(
        _record(
            quality=replace(QUALITY, permutation_p=None),
            cutoff=replace(CUTOFF, p_value=0.4),
        )
    )
    assert result.status is Status.NOT_QUALIFIED
    assert result.reason.startswith("cutoff:")
    assert result.checks["discrimination"] is None


def test_low_auc_fails_even_when_permutation_p_is_missing() -> None:
    result = qualify(_record(quality=replace(QUALITY, holdout_auc=0.55, permutation_p=None)))
    assert result.status is Status.NOT_QUALIFIED
    assert result.reason.startswith("discrimination:")


def test_document_text_boundary_is_not_qualified() -> None:
    result = qualify(
        _record(
            boundary=replace(BOUNDARY, domain="document_text"),
            evidence=Evidence(register_ids=("R26",), method="document_likelihood"),
        )
    )
    assert result.status is Status.NOT_QUALIFIED
    assert result.reason.startswith("boundary:")


def test_missing_boundary_or_evidence_is_not_tested() -> None:
    for changes in ({"boundary": None}, {"evidence": None}):
        result = qualify(_record(**changes))
        assert result.status is Status.NOT_TESTED
        assert result.checks["boundary"] is None


@pytest.mark.parametrize(
    "changes",
    [
        {},
        {"quality": None},
        {"cutoff": None},
        {"quality": replace(QUALITY, holdout_auc=0.5)},
        {"boundary": replace(BOUNDARY, domain="document_text")},
        {"quality": replace(QUALITY, permutation_p=None), "cutoff": replace(CUTOFF, p_value=0.4)},
    ],
)
def test_the_same_rule_applies_to_both_modes(changes: dict[str, object]) -> None:
    logprobs = qualify(_record(Mode.LOGPROBS, **changes))
    logits = qualify(_record(Mode.LOGITS, **changes))
    assert logprobs == logits


def test_availability_does_not_enter_the_rule() -> None:
    unavailable = _record(available=False, unavailable_reason="route returns no logprobs")
    assert qualify(unavailable) == qualify(_record())


def test_qualify_is_pure() -> None:
    record = _record()
    assert qualify(record) == qualify(record)
    assert record == _record()
