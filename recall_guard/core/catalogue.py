"""Qualification catalogue: entry and per-mode record types, and the single rule.

The catalogue records what was shown about each model on a serving route, per
evidence mode. A status is never stored: :func:`qualify` derives it from the
recorded inputs, so a status cannot disagree with its evidence.

The qualification rule, stated once. Four checks, evaluated in this order:

1. ``boundary``: the calibrator's recall boundary is a financial time series
   and its evidence method is reply features.
2. ``sample``: at least ``MIN_VALID`` valid draws in each class.
3. ``discrimination``: held-out AUC at least ``MIN_AUC`` and permutation p
   below ``ALPHA``.
4. ``cutoff``: the estimate inside the training range is above the one after
   the cutoff, with one-sided p below ``ALPHA``.

A check whose input was recorded and fails gives ``not_qualified`` with the
first failing check as the reason. If no check fails and an input is not
recorded, the status is ``not_tested`` with the first missing input as the
reason: a missing result is never read as a failure and never inferred. All
four true gives ``qualified``. The rule is the same for both modes.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum

MIN_AUC = 0.6
MIN_VALID = 50
ALPHA = 0.05

QUALIFYING_DOMAIN = "financial_time_series"
QUALIFYING_METHOD = "reply_features"

CHECK_NAMES = ("boundary", "sample", "discrimination", "cutoff")


class Mode(StrEnum):
    """Evidence mode of a route."""

    LOGPROBS = "logprobs"  # maps to EstimateSource.TOP_K
    LOGITS = "logits"  # maps to EstimateSource.FULL_DISTRIBUTION


class Status(StrEnum):
    """The one status of an entry and mode."""

    QUALIFIED = "qualified"
    NOT_QUALIFIED = "not_qualified"
    NOT_TESTED = "not_tested"


@dataclass(frozen=True)
class Evidence:
    """Register entries behind a record, and the method that produced them."""

    register_ids: tuple[str, ...]
    method: str  # "reply_features" | "document_likelihood"


@dataclass(frozen=True)
class RecallBoundary:
    """What the calibrator was fitted on."""

    domain: str  # "financial_time_series" | "document_text"
    exposed: str  # what stands for recall-exposed answers
    fenced: str  # what stands for fenced answers
    date_range: str  # relative to the training cutoff


@dataclass(frozen=True)
class Quality:
    """Held-out quality of a calibrator, with its interval and sample sizes."""

    holdout_auc: float
    interval_low: float
    interval_high: float
    interval_method: str  # "prompt_clustered_bootstrap" | "row_bootstrap"
    confidence: float
    n_exposed: int
    n_fenced: int
    n_prompts: int | None
    permutation_p: float | None


@dataclass(frozen=True)
class CutoffCheck:
    """Recall estimate inside the training range against after the cutoff."""

    inside_mean: float
    after_mean: float
    difference: float
    interval_low: float
    interval_high: float
    p_value: float
    n_inside: int
    n_after: int


@dataclass(frozen=True)
class ModeRecord:
    """What is recorded for one entry in one mode."""

    mode: Mode
    available: bool
    unavailable_reason: str | None
    requires: str | None  # what the adopter needs for this mode
    features: tuple[str, ...]
    features_unavailable: tuple[str, ...]
    boundary: RecallBoundary | None
    quality: Quality | None
    cutoff: CutoffCheck | None
    calibrator_fingerprint: str | None
    evidence: Evidence | None


@dataclass(frozen=True)
class CatalogueEntry:
    """One model on one serving route."""

    entry_id: str  # profile id, or a route slug
    profile_id: str | None
    model: str
    route: str
    weights_identity: str | None
    weights_group: str | None  # entries sharing weights
    modes: Mapping[Mode, ModeRecord]


@dataclass(frozen=True)
class Qualification:
    """Outcome of the rule: status, reason, and each check (None: not recorded)."""

    status: Status
    reason: str
    checks: Mapping[str, bool | None]


def _boundary_check(record: ModeRecord) -> tuple[bool | None, str]:
    boundary, evidence = record.boundary, record.evidence
    if boundary is not None and boundary.domain != QUALIFYING_DOMAIN:
        return False, f"recall boundary domain is {boundary.domain}, not {QUALIFYING_DOMAIN}"
    if evidence is not None and evidence.method != QUALIFYING_METHOD:
        return False, f"evidence method is {evidence.method}, not {QUALIFYING_METHOD}"
    if boundary is None:
        return None, "no recall boundary recorded"
    if evidence is None:
        return None, "no evidence recorded"
    return True, ""


def _sample_check(record: ModeRecord) -> tuple[bool | None, str]:
    quality = record.quality
    if quality is None:
        return None, "no quality record"
    if quality.n_exposed < MIN_VALID or quality.n_fenced < MIN_VALID:
        return False, (
            f"{quality.n_exposed} exposed and {quality.n_fenced} fenced valid draws, fewer than {MIN_VALID} in a class"
        )
    return True, ""


def _discrimination_check(record: ModeRecord) -> tuple[bool | None, str]:
    quality = record.quality
    if quality is None:
        return None, "no quality record"
    if not quality.holdout_auc >= MIN_AUC:
        return False, f"held-out AUC {quality.holdout_auc:g} is below {MIN_AUC:g}"
    if quality.permutation_p is None:
        return None, "no permutation p recorded"
    if not quality.permutation_p < ALPHA:
        return False, f"permutation p {quality.permutation_p:g} is not below {ALPHA:g}"
    return True, ""


def _cutoff_check(record: ModeRecord) -> tuple[bool | None, str]:
    cutoff = record.cutoff
    if cutoff is None:
        return None, "no cutoff check recorded"
    if not cutoff.inside_mean > cutoff.after_mean:
        return False, (
            f"estimate inside the training range ({cutoff.inside_mean:g}) is not above "
            f"the one after the cutoff ({cutoff.after_mean:g})"
        )
    if not cutoff.p_value < ALPHA:
        return False, f"one-sided p {cutoff.p_value:g} is not below {ALPHA:g}"
    return True, ""


_CHECKS = (_boundary_check, _sample_check, _discrimination_check, _cutoff_check)


def qualify(record: ModeRecord) -> Qualification:
    """Apply the qualification rule to one mode record.

    Pure function of the record; the mode and the availability do not enter.
    """
    outcomes = [check(record) for check in _CHECKS]
    checks: dict[str, bool | None] = {name: outcome for name, (outcome, _) in zip(CHECK_NAMES, outcomes, strict=True)}
    if record.quality is None:
        # No held-out result exists for this entry and mode: nothing was tested.
        return Qualification(Status.NOT_TESTED, "no quality record", checks)
    for name, (outcome, detail) in zip(CHECK_NAMES, outcomes, strict=True):
        if outcome is False:
            return Qualification(Status.NOT_QUALIFIED, f"{name}: {detail}", checks)
    for name, (outcome, detail) in zip(CHECK_NAMES, outcomes, strict=True):
        if outcome is None:
            return Qualification(Status.NOT_TESTED, f"{name}: {detail}", checks)
    return Qualification(Status.QUALIFIED, "all four checks passed", checks)
