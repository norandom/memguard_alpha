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

:func:`load_catalogue` reads and validates a catalogue data file (schema
``recall-guard-catalogue/1``) on call; importing this module reads no file.
The data file stores the inputs of the rule, never a status. For an entry with
a provider profile, the availability of a mode comes from the profile's
estimate sources; the data file adds only the reason and the requirement.
"""

import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, replace
from enum import StrEnum
from importlib import resources
from pathlib import Path
from types import MappingProxyType

from recall_guard.core.profiles import (
    FULL_DISTRIBUTION_SOURCE,
    TOP_K_SOURCE,
    ProfileError,
    available_profiles,
    load_profile,
)

MIN_AUC = 0.6
MIN_VALID = 50
ALPHA = 0.05

QUALIFYING_DOMAIN = "financial_time_series"
QUALIFYING_METHOD = "reply_features"

CHECK_NAMES = ("boundary", "sample", "discrimination", "cutoff")

CATALOGUE_SCHEMA = "recall-guard-catalogue/1"
CATALOGUE_RESOURCE = "catalogue_data.json"
# Recorded means and their difference are rounded separately; they must agree to this.
DIFFERENCE_TOLERANCE = 1e-3


class CatalogueError(ValueError):
    """The catalogue data is missing or invalid, or an entry, mode or status is unknown."""


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
    # Further calibrations recorded for the same entry and mode (for example a
    # document-text one), each as a record of its own; the rule does not read them.
    other_evidence: tuple["ModeRecord", ...] = ()


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


# --- loading and validation --------------------------------------------------

_SOURCE_OF_MODE = MappingProxyType({Mode.LOGPROBS: TOP_K_SOURCE, Mode.LOGITS: FULL_DISTRIBUTION_SOURCE})
_REGISTER_ID = re.compile(r"R\d+")

_TOP_KEYS = frozenset({"schema", "rule", "entries"})
_RULE = MappingProxyType({"min_auc": MIN_AUC, "min_valid": MIN_VALID, "alpha": ALPHA})
_ENTRY_KEYS = frozenset({"entry_id", "profile_id", "model", "route", "weights_identity", "weights_group", "modes"})
_MODE_KEYS = frozenset(
    {"available", "unavailable_reason", "requires", "features", "features_unavailable", "calibrations"}
)
_CALIBRATION_KEYS = frozenset({"boundary", "quality", "cutoff", "calibrator_fingerprint", "evidence"})
_BOUNDARY_KEYS = frozenset({"domain", "exposed", "fenced", "date_range"})
_QUALITY_KEYS = frozenset(
    {
        "holdout_auc",
        "interval_low",
        "interval_high",
        "interval_method",
        "confidence",
        "n_exposed",
        "n_fenced",
        "n_prompts",
        "permutation_p",
    }
)
_CUTOFF_KEYS = frozenset(
    {"inside_mean", "after_mean", "difference", "interval_low", "interval_high", "p_value", "n_inside", "n_after"}
)
_EVIDENCE_KEYS = frozenset({"register_ids", "method"})


def _exact_keys(data: object, expected: frozenset[str], where: str) -> Mapping:
    if not isinstance(data, Mapping):
        raise CatalogueError(f"{where}: must be an object")
    unknown, missing = sorted(map(str, data.keys() - expected)), sorted(expected - data.keys())
    if unknown or missing:
        raise CatalogueError(f"{where}: unknown keys {unknown}, missing keys {missing}")
    return data


def _text(data: Mapping, key: str, where: str, *, optional: bool = False) -> str | None:
    value = data[key]
    if value is None and optional:
        return None
    if not isinstance(value, str) or not value.strip():
        raise CatalogueError(f"{where}: {key} must be non-empty text{' or null' if optional else ''}")
    return value


def _texts(data: Mapping, key: str, where: str) -> tuple[str, ...]:
    value = data[key]
    if not isinstance(value, list) or not all(isinstance(item, str) and item.strip() for item in value):
        raise CatalogueError(f"{where}: {key} must be a list of non-empty text")
    return tuple(value)


def _number(data: Mapping, key: str, where: str, *, optional: bool = False) -> float | None:
    value = data[key]
    if value is None and optional:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise CatalogueError(f"{where}: {key} must be a finite number{' or null' if optional else ''}")
    return value


def _unit(data: Mapping, key: str, where: str, *, optional: bool = False) -> float | None:
    value = _number(data, key, where, optional=optional)
    if value is not None and not 0.0 <= value <= 1.0:
        raise CatalogueError(f"{where}: {key} must be in [0, 1]")
    return value


def _count(data: Mapping, key: str, where: str, *, optional: bool = False) -> int | None:
    value = data[key]
    if value is None and optional:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise CatalogueError(f"{where}: {key} must be a non-negative integer{' or null' if optional else ''}")
    return value


def _interval(data: Mapping, where: str) -> tuple[float, float]:
    low, high = _number(data, "interval_low", where), _number(data, "interval_high", where)
    if low > high:
        raise CatalogueError(f"{where}: interval_low {low:g} is above interval_high {high:g}")
    return low, high


def _boundary(data: object, where: str) -> RecallBoundary | None:
    if data is None:
        return None
    data = _exact_keys(data, _BOUNDARY_KEYS, where)
    return RecallBoundary(**{key: _text(data, key, where) for key in sorted(_BOUNDARY_KEYS)})


def _quality(data: object, where: str) -> Quality | None:
    if data is None:
        return None
    data = _exact_keys(data, _QUALITY_KEYS, where)
    low, high = _interval(data, where)
    confidence = _number(data, "confidence", where)
    if not 0.0 < confidence < 1.0:
        raise CatalogueError(f"{where}: confidence must be a level in (0, 1)")
    return Quality(
        holdout_auc=_unit(data, "holdout_auc", where),
        interval_low=low,
        interval_high=high,
        interval_method=_text(data, "interval_method", where),
        confidence=confidence,
        n_exposed=_count(data, "n_exposed", where),
        n_fenced=_count(data, "n_fenced", where),
        n_prompts=_count(data, "n_prompts", where, optional=True),
        permutation_p=_unit(data, "permutation_p", where, optional=True),
    )


def _cutoff(data: object, where: str) -> CutoffCheck | None:
    if data is None:
        return None
    data = _exact_keys(data, _CUTOFF_KEYS, where)
    low, high = _interval(data, where)
    inside, after = _number(data, "inside_mean", where), _number(data, "after_mean", where)
    difference = _number(data, "difference", where)
    if abs(difference - (inside - after)) > DIFFERENCE_TOLERANCE:
        raise CatalogueError(
            f"{where}: difference {difference:g} is not inside_mean minus after_mean ({inside:g} - {after:g})"
        )
    return CutoffCheck(
        inside_mean=inside,
        after_mean=after,
        difference=difference,
        interval_low=low,
        interval_high=high,
        p_value=_unit(data, "p_value", where),
        n_inside=_count(data, "n_inside", where),
        n_after=_count(data, "n_after", where),
    )


def _evidence(data: object, where: str) -> Evidence | None:
    if data is None:
        return None
    data = _exact_keys(data, _EVIDENCE_KEYS, where)
    ids = data["register_ids"]
    if not isinstance(ids, list) or not ids:
        raise CatalogueError(f"{where}: register_ids must be a non-empty list")
    for item in ids:
        if not isinstance(item, str) or not _REGISTER_ID.fullmatch(item):
            raise CatalogueError(f"{where}: evidence id {item!r} is not of the form R<digits>")
    return Evidence(register_ids=tuple(ids), method=_text(data, "method", where))


def _is_qualifying(record: ModeRecord) -> bool:
    boundary, evidence = record.boundary, record.evidence
    return (
        boundary is not None
        and boundary.domain == QUALIFYING_DOMAIN
        and evidence is not None
        and evidence.method == QUALIFYING_METHOD
    )


def _availability(stated: object, derived: bool | None, where: str) -> bool:
    if stated is not None and not isinstance(stated, bool):
        raise CatalogueError(f"{where}: available must be true, false or null")
    if derived is None:
        if stated is None:
            raise CatalogueError(f"{where}: available must be stated for an entry without a profile")
        return stated
    if stated is not None and stated != derived:
        raise CatalogueError(f"{where}: available={stated} contradicts the provider profile, which gives {derived}")
    return derived


def _mode_record(mode: Mode, data: object, derived: bool | None, where: str) -> ModeRecord:
    data = _exact_keys(data, _MODE_KEYS, where)
    available = _availability(data["available"], derived, where)
    reason = _text(data, "unavailable_reason", where, optional=True)
    requires = _text(data, "requires", where, optional=True)
    if available and reason is not None:
        raise CatalogueError(f"{where}: unavailable_reason is set but the mode is available")
    if not available and reason is None:
        raise CatalogueError(f"{where}: an unavailable mode must state its unavailable_reason")
    if not available and mode is Mode.LOGITS and requires is None:
        raise CatalogueError(f"{where}: an unavailable logits mode must state what it requires")
    base = ModeRecord(
        mode=mode,
        available=available,
        unavailable_reason=reason,
        requires=requires,
        features=_texts(data, "features", where),
        features_unavailable=_texts(data, "features_unavailable", where),
        boundary=None,
        quality=None,
        cutoff=None,
        calibrator_fingerprint=None,
        evidence=None,
    )
    if not isinstance(data["calibrations"], list):
        raise CatalogueError(f"{where}: calibrations must be a list")
    records = []
    for index, item in enumerate(data["calibrations"]):
        here = f"{where} calibration {index}"
        item = _exact_keys(item, _CALIBRATION_KEYS, here)
        evidence = _evidence(item["evidence"], f"{here} evidence")
        if evidence is None:
            raise CatalogueError(f"{here}: values recorded without evidence")
        records.append(
            replace(
                base,
                boundary=_boundary(item["boundary"], f"{here} boundary"),
                quality=_quality(item["quality"], f"{here} quality"),
                cutoff=_cutoff(item["cutoff"], f"{here} cutoff"),
                calibrator_fingerprint=_text(item, "calibrator_fingerprint", here, optional=True),
                evidence=evidence,
            )
        )
    if not records:
        return base
    qualifying = [record for record in records if _is_qualifying(record)]
    if len(qualifying) > 1:
        raise CatalogueError(
            f"{where}: more than one {QUALIFYING_DOMAIN} {QUALIFYING_METHOD} calibration; "
            "the status of a mode is read from one"
        )
    # The rule reads the time-series reply-features calibration; without one, the first recorded.
    primary = qualifying[0] if qualifying else records[0]
    return replace(primary, other_evidence=tuple(record for record in records if record is not primary))


def _entry(data: object, index: int, profiles: tuple[str, ...]) -> CatalogueEntry:
    where = f"entry {index}"
    if isinstance(data, Mapping) and isinstance(data.get("entry_id"), str) and data["entry_id"].strip():
        where = f"entry {data['entry_id']!r}"
    data = _exact_keys(data, _ENTRY_KEYS, where)
    entry_id = _text(data, "entry_id", where)
    profile_id = _text(data, "profile_id", where, optional=True)
    sources = None
    if profile_id is not None:
        if profile_id not in profiles:
            raise CatalogueError(f"{where}: profile_id {profile_id!r} is not a shipped provider profile")
        try:
            sources = load_profile(profile_id).estimate_sources
        except ProfileError as error:
            raise CatalogueError(f"{where}: cannot load provider profile {profile_id!r}: {error}") from None
    modes = data["modes"]
    if not isinstance(modes, Mapping) or set(modes) != {mode.value for mode in Mode}:
        raise CatalogueError(f"{where}: modes must hold exactly {', '.join(mode.value for mode in Mode)}")
    records = {
        mode: _mode_record(
            mode,
            modes[mode.value],
            None if sources is None else _SOURCE_OF_MODE[mode] in sources,
            f"{where} {mode.value}",
        )
        for mode in Mode
    }
    return CatalogueEntry(
        entry_id=entry_id,
        profile_id=profile_id,
        model=_text(data, "model", where),
        route=_text(data, "route", where),
        weights_identity=_text(data, "weights_identity", where, optional=True),
        weights_group=_text(data, "weights_group", where, optional=True),
        modes=MappingProxyType(records),
    )


def _packaged_catalogue():
    """The catalogue data file shipped inside the package."""
    return resources.files("recall_guard.core").joinpath(CATALOGUE_RESOURCE)


def _coerce(kind: type[StrEnum], value: object) -> StrEnum:
    try:
        return kind(value)
    except ValueError:
        known = ", ".join(member.value for member in kind)
        raise CatalogueError(f"Unknown {kind.__name__.lower()} {value!r}; known: {known}") from None


@dataclass(frozen=True)
class Catalogue:
    """The loaded catalogue: entries in file order, with listing and filters."""

    entries: tuple[CatalogueEntry, ...]

    def entry(self, entry_id: str) -> CatalogueEntry:
        """The entry with this id."""
        for entry in self.entries:
            if entry.entry_id == entry_id:
                return entry
        known = ", ".join(entry.entry_id for entry in self.entries) or "none"
        raise CatalogueError(f"Unknown catalogue entry {entry_id!r}; known: {known}")

    def qualification(self, entry_id: str, mode: Mode | str) -> Qualification:
        """Outcome of the qualification rule for one entry in one mode."""
        return qualify(self.entry(entry_id).modes[_coerce(Mode, mode)])

    def filter(
        self, *, mode: Mode | str | None = None, status: Status | str | None = None
    ) -> tuple[CatalogueEntry, ...]:
        """Entries matching a mode, a status or both.

        A mode alone gives the entries where that mode is available. A status
        alone gives the entries where either mode has it. Both give the entries
        whose status in that mode is the one asked for.
        """
        mode = None if mode is None else _coerce(Mode, mode)
        status = None if status is None else _coerce(Status, status)

        def matches(entry: CatalogueEntry) -> bool:
            if status is None:
                return mode is None or entry.modes[mode].available
            return any(qualify(entry.modes[item]).status is status for item in ((mode,) if mode else Mode))

        return tuple(entry for entry in self.entries if matches(entry))

    def without_logprobs(self) -> tuple[CatalogueEntry, ...]:
        """Entries that can be used on a route without logprobs: the logits mode is available."""
        return tuple(entry for entry in self.entries if entry.modes[Mode.LOGITS].available)

    def related(self, entry_id: str) -> tuple[CatalogueEntry, ...]:
        """The other entries that share this entry's weights."""
        group = self.entry(entry_id).weights_group
        if group is None:
            return ()
        return tuple(entry for entry in self.entries if entry.weights_group == group and entry.entry_id != entry_id)

    def usable(self, entry_id: str) -> bool:
        """False when neither mode is available for the entry."""
        return any(record.available for record in self.entry(entry_id).modes.values())


def load_catalogue(source: Mapping | str | Path | None = None) -> Catalogue:
    """Load and validate a catalogue.

    Without an argument, the data file shipped in the package; otherwise the
    path of a data file, or its parsed content.
    """
    if isinstance(source, Mapping):
        data = source
    else:
        path = _packaged_catalogue() if source is None else Path(source)
        if source is None and not path.is_file():
            raise CatalogueError(f"The packaged catalogue data file {path.name} does not exist: {path}")
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise CatalogueError(f"Cannot read catalogue data file {path.name}: {error.__class__.__name__}") from None
    data = _exact_keys(data, _TOP_KEYS, "catalogue")
    if data["schema"] != CATALOGUE_SCHEMA:
        raise CatalogueError(f"Unsupported catalogue schema {data['schema']!r}; expected {CATALOGUE_SCHEMA!r}")
    rule = _exact_keys(data["rule"], frozenset(_RULE), "rule")
    for key, constant in _RULE.items():
        value = _number(rule, key, "rule")
        if value != constant:
            raise CatalogueError(f"rule: {key} is {value:g} in the data file but {constant:g} in the rule")
    if not isinstance(data["entries"], list):
        raise CatalogueError("catalogue: entries must be a list")
    profiles = available_profiles()
    entries: list[CatalogueEntry] = []
    seen: set[str] = set()
    for index, item in enumerate(data["entries"]):
        entry = _entry(item, index, profiles)
        if entry.entry_id in seen:
            raise CatalogueError(f"Duplicate entry_id {entry.entry_id!r}")
        seen.add(entry.entry_id)
        entries.append(entry)
    return Catalogue(entries=tuple(entries))
