"""Frozen study plan: month split, EDGAR sets, frozen fields and changed-plan refusal (tasks 7.2 and 7.3).

Task 7.2 owns this part. It splits the decision months into calibration, pilot and
evaluation sets, freezes the EDGAR calibration rows and the negative-control items,
freezes the analysis settings and the accuracy definition, and hashes the plan.
Task 7.3 adds the arms, variant sets, limits, pilot settings and the scored
probability as further sections, and ``build_plan.py`` (7.3) wires the item
assembly (7.1) into :func:`freeze_plan` and writes ``runs/finance-sim/plan/plan.json``.

The inputs are plain values, so nothing here fetches data or contacts a host: one
:class:`MonthClass` per decision month with a pre-cutoff flag per model, and one
:class:`EdgarRow` per EDGAR release with its label and split tag.

Month split (Requirement 4.1), seeded by the plan's ``seed``:

- A month can be a calibration or pilot month only when it is on or before every
  model's cutoff. A month after any model's cutoff stays in evaluation.
- Calibration: 3 months per calendar year 2016 to 2023, drawn first.
- Pilot: from the months left, 1 month per calendar year 2016 to 2024, plus 1 more
  in each of 3 distinct years drawn from those 9.
- Evaluation: every other month. Pilot months are excluded from the confirmatory
  analysis (7.8).

One ``numpy.random.default_rng(seed)`` generator is consumed in this order: the
calibration months year by year, then the 3 extra pilot years, then the pilot
months year by year. Within a year, months are drawn uniformly without replacement
from that year's eligible months in calendar order.

Hashing and refusal (Requirement 8.2): every section has a content hash, and
``plan_sha256`` is the SHA-256 of the canonical JSON (sorted keys, compact
separators, UTF-8, no NaN) of every frozen field and the section hashes, excluding
``plan_sha256`` itself. :func:`load_plan` recomputes it from the file's content
before parsing anything and raises :class:`PlanChangedError` on any difference;
:func:`ensure_unchanged` refuses a plan whose hash differs from the one a recording
carries; :func:`write_plan` never replaces a different frozen plan.
"""

from __future__ import annotations

import hashlib
import json
import math
import numbers
import os
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import MISSING, dataclass, field, fields
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

from recall_guard.dataset.macro_regime import EXPOSURE

PLAN_SCHEMA = "finance-sim-plan/1"

#: Calendar years that give calibration months, and how many each gives.
CALIBRATION_YEARS: tuple[int, ...] = tuple(range(2016, 2024))
CALIBRATION_PER_YEAR = 3
#: Calendar years that give one pilot month each; PILOT_EXTRA_YEARS of them, drawn by the seed, give one more.
PILOT_YEARS: tuple[int, ...] = tuple(range(2016, 2025))
PILOT_EXTRA_YEARS = 3
SPLIT_RULE = (
    "calibration: 3 months per calendar year 2016 to 2023, drawn first; pilot: from the months left, "
    "1 month per calendar year 2016 to 2024 plus 1 more in each of 3 distinct years drawn from those 9; "
    "only months on or before every model's cutoff are drawn; evaluation: every other month; "
    "numpy.random.default_rng(seed) consumed as calibration months, extra pilot years, pilot months, "
    "months drawn uniformly without replacement within a year in calendar order"
)

#: EDGAR split tags (the directories under ``data/edgar/``) and the manifest's file keys.
EDGAR_SPLITS: tuple[str, ...] = ("calibration", "held_out")
EDGAR_FILES: tuple[str, ...] = (
    "calibration/is_memorized.jsonl", "calibration/oos_control.jsonl",
    "held_out/is_memorized.jsonl", "held_out/oos_control.jsonl",
)
EDGAR_CALIBRATION_MEMORIZED = 40
EDGAR_CALIBRATION_CONTROL = 100
EDGAR_NEGATIVE_CONTROL = 40

#: Names of the 7.10 and 7.11 statistics, in this order; those tasks use exactly these names.
METRICS: tuple[str, ...] = (
    "estimate_difference_pre_cutoff", "estimate_difference_post_cutoff",
    "accuracy_difference_pre_cutoff", "accuracy_difference_post_cutoff",
    "did_accuracy_gap", "brier_difference", "log_loss_difference", "achieved_power",
    "half_ensemble_variance_summary", "half_ensemble_variance_p_memorized",
    "first_draw_rate", "random_answerer_null",
)
#: Pre-registered DiD accuracy gap, the smallest effect of practical interest.
EFFECT_SIZE = 0.10
#: Negative control: allowed mean absolute change of the outlook against no adjustment, on the [-1, 1] scale.
TOLERANCE = 0.10
N_RESAMPLES = 2000
#: Moving-block length in months for the macro analyses.
BLOCK_LENGTH = 6
#: Moving-block length in filing quarters for the negative control (7.9).
NEGATIVE_CONTROL_BLOCK_LENGTH = 2
ALPHA = 0.05
POWER_THRESHOLD = 0.8
#: First-draw rank band of the ensemble (7.11): a first draw outside it counts as off-centre.
FIRST_DRAW_BAND: tuple[float, float] = (0.25, 0.75)

#: Accuracy (Requirement 6.6): one rule for every variant, with its documented price source.
ACCURACY_RULE = "macro_regime.is_correct"
PRICE_SOURCES: Mapping[str, str] = MappingProxyType({
    "world_equity": "FMP historical-price-eod/dividend-adjusted adjClose, SWDA.L",
    "tech_sector": "FMP historical-price-eod/dividend-adjusted adjClose, XLK",
    "gold_commodity": "FMP historical-price-eod/dividend-adjusted adjClose, IAU",
    "short_treasury_cash": "FRED DTB3, current vintage",
})
RETURN_RULE = (
    "for decision month M: asset return = adjClose(last trading day of M+1) / adjClose(last trading day of M) - 1; "
    "cash return = DTB3(last observation on or before the end of M) / 100 / 12"
)
OUTCOME_RULE = (
    "a spread is correct when (round(spread, 9) > 0) equals "
    "(round((gold_commodity + short_treasury_cash) - (world_equity + tech_sector), 9) > 0)"
)
MISSING_RULE = "a month without a complete M+1 keeps realized = None and is reported as missing, never substituted"

class PlanChangedError(ValueError):
    """The plan differs from its frozen content or from the hash a recording carries."""


_MONTH = re.compile(r"\d{4}-(0[1-9]|1[0-2])")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_METRIC = re.compile(r"[a-z][a-z0-9_]*")


# ---------- canonical JSON and hashes ----------

def canonical_json(value: Any) -> str:
    """Sorted keys, compact separators, no NaN or infinity: the form every plan hash is taken over."""
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def sha256_of(value: Any) -> str:
    """SHA-256 (lowercase hex) of the UTF-8 canonical JSON of ``value``."""
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def payload_sha256(payload: Mapping[str, Any]) -> str:
    """The plan hash of a serialised plan: every key except ``plan_sha256``."""
    return sha256_of({key: value for key, value in payload.items() if key != "plan_sha256"})


# ---------- validation helpers ----------

def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string; got {value!r}")
    return value


def _sequence(value: object, name: str) -> tuple:
    if isinstance(value, str | bytes) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be a sequence; got {type(value).__name__}")
    return tuple(value)


def _month(value: object, name: str) -> str:
    if not isinstance(value, str) or not _MONTH.fullmatch(value):
        raise ValueError(f"{name} must be a month as YYYY-MM; got {value!r}")
    return value


def _months(value: object, name: str) -> tuple[str, ...]:
    """Months in strictly increasing calendar order (sorted, no duplicates)."""
    months = tuple(_month(month, f"{name}[{index}]") for index, month in enumerate(_sequence(value, name)))
    if any(earlier >= later for earlier, later in zip(months, months[1:], strict=False)):
        raise ValueError(f"{name} must be in strictly increasing calendar order")
    return months


def _ids(value: object, name: str) -> tuple[str, ...]:
    ids = tuple(_text(item, f"{name}[{index}]") for index, item in enumerate(_sequence(value, name)))
    if len(set(ids)) != len(ids):
        raise ValueError(f"{name} must not repeat an id")
    return ids


def _mapping(value: object, name: str) -> Mapping:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping; got {type(value).__name__}")
    return value


def _sha256(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a SHA-256 hex string; got {type(value).__name__}")
    if not _SHA256.fullmatch(value):
        raise ValueError(f"{name} must be 64 lowercase hex characters; got {value!r}")
    return value


def _sha256_map(value: object, name: str, *, keys: tuple[str, ...] | None = None) -> Mapping[str, str]:
    mapping = _mapping(value, name)
    if not mapping:
        raise ValueError(f"{name} must not be empty")
    if keys is not None and set(mapping) != set(keys):
        raise ValueError(f"{name} must have exactly the keys {list(keys)}; got {sorted(map(str, mapping))}")
    return MappingProxyType({_text(key, f"{name} key"): _sha256(item, f"{name}[{key!r}]")
                             for key, item in sorted(mapping.items())})


def _text_map(value: object, name: str) -> Mapping[str, str]:
    mapping = _mapping(value, name)
    return MappingProxyType({_text(key, f"{name} key"): _text(item, f"{name}[{key!r}]")
                             for key, item in sorted(mapping.items())})


def _integer(value: object, name: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer; got {value!r}")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}; got {value}")
    return value


def _seed(value: object) -> int:
    seed = _integer(value, "seed", minimum=0)
    if seed >= 2**63:
        raise ValueError(f"seed must be below 2**63; got {seed}")
    return seed


def _real(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(f"{name} must be a number; got {value!r}")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite; got {value!r}")
    return number


def _timestamp(value: object) -> str:
    """An ISO 8601 UTC timestamp to the second, in ``datetime.isoformat`` form."""
    text = _text(value, "created_at")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        raise ValueError(f"created_at must be an ISO 8601 timestamp; got {text!r}") from None
    if parsed.utcoffset() is None or parsed.utcoffset().total_seconds() != 0:
        raise ValueError(f"created_at must be in UTC; got {text!r}")
    if parsed.isoformat(timespec="seconds") != text:
        raise ValueError(f"created_at must be written as {parsed.isoformat(timespec='seconds')!r}; got {text!r}")
    return text


def _plain(value: Any) -> Any:
    """JSON-ready form: sections by their ``to_dict``, mappings as dicts, tuples as lists."""
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_plain(item) for item in value]
    return value


def _section_dict(section: Any) -> dict[str, Any]:
    return {f.name: _plain(getattr(section, f.name)) for f in fields(section)}


def _section_from_dict(cls: type, data: object, name: str) -> Any:
    mapping = _mapping(data, name)
    names = {f.name for f in fields(cls)}
    if set(mapping) != names:
        raise ValueError(f"{name} must have exactly the fields {sorted(names)}; "
                         f"missing {sorted(names - set(mapping))}, undeclared {sorted(map(str, set(mapping) - names))}")
    return cls(**mapping)


def library_exposure() -> dict[str, dict[str, float]]:
    """``macro_regime.EXPOSURE`` as plain dicts (it is a read-only proxy of proxies, which JSON cannot take)."""
    return {category: {axis: float(value) for axis, value in row.items()} for category, row in EXPOSURE.items()}


# ---------- inputs ----------

@dataclass(frozen=True)
class MonthClass:
    """One decision month (``YYYY-MM``) and, per model id, whether it is on or before that model's cutoff."""

    month: str
    pre_cutoff: Mapping[str, bool]

    def __post_init__(self) -> None:
        _month(self.month, "month")
        flags = _mapping(self.pre_cutoff, "pre_cutoff")
        if not flags:
            raise ValueError("pre_cutoff must classify at least one model")
        for model, flag in flags.items():
            _text(model, "pre_cutoff model id")
            if not isinstance(flag, bool):
                raise TypeError(f"pre_cutoff[{model!r}] must be a bool; got {flag!r}")
        object.__setattr__(self, "pre_cutoff", MappingProxyType(dict(sorted(flags.items()))))


@dataclass(frozen=True)
class EdgarRow:
    """One EDGAR release: an id, its label (1 memorized, 0 control) and its split tag (:data:`EDGAR_SPLITS`)."""

    row_id: str
    label: int
    split: str

    def __post_init__(self) -> None:
        _text(self.row_id, "row_id")
        if isinstance(self.label, bool) or not isinstance(self.label, int):
            raise TypeError(f"label must be the integer 1 or 0; got {self.label!r}")
        if self.label not in (0, 1):
            raise ValueError(f"label must be 1 (memorized) or 0 (control); got {self.label}")
        if self.split not in EDGAR_SPLITS:
            raise ValueError(f"split must be one of {list(EDGAR_SPLITS)}; got {self.split!r}")


# ---------- sections ----------

@dataclass(frozen=True)
class MonthSplit:
    """The frozen month split: calibration, pilot and evaluation months, disjoint and covering every month.

    ``pre_cutoff`` lists, per model, the months on or before its cutoff; they are
    the earliest months, so every later month is post-cutoff for that model.
    """

    models: tuple[str, ...]
    months: tuple[str, ...]
    pre_cutoff: Mapping[str, tuple[str, ...]]
    calibration: tuple[str, ...]
    pilot: tuple[str, ...]
    evaluation: tuple[str, ...]
    rule: str = SPLIT_RULE

    def __post_init__(self) -> None:
        models = tuple(sorted(_ids(self.models, "models")))
        if not models:
            raise ValueError("models must name at least one model")
        months = _months(self.months, "months")
        if not months:
            raise ValueError("months must not be empty")
        pre = _mapping(self.pre_cutoff, "pre_cutoff")
        if set(pre) != set(models):
            raise ValueError(f"pre_cutoff must classify exactly the models {list(models)}")
        pre_cutoff = {}
        for model in models:
            listed = _months(pre[model], f"pre_cutoff[{model!r}]")
            if listed != months[:len(listed)]:
                raise ValueError(f"pre_cutoff[{model!r}] must be the months up to that model's cutoff")
            pre_cutoff[model] = listed
        calibration = _months(self.calibration, "calibration")
        pilot = _months(self.pilot, "pilot")
        evaluation = _months(self.evaluation, "evaluation")
        sets = {"calibration": set(calibration), "pilot": set(pilot), "evaluation": set(evaluation)}
        for first, second in (("calibration", "pilot"), ("calibration", "evaluation"), ("pilot", "evaluation")):
            shared = sets[first] & sets[second]
            if shared:
                raise ValueError(f"{first} and {second} months must be disjoint; both hold {sorted(shared)}")
        if set().union(*sets.values()) != set(months):
            raise ValueError("calibration, pilot and evaluation months must together hold every month exactly once")
        eligible = set(months[:min(len(listed) for listed in pre_cutoff.values())])
        late = sorted((sets["calibration"] | sets["pilot"]) - eligible)
        if late:
            raise ValueError(f"calibration and pilot months must be on or before every model's cutoff; got {late}")
        object.__setattr__(self, "models", models)
        object.__setattr__(self, "months", months)
        object.__setattr__(self, "pre_cutoff", MappingProxyType(pre_cutoff))
        object.__setattr__(self, "calibration", calibration)
        object.__setattr__(self, "pilot", pilot)
        object.__setattr__(self, "evaluation", evaluation)
        _text(self.rule, "rule")

    def to_dict(self) -> dict[str, Any]:
        return _section_dict(self)

    @classmethod
    def from_dict(cls, data: object) -> MonthSplit:
        return _section_from_dict(cls, data, "months")


@dataclass(frozen=True)
class EdgarSplit:
    """Frozen EDGAR row ids in input order: the calibration rows, the negative-control items and the unused rows.

    Calibration rows are the ``calibration`` split (label 1 memorized, label 0
    control); negative-control items are the ``held_out`` control rows; the
    ``held_out`` memorized rows are recorded as unused. No id appears twice.
    """

    calibration_memorized: tuple[str, ...]
    calibration_control: tuple[str, ...]
    negative_control: tuple[str, ...]
    unused: tuple[str, ...]

    def __post_init__(self) -> None:
        groups = {f.name: _ids(getattr(self, f.name), f.name) for f in fields(self)}
        for name, expected in (("calibration_memorized", EDGAR_CALIBRATION_MEMORIZED),
                               ("calibration_control", EDGAR_CALIBRATION_CONTROL),
                               ("negative_control", EDGAR_NEGATIVE_CONTROL)):
            if len(groups[name]) != expected:
                raise ValueError(f"{name} must hold {expected} EDGAR rows; got {len(groups[name])}")
        seen: dict[str, str] = {}
        for name, ids in groups.items():
            for row_id in ids:
                if row_id in seen:
                    raise ValueError(f"EDGAR sets must be disjoint; {row_id!r} is in {seen[row_id]} and {name}")
                seen[row_id] = name
        for name, ids in groups.items():
            object.__setattr__(self, name, ids)

    def to_dict(self) -> dict[str, Any]:
        return _section_dict(self)

    @classmethod
    def from_dict(cls, data: object) -> EdgarSplit:
        return _section_from_dict(cls, data, "edgar")


@dataclass(frozen=True)
class AccuracyDefinition:
    """Decision accuracy against realized next-month returns (Requirement 6.6), the same for every variant."""

    rule: str = ACCURACY_RULE
    exposure: Mapping[str, Mapping[str, float]] = field(default_factory=library_exposure)
    sources: Mapping[str, str] = field(default_factory=lambda: dict(PRICE_SOURCES))
    returns: str = RETURN_RULE
    outcome: str = OUTCOME_RULE
    missing: str = MISSING_RULE

    def __post_init__(self) -> None:
        for name in ("rule", "returns", "outcome", "missing"):
            _text(getattr(self, name), name)
        exposure = _mapping(self.exposure, "exposure")
        if not exposure:
            raise ValueError("exposure must not be empty")
        rows = {}
        for category, row in sorted(exposure.items()):
            _text(category, "exposure category")
            row = _mapping(row, f"exposure[{category!r}]")
            if not row:
                raise ValueError(f"exposure[{category!r}] must not be empty")
            rows[category] = MappingProxyType({
                _text(axis, "exposure axis"): _real(value, f"exposure[{category!r}][{axis!r}]")
                for axis, value in sorted(row.items())
            })
        if len({tuple(row) for row in rows.values()}) != 1:
            raise ValueError("every exposure row must name the same axes")
        sources = _text_map(self.sources, "sources")
        if set(sources) != set(rows):
            raise ValueError(f"sources must name exactly the exposure categories {sorted(rows)}")
        object.__setattr__(self, "exposure", MappingProxyType(rows))
        object.__setattr__(self, "sources", sources)

    def to_dict(self) -> dict[str, Any]:
        return _section_dict(self)

    @classmethod
    def from_dict(cls, data: object) -> AccuracyDefinition:
        return _section_from_dict(cls, data, "accuracy")


@dataclass(frozen=True)
class AnalysisSettings:
    """Pre-registered metrics and statistical settings, shared by every analysis (7.8 to 7.11)."""

    metrics: tuple[str, ...] = METRICS
    effect_size: float = EFFECT_SIZE
    tolerance: float = TOLERANCE
    n_resamples: int = N_RESAMPLES
    block_length: int = BLOCK_LENGTH
    negative_control_block_length: int = NEGATIVE_CONTROL_BLOCK_LENGTH
    alpha: float = ALPHA
    power_threshold: float = POWER_THRESHOLD
    first_draw_band: tuple[float, float] = FIRST_DRAW_BAND

    def __post_init__(self) -> None:
        metrics = _ids(self.metrics, "metrics")
        if not metrics:
            raise ValueError("metrics must name at least one statistic")
        for metric in metrics:
            if not _METRIC.fullmatch(metric):
                raise ValueError(f"metric names must be lowercase identifiers; got {metric!r}")
        effect_size = _real(self.effect_size, "effect_size")
        if not 0.0 < effect_size <= 2.0:
            raise ValueError(f"effect_size must be in (0, 2] (an accuracy gap); got {effect_size}")
        tolerance = _real(self.tolerance, "tolerance")
        if not 0.0 < tolerance <= 2.0:
            raise ValueError(f"tolerance must be in (0, 2] (a change on the [-1, 1] scale); got {tolerance}")
        _integer(self.n_resamples, "n_resamples", minimum=1)
        _integer(self.block_length, "block_length", minimum=1)
        _integer(self.negative_control_block_length, "negative_control_block_length", minimum=1)
        values = {}
        for name in ("alpha", "power_threshold"):
            values[name] = _real(getattr(self, name), name)
            if not 0.0 < values[name] < 1.0:
                raise ValueError(f"{name} must be in (0, 1); got {values[name]}")
        band = _sequence(self.first_draw_band, "first_draw_band")
        if len(band) != 2:
            raise ValueError(f"first_draw_band must be (lower, upper); got {band!r}")
        lower, upper = (_real(bound, "first_draw_band bound") for bound in band)
        if not 0.0 <= lower < upper <= 1.0:
            raise ValueError(f"first_draw_band must satisfy 0 <= lower < upper <= 1; got {band!r}")
        object.__setattr__(self, "metrics", metrics)
        object.__setattr__(self, "effect_size", effect_size)
        object.__setattr__(self, "tolerance", tolerance)
        object.__setattr__(self, "alpha", values["alpha"])
        object.__setattr__(self, "power_threshold", values["power_threshold"])
        object.__setattr__(self, "first_draw_band", (lower, upper))

    def to_dict(self) -> dict[str, Any]:
        return _section_dict(self)

    @classmethod
    def from_dict(cls, data: object) -> AnalysisSettings:
        return _section_from_dict(cls, data, "analysis")


# ---------- the plan ----------

@dataclass(frozen=True)
class StudyPlan:
    """The frozen study plan; immutable, and identified by :attr:`plan_sha256`.

    ``items`` maps each study item id (7.1) to the SHA-256 of its row and prompts;
    ``excluded_models`` maps a model without a sourced cutoff to the reason;
    ``edgar_files`` maps each :data:`EDGAR_FILES` key to its manifest sha256.

    Task 7.3 appends its sections (arms, variant sets, limits, pilot settings,
    scored probability) as keyword fields defaulting to None, each with a
    ``to_dict`` and a parser in ``_SECTION_PARSERS``. A section that is None is
    left out of the payload and the hash; a set one is serialised, gets a content
    hash and enters ``plan_sha256`` like the others.
    """

    created_at: str
    seed: int
    items: Mapping[str, str]
    excluded_models: Mapping[str, str]
    months: MonthSplit
    edgar: EdgarSplit
    edgar_files: Mapping[str, str]
    accuracy: AccuracyDefinition
    analysis: AnalysisSettings
    schema_version: str = PLAN_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != PLAN_SCHEMA:
            raise ValueError(f"unsupported plan schema {self.schema_version!r}; expected {PLAN_SCHEMA!r}")
        _timestamp(self.created_at)
        _seed(self.seed)
        object.__setattr__(self, "items", _sha256_map(self.items, "items"))
        object.__setattr__(self, "excluded_models", _text_map(self.excluded_models, "excluded_models"))
        object.__setattr__(self, "edgar_files", _sha256_map(self.edgar_files, "edgar_files", keys=EDGAR_FILES))
        for name, cls in (("months", MonthSplit), ("edgar", EdgarSplit), ("accuracy", AccuracyDefinition),
                          ("analysis", AnalysisSettings)):
            if not isinstance(getattr(self, name), cls):
                raise TypeError(f"{name} must be a {cls.__name__}; got {type(getattr(self, name)).__name__}")
        both = sorted(set(self.excluded_models) & set(self.months.models))
        if both:
            raise ValueError(f"excluded models must not be in the month split; got {both}")

    def frozen_fields(self) -> dict[str, Any]:
        """Every frozen field in JSON form; a None section (7.3, not set) is left out."""
        return {f.name: _plain(getattr(self, f.name)) for f in fields(self) if getattr(self, f.name) is not None}

    @property
    def content_sha256(self) -> dict[str, str]:
        """SHA-256 of each section (every field whose JSON form is an object)."""
        return {name: sha256_of(value) for name, value in self.frozen_fields().items() if isinstance(value, dict)}

    @property
    def plan_sha256(self) -> str:
        return payload_sha256({**self.frozen_fields(), "content_sha256": self.content_sha256})

    def to_dict(self) -> dict[str, Any]:
        body = {**self.frozen_fields(), "content_sha256": self.content_sha256}
        return {**body, "plan_sha256": payload_sha256(body)}

    @classmethod
    def from_dict(cls, data: object) -> StudyPlan:
        """Parse and validate a serialised plan; its section and plan hashes must match the content."""
        mapping = _mapping(data, "plan")
        declared = {f.name: f for f in fields(cls)}
        required = {name for name, f in declared.items()
                    if f.default is MISSING and f.default_factory is MISSING} | {"schema_version", "content_sha256"}
        extra = sorted(map(str, set(mapping) - set(declared) - {"content_sha256", "plan_sha256"}))
        missing = sorted(required - set(mapping))
        if extra or missing:
            raise ValueError(f"plan fields differ from the schema; missing {missing}, undeclared {extra}")
        values = {name: _SECTION_PARSERS.get(name, _unchanged)(mapping[name]) for name in declared if name in mapping}
        plan = cls(**values)
        recorded = _mapping(mapping["content_sha256"], "content_sha256")
        actual = plan.content_sha256
        if dict(recorded) != actual:
            changed = {name for name in actual if recorded.get(name) != actual[name]}
            differing = sorted(map(str, (set(recorded) ^ set(actual)) | changed))
            raise PlanChangedError(f"section content differs from its recorded hash: {differing}")
        if "plan_sha256" in mapping and mapping["plan_sha256"] != plan.plan_sha256:
            raise PlanChangedError("plan content differs from its recorded plan_sha256")
        return plan


def _unchanged(value: Any) -> Any:
    return value


#: Parsers for the fields whose JSON form is a section; other fields are validated by StudyPlan itself.
_SECTION_PARSERS: dict[str, Callable[[Any], Any]] = {
    "months": MonthSplit.from_dict,
    "edgar": EdgarSplit.from_dict,
    "accuracy": AccuracyDefinition.from_dict,
    "analysis": AnalysisSettings.from_dict,
}


# ---------- splits ----------

def _draw(rng: np.random.Generator, pool: Sequence[str], count: int, label: str) -> list[str]:
    """``count`` months of ``pool`` (calendar order) drawn uniformly without replacement, in calendar order."""
    if len(pool) < count:
        raise ValueError(f"{label} has {len(pool)} eligible months; the split needs {count}")
    return sorted(pool[int(index)] for index in rng.choice(len(pool), size=count, replace=False))


def split_months(months: Sequence[MonthClass], *, seed: int) -> MonthSplit:
    """Split decision months into calibration, pilot and evaluation sets by the frozen rule (module docstring)."""
    seed = _seed(seed)
    classes = _sequence(months, "months")
    if not classes:
        raise ValueError("months must not be empty")
    for item in classes:
        if not isinstance(item, MonthClass):
            raise TypeError(f"months must hold MonthClass values; got {type(item).__name__}")
    ordered = sorted(classes, key=lambda item: item.month)
    for earlier, later in zip(ordered, ordered[1:], strict=False):
        if earlier.month == later.month:
            raise ValueError(f"month {later.month} appears more than once")
    models = tuple(ordered[0].pre_cutoff)
    for item in ordered:
        if tuple(item.pre_cutoff) != models:
            raise ValueError(f"month {item.month} classifies {list(item.pre_cutoff)}; "
                             f"every month must classify {list(models)}")
    for model in models:
        for earlier, later in zip(ordered, ordered[1:], strict=False):
            if later.pre_cutoff[model] and not earlier.pre_cutoff[model]:
                raise ValueError(f"model {model!r}: month {later.month} is marked pre-cutoff after the "
                                 f"post-cutoff month {earlier.month}")
    eligible: dict[int, list[str]] = {}
    for item in ordered:
        if all(item.pre_cutoff.values()):
            eligible.setdefault(int(item.month[:4]), []).append(item.month)
    rng = np.random.default_rng(seed)
    calibration: list[str] = []
    for year in CALIBRATION_YEARS:
        calibration += _draw(rng, eligible.get(year, []), CALIBRATION_PER_YEAR, f"calibration year {year}")
    taken = set(calibration)
    extra_years = {PILOT_YEARS[int(index)]
                   for index in rng.choice(len(PILOT_YEARS), size=PILOT_EXTRA_YEARS, replace=False)}
    pilot: list[str] = []
    for year in PILOT_YEARS:
        pool = [month for month in eligible.get(year, []) if month not in taken]
        pilot += _draw(rng, pool, 1 + (year in extra_years), f"pilot year {year}")
    drawn = taken | set(pilot)
    all_months = tuple(item.month for item in ordered)
    return MonthSplit(
        models=models,
        months=all_months,
        pre_cutoff={model: tuple(item.month for item in ordered if item.pre_cutoff[model]) for model in models},
        calibration=tuple(sorted(calibration)),
        pilot=tuple(sorted(pilot)),
        evaluation=tuple(month for month in all_months if month not in drawn),
    )


def split_edgar(rows: Sequence[EdgarRow]) -> EdgarSplit:
    """Freeze the EDGAR calibration rows and negative-control items; refuse a repeated row id or other counts."""
    groups: dict[tuple[str, int], list[str]] = {
        ("calibration", 1): [], ("calibration", 0): [], ("held_out", 0): [], ("held_out", 1): [],
    }
    seen: set[str] = set()
    for row in _sequence(rows, "rows"):
        if not isinstance(row, EdgarRow):
            raise TypeError(f"rows must hold EdgarRow values; got {type(row).__name__}")
        if row.row_id in seen:
            raise ValueError(f"EDGAR row {row.row_id!r} appears more than once")
        seen.add(row.row_id)
        groups[(row.split, row.label)].append(row.row_id)
    return EdgarSplit(
        calibration_memorized=tuple(groups[("calibration", 1)]),
        calibration_control=tuple(groups[("calibration", 0)]),
        negative_control=tuple(groups[("held_out", 0)]),
        unused=tuple(groups[("held_out", 1)]),
    )


# ---------- freeze, write, load, refuse ----------

def freeze_plan(
    months: Sequence[MonthClass],
    edgar_rows: Sequence[EdgarRow],
    *,
    seed: int,
    created_at: datetime,
    items: Mapping[str, str],
    edgar_files: Mapping[str, str],
    excluded_models: Mapping[str, str] | None = None,
    analysis: AnalysisSettings | None = None,
) -> StudyPlan:
    """Freeze the 7.2 part of the plan before any study request; ``created_at`` must be timezone-aware."""
    if not isinstance(created_at, datetime) or created_at.utcoffset() is None:
        raise ValueError("created_at must be a timezone-aware datetime")
    return StudyPlan(
        created_at=created_at.astimezone(UTC).isoformat(timespec="seconds"),
        seed=seed,
        items=items,
        excluded_models={} if excluded_models is None else excluded_models,
        months=split_months(months, seed=seed),
        edgar=split_edgar(edgar_rows),
        edgar_files=edgar_files,
        accuracy=AccuracyDefinition(),
        analysis=AnalysisSettings() if analysis is None else analysis,
    )


def _no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    keys = [key for key, _ in pairs]
    if len(set(keys)) != len(keys):
        raise ValueError(f"duplicate keys in plan file: {sorted({key for key in keys if keys.count(key) > 1})}")
    return dict(pairs)


def _no_constant(name: str) -> Any:
    raise ValueError(f"non-finite number {name} in plan file")


def _read_payload(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_no_duplicate_keys,
                         parse_constant=_no_constant)
    if not isinstance(payload, dict):
        raise ValueError(f"plan file {path} does not hold a JSON object")
    return payload


def _check_accuracy(plan: StudyPlan) -> None:
    """The frozen accuracy rule must still be the library's (6.6): same function name, same EXPOSURE."""
    if plan.accuracy.rule != ACCURACY_RULE:
        raise PlanChangedError(f"frozen accuracy rule {plan.accuracy.rule!r} is not {ACCURACY_RULE!r}")
    if _plain(plan.accuracy.exposure) != library_exposure():
        raise PlanChangedError("frozen accuracy exposure differs from recall_guard.dataset.macro_regime.EXPOSURE")


def ensure_unchanged(plan: StudyPlan, recorded_sha256: str) -> None:
    """Refuse a plan whose hash differs from the one a recording or limit file carries."""
    if not isinstance(plan, StudyPlan):
        raise TypeError(f"plan must be a StudyPlan; got {type(plan).__name__}")
    recorded = _sha256(recorded_sha256, "recorded plan_sha256")
    actual = plan.plan_sha256
    if actual != recorded:
        raise PlanChangedError(f"plan changed: the recording carries plan_sha256 {recorded}, this plan is {actual}")


def write_plan(plan: StudyPlan, path: str | Path) -> Path:
    """Write the frozen plan as JSON; rewriting the same plan is a no-op, replacing a different one is refused."""
    if not isinstance(plan, StudyPlan):
        raise TypeError(f"plan must be a StudyPlan; got {type(plan).__name__}")
    target = Path(path)
    payload = plan.to_dict()
    if target.exists():
        try:
            existing = _read_payload(target).get("plan_sha256")
        except (OSError, ValueError) as exc:
            raise PlanChangedError(f"refusing to replace the unreadable plan file {target}: {exc}") from None
        if existing == payload["plan_sha256"]:
            return target
        raise PlanChangedError(f"refusing to replace the frozen plan {existing} at {target} "
                               f"with {payload['plan_sha256']}")
    target.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
    temporary = target.with_name(target.name + ".tmp")
    temporary.write_bytes(text.encode("utf-8"))
    os.replace(temporary, target)
    return target


def load_plan(path: str | Path, *, expected_sha256: str | None = None) -> StudyPlan:
    """Load a frozen plan, refusing any change.

    The hash is recomputed from the file's content before anything is parsed, so
    any edit raises :class:`PlanChangedError`, even one that also makes the file
    invalid. With ``expected_sha256`` (the hash a recording carries) a different
    plan is refused too. A consistently rehashed but invalid plan raises
    ``ValueError`` or ``TypeError`` from validation.
    """
    expected = None if expected_sha256 is None else _sha256(expected_sha256, "expected_sha256")
    payload = _read_payload(Path(path))
    recorded = payload.get("plan_sha256")
    if not isinstance(recorded, str) or not _SHA256.fullmatch(recorded):
        raise PlanChangedError("plan file carries no valid plan_sha256")
    actual = payload_sha256(payload)
    if actual != recorded:
        raise PlanChangedError(f"plan content changed: recorded plan_sha256 {recorded}, content hashes to {actual}")
    if expected is not None and recorded != expected:
        raise PlanChangedError(f"plan changed: the recording carries plan_sha256 {expected}, the file is {recorded}")
    plan = StudyPlan.from_dict(payload)
    _check_accuracy(plan)
    return plan
