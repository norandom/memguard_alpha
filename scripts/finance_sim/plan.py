"""Frozen study plan: month split, EDGAR sets, frozen fields and changed-plan refusal (tasks 7.2 and 7.3).

Task 7.2 splits the decision months into calibration, pilot and evaluation sets,
freezes the EDGAR calibration rows and the negative-control items, freezes the
analysis settings and the accuracy definition, and hashes the plan. Task 7.3 adds
four sections and ``build_plan.py`` wires the item assembly (7.1) into
:func:`freeze_plan` and writes ``runs/finance-sim/plan/plan.json``:

- ``arms`` (8.1, 8.10): the explicit arm-to-registry-id map and per arm its
  role, profile, GGUF artifact digest, temperature, seed base, draws, calibration
  draws, fit settings, evidence settings and variant set. The sampling arm
  computes the six variants; the serving arm no adjustment and top-k only, with
  its full-distribution variants listed as missing with the reason.
- ``pilot`` (8.9): 12 pilot months times 8 draws per rendering on the gated arm,
  with its own minimum parsed draws and the frozen evaluation month counts.
- ``limits`` (11.3): per arm and stage, in execution order, requests and output
  tokens sized to one recorded ensemble per item and rendering (never one per
  variant), time, and for the serving arm cost from the serving manifest's rates,
  with the manifest's SHA-256 (``LIMITS_RULE``). The StageLimits files rendered
  from it (:func:`stage_limits_files`) carry ``plan_sha256``.
- ``scored_probability`` (8.4): outcome, mapping per draw mode, clip, and the
  shrink toward 0.5, for Brier and log loss.

Arm roles (user decision 2026-11-10, :data:`ROLE_REASON`): the serving arm is
the primary arm (none and top-k on the full evaluation set, its stages as frozen
by 7.3); the sampling arm is the probe arm, sized to about one to two hours of
iGPU generation per stage at the planning rate of 6 s per generation: 8 draws per
rendering (min parsed 6), calibration on the 24 calibration months with 8 draws
and on the 140 EDGAR calibration rows with 4 draws, its full run on the frozen
probe evaluation months (``MonthSplit.probe_evaluation``, about 20 evaluation
months stratified pre and post cutoff), the negative control with 8 draws. Each
probe stage's time limit adds a headroom of 1.25 on that rate and one client
timeout (300 s), and is at most 9,000 s.

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
- Probe evaluation: a subset of :data:`PROBE_EVALUATION_MONTHS` evaluation months
  for the probe arm, stratified like the evaluation set: the months on or before
  every model's cutoff get ``round(PROBE_EVALUATION_MONTHS x pre / evaluation)``
  of them (half rounded up), the months after some model's cutoff the rest. It is
  a subset of the evaluation months, so disjoint from calibration and pilot.

One ``numpy.random.default_rng(seed)`` generator is consumed in this order: the
calibration months year by year, then the 3 extra pilot years, then the pilot
months year by year, then the pre-cutoff and the post-cutoff probe evaluation
months. Within a year (or a probe stratum), months are drawn uniformly without
replacement from the eligible months in calendar order.

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
from decimal import Decimal
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

from recall_guard.dataset.macro_regime import EXPOSURE, RealizedReturns

PLAN_SCHEMA = "finance-sim-plan/1"

#: Calendar years that give calibration months, and how many each gives.
CALIBRATION_YEARS: tuple[int, ...] = tuple(range(2016, 2024))
CALIBRATION_PER_YEAR = 3
#: Calendar years that give one pilot month each; PILOT_EXTRA_YEARS of them, drawn by the seed, give one more.
PILOT_YEARS: tuple[int, ...] = tuple(range(2016, 2025))
PILOT_EXTRA_YEARS = 3
#: Evaluation months of the probe arm's full run (user decision 2026-11-10: about 20).
PROBE_EVALUATION_MONTHS = 20
SPLIT_RULE = (
    "calibration: 3 months per calendar year 2016 to 2023, drawn first; pilot: from the months left, "
    "1 month per calendar year 2016 to 2024 plus 1 more in each of 3 distinct years drawn from those 9; "
    "only months on or before every model's cutoff are drawn; evaluation: every other month; "
    "probe evaluation: 20 evaluation months, round(20 x pre / evaluation) (half up) of the evaluation months on "
    "or before every model's cutoff and the rest of those after some model's cutoff; "
    "numpy.random.default_rng(seed) consumed as calibration months, extra pilot years, pilot months, then the "
    "pre-cutoff and the post-cutoff probe evaluation months, months drawn uniformly without replacement within a "
    "year or a probe stratum in calendar order"
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
    ``probe_evaluation`` is the probe arm's frozen subset of the evaluation months
    (so disjoint from calibration and pilot); its stratum counts are enforced when
    the split is drawn, like the per-year counts.
    """

    models: tuple[str, ...]
    months: tuple[str, ...]
    pre_cutoff: Mapping[str, tuple[str, ...]]
    calibration: tuple[str, ...]
    pilot: tuple[str, ...]
    evaluation: tuple[str, ...]
    probe_evaluation: tuple[str, ...] = ()
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
        probe = _months(self.probe_evaluation, "probe_evaluation")
        outside = sorted(set(probe) - sets["evaluation"])
        if outside:
            raise ValueError(f"probe_evaluation months must be evaluation months (so neither calibration nor pilot "
                             f"months); got {outside}")
        object.__setattr__(self, "models", models)
        object.__setattr__(self, "months", months)
        object.__setattr__(self, "pre_cutoff", MappingProxyType(pre_cutoff))
        object.__setattr__(self, "calibration", calibration)
        object.__setattr__(self, "pilot", pilot)
        object.__setattr__(self, "evaluation", evaluation)
        object.__setattr__(self, "probe_evaluation", probe)
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


# ---------- 7.3: arms and variant sets (Requirements 8.1, 8.10) ----------

SAMPLING_ARM = "sampling"
SERVING_ARM = "serving"
#: The two ensemble arms; every ensemble claim names its arm.
ARMS: tuple[str, ...] = (SAMPLING_ARM, SERVING_ARM)
GEMMA_REGISTRY_ID = "google/gemma-3-4b-it"
#: Explicit arm to cutoff-registry id map: both arms serve Gemma 3 4B (the llama.cpp profile names it gemma3:4b).
ARM_REGISTRY_IDS: Mapping[str, str] = MappingProxyType({SAMPLING_ARM: GEMMA_REGISTRY_ID,
                                                        SERVING_ARM: GEMMA_REGISTRY_ID})
#: Provider profile of each arm (recall_guard/core/provider_profiles).
ARM_PROFILES: Mapping[str, str] = MappingProxyType({SAMPLING_ARM: "llamacpp-gemma3-4b",
                                                    SERVING_ARM: "modal-gemma3-4b"})
#: The Gemma 3 4B GGUF the llama.cpp profile declares (q4_K_M, the 1.2 receipt's pin); both local hosts serve it.
GEMMA_GGUF_DIGEST = "sha256:f06e91e18fc74295d0c781bf91622508c75655e2c18ed869d52447ffe1e667e1"
#: What ``build_plan.py`` reads from those profiles; the defaults when no profile is read. ``artifact_digest``
#: is the declared GGUF digest (None for the serving arm, whose weights the serving manifest's digest pins).
ARM_PROFILE_FACTS: Mapping[str, Mapping[str, Any]] = MappingProxyType({
    SAMPLING_ARM: MappingProxyType({"profile_id": "llamacpp-gemma3-4b", "served_model": "gemma3:4b",
                                    "top_logprobs": 5, "max_tokens": 512,
                                    "estimate_sources": ("top_k", "full_distribution"),
                                    "artifact_digest": GEMMA_GGUF_DIGEST}),
    SERVING_ARM: MappingProxyType({"profile_id": "modal-gemma3-4b", "served_model": GEMMA_REGISTRY_ID,
                                   "top_logprobs": 5, "max_tokens": 256, "estimate_sources": ("top_k",),
                                   "artifact_digest": None}),
})
ARM_MEASURES: Mapping[str, str] = MappingProxyType({
    SAMPLING_ARM: ("the model's sampling distribution: local llama.cpp at the frozen temperature, with seed "
                   "seed_base + draw_index per draw; the probe arm, gated by the pilot and power gate"),
    SERVING_ARM: ("serving nondeterminism: Modal vLLM at temperature 0 without a seed; paid, with its own limits "
                  "and approval, no pilot or power gate, reporting its achieved power"),
})

#: Arm roles (user decision 2026-11-10): the serving arm is the primary arm, the sampling arm the probe arm.
PRIMARY_ROLE, PROBE_ROLE = "primary", "probe"
ROLES: tuple[str, ...] = (PRIMARY_ROLE, PROBE_ROLE)
ARM_ROLES: Mapping[str, str] = MappingProxyType({SAMPLING_ARM: PROBE_ROLE, SERVING_ARM: PRIMARY_ROLE})
ROLE_REASON = (
    "user decision 2026-11-10: \"the big tests take too long. we don't have 44h of time. so modal it is. for "
    "probes, smaller tests 1-2h local is fine. but the iGPU is just a small chip.\" The local iGPU cannot carry "
    "the 44-hour run, so the Modal serving arm is the primary arm, approved after the local probe results, and "
    "the local sampling arm is a probe arm of about one to two hours of iGPU time per stage"
)
#: What each arm supplies to the study (design, "Arm sizing decision").
ARM_SUPPLIES: Mapping[str, str] = MappingProxyType({
    SERVING_ARM: ("the pre-registered tests: the pre- minus post-cutoff accuracy gap, the Brier and log-loss "
                  "differences and the negative control, on the none and top_k variants over the full evaluation "
                  "set, with its achieved power"),
    SAMPLING_ARM: ("the top_k versus full_distribution comparison and the convergence analysis (7.12) and a small "
                   "paired check of the gap, with all six variants on its frozen probe evaluation months; the pilot "
                   "and power gate (7.8) gates this arm only"),
})
#: Evaluation months each role's full run executes (a MonthSplit field name).
FULL_RUN_MONTHS: Mapping[str, str] = MappingProxyType({PRIMARY_ROLE: "evaluation", PROBE_ROLE: "probe_evaluation"})
#: Calibration corpora, each with its own draws per prompt and rendering (7.6).
CALIBRATION_KINDS: tuple[str, ...] = ("macro", "edgar")

#: Calibrator quality gates (7.6), frozen per arm: weak below this held-out AUC, at least this many control rows.
FIT_MIN_AUC = 0.6
FIT_MIN_VALID = 50
#: Evidence pass (6.4, 7.6, 7.7), frozen per arm with a full-distribution source: one batch per prompt and rendering
#: with this timeout, the alternatives kept per position, and the host_busy retries and the wait between them.
EVIDENCE_TIMEOUT_S = 600.0
EVIDENCE_TOP_K_KEEP = 100
EVIDENCE_HOST_BUSY_RETRIES = 5
EVIDENCE_HOST_BUSY_WAIT_S = 60.0

NO_ADJUSTMENT = "none"
TOP_K = "top_k"
FULL_DISTRIBUTION = "full_distribution"
#: Estimate sources, named as the profiles' ``estimate_sources`` and ``EstimateSource``.
ESTIMATE_SOURCES: tuple[str, ...] = (TOP_K, FULL_DISTRIBUTION)
DRAW_MODES: tuple[str, ...] = ("single", "ensemble")
ADJUSTMENTS: tuple[str, ...] = (NO_ADJUSTMENT, TOP_K, FULL_DISTRIBUTION)
_VARIANT_PARTS: Mapping[str, tuple[str, str]] = MappingProxyType({
    f"{mode}_{adjustment}": (mode, adjustment) for mode in DRAW_MODES for adjustment in ADJUSTMENTS})
#: The six variants (8.1), ``<draw mode>_<adjustment>``: two draw modes crossed with three adjustments.
VARIANTS: tuple[str, ...] = tuple(_VARIANT_PARTS)
#: The single draw is the ensemble's first draw (7.7).
SINGLE_DRAW_INDEX = 0
VARIANT_RULE = (
    "one recorded ensemble per item and rendering; every variant derives from it without another generation "
    "request: an ensemble variant reduces every draw, a single variant uses draw index 0 only; adjustment none "
    "passes the decision through, top_k and full_distribution use that source's per-draw estimates on the same "
    "draws, so the variants are paired"
)
SERVING_FULL_DISTRIBUTION_REASON = (
    "missing for the serving arm: the full-distribution host serves the local GGUF, which are different weights "
    "from the served model, and values from different builds are not comparable (Requirement 9.4)"
)
#: The primary (serving) arm's ensemble, as frozen by 7.3.
ENSEMBLE_DRAWS = 32
ENSEMBLE_MIN_PARSED = 24
#: The probe (sampling) arm's ensemble: 8 draws per rendering, at least 6 parsed (user decision 2026-11-10).
PROBE_DRAWS = 8
PROBE_MIN_PARSED = 6
#: The probe arm's EDGAR calibration rows take 4 draws each; its macro calibration months take PROBE_DRAWS.
PROBE_EDGAR_CALIBRATION_DRAWS = 4
SAMPLING_TEMPERATURE = 0.7
SERVING_TEMPERATURE = 0.0
SAMPLING_SEED_BASE = 1000
#: Largest per-draw seed, so ``seed_base + draw_index`` is a non-negative 32-bit signed integer for every host.
MAX_SEED = 2**31 - 1
#: Macro renderings (6.1); an EDGAR item has its one outlook prompt.
RENDERINGS: tuple[str, ...] = ("identifying", "anonymised")


def _choice(value: object, name: str, options: Sequence[str]) -> str:
    if not isinstance(value, str) or value not in options:
        raise ValueError(f"{name} must be one of {list(options)}; got {value!r}")
    return value


def _subset(value: object, name: str, universe: Sequence[str]) -> tuple[str, ...]:
    """Distinct members of ``universe``, in its order."""
    members = _ids(value, name)
    for member in members:
        _choice(member, f"{name} entry", universe)
    if tuple(member for member in universe if member in members) != members:
        raise ValueError(f"{name} must follow the order {list(universe)}; got {list(members)}")
    return members


def _boolean(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a bool; got {value!r}")
    return value


def variant_parts(variant: str) -> tuple[str, str]:
    """``(draw mode, adjustment)`` of a variant name."""
    return _VARIANT_PARTS[_choice(variant, "variant", VARIANTS)]


def variants_for(sources: Sequence[str]) -> tuple[str, ...]:
    """The variants an arm computes: every draw mode with no adjustment and with each offered estimate source."""
    offered = (NO_ADJUSTMENT, *_subset(sources, "estimate_sources", ESTIMATE_SOURCES))
    return tuple(variant for variant in VARIANTS if _VARIANT_PARTS[variant][1] in offered)


def _section_with(cls: type, data: object, name: str, **parsers: Callable[[Any], Any]) -> Any:
    """Like ``_section_from_dict``, parsing the nested fields named in ``parsers``."""
    mapping = _mapping(data, name)
    names = {f.name for f in fields(cls)}
    if set(mapping) != names:
        raise ValueError(f"{name} must have exactly the fields {sorted(names)}; "
                         f"missing {sorted(names - set(mapping))}, undeclared {sorted(map(str, set(mapping) - names))}")
    return cls(**{key: parsers[key](value) if key in parsers else value for key, value in mapping.items()})


_GGUF_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")


@dataclass(frozen=True)
class FitPlan:
    """The calibrator fit's frozen quality gates (7.6): the weak-calibrator AUC threshold and the minimum control rows."""

    min_auc: float = FIT_MIN_AUC
    min_valid: int = FIT_MIN_VALID

    def __post_init__(self) -> None:
        min_auc = _real(self.min_auc, "min_auc")
        if not 0.0 <= min_auc <= 1.0:
            raise ValueError(f"min_auc must be in [0, 1]; got {min_auc}")
        _integer(self.min_valid, "min_valid", minimum=1)
        object.__setattr__(self, "min_auc", min_auc)

    def to_dict(self) -> dict[str, Any]:
        return _section_dict(self)

    @classmethod
    def from_dict(cls, data: object, name: str = "fit") -> FitPlan:
        return _section_from_dict(cls, data, name)


@dataclass(frozen=True)
class EvidencePlan:
    """The evidence pass's frozen settings (6.4, 7.6, 7.7): one batch per prompt and rendering.

    ``timeout_s`` bounds one batch, ``top_k_keep`` is the alternatives kept per
    position, and a ``host_busy`` refusal is retried ``host_busy_retries`` times,
    ``host_busy_wait_s`` apart. Evidence calls never count against the
    generation limits (``EVIDENCE_RULE``).
    """

    timeout_s: float = EVIDENCE_TIMEOUT_S
    top_k_keep: int = EVIDENCE_TOP_K_KEEP
    host_busy_retries: int = EVIDENCE_HOST_BUSY_RETRIES
    host_busy_wait_s: float = EVIDENCE_HOST_BUSY_WAIT_S

    def __post_init__(self) -> None:
        timeout = _real(self.timeout_s, "timeout_s")
        if timeout <= 0:
            raise ValueError(f"timeout_s must be above 0; got {timeout}")
        wait = _real(self.host_busy_wait_s, "host_busy_wait_s")
        if wait < 0:
            raise ValueError(f"host_busy_wait_s must not be negative; got {wait}")
        _integer(self.top_k_keep, "top_k_keep", minimum=1)
        _integer(self.host_busy_retries, "host_busy_retries", minimum=0)
        object.__setattr__(self, "timeout_s", timeout)
        object.__setattr__(self, "host_busy_wait_s", wait)

    def to_dict(self) -> dict[str, Any]:
        return _section_dict(self)

    @classmethod
    def from_dict(cls, data: object, name: str = "evidence") -> EvidencePlan:
        return _section_from_dict(cls, data, name)


@dataclass(frozen=True)
class ArmPlan:
    """One ensemble arm: its role, profile, pins, sampling settings and variant set (Requirements 8.1 and 8.10).

    ``role`` is ``primary`` or ``probe`` with ``role_reason`` (the user's
    decision) and ``supplies`` (what the arm contributes to the study).
    ``artifact_digest`` is the GGUF digest its profile declared at build time
    (``sha256:<hex>``), which both local hosts must serve; None for an arm
    without a local GGUF (the serving arm, whose weights the serving manifest's
    digest pins). ``variants`` is every draw mode crossed with no adjustment and
    with each estimate source the profile offers; ``missing_variants`` gives the
    reason for each of the others, which the analysis reports as missing, never
    substituted. ``seed_base`` is set exactly when the arm samples (temperature
    above 0): draw ``i`` uses seed ``seed_base + i``. ``max_tokens`` and
    ``top_logprobs`` are the profile's per-request output cap and top-k.
    ``draws`` and ``min_parsed`` size every evaluation ensemble of the arm;
    ``calibration_draws`` gives the draws per prompt and rendering of each
    calibration corpus (macro, EDGAR). ``fit`` freezes the calibrator fit's
    gates and ``evidence`` the evidence pass, set exactly when the arm has a
    full-distribution source.
    """

    profile_id: str
    served_model: str
    measures: str
    temperature: float
    seed_base: int | None
    draws: int
    min_parsed: int
    top_logprobs: int
    max_tokens: int
    estimate_sources: tuple[str, ...]
    variants: tuple[str, ...]
    missing_variants: Mapping[str, str]
    gated: bool
    role: str
    role_reason: str
    supplies: str
    artifact_digest: str | None
    calibration_draws: Mapping[str, int]
    fit: FitPlan
    evidence: EvidencePlan | None

    def __post_init__(self) -> None:
        for name in ("profile_id", "served_model", "measures", "role_reason", "supplies"):
            _text(getattr(self, name), name)
        _choice(self.role, "role", ROLES)
        temperature = _real(self.temperature, "temperature")
        if temperature < 0:
            raise ValueError(f"temperature must be at least 0; got {temperature}")
        draws = _integer(self.draws, "draws", minimum=2)
        min_parsed = _integer(self.min_parsed, "min_parsed", minimum=1)
        if min_parsed > draws:
            raise ValueError(f"min_parsed must not be above draws; got min_parsed={min_parsed}, draws={draws}")
        _integer(self.top_logprobs, "top_logprobs", minimum=1)
        _integer(self.max_tokens, "max_tokens", minimum=1)
        calibration = _mapping(self.calibration_draws, "calibration_draws")
        if set(calibration) != set(CALIBRATION_KINDS):
            raise ValueError(f"calibration_draws must give the draws of exactly the corpora {list(CALIBRATION_KINDS)}; "
                             f"got {sorted(map(str, calibration))}")
        calibration = {kind: _integer(calibration[kind], f"calibration_draws[{kind!r}]", minimum=1)
                       for kind in CALIBRATION_KINDS}
        if temperature > 0:
            if self.seed_base is None:
                raise ValueError("an arm that samples (temperature above 0) needs a seed_base, so every draw "
                                 "carries a recorded seed")
            seed_base = _integer(self.seed_base, "seed_base", minimum=0)
            widest = max(draws, *calibration.values())
            if seed_base + widest - 1 > MAX_SEED:
                raise ValueError(f"seed_base + draws - 1 must be at most {MAX_SEED} for every ensemble of the arm; "
                                 f"got {seed_base + widest - 1}")
        elif self.seed_base is not None:
            raise ValueError("an arm at temperature 0 sends no seed; seed_base must be None")
        sources = _subset(self.estimate_sources, "estimate_sources", ESTIMATE_SOURCES)
        if not sources:
            raise ValueError("estimate_sources must name at least one source")
        variants = _subset(self.variants, "variants", VARIANTS)
        expected = variants_for(sources)
        if variants != expected:
            raise ValueError(f"variants must be every draw mode crossed with no adjustment and each offered "
                             f"estimate source {list(sources)}: {list(expected)}; got {list(variants)}")
        missing = _text_map(self.missing_variants, "missing_variants")
        absent = [variant for variant in VARIANTS if variant not in variants]
        if set(missing) != set(absent):
            raise ValueError(f"missing_variants must give a reason for exactly the variants the arm cannot "
                             f"compute {absent}; got {sorted(missing)}")
        _boolean(self.gated, "gated")
        if self.artifact_digest is not None and (not isinstance(self.artifact_digest, str)
                                                 or not _GGUF_DIGEST.fullmatch(self.artifact_digest)):
            raise ValueError(f"artifact_digest must be sha256: and 64 lowercase hex characters, or None; "
                             f"got {self.artifact_digest!r}")
        full = FULL_DISTRIBUTION in sources
        if full and self.artifact_digest is None:
            raise ValueError("an arm with a full_distribution source needs the artifact_digest of the GGUF its "
                             "generation and full-distribution hosts serve")
        if not isinstance(self.fit, FitPlan):
            raise TypeError(f"fit must be a FitPlan; got {type(self.fit).__name__}")
        if full != isinstance(self.evidence, EvidencePlan) or (not full and self.evidence is not None):
            raise ValueError("evidence must be an EvidencePlan exactly when the arm has a full_distribution source, "
                             f"else None; got {type(self.evidence).__name__}")
        object.__setattr__(self, "temperature", temperature)
        object.__setattr__(self, "estimate_sources", sources)
        object.__setattr__(self, "variants", variants)
        object.__setattr__(self, "missing_variants", missing)
        object.__setattr__(self, "calibration_draws", MappingProxyType(calibration))

    @classmethod
    def for_arm(cls, arm: str, *, profile_id: str, served_model: str, top_logprobs: int, max_tokens: int,
                estimate_sources: Sequence[str], artifact_digest: str | None = None) -> ArmPlan:
        """The frozen settings of ``arm`` (role, temperature, seeds, draws, gate, fit, evidence) on its profile's facts.

        The sampling arm is the probe arm (8 draws, at least 6 parsed; macro
        calibration 8 and EDGAR calibration 4 draws per prompt); the serving arm
        the primary arm (32 draws, at least 24 parsed, both calibrations 32).
        """
        sampling = _choice(arm, "arm", ARMS) == SAMPLING_ARM
        variants = variants_for(estimate_sources)
        missing = {}
        for variant in VARIANTS:
            if variant not in variants:
                adjustment = _VARIANT_PARTS[variant][1]
                missing[variant] = (SERVING_FULL_DISTRIBUTION_REASON
                                    if arm == SERVING_ARM and adjustment == FULL_DISTRIBUTION
                                    else f"missing: the {profile_id} profile offers no {adjustment} estimate source")
        draws, min_parsed = (PROBE_DRAWS, PROBE_MIN_PARSED) if sampling else (ENSEMBLE_DRAWS, ENSEMBLE_MIN_PARSED)
        calibration = ({"macro": PROBE_DRAWS, "edgar": PROBE_EDGAR_CALIBRATION_DRAWS} if sampling
                       else {"macro": ENSEMBLE_DRAWS, "edgar": ENSEMBLE_DRAWS})
        return cls(profile_id=profile_id, served_model=served_model, measures=ARM_MEASURES[arm],
                   temperature=SAMPLING_TEMPERATURE if sampling else SERVING_TEMPERATURE,
                   seed_base=SAMPLING_SEED_BASE if sampling else None, draws=draws,
                   min_parsed=min_parsed, top_logprobs=top_logprobs, max_tokens=max_tokens,
                   estimate_sources=tuple(estimate_sources), variants=variants, missing_variants=missing,
                   gated=sampling, role=ARM_ROLES[arm], role_reason=ROLE_REASON, supplies=ARM_SUPPLIES[arm],
                   artifact_digest=artifact_digest, calibration_draws=calibration, fit=FitPlan(),
                   evidence=EvidencePlan() if FULL_DISTRIBUTION in estimate_sources else None)

    def to_dict(self) -> dict[str, Any]:
        return _section_dict(self)

    @classmethod
    def from_dict(cls, data: object, name: str = "arm") -> ArmPlan:
        def evidence(value: object) -> EvidencePlan | None:
            return None if value is None else EvidencePlan.from_dict(value, f"{name}.evidence")
        return _section_with(cls, data, name, fit=lambda value: FitPlan.from_dict(value, f"{name}.fit"),
                             evidence=evidence)


@dataclass(frozen=True)
class StudyArms:
    """Both ensemble arms, the explicit arm-to-registry-id map and the frozen variant names.

    The serving arm is the primary arm: it runs at temperature 0 with no pilot or
    power gate, computes no adjustment and top-k only, and carries the
    pre-registered tests. The sampling arm is the probe arm and the gated arm: it
    samples at a temperature above 0, computes the six variants on its frozen
    probe months and runs the local GGUF whose digest it freezes.
    """

    registry_ids: Mapping[str, str]
    arms: Mapping[str, ArmPlan]
    variants: tuple[str, ...] = VARIANTS
    single_draw_index: int = SINGLE_DRAW_INDEX
    rule: str = VARIANT_RULE

    def __post_init__(self) -> None:
        registry = _text_map(self.registry_ids, "registry_ids")
        if set(registry) != set(ARMS):
            raise ValueError(f"registry_ids must map exactly the arms {list(ARMS)}; got {sorted(registry)}")
        arms = _mapping(self.arms, "arms")
        if set(arms) != set(ARMS):
            raise ValueError(f"arms must hold exactly the arms {list(ARMS)}; got {sorted(map(str, arms))}")
        for name in ARMS:
            if not isinstance(arms[name], ArmPlan):
                raise TypeError(f"arms[{name!r}] must be an ArmPlan; got {type(arms[name]).__name__}")
        if tuple(_sequence(self.variants, "variants")) != VARIANTS:
            raise ValueError(f"variants must be the six frozen variants {list(VARIANTS)}")
        if _integer(self.single_draw_index, "single_draw_index", minimum=0) != SINGLE_DRAW_INDEX:
            raise ValueError("single_draw_index must be 0: the single draw is the ensemble's first draw")
        _text(self.rule, "rule")
        sampling, serving = arms[SAMPLING_ARM], arms[SERVING_ARM]
        if not sampling.gated or sampling.temperature <= 0 or sampling.variants != VARIANTS:
            raise ValueError("the sampling arm is the gated arm: it samples at a temperature above 0 and computes "
                             "the six variants")
        if serving.gated or serving.temperature != 0 or serving.variants != variants_for((TOP_K,)):
            raise ValueError("the serving arm runs at temperature 0 without a pilot or power gate and computes no "
                             "adjustment and top-k only")
        for name in ARMS:
            if arms[name].role != ARM_ROLES[name]:
                raise ValueError(f"the {name} arm is the {ARM_ROLES[name]} arm (user decision 2026-11-10): the "
                                 f"serving arm is primary and the sampling arm the probe arm; got "
                                 f"{arms[name].role!r}")
        if sampling.artifact_digest is None:
            raise ValueError("the sampling arm runs the local GGUF: it needs its artifact_digest")
        object.__setattr__(self, "registry_ids", registry)
        object.__setattr__(self, "arms", MappingProxyType({name: arms[name] for name in ARMS}))
        object.__setattr__(self, "variants", VARIANTS)

    def to_dict(self) -> dict[str, Any]:
        return _section_dict(self)

    @classmethod
    def from_dict(cls, data: object) -> StudyArms:
        def arms(value: object) -> dict[str, ArmPlan]:
            return {name: ArmPlan.from_dict(item, f"arms.arms[{name!r}]")
                    for name, item in _mapping(value, "arms.arms").items()}
        return _section_with(cls, data, "arms", arms=arms)


def study_arms(facts: Mapping[str, Mapping[str, Any]] | None = None, *,
               registry_ids: Mapping[str, str] = ARM_REGISTRY_IDS) -> StudyArms:
    """Both arms from their profile facts (``ARM_PROFILE_FACTS`` by default) and the arm-to-registry-id map."""
    facts = ARM_PROFILE_FACTS if facts is None else _mapping(facts, "facts")
    if set(facts) != set(ARMS):
        raise ValueError(f"facts must describe exactly the arms {list(ARMS)}")
    return StudyArms(registry_ids=registry_ids,
                     arms={arm: ArmPlan.for_arm(arm, **dict(_mapping(facts[arm], f"facts[{arm!r}]")))
                           for arm in ARMS})


# ---------- 7.3: pilot settings (Requirement 8.9) ----------

PILOT_MONTHS = len(PILOT_YEARS) + PILOT_EXTRA_YEARS
PILOT_DRAWS = 8
#: The pilot's own minimum parsed draws: three quarters of its draws, as 24 of 32 in the full run.
PILOT_MIN_PARSED = 6
#: The pilot's noise statistic is the correctness difference of this variant.
PILOT_NOISE_VARIANT = "ensemble_top_k"
#: What a gate below power_threshold does: "inform" lets the probe arm proceed after stop_for_decision with its
#: expected power reported, "block" waits for a decision (user decision 2026-11-10: "inform", because at effect
#: size 0.10 the gate cannot pass with month-level accuracy differences, and the serving arm carries the
#: confirmatory tests).
PILOT_GATE_POLICIES: tuple[str, ...] = ("inform", "block")
PILOT_GATE_POLICY = "inform"
PILOT_RULE = (
    "the pilot months of the month split run through the same guarded execution on the gated arm only, one "
    "recorded ensemble of `draws` draws per month and rendering with `min_parsed`, seeds seed_base + draw_index; "
    "per month the noise statistic is the identifying-minus-anonymised correctness difference of the noise "
    "variant (values in {-1, 0, 1}) and noise_sd its standard deviation over the pilot months; simulated power "
    "evaluates the frozen effect size with the pre-cutoff and post-cutoff counts of the months the gated arm's full "
    "run executes (the probe arm's probe evaluation months, so the gate does not overstate its power); the full run "
    "starts only at power of at least power_threshold; pilot months are excluded from the confirmatory analysis"
)


def _evaluation_counts(split: MonthSplit, model: str, months: Sequence[str] | None = None) -> tuple[int, int]:
    """Months on or before, and after, ``model``'s cutoff among ``months`` (every evaluation month by default)."""
    if model not in split.pre_cutoff:
        raise ValueError(f"{model!r} is not a model of the month split {list(split.models)}")
    months = split.evaluation if months is None else months
    pre = set(split.pre_cutoff[model])
    before = sum(month in pre for month in months)
    return before, len(months) - before


def _arm_counts(split: MonthSplit, arms: StudyArms, arm: str) -> tuple[int, int]:
    """Pre-cutoff and post-cutoff counts of the months ``arm``'s full run executes, by its model's cutoff."""
    return _evaluation_counts(split, arms.registry_ids[arm], _role_months(split, arms.arms[arm]))


@dataclass(frozen=True)
class PilotSettings:
    """The pilot of the gated arm: about 12 pre-cutoff months times 8 draws per rendering, its own minimum.

    ``pre_cutoff_evaluation_months`` and ``post_cutoff_evaluation_months`` are the
    frozen item counts the power simulation applies the pilot noise to: those of
    the months the gated arm's full run executes, the probe arm's probe
    evaluation months (15 and 5 for Gemma 3 4B with the default seed), so the
    gate never assumes the primary arm's 68 and 22. The primary arm's counts,
    for its achieved-power report, are :func:`evaluation_month_counts`.
    """

    pre_cutoff_evaluation_months: int
    post_cutoff_evaluation_months: int
    arm: str = SAMPLING_ARM
    months: int = PILOT_MONTHS
    draws: int = PILOT_DRAWS
    min_parsed: int = PILOT_MIN_PARSED
    renderings: tuple[str, ...] = RENDERINGS
    noise_variant: str = PILOT_NOISE_VARIANT
    rule: str = PILOT_RULE
    #: "inform" (the probe arm proceeds after stop_for_decision, its expected power reported) or "block": inform,
    #: because at effect size 0.10 the gate cannot pass with month-level accuracy differences, and the serving arm
    #: carries the confirmatory tests (user decision 2026-11-10).
    gate_policy: str = PILOT_GATE_POLICY

    def __post_init__(self) -> None:
        _choice(self.gate_policy, "gate_policy", PILOT_GATE_POLICIES)
        _integer(self.pre_cutoff_evaluation_months, "pre_cutoff_evaluation_months", minimum=1)
        _integer(self.post_cutoff_evaluation_months, "post_cutoff_evaluation_months", minimum=1)
        _choice(self.arm, "pilot arm", ARMS)
        _integer(self.months, "pilot months", minimum=1)
        draws = _integer(self.draws, "pilot draws", minimum=1)
        min_parsed = _integer(self.min_parsed, "pilot min_parsed", minimum=1)
        if min_parsed > draws:
            raise ValueError(f"pilot min_parsed must not be above the pilot's draws; got min_parsed={min_parsed}, "
                             f"draws={draws}")
        if tuple(_sequence(self.renderings, "renderings")) != RENDERINGS:
            raise ValueError(f"the pilot draws both renderings {list(RENDERINGS)}")
        _choice(self.noise_variant, "noise_variant", VARIANTS)
        _text(self.rule, "rule")
        object.__setattr__(self, "renderings", RENDERINGS)

    @classmethod
    def for_split(cls, split: MonthSplit, arms: StudyArms) -> PilotSettings:
        """The frozen pilot of the gated arm on this month split, with the counts of that arm's full-run months."""
        before, after = _arm_counts(split, arms, SAMPLING_ARM)
        return cls(pre_cutoff_evaluation_months=before, post_cutoff_evaluation_months=after, months=len(split.pilot))

    def to_dict(self) -> dict[str, Any]:
        return _section_dict(self)

    @classmethod
    def from_dict(cls, data: object) -> PilotSettings:
        return _section_from_dict(cls, data, "pilot")


def _check_pilot(pilot: PilotSettings, split: MonthSplit, arms: StudyArms) -> None:
    arm = arms.arms[pilot.arm]
    if not arm.gated:
        raise ValueError(f"the pilot runs on the gated arm only; {pilot.arm!r} is not gated")
    if pilot.months != len(split.pilot):
        raise ValueError(f"pilot months must be the {len(split.pilot)} pilot months of the split; got {pilot.months}")
    if pilot.draws > arm.draws:
        raise ValueError(f"pilot draws must not exceed the arm's {arm.draws} draws (the pilot seeds are the first "
                         f"of the arm's seeds); got {pilot.draws}")
    if pilot.noise_variant not in arm.variants:
        raise ValueError(f"the pilot noise variant {pilot.noise_variant!r} is not a variant of the {pilot.arm} arm")
    counts = _arm_counts(split, arms, pilot.arm)
    if (pilot.pre_cutoff_evaluation_months, pilot.post_cutoff_evaluation_months) != counts:
        raise ValueError(f"pilot evaluation month counts must be those of the {pilot.arm} arm's full-run months "
                         f"{counts} (pre-cutoff, post-cutoff); "
                         f"got {(pilot.pre_cutoff_evaluation_months, pilot.post_cutoff_evaluation_months)}")


# ---------- 7.3: limits per arm and stage (Requirement 11.3) ----------

#: Sampling-arm stages in execution order (local hosts, no cost).
SAMPLING_STAGES: tuple[str, ...] = ("calibration", "pilot", "full_run", "negative_control")
#: Serving-arm stages in execution order; equal to deploy/modal/stage_limits.STAGES.
SERVING_STAGES: tuple[str, ...] = ("weight_staging", "deployment", "smoke_run", "calibration", "full_run",
                                   "negative_control")
WEIGHT_STAGING = SERVING_STAGES[0]
#: The serving manifest of task 3.4, repository-relative.
SERVING_MANIFEST = "configs/modal/gemma-3-4b-measurement.json"
ROOT = Path(__file__).resolve().parents[2]
#: Sampling (probe) arm: the design's planning rate of 6 s per generation on the iGPU (user decision 2026-11-10).
SAMPLING_SECONDS_PER_GENERATION = 6
#: Headroom on the planning rate, so a stage slower than 6 s per generation still completes (decision 2026-11-10).
PROBE_TIME_HEADROOM = 1.25
#: The probe arm's client timeout (llamacpp-gemma3-4b timeout_s): the ledger admits a request only while elapsed time
#: plus this timeout fits max_seconds, so each probe stage adds it once.
PROBE_REQUEST_TIMEOUT_SECONDS = 300
#: Every probe stage is bounded to two and a half hours (within the user's one to two hours of generation at the
#: planning rate, plus the headroom and one timeout).
PROBE_STAGE_MAX_SECONDS = 9000
#: CPU-only weight staging timeout: about 8.6 GB of pinned weights at 2.4 MB/s or faster.
WEIGHT_STAGING_SECONDS = 3600
#: The expiry controller of deploy/modal/serve.py (scripts/measure_modal_model.py prices it the same way).
CONTROLLER_CORES = 0.125
CONTROLLER_MEMORY_GIB = 0.125
CONTROLLER_ALLOWANCE_SECONDS = 150
EVIDENCE_RULE = (
    "full-distribution evidence calls do not count against max_requests, max_output_tokens or max_seconds: they "
    "are teacher-forcing calls on the local full-distribution host, issued after generation with that host loaded "
    "alone, each bounded by its own timeout, at most one per recorded response (or one per item and rendering in "
    "batch mode); they never add a generation request"
)
LIMITS_RULE = (
    "one recorded ensemble per item and rendering, never one per variant: max_requests = the sum over ensemble "
    "groups of ensembles x that group's draws (draws_by_group, else draws), plus the deployment's one eligibility "
    "post; max_output_tokens = max_requests x the arm's max_tokens. Each arm's calibration takes its "
    "calibration_draws per corpus (macro, edgar), its pilot the pilot's draws and every other stage the arm's draws; "
    "the full run covers the evaluation months of the arm's role (primary: every evaluation month; probe: the "
    "split's probe_evaluation months). Sampling arm (local probe arm, no cost): max_seconds = ceil(max_requests x "
    "sampling_seconds_per_generation x probe_time_headroom) + probe_request_timeout_seconds (the ledger admits a "
    "request only while elapsed time plus the client timeout fits), at most probe_stage_max_seconds per stage. "
    "Serving arm (paid), stages "
    "in execution order weight_staging, deployment, smoke_run, calibration (macro and EDGAR in one serve), "
    "full_run, negative_control, each stage after weight staging one serve lifecycle, five in all: max_seconds = "
    "startup_seconds + max_requests x serving_seconds_per_post, where serving_seconds_per_post = "
    "ceil((gpu_phase_seconds - 5 x startup_seconds) / inference_posts) is the manifest's worst-case post; "
    "max_usd = rate x (max_seconds + cleanup_seconds) + controller rate x (max_seconds + "
    "controller_allowance_seconds), rounded up to the cent, with rate = gpu x gpu_count + cores x core + memory x "
    "memory rate (scripts/measure_modal_model.compute_rate); weight staging is CPU-only with 0 requests and 0 "
    "tokens: max_usd = (cores x core + memory x memory rate) x weight_staging_seconds, rounded up to the cent; "
    "every stage, and the stages together, fit the manifest's reviewed ceiling"
)


def _usd_up(value: float) -> float:
    """USD rounded up to the cent (an upper bound, never below the value)."""
    return math.ceil(round(value * 100, 6)) / 100


@dataclass(frozen=True)
class StageBudget:
    """One stage's frozen limits, sized to one recorded ensemble per item and rendering.

    ``ensembles`` counts the ensembles per item group (items times renderings);
    each group's ensemble has ``draws`` draws unless ``draws_by_group`` names
    another count for it (the probe arm's EDGAR calibration rows).
    ``max_requests`` is the sum of ensembles times their draws plus
    ``extra_requests`` (the serving deployment's eligibility post). ``max_usd``
    is set for the paid serving arm only.
    """

    stage: str
    ensembles: Mapping[str, int]
    draws: int
    extra_requests: int
    max_requests: int
    max_output_tokens: int
    max_seconds: int
    max_usd: float | None
    draws_by_group: Mapping[str, int] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        _text(self.stage, "stage")
        groups = _mapping(self.ensembles, "ensembles")
        ensembles = {_text(group, "ensembles group"): _integer(count, f"ensembles[{group!r}]", minimum=1)
                     for group, count in sorted(groups.items())}
        draws = _integer(self.draws, "draws", minimum=0)
        overrides = _mapping(self.draws_by_group, "draws_by_group")
        by_group = {}
        for group, count in sorted(overrides.items()):
            if group not in ensembles:
                raise ValueError(f"{self.stage}: draws_by_group names {group!r}, which is not an ensemble group of "
                                 f"the stage {sorted(ensembles)}")
            if _integer(count, f"draws_by_group[{group!r}]", minimum=1) == draws:
                raise ValueError(f"{self.stage}: draws_by_group[{group!r}] repeats the stage's draws {draws}; list "
                                 "only the groups whose ensembles take another count")
            by_group[group] = count
        extra = _integer(self.extra_requests, "extra_requests", minimum=0)
        expected = sum(count * by_group.get(group, draws) for group, count in ensembles.items()) + extra
        if _integer(self.max_requests, "max_requests", minimum=0) != expected:
            raise ValueError(f"{self.stage}: max_requests must be the ensembles times their draws plus the extra "
                             f"requests ({expected}); got {self.max_requests}")
        _integer(self.max_output_tokens, "max_output_tokens", minimum=0)
        _integer(self.max_seconds, "max_seconds", minimum=1)
        if self.max_usd is not None:
            usd = _real(self.max_usd, "max_usd")
            if usd <= 0:
                raise ValueError(f"{self.stage}: max_usd must be positive or None; got {usd}")
            object.__setattr__(self, "max_usd", usd)
        object.__setattr__(self, "ensembles", MappingProxyType(ensembles))
        object.__setattr__(self, "draws_by_group", MappingProxyType(by_group))

    def group_draws(self) -> dict[str, int]:
        """The draws of each ensemble group's ensembles."""
        return {group: self.draws_by_group.get(group, self.draws) for group in self.ensembles}

    def to_dict(self) -> dict[str, Any]:
        return _section_dict(self)

    @classmethod
    def from_dict(cls, data: object, name: str = "stage") -> StageBudget:
        return _section_from_dict(cls, data, name)


@dataclass(frozen=True)
class ServingManifest:
    """The serving manifest's reviewed rates and ceiling (task 3.4), identified by its file's SHA-256."""

    path: str
    sha256: str
    pricing_checked_at: str
    gpu_usd_per_second: float
    physical_core_usd_per_second: float
    memory_gib_usd_per_second: float
    gpu_count: int
    cpu_physical_cores: int
    memory_gib: int
    inference_posts: int
    eligibility_posts: int
    output_tokens_per_post: int
    total_output_tokens: int
    gpu_phase_seconds: int
    startup_seconds: int
    cleanup_seconds: int
    temperature: float
    top_logprobs: int
    target_usd: float

    def __post_init__(self) -> None:
        path = _text(self.path, "manifest path")
        if "\\" in path or path.startswith("/") or ".." in path.split("/"):
            raise ValueError(f"manifest path must be repository-relative in POSIX form; got {path!r}")
        _sha256(self.sha256, "manifest sha256")
        _text(self.pricing_checked_at, "pricing_checked_at")
        for name in ("gpu_usd_per_second", "physical_core_usd_per_second", "memory_gib_usd_per_second"):
            rate = _real(getattr(self, name), name)
            if not 0 < rate < 1:
                raise ValueError(f"{name} must be in (0, 1); got {rate}")
            object.__setattr__(self, name, rate)
        for name in ("gpu_count", "cpu_physical_cores", "memory_gib", "inference_posts", "output_tokens_per_post",
                     "total_output_tokens", "gpu_phase_seconds", "top_logprobs"):
            _integer(getattr(self, name), name, minimum=1)
        for name in ("eligibility_posts", "startup_seconds", "cleanup_seconds"):
            _integer(getattr(self, name), name, minimum=0)
        temperature = _real(self.temperature, "manifest temperature")
        target = _real(self.target_usd, "target_usd")
        if temperature < 0 or target <= 0:
            raise ValueError("the manifest temperature must be at least 0 and target_usd positive")
        object.__setattr__(self, "temperature", temperature)
        object.__setattr__(self, "target_usd", target)

    @classmethod
    def from_manifest(cls, path: str, data: bytes) -> ServingManifest:
        """The record of a serving manifest's bytes; ``deploy.modal.manifest.load_plan`` validates the rest."""
        try:
            raw = json.loads(data.decode("utf-8"), object_pairs_hook=_no_duplicate_keys, parse_constant=_no_constant)
            cost, resources, limits = raw["cost"], raw["resources"], raw["limits"]
            values = {
                "pricing_checked_at": cost["pricing_checked_at"], "gpu_usd_per_second": cost["gpu_usd_per_second"],
                "physical_core_usd_per_second": cost["physical_core_usd_per_second"],
                "memory_gib_usd_per_second": cost["memory_gib_usd_per_second"], "target_usd": cost["target_usd"],
                **{name: resources[name] for name in ("gpu_count", "cpu_physical_cores", "memory_gib")},
                **{name: limits[name] for name in ("inference_posts", "eligibility_posts", "output_tokens_per_post",
                                                   "total_output_tokens", "gpu_phase_seconds", "startup_seconds",
                                                   "cleanup_seconds", "temperature", "top_logprobs")},
            }
        except (KeyError, TypeError, UnicodeDecodeError) as exc:
            raise ValueError(f"serving manifest {path} lacks a reviewed field: {exc!r}") from None
        return cls(path=path, sha256=hashlib.sha256(data).hexdigest(), **values)

    @property
    def cpu_usd_per_second(self) -> float:
        return (self.cpu_physical_cores * self.physical_core_usd_per_second
                + self.memory_gib * self.memory_gib_usd_per_second)

    @property
    def usd_per_second(self) -> float:
        """The serving app's configured rate, as ``measure_modal_model.compute_rate``."""
        return (self.gpu_usd_per_second * self.gpu_count + self.physical_core_usd_per_second * self.cpu_physical_cores
                + self.memory_gib_usd_per_second * self.memory_gib)

    def to_dict(self) -> dict[str, Any]:
        return _section_dict(self)

    @classmethod
    def from_dict(cls, data: object) -> ServingManifest:
        return _section_from_dict(cls, data, "limits.manifest")


def load_serving_manifest(path: str = SERVING_MANIFEST, *, root: Path = ROOT) -> ServingManifest:
    """The serving manifest record of ``root / path``."""
    return ServingManifest.from_manifest(path, (Path(root) / path).read_bytes())


@dataclass(frozen=True)
class PlanLimits:
    """Frozen request, token, time and (serving arm) cost limits per arm and stage, in execution order.

    ``stages`` maps each arm to its :class:`StageBudget` tuple; ``manifest`` holds
    the serving manifest's digest, rates and ceiling the serving limits are derived
    from. The sizing inputs are fields too, so the plan recomputes every stage
    with :func:`size_limits` and refuses any other value. The serving arm's
    StageLimits files are rendered by :func:`stage_limits_files` and carry the
    plan's ``plan_sha256``, which cannot be part of the content it hashes.
    """

    manifest: ServingManifest
    stages: Mapping[str, tuple[StageBudget, ...]]
    serving_seconds_per_post: int
    sampling_seconds_per_generation: int = SAMPLING_SECONDS_PER_GENERATION
    weight_staging_seconds: int = WEIGHT_STAGING_SECONDS
    controller_cores: float = CONTROLLER_CORES
    controller_memory_gib: float = CONTROLLER_MEMORY_GIB
    controller_allowance_seconds: int = CONTROLLER_ALLOWANCE_SECONDS
    probe_stage_max_seconds: int = PROBE_STAGE_MAX_SECONDS
    #: Headroom on the planning rate, so a probe stage somewhat slower than 6 s per generation still completes.
    probe_time_headroom: float = PROBE_TIME_HEADROOM
    #: The probe arm's client timeout, added once per stage because admission needs elapsed time plus it to fit.
    probe_request_timeout_seconds: int = PROBE_REQUEST_TIMEOUT_SECONDS
    evidence_requests: str = EVIDENCE_RULE
    rule: str = LIMITS_RULE

    def __post_init__(self) -> None:
        if not isinstance(self.manifest, ServingManifest):
            raise TypeError(f"manifest must be a ServingManifest; got {type(self.manifest).__name__}")
        groups = _mapping(self.stages, "stages")
        if set(groups) != set(ARMS):
            raise ValueError(f"stages must hold exactly the arms {list(ARMS)}")
        stages = {}
        for arm, names in ((SAMPLING_ARM, SAMPLING_STAGES), (SERVING_ARM, SERVING_STAGES)):
            budgets = _sequence(groups[arm], f"stages[{arm!r}]")
            if any(not isinstance(budget, StageBudget) for budget in budgets):
                raise TypeError(f"stages[{arm!r}] must hold StageBudget values")
            if tuple(budget.stage for budget in budgets) != names:
                raise ValueError(f"the {arm} arm's stages must be {list(names)} in execution order")
            if any((budget.max_usd is None) != (arm == SAMPLING_ARM) for budget in budgets):
                raise ValueError("max_usd is set for every serving-arm stage and for no sampling-arm stage")
            stages[arm] = budgets
        for name in ("serving_seconds_per_post", "sampling_seconds_per_generation", "weight_staging_seconds",
                     "controller_allowance_seconds", "probe_stage_max_seconds", "probe_request_timeout_seconds"):
            _integer(getattr(self, name), name, minimum=1)
        for name in ("controller_cores", "controller_memory_gib"):
            value = _real(getattr(self, name), name)
            if value <= 0:
                raise ValueError(f"{name} must be positive; got {value}")
            object.__setattr__(self, name, value)
        object.__setattr__(self, "probe_time_headroom", _headroom(self.probe_time_headroom))
        _text(self.evidence_requests, "evidence_requests")
        _text(self.rule, "rule")
        _check_ceiling(stages[SERVING_ARM], self.manifest)
        _check_probe_bound(stages[SAMPLING_ARM], self.probe_stage_max_seconds)
        object.__setattr__(self, "stages", MappingProxyType(stages))

    def settings(self) -> dict[str, Any]:
        """The sizing inputs :func:`size_limits` takes besides the plan sections and the manifest."""
        return {name: getattr(self, name) for name in (
            "sampling_seconds_per_generation", "weight_staging_seconds", "controller_cores", "controller_memory_gib",
            "controller_allowance_seconds", "probe_stage_max_seconds", "probe_time_headroom",
            "probe_request_timeout_seconds")}

    def stage(self, arm: str, stage: str) -> StageBudget:
        for budget in self.stages[_choice(arm, "arm", ARMS)]:
            if budget.stage == stage:
                return budget
        raise ValueError(f"the {arm} arm has no stage {stage!r}")

    @property
    def serving_usd_total(self) -> float:
        """The serving arm's maximum cost: every stage's ``max_usd``, summed exactly."""
        return float(sum((Decimal(str(budget.max_usd)) for budget in self.stages[SERVING_ARM]), Decimal(0)))

    def to_dict(self) -> dict[str, Any]:
        return _section_dict(self)

    @classmethod
    def from_dict(cls, data: object) -> PlanLimits:
        def stages(value: object) -> dict[str, tuple[StageBudget, ...]]:
            return {arm: tuple(StageBudget.from_dict(item, f"limits.stages[{arm!r}][{index}]")
                               for index, item in enumerate(_sequence(items, f"limits.stages[{arm!r}]")))
                    for arm, items in _mapping(value, "limits.stages").items()}
        return _section_with(cls, data, "limits", manifest=ServingManifest.from_dict, stages=stages)


def _headroom(value: object) -> float:
    headroom = _real(value, "probe_time_headroom")
    if headroom < 1.0:
        raise ValueError(f"probe_time_headroom must be at least 1 (a margin above the planning rate); got {headroom}")
    return headroom


def _probe_seconds(requests: int, rate: int, headroom: float, timeout: int) -> int:
    """``ceil(requests x rate x headroom) + timeout``, in exact decimal arithmetic."""
    return math.ceil(Decimal(requests * rate) * Decimal(str(headroom))) + timeout


def _check_probe_bound(stages: Sequence[StageBudget], bound: int) -> None:
    """Every stage of the probe (sampling) arm fits its time bound of about two hours of iGPU time."""
    over = {budget.stage: budget.max_seconds for budget in stages if budget.max_seconds > bound}
    if over:
        raise ValueError(f"the probe arm's stages must each fit probe_stage_max_seconds = {bound} s at the planning "
                         f"rate; {over} exceed it")


def _check_ceiling(stages: Sequence[StageBudget], manifest: ServingManifest) -> None:
    """Each serving stage, and the stages together, fit the manifest's reviewed ceiling.

    Requests and tokens count every stage; seconds and USD the GPU stages only,
    since the manifest's GPU phase and target exclude the CPU-only weight staging.
    """
    ceiling = {"max_requests": manifest.inference_posts, "max_output_tokens": manifest.total_output_tokens,
               "max_seconds": manifest.gpu_phase_seconds, "max_usd": manifest.target_usd}
    for budget in stages:
        for name, limit in ceiling.items():
            if getattr(budget, name) > limit:
                raise ValueError(f"serving stage {budget.stage}: {name} {getattr(budget, name)} exceeds the "
                                 f"manifest's reviewed ceiling of {limit}")
    staging, gpu = stages[0], stages[1:]
    if staging.max_requests or staging.max_output_tokens:
        raise ValueError("weight staging is CPU-only: 0 requests and 0 output tokens")
    if staging.max_usd < manifest.cpu_usd_per_second * staging.max_seconds:
        raise ValueError("weight staging's max_usd must cover running to its timeout")
    totals = {"max_requests": sum(budget.max_requests for budget in stages),
              "max_output_tokens": sum(budget.max_output_tokens for budget in stages),
              "max_seconds": sum(budget.max_seconds for budget in gpu),
              "max_usd": sum((Decimal(str(budget.max_usd)) for budget in gpu), Decimal(0))}
    for name, total in totals.items():
        limit = Decimal(str(ceiling[name])) if name == "max_usd" else ceiling[name]
        if total > limit:
            raise ValueError(f"the serving stages together need {name} = {total}, above the manifest's reviewed "
                             f"ceiling of {ceiling[name]} for the whole arm")


def size_limits(split: MonthSplit, edgar: EdgarSplit, arms: StudyArms, pilot: PilotSettings,
                manifest: ServingManifest, *, sampling_seconds_per_generation: int = SAMPLING_SECONDS_PER_GENERATION,
                weight_staging_seconds: int = WEIGHT_STAGING_SECONDS, controller_cores: float = CONTROLLER_CORES,
                controller_memory_gib: float = CONTROLLER_MEMORY_GIB,
                controller_allowance_seconds: int = CONTROLLER_ALLOWANCE_SECONDS,
                probe_stage_max_seconds: int = PROBE_STAGE_MAX_SECONDS,
                probe_time_headroom: float = PROBE_TIME_HEADROOM,
                probe_request_timeout_seconds: int = PROBE_REQUEST_TIMEOUT_SECONDS) -> PlanLimits:
    """Size every stage to one recorded ensemble per item and rendering (module rule ``LIMITS_RULE``).

    The calibration stage takes each arm's ``calibration_draws`` per corpus,
    the pilot the pilot's draws and every other stage the arm's draws; the full
    run covers the evaluation months of the arm's role (:func:`full_run_months`).
    A probe-arm stage's time limit is ``ceil(requests x rate x headroom)`` plus
    one client timeout, and must fit ``probe_stage_max_seconds``.
    """
    for value, cls, name in ((split, MonthSplit, "split"), (edgar, EdgarSplit, "edgar"), (arms, StudyArms, "arms"),
                             (pilot, PilotSettings, "pilot"), (manifest, ServingManifest, "manifest")):
        if not isinstance(value, cls):
            raise TypeError(f"{name} must be a {cls.__name__}; got {type(value).__name__}")
    sampling, serving = arms.arms[SAMPLING_ARM], arms.arms[SERVING_ARM]
    if serving.max_tokens > manifest.output_tokens_per_post:
        raise ValueError(f"the serving arm's max_tokens {serving.max_tokens} exceeds the manifest's "
                         f"output_tokens_per_post {manifest.output_tokens_per_post}")
    if serving.top_logprobs > manifest.top_logprobs:
        raise ValueError(f"the serving arm's top_logprobs {serving.top_logprobs} exceeds the manifest's "
                         f"top_logprobs {manifest.top_logprobs}")
    if serving.temperature != manifest.temperature:
        raise ValueError(f"the serving arm's temperature {serving.temperature} is not the manifest's "
                         f"{manifest.temperature}")
    calibration = {"macro": len(split.calibration) * len(RENDERINGS),
                   "edgar": len(edgar.calibration_memorized) + len(edgar.calibration_control)}
    negative_control = {"edgar": len(edgar.negative_control)}

    def full_run(arm: str) -> dict[str, int]:
        months = _role_months(split, arms.arms[arm])
        if not months:
            raise ValueError(f"the {arm} arm's full run needs the split's {FULL_RUN_MONTHS[arms.arms[arm].role]} "
                             "months; the split has none")
        return {"macro": len(months) * len(RENDERINGS)}

    def calibration_draws(arm: ArmPlan) -> tuple[int, dict[str, int]]:
        """The calibration stage's draws (macro) and the EDGAR override when it differs."""
        draws = arm.calibration_draws["macro"]
        edgar_draws = arm.calibration_draws["edgar"]
        return draws, ({} if edgar_draws == draws else {"edgar": edgar_draws})

    def requests_of(ensembles: Mapping[str, int], draws: int, by_group: Mapping[str, int]) -> int:
        return sum(count * by_group.get(group, draws) for group, count in ensembles.items())

    rate = _integer(sampling_seconds_per_generation, "sampling_seconds_per_generation", minimum=1)
    headroom = _headroom(probe_time_headroom)
    timeout = _integer(probe_request_timeout_seconds, "probe_request_timeout_seconds", minimum=1)

    def local(stage: str, ensembles: Mapping[str, int], draws: int,
              by_group: Mapping[str, int] | None = None) -> StageBudget:
        by_group = {} if by_group is None else by_group
        requests = requests_of(ensembles, draws, by_group)
        return StageBudget(stage=stage, ensembles=ensembles, draws=draws, extra_requests=0, max_requests=requests,
                           max_output_tokens=requests * sampling.max_tokens,
                           max_seconds=_probe_seconds(requests, rate, headroom, timeout), max_usd=None,
                           draws_by_group=by_group)

    sampling_stages = (local("calibration", calibration, *calibration_draws(sampling)),
                       local("pilot", {"macro": len(split.pilot) * len(pilot.renderings)}, pilot.draws),
                       local("full_run", full_run(SAMPLING_ARM), sampling.draws),
                       local("negative_control", negative_control, sampling.draws))
    bound = _integer(probe_stage_max_seconds, "probe_stage_max_seconds", minimum=1)
    _check_probe_bound(sampling_stages, bound)

    lifecycles = len(SERVING_STAGES) - 1
    budget = manifest.gpu_phase_seconds - lifecycles * manifest.startup_seconds
    if budget <= 0:
        raise ValueError("the manifest's GPU phase leaves no time for posts after five startups")
    seconds_per_post = -(-budget // manifest.inference_posts)
    controller_rate = (_real(controller_cores, "controller_cores") * manifest.physical_core_usd_per_second
                       + _real(controller_memory_gib, "controller_memory_gib") * manifest.memory_gib_usd_per_second)
    allowance = _integer(controller_allowance_seconds, "controller_allowance_seconds", minimum=1)
    staging_seconds = _integer(weight_staging_seconds, "weight_staging_seconds", minimum=1)
    serving_stages = [StageBudget(stage=WEIGHT_STAGING, ensembles={}, draws=0, extra_requests=0, max_requests=0,
                                  max_output_tokens=0, max_seconds=staging_seconds,
                                  max_usd=_usd_up(manifest.cpu_usd_per_second * staging_seconds))]
    for stage, ensembles, (draws, by_group), extra in (
            ("deployment", {}, (0, {}), manifest.eligibility_posts),
            ("smoke_run", {"smoke": 1}, (serving.draws, {}), 0),
            ("calibration", calibration, calibration_draws(serving), 0),
            ("full_run", full_run(SERVING_ARM), (serving.draws, {}), 0),
            ("negative_control", negative_control, (serving.draws, {}), 0)):
        requests = requests_of(ensembles, draws, by_group) + extra
        seconds = manifest.startup_seconds + requests * seconds_per_post
        usd = _usd_up(manifest.usd_per_second * (seconds + manifest.cleanup_seconds)
                      + controller_rate * (seconds + allowance))
        serving_stages.append(StageBudget(stage=stage, ensembles=ensembles, draws=draws, extra_requests=extra,
                                          max_requests=requests, max_output_tokens=requests * serving.max_tokens,
                                          max_seconds=seconds, max_usd=usd, draws_by_group=by_group))
    return PlanLimits(manifest=manifest, stages={SAMPLING_ARM: sampling_stages, SERVING_ARM: tuple(serving_stages)},
                      serving_seconds_per_post=seconds_per_post, sampling_seconds_per_generation=rate,
                      weight_staging_seconds=staging_seconds, controller_cores=controller_cores,
                      controller_memory_gib=controller_memory_gib, controller_allowance_seconds=allowance,
                      probe_stage_max_seconds=bound, probe_time_headroom=headroom,
                      probe_request_timeout_seconds=timeout)


def _role_months(split: MonthSplit, arm: ArmPlan) -> tuple[str, ...]:
    """The evaluation months the full run of an arm with this role executes."""
    return getattr(split, FULL_RUN_MONTHS[arm.role])


# ---------- 7.3: scored probability for Brier and log loss (Requirement 8.4) ----------

#: Decimals a spread or return keeps before its sign is tested, as macro_regime.is_correct.
SIGN_DECIMALS = 9
SCORED_CLIP: tuple[float, float] = (0.01, 0.99)
SCORED_OUTCOME = (
    "1 when the defensive pair beats the risk pair over the next month, "
    "round((gold_commodity + short_treasury_cash) - (world_equity + tech_sector), 9) > 0, from the realized "
    "returns of the accuracy definition (FMP adjusted closes, DTB3 as the cash leg); else 0"
)
SCORED_MAPPING: Mapping[str, str] = MappingProxyType({
    "ensemble": "share of the parsed draws whose defensive spread is positive (round(spread, 9) > 0)",
    "single": "1 when draw index 0 parsed with a positive defensive spread (round(spread, 9) > 0), else 0",
})
SCORED_ADJUSTMENT = (
    "clip p_raw to [0.01, 0.99] first; then p_adj = 0.5 + (1 - clip(p_memorized, 0, 1)) * (p_raw - 0.5) with the "
    "variant's p_memorized, the factor that scales the components; a variant without an adjustment (no "
    "adjustment, no estimate or a weak calibrator) keeps p_adj = p_raw"
)
SCORED_LOSSES: Mapping[str, str] = MappingProxyType({
    "brier": "(p_adj - outcome) ** 2",
    "log_loss": "-(outcome * ln(p_adj) + (1 - outcome) * ln(1 - p_adj)), natural logarithm",
})
SCORED_MISSING = ("a month with realized = None, or a variant without a decision (fewer parsed draws than "
                  "min_parsed, or draw index 0 unparsed), has no scored probability and is reported as missing, "
                  "never substituted")


@dataclass(frozen=True)
class ScoredProbability:
    """The forecast scored by Brier and log loss: outcome, mapping per draw mode, clip and shrink toward 0.5."""

    outcome: str = SCORED_OUTCOME
    mapping: Mapping[str, str] = field(default_factory=lambda: dict(SCORED_MAPPING))
    clip: tuple[float, float] = SCORED_CLIP
    adjustment: str = SCORED_ADJUSTMENT
    losses: Mapping[str, str] = field(default_factory=lambda: dict(SCORED_LOSSES))
    missing: str = SCORED_MISSING
    sign_decimals: int = SIGN_DECIMALS

    def __post_init__(self) -> None:
        for name in ("outcome", "adjustment", "missing"):
            _text(getattr(self, name), name)
        mapping = _text_map(self.mapping, "mapping")
        if set(mapping) != set(DRAW_MODES):
            raise ValueError(f"mapping must give the rule of each draw mode {list(DRAW_MODES)}")
        losses = _text_map(self.losses, "losses")
        if set(losses) != {"brier", "log_loss"}:
            raise ValueError("losses must define brier and log_loss")
        bounds = _sequence(self.clip, "clip")
        if len(bounds) != 2:
            raise ValueError(f"clip must be (lower, upper); got {bounds!r}")
        lower, upper = (_real(bound, "clip bound") for bound in bounds)
        if not 0.0 < lower < 0.5 < upper < 1.0:
            raise ValueError(f"clip must satisfy 0 < lower < 0.5 < upper < 1; got {bounds!r}")
        _integer(self.sign_decimals, "sign_decimals", minimum=0)
        object.__setattr__(self, "mapping", mapping)
        object.__setattr__(self, "losses", losses)
        object.__setattr__(self, "clip", (lower, upper))

    def _positive(self, value: float) -> bool:
        return round(value, self.sign_decimals) > 0

    def outcome_of(self, realized: RealizedReturns) -> int:
        """1 when the defensive pair beat the risk pair over the next month, else 0."""
        if not isinstance(realized, RealizedReturns):
            raise TypeError(f"realized must be RealizedReturns; got {type(realized).__name__}")
        return int(self._positive((realized.gold_commodity + realized.short_treasury_cash)
                                  - (realized.world_equity + realized.tech_sector)))

    def probability(self, spreads: Sequence[float], *, mode: str) -> float:
        """The clipped raw probability from the parsed draws' defensive spreads (draw order) of one draw mode."""
        mode = _choice(mode, "mode", DRAW_MODES)
        values = [_real(value, "spread") for value in _sequence(spreads, "spreads")]
        if mode == "single" and len(values) != 1:
            raise ValueError(f"a single draw has exactly one spread (draw index 0); got {len(values)}")
        if not values:
            raise ValueError("an ensemble probability needs at least one parsed draw")
        share = sum(self._positive(value) for value in values) / len(values)
        lower, upper = self.clip
        return min(max(share, lower), upper)

    def adjusted(self, p_raw: float, p_memorized: float | None) -> float:
        """Shrink a clipped probability toward 0.5 by ``1 - clip(p_memorized, 0, 1)``; None keeps it."""
        value = _real(p_raw, "p_raw")
        lower, upper = self.clip
        if not lower <= value <= upper:
            raise ValueError(f"p_raw must be a clipped probability in [{lower}, {upper}]; got {value}")
        if p_memorized is None:
            return value
        scale = 1.0 - min(max(_real(p_memorized, "p_memorized"), 0.0), 1.0)
        # A factor of 1 (estimate 0) leaves the forecast exactly as no adjustment does.
        return value if scale == 1.0 else 0.5 + scale * (value - 0.5)

    def to_dict(self) -> dict[str, Any]:
        return _section_dict(self)

    @classmethod
    def from_dict(cls, data: object) -> ScoredProbability:
        return _section_from_dict(cls, data, "scored_probability")


#: The 7.3 sections of the plan, in dependency order.
STUDY_SECTIONS: tuple[str, ...] = ("arms", "pilot", "limits", "scored_probability")


# ---------- the plan ----------

@dataclass(frozen=True)
class StudyPlan:
    """The frozen study plan; immutable, and identified by :attr:`plan_sha256`.

    ``items`` maps each study item id (7.1) to the SHA-256 of its row and prompts;
    ``excluded_models`` maps a model without a sourced cutoff to the reason;
    ``edgar_files`` maps each :data:`EDGAR_FILES` key to its manifest sha256.

    Task 7.3 appends its sections (arms with their variant sets, pilot settings,
    limits, scored probability) as keyword fields defaulting to None, each with a
    ``to_dict`` and a parser in ``_SECTION_PARSERS``. A section that is None is
    left out of the payload and the hash; a set one is serialised, gets a content
    hash and enters ``plan_sha256`` like the others. ``pilot`` needs ``arms``, and
    ``limits`` needs both; the limits must equal :func:`size_limits` of the plan.
    The runner (7.4 onward) runs a plan only with every section set
    (:func:`require_complete`).
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
    arms: StudyArms | None = None
    pilot: PilotSettings | None = None
    limits: PlanLimits | None = None
    scored_probability: ScoredProbability | None = None

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
        self._check_study_sections()

    def _check_study_sections(self) -> None:
        """The 7.3 sections: their types, their dependencies and their agreement with the 7.2 part."""
        for name, cls in (("arms", StudyArms), ("pilot", PilotSettings), ("limits", PlanLimits),
                          ("scored_probability", ScoredProbability)):
            value = getattr(self, name)
            if value is not None and not isinstance(value, cls):
                raise TypeError(f"{name} must be a {cls.__name__} or None; got {type(value).__name__}")
        if self.pilot is not None and self.arms is None:
            raise ValueError("the pilot section needs the arms section")
        if self.limits is not None and (self.arms is None or self.pilot is None):
            raise ValueError("the limits section needs the arms and pilot sections")
        if self.arms is not None:
            for arm, registry_id in self.arms.registry_ids.items():
                if registry_id in self.excluded_models:
                    raise ValueError(f"arm {arm!r} has no included model: {registry_id} is excluded "
                                     f"({self.excluded_models[registry_id]})")
                if registry_id not in self.months.models:
                    raise ValueError(f"arm {arm!r} maps to {registry_id!r}, which is not an included model of the "
                                     f"month split {list(self.months.models)}")
        if self.pilot is not None:
            _check_pilot(self.pilot, self.months, self.arms)
        if self.limits is not None:
            expected = size_limits(self.months, self.edgar, self.arms, self.pilot, self.limits.manifest,
                                   **self.limits.settings())
            differing = [f"{arm}/{budget.stage}" for arm in ARMS
                         for budget, frozen in zip(expected.stages[arm], self.limits.stages[arm], strict=True)
                         if budget != frozen]
            if expected.serving_seconds_per_post != self.limits.serving_seconds_per_post:
                differing.append("serving_seconds_per_post")
            if differing:
                raise ValueError(f"limits differ from the sizing rule of the plan's items, arms and manifest: "
                                 f"{differing}")

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
    "arms": StudyArms.from_dict,
    "pilot": PilotSettings.from_dict,
    "limits": PlanLimits.from_dict,
    "scored_probability": ScoredProbability.from_dict,
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
    evaluation = tuple(month for month in all_months if month not in drawn)
    return MonthSplit(
        models=models,
        months=all_months,
        pre_cutoff={model: tuple(item.month for item in ordered if item.pre_cutoff[model]) for model in models},
        calibration=tuple(sorted(calibration)),
        pilot=tuple(sorted(pilot)),
        evaluation=evaluation,
        probe_evaluation=_probe_evaluation(rng, evaluation, {item.month for item in ordered
                                                             if all(item.pre_cutoff.values())}),
    )


def probe_strata(evaluation: int, pre_cutoff: int, months: int = PROBE_EVALUATION_MONTHS) -> tuple[int, int]:
    """Probe months on or before every model's cutoff and after it, in the evaluation set's proportion.

    ``round(months x pre_cutoff / evaluation)`` with a half rounded up, in exact
    integer arithmetic; the post-cutoff stratum gets the rest (68 and 22
    evaluation months give 15 and 5).
    """
    _integer(evaluation, "evaluation months", minimum=1)
    _integer(pre_cutoff, "pre-cutoff evaluation months", minimum=0)
    _integer(months, "probe evaluation months", minimum=1)
    if pre_cutoff > evaluation:
        raise ValueError(f"pre-cutoff evaluation months {pre_cutoff} exceed the {evaluation} evaluation months")
    before = (2 * months * pre_cutoff + evaluation) // (2 * evaluation)
    return before, months - before


def _probe_evaluation(rng: np.random.Generator, evaluation: Sequence[str], pre_cutoff: set[str]) -> tuple[str, ...]:
    """The probe arm's evaluation subset, drawn after the pilot: the pre-cutoff stratum, then the post-cutoff one."""
    before = [month for month in evaluation if month in pre_cutoff]
    after = [month for month in evaluation if month not in pre_cutoff]
    counts = probe_strata(len(evaluation), len(before))
    probe: list[str] = []
    for pool, count, label in ((before, counts[0], "the pre-cutoff probe stratum"),
                               (after, counts[1], "the post-cutoff probe stratum")):
        if count:
            probe += _draw(rng, pool, count, label)
    return tuple(sorted(probe))


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
    arms: StudyArms | None = None,
    serving_manifest: ServingManifest | None = None,
    pilot: PilotSettings | None = None,
    scored_probability: ScoredProbability | None = None,
) -> StudyPlan:
    """Freeze the plan before any study request; ``created_at`` must be timezone-aware.

    Without ``arms`` and ``serving_manifest`` this is the 7.2 part only. With both
    it is the complete plan: the pilot defaults to :meth:`PilotSettings.for_split`,
    the scored probability to :class:`ScoredProbability`, and the limits are
    :func:`size_limits` of the split, the EDGAR sets, the arms, the pilot and the
    manifest.
    """
    if not isinstance(created_at, datetime) or created_at.utcoffset() is None:
        raise ValueError("created_at must be a timezone-aware datetime")
    split = split_months(months, seed=seed)
    edgar = split_edgar(edgar_rows)
    sections: dict[str, Any] = {}
    if any(value is not None for value in (arms, serving_manifest, pilot, scored_probability)):
        if arms is None:
            raise ValueError("freeze_plan needs the arms to freeze the 7.3 sections")
        if serving_manifest is None:
            raise ValueError("freeze_plan needs the serving manifest to size the serving arm's limits")
        if not isinstance(arms, StudyArms):
            raise TypeError(f"arms must be a StudyArms; got {type(arms).__name__}")
        pilot = PilotSettings.for_split(split, arms) if pilot is None else pilot
        sections = {"arms": arms, "pilot": pilot,
                    "limits": size_limits(split, edgar, arms, pilot, serving_manifest),
                    "scored_probability": ScoredProbability() if scored_probability is None else scored_probability}
    return StudyPlan(
        created_at=created_at.astimezone(UTC).isoformat(timespec="seconds"),
        seed=seed,
        items=items,
        excluded_models={} if excluded_models is None else excluded_models,
        months=split,
        edgar=edgar,
        edgar_files=edgar_files,
        accuracy=AccuracyDefinition(),
        analysis=AnalysisSettings() if analysis is None else analysis,
        **sections,
    )


def require_complete(plan: StudyPlan) -> StudyPlan:
    """Refuse a plan without every 7.3 section; the runner (7.4 onward) runs complete plans only."""
    if not isinstance(plan, StudyPlan):
        raise TypeError(f"plan must be a StudyPlan; got {type(plan).__name__}")
    missing = [name for name in STUDY_SECTIONS if getattr(plan, name) is None]
    if missing:
        raise ValueError(f"plan {plan.plan_sha256} lacks the sections {missing}; freeze it with "
                         f"scripts/finance_sim/build_plan.py")
    return plan


def full_run_months(plan: StudyPlan, arm: str) -> tuple[str, ...]:
    """The evaluation months the full run of ``arm`` executes, by its role.

    The primary (serving) arm runs every evaluation month; the probe (sampling)
    arm its frozen ``probe_evaluation`` months, a subset of them.
    """
    if not isinstance(plan, StudyPlan):
        raise TypeError(f"plan must be a StudyPlan; got {type(plan).__name__}")
    if plan.arms is None:
        raise ValueError(f"plan {plan.plan_sha256} lacks the sections ['arms']; the full-run months follow the arm's "
                         "role")
    return _role_months(plan.months, plan.arms.arms[_choice(arm, "arm", ARMS)])


def evaluation_month_counts(plan: StudyPlan, arm: str) -> tuple[int, int]:
    """Pre-cutoff and post-cutoff months among the full-run months of ``arm``, by its model's cutoff.

    The primary (serving) arm's are every evaluation month's (68 and 22 for Gemma
    3 4B), the counts of its achieved-power report; the probe (sampling) arm's
    are its probe evaluation months' (15 and 5), which the pilot freezes for the
    power gate.
    """
    full_run_months(plan, arm)  # the arm and the arms section are checked there
    return _arm_counts(plan.months, plan.arms, arm)


def stage_limits_files(plan: StudyPlan) -> tuple[tuple[str, bytes], ...]:
    """The serving arm's StageLimits files (deploy/modal/stage_limits.py) in execution order, as exact bytes.

    Each carries the plan's ``plan_sha256``; ``build_plan.py`` writes these bytes
    unchanged, so their SHA-256 values, and the ``stage_limits_sha256`` over them,
    follow from the plan alone.
    """
    require_complete(plan)
    files = []
    for budget in plan.limits.stages[SERVING_ARM]:
        document = {"plan_sha256": plan.plan_sha256, "stage": budget.stage, "max_requests": budget.max_requests,
                    "max_output_tokens": budget.max_output_tokens, "max_seconds": budget.max_seconds,
                    "max_usd": budget.max_usd}
        files.append((budget.stage, (json.dumps(document, indent=2, sort_keys=True) + "\n").encode("utf-8")))
    return tuple(files)


def serving_limits_record(plan: StudyPlan) -> dict[str, Any]:
    """The serving arm's limits with the hashes the approval checks: plan, manifest and stage limits.

    ``stage_limits_sha256`` is the SHA-256 of the canonical JSON list of
    ``[stage, file sha256]`` pairs in execution order, as
    ``scripts/finance_sim/limits.py`` computes it from the files, so the
    presentation, the approval record and the plan agree. These hashes cannot be
    inside the plan's own content (they depend on ``plan_sha256``); the build
    receipt records this dictionary.
    """
    files = stage_limits_files(plan)
    stages = [{"stage": budget.stage, "sha256": hashlib.sha256(data).hexdigest(),
               "max_requests": budget.max_requests, "max_output_tokens": budget.max_output_tokens,
               "max_seconds": budget.max_seconds, "max_usd": budget.max_usd}
              for budget, (_, data) in zip(plan.limits.stages[SERVING_ARM], files, strict=True)]
    return {"plan_sha256": plan.plan_sha256, "manifest": plan.limits.manifest.path,
            "manifest_sha256": plan.limits.manifest.sha256,
            "stage_limits_sha256": sha256_of([[stage["stage"], stage["sha256"]] for stage in stages]),
            "stages": stages, "max_usd_total": plan.limits.serving_usd_total}


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
