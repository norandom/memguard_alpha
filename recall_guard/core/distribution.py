"""Recorded draws and the distribution summary of an ensemble.

An ensemble answers two questions at once: what the model decides, and how
firmly. The consensus answers the first; this module answers the second, from
the recorded draws alone. Every number here is a pure function of the
:class:`RecordedDraw` records and the :class:`~recall_guard.core.decision.DecisionDefinition`,
so a stored ensemble can be summarised again later without a new request.

The summary is order-free, with two deliberate exceptions. The first-draw
position asks where draw 0 sits in the distribution, because a single-draw
caller would have seen exactly that draw. The lag-one dependence runs over the
parsed draws in ``draw_index`` order, with failed draws skipped, because
dependence between neighbouring draws is the thing an interval silently assumes
away. Both read ``draw_index``, never arrival order.

Agreement and the separated-cluster verdicts are not recomputed here. They come
from :func:`~recall_guard.core.ensemble.reduce_draws`, so the summary and the
consensus can never disagree about whether a component is split.
"""

from __future__ import annotations

import math
import numbers
import statistics
from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import dataclass, fields
from types import MappingProxyType
from typing import Any

import numpy as np

from recall_guard.core.consensus import Tail
from recall_guard.core.decision import DecisionDefinition
from recall_guard.core.ensemble import EnsembleSpec, reduce_draws

#: Confidence and tail of the decision-agreement Wilson interval: the package
#: convention, so the lower bound is the 95 percent lower bound and the pair
#: covers 90 percent two-sided.
AGREEMENT_CONFIDENCE = 0.95
AGREEMENT_TAIL = Tail.ONE_SIDED

#: Quantile levels, in percent, reported for every distribution.
QUANTILE_LEVELS: tuple[int, ...] = (5, 25, 50, 75, 95)

#: Stated on every summary next to its intervals.
INDEPENDENCE_NOTE = (
    "Intervals assume independent draws; the lag-one dependence between draws is reported, not corrected."
)

#: Name of the distribution of the definition's summary value.
SUMMARY_VALUE_NAME = "summary"


def _is_finite_real(value: object) -> bool:
    # bool is an int subclass; a True estimate is a recording bug, not 1.0.
    return isinstance(value, numbers.Real) and not isinstance(value, bool) and math.isfinite(value)


def _exact_fields(data: object, names: Sequence[str], what: str) -> Mapping[str, Any]:
    """``data`` as a mapping holding exactly ``names``; anything else is refused, naming the difference."""
    if not isinstance(data, Mapping):
        raise TypeError(f"{what} must be a mapping; got {type(data).__name__}")
    missing = [name for name in names if name not in data]
    unexpected = sorted(str(key) for key in data if key not in names)
    if missing or unexpected:
        raise ValueError(f"{what} fields differ: missing {missing}, unexpected {unexpected}")
    return data


def _finite_field(value: object, name: str, *, optional: bool = False) -> float | None:
    if value is None and optional:
        return None
    if not _is_finite_real(value):
        raise ValueError(f"{name} must be a finite number{' or None' if optional else ''}; got {value!r}")
    return float(value)


def _count_field(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer; got {value!r}")
    return value


def _quantiles_to_dict(quantiles: Mapping[int, float]) -> dict[str, float]:
    """JSON object keys are strings, so the levels are written as ``"5"``, ``"25"`` and so on."""
    return {str(level): float(quantiles[level]) for level in QUANTILE_LEVELS}


def _quantiles_from_dict(value: object, name: str) -> dict[int, float]:
    expected = [str(level) for level in QUANTILE_LEVELS]
    keys = sorted(str(key) for key in value) if isinstance(value, Mapping) else None
    if keys is None or len(value) != len(expected) or set(keys) != set(expected):
        raise ValueError(f"{name} must hold exactly the quantile levels {expected}; got {keys if keys else value!r}")
    by_level = {str(key): number for key, number in value.items()}
    return {level: _finite_field(by_level[str(level)], f"{name}[{level}]") for level in QUANTILE_LEVELS}


@dataclass(frozen=True)
class RecordedDraw:
    """One draw as recorded by the guarded-decision API and the study runner.

    ``response_sha256`` stands in for the reply text, which stays in the run
    directory. ``parsed`` is ``None`` exactly when the reply did not parse, and
    then ``failure_code`` says why. Item and variant keys are not fields: one
    ensemble per item and rendering feeds every variant, so they belong to the
    study recording around this record.

    ``parsed`` is held as a read-only mapping, a copy taken at construction, so
    the frozen record cannot change afterwards. :meth:`to_dict` and
    :meth:`from_dict` are the supported serialisation path;
    ``dataclasses.asdict``, ``copy.deepcopy`` and pickle are not supported.
    """

    draw_index: int
    temperature: float | None
    seed: int | None
    response_sha256: str
    parsed: Mapping[str, float] | None
    failure_code: str | None
    estimate: float | None
    evidence_ref: str | None

    def __post_init__(self) -> None:
        if not isinstance(self.draw_index, int) or isinstance(self.draw_index, bool) or self.draw_index < 0:
            raise ValueError(f"draw_index must be a non-negative integer; got {self.draw_index!r}")
        if not isinstance(self.response_sha256, str) or not self.response_sha256:
            raise ValueError(f"response_sha256 must be a non-empty string; got {self.response_sha256!r}")
        if (self.parsed is None) != (self.failure_code is not None):
            raise ValueError(
                "failure_code must be set exactly when parsed is None; "
                f"got parsed={self.parsed!r}, failure_code={self.failure_code!r}"
            )
        if self.failure_code is not None and (not isinstance(self.failure_code, str) or not self.failure_code):
            raise ValueError(f"failure_code must be a non-empty string when set; got {self.failure_code!r}")
        if self.seed is not None and (not isinstance(self.seed, int) or isinstance(self.seed, bool)):
            raise ValueError(f"seed must be an integer when set; got {self.seed!r}")
        if self.parsed is not None and not all(_is_finite_real(v) for v in self.parsed.values()):
            raise ValueError(f"parsed values must be finite numbers; got {dict(self.parsed)!r}")
        if self.parsed is not None:
            # A read-only copy: the record is frozen, and so is the vector it holds.
            object.__setattr__(self, "parsed", MappingProxyType(dict(self.parsed)))
        if self.estimate is not None and not _is_finite_real(self.estimate):
            raise ValueError(f"estimate must be a finite number when set; got {self.estimate!r}")
        if self.temperature is not None and not _is_finite_real(self.temperature):
            raise ValueError(f"temperature must be a finite number when set; got {self.temperature!r}")

    def to_dict(self) -> dict[str, Any]:
        """JSON-compatible dict, one line of the study recording."""
        data = {f.name: getattr(self, f.name) for f in fields(self)}
        if self.parsed is not None:
            data["parsed"] = dict(self.parsed)
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> RecordedDraw:
        """Inverse of :meth:`to_dict`; missing or unexpected keys are refused."""
        names = [f.name for f in fields(cls)]
        missing = [name for name in names if name not in data]
        if missing:
            raise ValueError(f"recorded draw is missing fields {missing}")
        unexpected = sorted(str(key) for key in data if key not in names)
        if unexpected:
            raise ValueError(f"recorded draw has unexpected fields {unexpected}")
        values = {name: data[name] for name in names}
        if values["parsed"] is not None:
            values["parsed"] = dict(values["parsed"])
        return cls(**values)


@dataclass(frozen=True)
class ComponentDistribution:
    """Distribution of one component, or of the summary value, over parsed draws.

    ``quantiles`` is keyed by :data:`QUANTILE_LEVELS`. ``unresolved`` is the
    separated-cluster verdict of the ensemble reduction: ``True`` when a split
    was found, ``False`` when the check ran and found none, and ``None`` when it
    could not run (no grid, or too few parsed draws). ``None`` is never a
    resolved component. A split that the density guard in
    :mod:`~recall_guard.core.consensus` skips reads as ``False``, as
    :func:`~recall_guard.core.ensemble.reduce_draws` reports it. The summary
    value has no grid of its own, so it is ``True`` when any component is split
    and ``None`` otherwise.
    """

    name: str
    quantiles: Mapping[int, float]
    sd: float
    iqr: float
    sign_agreement: float
    unresolved: bool | None
    first_draw_rank: float | None

    def to_dict(self) -> dict[str, Any]:
        """JSON-compatible dict with string quantile levels; the inverse of :meth:`from_dict`."""
        return {"name": self.name, "quantiles": _quantiles_to_dict(self.quantiles), "sd": self.sd, "iqr": self.iqr,
                "sign_agreement": self.sign_agreement, "unresolved": self.unresolved,
                "first_draw_rank": self.first_draw_rank}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ComponentDistribution:
        """Load the exact field set written by :meth:`to_dict`; quantile levels become integers again."""
        values = _exact_fields(data, [f.name for f in fields(cls)], "component distribution")
        name = values["name"]
        if not isinstance(name, str) or not name:
            raise ValueError(f"name must be a non-empty string; got {name!r}")
        unresolved = values["unresolved"]
        if unresolved is not None and not isinstance(unresolved, bool):
            raise ValueError(f"unresolved must be true, false or None; got {unresolved!r}")
        return cls(name=name, quantiles=_quantiles_from_dict(values["quantiles"], f"{name}.quantiles"),
                   sd=_finite_field(values["sd"], f"{name}.sd"), iqr=_finite_field(values["iqr"], f"{name}.iqr"),
                   sign_agreement=_finite_field(values["sign_agreement"], f"{name}.sign_agreement"),
                   unresolved=unresolved,
                   first_draw_rank=_finite_field(values["first_draw_rank"], f"{name}.first_draw_rank",
                                                 optional=True))


@dataclass(frozen=True)
class DistributionSummary:
    """Distribution of an ensemble's parsed draws and per-draw estimates.

    ``decision_agreement_ci`` is a Wilson interval, one-sided at 95 percent
    (:data:`AGREEMENT_CONFIDENCE`, :data:`AGREEMENT_TAIL`): its lower bound is
    the 95 percent lower bound, and the pair covers 90 percent two-sided. Like
    every interval here it assumes independent draws; see ``independence_note``.
    """

    n_requested: int
    n_parsed: int
    n_failed: int
    components: tuple[ComponentDistribution, ...]
    summary_value: ComponentDistribution | None
    decision_agreement: float | None
    decision_agreement_ci: tuple[float, float] | None
    estimate_quantiles: Mapping[int, float] | None
    estimate_sd: float | None
    distinct_answers: int
    lag1_dependence: float | None
    independence_note: str

    def to_dict(self) -> dict[str, Any]:
        """JSON-compatible dict, as the study recording stores it; the inverse of :meth:`from_dict`."""
        ci = self.decision_agreement_ci
        return {
            "n_requested": self.n_requested, "n_parsed": self.n_parsed, "n_failed": self.n_failed,
            "components": [component.to_dict() for component in self.components],
            "summary_value": None if self.summary_value is None else self.summary_value.to_dict(),
            "decision_agreement": self.decision_agreement,
            "decision_agreement_ci": None if ci is None else [ci[0], ci[1]],
            "estimate_quantiles": None if self.estimate_quantiles is None else _quantiles_to_dict(
                self.estimate_quantiles),
            "estimate_sd": self.estimate_sd, "distinct_answers": self.distinct_answers,
            "lag1_dependence": self.lag1_dependence, "independence_note": self.independence_note,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> DistributionSummary:
        """Load the exact field set written by :meth:`to_dict`; unknown, missing or invalid fields are refused."""
        values = _exact_fields(data, [f.name for f in fields(cls)], "distribution summary")
        components = values["components"]
        if not isinstance(components, list | tuple) or not components:
            raise ValueError("components must be a non-empty list of component distributions")
        ci = values["decision_agreement_ci"]
        if ci is not None:
            if not isinstance(ci, list | tuple) or len(ci) != 2:
                raise ValueError(f"decision_agreement_ci must be a pair of bounds or None; got {ci!r}")
            ci = (_finite_field(ci[0], "decision_agreement_ci[0]"), _finite_field(ci[1], "decision_agreement_ci[1]"))
        quantiles = values["estimate_quantiles"]
        note = values["independence_note"]
        if not isinstance(note, str) or not note:
            raise ValueError("independence_note must be a non-empty string")
        return cls(
            n_requested=_count_field(values["n_requested"], "n_requested"),
            n_parsed=_count_field(values["n_parsed"], "n_parsed"),
            n_failed=_count_field(values["n_failed"], "n_failed"),
            components=tuple(ComponentDistribution.from_dict(component) for component in components),
            summary_value=None if values["summary_value"] is None else ComponentDistribution.from_dict(
                values["summary_value"]),
            decision_agreement=_finite_field(values["decision_agreement"], "decision_agreement", optional=True),
            decision_agreement_ci=ci,
            estimate_quantiles=None if quantiles is None else _quantiles_from_dict(quantiles, "estimate_quantiles"),
            estimate_sd=_finite_field(values["estimate_sd"], "estimate_sd", optional=True),
            distinct_answers=_count_field(values["distinct_answers"], "distinct_answers"),
            lag1_dependence=_finite_field(values["lag1_dependence"], "lag1_dependence", optional=True),
            independence_note=note,
        )


@dataclass(frozen=True)
class _DrawView:
    """Adapter for :func:`reduce_draws`, which reads only ``content``.

    The response hash is the content: it is what the canonical order and the
    draw-set digest need, and the reply text is not part of the record.
    """

    content: str
    vector: Mapping[str, float]


def _quantiles(values: np.ndarray) -> dict[int, float]:
    # numpy's default method is linear interpolation between order statistics.
    levels = np.quantile(values, [level / 100 for level in QUANTILE_LEVELS])
    return {level: float(q) for level, q in zip(QUANTILE_LEVELS, levels, strict=True)}


def _sd(values: np.ndarray) -> float:
    # statistics.stdev sums exactly, so identical draws give exactly 0.0; np.std
    # leaves a rounding residue there that would read as spread.
    return statistics.stdev(values.tolist()) if values.size > 1 else 0.0


def _distribution(
    name: str, values: np.ndarray, first: float | None, unresolved: bool | None
) -> ComponentDistribution:
    """``values`` holds one scalar per parsed draw; ``first`` is draw 0's, if it parsed."""
    quantiles = _quantiles(values)
    median_sign = np.sign(quantiles[50])
    rank = None
    if first is not None:
        # Mid-rank: share strictly below plus half the share tied, draw 0 included.
        rank = float((np.count_nonzero(values < first) + 0.5 * np.count_nonzero(values == first)) / values.size)
    return ComponentDistribution(
        name=name,
        quantiles=quantiles,
        sd=_sd(values),
        iqr=quantiles[75] - quantiles[25],
        sign_agreement=float(np.mean(np.sign(values) == median_sign)),
        unresolved=unresolved,
        first_draw_rank=rank,
    )


def _lag1(series: np.ndarray) -> float | None:
    """Pearson correlation of consecutive pairs; ``None`` when it is undefined."""
    if series.size < 3:
        return None
    head, tail = series[:-1], series[1:]
    # Tested exactly: a rounded mean of equal values leaves tiny deviations,
    # whose correlation would be noise reported as dependence.
    if np.all(head == head[0]) or np.all(tail == tail[0]):
        return None
    head_dev, tail_dev = head - head.mean(), tail - tail.mean()
    denominator = math.sqrt(float(np.sum(head_dev**2)) * float(np.sum(tail_dev**2)))
    return float(np.sum(head_dev * tail_dev)) / denominator


def _unresolved(views: Sequence[_DrawView], definition: DecisionDefinition) -> dict[str, bool | None]:
    """Separated-cluster verdict per component, from the ensemble reduction.

    Read from ``component_verdicts`` rather than ``multimodal``: the flagged
    list cannot tell "checked, no split" from "not checked", and only the first
    is a resolved component. :class:`EnsembleSpec` holds one grid for every
    component, while a definition declares one per component, so components are
    reduced in groups that share a grid.
    """
    by_grid: dict[float | None, list[str]] = {}
    for spec in definition.components:
        by_grid.setdefault(spec.grid, []).append(spec.name)
    verdicts: dict[str, bool | None] = {}
    for grid, names in by_grid.items():
        result = reduce_draws(
            views,
            EnsembleSpec(draws=len(views), min_parsed=1, grid=grid),
            decide=lambda _view: None,
            components=lambda view, names=names: {name: view.vector[name] for name in names},
        )
        for name, verdict in result.component_verdicts:
            verdicts[name] = None if verdict is None else verdict.separated
    return {spec.name: verdicts[spec.name] for spec in definition.components}


def _agreement(
    views: Sequence[_DrawView], decide: Callable[[Mapping[str, float]], Hashable]
) -> tuple[float, tuple[float, float] | None]:
    spec = EnsembleSpec(draws=len(views), min_parsed=1, confidence=AGREEMENT_CONFIDENCE, tail=AGREEMENT_TAIL)
    result = reduce_draws(views, spec, decide=lambda view: decide(view.vector))
    return result.agreement, result.agreement_ci


def summarize_distribution(draws: Sequence[RecordedDraw], definition: DecisionDefinition) -> DistributionSummary:
    """Summarise recorded draws without new requests.

    Statistics cover the parsed draws only; failed draws count towards
    ``n_requested`` and ``n_failed`` and nothing else. Estimate quantiles use
    the parsed draws that carry an estimate and are ``None`` when none does.

    Raises ``ValueError`` for an empty draw set, a set in which no draw parsed,
    a repeated ``draw_index``, or a parsed vector whose names differ from the
    definition's components.
    """
    if not draws:
        raise ValueError("cannot summarise an empty draw set")
    ordered = sorted(draws, key=lambda d: d.draw_index)
    for previous, current in zip(ordered, ordered[1:], strict=False):
        if previous.draw_index == current.draw_index:
            raise ValueError(f"duplicate draw_index {current.draw_index}")
    parsed = [d for d in ordered if d.parsed is not None]
    if not parsed:
        raise ValueError(f"no draw parsed out of {len(ordered)}; nothing to summarise")

    names = [spec.name for spec in definition.components]
    vectors: list[dict[str, float]] = []
    for draw in parsed:
        recorded = draw.parsed or {}
        if set(recorded) != set(names):
            raise ValueError(
                f"draw {draw.draw_index}: parsed components {sorted(recorded)} differ from "
                f"definition {definition.name!r} components {sorted(names)}"
            )
        vectors.append({name: float(recorded[name]) for name in names})

    # Rows are parsed draws in draw_index order; columns are components in declared order.
    matrix = np.array([[vector[name] for name in names] for vector in vectors], dtype=float)
    first_row = 0 if parsed[0].draw_index == 0 else None

    views = [_DrawView(content=d.response_sha256, vector=v) for d, v in zip(parsed, vectors, strict=True)]
    unresolved = _unresolved(views, definition)
    components = tuple(
        _distribution(
            name,
            matrix[:, column],
            None if first_row is None else float(matrix[first_row, column]),
            unresolved[name],
        )
        for column, name in enumerate(names)
    )

    summary_value = None
    lag_series = matrix[:, 0]
    if definition.summary is not None:
        series = np.array([float(definition.summary(v)) for v in vectors], dtype=float)
        if not np.all(np.isfinite(series)):
            raise ValueError(f"definition {definition.name!r}: summary returned a non-finite value")
        # The summary value has no grid, so the split check cannot run on it
        # directly; a split component makes it unresolved, and nothing else can
        # make it resolved.
        summary_unresolved = True if any(c.unresolved is True for c in components) else None
        summary_value = _distribution(
            SUMMARY_VALUE_NAME, series, None if first_row is None else float(series[first_row]), summary_unresolved
        )
        lag_series = series

    agreement = agreement_ci = None
    if definition.decide is not None:
        agreement, agreement_ci = _agreement(views, definition.decide)

    estimates = np.array([d.estimate for d in parsed if d.estimate is not None], dtype=float)
    estimate_quantiles = _quantiles(estimates) if estimates.size else None
    estimate_sd = _sd(estimates) if estimates.size else None

    return DistributionSummary(
        n_requested=len(ordered),
        n_parsed=len(parsed),
        n_failed=len(ordered) - len(parsed),
        components=components,
        summary_value=summary_value,
        decision_agreement=agreement,
        decision_agreement_ci=agreement_ci,
        estimate_quantiles=estimate_quantiles,
        estimate_sd=estimate_sd,
        distinct_answers=len({tuple(row) for row in matrix.tolist()}),
        lag1_dependence=_lag1(lag_series),
        independence_note=INDEPENDENCE_NOTE,
    )


__all__ = [
    "AGREEMENT_CONFIDENCE",
    "AGREEMENT_TAIL",
    "INDEPENDENCE_NOTE",
    "QUANTILE_LEVELS",
    "SUMMARY_VALUE_NAME",
    "ComponentDistribution",
    "DistributionSummary",
    "RecordedDraw",
    "summarize_distribution",
]
