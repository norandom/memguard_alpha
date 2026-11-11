"""Convergence figure F4 of the worked example (task 8.3; Requirements 10.5 and 10.6).

No request is sent and nothing is recomputed: every number comes from the
stored convergence tables of task 7.12
(``runs/finance-sim/analysis/convergence-<arm>-<stage>.json``,
:func:`~scripts.finance_sim.convergence.table_path`), one per analysed stage
(``full_run``, the positive control, and ``negative_control``) of the arm that
records full-distribution evidence (the probe arm).

Form (design: Figures, F4): lines, small multiples by feature (``loss``,
``min_k``, ``min_k_pp``, ``zlib_ratio`` and the estimate) and a separate mass
panel; x is k on a log scale, y the absolute difference to the
full-distribution value (the ``features`` rows of evidence ``host_top_k``:
mean and maximum over the compared draws) or the probability mass the first k
alternatives cover (the ``mass`` rows: mean and minimum). With two stages that
is at most four direct-labelled lines per panel; they need palette slot 4,
whose light step is below 3:1 contrast, so direct labels, marker shape and
line style carry identity there, and the table carries every value. The
served row (evidence ``served_top_k``) uses the generation host's own top-k
through the top-k calibrator, a different calibrator, so it is a separate open
marker at the served k, never joined to the lines. A k above the kept count is
not drawn and is named in the caption. The caption states that values compare
only within one model and one set of weights, and the figure refuses tables of
different weights.

Outputs, style and the write-once rule are those of
:mod:`scripts.finance_sim.figures`: ``convergence-light.svg``,
``convergence-dark.svg``, ``convergence.csv`` and ``convergence.md``; derived
statistics only. Matplotlib is imported inside the functions that draw.
"""

from __future__ import annotations

import argparse
import math
import sys
import textwrap
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from scripts.finance_sim.calibrate import MACRO
from scripts.finance_sim.convergence import (
    CONVERGENCE_DIR,
    CONVERGENCE_STAGES,
    EVIDENCE_KINDS,
    FEATURES,
    HOST_TOP_K,
    SERVED_TOP_K,
    TABLE_SCHEMA,
    table_path,
)
from scripts.finance_sim.figures import (
    _HEX64,
    _RC,
    _TITLE_LINE_IN,
    _WIDTH_IN,
    CONVERGENCE,
    DERIVED_ONLY,
    DRAWN,
    EDGAR_SOURCE,
    EXIT_REFUSED,
    EXIT_WRITTEN,
    FIGURES_DIR,
    FMP_STUDY_SOURCE,
    FRED_SOURCE,
    MISSING,
    ROLE_PURPOSE,
    FigureError,
    FigureTable,
    _around,
    _count,
    _flat,
    _grouped,
    _number,
    _obstacles,
    _place_text,
    _read_json,
    _real,
    _style_axes,
    _text,
    _theme_rc,
    figure_texts,
    write_outputs,
)
from scripts.finance_sim.limits import ARMS, SAMPLING_ARM
from scripts.finance_sim.plan import FULL_DISTRIBUTION, StudyPlan, load_plan

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from recall_guard.harness.plots import FigureTheme

#: The arm with full-distribution evidence (7.12 refuses the serving arm).
DEFAULT_CONVERGENCE_ARM = SAMPLING_ARM
STAGE_NAMES: Mapping[str, str] = {"full_run": "full run", "negative_control": "negative control"}
FEATURE_NAMES: Mapping[str, str] = {"loss": "Loss", "min_k": "Min-K%", "min_k_pp": "Min-K%++",
                                    "zlib_ratio": "zlib ratio", "estimate": "p_memorized estimate"}
#: The two statistics drawn per stage: the line style tells them apart (solid, dashed).
FEATURE_STATISTICS: tuple[tuple[str, str], ...] = (("mean_abs_diff", "mean"), ("max_abs_diff", "max"))
MASS_STATISTICS: tuple[tuple[str, str], ...] = (("mean_mass", "mean"), ("min_mass", "min"))
#: Short stage names of the direct labels, by the prompts each stage scores (the caption spells them out).
STAGE_SHORT: Mapping[str, str] = {"full_run": "macro", "negative_control": "EDGAR"}
MASS = "mass"
CONVERGENCE_COLUMNS: tuple[str, ...] = (
    "arm", "role", "stage", "kind", "quantity", "evidence", "k", "n", "mean_abs_diff", "max_abs_diff",
    "share_within", "mean_mass", "min_mass", "drawn", "missing")
#: At most four lines per panel: two stages times two statistics, palette slots 1 to 4 in this order.
MAX_LINES = 4
_PANEL_IN, _FRAME_IN = 2.05, 1.15
_MARKER_PT, _SERVED_PT, _LINE_PT = 5.0, 7.5, 1.4
_LABEL_PT = 6.5
_BOX_PAD_PT, _LABEL_CHARS = 0.8, 26
#: A label's surface box keeps this much space from every marker as drawn (its path at the marker size: a diamond
#: reaches 0.71 of the size from its centre, a circle 0.5) and from every other label's box.
_CLEARANCE_PT = 3.0
#: The served label's distance from its marker's centre, nearest ring first.
_SERVED_RINGS_PT = (6.0, 10.0, 14.0, 24.0)
#: The whole-vocabulary label sits this far above (or below) its line: clear of a marker centred on the line.
_REFERENCE_OFFSET_PT = 10.0
#: The share of the mass axis kept above the whole-vocabulary line: room for the labels placed there.
_MASS_HEADROOM = 0.25
_TITLE_CHARS = 52


# ---------- plain data ----------

@dataclass(frozen=True)
class AgreementRow:
    """One ``features`` row of a stored convergence table: one evidence, k and feature."""

    evidence: str
    k: int
    feature: str
    n: int
    mean_abs_diff: float | None
    max_abs_diff: float | None
    share_within: float | None
    missing: str
    missing_reason: str | None


@dataclass(frozen=True)
class MassRow:
    """One ``mass`` row of a stored convergence table: one evidence and k."""

    evidence: str
    k: int
    n_draws: int
    mean_mass: float | None
    min_mass: float | None
    missing: str
    missing_reason: str | None


@dataclass(frozen=True)
class ConvergenceStage:
    """One stage's stored convergence table, read and checked, or the reason it is missing (``status``)."""

    plan_sha256: str
    arm: str
    stage: str
    status: str
    reason: str | None = None
    kind: str | None = None
    model: str | None = None
    model_digest: str | None = None
    n_vocab: int | None = None
    kept: int | None = None
    ks: tuple[int, ...] = ()
    served_k: int | None = None
    tolerance: float | None = None
    comparability: str | None = None
    scope: str | None = None
    estimator: str | None = None
    n_items: int = 0
    n_missing_items: int = 0
    n_draws: int = 0
    n_missing_draws: int = 0
    full_mean_mass: float | None = None
    full_min_mass: float | None = None
    features: tuple[AgreementRow, ...] = ()
    mass: tuple[MassRow, ...] = ()

    @property
    def name(self) -> str:
        return STAGE_NAMES.get(self.stage, self.stage)

    def agreement(self, evidence: str, feature: str) -> tuple[AgreementRow, ...]:
        return tuple(row for row in self.features if (row.evidence, row.feature) == (evidence, feature))

    def coverage(self, evidence: str) -> tuple[MassRow, ...]:
        return tuple(row for row in self.mass if row.evidence == evidence)

    @property
    def above_kept(self) -> tuple[int, ...]:
        """The host k values the table reports missing because they exceed the kept alternatives."""
        return tuple(row.k for row in self.coverage(HOST_TOP_K) if row.missing_reason is not None
                     and self.kept is not None and row.k > self.kept)


def _missing_text(counts: object) -> str:
    if not isinstance(counts, Mapping):
        raise FigureError("a row's missing counts must be a mapping")
    return "; ".join(f"{_flat(reason)}: {_count(count, 'missing count')}" for reason, count in counts.items())


def _agreement_row(entry: Mapping[str, Any]) -> AgreementRow:
    evidence, feature = entry["evidence"], entry["feature"]
    if evidence not in EVIDENCE_KINDS or feature not in FEATURES:
        raise FigureError(f"unknown convergence row {evidence!r}/{feature!r}")
    reason = entry["missing_reason"]
    return AgreementRow(evidence=evidence, k=_count(entry["k"], "k"), feature=feature, n=_count(entry["n"], "n"),
                        mean_abs_diff=_real(entry["mean_abs_diff"], "mean_abs_diff"),
                        max_abs_diff=_real(entry["max_abs_diff"], "max_abs_diff"),
                        share_within=_real(entry["share_within"], "share_within"),
                        missing=_missing_text(entry["missing"]),
                        missing_reason=None if reason is None else _flat(_text(reason, "missing_reason")))


def _mass_row(entry: Mapping[str, Any]) -> MassRow:
    if entry["evidence"] not in EVIDENCE_KINDS:
        raise FigureError(f"unknown convergence evidence {entry['evidence']!r}")
    reason = entry["missing_reason"]
    return MassRow(evidence=entry["evidence"], k=_count(entry["k"], "k"), n_draws=_count(entry["n_draws"], "n_draws"),
                   mean_mass=_real(entry["mean_mass"], "mean_mass"), min_mass=_real(entry["min_mass"], "min_mass"),
                   missing=_missing_text(entry["missing"]),
                   missing_reason=None if reason is None else _flat(_text(reason, "missing_reason")))


def convergence_stage(data: Mapping[str, Any], *, plan_sha256: str | None = None) -> ConvergenceStage:
    """One stored 7.12 convergence table (its JSON payload), read and checked, not recomputed.

    Refuses another schema, an arm or stage outside the analysis, another plan
    than ``plan_sha256`` (when given) and malformed rows.
    """
    if not isinstance(data, Mapping) or data.get("schema") != TABLE_SCHEMA:
        raise FigureError(f"not a {TABLE_SCHEMA} convergence table")
    arm, stage = data.get("arm"), data.get("stage")
    if arm not in ARMS or stage not in CONVERGENCE_STAGES:
        raise FigureError(f"the convergence table names {arm!r}/{stage!r}; expected an arm of {list(ARMS)} and a stage "
                          f"of {list(CONVERGENCE_STAGES)}")
    sha = data.get("plan_sha256")
    if not isinstance(sha, str) or not _HEX64.fullmatch(sha):
        raise FigureError(f"the convergence table plan_sha256 must be 64 lowercase hex characters; got {sha!r}")
    if plan_sha256 is not None and sha != plan_sha256:
        raise FigureError(f"the {arm}/{stage} convergence table was written under plan {sha[:12]}, not the frozen "
                          f"plan {plan_sha256[:12]}")
    try:
        ks = tuple(_count(k, "k") for k in data["ks"])
        return ConvergenceStage(
            plan_sha256=sha, arm=arm, stage=stage, status=DRAWN, kind=_text(data["kind"], "kind"),
            model=_text(data["model"], "model"), model_digest=_text(data["model_digest"], "model_digest"),
            n_vocab=_count(data["n_vocab"], "n_vocab"), kept=_count(data["kept"], "kept"), ks=ks,
            served_k=_count(data["served_k"], "served_k"), tolerance=_real(data["tolerance"], "tolerance"),
            comparability=_flat(_text(data["comparability"], "comparability")),
            scope=_flat(_text(data["scope"], "scope")), estimator=_flat(_text(data["estimator"], "estimator")),
            n_items=len(data["items"]), n_missing_items=len(data["missing_items"]),
            n_draws=_count(data["n_draws"], "n_draws"), n_missing_draws=len(data["missing_draws"]),
            full_mean_mass=_real(data["full_mean_mass"], "full_mean_mass"),
            full_min_mass=_real(data["full_min_mass"], "full_min_mass"),
            features=tuple(_agreement_row(entry) for entry in data["features"]),
            mass=tuple(_mass_row(entry) for entry in data["mass"]))
    except (KeyError, TypeError, AttributeError) as exc:
        raise FigureError(f"the {arm}/{stage} convergence table is malformed: {exc!r}") from None


def read_convergence_stage(path: str | Path, *, plan_sha256: str | None = None) -> ConvergenceStage:
    """The convergence table at ``path`` (duplicate keys and non-finite numbers refused)."""
    return convergence_stage(_read_json(Path(path)), plan_sha256=plan_sha256)


def convergence_stages(plan: StudyPlan, *, arm: str = DEFAULT_CONVERGENCE_ARM,
                       stages: Sequence[str] = CONVERGENCE_STAGES,
                       analysis_dir: str | Path = CONVERGENCE_DIR) -> tuple[ConvergenceStage, ...]:
    """The stored convergence table of each of ``stages`` of ``arm``, or the reason it is missing.

    An arm without full-distribution evidence is missing with the plan's
    reason; a table not written yet is missing with its path under the
    analysis directory.
    """
    if arm not in ARMS:
        raise FigureError(f"unknown arm {arm!r}; expected one of {list(ARMS)}")
    stages = (stages,) if isinstance(stages, str) else tuple(stages)
    if not stages or len(set(stages)) != len(stages) or not set(stages) <= set(CONVERGENCE_STAGES):
        raise FigureError(f"stages must be distinct stages of {list(CONVERGENCE_STAGES)}; got {list(stages)}")
    arm_plan = plan.arms.arms[arm]
    found = []
    for stage in stages:
        if FULL_DISTRIBUTION not in arm_plan.estimate_sources:
            reason = arm_plan.missing_variants.get(f"ensemble_{FULL_DISTRIBUTION}") or (
                f"the {arm} arm records no full-distribution evidence")
            found.append(ConvergenceStage(plan.plan_sha256, arm, stage, MISSING, reason=_flat(reason)))
            continue
        path = table_path(arm, stage, analysis_dir)
        if not path.exists():
            found.append(ConvergenceStage(plan.plan_sha256, arm, stage, MISSING, reason=(
                f"no convergence table at {path.name} under the analysis directory: the analysis has not been run")))
            continue
        table = read_convergence_stage(path, plan_sha256=plan.plan_sha256)
        if (table.arm, table.stage) != (arm, stage):
            raise FigureError(f"{path} holds the table of {table.arm}/{table.stage}, not {arm}/{stage}")
        found.append(table)
    return tuple(found)


def _checked_stages(stages: Iterable[ConvergenceStage]) -> tuple[ConvergenceStage, ...]:
    stages = tuple(stages)
    if not stages:
        raise FigureError("the convergence figure needs at least one stage")
    for stage in stages:
        if not isinstance(stage, ConvergenceStage):
            raise TypeError(f"stages must be ConvergenceStage values; got {type(stage).__name__}")
    if len({stage.plan_sha256 for stage in stages}) != 1 or len({stage.arm for stage in stages}) != 1:
        raise FigureError("the convergence tables must come from one plan and one arm")
    names = [stage.stage for stage in stages]
    if len(set(names)) != len(names):
        raise FigureError(f"a stage is given twice: {names}")
    drawn = [stage for stage in stages if stage.status == DRAWN]
    if len({(stage.model, stage.model_digest) for stage in drawn}) > 1:
        raise FigureError("the convergence tables come from different models or weights "
                          f"{sorted({str(stage.model_digest) for stage in drawn})}; values compare only within one "
                          "model and one set of weights (Requirement 9.4)")
    if 2 * len(drawn) > MAX_LINES:
        raise FigureError(f"at most {MAX_LINES // 2} stages fit the at most {MAX_LINES} lines of a panel")
    return stages


# ---------- table and caption ----------

def _cells(**values: object) -> tuple[str, ...]:
    def cell(value: object) -> str:
        if value is None:
            return ""
        return _number(value) if isinstance(value, bool | int | float) else _flat(value)

    return tuple(cell(values.get(column)) for column in CONVERGENCE_COLUMNS)


def convergence_table(stages: Iterable[ConvergenceStage], plan: StudyPlan | None = None) -> FigureTable:
    """F4's table (every stored features and mass row of each stage, missing stages with the reason) and caption."""
    stages = _checked_stages(stages)
    role = None if plan is None else plan.arms.arms[stages[0].arm].role
    rows = []
    for stage in stages:
        head = {"arm": stage.arm, "role": role, "stage": stage.stage, "kind": stage.kind}
        if stage.status != DRAWN:
            rows.append(_cells(**head, drawn=False, missing=stage.reason))
            continue
        for row in stage.features:
            rows.append(_cells(**head, quantity=row.feature, evidence=row.evidence, k=row.k, n=row.n,
                               mean_abs_diff=row.mean_abs_diff, max_abs_diff=row.max_abs_diff,
                               share_within=row.share_within, drawn=row.missing_reason is None,
                               missing=row.missing_reason or row.missing))
        for row in stage.mass:
            rows.append(_cells(**head, quantity=MASS, evidence=row.evidence, k=row.k, n=row.n_draws,
                               mean_mass=row.mean_mass, min_mass=row.min_mass, drawn=row.missing_reason is None,
                               missing=row.missing_reason or row.missing))
    return FigureTable(name=CONVERGENCE, caption=_convergence_caption(stages, role), columns=CONVERGENCE_COLUMNS,
                       rows=tuple(rows))


def _convergence_caption(stages: Sequence[ConvergenceStage], role: str | None) -> str:
    drawn = [stage for stage in stages if stage.status == DRAWN]
    first = drawn[0] if drawn else None
    arm = stages[0].arm
    role_text = "" if role is None else f" ({role} role, {ROLE_PURPOSE.get(role, role)})"
    parts = ["Figure F4. Top-k versus full-distribution agreement as k grows."]
    if first is not None:
        parts.append(
            f"Data: the stored convergence tables (task 7.12; read, not recomputed) of the {arm} arm{role_text}, model "
            f"{first.model}, plan {first.plan_sha256[:12]}; per feature and k, the mean and the maximum over the "
            "compared draws of the absolute difference between the value from each draw's full-distribution evidence "
            "truncated to its first k alternatives and the value from the full distribution, and the probability "
            "mass the first k alternatives cover (mean and minimum over draws).")
        parts.append(f"Estimator: {first.estimator}")
        parts.append(f"Scope: {first.scope}")
        served = sorted({stage.served_k for stage in drawn if stage.served_k is not None})
        parts.append(f"The served row (the generation host's own top-{', '.join(map(str, served))} logprobs through "
                     "the top-k calibrator, a different calibrator) is a separate open marker at k = "
                     f"{', '.join(map(str, served))}, not joined to the lines.")
        sizes = "; ".join(
            f"{stage.name}: {stage.n_draws} draws of {stage.n_items} items ({stage.n_missing_draws} draws and "
            f"{stage.n_missing_items} planned items missing), {stage.kept} kept alternatives of a {stage.n_vocab}-token "
            f"vocabulary covering a mean mass of {_number(stage.full_mean_mass)} (minimum "
            f"{_number(stage.full_min_mass)})" for stage in drawn)
        parts.append(f"Sample size: {sizes}.")
        parts.append("Interval: none; the tables report the mean and the maximum absolute difference over draws and "
                     f"the share of draws within {_number(first.tolerance)} (in the table), and no interval.")
        parts.append(first.comparability)
    above = [(stage.name, f"k {', '.join(map(str, stage.above_kept))} above the {stage.kept} alternatives the "
              "evidence keeps") for stage in drawn if stage.above_kept]
    if above:
        parts.append(f"Not drawn: {_grouped(above)}; the table lists these rows as missing.")
    other = [(f"{stage.name} {row.evidence} k {row.k} {row.feature}", row.missing_reason) for stage in drawn
             for row in stage.features if row.missing_reason is not None and row.k not in stage.above_kept]
    if other:
        parts.append(f"Missing: {_grouped(other)}.")
    absent = [(stage.name, _flat(stage.reason)) for stage in stages if stage.status != DRAWN]
    if absent:
        parts.append(f"Missing (not drawn): {_grouped(absent)}.")
    parts.append("Encoding: one colour, marker shape and direct label per line (stage and statistic, palette slots 1 "
                 "to 4 in order): macro is the full run (the macro prompts of the positive control), EDGAR the "
                 "negative control (the EDGAR releases); solid lines are means, dashed lines the maximum (max, "
                 "agreement) or the minimum (min, mass); identical lines share one label; open markers are the served "
                 "row; the muted line marks no difference (agreement) or the whole vocabulary (mass); the x axis is "
                 "logarithmic.")
    kinds = {stage.kind for stage in drawn}
    sources = [*([FRED_SOURCE] if MACRO in kinds else []), *([EDGAR_SOURCE] if kinds - {MACRO} else []),
               FMP_STUDY_SOURCE]
    parts += [f"Sources: {'; '.join(sources)}.", DERIVED_ONLY]
    return " ".join(parts)


# ---------- drawing ----------

@dataclass(frozen=True)
class _Series:
    """One line of a panel: its stage, statistic, palette slot, points and the served value."""

    key: str
    label: str
    slot: int
    dashed: bool
    ks: tuple[int, ...]
    values: tuple[float, ...]
    served_k: int | None
    served: float | None


def _series(drawn: Sequence[ConvergenceStage], feature: str) -> list[_Series]:
    statistics = MASS_STATISTICS if feature == MASS else FEATURE_STATISTICS
    found = []
    for index, stage in enumerate(drawn):
        if feature == MASS:
            host, served_rows = stage.coverage(HOST_TOP_K), stage.coverage(SERVED_TOP_K)
        else:
            host, served_rows = stage.agreement(HOST_TOP_K, feature), stage.agreement(SERVED_TOP_K, feature)
        for position, (field_name, name) in enumerate(statistics):
            points = [(row.k, getattr(row, field_name)) for row in sorted(host, key=lambda row: row.k)
                      if row.missing_reason is None and getattr(row, field_name) is not None]
            served = next((getattr(row, field_name) for row in served_rows if row.missing_reason is None), None)
            found.append(_Series(key=f"{stage.stage}-{field_name}", label=f"{STAGE_SHORT[stage.stage]}, {name}",
                                 slot=2 * index + position, dashed=position == 1,
                                 ks=tuple(k for k, _ in points), values=tuple(value for _, value in points),
                                 served_k=stage.served_k, served=served))
    return found


def _groups(series: Sequence[_Series]) -> list[list[_Series]]:
    """Lines with identical points, which one label names together."""
    groups: dict[tuple[tuple[int, ...], tuple[float, ...]], list[_Series]] = {}
    for line in series:
        if line.ks:
            groups.setdefault((line.ks, line.values), []).append(line)
    return list(groups.values())


def _group_label(group: Sequence[_Series], total: int) -> str:
    """The label of identical lines: their names, or "all lines" when every line of the panel is among them."""
    if len(group) == total > 1:
        text = "all lines"
    else:
        text = " = ".join(line.label for line in group)
    if all(value == 0.0 for value in group[0].values):
        text += ": 0 at every k"
    return textwrap.fill(text, width=_LABEL_CHARS)


def _draw_panel(axes: Axes, feature: str, series: Sequence[_Series], style: FigureTheme, theme: str) -> None:
    from recall_guard.harness.plots import PAPER_MARKERS, figure_theme

    palette = figure_theme(theme).palette
    _style_axes(axes, style)
    ks = sorted({k for line in series for k in (*line.ks, *([line.served_k] if line.served is not None else []))
                 if k is not None})
    if ks:
        # From the first k rightwards only: the direct labels take the space left of it (_label_column).
        reference = 1.0 if feature == MASS else 0.0
        axes.plot([ks[0], ks[-1] * 1.6], [reference, reference], color=style.muted, linewidth=1.0, linestyle="-",
                  marker="none", zorder=1, gid=f"convergence-reference-{feature}")
    for line in series:
        colour, marker = palette[line.slot], PAPER_MARKERS[line.slot]
        if line.ks:
            axes.plot(line.ks, line.values, color=colour, linewidth=_LINE_PT, linestyle="--" if line.dashed else "-",
                      marker=marker, markersize=_MARKER_PT, markerfacecolor=colour, markeredgecolor=colour,
                      zorder=3, gid=f"convergence-{feature}-{line.key}")
        if line.served is not None and line.served_k is not None:
            axes.plot([line.served_k], [line.served], linestyle="none", marker=marker, markersize=_SERVED_PT,
                      markerfacecolor=style.surface, markeredgecolor=colour, markeredgewidth=1.4, zorder=4,
                      gid=f"convergence-served-{feature}-{line.key}")
    values = [value for line in series for value in (*line.values, line.served) if value is not None]
    if feature == MASS:
        bottom = min([*values, 0.0]) - 0.05
        axes.set_ylim(bottom, 1.0 + _MASS_HEADROOM * (1.0 - bottom))
    else:
        top = max([*values, 0.0])
        top = top * 1.3 if top > 0 else 1.0
        axes.set_ylim(-0.06 * top, top)
    axes.set_xscale("log")
    if ks:
        axes.set_xlim(ks[0] / 1.6, ks[-1] * 1.6)
        axes.set_xticks(ks, labels=[str(k) for k in ks])
        axes.xaxis.set_minor_locator(_no_ticks())
    title = (f"{FEATURE_NAMES[feature]}: absolute difference to the full distribution" if feature != MASS
             else "Probability mass the first k alternatives cover")
    axes.set_title(textwrap.fill(title, width=_TITLE_CHARS), loc="left", color=style.ink)
    axes.set_ylabel("mass covered" if feature == MASS else "absolute difference", fontsize=_RC["axes.labelsize"])


def _no_ticks() -> Any:
    from matplotlib.ticker import NullLocator

    return NullLocator()


def _marker_boxes(axes: Axes, clearance_pt: float = 0.0, *, k: float | None = None) -> list[Any]:
    """Every visible marker of ``axes`` as drawn, as a display-space box padded by ``clearance_pt``.

    The marker's path at the marker size (a diamond reaches 0.71 of the size
    from its centre, a circle or a square 0.5) plus half the edge width, so a
    label kept clear of these boxes is clear of what is drawn; the half-size
    boxes of :func:`~scripts.finance_sim.figures._obstacles` understate a
    diamond. With ``k``, only the points at that x. Call after the layout is
    fixed.
    """
    from matplotlib.markers import MarkerStyle
    from matplotlib.transforms import Bbox

    scale = axes.figure.dpi / 72.0
    boxes = []
    for line in axes.lines:
        marker = line.get_marker()
        if not line.get_visible() or marker in (None, "None", "none", "", " "):
            continue
        marker_style = MarkerStyle(marker)
        reach = marker_style.get_path().transformed(marker_style.get_transform()).get_extents()
        size = line.get_markersize() * scale
        edge = (line.get_markeredgewidth() / 2.0 + clearance_pt) * scale
        points = line.get_xydata()
        for (x, _), (cx, cy) in zip(points, line.get_transform().transform(points), strict=True):
            if k is None or x == k:
                boxes.append(Bbox.from_extents(cx + reach.x0 * size - edge, cy + reach.y0 * size - edge,
                                               cx + reach.x1 * size + edge, cy + reach.y1 * size + edge))
    return boxes


def _marker_reach(axes: Axes, k: int) -> float:
    """How far left of their centre the markers drawn at ``k`` reach, in pixels (0 without a marker there)."""
    centre = axes.transData.transform((k, 0.0))[0]
    return max((centre - box.x0 for box in _marker_boxes(axes, k=k)), default=0.0)


def _label_box(label: Any, renderer: Any, scale: float) -> Any:
    """A placed label as an obstacle: its surface box plus the clearance the next label keeps from it."""
    return label.get_window_extent(renderer).padded((_BOX_PAD_PT + _CLEARANCE_PT) * scale)


def _label_column(axes: Axes, feature: str, series: Sequence[_Series], style: FigureTheme, renderer: Any) -> None:
    """Direct labels in a column left of the lines' first points, where the lines lie furthest apart.

    Runs after the layout is fixed: the x axis is extended to the left just
    enough for the widest label (the axes box does not move), each label is
    set beside its line's first point with its box ``_CLEARANCE_PT`` clear of
    the markers drawn there, and the column is spread vertically so no two
    boxes come closer than that and all stay inside the frame. A label moved
    off its point gets a thin leader to it. Identical lines share one label.
    """
    groups = _groups(series)
    if not groups:
        return
    scale = axes.figure.dpi / 72.0
    texts = [_group_label(group, len(series)) for group in groups]
    style_kwargs = {"fontsize": _LABEL_PT, "linespacing": 1.1}
    probes = [axes.text(0.0, 0.0, text, transform=axes.transAxes, **style_kwargs) for text in texts]
    sizes = [probe.get_window_extent(renderer) for probe in probes]
    for probe in probes:
        probe.remove()
    first_k = min(line.ks[0] for group in groups for line in group)
    # From the first points' centre to a label's right edge: the markers' reach there (the same under any x
    # limits), the clearance, then the label's own box pad.
    gap = _marker_reach(axes, first_k) + (_CLEARANCE_PT + _BOX_PAD_PT) * scale
    low, high = axes.get_xlim()
    share = min((max(size.width for size in sizes) + gap + 4.0 * scale) / axes.bbox.width, 0.55)
    span = math.log(high) - math.log(first_k)
    axes.set_xlim(math.exp(math.log(first_k) - share / (1.0 - share) * span), high)
    point_x = axes.transData.transform((first_k, 0.0))[0]
    targets = [float(axes.transData.transform((group[0].ks[0], group[0].values[0]))[1]) for group in groups]
    heights = [size.height + 2.0 * _BOX_PAD_PT * scale for size in sizes]
    frame = axes.bbox
    positions: dict[int, float] = {}
    floor = frame.y0 + scale
    for index in sorted(range(len(groups)), key=lambda i: targets[i]):
        centre = max(targets[index], floor + heights[index] / 2.0)
        positions[index] = centre
        floor = centre + heights[index] / 2.0 + _CLEARANCE_PT * scale
    overflow = floor - _CLEARANCE_PT * scale - (frame.y1 - scale)
    if overflow > 0:
        positions = {index: centre - overflow for index, centre in positions.items()}
    inverse = axes.transData.inverted()
    for index, (group, text) in enumerate(zip(groups, texts, strict=True)):
        x, y = inverse.transform((point_x - gap, positions[index]))
        axes.annotate(text, (x, y), ha="right", va="center", color=style.ink, zorder=5, annotation_clip=False,
                      in_layout=False, gid=f"convergence-label-{feature}-{index}",
                      bbox={"facecolor": style.surface, "edgecolor": "none", "pad": _BOX_PAD_PT}, **style_kwargs)
        if abs(positions[index] - targets[index]) > 2.0 * scale:
            line = group[0]
            axes.plot([x, line.ks[0]], [y, line.values[0]], color=style.secondary_ink, linewidth=0.6, linestyle="-",
                      marker="none", zorder=2, gid=f"convergence-leader-{feature}-{index}")


def _place_line_labels(axes: Axes, feature: str, series: Sequence[_Series], style: FigureTheme,
                       renderer: Any) -> None:
    """Direct labels: the line column, then one label for the served markers and one for the mass reference.

    The two single labels are placed by their boxes (:func:`_place_text`)
    against the lines, the markers as drawn and the labels placed before,
    each kept ``_CLEARANCE_PT`` away. The served label tries the rings of
    ``_SERVED_RINGS_PT`` around the served markers, highest marker first. The
    whole-vocabulary label goes to the right end of its line, above it, which
    neither the column, the served label nor the lines that approach the line
    from below reach; earlier points along the line are its fallbacks.
    """
    _label_column(axes, feature, series, style, renderer)
    scale = axes.figure.dpi / 72.0
    paths, _ = _obstacles(axes)
    boxes = _marker_boxes(axes, _CLEARANCE_PT)
    boxes += [_label_box(text, renderer, scale) for text in axes.texts]
    served = [(line.served_k, line.served) for line in series if line.served is not None and line.served_k is not None]
    if served:
        k = served[0][0]
        # Nearest first; farther rings clear lines that converge at the served k.
        candidates = [placement for offset in _SERVED_RINGS_PT
                      for anchor in sorted(served, key=lambda point: -point[1]) for placement in _around(anchor, offset)]
        # Only the estimate goes through a calibrator; the features and the mass are the host's top-k alone.
        text = f"served top-{k}\n(top-k calibrator)" if feature == "estimate" else f"served top-{k}"
        label = _place_text(axes, text, candidates, renderer, paths=paths,
                            boxes=boxes, fontsize=_LABEL_PT, color=style.secondary_ink, zorder=5,
                            gid=f"convergence-served-label-{feature}", linespacing=1.1,
                            bbox={"facecolor": style.surface, "edgecolor": "none", "pad": _BOX_PAD_PT})
        boxes.append(_label_box(label, renderer, scale))
    if feature == MASS:
        low, high = axes.get_xlim()
        end = (high, 1.0)
        candidates = [(end, -3.0, _REFERENCE_OFFSET_PT, "right", "bottom"),
                      (end, -3.0, -_REFERENCE_OFFSET_PT, "right", "top")]
        anchors = [(math.exp(math.log(low) + share * (math.log(high) - math.log(low))), 1.0)
                   for share in (0.55, 0.3, 0.05)]
        candidates += [(anchor, 0.0, _REFERENCE_OFFSET_PT, "left", "bottom") for anchor in anchors]
        candidates += [(anchor, 0.0, -_REFERENCE_OFFSET_PT, "left", "top") for anchor in anchors]
        _place_text(axes, "whole vocabulary", candidates, renderer, paths=paths, boxes=boxes, fontsize=_LABEL_PT,
                    color=style.secondary_ink, zorder=5, gid="convergence-reference-label",
                    bbox={"facecolor": style.surface, "edgecolor": "none", "pad": _BOX_PAD_PT})


def plot_convergence(stages: Sequence[ConvergenceStage], *, theme: str = "light") -> Figure:
    """F4: agreement per feature and mass covered as k grows, one line per stage and statistic (Requirement 10.5).

    ``stages`` are plain data read from the stored tables
    (:func:`convergence_stages`); missing stages are not drawn. Raises
    :class:`FigureError` when every stage is missing. Returns a matplotlib
    Figure (no pyplot, no global state); labels are placed after the layout.
    """
    import matplotlib
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    from recall_guard.harness.plots import figure_theme

    stages = _checked_stages(stages)
    drawn = [stage for stage in stages if stage.status == DRAWN]
    if not drawn:
        raise FigureError("nothing to draw: every convergence table is missing (see the table)")
    style = figure_theme(theme)
    quantities = (*FEATURES, MASS)
    rows = math.ceil(len(quantities) / 2)
    with matplotlib.rc_context(_theme_rc(theme)):
        figure = Figure(figsize=(_WIDTH_IN, rows * (_PANEL_IN + 2 * _TITLE_LINE_IN) + _FRAME_IN),
                        layout="constrained")
        grid = figure.subplots(rows, 2, squeeze=False)
        cells = list(grid.ravel())
        drawn_series = {}
        for axes, quantity in zip(cells, quantities, strict=False):
            drawn_series[quantity] = _series(drawn, quantity)
            _draw_panel(axes, quantity, drawn_series[quantity], style, theme)
        for axes in cells[len(quantities):]:
            axes.set_axis_off()
        figure.supxlabel("k, alternatives kept per position (log scale)")
        digest = str(drawn[0].model_digest)
        figure.suptitle(
            "Top-k versus full-distribution agreement as k grows\n"
            f"Values compare only within one model and one set of weights: {drawn[0].model}, {digest[:19]}",
            x=0.01, ha="left")
        renderer = FigureCanvasAgg(figure).get_renderer()
        figure.draw_without_rendering()
        for axes, quantity in zip(cells, quantities, strict=False):
            _place_line_labels(axes, quantity, drawn_series[quantity], style, renderer)
        figure.draw_without_rendering()
    return figure


# ---------- outputs ----------

def render_convergence(plan: StudyPlan, *, arm: str = DEFAULT_CONVERGENCE_ARM,
                       stages: Sequence[str] = CONVERGENCE_STAGES, analysis_dir: str | Path = CONVERGENCE_DIR,
                       out_dir: str | Path = FIGURES_DIR) -> tuple[Path, ...]:
    """Render F4 from the stored convergence tables of ``plan`` and write its four files once."""
    found = convergence_stages(plan, arm=arm, stages=stages, analysis_dir=analysis_dir)
    texts = figure_texts(CONVERGENCE, convergence_table(found, plan),
                         lambda theme: plot_convergence(found, theme=theme), out_dir)
    return write_outputs(texts)


def main(argv: Sequence[str] | None = None) -> int:
    """Render the convergence figure of the frozen plan from its stored convergence tables.

    Exit codes: 0 written (or already written identically), 2 refused (another
    plan, a malformed table, tables of different weights, nothing to draw, or
    a different figure at the output path).
    """
    parser = argparse.ArgumentParser(description="Convergence figure F4 (task 8.3), from the stored convergence "
                                                 "tables; sends no request.")
    parser.add_argument("--plan", type=Path, required=True, help="the frozen plan.json")
    parser.add_argument("--plan-sha256", required=True, help="the plan hash the tables carry")
    parser.add_argument("--arm", default=DEFAULT_CONVERGENCE_ARM, choices=ARMS)
    parser.add_argument("--stage", action="append", choices=CONVERGENCE_STAGES,
                        help="a stage to draw (repeatable; default both, the full run first)")
    parser.add_argument("--analysis-dir", type=Path, default=CONVERGENCE_DIR)
    parser.add_argument("--out-dir", type=Path, default=FIGURES_DIR)
    args = parser.parse_args(argv)
    try:
        plan = load_plan(args.plan, expected_sha256=args.plan_sha256)
        paths = render_convergence(plan, arm=args.arm, stages=args.stage or CONVERGENCE_STAGES,
                                   analysis_dir=args.analysis_dir, out_dir=args.out_dir)
    except (ValueError, TypeError, OSError) as exc:
        print(f"figures_convergence: refused: {exc}", file=sys.stderr)
        return EXIT_REFUSED
    for path in paths:
        print(path)
    return EXIT_WRITTEN


__all__ = [
    "CONVERGENCE_COLUMNS",
    "DEFAULT_CONVERGENCE_ARM",
    "FEATURE_NAMES",
    "MASS",
    "MAX_LINES",
    "STAGE_NAMES",
    "AgreementRow",
    "ConvergenceStage",
    "MassRow",
    "convergence_stage",
    "convergence_stages",
    "convergence_table",
    "main",
    "plot_convergence",
    "read_convergence_stage",
    "render_convergence",
]


if __name__ == "__main__":
    raise SystemExit(main())
