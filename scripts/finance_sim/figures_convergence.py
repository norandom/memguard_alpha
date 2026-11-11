"""Convergence figure (Figure 4) of the worked example.

No request is sent and nothing is recomputed: every number comes from the
stored convergence tables
(``runs/finance-sim/analysis/convergence-<arm>-<stage>.json``,
:func:`~scripts.finance_sim.convergence.table_path`), one per analysed stage
(``full_run``, the macro prompts, and ``negative_control``, the EDGAR
releases) of the arm that records full-distribution evidence.

The figure carries one message: how fast the recall estimate computed from
the first k alternatives per token approaches the estimate from logits (the
kept top alternatives of the next-token vector, with the stored kept count in
the label). Three panels in one row, labelled (a) to (c)
above their plot areas, in the one academic style
(:mod:`scripts.finance_sim.figure_style`): the absolute difference to the
value from logits for Min-K%++ and for the recall estimate, and the
probability mass the first k alternatives cover; x is k on a log scale. Per
stage one line joins the means over the compared draws and one the worst
draw; with two stages that is at most four lines per panel, told apart by
marker shape, line style and grey level and named in one legend above the
panels. The served values (the route's own top alternatives through the
calibrator of logprobs, a different calibrator) are one separate marker per
stage at the served k, never joined to the lines: the mean only, because the
mean and the worst draw would be two markers in one place; the worst draw is
in the table.

What is not plotted, and why. Loss, Min-K% and the zlib ratio use only the
logprob of each generated token, which truncation keeps exact, so their
differences are zero at every k by construction: the caption says so in one
clause, a note below the table gives the reason and the table keeps their
rows. Min-K%++ at k = 1 is undefined (one alternative has no spread; the
stored value only reflects the floor on the standard deviation): the point is
not plotted, the caption says so, and the table keeps the stored value with a
note. A k above the kept count is not drawn and is a row of the table with
its reason. The figure refuses tables of different
weights; a note below the table states that values compare only within one
model and one set of weights.

Outputs and the write-once rule are those of
:mod:`scripts.finance_sim.figures`: ``convergence.svg``, ``convergence.csv``
and ``convergence.md``; derived statistics only. Matplotlib is imported inside
the functions that draw.
"""

from __future__ import annotations

import argparse
import sys
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
from scripts.finance_sim.figure_style import (
    FIGURE_WIDTH_IN,
    FIGURES_DIR,
    FigureError,
    caption,
    figure_rc,
    label_panel,
    lay_out,
    panel_label,
    series_style,
    write_outputs,
)
from scripts.finance_sim.figures import (
    _HEX64,
    CONVERGENCE,
    DRAWN,
    EDGAR_SOURCE,
    EXIT_REFUSED,
    EXIT_WRITTEN,
    FIGURE_NUMBERS,
    FMP_STUDY_SOURCE,
    FRED_SOURCE,
    MISSING,
    FigureTable,
    _count,
    _flat,
    _listed,
    _number,
    _read_json,
    _real,
    _text,
    evidence_label,
    figure_texts,
    rounded,
    route_label,
)
from scripts.finance_sim.limits import ARMS, SAMPLING_ARM
from scripts.finance_sim.plan import FULL_DISTRIBUTION, TOP_K, StudyPlan, load_plan

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

#: The arm with full-distribution evidence (the convergence analysis refuses the other).
DEFAULT_CONVERGENCE_ARM = SAMPLING_ARM
STAGE_NAMES: Mapping[str, str] = {"full_run": "full run", "negative_control": "negative control"}
FEATURE_NAMES: Mapping[str, str] = {"loss": "loss", "min_k": "Min-K%", "min_k_pp": "Min-K%++",
                                    "zlib_ratio": "the zlib ratio", "estimate": "the recall estimate"}
#: The two statistics drawn per stage, the mean over draws and the worst draw: marker, line style and grey level
#: tell them apart.
FEATURE_STATISTICS: tuple[tuple[str, str], ...] = (("mean_abs_diff", "mean"), ("max_abs_diff", "max"))
MASS_STATISTICS: tuple[tuple[str, str], ...] = (("mean_mass", "mean"), ("min_mass", "min"))
#: Stage names of the legend and the caption, by the prompts each stage scores.
STAGE_SHORT: Mapping[str, str] = {"full_run": "macro prompts", "negative_control": "EDGAR releases"}
MASS = "mass"
#: The panels in order, (a) to (c): the two quantities that change with k, then the mass.
PLOTTED: tuple[str, ...] = ("min_k_pp", "estimate", MASS)
#: What each panel shows, for the caption, and the y label of every panel with its unit.
PANEL_NAMES: Mapping[str, str] = {"min_k_pp": "Min-K%++", "estimate": "the recall estimate",
                                  MASS: "the mass covered"}
Y_LABELS: Mapping[str, str] = {
    "min_k_pp": "absolute difference to logits,\nMin-K%++ (z-score units)",
    "estimate": "absolute difference to logits,\nrecall estimate (probability)",
    MASS: "probability mass covered\n(share)",
}
#: Features computed from the logprob of each generated token alone, which the truncation keeps exact.
ACTUAL_TOKEN_FEATURES: tuple[str, ...] = ("loss", "min_k", "zlib_ratio")
#: The feature that needs the spread of the alternatives, and the k at which there is none.
SPREAD_FEATURE, NO_SPREAD_K = "min_k_pp", 1
UNDEFINED_NOTE = ("undefined at k = 1: one alternative has no spread, so the stored value reflects only the floor "
                  "on the standard deviation; not plotted")
ZERO_NOTE, UNPLOTTED_NOTE = "zero at every k by construction; not plotted", "not plotted"
CONVERGENCE_COLUMNS: tuple[str, ...] = (
    "arm", "route", "role", "stage", "kind", "quantity", "evidence", "k", "n", "mean_abs_diff", "max_abs_diff",
    "share_within", "mean_mass", "min_mass", "drawn", "missing", "note")
#: At most four lines per panel: two stages times two statistics, series styles 0 to 3 in this order; the served
#: markers of the stages take the next styles.
MAX_LINES = 4
#: Figure size in inches: the legend, one row of three panels and the x label below them.
_SIZE_IN = (FIGURE_WIDTH_IN, 3.6)


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


def is_plotted(quantity: str, k: int) -> bool:
    """Whether the figure plots ``quantity`` at ``k``: a panel's quantity, and not Min-K%++ without a spread."""
    return quantity in PLOTTED and not (quantity == SPREAD_FEATURE and k == NO_SPREAD_K)


def _feature_note(row: AgreementRow, zero: Sequence[str]) -> str | None:
    """Why a stored features row with values is not plotted, for the table; None when it is plotted or missing."""
    if row.missing_reason is not None or is_plotted(row.feature, row.k):
        return None
    if row.feature == SPREAD_FEATURE:
        return UNDEFINED_NOTE
    return ZERO_NOTE if row.feature in zero and row.evidence == HOST_TOP_K else UNPLOTTED_NOTE


def convergence_table(stages: Iterable[ConvergenceStage], plan: StudyPlan | None = None) -> FigureTable:
    """Figure 4's table (every stored features and mass row of each stage, missing stages with the reason), its
    caption and its notes. With ``plan`` the rows and the caption name the serving route."""
    stages = _checked_stages(stages)
    role = None if plan is None else plan.arms.arms[stages[0].arm].role
    route = None if plan is None else route_label(plan, stages[0].arm)
    drawn = [stage for stage in stages if stage.status == DRAWN]
    zero = _zero_by_construction(drawn)
    rows = []
    for stage in stages:
        head = {"arm": stage.arm, "route": route, "role": role, "stage": stage.stage, "kind": stage.kind}
        if stage.status != DRAWN:
            rows.append(_cells(**head, drawn=False, missing=stage.reason))
            continue
        for row in stage.features:
            rows.append(_cells(**head, quantity=row.feature, evidence=row.evidence, k=row.k, n=row.n,
                               mean_abs_diff=row.mean_abs_diff, max_abs_diff=row.max_abs_diff,
                               share_within=row.share_within,
                               drawn=row.missing_reason is None and is_plotted(row.feature, row.k),
                               missing=row.missing_reason or row.missing, note=_feature_note(row, zero)))
        for row in stage.mass:
            rows.append(_cells(**head, quantity=MASS, evidence=row.evidence, k=row.k, n=row.n_draws,
                               mean_mass=row.mean_mass, min_mass=row.min_mass, drawn=row.missing_reason is None,
                               missing=row.missing_reason or row.missing))
    return FigureTable(name=CONVERGENCE, caption=_convergence_caption(stages, route), columns=CONVERGENCE_COLUMNS,
                       rows=tuple(rows), notes=_convergence_notes(drawn), sources=_convergence_sources(drawn))


def _zero_by_construction(drawn: Sequence[ConvergenceStage]) -> list[str]:
    """The actual-token features whose stored host rows are zero at every k in every drawn stage."""
    found = []
    for feature in ACTUAL_TOKEN_FEATURES:
        rows = [row for stage in drawn for row in stage.agreement(HOST_TOP_K, feature) if row.missing_reason is None]
        if rows and all(row.mean_abs_diff == 0.0 and row.max_abs_diff == 0.0 for row in rows):
            found.append(feature)
    return found


def _sentence(names: Sequence[str]) -> str:
    """The names as the subject of a sentence: listed, the first letter a capital."""
    text = _listed(names)
    return text[0].upper() + text[1:]


def _plotted_points(stage: ConvergenceStage, quantity: str) -> list[AgreementRow | MassRow]:
    """The stored truncation rows of ``quantity`` the figure plots for ``stage``, by k."""
    rows = stage.coverage(HOST_TOP_K) if quantity == MASS else stage.agreement(HOST_TOP_K, quantity)
    mean = (MASS_STATISTICS if quantity == MASS else FEATURE_STATISTICS)[0][0]
    return sorted((row for row in rows if row.missing_reason is None and getattr(row, mean) is not None
                   and is_plotted(quantity, row.k)), key=lambda row: row.k)


def _key_numbers(drawn: Sequence[ConvergenceStage]) -> tuple[int, float, float] | None:
    """The k the caption quotes, the largest mean difference of the estimate there and the smallest mean mass.

    The k is the served k, where truncated logits and served logprobs hold the
    same number of alternatives; without a served k among the plotted points
    it is the smallest plotted k above one alternative. A k at which every
    kept alternative is used is never quoted: its difference is zero by
    construction.
    """
    if not drawn:
        return None
    estimate = [{row.k: row.mean_abs_diff for row in _plotted_points(stage, "estimate")} for stage in drawn]
    mass = [{row.k: row.mean_mass for row in _plotted_points(stage, MASS)} for stage in drawn]
    kept = {stage.kept for stage in drawn}
    common = sorted(k for k in set.intersection(*(set(found) for found in (*estimate, *mass)))
                    if k not in kept and k != NO_SPREAD_K)
    if not common:
        return None
    served = {stage.served_k for stage in drawn}
    k = next((k for k in common if served == {k}), common[0])
    return k, max(found[k] for found in estimate), min(found[k] for found in mass)


def _one(values: Iterable[int | None]) -> int | None:
    """The one count every drawn stage states, or None when they state none or different ones."""
    found = set(values)
    return found.pop() if len(found) == 1 else None


def _full_ks(drawn: Sequence[ConvergenceStage]) -> list[int]:
    """The k values at which a drawn stage uses every kept alternative, so its differences are zero by construction."""
    return sorted({stage.kept for stage in drawn if stage.kept is not None and stage.kept in stage.ks})


def _convergence_caption(stages: Sequence[ConvergenceStage], route: str | None) -> str:
    """Figure 4's caption: the difference at the quoted k, the draws, the panels, what is zero by construction.

    The label of logits is built from the kept count the tables store
    (:func:`~scripts.finance_sim.figures.evidence_label`). Why a value is zero
    by construction and what the served marker is are notes below the table
    (:func:`_convergence_notes`); a k above the kept count is a row of the
    table.
    """
    drawn = [stage for stage in stages if stage.status == DRAWN]
    key = _key_numbers(drawn)
    logits = evidence_label(FULL_DISTRIBUTION, _one(stage.kept for stage in drawn))
    if not drawn:
        message = "No convergence table is stored, so nothing is plotted"
    elif key is None:
        message = (f"The recall estimate from the first k alternatives per token against the estimate from {logits}, "
                   "as k grows")
    else:
        k, difference, mass = key
        message = (f"The recall estimate from the first k alternatives per token is within {rounded(difference)} on "
                   f"average of that from {logits} at k = {k}, where the mean mass covered is at least "
                   f"{rounded(mass)}")
    sizes = _listed([f"{stage.n_draws} draws of {stage.n_items} {STAGE_SHORT[stage.stage]}" for stage in drawn]
                    ) if drawn else "no compared draws"
    if route is not None and drawn:
        sizes += f" on {route}"
    letters = [panel_label(index) for index in range(len(PLOTTED))]
    panels = _listed([f"{letter} {PANEL_NAMES[quantity]}" for letter, quantity in zip(letters, PLOTTED, strict=True)])
    notes = [f"{panels}; no interval is drawn."]
    if drawn:
        zero = _zero_by_construction(drawn)
        pieces = [f"at k = {_listed([str(k) for k in _full_ks(drawn)])}"] if _full_ks(drawn) else []
        if zero:
            pieces.append(f"{'at every k, ' if pieces else 'at every k '}for "
                          f"{_listed([FEATURE_NAMES[feature] for feature in zero])} (not plotted)")
        if pieces:
            notes.append(f"By construction the differences are zero {' and, '.join(pieces)}.")
        other = [feature for feature in ACTUAL_TOKEN_FEATURES if feature not in zero]
        if other:
            several = len(other) > 1
            notes.append(f"{_sentence([FEATURE_NAMES[feature] for feature in other])} {'are' if several else 'is'} "
                         f"not plotted{' either' if zero else ''} (see the table).")
        if any(row.k == NO_SPREAD_K and row.missing_reason is None
               for stage in drawn for row in stage.agreement(HOST_TOP_K, SPREAD_FEATURE)):
            notes.append(f"{FEATURE_NAMES[SPREAD_FEATURE]} is not plotted at k = {NO_SPREAD_K} (see the table).")
        if any(row.missing_reason is not None and row.k not in stage.above_kept
               for stage in drawn for row in (*stage.features, *stage.mass)):
            notes.append("Some points are missing (see the table).")
    for stage in stages:
        if stage.status != DRAWN:
            name = STAGE_SHORT.get(stage.stage, stage.name)
            notes.append(f"{name[0].upper()}{name[1:]} are not drawn (see the table).")
    return caption(FIGURE_NUMBERS[CONVERGENCE], message, n=sizes, interval=" ".join(notes))


def _convergence_notes(drawn: Sequence[ConvergenceStage]) -> tuple[str, ...]:
    """The lines below Figure 4's table: what a row compares, why some differences are zero by construction, what
    the served marker is, and within what the values compare."""
    if not drawn:
        return ()
    first = drawn[0]
    kept = _one(stage.kept for stage in drawn)
    notes = [
        "A difference is the absolute difference to the value from logits, over the compared draws; the worst draw "
        "is the largest difference or the smallest mass.",
        f"A truncated row keeps the first k of the {'' if kept is None else f'{kept} '}alternatives stored per token "
        "and the calibrator of logits; the served row uses the route's own top alternatives through the calibrator "
        "of logprobs.",
    ]
    zero = _zero_by_construction(drawn)
    reasons = []
    if zero:
        several = len(zero) > 1
        reasons.append(f"{_sentence([FEATURE_NAMES[feature] for feature in zero])} use{'' if several else 's'} only "
                       "the logprob of each generated token, which truncation keeps exact")
    if _full_ks(drawn):
        reasons.append(f"at k = {_listed([str(k) for k in _full_ks(drawn)])} every kept alternative is used")
    if reasons:
        text = "; ".join(reasons)
        notes.append(f"{text[0].upper()}{text[1:]}.")
    served = _one(stage.served_k for stage in drawn)
    if served is not None:
        notes.append(f"The figure shows the mean of the served row as a separate marker at k = {served}; its worst "
                     "draw is in the table.")
    notes.append(f"Values compare only within one model and one set of weights ({first.model}, weights "
                 f"{first.model_digest}).")
    return tuple(notes)


def _convergence_sources(drawn: Sequence[ConvergenceStage]) -> tuple[str, ...]:
    kinds = {stage.kind for stage in drawn}
    return (*((FRED_SOURCE,) if MACRO in kinds else ()), *((EDGAR_SOURCE,) if kinds - {MACRO} else ()),
            FMP_STUDY_SOURCE)


# ---------- drawing ----------

def _plain(value: float, _position: object = None) -> str:
    """A tick value written in full, without an exponent: no offset text stands above an axis."""
    text = f"{value:.10f}".rstrip("0").rstrip(".")
    return "0" if text in ("", "-0") else text


def _draw_panel(axes: Axes, quantity: str, drawn: Sequence[ConvergenceStage], ks: Sequence[int]) -> None:
    """One panel: per stage a line of the means and one of the worst draw over k, and the served mean as a marker."""
    from matplotlib.ticker import FuncFormatter, NullLocator

    statistics = MASS_STATISTICS if quantity == MASS else FEATURE_STATISTICS
    values: list[float] = []
    for index, stage in enumerate(drawn):
        points = _plotted_points(stage, quantity)
        for position, (field_name, _) in enumerate(statistics):
            found = [(row.k, getattr(row, field_name)) for row in points if getattr(row, field_name) is not None]
            if found:
                axes.plot([k for k, _ in found], [value for _, value in found], zorder=3, clip_on=False,
                          gid=f"convergence-{quantity}-{stage.stage}-{field_name}",
                          **series_style(2 * index + position))
                values += [value for _, value in found]
        served_rows = stage.coverage(SERVED_TOP_K) if quantity == MASS else stage.agreement(SERVED_TOP_K, quantity)
        served = next((getattr(row, statistics[0][0]) for row in served_rows if row.missing_reason is None), None)
        if served is not None and stage.served_k is not None:
            axes.plot([stage.served_k], [served], zorder=4, clip_on=False,
                      gid=f"convergence-served-{quantity}-{stage.stage}",
                      **{**series_style(MAX_LINES + index), "linestyle": "none"})
            values.append(served)
    if quantity == MASS:
        axes.set_ylim(min([*values, 0.0]) - 0.05, 1.05)
    else:
        top = max([*values, 0.0])
        top = top * 1.15 if top > 0 else 1.0
        axes.set_ylim(-0.06 * top, top)
    axes.set_xscale("log")
    if ks:
        axes.set_xlim(ks[0] / 1.5, ks[-1] * 1.5)
        axes.set_xticks(list(ks), labels=[str(k) for k in ks])
        axes.xaxis.set_minor_locator(NullLocator())
    axes.yaxis.set_major_formatter(FuncFormatter(_plain))
    axes.set_ylabel(Y_LABELS[quantity])


def _legend(figure: Figure, drawn: Sequence[ConvergenceStage]) -> None:
    """The figure's one legend, above the panels: per stage its two lines and the marker of its served mean."""
    from matplotlib.lines import Line2D

    handles = []
    for index, stage in enumerate(drawn):
        name = STAGE_SHORT[stage.stage]
        handles += [
            Line2D([], [], label=f"{name}, mean over draws", **series_style(2 * index)),
            Line2D([], [], label=f"{name}, worst draw", **series_style(2 * index + 1)),
            Line2D([], [], label=f"{name}, {evidence_label(TOP_K, stage.served_k)}, mean",
                   **{**series_style(MAX_LINES + index), "linestyle": "none"}),
        ]
    figure.legend(handles=handles, loc="outside upper center", ncols=max(len(drawn), 1))


def plot_convergence(stages: Sequence[ConvergenceStage]) -> Figure:
    """Figure 4: how the values from the first k alternatives approach the values from logits as k grows.

    Three panels in one row in the academic style, labelled (a) to (c) above
    their plot areas, with one legend above them and one x label below:
    Min-K%++, the recall estimate, and the probability mass covered. Per
    stage one line joins the means and one the worst draw; the served mean is
    a separate marker. ``stages`` are plain data read from the stored tables
    (:func:`convergence_stages`); missing stages are not drawn. Raises
    :class:`FigureError` when every stage is missing. Returns a matplotlib
    Figure (no pyplot, no global state).
    """
    import matplotlib
    from matplotlib.figure import Figure

    stages = _checked_stages(stages)
    drawn = [stage for stage in stages if stage.status == DRAWN]
    if not drawn:
        raise FigureError("nothing to draw: every convergence table is missing (see the table)")
    ks = sorted({row.k for stage in drawn for quantity in PLOTTED for row in _plotted_points(stage, quantity)}
                | {stage.served_k for stage in drawn if stage.served_k is not None})
    with matplotlib.rc_context(figure_rc()):
        figure = Figure(figsize=_SIZE_IN, layout="constrained")
        cells = list(figure.subplots(1, len(PLOTTED), squeeze=False).ravel())
        for index, (axes, quantity) in enumerate(zip(cells, PLOTTED, strict=True)):
            _draw_panel(axes, quantity, drawn, ks)
            label_panel(axes, index)
        figure.supxlabel("alternatives kept per token, k (count, log scale)")
        _legend(figure, drawn)
        lay_out(figure)
    return figure


# ---------- outputs ----------

def render_convergence(plan: StudyPlan, *, arm: str = DEFAULT_CONVERGENCE_ARM,
                       stages: Sequence[str] = CONVERGENCE_STAGES, analysis_dir: str | Path = CONVERGENCE_DIR,
                       out_dir: str | Path = FIGURES_DIR) -> tuple[Path, ...]:
    """Render F4 from the stored convergence tables of ``plan`` and write its three files once."""
    found = convergence_stages(plan, arm=arm, stages=stages, analysis_dir=analysis_dir)
    texts = figure_texts(CONVERGENCE, convergence_table(found, plan), lambda: plot_convergence(found), out_dir)
    return write_outputs(texts)


def main(argv: Sequence[str] | None = None) -> int:
    """Render the convergence figure of the frozen plan from its stored convergence tables.

    Exit codes: 0 written (or already written identically), 2 refused (another
    plan, a malformed table, tables of different weights, nothing to draw, or
    a different figure at the output path).
    """
    parser = argparse.ArgumentParser(description="Convergence figure, from the stored convergence tables; sends no "
                                                 "request.")
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
    "PLOTTED",
    "STAGE_NAMES",
    "AgreementRow",
    "ConvergenceStage",
    "MassRow",
    "convergence_stage",
    "convergence_stages",
    "convergence_table",
    "is_plotted",
    "main",
    "plot_convergence",
    "read_convergence_stage",
    "render_convergence",
]


if __name__ == "__main__":
    raise SystemExit(main())
