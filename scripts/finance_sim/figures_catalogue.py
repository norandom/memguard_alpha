"""Figures 5 to 7 of the research record: the three analyses behind the catalogue (Requirements 5.6 and 8.1 to 8.6).

One figure per analysis, each drawn from the RESULT OBJECT of that analysis:

* Figure 5, ``mode-comparison``: the held-out AUC of the logprobs and the
  logits mode with its prompt-clustered interval, from a
  :class:`~scripts.finance_sim.mode_comparison.ModeComparison`. One panel; a
  dotted line at chance (0.5) and a dashed line at the minimum of the
  qualification rule, both explained in the caption.
* Figure 6, ``cutoff-check``: the mean recall estimate for analysis months
  inside the training range and after the cutoff, per mode (panels (a) and
  (b)) and prompt kind (marker and line style), from a
  :class:`~scripts.finance_sim.cutoff_check.CutoffCheckResult`.
* Figure 7, ``repeat-baseline``: the number of prompts by the number of
  different answers the same prompt returned, one panel per arm, from a
  :class:`~scripts.finance_sim.repeat_baseline.RepeatBaseline`.

This module computes no statistic. It plots the values and intervals the
analysis already reports and writes the same records to the table, so a figure
and its table cannot disagree. A statistic without a value is not drawn and is
listed in the table with the input that is missing; an analysis with nothing to
plot gives empty panels and a caption that says why.

Style: the one academic style of :mod:`scripts.finance_sim.figure_style`. Both
axes carry the quantity and its unit, nothing is written inside a plot area
(panels carry "(a)" and "(b)" above it), and the one message of a figure is in
its numbered caption, with the sample size and what the interval is.

Each figure is three files in the output directory: ``<name>.svg``,
``<name>.csv`` (one row per reported statistic, the plotted ones marked, with
the interval bounds and n at full precision) and ``<name>.md`` (the caption,
the same table with numbers rounded to four significant digits, a line that
says so and names the CSV, and the derived-statistics line). They are written
once and never replaced by different ones. Derived statistics only.

Numbers in text: a caption shows every real number to three decimals through
one function, and no rounding, in a caption or in the markdown table, turns a
nonzero value into zero or changes its sign; the caption and the rounding note
state that precision in the same sentence. A plotted interval with equal
bounds is named in the caption, so a bar without length is not read as a
missing interval. A caption is built from the data of the result object, never
from the wording of its reason; the full reason is a note under the table.

Reasons: why a value is missing, why a comparison is refused and why it gave
its answer are stored in the working vocabulary of the study. Every one of
them is published through
:func:`~scripts.finance_sim.published_text.published_reason`, in the table, in
the notes and in the caption; a reason without a published wording is refused.
The sources of the prompts are cited once under each table.

What an interval is: the words of a caption follow the method the analysis
states on the interval itself, the block length in the inputs of the statistic
and its independence flag. A method without wording here, or a block length
the statistics of a cell do not agree on, is refused with ``FigureError``, so
a change in an analysis never leaves a caption that describes another interval.
"""

from __future__ import annotations

import csv
import io
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from recall_guard.core.catalogue import MIN_AUC
from recall_guard.core.significance import BLOCK_BOOTSTRAP_METHOD, CLUSTERED_AUC_METHOD, PAIRED_AUC_METHOD
from scripts.finance_sim.calibrate import MACRO
from scripts.finance_sim.cutoff_check import (
    CATALOGUE_MODES,
    DIFFERENCE_INTERVAL_METHOD,
    MODES,
    PROMPT_KINDS,
    RULE_PROMPT,
    CutoffCell,
    CutoffCheckResult,
)
from scripts.finance_sim.figure_style import FIGURE_WIDTH_IN, caption, label_panel, panel_label, series_style
from scripts.finance_sim.figures import (
    DERIVED_ONLY,
    EDGAR_SOURCE,
    FIGURES_DIR,
    FMP_STUDY_SOURCE,
    FRED_SOURCE,
    FigureError,
    figure_rc,
    figure_svg,
    lay_out,
    write_outputs,
)
from scripts.finance_sim.limits import ARMS
from scripts.finance_sim.mode_comparison import IMPROVE, LOGITS, LOGPROBS, UNDECIDED, ModeComparison
from scripts.finance_sim.published_text import published_reason
from scripts.finance_sim.repeat_baseline import (
    ANALYSED,
    ONE_ANSWER_SHARE,
    WILSON_METHOD,
    RepeatBaseline,
    distribution_name,
    route_text,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from recall_guard.core.significance import Statistic

MODE_COMPARISON, CUTOFF_CHECK, REPEAT_BASELINE = "mode-comparison", "cutoff-check", "repeat-baseline"
#: The figures of this module, in the order the record shows them.
CATALOGUE_FIGURES: tuple[str, ...] = (MODE_COMPARISON, CUTOFF_CHECK, REPEAT_BASELINE)
#: The number each caption starts with ("Figure N."); 1 to 4 are the figures of the worked example.
FIGURE_NUMBERS: Mapping[str, int] = MappingProxyType({MODE_COMPARISON: 5, CUTOFF_CHECK: 6, REPEAT_BASELINE: 7})
#: The columns of every table: where the mark stands, the statistic as the analysis reports it, whether it is drawn.
COLUMNS: tuple[str, ...] = ("panel", "series", "x", "statistic", "value", "interval_low", "interval_high",
                            "confidence", "p_value", "n", "unit", "plotted", "missing")

#: The decimals of every real number in a caption, and the significant digits of a number in the markdown table.
CAPTION_DECIMALS = 3
ROUNDED_DIGITS = 4
#: The precision of the statistics in a caption, said once at its end and again in the rounding note of the markdown.
CAPTION_PRECISION = ("Statistics in the caption are shown to three decimals; a nonzero value that would round to "
                     "zero is shown with two significant digits.")

#: What a caption calls an interval, by the method the analysis states on it. Both AUC methods resample whole prompts.
_AUC_INTERVALS: Mapping[str, str] = MappingProxyType({CLUSTERED_AUC_METHOD: "prompt-clustered bootstrap",
                                                      PAIRED_AUC_METHOD: "prompt-clustered bootstrap"})
_SHARE_INTERVALS: Mapping[str, str] = MappingProxyType({WILSON_METHOD: "Wilson score"})
#: The interval method of each statistic of a cutoff cell; the caption words it by the block length that was used.
_CUTOFF_METHODS: Mapping[str, str] = MappingProxyType({"mean_inside": BLOCK_BOOTSTRAP_METHOD,
                                                       "mean_after": BLOCK_BOOTSTRAP_METHOD,
                                                       "difference": DIFFERENCE_INTERVAL_METHOD})

_MODES: tuple[str, ...] = (LOGPROBS, LOGITS)
_GROUPS: tuple[str, ...] = ("inside", "after")
_AGREEMENT = ("estimate_agreement_mean_difference", "estimate_agreement_lower_limit_of_agreement",
              "estimate_agreement_upper_limit_of_agreement")
_ONE_PANEL_SIZE_IN = (3.6, 2.9)
_TWO_PANEL_SIZE_IN = (FIGURE_WIDTH_IN, 3.1)
_REFERENCE_GREY = "0.4"
_REFERENCE_PT = 0.7
_BAR_GREY = "0.8"
_BAR_WIDTH = 0.6
#: Half the distance between the two prompt kinds of the cutoff figure, so their interval bars do not overlap.
_KIND_DODGE = 0.06


# ---------- the table ----------

def _flat(text: object) -> str:
    return " ".join(str(text).split())


def _text(value: object, *, exact: bool) -> str:
    """A table cell: empty when missing, yes or no, integers as they are, reals in full (CSV) or rounded."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        if exact:
            return repr(value)
        # Significant digits keep a nonzero value nonzero and its sign; zero is written without a sign.
        return "0" if value == 0 else f"{value:.{ROUNDED_DIGITS}g}"
    return _flat(value)


def _rounding_note(name: str) -> str:
    return (f"{CAPTION_PRECISION} Numbers in the table are rounded to {ROUNDED_DIGITS} significant digits; the exact "
            f"values are in `{name}.csv`.")


@dataclass(frozen=True)
class CatalogueTable:
    """The table view of one figure: one row per reported statistic, the caption and the notes below the table."""

    name: str
    caption: str
    rows: tuple[tuple[object, ...], ...]
    notes: tuple[str, ...] = ()
    sources: tuple[str, ...] = ()

    def to_csv(self) -> str:
        """The rows with every number at full precision, so the file holds the values of the analysis exactly."""
        buffer = io.StringIO()
        writer = csv.writer(buffer, lineterminator="\n")
        writer.writerow(COLUMNS)
        writer.writerows([_text(value, exact=True) for value in row] for row in self.rows)
        return buffer.getvalue()

    def to_markdown(self) -> str:
        """The caption, the rows as a markdown table with rounded numbers, the rounding, the notes, derived-only."""

        def line(cells: tuple[object, ...]) -> str:
            return "| " + " | ".join(_text(cell, exact=False).replace("|", "\\|") for cell in cells) + " |"

        lines = [self.caption, "", line(COLUMNS), "|" + "|".join("---" for _ in COLUMNS) + "|"]
        lines += [line(row) for row in self.rows]
        sources = [f"Sources: {'; '.join(self.sources)}."] if self.sources else []
        return "\n".join([*lines, "", _rounding_note(self.name), *self.notes, *sources, DERIVED_ONLY]) + "\n"


def _row(panel: str, series: str, x: object, statistic: Statistic, *, plotted: bool) -> tuple[object, ...]:
    """The table row of ``statistic`` as the analysis reports it; a statistic without a value is never plotted."""
    interval, test = statistic.interval, statistic.test
    return (panel, series, x, statistic.name, statistic.value,
            None if interval is None else interval.lo, None if interval is None else interval.hi,
            None if interval is None else interval.confidence, None if test is None else test.p_value,
            statistic.n, statistic.unit, plotted and statistic.value is not None,
            None if statistic.missing is None else published_reason(statistic.missing))


def _percent(confidence: float) -> str:
    return f"{confidence * 100:g} percent"


def _number(value: float, *, signed: bool = False) -> str:
    """Every real number of a caption: three decimals, and a rounding never hides or changes a sign.

    A nonzero value that three decimals would round to zero is written with two
    significant digits (-0.0003297 is "-0.00033"), so a reader still sees which
    side of zero it lies on. ``signed`` puts a plus sign before a positive
    difference. Zero has no sign and is "0.000".
    """
    if value == 0:
        return "0.000"
    sign = "+" if signed else "-"
    text = f"{value:{sign}.{CAPTION_DECIMALS}f}"
    if float(text) != 0:
        return text
    exponent = int(f"{abs(value):.1e}".split("e")[1])  # of the value as rounded to two significant digits
    return f"{value:{sign}.{1 - exponent}f}"


def _bounds(statistic: Statistic, *, signed: bool = False, word: bool = False) -> str:
    """The interval of ``statistic`` as "lo to hi", after the word "interval" where the caption first shows one."""
    interval = statistic.interval
    if interval is None:
        return "no interval"
    return (f"{'interval ' if word else ''}{_number(interval.lo, signed=signed)} to "
            f"{_number(interval.hi, signed=signed)}")


def _no_length(plotted: list[tuple[str, Statistic]]) -> list[str]:
    """One caption sentence per plotted interval with equal bounds, so a bar without length is not read as missing."""
    return [f"The {label} interval is {_bounds(statistic)}, so its bar has no length."
            for label, statistic in plotted
            if statistic.value is not None and statistic.interval is not None
            and statistic.interval.lo == statistic.interval.hi]


def _interval_wording(statistics: list[Statistic], wording: Mapping[str, str], what: str) -> str | None:
    """What the caption calls the intervals of ``statistics``, by the method the analysis states; None without one.

    Raises:
        FigureError: a method this module has no wording for, or intervals that need different words.
    """
    methods = sorted({statistic.interval.method for statistic in statistics if statistic.interval is not None})
    unknown = [method for method in methods if method not in wording]
    if unknown:
        raise FigureError(f"the caption of {what} has no wording for the interval method {unknown[0]!r}; it "
                          f"describes {sorted(wording)}")
    words = sorted({wording[method] for method in methods})
    if len(words) > 1:
        raise FigureError(f"the intervals of {what} are of different kinds ({words}); the caption names one")
    return words[0] if words else None


def _probability_axis(axes: Axes) -> None:
    """A y axis for a probability: ticks from 0 to 1, with room above 1 so a mark at 1 is drawn whole."""
    axes.set_ylim(0.0, 1.05)
    axes.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])


def _reason(text: object) -> str:
    """A stored reason in its published wording, as part of a caption sentence or of a note."""
    return published_reason(text)


def _sources(kind: str) -> tuple[str, ...]:
    """The citations under a table: where the prompts of ``kind`` come from, and that no price enters the figure."""
    return (FRED_SOURCE if kind == MACRO else EDGAR_SOURCE, FMP_STUDY_SOURCE)


def _checked(result: object, expected: type, what: str) -> Any:
    if not isinstance(result, expected):
        raise TypeError(f"{what} must be a {expected.__name__}; got {type(result).__name__}")
    return result


def _draw_interval(axes: Axes, x: float, statistic: Statistic, colour: object, gid: str) -> None:
    """The interval of ``statistic`` as the analysis reports it, capped at both ends; nothing without one."""
    if statistic.interval is not None:
        axes.plot([x, x], [statistic.interval.lo, statistic.interval.hi], color=colour, linestyle="-", marker="_",
                  markeredgecolor=colour, zorder=2, gid=gid)


# ---------- Figure 5: logprobs against logits ----------

def _comparison_statistics(comparison: ModeComparison) -> dict[str, Statistic]:
    return {statistic.name: statistic for statistic in comparison.statistics}


def _mode_comparison_caption(comparison: ModeComparison) -> str:
    number = FIGURE_NUMBERS[MODE_COMPARISON]
    if comparison.refusal is not None:
        return caption(number, "The comparison of logits with logprobs is refused and nothing is plotted: "
                               f"{_reason(comparison.refusal)}", n="0 held-out prompts", interval=None)
    statistics = _comparison_statistics(comparison)
    aucs = {mode: statistics[f"holdout_auc_{mode}"] for mode in _MODES}
    drawn = {mode: statistic for mode, statistic in aucs.items() if statistic.value is not None}
    levels = []
    for mode in _MODES:
        if mode in drawn:
            levels.append(f"{mode} {_number(aucs[mode].value)} ({_bounds(aucs[mode], word=not levels)})")
        else:
            levels.append(f"{mode} not computed")
    paired = statistics["paired_auc_difference"]
    if paired.value is None:
        difference = "The paired difference, logits minus logprobs, cannot be computed"
    else:
        difference = (f"The paired difference, logits minus logprobs, is {_number(paired.value, signed=True)} "
                      f"({_bounds(paired, signed=True)})")
    cause = _comparison_cause(comparison, paired)
    message = (f"Held-out AUC of the recall estimate by evidence mode, on the same weights and prompts: "
               f"{', '.join(levels)}. {difference}; the analysis answers {comparison.answer}"
               f"{'' if cause is None else f', because {cause}'}")

    def size(statistic: Statistic) -> str:
        return f"{statistic.n} held-out prompts ({statistic.inputs['n_rows']} draws)"

    if not drawn:
        n = "0 held-out prompts"
    elif len({size(statistic) for statistic in drawn.values()}) == 1:
        n = size(next(iter(drawn.values())))
    else:
        n = " and ".join(f"{size(statistic)} with {mode}" for mode, statistic in drawn.items())
    notes = []
    shown = [statistic for statistic in (*drawn.values(), paired) if statistic.value is not None]
    confidences = {statistic.interval.confidence for statistic in shown if statistic.interval is not None}
    kind = _interval_wording(shown, _AUC_INTERVALS, "the mode comparison")
    if confidences:
        notes.append(f"Intervals are {_percent(max(confidences))} {kind} intervals.")
    notes.append(f"The dotted line marks chance (0.5), the dashed line the rule's minimum ({MIN_AUC:g}).")
    notes += [f"No mark is drawn for {mode}: {_reason(aucs[mode].missing)}." for mode in _MODES if mode not in drawn]
    notes += _no_length(list(drawn.items()))
    if shown:
        notes.append(CAPTION_PRECISION)
    return caption(number, message, n=n, interval=" ".join(notes))


def _comparison_cause(comparison: ModeComparison, paired: Statistic) -> str | None:
    """The one short cause of the answer that the comparison holds as data, or None; the reason text is never read.

    Only the causes that follow from the answer, the supplied cutoff result and
    the interval of the paired difference are stated. Any other answer keeps
    its full reason in the note under the table.
    """
    if paired.value is None or paired.interval is None or paired.interval.lo <= 0.0:
        return None
    if comparison.answer == IMPROVE and comparison.cutoff_check is True:
        return "the difference interval excludes 0 and the cutoff check for logits passed"
    if comparison.answer != UNDECIDED or comparison.kind != MACRO:
        return None
    if comparison.cutoff_check is None:
        return "no cutoff check for logits was supplied"
    if comparison.cutoff_check is False:
        return "the cutoff check for logits did not pass"
    return None


def mode_comparison_table(comparison: ModeComparison) -> CatalogueTable:
    """Figure 5's table and caption: the two held-out AUCs (plotted), the paired difference and the agreement."""
    comparison = _checked(comparison, ModeComparison, "comparison")
    if comparison.refusal is not None:
        rows = tuple(("", "", mode, f"holdout_auc_{mode}", None, None, None, None, None, 0, "prompt", False,
                      _reason(comparison.refusal)) for mode in _MODES)
        return CatalogueTable(MODE_COMPARISON, _mode_comparison_caption(comparison), rows,
                              ("The comparison is refused; no value is compared.",), _sources(comparison.kind))
    statistics = _comparison_statistics(comparison)
    rows = [_row("", "", mode, statistics[f"holdout_auc_{mode}"], plotted=True) for mode in _MODES]
    rows += [_row("", "", "", statistics[name], plotted=False) for name in ("paired_auc_difference", *_AGREEMENT)]
    note = f"Answer of the comparison: {comparison.answer}. Reason: {_reason(comparison.reason)}."
    return CatalogueTable(MODE_COMPARISON, _mode_comparison_caption(comparison), tuple(rows), (note,),
                          _sources(comparison.kind))


def plot_mode_comparison(comparison: ModeComparison) -> Figure:
    """Figure 5: the held-out AUC of each mode with its interval bar; a refused comparison gives an empty panel."""
    import matplotlib
    from matplotlib.figure import Figure

    comparison = _checked(comparison, ModeComparison, "comparison")
    statistics = _comparison_statistics(comparison)
    style = {**series_style(0), "linestyle": "none"}
    with matplotlib.rc_context(figure_rc()):
        figure = Figure(figsize=_ONE_PANEL_SIZE_IN, layout="constrained")
        axes = figure.subplots()
        axes.axhline(0.5, color=_REFERENCE_GREY, linewidth=_REFERENCE_PT, linestyle=":", zorder=1,
                     gid="reference-chance")
        axes.axhline(MIN_AUC, color=_REFERENCE_GREY, linewidth=_REFERENCE_PT, linestyle="--", zorder=1,
                     gid="reference-minimum")
        for place, mode in enumerate(_MODES):
            statistic = statistics.get(f"holdout_auc_{mode}")
            if statistic is None or statistic.value is None:
                continue
            _draw_interval(axes, place, statistic, style["color"], f"auc-interval-{mode}")
            axes.plot([place], [statistic.value], zorder=3, gid=f"auc-{mode}", **style)
        axes.set_xlim(-0.6, len(_MODES) - 0.4)
        axes.set_xticks(range(len(_MODES)), labels=list(_MODES))
        _probability_axis(axes)
        axes.set_xlabel("Evidence mode")
        axes.set_ylabel("Held-out AUC (probability)")
        lay_out(figure)
    return figure


# ---------- Figure 6: inside the training range against after the cutoff ----------

def _cutoff_panel(index: int, mode: str) -> str:
    return f"{panel_label(index)} {CATALOGUE_MODES[mode]}"


def _block_length(cell: CutoffCell) -> int:
    """The block length the intervals of a computed ``cell`` were resampled with, read from its statistics.

    Raises:
        FigureError: an interval method the caption has no wording for, or statistics that do not state one block
            length of at least 1.
    """
    where = f"the {CATALOGUE_MODES[cell.mode]} mode under the {cell.prompt_kind} prompt"
    lengths = set()
    for name, method in _CUTOFF_METHODS.items():
        statistic = cell.statistics[name]
        if statistic.interval is not None and statistic.interval.method != method:
            raise FigureError(f"the caption of the cutoff check has no wording for the interval method "
                              f"{statistic.interval.method!r} of {name} in {where}; it describes {method!r}")
        lengths.add(statistic.inputs.get("block_length"))
    (length,) = lengths if len(lengths) == 1 else (None,)
    if isinstance(length, bool) or not isinstance(length, int) or length < 1:
        raise FigureError(f"the statistics of {where} do not state one block length of at least 1; their inputs "
                          f"give {sorted(lengths, key=repr)}")
    return length


def _cutoff_intervals(check: CutoffCheckResult, plotted: list[CutoffCell], confidence: str) -> list[str]:
    """What the intervals of the plotted cells are, from the block length each cell used and the plan's.

    At block length 1 the months are resampled singly, so the caption says that
    the interval allows for no serial dependence, and why the plan's length was
    not used when a group has fewer months than it. Cells that differ are named
    per mode, or per mode and prompt kind where the kinds of one mode differ.
    """
    planned = check.to_dict()["settings"]["block_length"]

    def used(cell: CutoffCell) -> tuple[int, bool]:
        return _block_length(cell), min(cell.summary.n_inside, cell.summary.n_after) < planned

    def singly(fewer: bool) -> str:
        why = f", because a group has fewer months than the plan's block length of {planned}" if fewer else ""
        return f"resampled singly (block length 1{why})"

    by_cell = [(cell, used(cell)) for cell in plotted]
    if len({found for _, found in by_cell}) == 1:
        length, fewer = by_cell[0][1]
        if length == 1:
            return [f"Intervals are {confidence} bootstrap intervals over months, {singly(fewer)}, so they do not "
                    "allow for serial dependence between months."]
        return [f"Intervals are {confidence} moving-block bootstrap intervals over months with block length {length}."]
    notes = [f"Intervals are {confidence} bootstrap intervals over months."]
    for mode in MODES:
        cells = [(cell, found) for cell, found in by_cell if cell.mode == mode]
        one = len({found for _, found in cells}) == 1
        for cell, (length, fewer) in cells[:1] if one else cells:
            where = f"In the {CATALOGUE_MODES[mode]} mode" + ("" if one else f" under the {cell.prompt_kind} prompt")
            if length == 1:
                notes.append(f"{where} months are {singly(fewer)}, so the intervals do not allow for serial "
                             "dependence between months.")
            else:
                notes.append(f"{where} they are moving-block intervals with block length {length}.")
    return notes


def _cutoff_caption(check: CutoffCheckResult) -> str:
    number = FIGURE_NUMBERS[CUTOFF_CHECK]
    rule = {mode: check.cell(mode, RULE_PROMPT) for mode in MODES}
    computed = {mode: cell for mode, cell in rule.items() if cell.summary.passes is not None}
    # The message is about the prompt the rule reads; the months of the other prompt kind are in the table.
    n = f"months under the {RULE_PROMPT} prompt: " + ", ".join(
        f"{CATALOGUE_MODES[mode]} {cell.summary.n_inside} inside and {cell.summary.n_after} after"
        for mode, cell in rule.items())
    plotted = [cell for cell in check.cells if cell.statistics["mean_inside"].value is not None]
    if not plotted:
        why = _reason(rule[MODES[0]].statistics["difference"].missing)
        return caption(number, f"The cutoff check cannot be computed and nothing is plotted: {why}", n=n,
                       interval=None)
    start = f"Under the {RULE_PROMPT} prompt the mean recall estimate is "
    than = "higher for analysis months inside the training range than after the cutoff"
    passing = [CATALOGUE_MODES[mode] for mode, cell in computed.items() if cell.summary.passes]
    if not computed:
        lead = f"{start}missing in both modes"
    elif len(passing) == len(computed):
        lead = f"{start}{than}"
    elif not passing:
        lead = f"{start}not shown to be {than}"
    else:
        lead = f"{start}{than} in the {' and the '.join(passing)} mode only"
    details = []
    for mode, cell in computed.items():
        difference = cell.statistics["difference"]
        details.append(f"{CATALOGUE_MODES[mode]} difference {_number(cell.summary.difference, signed=True)} "
                       f"({_bounds(difference, signed=True, word=not details)}, one-sided p "
                       f"{_number(cell.summary.p_value)})")
    message = f"{lead}: {', '.join(details)}" if details else lead
    means = [(f"{CATALOGUE_MODES[cell.mode]} {cell.prompt_kind} {group}", cell.statistics[f"mean_{group}"])
             for cell in plotted for group in _GROUPS]
    confidences = {statistic.interval.confidence for _, statistic in means if statistic.interval is not None}
    notes = []
    if confidences:
        notes += _cutoff_intervals(check, plotted, _percent(max(confidences)))
    notes.append(f"Panel (a) is the {CATALOGUE_MODES[MODES[0]]} mode and panel (b) the {CATALOGUE_MODES[MODES[1]]} "
                 "mode; marker and line style give the prompt kind.")
    others = [kind for kind in PROMPT_KINDS
              if kind != RULE_PROMPT and any(cell.prompt_kind == kind for cell in plotted)]
    if others:
        notes.append(f"The {' and the '.join(others)} prompt is drawn for comparison; its numbers are in the table.")
    for index, mode in enumerate(MODES):
        if not any(cell.mode == mode for cell in plotted):
            why = _reason(rule[mode].statistics["difference"].missing)
            notes.append(f"Panel {panel_label(index)} is empty: {why}.")
    flat = _no_length(means)
    notes += flat
    if details or flat:
        notes.append(CAPTION_PRECISION)
    return caption(number, message, n=n, interval=" ".join(notes))


def cutoff_check_table(check: CutoffCheckResult) -> CatalogueTable:
    """Figure 6's table and caption: per mode and prompt kind the two means (plotted) and their difference."""
    check = _checked(check, CutoffCheckResult, "check")
    rows = []
    for index, mode in enumerate(MODES):
        for kind in PROMPT_KINDS:
            cell = check.cell(mode, kind)
            rows += [_row(_cutoff_panel(index, mode), kind, group, cell.statistics[f"mean_{group}"], plotted=True)
                     for group in _GROUPS]
            rows.append(_row(_cutoff_panel(index, mode), kind, "inside minus after", cell.statistics["difference"],
                             plotted=False))
    note = (f"The qualification rule reads the {RULE_PROMPT} prompt; the p-value of a difference is one-sided "
            "(inside above after).")
    return CatalogueTable(CUTOFF_CHECK, _cutoff_caption(check), tuple(rows), (note,), _sources(MACRO))


def plot_cutoff_check(check: CutoffCheckResult) -> Figure:
    """Figure 6: the mean estimate inside and after, per mode (panel) and prompt kind (marker and line style)."""
    import matplotlib
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D

    check = _checked(check, CutoffCheckResult, "check")
    with matplotlib.rc_context(figure_rc()):
        figure = Figure(figsize=_TWO_PANEL_SIZE_IN, layout="constrained")
        panels = figure.subplots(1, len(MODES), sharey=True)
        drawn: set[int] = set()
        for index, (axes, mode) in enumerate(zip(panels, MODES, strict=True)):
            label_panel(axes, index)
            for series, kind in enumerate(PROMPT_KINDS):
                found = [check.cell(mode, kind).statistics[f"mean_{group}"] for group in _GROUPS]
                if any(statistic.value is None for statistic in found):
                    continue
                style = series_style(series)
                places = [place + (2 * series - 1) * _KIND_DODGE for place in range(len(_GROUPS))]
                for place, group, statistic in zip(places, _GROUPS, found, strict=True):
                    _draw_interval(axes, place, statistic, style["color"], f"mean-interval-{mode}-{kind}-{group}")
                axes.plot(places, [statistic.value for statistic in found], zorder=3, gid=f"mean-{mode}-{kind}",
                          **style)
                drawn.add(series)
            axes.set_xlim(-0.5, len(_GROUPS) - 0.5)
            axes.set_xticks(range(len(_GROUPS)), labels=list(_GROUPS))
            _probability_axis(axes)
            axes.tick_params(axis="y", labelleft=True)
            axes.set_xlabel("Analysis months relative to the training cutoff")
            axes.set_ylabel("Mean recall estimate (probability)")
        if drawn:
            handles = [Line2D([], [], label=f"{kind} prompt", **series_style(series))
                       for series, kind in enumerate(PROMPT_KINDS) if series in drawn]
            figure.legend(handles=handles, loc="outside lower center", ncols=len(handles))
        lay_out(figure)
    return figure


# ---------- Figure 7: different answers on a repeated prompt ----------

def _distribution(baseline: RepeatBaseline, arm: str) -> list[tuple[int, Statistic]]:
    """The counts of prompts with 1, 2, 3, ... different answers of ``arm``, as the analysis reports them."""
    by_name = {statistic.name: statistic for statistic in baseline.statistics[arm]}
    found, answers = [], 1
    while distribution_name(answers) in by_name:
        found.append((answers, by_name[distribution_name(answers)]))
        answers += 1
    return found


def _route(header: Mapping[str, Any]) -> str:
    route = header["route"]
    return route_text(route["profile_id"], route["provider"], route["served_model"])


def _repeat_caption(baseline: RepeatBaseline) -> str:
    headers = baseline.to_dict()["arms"]
    parts, sizes, confidences, shares, dependent = [], [], set(), [], []
    for index, arm in enumerate(ARMS):
        share = baseline.statistic(arm, ONE_ANSWER_SHARE)
        where = f"panel {panel_label(index)}, route {_route(headers[arm])} at temperature " \
                f"{headers[arm]['temperature']:g}"
        if share.value is None:
            parts.append(f"{where}, is empty: {_reason(share.missing)}")
            sizes.append(f"0 prompts in panel {panel_label(index)}")
            continue
        one = baseline.statistic(arm, distribution_name(1)).value
        shares.append(share)
        if share.interval is not None:
            confidences.add(share.interval.confidence)
            if not share.independent_units:
                dependent.append(panel_label(index))
        # The prompts left out and why are rows of the table; the caption keeps what the bars show.
        parts.append(f"{where}: {one} of {share.n} prompts returned one answer (share {_number(share.value)}, "
                     f"{_bounds(share, word=True)})")
        sizes.append(f"{baseline.statistic(arm, ANALYSED).value} prompts in panel {panel_label(index)}")
    message = ("Number of different answers the same prompt returned when it was asked again, by serving route and "
               "temperature: " + "; ".join(parts))
    kind = _interval_wording(shares, _SHARE_INTERVALS, "the repeat baseline")
    notes = []
    if confidences:
        # The interval takes prompts as independent; where the analysis marks them as not, the caption says so.
        if not dependent:
            narrow = ""
        else:
            everywhere = len(dependent) == sum(share.interval is not None for share in shares)
            where = "" if everywhere else f"in panel{'s' if len(dependent) > 1 else ''} {' and '.join(dependent)} "
            narrow = (f"; {where}the identifying and the anonymised prompt of one month share the month, so the "
                      "prompts are not independent and the interval may be too narrow")
        notes.append("The bars are counts and carry no interval; the interval of a share is a "
                     f"{_percent(max(confidences))} {kind} interval{narrow}.")
    if shares:
        notes.append(CAPTION_PRECISION)
    return caption(FIGURE_NUMBERS[REPEAT_BASELINE], message, n=" and ".join(sizes),
                   interval=" ".join(notes) if notes else None)


def repeat_baseline_table(baseline: RepeatBaseline) -> CatalogueTable:
    """Figure 7's table and caption: every statistic of both arms; the counts by different answers are plotted."""
    baseline = _checked(baseline, RepeatBaseline, "baseline")
    headers = baseline.to_dict()["arms"]
    rows = []
    for index, arm in enumerate(ARMS):
        panel = f"{panel_label(index)} route {_route(headers[arm])}, temperature {headers[arm]['temperature']:g}"
        answers = {statistic.name: count for count, statistic in _distribution(baseline, arm)}
        rows += [_row(panel, "", answers.get(statistic.name, ""), statistic, plotted=statistic.name in answers)
                 for statistic in baseline.statistics[arm]]
    note = ("x is the number of different answers of a plotted count. The identifying and the anonymised prompt of "
            "one month share the month, so prompts are not independent where both are counted.")
    return CatalogueTable(REPEAT_BASELINE, _repeat_caption(baseline), tuple(rows), (note,), _sources(MACRO))


def plot_repeat_baseline(baseline: RepeatBaseline) -> Figure:
    """Figure 7: per arm, the number of prompts by the number of different answers, as plain outlined bars."""
    import matplotlib
    from matplotlib.figure import Figure
    from matplotlib.ticker import MaxNLocator

    baseline = _checked(baseline, RepeatBaseline, "baseline")
    counts = {arm: [(answers, statistic.value) for answers, statistic in _distribution(baseline, arm)
                    if statistic.value is not None] for arm in ARMS}
    widest = max([answers for found in counts.values() for answers, _ in found], default=1)
    tallest = max([count for found in counts.values() for _, count in found], default=0)
    with matplotlib.rc_context(figure_rc()):
        figure = Figure(figsize=_TWO_PANEL_SIZE_IN, layout="constrained")
        panels = figure.subplots(1, len(ARMS))
        for index, (axes, arm) in enumerate(zip(panels, ARMS, strict=True)):
            label_panel(axes, index)
            for answers, count in counts[arm]:
                axes.bar(answers, count, width=_BAR_WIDTH, facecolor=_BAR_GREY, edgecolor="black", linewidth=0.8,
                         gid=f"answers-{arm}-{answers}")
            axes.set_xlim(1 - _BAR_WIDTH, widest + _BAR_WIDTH)
            axes.set_xticks(range(1, widest + 1))
            axes.set_ylim(0, max(1, tallest) * 1.08)
            axes.yaxis.set_major_locator(MaxNLocator(integer=True))
            axes.set_xlabel("Different answers among the draws\nof one prompt (count)")
            axes.set_ylabel("Prompts (count)")
        lay_out(figure)
    return figure


# ---------- the written outputs ----------

def catalogue_figure_paths(name: str, out_dir: str | Path = FIGURES_DIR) -> dict[str, Path]:
    """The three files of figure ``name`` in ``out_dir``: the SVG, the CSV and the markdown, in that order."""
    if name not in CATALOGUE_FIGURES:
        raise FigureError(f"unknown figure {name!r}; expected one of {list(CATALOGUE_FIGURES)}")
    out = Path(out_dir)
    return {"svg": out / f"{name}.svg", "csv": out / f"{name}.csv", "markdown": out / f"{name}.md"}


def catalogue_figure_texts(name: str, table: CatalogueTable, draw: Callable[[], Figure],
                           out_dir: str | Path = FIGURES_DIR) -> dict[Path, str]:
    """The three files of one figure as text by path: ``draw()`` saved as SVG, the table as CSV and markdown."""
    paths = catalogue_figure_paths(name, out_dir)
    return {paths["svg"]: figure_svg(draw()), paths["csv"]: table.to_csv(), paths["markdown"]: table.to_markdown()}


def render_catalogue_figures(mode_comparison: ModeComparison, cutoff_check: CutoffCheckResult,
                             repeat_baseline: RepeatBaseline, *,
                             out_dir: str | Path = FIGURES_DIR) -> tuple[Path, ...]:
    """Render Figures 5 to 7 from the results of their analyses and write them once.

    Every figure is built before any file is written, so a refusal leaves the
    output directory unchanged. Returns the written (or already identical)
    paths, per figure in the order of :func:`catalogue_figure_paths`.
    """
    figures = (
        (MODE_COMPARISON, mode_comparison_table(mode_comparison), lambda: plot_mode_comparison(mode_comparison)),
        (CUTOFF_CHECK, cutoff_check_table(cutoff_check), lambda: plot_cutoff_check(cutoff_check)),
        (REPEAT_BASELINE, repeat_baseline_table(repeat_baseline), lambda: plot_repeat_baseline(repeat_baseline)),
    )
    texts: dict[Path, str] = {}
    for name, table, draw in figures:
        texts.update(catalogue_figure_texts(name, table, draw, out_dir))
    return write_outputs(texts)


__all__ = [
    "CATALOGUE_FIGURES",
    "COLUMNS",
    "CUTOFF_CHECK",
    "FIGURE_NUMBERS",
    "MODE_COMPARISON",
    "REPEAT_BASELINE",
    "CatalogueTable",
    "catalogue_figure_paths",
    "catalogue_figure_texts",
    "cutoff_check_table",
    "mode_comparison_table",
    "plot_cutoff_check",
    "plot_mode_comparison",
    "plot_repeat_baseline",
    "render_catalogue_figures",
    "repeat_baseline_table",
]
