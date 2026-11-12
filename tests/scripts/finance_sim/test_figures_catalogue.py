"""Figures 5 to 7: the mode comparison, the cutoff check and the repeat baseline (qualified-model-catalogue 3.4).

Requirements 5.6 and 8.1 to 8.6. Every figure is built from the result object
of its analysis, through the analyses' own functions on the constructed inputs
of their tests: labelled axes, nothing written inside a plot area, one SVG with
its CSV and markdown, grey levels only, a numbered caption with the sample
size, a table that holds the values of the analysis exactly, interval bars
where the analysis reports an interval, and missing cells listed, not drawn.
"""

from __future__ import annotations

import csv
import io
import re
from dataclasses import replace

import matplotlib

matplotlib.use("Agg")

import pytest  # noqa: E402
from matplotlib.colors import to_rgb  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402

from recall_guard.core.catalogue import MIN_AUC  # noqa: E402
from recall_guard.core.significance import (  # noqa: E402
    BLOCK_BOOTSTRAP_METHOD,
    CLUSTERED_AUC_METHOD,
    PAIRED_AUC_METHOD,
)
from scripts.finance_sim import figures_catalogue  # noqa: E402
from scripts.finance_sim.cutoff_check import (  # noqa: E402
    CATALOGUE_MODES,
    DIFFERENCE_INTERVAL_METHOD,
    MODES,
    PROMPT_KINDS,
    cutoff_check,
)
from scripts.finance_sim.figures import DERIVED_ONLY, FigureError  # noqa: E402
from scripts.finance_sim.figures_catalogue import (  # noqa: E402
    CATALOGUE_FIGURES,
    COLUMNS,
    CUTOFF_CHECK,
    FIGURE_NUMBERS,
    MODE_COMPARISON,
    REPEAT_BASELINE,
    catalogue_figure_paths,
    catalogue_figure_texts,
    cutoff_check_table,
    mode_comparison_table,
    plot_cutoff_check,
    plot_mode_comparison,
    plot_repeat_baseline,
    render_catalogue_figures,
    repeat_baseline_table,
)
from scripts.finance_sim.limits import ARMS  # noqa: E402
from scripts.finance_sim.mode_comparison import IMPROVE, LOGITS, LOGPROBS, MATCH, UNDECIDED  # noqa: E402
from scripts.finance_sim.published_text import published_reason  # noqa: E402
from scripts.finance_sim.repeat_baseline import (  # noqa: E402
    ANALYSED,
    NO_DRAWS,
    ONE_ANSWER_SHARE,
    WILSON_METHOD,
    distribution_name,
)
from tests.scripts.finance_sim import test_cutoff_check as cutoff_tests  # noqa: E402
from tests.scripts.finance_sim import test_mode_comparison as mode_tests  # noqa: E402
from tests.scripts.finance_sim import test_repeat_baseline as repeat_tests  # noqa: E402

#: The module fixtures of the three analyses' tests, registered in this module under the same names.
recorded = mode_tests.recorded
sampling = cutoff_tests.sampling
serving = cutoff_tests.serving
baseline = repeat_tests.baseline

#: The one statement of the precision of a caption, the same in the three captions and in the rounding note.
PRECISION = ("Statistics in the caption are shown to three decimals; a nonzero value that would round to zero is "
             "shown with two significant digits.")
SINGLY = "resampled singly (block length 1, because a group has fewer months than the plan's block length of 6)"
NOT_INDEPENDENT = ("the identifying and the anonymised prompt of one month share the month, so the prompts are not "
                   "independent and the interval may be too narrow")

AGREEMENT = ("estimate_agreement_mean_difference", "estimate_agreement_lower_limit_of_agreement",
             "estimate_agreement_upper_limit_of_agreement")


@pytest.fixture(scope="module")
def comparison(recorded):
    return mode_tests.compare(recorded)


@pytest.fixture(scope="module")
def check(sampling):
    return sampling.check


@pytest.fixture(scope="module")
def drawn(comparison, check, baseline):
    """Per figure name: the result of the analysis, its table and its drawn figure."""
    return {
        MODE_COMPARISON: (comparison, mode_comparison_table(comparison), plot_mode_comparison(comparison)),
        CUTOFF_CHECK: (check, cutoff_check_table(check), plot_cutoff_check(check)),
        REPEAT_BASELINE: (baseline, repeat_baseline_table(baseline), plot_repeat_baseline(baseline)),
    }


def rows_of(table) -> list[dict[str, str]]:
    return list(csv.DictReader(io.StringIO(table.to_csv())))


def one_row(table, **where: str) -> dict[str, str]:
    found = [row for row in rows_of(table) if all(row[column] == value for column, value in where.items())]
    assert len(found) == 1, (where, found)
    return found[0]


def artists(figure, gid: str) -> list:
    return [artist for axes in figure.axes for artist in (*axes.lines, *axes.patches) if artist.get_gid() == gid]


def marks(figure) -> list:
    """Everything drawn in the plot areas except the reference lines."""
    return [artist for axes in figure.axes for artist in (*axes.lines, *axes.patches, *axes.collections)
            if not (artist.get_gid() or "").startswith("reference-")]


def assert_matches(row: dict[str, str], statistic, *, plotted: bool) -> None:
    """The CSV row holds the value, the interval bounds and n of ``statistic`` exactly."""
    assert row["statistic"] == statistic.name
    assert float(row["value"]) == statistic.value
    assert int(row["n"]) == statistic.n and row["unit"] == statistic.unit
    if statistic.interval is None:
        assert (row["interval_low"], row["interval_high"], row["confidence"]) == ("", "", "")
    else:
        assert float(row["interval_low"]) == statistic.interval.lo
        assert float(row["interval_high"]) == statistic.interval.hi
        assert float(row["confidence"]) == statistic.interval.confidence
    assert row["plotted"] == ("yes" if plotted else "no") and row["missing"] == ""


def is_grey(colour: object) -> bool:
    red, green, blue = to_rgb(colour)
    return red == green == blue


# ---------- the style of every figure (8.1, 8.2, 8.4, 8.6) ----------

def test_the_three_figures_are_numbered_five_to_seven():
    assert CATALOGUE_FIGURES == (MODE_COMPARISON, CUTOFF_CHECK, REPEAT_BASELINE)
    assert CATALOGUE_FIGURES == ("mode-comparison", "cutoff-check", "repeat-baseline")
    assert [FIGURE_NUMBERS[name] for name in CATALOGUE_FIGURES] == [5, 6, 7]


@pytest.mark.parametrize(("name", "x_label", "y_label"), [
    (MODE_COMPARISON, "Evidence mode", "Held-out AUC (probability)"),
    (CUTOFF_CHECK, "Analysis months relative to the training cutoff", "Mean recall estimate (probability)"),
    (REPEAT_BASELINE, "Different answers among the draws\nof one prompt (count)", "Prompts (count)"),
])
def test_both_axes_of_every_panel_are_labelled_with_quantity_and_unit(drawn, name, x_label, y_label):
    figure = drawn[name][2]
    assert len(figure.axes) == (1 if name == MODE_COMPARISON else 2)
    for axes in figure.axes:
        assert axes.get_xlabel() == x_label and axes.get_ylabel() == y_label
        assert axes.xaxis.label.get_visible() and axes.yaxis.label.get_visible()


@pytest.mark.parametrize("name", ["mode-comparison", "cutoff-check", "repeat-baseline"])
def test_no_title_and_no_text_inside_a_plot_area(drawn, name):
    figure = drawn[name][2]
    figure.draw_without_rendering()
    assert figure.get_suptitle() == "" and figure.texts == []
    for index, axes in enumerate(figure.axes):
        assert [axes.get_title(loc=place) for place in ("left", "center", "right")] == ["", "", ""]
        # The only text of a panel besides its axis and tick labels is its letter, above the plot area.
        letters = [] if len(figure.axes) == 1 else [f"({'ab'[index]})"]
        assert [text.get_text() for text in axes.texts] == letters
        for text in axes.texts:
            assert text.get_window_extent().y0 > axes.bbox.y1
        assert not axes.tables and not axes.images
    legends = [axes.get_legend() for axes in figure.axes if axes.get_legend() is not None] + list(figure.legends)
    if name == CUTOFF_CHECK:
        (legend,) = legends
        assert [text.get_text() for text in legend.get_texts()] == [f"{kind} prompt" for kind in PROMPT_KINDS]
        assert legend.get_title().get_text() == "" and not legend.get_frame_on()
    else:
        assert legends == []


@pytest.mark.parametrize("name", ["mode-comparison", "cutoff-check", "repeat-baseline"])
def test_every_colour_is_a_grey_level_and_glyphs_are_paths(drawn, name, tmp_path):
    _, table, figure = drawn[name]
    svg = catalogue_figure_texts(name, table, lambda: figure, tmp_path)[catalogue_figure_paths(name, tmp_path)["svg"]]
    colours = set(re.findall(r"#[0-9a-fA-F]{6}\b", svg))
    assert colours and all(is_grey(colour) for colour in colours), sorted(colours)
    assert "<text" not in svg


def test_the_repeat_baseline_bars_are_plain_rectangles_without_value_labels(drawn, baseline):
    figure = drawn[REPEAT_BASELINE][2]
    for letter, (arm, axes) in zip("ab", zip(ARMS, figure.axes, strict=True), strict=True):
        counts = {}
        for answers in range(1, 10):
            try:
                counts[answers] = baseline.statistic(arm, distribution_name(answers)).value
            except KeyError:
                break
        bars = {answers: artists(figure, f"answers-{arm}-{answers}") for answers in counts}
        assert all(len(found) == 1 for found in bars.values())
        assert sorted(patch.get_gid() for patch in axes.patches) == sorted(f"answers-{arm}-{k}" for k in counts)
        for answers, (patch,) in bars.items():
            assert type(patch) is Rectangle and not isinstance(patch, FancyBboxPatch)
            assert patch.get_height() == counts[answers]
            assert patch.get_x() + patch.get_width() / 2 == pytest.approx(answers)
            assert to_rgb(patch.get_edgecolor()) == (0.0, 0.0, 0.0)
            assert is_grey(patch.get_facecolor()) and 0.5 < to_rgb(patch.get_facecolor())[0] < 1.0
        assert [text.get_text() for text in axes.texts] == [f"({letter})"]
        for axis in (axes.xaxis, axes.yaxis):
            assert all(float(tick) == int(tick) for tick in axis.get_ticklocs())


def test_the_mode_comparison_has_reference_lines_at_chance_and_the_rule_minimum_explained_in_the_caption(drawn):
    _, table, figure = drawn[MODE_COMPARISON]
    (axes,) = figure.axes
    # The axis runs from 0 to 1 with a little room above, so a mark at 1 is drawn whole.
    assert axes.get_ylim() == (0.0, 1.05) and list(axes.get_yticks()) == [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    assert [tick.get_text() for tick in axes.get_xticklabels()] == [LOGPROBS, LOGITS]
    (chance,), (minimum,) = artists(figure, "reference-chance"), artists(figure, "reference-minimum")
    assert set(chance.get_ydata()) == {0.5} and set(minimum.get_ydata()) == {MIN_AUC}
    assert chance.get_linestyle() != minimum.get_linestyle()
    assert "chance (0.5)" in table.caption and f"({MIN_AUC:g})" in table.caption


def test_the_two_prompt_kinds_differ_by_marker_and_line_style(drawn):
    figure = drawn[CUTOFF_CHECK][2]
    for mode in MODES:
        (first,), (second,) = (artists(figure, f"mean-{mode}-{kind}") for kind in PROMPT_KINDS)
        assert first.get_marker() != second.get_marker() and first.get_linestyle() != second.get_linestyle()


# ---------- the files (8.3, design: one SVG with its CSV and markdown) ----------

def test_each_figure_is_one_svg_with_its_csv_and_markdown_and_no_themed_variant(comparison, check, baseline, tmp_path):
    written = render_catalogue_figures(comparison, check, baseline, out_dir=tmp_path)
    expected = [f"{name}.{suffix}" for name in CATALOGUE_FIGURES for suffix in ("svg", "csv", "md")]
    assert [path.name for path in written] == expected
    assert sorted(path.name for path in tmp_path.iterdir()) == sorted(expected)
    assert not any("light" in name or "dark" in name for name in expected)
    for name in CATALOGUE_FIGURES:
        paths = catalogue_figure_paths(name, tmp_path)
        assert paths["svg"].read_text(encoding="utf-8").startswith("<?xml")
        assert paths["csv"].read_text(encoding="utf-8").splitlines()[0] == ",".join(COLUMNS)


def test_rendering_twice_gives_identical_bytes_and_a_different_figure_is_never_replaced(comparison, check, baseline,
                                                                                         tmp_path):
    first = render_catalogue_figures(comparison, check, baseline, out_dir=tmp_path / "one")
    second = render_catalogue_figures(comparison, check, baseline, out_dir=tmp_path / "two")
    assert [path.read_bytes() for path in first] == [path.read_bytes() for path in second]
    assert b"\r" not in b"".join(path.read_bytes() for path in first)
    again = render_catalogue_figures(comparison, check, baseline, out_dir=tmp_path / "one")
    assert again == first
    catalogue_figure_paths(MODE_COMPARISON, tmp_path / "one")["csv"].write_text("changed\n", encoding="utf-8")
    with pytest.raises(FigureError, match="refusing to replace"):
        render_catalogue_figures(comparison, check, baseline, out_dir=tmp_path / "one")


def test_an_unknown_figure_name_is_refused(tmp_path):
    with pytest.raises(FigureError, match="unknown figure"):
        catalogue_figure_paths("gap", tmp_path)


# ---------- the caption (8.2, 8.5) ----------

@pytest.mark.parametrize(("name", "number"), [("mode-comparison", 5), ("cutoff-check", 6), ("repeat-baseline", 7)])
def test_the_markdown_is_the_numbered_caption_then_the_table_then_the_derived_statistics_line(drawn, name, number):
    table = drawn[name][1]
    lines = table.to_markdown().splitlines()
    assert lines[0] == table.caption and table.caption.startswith(f"Figure {number}. ")
    assert " n = " in table.caption and "—" not in table.caption and "\n" not in table.caption
    assert lines[1] == "" and lines[2] == "| " + " | ".join(COLUMNS) + " |"
    assert len([line for line in lines if line.startswith("| ")]) == len(rows_of(table)) + 1
    assert lines[-1] == DERIVED_ONLY


@pytest.mark.parametrize("name", ["mode-comparison", "cutoff-check", "repeat-baseline"])
def test_the_markdown_states_its_rounding_and_where_the_exact_values_are(drawn, name):
    lines = drawn[name][1].to_markdown().splitlines()
    last_row = max(index for index, line in enumerate(lines) if line.startswith("| "))
    assert lines[last_row + 1] == ""
    assert lines[last_row + 2] == (f"{PRECISION} Numbers in the table are rounded to 4 significant digits; the "
                                   f"exact values are in `{name}.csv`.")


@pytest.mark.parametrize("name", ["mode-comparison", "cutoff-check", "repeat-baseline"])
def test_every_caption_states_its_precision_once_in_the_same_words_as_the_rounding_note(drawn, name):
    """A caption mixes "+0.096" with "-0.00033"; the reader is told why, at the end of the caption."""
    caption = drawn[name][1].caption
    assert caption.count(PRECISION) == 1 and caption.endswith(f". {PRECISION}")
    assert figures_catalogue.CAPTION_PRECISION == PRECISION


@pytest.mark.parametrize(("value", "signed", "text"), [
    (-0.0003297, True, "-0.00033"),   # the recorded lower bound of the logprobs identifying difference
    (0.0003, True, "+0.00030"),
    (0.0003, False, "0.00030"),
    (-0.00004999, True, "-0.000050"),
    (0.0, True, "0.000"),
    (-0.0, True, "0.000"),
    (0.0, False, "0.000"),
    (0.09555, True, "+0.096"),
    (-0.1659, True, "-0.166"),
    (0.7217, False, "0.722"),
    (1.0, False, "1.000"),
    (0.0005001, True, "+0.001"),
])
def test_a_caption_number_has_three_decimals_and_never_loses_or_changes_its_sign(value, signed, text):
    assert figures_catalogue._number(value, signed=signed) == text


@pytest.mark.parametrize(("value", "text"), [
    (-0.0003297, "-0.0003297"), (0.09555123, "0.09555"), (1.5e-07, "1.5e-07"), (-1.5e-07, "-1.5e-07"),
    (0.0, "0"), (-0.0, "0"), (1.0, "1"), (-0.19734, "-0.1973"),
])
def test_a_rounded_table_cell_never_turns_a_nonzero_value_into_zero_or_changes_its_sign(value, text):
    assert figures_catalogue._text(value, exact=False) == text
    assert figures_catalogue._text(value, exact=True) == repr(value)


def shown(value: float, *, signed: bool = False) -> str:
    """A number as a caption shows it; the rule itself is tested above on fixed values."""
    return figures_catalogue._number(value, signed=signed)


def numbers_in(caption: str) -> list[str]:
    return re.findall(r"\d+\.\d+", caption)


@pytest.mark.parametrize("name", ["mode-comparison", "cutoff-check", "repeat-baseline"])
def test_every_real_number_of_a_caption_has_three_decimals(drawn, name):
    """One precision per caption; the two reference lines and a temperature are named as the settings they are."""
    result, table, _ = drawn[name]
    settings = {"0.5", f"{MIN_AUC:g}"} if name == MODE_COMPARISON else set()
    if name == REPEAT_BASELINE:
        settings = {f"{header['temperature']:g}" for header in result.to_dict()["arms"].values()}
    # A longer number is one that three decimals would round to zero; it keeps two significant digits instead.
    odd = [number for number in numbers_in(table.caption)
           if len(number.split(".")[1]) != 3 and number not in settings and float(number) > 0.0005]
    assert odd == [] and numbers_in(table.caption)
    assert "%" not in table.caption and "—" not in table.caption


def with_statistic(comparison, name: str, **changes):
    """``comparison`` with its statistic ``name`` changed, as data."""
    return replace(comparison, statistics=tuple(
        replace(statistic, **changes) if statistic.name == name else statistic
        for statistic in comparison.statistics))


def with_cell_interval(check, mode: str, kind: str, name: str, lo: float, hi: float):
    """``check`` with the interval of statistic ``name`` of one cell changed, as data."""
    cell = check.cell(mode, kind)
    statistic = cell.statistics[name]
    changed = replace(cell, statistics={**cell.statistics,
                                        name: replace(statistic, interval=replace(statistic.interval, lo=lo, hi=hi))})
    return replace(check, cells=tuple(changed if found is cell else found for found in check.cells))


def with_cell_statistic(check, mode: str, kind: str, name: str, **changes):
    """``check`` with statistic ``name`` of one cell changed, as data."""
    cell = check.cell(mode, kind)
    changed = replace(cell, statistics={**cell.statistics, name: replace(cell.statistics[name], **changes)})
    return replace(check, cells=tuple(changed if found is cell else found for found in check.cells))


def with_block_length(check, mode: str, length: int, kinds=PROMPT_KINDS):
    """``check`` as if the intervals of ``mode`` had been resampled in blocks of ``length`` months, as data."""
    for kind in kinds:
        for name, statistic in check.cell(mode, kind).statistics.items():
            check = with_cell_statistic(check, mode, kind, name, inputs={**statistic.inputs, "block_length": length})
    return check


def with_share(baseline, arm: str, **changes):
    """``baseline`` with the share of prompts with one answer of ``arm`` changed, as data."""
    changed = tuple(replace(statistic, **changes) if statistic.name == ONE_ANSWER_SHARE else statistic
                    for statistic in baseline.statistics[arm])
    return replace(baseline, statistics={**baseline.statistics, arm: changed})


def test_the_mode_comparison_caption_is_one_message_built_from_the_statistics_and_the_answer(drawn):
    comparison, table, _ = drawn[MODE_COMPARISON]
    statistics = mode_tests.by_name(comparison)
    logprobs, logits = (statistics[f"holdout_auc_{mode}"] for mode in (LOGPROBS, LOGITS))
    paired = statistics["paired_auc_difference"]
    assert logits.n == logprobs.n and logits.inputs["n_rows"] == logprobs.inputs["n_rows"]
    assert table.caption.startswith(
        "Figure 5. Held-out AUC of the recall estimate by evidence mode, on the same weights and prompts: "
        f"logprobs {shown(logprobs.value)} (interval {shown(logprobs.interval.lo)} to {shown(logprobs.interval.hi)}), "
        f"logits {shown(logits.value)} ({shown(logits.interval.lo)} to {shown(logits.interval.hi)}). "
        f"The paired difference, logits minus logprobs, is {shown(paired.value, signed=True)} "
        f"({shown(paired.interval.lo, signed=True)} to {shown(paired.interval.hi, signed=True)}); "
        f"the analysis answers {comparison.answer}")
    assert (f" n = {logits.n} held-out prompts ({logits.inputs['n_rows']} draws). Intervals are 95 percent "
            "prompt-clustered bootstrap intervals. The dotted line marks chance (0.5), the dashed line the rule's "
            f"minimum ({MIN_AUC:g}).") in table.caption
    # The phrase describes the method the analysis states for these intervals.
    assert logprobs.interval.method == logits.interval.method == CLUSTERED_AUC_METHOD
    assert paired.interval.method == PAIRED_AUC_METHOD
    # The agreement of the prompt estimates is not plotted: it stays in the table.
    assert "agreement" not in table.caption and "limits" not in table.caption
    assert all(one_row(table, statistic=name)["value"] != "" for name in AGREEMENT)
    assert f"Answer of the comparison: {comparison.answer}." in table.to_markdown()


@pytest.mark.parametrize(("answer", "cutoff", "low", "ending"), [
    (UNDECIDED, False, 0.1403, "the analysis answers undecided, because the cutoff check for logits did not pass. "),
    (UNDECIDED, None, 0.1403, "the analysis answers undecided, because no cutoff check for logits was supplied. "),
    (IMPROVE, True, 0.1403, "the analysis answers improve, because the difference interval excludes 0 and the "
                            "cutoff check for logits passed. "),
    (UNDECIDED, False, -0.02, "the analysis answers undecided. "),
    (MATCH, None, -0.02, "the analysis answers match. "),
])
def test_the_cause_in_the_mode_comparison_caption_is_read_from_data_never_from_the_reason_text(
        comparison, answer, cutoff, low, ending):
    paired = mode_tests.by_name(comparison)["paired_auc_difference"]
    changed = with_statistic(replace(comparison, answer=answer, cutoff_check=cutoff), "paired_auc_difference",
                             value=0.2783, interval=replace(paired.interval, lo=low, hi=0.4106))
    caption = mode_comparison_table(changed).caption
    assert f"is +0.278 ({low:+.3f} to +0.411); {ending}n = " in caption
    # A reworded reason changes the note under the table and nothing in the caption.
    other = ("the intervals of the limits of agreement cannot be computed (no month resample gave a defined limit), "
             "so the limits cannot be checked on their interval bounds")
    table = mode_comparison_table(replace(changed, reason=other))
    assert table.caption == caption
    assert f"Answer of the comparison: {answer}. Reason: {other}." in table.to_markdown().splitlines()
    # The note gives the published sentence of the stored reason; a reason without one is refused, not published.
    stored = ("holdout_auc_logits cannot be computed; missing: the calibrator record of the logits mode "
              "(full_distribution)")
    assert (f"Answer of the comparison: {answer}. Reason: the held-out AUC of logits cannot be computed; missing: "
            "the logits calibrator.") in mode_comparison_table(replace(changed, reason=stored)).to_markdown()
    with pytest.raises(FigureError, match="no published wording"):
        mode_comparison_table(replace(changed, reason="worded differently; in two parts"))


def test_a_plotted_interval_with_equal_bounds_is_named_in_the_caption(drawn):
    comparison, _, _ = drawn[MODE_COMPARISON]
    auc = mode_tests.by_name(comparison)[f"holdout_auc_{LOGITS}"]
    # The recorded case: logits separate the held-out prompts completely, logprobs do not.
    separated = with_statistic(comparison, f"holdout_auc_{LOGITS}", value=1.0,
                               interval=replace(auc.interval, lo=1.0, hi=1.0))
    separated = with_statistic(separated, f"holdout_auc_{LOGPROBS}", value=0.7217,
                               interval=replace(auc.interval, lo=0.5858, hi=0.8551))
    caption = mode_comparison_table(separated).caption
    assert "logprobs 0.722 (interval 0.586 to 0.855), logits 1.000 (1.000 to 1.000). " in caption
    assert caption.endswith(f" The logits interval is 1.000 to 1.000, so its bar has no length. {PRECISION}")
    assert "logprobs interval is" not in caption
    wide = with_statistic(separated, f"holdout_auc_{LOGITS}", interval=replace(auc.interval, lo=0.9, hi=1.0))
    assert "no length" not in mode_comparison_table(wide).caption

    check, table, _ = drawn[CUTOFF_CHECK]
    assert "no length" not in table.caption
    flat = with_cell_interval(check, "top_k", "anonymised", "mean_after", 0.25, 0.25)
    assert "The logprobs anonymised after interval is 0.250 to 0.250, so its bar has no length." in \
        cutoff_check_table(flat).caption


def test_the_cutoff_caption_names_the_months_the_difference_and_the_one_sided_p(drawn):
    check, table, _ = drawn[CUTOFF_CHECK]
    details, sizes = [], []
    for mode in MODES:
        cell = check.cell(mode, "identifying")
        summary, interval = cell.summary, cell.statistics["difference"].interval
        word = "interval " if not details else ""
        details.append(f"{CATALOGUE_MODES[mode]} difference {shown(summary.difference, signed=True)} "
                       f"({word}{shown(interval.lo, signed=True)} to {shown(interval.hi, signed=True)}, "
                       f"one-sided p {shown(summary.p_value)})")
        sizes.append(f"{CATALOGUE_MODES[mode]} {summary.n_inside} inside and {summary.n_after} after")
    # The fixture passes with logprobs and fails with logits: the caption does not claim more.
    assert check.cell("top_k", "identifying").summary.passes is True
    assert check.cell("full_distribution", "identifying").summary.passes is False
    assert table.caption == (
        "Figure 6. Under the identifying prompt the mean recall estimate is higher for analysis months inside the "
        f"training range than after the cutoff in the logprobs mode only: {', '.join(details)}. "
        f"n = months under the identifying prompt: {', '.join(sizes)}. "
        # Five months after the cutoff are fewer than the plan's block length of 6: every cell used block length 1.
        f"Intervals are 95 percent bootstrap intervals over months, {SINGLY}, so they do not allow for serial "
        "dependence between months. Panel (a) is the logprobs mode and "
        "panel (b) the logits mode; marker and line style give the prompt kind. The anonymised prompt is drawn for "
        f"comparison; its numbers are in the table. {PRECISION}")
    assert "moving-block" not in table.caption
    for cell in check.cells:
        assert cell.block_length == 1 and cutoff_tests.BLOCK_LENGTH == 6
        assert {statistic.inputs["block_length"] for statistic in cell.statistics.values()} == {1}
        # The wording describes the methods the analysis states for these intervals.
        assert cell.statistics["mean_inside"].interval.method == BLOCK_BOOTSTRAP_METHOD
        assert cell.statistics["mean_after"].interval.method == BLOCK_BOOTSTRAP_METHOD
        assert cell.statistics["difference"].interval.method == DIFFERENCE_INTERVAL_METHOD


def test_the_cutoff_caption_names_the_planned_block_length_where_it_was_used(serving, check):
    cell = serving.check.cell("top_k", "identifying")
    assert cell.block_length == 6 and cell.statistics["difference"].inputs["block_length"] == 6
    caption = cutoff_check_table(serving.check).caption
    assert (" Intervals are 95 percent moving-block bootstrap intervals over months with block length 6. Panel (a) "
            in caption)
    assert "singly" not in caption and "serial dependence" not in caption
    # A length that is neither 1 nor the plan's is named as it is.
    other = with_block_length(with_block_length(check, "top_k", 3), "full_distribution", 3)
    assert (" Intervals are 95 percent moving-block bootstrap intervals over months with block length 3. Panel (a) "
            in cutoff_check_table(other).caption)


def test_the_cutoff_caption_names_the_block_length_per_mode_where_the_modes_differ(check):
    mixed = with_block_length(check, "top_k", 6)
    assert (" Intervals are 95 percent bootstrap intervals over months. In the logprobs mode they are moving-block "
            "intervals with block length 6. In the logits mode months are resampled singly (block length 1, because "
            "a group has fewer months than the plan's block length of 6), so the intervals do not allow for serial "
            "dependence between months. Panel (a) ") in cutoff_check_table(mixed).caption
    # Where the two prompt kinds of one mode differ, the cells of that mode are named one by one.
    by_kind = with_block_length(check, "top_k", 6, kinds=("identifying",))
    assert (" Intervals are 95 percent bootstrap intervals over months. In the logprobs mode under the identifying "
            "prompt they are moving-block intervals with block length 6. In the logprobs mode under the anonymised "
            f"prompt months are {SINGLY}, so the intervals do not allow for serial dependence between months. In the "
            f"logits mode months are {SINGLY}, so the intervals do not allow for serial dependence between months. "
            "Panel (a) ") in cutoff_check_table(by_kind).caption


def test_a_caption_refuses_an_interval_method_or_block_length_it_has_no_wording_for(comparison, check, baseline):
    """A change of method in an analysis fails here instead of leaving an untrue caption."""
    auc = mode_tests.by_name(comparison)[f"holdout_auc_{LOGITS}"]
    paired = mode_tests.by_name(comparison)["paired_auc_difference"]
    for name, statistic in ((auc.name, auc), (paired.name, paired)):
        changed = with_statistic(comparison, name, interval=replace(statistic.interval, method="row bootstrap"))
        with pytest.raises(FigureError, match="row bootstrap"):
            mode_comparison_table(changed)
    for name in ("mean_inside", "mean_after", "difference"):
        statistic = check.cell("top_k", "anonymised").statistics[name]
        changed = with_cell_statistic(check, "top_k", "anonymised", name,
                                      interval=replace(statistic.interval, method="normal approximation"))
        with pytest.raises(FigureError, match="normal approximation"):
            cutoff_check_table(changed)
    statistic = check.cell("top_k", "identifying").statistics["mean_after"]
    for inputs in ({**statistic.inputs, "block_length": 2},
                   {key: value for key, value in statistic.inputs.items() if key != "block_length"},
                   {**statistic.inputs, "block_length": 0}):
        with pytest.raises(FigureError, match="block length"):
            cutoff_check_table(with_cell_statistic(check, "top_k", "identifying", "mean_after", inputs=inputs))
    share = baseline.statistic(ARMS[0], ONE_ANSWER_SHARE)
    with pytest.raises(FigureError, match="Wald"):
        repeat_baseline_table(with_share(baseline, ARMS[0], interval=replace(share.interval, method="Wald")))


def test_a_bound_that_rounds_to_zero_keeps_its_sign_in_the_cutoff_caption(check):
    """The recorded case: a lower bound of -0.0003297 means the interval includes 0, and the caption shows it."""
    high = check.cell("top_k", "identifying").statistics["difference"].interval.hi
    recorded = with_cell_interval(check, "top_k", "identifying", "difference", -0.0003297, high)
    caption = cutoff_check_table(recorded).caption
    assert f"(interval -0.00033 to {high:+.3f}, one-sided p " in caption
    assert "+0.000" not in caption and "-0.000 " not in caption
    table = cutoff_check_table(recorded).to_markdown()
    assert "| -0.0003297 |" in table


def test_the_repeat_baseline_caption_names_route_temperature_share_and_the_prompts_analysed(drawn):
    baseline, table, _ = drawn[REPEAT_BASELINE]
    arms = baseline.to_dict()["arms"]
    sizes = []
    for letter, arm in zip("ab", ARMS, strict=True):
        route, share = arms[arm]["route"], baseline.statistic(arm, ONE_ANSWER_SHARE)
        one = baseline.statistic(arm, distribution_name(1)).value
        assert f"panel ({letter}), route {route['profile_id']} ({route['provider']}, model {route['served_model']}) " \
               f"at temperature {arms[arm]['temperature']:g}: {one} of {share.n} prompts returned one answer " \
               f"(share {share.value:.3f}, interval {share.interval.lo:.3f} to {share.interval.hi:.3f})" in \
               table.caption
        sizes.append(f"{share.n} prompts in panel ({letter})")
    assert table.caption.startswith("Figure 7. Number of different answers the same prompt returned when it was "
                                    "asked again, by serving route and temperature: panel (a), route ")
    assert table.caption.endswith(
        f" n = {' and '.join(sizes)}. The bars are counts and carry no interval; the interval of a share is a "
        f"95 percent Wilson score interval; {NOT_INDEPENDENT}. {PRECISION}")
    # The analysis marks the prompts of both routes as not independent, and states the method the caption names.
    for arm in ARMS:
        share = baseline.statistic(arm, ONE_ANSWER_SHARE)
        assert share.independent_units is False and share.interval.method == WILSON_METHOD
    assert "assumes independent" not in table.caption
    # The exclusions are in the table; the published figure names the route, not the arm of the study.
    assert "excluded" not in table.caption and "not fully recorded" not in table.caption
    assert not re.search(r"\barm\b", table.caption)
    assert not any(re.search(r"\barm\b", row["panel"]) for row in rows_of(table))


def test_the_repeat_baseline_caption_says_only_what_the_interval_is_where_the_prompts_are_independent(baseline):
    independent = with_share(with_share(baseline, ARMS[0], independent_units=True), ARMS[1], independent_units=True)
    caption = repeat_baseline_table(independent).caption
    assert caption.endswith(". The bars are counts and carry no interval; the interval of a share is a 95 percent "
                            f"Wilson score interval. {PRECISION}")
    assert "independent" not in caption and "narrow" not in caption
    # The recorded case of the analysis's own test: one rendering per month, the other route without draws.
    keys, a, b = repeat_tests.SAMPLING_KEYS, repeat_tests.A, repeat_tests.B
    single = repeat_tests.build(sampling=repeat_tests.ensemble(ARMS[0], keys[0], (a, a, b, b))
                                + repeat_tests.ensemble(ARMS[0], keys[2], (a, a, a, a)))
    assert single.statistic(ARMS[0], ONE_ANSWER_SHARE).independent_units is True
    caption = repeat_baseline_table(single).caption
    assert f"Wilson score interval. {PRECISION}" in caption and "not independent" not in caption


def test_the_repeat_baseline_caption_names_the_panel_whose_prompts_are_not_independent(baseline):
    mixed = with_share(baseline, ARMS[1], independent_units=True)
    assert repeat_baseline_table(mixed).caption.endswith(
        f"95 percent Wilson score interval; in panel (a) {NOT_INDEPENDENT}. {PRECISION}")
    mixed = with_share(baseline, ARMS[0], independent_units=True)
    assert repeat_baseline_table(mixed).caption.endswith(
        f"95 percent Wilson score interval; in panel (b) {NOT_INDEPENDENT}. {PRECISION}")


# ---------- the table holds the values of the analysis exactly (8.3) ----------

def test_the_mode_comparison_table_holds_the_values_of_the_comparison_exactly(drawn):
    comparison, table, _ = drawn[MODE_COMPARISON]
    statistics = mode_tests.by_name(comparison)
    for mode in (LOGPROBS, LOGITS):
        statistic = statistics[f"holdout_auc_{mode}"]
        assert statistic.interval is not None
        assert_matches(one_row(table, x=mode), statistic, plotted=True)
    paired = statistics["paired_auc_difference"]
    row = one_row(table, statistic="paired_auc_difference")
    assert_matches(row, paired, plotted=False)
    assert float(row["p_value"]) == paired.test.p_value
    for name in AGREEMENT:
        assert_matches(one_row(table, statistic=name), statistics[name], plotted=False)
    assert len(rows_of(table)) == 6
    assert [row["plotted"] for row in rows_of(table)] == ["yes", "yes", "no", "no", "no", "no"]


def test_the_cutoff_table_holds_the_values_of_every_cell_exactly(drawn):
    check, table, _ = drawn[CUTOFF_CHECK]
    for letter, mode in zip("ab", MODES, strict=True):
        for kind in PROMPT_KINDS:
            cell = check.cell(mode, kind)
            where = {"panel": f"({letter}) {CATALOGUE_MODES[mode]}", "series": kind}
            for name, place in (("mean_inside", "inside"), ("mean_after", "after")):
                row = one_row(table, **where, statistic=name)
                assert row["x"] == place and cell.statistics[name].interval is not None
                assert_matches(row, cell.statistics[name], plotted=True)
            row = one_row(table, **where, statistic="difference")
            assert_matches(row, cell.statistics["difference"], plotted=False)
            assert float(row["p_value"]) == cell.summary.p_value
            assert float(row["interval_low"]) == cell.summary.interval_low
    assert len(rows_of(table)) == 12


def test_the_repeat_baseline_table_holds_every_statistic_of_both_arms_exactly(drawn):
    baseline, table, _ = drawn[REPEAT_BASELINE]
    arms = baseline.to_dict()["arms"]
    for letter, arm in zip("ab", ARMS, strict=True):
        route = arms[arm]["route"]
        panel = (f"({letter}) route {route['profile_id']} ({route['provider']}, model {route['served_model']}), "
                 f"temperature {arms[arm]['temperature']:g}")
        for statistic in baseline.statistics[arm]:
            row = one_row(table, panel=panel, statistic=statistic.name)
            is_count = statistic.name.startswith("prompts_with_") and statistic.name.endswith("different_answers")
            assert_matches(row, statistic, plotted=is_count)
            if is_count:
                assert distribution_name(int(row["x"])) == statistic.name
        share = one_row(table, panel=panel, statistic=ONE_ANSWER_SHARE)
        assert share["interval_low"] != "" and int(one_row(table, panel=panel, statistic=ANALYSED)["value"]) > 0
    assert len(rows_of(table)) == sum(len(found) for found in baseline.statistics.values())


# ---------- interval bars (8.5) ----------

def test_interval_bars_are_drawn_where_the_analysis_reports_an_interval(drawn):
    comparison, _, figure = drawn[MODE_COMPARISON]
    statistics = mode_tests.by_name(comparison)
    for place, mode in enumerate((LOGPROBS, LOGITS)):
        statistic = statistics[f"holdout_auc_{mode}"]
        (mark,), (bar,) = artists(figure, f"auc-{mode}"), artists(figure, f"auc-interval-{mode}")
        assert list(mark.get_xdata()) == [place] and list(mark.get_ydata()) == [statistic.value]
        assert list(bar.get_ydata()) == [statistic.interval.lo, statistic.interval.hi]
        assert list(bar.get_xdata()) == [place, place]
    assert len(marks(figure)) == 4

    check, _, figure = drawn[CUTOFF_CHECK]
    for mode in MODES:
        for kind in PROMPT_KINDS:
            cell = check.cell(mode, kind)
            (line,) = artists(figure, f"mean-{mode}-{kind}")
            assert list(line.get_ydata()) == [cell.statistics["mean_inside"].value, cell.statistics["mean_after"].value]
            for place, group in enumerate(("inside", "after")):
                (bar,) = artists(figure, f"mean-interval-{mode}-{kind}-{group}")
                interval = cell.statistics[f"mean_{group}"].interval
                assert list(bar.get_ydata()) == [interval.lo, interval.hi]
                assert bar.get_xdata()[0] == bar.get_xdata()[1] == line.get_xdata()[place]
    assert len(marks(figure)) == 12
    first, second = figure.axes
    assert first.get_ylim() == second.get_ylim() == (0.0, 1.05)
    assert [tick.get_text() for tick in first.get_xticklabels()] == ["inside", "after"]


# ---------- missing cells are listed, not drawn (6.9) ----------

def test_a_mode_without_a_calibrator_is_not_plotted_and_is_listed_with_its_reason(recorded):
    comparison = mode_tests.compare(recorded, logits=None)
    table, figure = mode_comparison_table(comparison), plot_mode_comparison(comparison)
    missing = mode_tests.by_name(comparison)[f"holdout_auc_{LOGITS}"].missing
    assert artists(figure, f"auc-{LOGITS}") == [] and artists(figure, f"auc-interval-{LOGITS}") == []
    assert len(artists(figure, f"auc-{LOGPROBS}")) == 1 and len(marks(figure)) == 2
    row = one_row(table, x=LOGITS)
    assert missing == "the calibrator record of the logits mode (full_distribution)"
    assert (row["value"], row["interval_low"], row["plotted"], row["missing"]) == ("", "", "no",
                                                                                 "the logits calibrator")
    assert one_row(table, statistic="paired_auc_difference")["missing"] == "the logits calibrator"
    assert f"No mark is drawn for {LOGITS}: the logits calibrator." in table.caption
    assert "full_distribution" not in table.to_markdown() and "full_distribution" not in table.to_csv()
    assert f"logits not computed. The paired difference, logits minus logprobs, cannot be computed; the analysis " \
           f"answers {comparison.answer}. n = " in table.caption
    (axes,) = figure.axes
    assert axes.get_xlabel() and axes.get_ylabel()


def test_a_refused_comparison_gives_an_empty_panel_whose_caption_says_why(recorded):
    comparison = mode_tests.compare(recorded, logits=mode_tests.with_weights(recorded.logits, mode_tests.OTHER_DIGEST))
    table, figure = mode_comparison_table(comparison), plot_mode_comparison(comparison)
    assert comparison.statistics == () and marks(figure) == []
    assert table.caption.startswith("Figure 5. ") and "refused" in table.caption and "different weights" in \
        table.caption
    assert "n = 0 held-out prompts." in table.caption
    assert [(row["plotted"], row["missing"]) for row in rows_of(table)] == [
        ("no", published_reason(comparison.refusal))] * 2
    assert published_reason(comparison.refusal) in table.caption
    (axes,) = figure.axes
    assert axes.get_xlabel() and axes.get_ylabel() and list(axes.texts) == []


def test_cells_of_a_mode_the_arm_does_not_record_leave_their_panel_empty_and_are_listed(serving):
    check = serving.check
    table, figure = cutoff_check_table(check), plot_cutoff_check(check)
    logprobs, logits = figure.axes
    assert len(logprobs.lines) == 6 and list(logits.lines) == []
    assert logits.get_xlabel() and logits.get_ylabel()
    reason = check.cell("full_distribution", "identifying").statistics["difference"].missing
    for kind in PROMPT_KINDS:
        for name in ("mean_inside", "mean_after", "difference"):
            row = one_row(table, panel="(b) logits", series=kind, statistic=name)
            assert (row["value"], row["interval_low"], row["plotted"]) == ("", "", "no")
            assert row["missing"] == published_reason(
                check.cell("full_distribution", kind).statistics[name].missing)
    assert reason == "the serving arm records no full_distribution estimate, so no month carries one"
    assert published_reason(reason) == "this serving route records no logits recall estimate, so no month has one"
    assert f"Panel (b) is empty: {published_reason(reason)}." in table.caption
    assert "serving arm" not in table.to_markdown() and "full_distribution" not in table.to_markdown()
    assert "higher for analysis months inside the training range than after the cutoff" in table.caption
    assert "not shown" not in table.caption
    assert_matches(one_row(table, panel="(a) logprobs", series="identifying", statistic="mean_inside"),
                   check.cell("top_k", "identifying").statistics["mean_inside"], plotted=True)


def test_a_cutoff_check_with_nothing_to_plot_gives_empty_panels_whose_caption_says_why():
    case = cutoff_tests.sampling_case(after_months=1)
    check = cutoff_check(cutoff_tests.PLAN, case.results, arm="sampling")
    table, figure = cutoff_check_table(check), plot_cutoff_check(check)
    assert marks(figure) == [] and figure.legends == []
    assert all(axes.get_xlabel() and axes.get_ylabel() for axes in figure.axes)
    reason = check.cell("top_k", "identifying").statistics["difference"].missing
    assert "cannot be computed and nothing is plotted" in table.caption
    assert reason not in table.caption and published_reason(reason) in table.caption
    assert "months with a logprobs recall estimate under the identifying prompt: " in table.caption
    assert all(row["plotted"] == "no" and row["missing"] for row in rows_of(table))


def test_an_arm_without_recorded_draws_leaves_its_panel_empty_and_is_listed():
    baseline = repeat_tests.build(sampling=repeat_tests.sampling_draws())
    table, figure = repeat_baseline_table(baseline), plot_repeat_baseline(baseline)
    sampling_axes, serving_axes = figure.axes
    assert len(sampling_axes.patches) == 3 and list(serving_axes.patches) == []
    assert serving_axes.get_xlabel() and serving_axes.get_ylabel()
    rows = [row for row in rows_of(table) if row["panel"].startswith("(b) route ")]
    assert len(rows) == len(baseline.statistics["serving"])
    no_draws = "no draw of the full run is recorded for this serving route"
    assert published_reason(NO_DRAWS) == no_draws
    assert all((row["value"], row["plotted"], row["missing"]) == ("", "no", no_draws) for row in rows)
    assert f"is empty: {no_draws}" in table.caption and "0 prompts in panel (b)" in table.caption
    assert NO_DRAWS not in table.to_markdown() and NO_DRAWS not in table.to_csv()


def test_the_figures_refuse_anything_but_the_result_of_their_analysis(comparison, check, baseline):
    for function, wrong in ((mode_comparison_table, check), (plot_mode_comparison, baseline),
                            (cutoff_check_table, baseline), (plot_cutoff_check, comparison),
                            (repeat_baseline_table, comparison), (plot_repeat_baseline, check)):
        with pytest.raises(TypeError):
            function(wrong)


def test_the_figure_code_computes_no_statistic():
    """It reads the result objects; no resampling, test or interval function is used."""
    with open(figures_catalogue.__file__, encoding="utf-8") as stream:
        source = stream.read()
    for name in ("wilson_interval", "block_bootstrap", "clustered_auc_interval", "paired_auc_difference(",
                 "permutation_", "agreement(", "import numpy"):
        assert name not in source, name
