"""The one academic figure style (qualified-model-catalogue task 1.2; Requirements 8.2, 8.4, 8.5, 8.6, 8.8).

Style values, series styles that differ without colour, the numbered caption,
a render under the style that is grey only and byte-stable, and the helpers
every figure module shares to render and write a figure.
"""

from __future__ import annotations

import io
import itertools
import re
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import pytest  # noqa: E402
from matplotlib.colors import to_rgb  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from scripts.finance_sim import figure_style  # noqa: E402
from scripts.finance_sim.figure_style import academic_rc, caption, series_style  # noqa: E402
from scripts.finance_sim.figures import SVG_HASH_SALT  # noqa: E402

_INK_KEYS = ("text.color", "axes.edgecolor", "axes.labelcolor", "xtick.color", "ytick.color",
             "xtick.labelcolor", "ytick.labelcolor", "legend.labelcolor", "lines.color", "patch.edgecolor")
_PAPER_KEYS = ("figure.facecolor", "figure.edgecolor", "axes.facecolor", "savefig.facecolor", "savefig.edgecolor",
               "legend.facecolor")


def _is_grey(colour: object) -> bool:
    red, green, blue = to_rgb(colour)
    return red == green == blue


def _render() -> str:
    """Six series with interval bars, a legend and both axis labels, saved as SVG under the style."""
    with matplotlib.rc_context(academic_rc()):
        figure = Figure(figsize=(4.0, 3.0))
        axes = figure.subplots()
        for index in range(6):
            axes.errorbar([0, 1, 2], [index, index + 1, index + 0.5], yerr=0.2, label=f"series {index}",
                          **series_style(index))
        axes.set_xlabel("Horizon (months)")
        axes.set_ylabel("Accuracy (share correct, $p_{mem}$)")
        axes.legend()
        buffer = io.StringIO()
        figure.savefig(buffer, format="svg", metadata={"Date": None})
    return buffer.getvalue()


def test_rc_keys_are_valid_matplotlib_settings() -> None:
    rc = academic_rc()
    assert set(rc) <= set(matplotlib.rcParams)
    with matplotlib.rc_context(rc):
        for key in ("font.family", "axes.grid", "svg.fonttype"):
            assert matplotlib.rcParams[key] is not None


def test_rc_has_a_white_background() -> None:
    rc = academic_rc()
    for key in _PAPER_KEYS:
        assert to_rgb(rc[key]) == (1.0, 1.0, 1.0), key


def test_rc_has_black_axes_and_text() -> None:
    rc = academic_rc()
    for key in _INK_KEYS:
        assert to_rgb(rc[key]) == (0.0, 0.0, 0.0), key


def test_rc_uses_a_serif_typeface_shipped_with_matplotlib() -> None:
    rc = academic_rc()
    assert rc["font.family"] == "serif"
    assert rc["font.serif"] == ["DejaVu Serif"]
    assert rc["mathtext.fontset"] == "stix"


def test_rc_sizes_a_label_shared_by_the_panels_like_an_axis_label() -> None:
    rc = academic_rc()
    assert rc["figure.labelsize"] == rc["axes.labelsize"] == rc["font.size"] == 9.0


def test_rc_has_no_grid_and_no_top_or_right_spine() -> None:
    rc = academic_rc()
    assert rc["axes.grid"] is False
    assert rc["axes.spines.top"] is False
    assert rc["axes.spines.right"] is False
    assert rc["axes.spines.left"] is True
    assert rc["axes.spines.bottom"] is True


def test_rc_default_colour_cycle_is_black_only() -> None:
    with matplotlib.rc_context(academic_rc()):
        colours = matplotlib.rcParams["axes.prop_cycle"].by_key()["color"]
    assert [to_rgb(colour) for colour in colours] == [(0.0, 0.0, 0.0)]


def test_rc_keeps_svg_output_reproducible() -> None:
    rc = academic_rc()
    assert rc["svg.fonttype"] == "path"
    assert rc["svg.hashsalt"] == SVG_HASH_SALT == figure_style.SVG_HASH_SALT


def test_rc_is_a_fresh_dict_on_every_call() -> None:
    first = academic_rc()
    first["font.family"] = "sans-serif"
    first["font.serif"].append("Times")
    assert academic_rc()["font.family"] == "serif"
    assert academic_rc()["font.serif"] == ["DejaVu Serif"]


def test_series_styles_carry_marker_line_style_and_grey_level() -> None:
    for index in range(figure_style.SERIES_STYLES):
        style = series_style(index)
        assert {"marker", "linestyle", "color", "markerfacecolor", "markeredgecolor"} <= set(style)
        for key in ("color", "markerfacecolor", "markeredgecolor"):
            assert _is_grey(style[key]), (index, key)
        assert to_rgb(style["color"])[0] <= 0.5, "a series must stay legible on white"


def test_at_least_six_series_styles_differ_without_colour() -> None:
    assert figure_style.SERIES_STYLES >= 6
    shapes = [(series_style(index)["marker"], series_style(index)["linestyle"])
              for index in range(figure_style.SERIES_STYLES)]
    for first, second in itertools.combinations(shapes, 2):
        assert first != second
    assert len({marker for marker, _ in shapes[:6]}) == 6


def test_series_style_is_deterministic_and_accepted_by_matplotlib() -> None:
    assert series_style(3) == series_style(3)
    axes = Figure().subplots()
    for index in range(figure_style.SERIES_STYLES):
        (line,) = axes.plot([0, 1], [index, index], **series_style(index))
        assert line.get_marker() == series_style(index)["marker"]


def test_series_style_wraps_after_the_last_style_and_rejects_a_negative_index() -> None:
    assert series_style(figure_style.SERIES_STYLES) == series_style(0)
    with pytest.raises(ValueError, match="index"):
        series_style(-1)


def test_caption_starts_with_the_figure_number() -> None:
    text = caption(3, "Recall falls after the cutoff", n="120 prompts in 12 months",
                   interval="Bars are 95% cluster bootstrap intervals")
    assert text.startswith("Figure 3. ")
    assert text == ("Figure 3. Recall falls after the cutoff. n = 120 prompts in 12 months. "
                    "Bars are 95% cluster bootstrap intervals.")


def test_caption_states_message_then_sample_size_then_interval() -> None:
    text = caption(1, "The two modes agree.", n="40 prompts", interval="Bands are 95% intervals.")
    assert text.index("The two modes agree.") < text.index("n = 40 prompts.") < text.index("Bands are 95%")
    assert ".." not in text


def test_caption_without_an_interval_ends_at_the_sample_size() -> None:
    assert caption(12, "Counts per month", n="7 months", interval=None) == "Figure 12. Counts per month. n = 7 months."


def test_caption_is_one_line_without_trailing_spaces_or_doubled_spaces() -> None:
    text = caption(2, "  Accuracy by\n arm ", n=" 30 draws ", interval="  Bars are ranges \n")
    assert text == "Figure 2. Accuracy by arm. n = 30 draws. Bars are ranges."
    assert "\n" not in text and "  " not in text and text == text.strip()


@pytest.mark.parametrize(("number", "message", "n", "interval"), [
    (0, "Message", "10 prompts", None),
    (True, "Message", "10 prompts", None),
    (1, " ", "10 prompts", None),
    (1, "Message", "", None),
    (1, "Message", "10 prompts", "  "),
    (1, "Before — after", "10 prompts", None),
    (1, "Message", "10 prompts", "Bars — ranges"),
])
def test_caption_rejects_unusable_input(number: int, message: str, n: str, interval: str | None) -> None:
    with pytest.raises(ValueError):
        caption(number, message, n=n, interval=interval)


def test_a_render_under_the_style_uses_grey_levels_only() -> None:
    svg = _render()
    colours = set(re.findall(r"#[0-9a-fA-F]{6}\b", svg))
    assert colours, "the render should name its colours"
    assert all(_is_grey(colour) for colour in colours), sorted(colours)
    assert "<text" not in svg, "glyphs are drawn as paths"


def test_a_render_under_the_style_is_byte_stable() -> None:
    assert _render() == _render()


def test_panel_labels_are_lower_case_letters_in_parentheses() -> None:
    assert [figure_style.panel_label(index) for index in (0, 1, 5, 25)] == ["(a)", "(b)", "(f)", "(z)"]
    for index in (-1, 26, True, 1.0):
        with pytest.raises(ValueError, match="index"):
            figure_style.panel_label(index)


def test_a_panel_label_sits_outside_the_plot_area_above_its_top_left_corner() -> None:
    with matplotlib.rc_context(academic_rc()):
        figure = Figure(figsize=(figure_style.FIGURE_WIDTH_IN, 3.0), layout="constrained")
        panels = figure.subplots(1, 2)
        labels = [figure_style.label_panel(axes, index) for index, axes in enumerate(panels)]
        figure.draw_without_rendering()
        assert [label.get_text() for label in labels] == ["(a)", "(b)"]
        assert [label.get_gid() for label in labels] == ["panel-label-a", "panel-label-b"]
        for axes, label in zip(panels, labels, strict=True):
            extent, frame = label.get_window_extent(), axes.bbox
            assert extent.y0 > frame.y1, "above the plot area, not inside it"
            assert abs(extent.x0 - frame.x0) < 1.0, "at the left edge of the plot area"
            assert extent.y1 <= figure.bbox.y1, "the layout keeps the label on the page"
            assert axes.get_title() == ""


def test_the_figure_width_is_a_full_text_column() -> None:
    assert figure_style.FIGURE_WIDTH_IN == 7.2


def test_the_style_module_holds_the_helpers_every_figure_module_shares() -> None:
    for name in ("figure_rc", "figure_svg", "lay_out", "freeze_layout", "write_outputs", "FigureError",
                 "DERIVED_ONLY", "FIGURES_DIR"):
        assert name in figure_style.__all__ and hasattr(figure_style, name), name
    assert issubclass(figure_style.FigureError, ValueError)
    root = Path(__file__).resolve().parents[3]
    assert figure_style.FIGURES_DIR == root / "docs" / "assets" / "finance-sim"
    assert figure_style.DERIVED_ONLY.startswith("Derived statistics only")
    # The complete rcParams of a render: matplotlib's own defaults under the style, nothing of the process.
    with matplotlib.rc_context({"savefig.format": "pdf", "lines.markersize": 12.0}):
        rc = figure_style.figure_rc()
    assert rc["savefig.format"] == "png" and rc["lines.markersize"] == 5.0 and "backend" not in rc
    assert {key: rc[key] for key in academic_rc()} == academic_rc()


def test_figure_svg_is_byte_stable_and_freezes_the_layout() -> None:
    from matplotlib.layout_engine import PlaceHolderLayoutEngine

    def render() -> str:
        with matplotlib.rc_context(figure_style.figure_rc()):
            figure = Figure(figsize=(4.0, 3.0), layout="constrained")
            axes = figure.subplots()
            axes.plot([0, 1], [0, 1], **series_style(0))
            axes.set_xlabel("Horizon (months)")
            axes.set_ylabel("Accuracy (share correct)")
            assert figure_style.lay_out(figure) is figure
        text = figure_style.figure_svg(figure)
        assert isinstance(figure.get_layout_engine(), PlaceHolderLayoutEngine)
        return text

    svg = render()
    assert svg == render() and "<dc:date>" not in svg and "<text" not in svg


def test_outputs_are_written_once_and_a_different_output_is_refused(tmp_path: Path) -> None:
    first, second = tmp_path / "assets" / "figure.md", tmp_path / "assets" / "figure.csv"
    texts = {first: "caption\n", second: "a,b\n"}
    assert figure_style.write_outputs(texts) == (first, second)
    assert first.read_bytes() == b"caption\n" and second.read_bytes() == b"a,b\n"
    assert figure_style.write_outputs(texts) == (first, second)  # the same output again is a no-op
    with pytest.raises(figure_style.FigureError, match="refusing to replace"):
        figure_style.write_outputs({second: "a,b\n", first: "another caption\n"})
    assert first.read_bytes() == b"caption\n" and not list(first.parent.glob("*.tmp"))


def test_the_style_module_imports_without_matplotlib() -> None:
    code = ("import sys\n"
            "import scripts.finance_sim.figure_style\n"
            "heavy = sorted(m for m in sys.modules if m.split('.')[0] == 'matplotlib')\n"
            "assert not heavy, heavy\n"
            "print('lean-ok')\n")
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                            cwd=Path(__file__).resolve().parents[3])
    assert result.returncode == 0, result.stderr
    assert "lean-ok" in result.stdout
