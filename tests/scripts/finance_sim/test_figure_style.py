"""The one academic figure style (qualified-model-catalogue task 1.2; Requirements 8.2, 8.4, 8.5, 8.6, 8.8).

Style values, series styles that differ without colour, the numbered caption,
and a render under the style that is grey only and byte-stable.
"""

from __future__ import annotations

import io
import itertools
import re

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
