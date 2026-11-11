"""The one academic figure style of the research record (Requirements 8.2, 8.4, 8.5, 8.6, 8.8).

Every figure is drawn the way a quantitative journal article prints one: a
plain white background, black axes, ticks and text, a serif typeface, no grid
and no top or right spine. Series are told apart by marker shape, line style
and grey level, never by colour, so a figure reads in black and white print.
The message of a figure is not written inside the plot; it goes in a numbered
caption below it (:func:`caption`), which also states the sample size and what
the interval is. A multi-panel figure labels its panels "(a)", "(b)" above each
plot area (:func:`label_panel`) and the caption says what each panel shows.

The module also holds what every figure module shares to render and write a
figure: the complete rcParams a figure is built under (:func:`figure_rc`), the
layout passes (:func:`lay_out`, :func:`freeze_layout`), the byte-stable SVG
text (:func:`figure_svg`), the write-once outputs (:func:`write_outputs`), the
refusal (:class:`FigureError`), the published figure folder
(:data:`FIGURES_DIR`) and the line every published table ends with
(:data:`DERIVED_ONLY`).

:func:`academic_rc` returns plain rcParams values and matplotlib is imported
only inside the functions that render, so importing this module is free. The
fonts are the ones that ship with matplotlib (DejaVu Serif, STIX for math),
glyphs are saved as paths and the SVG ids are salted with a fixed value, so a
saved SVG has the same bytes in every process.
"""

from __future__ import annotations

import io
import os
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from matplotlib.figure import Figure

__all__ = ["DERIVED_ONLY", "FIGURES_DIR", "FIGURE_WIDTH_IN", "SERIES_STYLES", "SVG_HASH_SALT", "FigureError",
           "academic_rc", "caption", "figure_rc", "figure_svg", "freeze_layout", "label_panel", "lay_out",
           "panel_label", "series_style", "write_outputs"]

ROOT = Path(__file__).resolve().parents[2]
#: The docs figure folder every published figure is written to; tests write under tmp_path instead.
FIGURES_DIR = ROOT / "docs" / "assets" / "finance-sim"
#: The last line of every published table.
DERIVED_ONLY = "Derived statistics only: no reply text, no realized returns and no raw licensed series."

#: Salt of the ids in a saved SVG; the value the published finance-simulation figures already use.
SVG_HASH_SALT = "memorization-adjusted-finance-sim"
#: Width of a figure that spans the text column, in inches; a smaller figure states its own width.
FIGURE_WIDTH_IN = 7.2
#: Distance of a panel label from the top edge of its plot area, in points.
_PANEL_LABEL_GAP_PT = 4.0

_PAPER = "white"
_INK = "black"
#: Marker shapes, line styles and grey levels a series cycles through. The numbers of markers and of line styles
#: share no factor, so every index below their product has its own (marker, line style) pair.
_MARKERS: tuple[str, ...] = ("o", "s", "^", "D", "v", "X", "P")
_LINESTYLES: tuple[str, ...] = ("-", "--", ":", "-.")
_GREYS: tuple[str, ...] = ("black", "0.4")
#: Number of mutually distinct series styles; :func:`series_style` repeats after this many.
SERIES_STYLES = len(_MARKERS) * len(_LINESTYLES)

_EM_DASH = "—"
_SENTENCE_END = (".", "!", "?")
#: Decimals of the figure fraction kept when the layout is frozen (about 1e-9 of the figure).
_LAYOUT_DECIMALS = 9


class FigureError(ValueError):
    """An input or output the figures refuse; nothing is written for it."""


def academic_rc() -> dict[str, object]:
    """The rcParams of the academic style, as a new dict: white, black ink, serif, no grid, reproducible SVG."""
    return {
        # Plain white background.
        "figure.facecolor": _PAPER, "figure.edgecolor": _PAPER, "axes.facecolor": _PAPER,
        "savefig.facecolor": _PAPER, "savefig.edgecolor": _PAPER,
        # Black axes, ticks and text.
        "text.color": _INK, "axes.edgecolor": _INK, "axes.labelcolor": _INK, "xtick.color": _INK,
        "ytick.color": _INK, "xtick.labelcolor": _INK, "ytick.labelcolor": _INK, "lines.color": _INK,
        "patch.edgecolor": _INK, "axes.prop_cycle": f"cycler('color', ['{_INK}'])",
        # A serif typeface that ships with matplotlib, STIX for math.
        "font.family": "serif", "font.serif": ["DejaVu Serif"], "mathtext.fontset": "stix",
        "font.size": 9.0, "axes.labelsize": 9.0, "xtick.labelsize": 8.0, "ytick.labelsize": 8.0,
        "legend.fontsize": 8.0, "figure.labelsize": 9.0,
        # No grid, an open frame, ticks pointing out.
        "axes.grid": False, "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": True,
        "axes.spines.bottom": True, "axes.linewidth": 0.8, "xtick.direction": "out", "ytick.direction": "out",
        "xtick.major.width": 0.8, "ytick.major.width": 0.8,
        # Thin lines, small markers, capped interval bars.
        "lines.linewidth": 1.0, "lines.markersize": 5.0, "lines.markeredgewidth": 0.9, "errorbar.capsize": 2.5,
        # A plain legend: no frame, no shadow, no rounded box.
        "legend.frameon": False, "legend.fancybox": False, "legend.shadow": False, "legend.facecolor": _PAPER,
        "legend.edgecolor": _INK, "legend.labelcolor": _INK,
        # Reproducible SVG: glyphs as paths, fixed id salt.
        "svg.fonttype": "path", "svg.hashsalt": SVG_HASH_SALT,
    }


def series_style(index: int) -> dict[str, object]:
    """Plot keyword arguments of series ``index``: marker, line style and grey level, no colour.

    Any two of the first ``SERIES_STYLES`` indexes differ in marker or line
    style, and the first seven differ in marker alone; the grey level and the
    marker fill (solid for the black series, open for the grey ones) only add
    to that. The styles repeat after ``SERIES_STYLES``.
    """
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise ValueError(f"index must be an integer of at least 0, got {index!r}")
    grey = _GREYS[index % len(_GREYS)]
    return {
        "marker": _MARKERS[index % len(_MARKERS)],
        "linestyle": _LINESTYLES[index % len(_LINESTYLES)],
        "color": grey,
        "markerfacecolor": grey if grey == _GREYS[0] else _PAPER,
        "markeredgecolor": grey,
    }


def panel_label(index: int) -> str:
    """The label of panel ``index`` of a multi-panel figure: "(a)" for 0, "(b)" for 1, up to "(z)"."""
    if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < 26:
        raise ValueError(f"index must be an integer from 0 to 25, got {index!r}")
    return f"({chr(ord('a') + index)})"


def label_panel(axes: Any, index: int) -> Any:
    """Write the label of panel ``index`` above the top-left corner of ``axes``, outside the plot area.

    The caption says what each panel shows; nothing else is written on a
    panel. The label takes part in the layout, so a constrained layout keeps
    room for it. Returns the text artist (gid ``panel-label-<letter>``).
    """
    text = panel_label(index)
    return axes.annotate(text, xy=(0.0, 1.0), xycoords="axes fraction", xytext=(0.0, _PANEL_LABEL_GAP_PT),
                         textcoords="offset points", ha="left", va="bottom", annotation_clip=False,
                         gid=f"panel-label-{text[1]}")


def _sentence(text: str, name: str) -> str:
    """``text`` on one line with single spaces and a closing full stop; empty text and em dashes are refused."""
    line = " ".join(text.split())
    if not line:
        raise ValueError(f"{name} must not be empty")
    if _EM_DASH in line:
        raise ValueError(f"{name} must not contain an em dash")
    return line if line.endswith(_SENTENCE_END) else f"{line}."


def caption(number: int, message: str, *, n: str, interval: str | None) -> str:
    """The caption below figure ``number``, one line: "Figure N.", the message, the sample size, the interval.

    ``message`` is the one statement the figure makes, ``n`` the sample size
    with its unit (for example "120 prompts in 12 months") and ``interval``
    what the bars or bands are (for example "Bars are 95% cluster bootstrap
    intervals"), or None when the figure shows no interval.
    """
    if isinstance(number, bool) or not isinstance(number, int) or number < 1:
        raise ValueError(f"number must be an integer of at least 1, got {number!r}")
    parts = [f"Figure {number}.", _sentence(message, "message"), _sentence(f"n = {_sentence(n, 'n')}", "n")]
    if interval is not None:
        parts.append(_sentence(interval, "interval"))
    return " ".join(parts)


# ---------- rendering and writing, shared by every figure module ----------

def _default_rc() -> dict[str, Any]:
    """matplotlib's own defaults for every style key, so a render starts from the same rcParams in every process.

    The plot functions build and lay out a figure inside
    ``matplotlib.rc_context`` with these defaults under the academic style
    (:func:`figure_rc`), and :func:`figure_svg` saves under them, so
    whatever the process set before is overlaid, not inherited: a paper style
    (``recall_guard.harness.plots.configure_paper_style`` leaves
    ``savefig.format`` at ``pdf``, which is the renderer
    ``Figure.draw_without_rendering`` lays the figure out and measures its
    text with, and ``savefig.bbox`` at ``tight``, which re-crops the saved
    file), a style sheet or a user's matplotlibrc. The keys matplotlib keeps
    out of a style (``STYLE_BLACKLIST``: the backend, interactive mode, window
    and directory settings) stay as they are; the backend cannot be reset at
    runtime, and none of them reaches a render. The fonts are the ones shipped
    with matplotlib (:func:`academic_rc`) and nothing here touches the font
    manager.
    """
    import matplotlib
    from matplotlib.style.core import STYLE_BLACKLIST

    return {key: value for key, value in matplotlib.rcParamsDefault.items() if key not in STYLE_BLACKLIST}


def figure_rc() -> dict[str, Any]:
    """The complete rcParams every figure is built, laid out and saved under.

    matplotlib's defaults (:func:`_default_rc`), then the one academic style
    (:func:`academic_rc`); nothing of the process's state.
    """
    return {**_default_rc(), **academic_rc()}


def lay_out(figure: Figure) -> Figure:
    """Run the constrained layout of ``figure`` to its fixed point; returns the same figure.

    One pass places the panels; a second settles a panel of fixed aspect and
    the room its labels and the legend need, so nothing is cut off at the edge
    of the page. Call inside ``matplotlib.rc_context(figure_rc())``.
    """
    for _ in range(2):
        figure.draw_without_rendering()
    return figure


def freeze_layout(figure: Figure) -> Figure:
    """Fix the last layout of ``figure``: positions rounded, layout engine off; returns the same figure.

    The constrained layout can return axes positions that differ in their last
    bits from one process to the next, and the SVG clip-path ids hash the
    unrounded clip rectangles, so the same figure would not be byte-identical
    across processes. After the final layout pass every axes position and
    every figure-level text position is rounded to ``_LAYOUT_DECIMALS``
    decimals of the figure (a shift far below a point) and the layout engine
    is switched off, so saving draws exactly these positions. A legend
    placed outside the panels keeps the place the layout gave it.
    """
    for axes in figure.axes:
        bounds = [round(float(value), _LAYOUT_DECIMALS) for value in axes.get_position().bounds]
        axes.set_position(bounds, which="both")
    texts = [*figure.texts, *(text for text in (getattr(figure, name, None) for name in (
        "_suptitle", "_supxlabel", "_supylabel")) if text is not None)]
    for text in texts:
        x, y = text.get_position()
        text.set_position((round(float(x), _LAYOUT_DECIMALS), round(float(y), _LAYOUT_DECIMALS)))
    figure.set_layout_engine("none")
    return figure


def figure_svg(figure: Figure) -> str:
    """The SVG text of ``figure``: glyphs as paths, a fixed id salt and no timestamp, so it is byte-stable.

    The figure's layout is frozen first (:func:`freeze_layout`), and the file
    is saved under matplotlib's own defaults plus the academic style
    (:func:`figure_rc`: no tight bounding box, nothing from the process's
    global rcParams), so the bytes do not depend on the process that renders
    it.
    """
    import matplotlib

    with matplotlib.rc_context(figure_rc()):
        freeze_layout(figure)
        buffer = io.StringIO()
        figure.savefig(buffer, format="svg", metadata={"Date": None}, facecolor=figure.get_facecolor(),
                       edgecolor=figure.get_edgecolor())
    return buffer.getvalue()


def _write_text(path: Path, text: str) -> None:
    """Write atomically: a crash leaves the previous file or the new one, never half of one."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def write_outputs(texts: Mapping[Path, str]) -> tuple[Path, ...]:
    """Write every file once: all are checked first; the same text is a no-op, a different one refused."""
    for path, text in texts.items():
        if path.exists() and path.read_text(encoding="utf-8") != text:
            raise FigureError(f"refusing to replace the different figure output at {path}; remove the old figure "
                              "files first")
    for path, text in texts.items():
        if not path.exists():
            _write_text(path, text)
    return tuple(texts)
