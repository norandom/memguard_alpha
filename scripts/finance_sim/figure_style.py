"""The one academic figure style of the research record (Requirements 8.2, 8.4, 8.5, 8.6, 8.8).

Every figure is drawn the way a quantitative journal article prints one: a
plain white background, black axes, ticks and text, a serif typeface, no grid
and no top or right spine. Series are told apart by marker shape, line style
and grey level, never by colour, so a figure reads in black and white print.
The message of a figure is not written inside the plot; it goes in a numbered
caption below it (:func:`caption`), which also states the sample size and what
the interval is.

:func:`academic_rc` returns plain rcParams values and this module imports
nothing from matplotlib, so importing it is free. The fonts are the ones that
ship with matplotlib (DejaVu Serif, STIX for math), glyphs are saved as paths
and the SVG ids are salted with a fixed value, so a saved SVG has the same
bytes in every process.
"""

from __future__ import annotations

__all__ = ["SERIES_STYLES", "SVG_HASH_SALT", "academic_rc", "caption", "series_style"]

#: Salt of the ids in a saved SVG; the value the published finance-simulation figures already use.
SVG_HASH_SALT = "memorization-adjusted-finance-sim"

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
        "legend.fontsize": 8.0,
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
