"""Offline worked-example replay and licensing guard (task 8.4; Requirements 10.1 and 10.7).

One command rebuilds the worked example from the recorded draws and scores
alone, with no model request: both pre-registered analyses (task 7.10
:func:`~scripts.finance_sim.analysis.analysis_from_records`, task 7.11
:func:`~scripts.finance_sim.ensemble_metrics.ensemble_metrics_from_records`),
the negative-control report (task 7.9
:func:`~scripts.finance_sim.negative_control.negative_control_from_records`)
and the convergence analysis (task 7.12
:func:`~scripts.finance_sim.convergence.convergence_from_records`), then the five
figures (the accuracy gap and the reliability diagram, the first-draw figure with the
draw-spread figure, and the convergence figure: :func:`~scripts.finance_sim.figures.render_figures`,
:func:`~scripts.finance_sim.figures_draws.render_draws` on the probe arm with the
frozen example rule, :func:`~scripts.finance_sim.figures_convergence.render_convergence`),
all written into one fresh output root.

Layout under the output root (design: File Structure Plan and the Figures
layout). The analysis JSON documents carry per-item rows and stay private, so
they go to ``runs/finance-sim/analysis/`` with the markdown each writer puts
beside them; the published files go to ``docs/assets/finance-sim/``: the five
figures (one SVG in the academic style, CSV and markdown each) and one published copy of
every analysis markdown with the sources above it (Moody's for BAA10Y, FMP for
the prices, SEC EDGAR for the releases). The root also gets ``replay.json``,
the manifest with the SHA-256 of every file, the matplotlib version and the
example rule. The dated report of task 9.5 (``docs/results/``) cites these
files; this module does not write it.

Reproducibility (Requirement 10.7). Every statistic is recomputed from the
frozen plan's seeds and the stored results, and the figure bytes are stable
across processes (task 8.3's frozen layout), so a replay in a fresh process
reproduces the stored outputs byte for byte. The SVG metadata carries the
matplotlib version (``Matplotlib v<version>``); another version changes the
glyph paths and the layout, so the replay refuses to run with a version other
than :data:`MATPLOTLIB_VERSION` and checks the metadata of every SVG it wrote.
``--compare DIR`` then compares the published files with a committed copy.

Licensing guard. Published tables carry derived statistics only. Before the
manifest is written, every published markdown, CSV and SVG is scanned for the
raw licensed series: each macro row's raw FRED levels (BAA10Y among them) and
the realized returns from the private ``items.json``, the FMP adjusted closes
of every cached body under the private FMP folder (``<symbol>_<start>_<end>.json``,
read with the item assembly's own parser, per symbol in date order, both the
daily series and the study's month-end series, the last close of each calendar
month as ``items.realized_returns`` takes it) and the DTB3 rates of the cached
FRED body, each as single values where they are distinctive and as runs of
consecutive values, and for the reply text of every recorded draw. A match on
a refusing label removes the published tree and refuses the replay, naming the
file, the series and the count, never the value. DTB3 is public data (the FRED
H.15 three-month bill rate, not a licensed series; the design's licensing rule
names BAA10Y and the FMP closes), so its label is advisory
(:data:`WARNING_LABELS`): a match is recorded in the manifest under
``guard.warnings`` by file, label and count, and the replay publishes. The 9.3
replay of the real records showed why: the rates are short two-decimal
numbers, and runs of three of them coincide with SVG style numbers, glyph
transform scale factors and the reliability diagram's bin edges (fix 9.3a).
The probes are counted in the manifest by label, never listed. With
``--require-guard`` (the publication run of task 9.5) the replay refuses before
writing anything unless the items, one FMP body per priced symbol and the DTB3
body were all there to probe; the DTB3 body stays required.

Matplotlib is imported only when the figures render, so importing this module
loads no plotting library.
"""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import math
import os
import re
import shutil
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from importlib import metadata
from pathlib import Path
from typing import Any

from scripts.finance_sim.analysis import analysis_from_records, write_analysis
from scripts.finance_sim.calibrate import CALIBRATORS_DIR
from scripts.finance_sim.convergence import CONVERGENCE_STAGES, convergence_from_records, write_table
from scripts.finance_sim.ensemble_metrics import ANALYSED_STAGES, ensemble_metrics_from_records, write_metrics
from scripts.finance_sim.execute import EXECUTION_STAGES, RESULTS_DIR, ExecutionDraw
from scripts.finance_sim.figures import (
    DEFAULT_ARMS,
    EDGAR_SOURCE,
    FIGURE_NAMES,
    FMP_SOURCE,
    FRED_SOURCE,
    render_figures,
)
from scripts.finance_sim.figures_convergence import render_convergence
from scripts.finance_sim.figures_draws import EXAMPLE_DEFAULT, render_draws
from scripts.finance_sim.items import (
    CASH_SERIES,
    DEFAULT_EDGAR_DIR,
    ITEMS_SCHEMA,
    PRICE_SYMBOLS,
    PRIVATE_ROOT,
    PrivatePaths,
    _month_index,
    _parse_closes,
    _parse_rates,
    load_edgar_items,
)
from scripts.finance_sim.limits import ARMS, SAMPLING_ARM
from scripts.finance_sim.negative_control import negative_control_from_records, write_report
from scripts.finance_sim.pilot import CONFIRMATORY_STAGE
from scripts.finance_sim.plan import StudyPlan, load_plan, require_complete
from scripts.finance_sim.recording import RECORDS_DIR, DrawLog, draw_log_path

#: The matplotlib version the published SVGs were rendered with (their ``dc:title`` metadata); a replay with
#: another version is refused, because the glyph paths and the layout would change the bytes.
MATPLOTLIB_VERSION = "3.10.9"
#: The creator string matplotlib writes into every SVG it saves.
SVG_CREATOR = f"Matplotlib v{MATPLOTLIB_VERSION}, https://matplotlib.org/"
#: The F3 example rule, frozen for the replay: figures_draws' default (no ``--example`` override is passed).
EXAMPLE_RULE = EXAMPLE_DEFAULT
#: F3 and F4 are drawn on the probe (sampling) arm (design: Figures, amendment after 8.2 and 8.3).
PROBE_ARM = SAMPLING_ARM
#: Under the output root: the private analysis documents (per-item rows) and the published files.
PRIVATE_DIR = Path("runs") / "finance-sim" / "analysis"
PUBLISHED_DIR = Path("docs") / "assets" / "finance-sim"
MANIFEST_NAME = "replay.json"
REPLAY_SCHEMA = "finance-sim-replay/1"
EXIT_WRITTEN, EXIT_DIFFERENT, EXIT_REFUSED = 0, 1, 2
#: The published file kinds the guard scans.
PUBLISHED_SUFFIXES: tuple[str, ...] = (".md", ".csv", ".svg")
#: A single value is probed only when its shortest float repr has this many characters (about six significant
#: digits): shorter values coincide with derived statistics; they are covered by the runs of consecutive values.
MIN_VALUE_PROBE = 8
#: A reply is probed only from this length: a one-component outlook reply such as ``{"outlook": 0.2}`` is the
#: prompt's JSON example, which the negative-control report states by design, so it cannot count as a leak.
MIN_TEXT_PROBE = 24
#: Consecutive values of one series probed together, and how many number tokens apart they may stand.
SERIES_WINDOW, SERIES_SPAN = 3, 80
#: The published copy of an analysis markdown carries this above the table. It is published text: the sources, what
#: was done and what stays private, with no task number, script path or private path.
PUBLISHED_HEADER = (
    f"Sources: {FRED_SOURCE}; {FMP_SOURCE}; {EDGAR_SOURCE}. Replayed offline from the recorded draws and scores with "
    "no model request; the per-item rows stay private. Derived statistics only: no reply text, no realized return "
    "and no raw licensed series.")
_SVG_CREATOR = re.compile(r"<dc:title>(Matplotlib v[^<]*)</dc:title>")
#: A number standing on its own: not part of an identifier, a hash, a date or a longer number.
_NUMBER = re.compile(r"(?<![0-9A-Za-z._-])-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?(?![0-9A-Za-z._])")
#: The private folders the item assembly fills (design: ``runs/finance-sim/``): the FMP bodies and the FRED cache.
PRIVATE_PATHS = PrivatePaths(PRIVATE_ROOT)
#: A cached FMP body, named by ``items.fetch_adjusted_closes``; a cached DTB3 body, named by ``fetch_cash_rates``.
_FMP_FILE = re.compile(r"(?P<symbol>[A-Za-z0-9][A-Za-z0-9.\-]*)_(?P<start>\d{4}-\d{2}-\d{2})_(?P<end>\d{4}-\d{2}-\d{2})"
                       r"\.json")
_DTB3_FILE = re.compile(re.escape(CASH_SERIES) + r"_current_[0-9a-f]{16}\.json")
_ANY_DAY = (date(1900, 1, 1), date(2199, 12, 31))
_RAW_LABELS: Mapping[str, str] = {"baa10y": "BAA10Y raw levels (FRED, Moody's)",
                                  "cpi_yoy": "CPI year-over-year raw levels (FRED)",
                                  "t10y2y": "T10Y2Y raw levels (FRED)"}
RETURNS_LABEL = "realized returns (from FMP adjusted closes and DTB3)"
CLOSES_LABEL = "FMP adjusted closes (licensed)"
DTB3_LABEL = "DTB3 rates (FRED, current vintage)"
REPLY_LABEL = "reply text of a recorded draw"
#: The labels whose matches are advisory: recorded in the manifest as ``guard.warnings`` (file, label, count),
#: never a refusal. DTB3 is the FRED H.15 three-month Treasury bill rate, public data and not a licensed series
#: (the design's licensing rule names BAA10Y and the FMP closes); its rates are still probed because the match
#: is informative. Every other label (the macro rows' raw FRED levels, BAA10Y among them, the realized returns,
#: the FMP closes and the reply text) removes the published tree and refuses.
WARNING_LABELS: frozenset[str] = frozenset({DTB3_LABEL})


class ReplayError(ValueError):
    """The replay refuses: nothing published, and the reason named."""


# ---------- the matplotlib pin ----------

def require_matplotlib(installed: str | None = None) -> str:
    """The installed matplotlib version, which must be :data:`MATPLOTLIB_VERSION`; refused otherwise.

    ``installed`` overrides the lookup (tests); the distribution metadata is
    read, so matplotlib itself is not imported here.
    """
    if installed is None:
        try:
            installed = metadata.version("matplotlib")
        except metadata.PackageNotFoundError:
            raise ReplayError(f"matplotlib is not installed; the published figures render with matplotlib "
                              f"{MATPLOTLIB_VERSION}") from None
    if installed != MATPLOTLIB_VERSION:
        raise ReplayError(f"matplotlib {installed} is installed, but the published figures were rendered with "
                          f"matplotlib {MATPLOTLIB_VERSION} (the version the SVG metadata carries); another version "
                          "changes the SVG bytes, so the replay refuses. Install the locked version, or re-render "
                          "every figure and move the pin in scripts/finance_sim/replay.py deliberately")
    return installed


def svg_version(text: str) -> str:
    """The creator string of an SVG matplotlib saved (``Matplotlib v<version>, <url>``)."""
    match = _SVG_CREATOR.search(text)
    if match is None:
        raise ReplayError("the SVG carries no matplotlib creator metadata")
    return match.group(1)


def check_svg_version(text: str, where: str | Path = "the SVG") -> str:
    """Refuse an SVG whose metadata names another matplotlib version than the pin; returns the creator string."""
    found = svg_version(text)
    if found != SVG_CREATOR:
        raise ReplayError(f"{where} was rendered by {found!r}, not {SVG_CREATOR!r}")
    return found


# ---------- the licensing guard ----------

@dataclass(frozen=True)
class Probe:
    """What the guard looks for: one distinctive number, a run of consecutive numbers, or a text."""

    label: str
    kind: str
    texts: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.kind not in ("value", "series", "text") or not self.texts:
            raise ValueError("a probe is a value, a series or a text with at least one entry")


@dataclass(frozen=True)
class Leak:
    """A probe found in a published file.

    ``found`` is the matched text, kept for the caller's inspection in process;
    it is never printed: :meth:`__str__`, the replay's refusal and its stderr
    name the file and the series only.
    """

    path: Path
    label: str
    found: str

    def __str__(self) -> str:
        return f"{self.path.name}: {self.label}"


def leak_summary(leaks: Sequence[Leak]) -> str:
    """The leaks by file and series with their counts, without any matched value."""
    counts: dict[str, int] = {}
    for leak in leaks:
        counts[str(leak)] = counts.get(str(leak), 0) + 1
    return "; ".join(f"{key} ({count})" for key, count in counts.items())


def partition_leaks(leaks: Sequence[Leak]) -> tuple[list[Leak], list[Leak]]:
    """The leaks that refuse the replay, then the advisory ones (a label in :data:`WARNING_LABELS`), in order."""
    refusing = [leak for leak in leaks if leak.label not in WARNING_LABELS]
    warnings = [leak for leak in leaks if leak.label in WARNING_LABELS]
    return refusing, warnings


def warning_records(leaks: Sequence[Leak]) -> list[dict[str, Any]]:
    """The manifest's ``guard.warnings``: one record per file and label with the match count, sorted; no value."""
    counts: dict[tuple[str, str], int] = {}
    for leak in leaks:
        key = (leak.path.name, leak.label)
        counts[key] = counts.get(key, 0) + 1
    return [{"file": file, "label": label, "count": count} for (file, label), count in sorted(counts.items())]


def _number_text(value: object) -> str | None:
    """The canonical text of a finite number: the shortest float repr, so ``1.930`` and ``1.93`` agree."""
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        return None
    return repr(float(value))


def value_probe(label: str, value: object) -> Probe | None:
    """A probe for one number, or None when the number is too short to be distinctive."""
    text = _number_text(value)
    if text is None or len(text) < MIN_VALUE_PROBE:
        return None
    return Probe(label, "value", (text,))


def series_probe(label: str, values: Sequence[object]) -> Probe | None:
    """A probe for consecutive values of one series in order (each may be short), or None without two numbers."""
    texts = [_number_text(value) for value in values]
    if len(texts) < 2 or any(text is None for text in texts):
        return None
    return Probe(label, "series", tuple(text for text in texts if text is not None))


def text_probe(label: str, text: str) -> Probe | None:
    """A probe for a text (a reply), or None when it is shorter than :data:`MIN_TEXT_PROBE`."""
    flat = text.strip()
    if len(flat) < MIN_TEXT_PROBE:
        return None
    return Probe(label, "text", (flat,))


def items_probes(record: Mapping[str, Any]) -> list[Probe]:
    """The probes of a private ``items.json`` (:func:`scripts.finance_sim.items.write_items`): raw rows and returns.

    Per macro series (``baa10y``, ``cpi_yoy``, ``t10y2y``, in month order) and
    per realized-return category: every distinctive value and every run of
    :data:`SERIES_WINDOW` consecutive values.
    """
    if not isinstance(record, Mapping) or record.get("schema") != ITEMS_SCHEMA:
        raise ReplayError(f"items.json must carry the schema {ITEMS_SCHEMA!r}")
    entries = record.get("macro_items")
    if not isinstance(entries, list):
        raise ReplayError("items.json must list macro_items")
    raw: dict[str, list[object]] = {}
    returns: dict[str, list[object]] = {}
    for entry in entries:
        row = entry.get("row") if isinstance(entry, Mapping) else None
        values = row.get("raw") if isinstance(row, Mapping) else None
        if not isinstance(values, Mapping):
            raise ReplayError("every macro item of items.json carries row.raw")
        for series, value in values.items():
            raw.setdefault(str(series), []).append(value)
        realized = entry.get("realized")
        if isinstance(realized, Mapping):
            for category, value in realized.items():
                returns.setdefault(str(category), []).append(value)
    probes: list[Probe] = []
    for series, values in raw.items():
        probes += _sequence_probes(_RAW_LABELS.get(series, f"{series} raw levels (FRED)"), values)
    for values in returns.values():
        probes += _sequence_probes(RETURNS_LABEL, values)
    return probes


def _sequence_probes(label: str, values: Sequence[object]) -> list[Probe]:
    probes = [probe for value in values if (probe := value_probe(label, value)) is not None]
    for start in range(0, max(0, len(values) - SERIES_WINDOW + 1)):
        probe = series_probe(label, values[start:start + SERIES_WINDOW])
        if probe is not None:
            probes.append(probe)
    return probes


def _cached_body(path: Path, what: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ReplayError(f"the cached {what} body {path.name} cannot be read: {exc}") from None


def month_end_closes(closes: Sequence[tuple[date, float]]) -> list[float]:
    """The study's month-end series of one symbol: the last close of each calendar month, in month order.

    The rule of :func:`scripts.finance_sim.items.realized_returns`: closes
    oldest first, the month's last close wins.
    """
    by_month: dict[int, float] = {}
    for day, value in closes:
        by_month[_month_index(day)] = float(value)
    return [by_month[index] for index in sorted(by_month)]


def _distinct(probes: Iterable[Probe]) -> list[Probe]:
    unique: dict[tuple[str, str, tuple[str, ...]], Probe] = {}
    for probe in probes:
        unique.setdefault((probe.label, probe.kind, probe.texts), probe)
    return list(unique.values())


def fmp_probes(cache_dir: str | Path, *, month_ends: bool = True) -> tuple[list[Probe], int]:
    """Probes for the adjusted closes of every cached FMP body under ``cache_dir``, and the number of bodies.

    Each body (``<symbol>_<start>_<end>.json``) is read with the item
    assembly's parser, so a body that would not assemble refuses the replay.
    Per symbol the daily closes are probed in date order and, with
    ``month_ends`` (the default), so is the study's month-end series
    (:func:`month_end_closes`): in a daily body three consecutive month-end
    closes stand about twenty-one closes apart, so a published month-end run
    is found only through that series. Duplicate probes are kept once. A
    missing folder gives no probe and a count of zero.
    """
    cache = Path(cache_dir)
    if not cache.is_dir():
        return [], 0
    probes: list[Probe] = []
    bodies = 0
    for path in sorted(cache.iterdir()):
        match = _FMP_FILE.fullmatch(path.name)
        if match is None or not path.is_file():
            continue
        start, end = (date.fromisoformat(match.group(name)) for name in ("start", "end"))
        try:
            closes = _parse_closes(_cached_body(path, "FMP"), match.group("symbol"), start, end)
        except ValueError as exc:
            raise ReplayError(f"the cached FMP body {path.name} cannot be read: {exc}") from None
        probes += _sequence_probes(CLOSES_LABEL, [value for _, value in closes])
        if month_ends:
            probes += _sequence_probes(CLOSES_LABEL, month_end_closes(closes))
        bodies += 1
    return _distinct(probes), bodies


def dtb3_probes(cache_dir: str | Path) -> tuple[list[Probe], int]:
    """Probes for the DTB3 rates of every cached FRED body under ``cache_dir``, and the number of bodies."""
    cache = Path(cache_dir)
    if not cache.is_dir():
        return [], 0
    probes: list[Probe] = []
    bodies = 0
    for path in sorted(cache.iterdir()):
        if _DTB3_FILE.fullmatch(path.name) is None or not path.is_file():
            continue
        try:
            rates = _parse_rates(_cached_body(path, "FRED"), *_ANY_DAY)
        except ValueError as exc:
            raise ReplayError(f"the cached FRED body {path.name} cannot be read: {exc}") from None
        probes += _sequence_probes(DTB3_LABEL, [value for _, value in rates])
        bodies += 1
    return probes, bodies


def reply_probes(plan: StudyPlan, records_dir: str | Path = RECORDS_DIR) -> list[Probe]:
    """A text probe per distinct reply in the draw logs of every arm and stage under ``records_dir``."""
    replies: dict[str, None] = {}
    for arm in ARMS:
        for stage in EXECUTION_STAGES:
            try:
                path = draw_log_path(arm, stage, records_dir)
            except ValueError:
                continue  # a stage the arm does not run (the serving arm has no pilot)
            if not path.exists():
                continue
            for draw in DrawLog(path, plan=plan, loads=ExecutionDraw.from_dict).latest():
                if draw.response is not None:
                    replies.setdefault(draw.response.content)
    return [probe for text in replies if (probe := text_probe(REPLY_LABEL, text)) is not None]


def number_tokens(content: str) -> tuple[str, ...]:
    """Every number standing on its own in ``content``, as canonical text, in file order (non-finite ones left out)."""
    texts = (_number_text(float(match.group())) for match in _NUMBER.finditer(content))
    return tuple(text for text in texts if text is not None)


def _tokens(content: str) -> dict[str, list[int]]:
    """Positions of every number token of a file by canonical text, in file order."""
    positions: dict[str, list[int]] = {}
    for index, text in enumerate(number_tokens(content)):
        positions.setdefault(text, []).append(index)
    return positions


def _in_order(positions: Mapping[str, Sequence[int]], texts: Sequence[str], span: int) -> bool:
    """Whether the texts occur as number tokens in order, each within ``span`` tokens of the previous one."""
    if any(text not in positions for text in texts):
        return False
    for start in positions[texts[0]]:
        last = start
        for text in texts[1:]:
            found = positions[text]
            index = bisect.bisect_right(found, last)
            if index == len(found) or found[index] - last > span:
                last = None
                break
            last = found[index]
        if last is not None:
            return True
    return False


def find_leaks(paths: Iterable[str | Path], probes: Iterable[Probe]) -> list[Leak]:
    """Every probe found in the text files ``paths`` (markdown, CSV, SVG), in path order."""
    probes = tuple(probes)
    leaks = []
    for path in (Path(path) for path in paths):
        content = path.read_text(encoding="utf-8")
        positions = _tokens(content)
        for probe in probes:
            if probe.kind == "text":
                if probe.texts[0] in content:
                    leaks.append(Leak(path, probe.label, probe.texts[0]))
            elif probe.kind == "value":
                if probe.texts[0] in positions:
                    leaks.append(Leak(path, probe.label, probe.texts[0]))
            elif _in_order(positions, probe.texts, SERIES_SPAN):
                leaks.append(Leak(path, probe.label, ", ".join(probe.texts)))
    return leaks


# ---------- the replay ----------

@dataclass(frozen=True)
class ReplayInputs:
    """Where the replay reads: the frozen plan and the private records of the study (nothing is written there)."""

    plan: StudyPlan
    results_dir: Path = RESULTS_DIR
    records_dir: Path = RECORDS_DIR
    calibrators_dir: Path = CALIBRATORS_DIR
    edgar_dir: Path = DEFAULT_EDGAR_DIR
    items_path: Path | None = None
    fmp_cache: Path = PRIVATE_PATHS.fmp_cache
    fred_cache: Path = PRIVATE_PATHS.fred_cache


@dataclass(frozen=True)
class GuardProbes:
    """The licensing probes of one replay and which private sources supplied them.

    ``fmp_bodies`` counts the cached FMP bodies read (one per priced symbol
    when the assembly ran), ``dtb3_bodies`` the cached DTB3 bodies.
    """

    probes: tuple[Probe, ...]
    items_probed: bool
    closes_probed: bool
    dtb3_probed: bool
    fmp_bodies: int = 0
    dtb3_bodies: int = 0

    def counts(self) -> dict[str, int]:
        """Probes per label, sorted by label (what the manifest records; never a value)."""
        counts: dict[str, int] = {}
        for probe in self.probes:
            counts[probe.label] = counts.get(probe.label, 0) + 1
        return dict(sorted(counts.items()))


@dataclass(frozen=True)
class ReplayOutput:
    """What one replay wrote: the root, its manifest and the private and published files."""

    root: Path
    manifest: Mapping[str, Any]
    private: tuple[Path, ...]
    published: tuple[Path, ...]


def fresh_root(path: str | Path) -> Path:
    """``path`` as the output root: a new directory, or an existing empty one; never a drive root."""
    root = Path(path)
    resolved = root.resolve()
    if resolved.parent == resolved:
        raise ReplayError(f"the output root must not be a drive or filesystem root; got {root}")
    if root.exists():
        if not root.is_dir() or any(root.iterdir()):
            raise ReplayError(f"the output root must be a new or empty directory; {root} is not")
    else:
        root.mkdir(parents=True)
    return root


def published_markdown(text: str) -> str:
    """The published copy of an analysis markdown: the sources and the derived-only statement above it."""
    return f"{PUBLISHED_HEADER}\n\n{text}"


def _write_text(path: Path, text: str) -> None:
    """Write atomically: a crash leaves the previous file or the new one, never half of one."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _relative(paths: Iterable[Path], root: Path) -> dict[str, str]:
    return {path.relative_to(root).as_posix(): _sha256(path) for path in sorted(paths)}


def _read_items(path: Path) -> Mapping[str, Any]:
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ReplayError(f"items.json at {path} cannot be read: {exc}") from None
    if not isinstance(record, Mapping):
        raise ReplayError(f"items.json at {path} must hold an object")
    return record


def replay_probes(inputs: ReplayInputs) -> GuardProbes:
    """The guard's probes for one replay: the items' raw series and returns (when given), the cached FMP closes,
    the cached DTB3 rates and every recorded reply."""
    probes = [] if inputs.items_path is None else items_probes(_read_items(Path(inputs.items_path)))
    closes, fmp_bodies = fmp_probes(inputs.fmp_cache)
    rates, fred_bodies = dtb3_probes(inputs.fred_cache)
    replies = reply_probes(inputs.plan, inputs.records_dir)
    return GuardProbes(probes=tuple(probes + closes + rates + replies), items_probed=inputs.items_path is not None,
                       closes_probed=fmp_bodies > 0, dtb3_probed=fred_bodies > 0, fmp_bodies=fmp_bodies,
                       dtb3_bodies=fred_bodies)


def check_guard_inputs(guard: GuardProbes) -> None:
    """Refuse unless every private source was there to probe: the items, one FMP body per priced symbol, DTB3."""
    missing = []
    if not guard.items_probed:
        missing.append("items.json (--items)")
    if guard.fmp_bodies != len(PRICE_SYMBOLS):
        missing.append(f"{len(PRICE_SYMBOLS)} cached FMP bodies, one per priced symbol (--fmp-cache holds "
                       f"{guard.fmp_bodies})")
    if not guard.dtb3_probed:
        missing.append(f"the cached {CASH_SERIES} body (--fred-cache)")
    if missing:
        raise ReplayError("the licensing guard requires every private source and is missing: " + "; ".join(missing))


def replay(inputs: ReplayInputs, out_root: str | Path, *, require_guard: bool = False) -> ReplayOutput:
    """Rebuild the analyses, the negative-control report, the convergence tables and the five figures offline.

    Order: the matplotlib pin, the licensing probes (with ``require_guard``,
    every private source must be there: :func:`check_guard_inputs`) and the
    fresh root are checked before anything is written; the private documents
    are written (both arms' analyses and ensemble metrics, both arms'
    negative-control reports, the probe arm's convergence tables); the figures
    and the published markdown copies follow; then every SVG's metadata is
    checked against the pin and every published file against the probes. A
    leak on a refusing label removes the published tree and refuses; the
    private documents stay for inspection. A match on an advisory label
    (:data:`WARNING_LABELS`) is recorded in the manifest under
    ``guard.warnings`` by file, label and count, and the replay publishes. No
    request is sent: nothing here builds a client.
    """
    if not isinstance(inputs, ReplayInputs):
        raise TypeError(f"inputs must be ReplayInputs; got {type(inputs).__name__}")
    plan = require_complete(inputs.plan)
    version = require_matplotlib()
    guard = replay_probes(inputs)
    if require_guard:
        check_guard_inputs(guard)
    root = fresh_root(out_root)
    private, published = root / PRIVATE_DIR, root / PUBLISHED_DIR
    releases = load_edgar_items(inputs.edgar_dir).negative_control
    private_paths: list[Path] = []
    for arm in DEFAULT_ARMS:
        private_paths += write_analysis(analysis_from_records(plan, arm=arm, results_dir=inputs.results_dir),
                                        private)
        for stage in ANALYSED_STAGES:
            private_paths += write_metrics(
                ensemble_metrics_from_records(plan, arm=arm, stage=stage, results_dir=inputs.results_dir), private)
        private_paths += write_report(
            negative_control_from_records(plan, releases, arm=arm, results_dir=inputs.results_dir), private)
    for stage in CONVERGENCE_STAGES:
        private_paths += write_table(
            convergence_from_records(plan, arm=PROBE_ARM, stage=stage, results_dir=inputs.results_dir,
                                     records_dir=inputs.records_dir, calibrators_dir=inputs.calibrators_dir),
            private)
    published_paths = list(render_figures(plan, arms=DEFAULT_ARMS, figures=FIGURE_NAMES, analysis_dir=private,
                                          calibrators_dir=inputs.calibrators_dir, out_dir=published))
    published_paths += render_draws(plan, arm=PROBE_ARM, stage=CONFIRMATORY_STAGE, analysis_dir=private,
                                    results_dir=inputs.results_dir, example=None, out_dir=published)
    published_paths += render_convergence(plan, arm=PROBE_ARM, stages=CONVERGENCE_STAGES, analysis_dir=private,
                                          out_dir=published)
    for path in private_paths:
        if path.suffix == ".md":
            target = published / path.name
            _write_text(target, published_markdown(path.read_text(encoding="utf-8")))
            published_paths.append(target)
    for path in published_paths:
        if path.suffix == ".svg":
            check_svg_version(path.read_text(encoding="utf-8"), path)
    refusing, warnings = partition_leaks(
        find_leaks([path for path in published_paths if path.suffix in PUBLISHED_SUFFIXES], guard.probes))
    if refusing:
        shutil.rmtree(root / PUBLISHED_DIR.parts[0])
        raise ReplayError(f"{len(refusing)} licensing probe(s) matched a published file, so nothing is published "
                          f"(file: series (matches)): {leak_summary(refusing)}")
    manifest = {
        "schema": REPLAY_SCHEMA, "plan_sha256": plan.plan_sha256, "matplotlib": version,
        "svg_creator": SVG_CREATOR, "arms": list(DEFAULT_ARMS), "probe_arm": PROBE_ARM,
        "example_rule": EXAMPLE_RULE, "analysed_stages": list(ANALYSED_STAGES),
        "convergence_stages": list(CONVERGENCE_STAGES), "private_dir": PRIVATE_DIR.as_posix(),
        "published_dir": PUBLISHED_DIR.as_posix(), "private": _relative(private_paths, root),
        "published": _relative(published_paths, root),
        "guard": {"required": require_guard, "items_probed": guard.items_probed,
                  "closes_probed": guard.closes_probed, "dtb3_probed": guard.dtb3_probed,
                  "fmp_bodies": guard.fmp_bodies, "priced_symbols": len(PRICE_SYMBOLS),
                  "dtb3_bodies": guard.dtb3_bodies, "probes": guard.counts(),
                  "files_scanned": sum(1 for path in published_paths if path.suffix in PUBLISHED_SUFFIXES),
                  "leaks": 0, "warning_labels": sorted(WARNING_LABELS), "warnings": warning_records(warnings)},
    }
    _write_text(root / MANIFEST_NAME, json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return ReplayOutput(root=root, manifest=manifest, private=tuple(private_paths), published=tuple(published_paths))


def compare_published(output: ReplayOutput, committed: str | Path) -> list[str]:
    """The published files of ``output`` that differ from, or are missing in, the directory ``committed``, then
    every file there that the replay did not produce (stale), each marked ``(stale)``."""
    base = output.root / PUBLISHED_DIR
    other = Path(committed)
    differing = []
    produced = set()
    for path in output.published:
        relative = path.relative_to(base).as_posix()
        produced.add(relative)
        candidate = other / relative
        if not candidate.is_file() or candidate.read_bytes() != path.read_bytes():
            differing.append(relative)
    stale = sorted(path.relative_to(other).as_posix() for path in other.rglob("*")
                   if path.is_file() and path.relative_to(other).as_posix() not in produced)
    return differing + [f"{name} (stale)" for name in stale]


def main(argv: Sequence[str] | None = None) -> int:
    """Replay the worked example offline into a fresh output root.

    Exit codes: 0 written (every file listed on stdout), 1 written but a
    published file differs from the ``--compare`` directory (the differing
    files listed), 2 refused (another plan, an unpinned matplotlib, a used
    output root, records that do not belong to the plan, or a licensing probe
    on a refusing label found in a published file). Advisory matches
    (``guard.warnings`` in the manifest) are summarised on stderr by file,
    label and count and do not change the exit code.
    """
    parser = argparse.ArgumentParser(description="Offline worked-example replay (task 8.4): both pre-registered "
                                                 "analyses, the negative-control report, the convergence tables and "
                                                 "the five figures from recorded draws and scores; sends no request.")
    parser.add_argument("--plan", type=Path, required=True, help="the frozen plan.json")
    parser.add_argument("--plan-sha256", required=True, help="the plan hash the records carry")
    parser.add_argument("--out-root", type=Path, required=True,
                        help="a new or empty directory; gets runs/finance-sim/analysis (private), "
                             "docs/assets/finance-sim (published) and replay.json")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--records-dir", type=Path, default=RECORDS_DIR)
    parser.add_argument("--calibrators-dir", type=Path, default=CALIBRATORS_DIR)
    parser.add_argument("--edgar-dir", type=Path, default=DEFAULT_EDGAR_DIR, help="the frozen EDGAR corpus")
    parser.add_argument("--items", type=Path, help="the private items.json; its raw series and realized returns "
                                                   "are probed in every published file")
    parser.add_argument("--fmp-cache", type=Path, default=PRIVATE_PATHS.fmp_cache,
                        help="the private FMP folder; every cached body's adjusted closes are probed")
    parser.add_argument("--fred-cache", type=Path, default=PRIVATE_PATHS.fred_cache,
                        help="the FRED loader cache; the cached DTB3 body's rates are probed")
    parser.add_argument("--compare", type=Path, help="a committed docs/assets/finance-sim to compare the published "
                                                     "files with, byte for byte")
    parser.add_argument("--require-guard", action="store_true",
                        help="refuse before writing anything unless items.json, one cached FMP body per priced "
                             "symbol and the cached DTB3 body were all probed (the publication run)")
    args = parser.parse_args(argv)
    try:
        plan = load_plan(args.plan, expected_sha256=args.plan_sha256)
        output = replay(ReplayInputs(plan, results_dir=args.results_dir, records_dir=args.records_dir,
                                     calibrators_dir=args.calibrators_dir, edgar_dir=args.edgar_dir,
                                     items_path=args.items, fmp_cache=args.fmp_cache, fred_cache=args.fred_cache),
                        args.out_root, require_guard=args.require_guard)
        differing = [] if args.compare is None else compare_published(output, args.compare)
    except (ValueError, TypeError, OSError) as exc:
        print(f"replay: refused: {exc}", file=sys.stderr)
        return EXIT_REFUSED
    for path in (*output.private, *output.published, output.root / MANIFEST_NAME):
        print(path)
    warnings = output.manifest["guard"]["warnings"]
    if warnings:
        summary = "; ".join(f"{record['file']}: {record['label']} ({record['count']})" for record in warnings)
        print(f"replay: {sum(record['count'] for record in warnings)} advisory licensing match(es) recorded in "
              f"{MANIFEST_NAME} as guard.warnings (file: series (matches)): {summary}", file=sys.stderr)
    if differing:
        print(f"replay: {len(differing)} published file(s) differ from {args.compare}: {', '.join(differing)}",
              file=sys.stderr)
        return EXIT_DIFFERENT
    return EXIT_WRITTEN


__all__ = [
    "CLOSES_LABEL",
    "DTB3_LABEL",
    "EXAMPLE_RULE",
    "MANIFEST_NAME",
    "MATPLOTLIB_VERSION",
    "MIN_TEXT_PROBE",
    "MIN_VALUE_PROBE",
    "PRIVATE_DIR",
    "PRIVATE_PATHS",
    "PROBE_ARM",
    "PUBLISHED_DIR",
    "PUBLISHED_HEADER",
    "PUBLISHED_SUFFIXES",
    "REPLAY_SCHEMA",
    "REPLY_LABEL",
    "RETURNS_LABEL",
    "SERIES_SPAN",
    "SERIES_WINDOW",
    "SVG_CREATOR",
    "WARNING_LABELS",
    "GuardProbes",
    "Leak",
    "Probe",
    "ReplayError",
    "ReplayInputs",
    "ReplayOutput",
    "check_guard_inputs",
    "check_svg_version",
    "compare_published",
    "dtb3_probes",
    "find_leaks",
    "fmp_probes",
    "fresh_root",
    "items_probes",
    "leak_summary",
    "main",
    "month_end_closes",
    "number_tokens",
    "partition_leaks",
    "published_markdown",
    "replay",
    "replay_probes",
    "reply_probes",
    "require_matplotlib",
    "series_probe",
    "svg_version",
    "text_probe",
    "value_probe",
    "warning_records",
]


if __name__ == "__main__":
    raise SystemExit(main())
