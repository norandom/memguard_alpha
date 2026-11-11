"""Render the catalogue page (``docs/catalogue.md``) from the catalogue data.

The page lets a reader choose a model, serving route and evidence mode without
reading the research reports. It states the qualification rule once, then one
table row per entry and mode, the entries on routes without logprobs, and the
entries that share weights. Every status is derived with
:func:`recall_guard.core.catalogue.qualify`; the thresholds in the rule text
are the constants of that module. Recorded numbers are printed as recorded,
and a value that is not recorded is never printed as a number.

Rendering is a pure function of the catalogue: the same catalogue gives the
same bytes.

Usage::

    python -m scripts.catalogue.render_catalogue [--catalogue PATH] [--out PATH]
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from decimal import Decimal
from pathlib import Path

from recall_guard.core.catalogue import (
    ALPHA,
    CHECK_NAMES,
    MIN_AUC,
    MIN_VALID,
    QUALIFYING_DOMAIN,
    QUALIFYING_METHOD,
    Catalogue,
    CatalogueEntry,
    CatalogueError,
    Mode,
    ModeRecord,
    Status,
    load_catalogue,
    qualify,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "docs" / "catalogue.md"
REGISTER_PAGE = "research-register.md"
NOT_RECORDED = "not recorded"

_STATUS_LABEL = {
    Status.QUALIFIED: "qualified",
    Status.NOT_QUALIFIED: "not qualified",
    Status.NOT_TESTED: "not tested",
}

_CHECK_TEXT = {
    "boundary": (
        f"the calibrator's recall boundary is `{QUALIFYING_DOMAIN}` and its evidence method is `{QUALIFYING_METHOD}`."
    ),
    "sample": f"there are at least {MIN_VALID} valid draws in each class (exposed and fenced).",
    "discrimination": f"the held-out AUC is at least {MIN_AUC:g} and the permutation p is below {ALPHA:g}.",
    "cutoff": (
        "the recall estimate inside the training range is above the one after the cutoff, "
        f"with a one-sided p below {ALPHA:g}."
    ),
}

_RECORD_COLUMNS = (
    "Entry",
    "Model",
    "Route",
    "Mode",
    "Available",
    "Recall boundary",
    "Held-out AUC [interval]",
    "Interval method, confidence",
    "Sample sizes",
    "Permutation p",
    "Cutoff check",
    "Status",
    "Evidence",
)


def _number(value: float | int) -> str:
    """A recorded number with its recorded digits, never in scientific notation.

    The text converts back to the recorded value: an integer stays an integer
    and a float keeps the shortest digits that identify it.
    """
    if isinstance(value, int):
        return str(value)
    text = format(Decimal(repr(value)), "f")
    return text if "." in text else f"{text}.0"


def _cell(text: str | None) -> str:
    """Text made safe for one table cell; a missing text is an empty cell."""
    if text is None:
        return ""
    return " ".join(str(text).split()).replace("|", "\\|")


def _table(columns: Sequence[str], rows: Sequence[Sequence[str | None]]) -> list[str]:
    lines = ["| " + " | ".join(columns) + " |", "|" + " --- |" * len(columns)]
    for row in rows:
        cells = [_cell(value) for value in row]
        lines.append("|" + "".join(f" {cell} |" if cell else " |" for cell in cells))
    return lines


def _count(number: int, singular: str, plural: str) -> str:
    return f"{number} {singular if number == 1 else plural}"


def _status(record: ModeRecord, *, reason: bool = True) -> str:
    outcome = qualify(record)
    label = _STATUS_LABEL[outcome.status]
    return f"{label}: {outcome.reason}" if reason else label


def _available(record: ModeRecord) -> str:
    if record.available:
        return "yes"
    if record.unavailable_reason is None:
        return "no (reason not recorded)"
    return f"no: {record.unavailable_reason}"


def _boundary(record: ModeRecord) -> str | None:
    boundary = record.boundary
    if boundary is None:
        return None
    return f"{boundary.domain}; exposed: {boundary.exposed}; fenced: {boundary.fenced}; range: {boundary.date_range}"


def _quality_cells(record: ModeRecord) -> list[str | None]:
    quality = record.quality
    if quality is None:
        return [None, None, None, None]
    sizes = f"exposed {_number(quality.n_exposed)}, fenced {_number(quality.n_fenced)}"
    if quality.n_prompts is not None:
        sizes += f", prompts {_number(quality.n_prompts)}"
    return [
        f"{_number(quality.holdout_auc)} [{_number(quality.interval_low)}, {_number(quality.interval_high)}]",
        f"{quality.interval_method}, {_number(quality.confidence)}",
        sizes,
        None if quality.permutation_p is None else _number(quality.permutation_p),
    ]


def _cutoff(record: ModeRecord) -> str | None:
    cutoff = record.cutoff
    if cutoff is None:
        return None
    return (
        f"inside {_number(cutoff.inside_mean)}, after {_number(cutoff.after_mean)}, "
        f"difference {_number(cutoff.difference)} "
        f"[{_number(cutoff.interval_low)}, {_number(cutoff.interval_high)}], "
        f"p {_number(cutoff.p_value)}, n {_number(cutoff.n_inside)} and {_number(cutoff.n_after)}"
    )


def _evidence(record: ModeRecord) -> str | None:
    if record.evidence is None:
        return None
    return ", ".join(f"[{register_id}]({REGISTER_PAGE})" for register_id in record.evidence.register_ids)


def _record_row(entry: CatalogueEntry, record: ModeRecord) -> list[str | None]:
    return [
        entry.entry_id,
        entry.model,
        entry.route,
        record.mode.value,
        _available(record),
        _boundary(record),
        *_quality_cells(record),
        _cutoff(record),
        _status(record),
        _evidence(record),
    ]


def _records(catalogue: Catalogue) -> list[tuple[CatalogueEntry, ModeRecord]]:
    """Every entry and mode, in catalogue order and in the order of ``Mode``."""
    return [(entry, entry.modes[mode]) for entry in catalogue.entries for mode in Mode if mode in entry.modes]


def _summary(catalogue: Catalogue) -> list[str]:
    records = _records(catalogue)
    counts = {status: 0 for status in Status}
    for _, record in records:
        counts[qualify(record).status] += 1
    parts = [f"{counts[status]} {'is' if counts[status] == 1 else 'are'} {_STATUS_LABEL[status]}" for status in Status]
    return [
        f"This catalogue holds {_count(len(catalogue.entries), 'entry', 'entries')}, each a model on a serving "
        "route, and records two evidence modes for each: logprobs and logits. "
        f"Of the {_count(len(records), 'entry and mode combination', 'entry and mode combinations')}, "
        f"{', '.join(parts[:-1])} and {parts[-1]}. "
        "A status is derived from the recorded results by the rule below; it is never entered by hand.",
    ]


def _rule() -> list[str]:
    lines = [
        "## Qualification rule",
        "",
        f"The same rule applies to every entry and mode. It has {len(CHECK_NAMES)} checks, read in this order:",
        "",
    ]
    for index, name in enumerate(CHECK_NAMES, start=1):
        lines.append(f"{index}. **{name}**: {_CHECK_TEXT.get(name, 'see the rule in the package.')}")
    lines += [
        "",
        "The status is decided in this order, and the first case that applies gives it:",
        "",
        f"1. A mode with no held-out quality record is **{_STATUS_LABEL[Status.NOT_TESTED]}**, "
        "whatever else is recorded.",
        "2. Otherwise, when a quality record exists and a check fails on the recorded values, "
        f"the mode is **{_STATUS_LABEL[Status.NOT_QUALIFIED]}**. "
        "The reason names the first failing check, in the order of the checks above.",
        "3. Otherwise, when an input of a check is not recorded, "
        f"the mode is **{_STATUS_LABEL[Status.NOT_TESTED]}**. "
        "The reason names the first missing input, in the same order.",
        f"4. Otherwise all {len(CHECK_NAMES)} checks pass and the mode is **{_STATUS_LABEL[Status.QUALIFIED]}**.",
        "",
        "A missing result is never read as a failure and is never inferred from another entry or mode.",
    ]
    return lines


def _catalogue_table(catalogue: Catalogue) -> list[str]:
    records = _records(catalogue)
    lines = [
        "## Catalogue table",
        "",
        "One row per entry and mode. The recall boundary names the domain, what stands for exposed and for "
        "fenced answers, and the date range relative to the training cutoff. The interval belongs to the "
        "held-out AUC. Evidence ids refer to entries of the research register. "
        "An empty cell means that the value is not recorded.",
        "",
        *_table(_RECORD_COLUMNS, [_record_row(entry, record) for entry, record in records]),
        "",
        "### Other recorded evidence",
        "",
    ]
    others = [(entry, other) for entry, record in records for other in record.other_evidence]
    if not others:
        return [*lines, "No further evidence is recorded."]
    rows = []
    for entry, other in others:
        row = _record_row(entry, other)
        method = NOT_RECORDED if other.evidence is None else other.evidence.method
        rows.append([*row[:4], method, *row[4:]])
    return [
        *lines,
        "Further results recorded for an entry and mode, each with the method that produced it. "
        "These records do not set the status of a mode. The status column shows what the rule gives "
        "for the record taken on its own.",
        "",
        *_table([*_RECORD_COLUMNS[:4], "Method", *_RECORD_COLUMNS[4:]], rows),
    ]


def _text(value: str | None) -> str:
    return NOT_RECORDED if value is None else value


def _without_logprobs(catalogue: Catalogue) -> list[str]:
    lines = ["## Entries without logprobs", "", "### Logprobs mode unavailable", ""]
    lacking = [entry for entry in catalogue.entries if not entry.modes[Mode.LOGPROBS].available]
    if lacking:
        rows = []
        for entry in lacking:
            logits = entry.modes[Mode.LOGITS]
            rows.append(
                [
                    entry.entry_id,
                    entry.model,
                    entry.route,
                    _text(entry.modes[Mode.LOGPROBS].unavailable_reason),
                    _available(logits),
                    _status(logits),
                    _text(logits.requires),
                ]
            )
        lines += [
            "For each entry whose route returns no logprobs: why, whether the logits mode is available, "
            "and what the logits mode requires.",
            "",
            *_table(
                (
                    "Entry",
                    "Model",
                    "Route",
                    "Why the logprobs mode is unavailable",
                    "Logits mode available",
                    "Logits status",
                    "What the logits mode requires",
                ),
                rows,
            ),
        ]
    else:
        lines.append("No entry has an unavailable logprobs mode.")

    lines += ["", "### Not usable for recall-aware answers", ""]
    unusable = [entry for entry in catalogue.entries if not catalogue.usable(entry.entry_id)]
    if unusable:
        lines += [
            "Neither mode is available for these entries, so they are not usable for recall-aware answers. "
            "The reasons are in the table above.",
            "",
            *_table(("Entry", "Model", "Route"), [[entry.entry_id, entry.model, entry.route] for entry in unusable]),
        ]
    else:
        lines.append("Every entry has at least one available mode.")

    lines += ["", "### Entries with a logits mode", ""]
    usable = catalogue.without_logprobs()
    if usable:
        lines += [
            "Entries whose logits mode is available, with the status of that mode. "
            "These are the entries that can be used when logprobs are not wanted or not available. "
            "Only a qualified logits mode gives a recall-aware answer.",
            "",
            *_table(
                (
                    "Entry",
                    "Model",
                    "Route",
                    "Logprobs mode available",
                    "Logits status",
                    "What the logits mode requires",
                ),
                [
                    [
                        entry.entry_id,
                        entry.model,
                        entry.route,
                        _available(entry.modes[Mode.LOGPROBS]),
                        _status(entry.modes[Mode.LOGITS]),
                        _text(entry.modes[Mode.LOGITS].requires),
                    ]
                    for entry in usable
                ],
            ),
        ]
    else:
        lines.append("No entry has an available logits mode.")
    return lines


def _related(catalogue: Catalogue) -> list[str]:
    groups: dict[str, list[CatalogueEntry]] = {}
    for entry in catalogue.entries:
        if entry.weights_group is not None:
            groups.setdefault(entry.weights_group, []).append(entry)
    rows = [
        [
            group,
            entry.entry_id,
            entry.model,
            entry.route,
            _text(entry.weights_identity),
            _status(entry.modes[Mode.LOGPROBS], reason=False),
            _status(entry.modes[Mode.LOGITS], reason=False),
        ]
        for group, members in groups.items()
        if len(members) > 1
        for entry in members
    ]
    lines = ["## Related entries", ""]
    if not rows:
        return [*lines, "No two entries share weights."]
    return [
        *lines,
        "Entries in one weights group serve the same weights, so their rows compare routes for one model.",
        "",
        *_table(
            ("Weights group", "Entry", "Model", "Route", "Weights identity", "Logprobs status", "Logits status"),
            rows,
        ),
    ]


def render_catalogue(catalogue: Catalogue) -> str:
    """The catalogue page as Markdown text with LF newlines."""
    blocks = (
        ["# Model catalogue", "", *_summary(catalogue)],
        _rule(),
        _catalogue_table(catalogue),
        _without_logprobs(catalogue),
        _related(catalogue),
    )
    return "\n\n".join("\n".join(block) for block in blocks) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    """Write the catalogue page; 0 on success, 2 when the catalogue cannot be loaded."""
    parser = argparse.ArgumentParser(description="Render the catalogue page from the catalogue data file.")
    parser.add_argument(
        "--catalogue", type=Path, default=None, help="catalogue data file (default: the one shipped in the package)"
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="page to write (default: docs/catalogue.md)")
    args = parser.parse_args(argv)
    try:
        catalogue = load_catalogue(args.catalogue)
    except CatalogueError as error:
        print(f"render_catalogue: {error}", file=sys.stderr)
        return 2
    page = render_catalogue(catalogue)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(page.encode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
