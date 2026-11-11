"""Form checks for the research record (qualified-model-catalogue, task 5.1).

Three scopes:

- Scope A, every dated report under ``docs/results/``: the four-line summary block in
  the template's form and a class from the fixed list (Requirements 7.4, 7.9).
- Scope B, the restructured documents and the summary blocks: no infrastructure terms,
  no em dash, no model, provider or route name in a section head, numbered figure
  captions, no themed figure variants (Requirements 7.8, 7.11, 7.13, 8.2, 8.6). The
  findings of the other dated reports are not checked for content.
- Scope C, the report template and the reporting rules (Requirements 9.1 to 9.5).

Documents not yet in form are listed in ``SUMMARY_BLOCK_PENDING`` and
``CONTENT_PENDING``. A listed document must still fail its check, so a task that brings
a document into form has to delete its entry here; the lists cannot go stale.

The test reads ``docs/`` and the shipped package data only. The steering file lives
under ``.kiro``, which the CI build context excludes, so its checks skip when the file
is absent. Nothing here reads ``runs/``.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import NamedTuple

import pytest

import recall_guard
from recall_guard.core import profiles

_ROOT = Path(__file__).resolve().parents[2]
_DOCS = _ROOT / "docs"
_RESULTS = _DOCS / "results"
_TEMPLATE = _RESULTS / "template.md"
_STEERING = _ROOT / ".kiro" / "steering" / "research-reports.md"

# --- fixed vocabulary ---------------------------------------------------------

CLASSES = (
    "qualification evidence",
    "mode comparison",
    "foundation",
    "maintenance",
    "convenience",
    "not aligned",
)
SUMMARY_LABELS = ("Result", "Catalogue effect", "Not shown", "Class")
REPORT_SECTIONS = ("Question", "Methods", "Data", "Results", "Limits", "References")

EM_DASH = "—"

# Requirement 6.11, in the same words in the template and in the reporting rules.
DETECTABLE_EFFECT = "the detectable effect for each main comparison at its sample size"

# Infrastructure vocabulary that has no place in a quant result (Requirement 7.11).
# Matched as words or phrases. "memory" and "window" alone are not listed: memorization
# is the subject of the research and rolling windows are ordinary statistics.
_I = re.IGNORECASE
BANNED_TERMS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("Windows", re.compile(r"\bWindows\b")),  # capitalised only
    ("Defender", re.compile(r"\bDefender\b", _I)),
    ("antivirus", re.compile(r"\banti-?virus\b", _I)),
    ("exit code", re.compile(r"\bexit(?:ed with)? (?:codes?|status)\b", _I)),
    ("WinError", re.compile(r"WinError", _I)),
    ("PowerShell", re.compile(r"\bPowerShell\b", _I)),
    ("WSL", re.compile(r"\bWSL2?\b", _I)),
    ("Podman", re.compile(r"\bPodman\b", _I)),
    ("Docker", re.compile(r"\bDocker\b", _I)),
    ("RAM", re.compile(r"\bRAM\b")),  # upper case only
    ("memory pressure", re.compile(r"\bmemory pressure\b", _I)),
    ("out of memory", re.compile(r"\bout[- ]of[- ]memory\b", _I)),
    ("OOM", re.compile(r"\bOOM\b", _I)),
    ("process ended", re.compile(r"\bprocess ended\b", _I)),
    ("process exit", re.compile(r"\bprocess exit", _I)),
    ("reaper", re.compile(r"\breap(?:er|ed|ing)\b", _I)),
    ("port number", re.compile(r"\bports? \d{2,5}\b", _I)),
    ("local address", re.compile(r"\b(?:localhost|127\.0\.0\.1):\d+", _I)),
    ("driver script", re.compile(r"\bdriver scripts?\b", _I)),
    ("hashsalt", re.compile(r"hashsalt", _I)),
)

# A themed figure variant (Requirement 8.6): an image file whose name ends in
# "-light" or "-dark", or any target with the MkDocs theme suffix.
_THEME_SUFFIX = r"#only-(?:light|dark)"
_THEMED_VARIANT = re.compile(
    rf"[\w./-]*(?:[-_](?:light|dark)\.(?:svg|png|jpe?g|gif|webp|avif|pdf)(?![A-Za-z0-9])(?:{_THEME_SUFFIX})?"
    rf"|{_THEME_SUFFIX})"
)

# Generic words that the catalogue and profile names contain and that are not names.
_STOP_TOKENS = frozenset({"ai", "api", "it", "local", "managed", "remote", "the", "top"})

# --- documents not yet in form ------------------------------------------------

# Dated reports without the summary block. Tasks 5.4 and 5.5 delete entries.
SUMMARY_BLOCK_PENDING = frozenset(
    {
        "2026-10-30-nim-measurement.md",
        "2026-10-31-completion-api-pilot.md",
        "2026-11-01-anthropic-reweighting-prototype.md",
        "2026-11-01-fomc-statistical-signature.md",
        "2026-11-01-nim-refresh.md",
        "2026-11-01-recall-guard-ensemble.md",
        "2026-11-01-recall-guard-single.md",
        "2026-11-05-local-provider-feasibility.md",
        "2026-11-05-nim-request-diagnostic.md",
        "2026-11-05-sp500-pilot.md",
        "2026-11-06-gpt-oss-response-mapping.md",
        "2026-11-06-local-model-smoke.md",
        "2026-11-06-local-model-storage.md",
        "2026-11-06-modal-pilot.md",
        "2026-11-08-gemma3-document-likelihood.md",
        "2026-11-08-gemma4-document-likelihood-npu.md",
        "2026-11-08-phase-a-single-draw.md",
        "2026-11-08-phi4-mini-disqualified.md",
        "2026-11-11-memorization-adjusted-finance-sim.md",
    }
)

# Scope B documents, as paths relative to the repository root.
SYNTHESIS = "docs/research-report.md"
R43_REPORT = "docs/results/2026-11-11-memorization-adjusted-finance-sim.md"
METHODS_PAGE = "docs/research-methods.md"
CATALOGUE_PAGE = "docs/catalogue.md"
REGISTER = "docs/research-register.md"
CONTENT_DOCUMENTS = (SYNTHESIS, R43_REPORT, METHODS_PAGE, CATALOGUE_PAGE, REGISTER)

# Scope B documents that do not exist yet or still fail a content check. The tasks
# that write or rewrite them (4.3, 5.2 to 5.6) delete entries.
CONTENT_PENDING = frozenset({SYNTHESIS, R43_REPORT, METHODS_PAGE, CATALOGUE_PAGE, REGISTER})

# The governing files name what does not belong in the record, so that one list item
# is taken out before the banned-terms check. It starts with this marker.
EXCLUSION_MARKER = "Not in the record:"


# --- helpers (used by tasks 5.3 to 5.6) ---------------------------------------


class Image(NamedTuple):
    """An embedded image and the first non-blank line below it."""

    target: str
    caption: str | None


def prose_lines(text: str) -> list[str]:
    """The lines of ``text`` outside fenced code blocks."""
    lines: list[str] = []
    fenced = False
    for line in text.splitlines():
        if line.lstrip().startswith(("```", "~~~")):
            fenced = not fenced
            continue
        if not fenced:
            lines.append(line)
    return lines


def section_heads(text: str) -> list[tuple[int, str]]:
    """Every Markdown head outside code fences as ``(level, title)``."""
    heads: list[tuple[int, str]] = []
    for line in prose_lines(text):
        match = re.match(r"(#{1,6})\s+(.*?)\s*#*\s*$", line)
        if match:
            heads.append((len(match.group(1)), match.group(2)))
    return heads


def _cells(row: str) -> list[str]:
    inner = row.strip()
    if not (inner.startswith("|") and inner.endswith("|") and len(inner) >= 2):
        return []
    return [cell.strip().replace(r"\|", "|") for cell in re.split(r"(?<!\\)\|", inner[1:-1])]


def parse_summary_block(text: str) -> dict[str, str] | None:
    """The summary block of a report as ``{label: content}`` in row order.

    The form is the template's: a table whose header row is ``| Summary | |``, a
    separator row, then one row per line. Returns ``None`` when the report has no
    such table. Labels are returned as written; ``summary_block_problems`` judges them.
    """
    lines = prose_lines(text)
    for index, line in enumerate(lines):
        if _cells(line) != ["Summary", ""]:
            continue
        if index + 1 >= len(lines) or not re.fullmatch(r"\|\s*:?-+:?\s*\|\s*:?-+:?\s*\|", lines[index + 1].strip()):
            return None
        block: dict[str, str] = {}
        for row in lines[index + 2 :]:
            cells = _cells(row)
            if len(cells) != 2:
                break
            block[cells[0]] = cells[1]
        return block
    return None


def summary_block_problems(text: str) -> list[str]:
    """What keeps ``text`` from having a valid summary block; empty when it is valid."""
    block = parse_summary_block(text)
    if block is None:
        return ["no summary block"]
    problems: list[str] = []
    if tuple(block) != SUMMARY_LABELS:
        problems.append(f"summary lines are {list(block)}, expected {list(SUMMARY_LABELS)}")
    problems.extend(f"summary line {label!r} is empty" for label, content in block.items() if not content)
    if "Class" in block and block["Class"] not in CLASSES:
        problems.append(f"class {block['Class']!r} is not in the fixed list")
    return problems


def embedded_images(text: str) -> list[Image]:
    """Every image embedded outside code fences, with the line that follows it."""
    lines = prose_lines(text)
    images: list[Image] = []
    for index, line in enumerate(lines):
        targets = re.findall(r"!\[[^\]]*\]\(\s*([^)\s]+)[^)]*\)", line)
        targets += re.findall(r"<img\b[^>]*\bsrc=[\"']([^\"']+)[\"']", line, flags=_I)
        if not targets:
            continue
        caption = next((below.strip() for below in lines[index + 1 :] if below.strip()), None)
        images.extend(Image(target, caption) for target in targets)
    return images


def figure_problems(text: str) -> list[str]:
    """Images without a numbered caption below, and themed figure variants."""
    problems: list[str] = []
    for image in embedded_images(text):
        caption = re.sub(r"^(?:<figcaption>|[*_]+)\s*", "", image.caption or "")
        if not re.match(r"Figure \d+\.", caption):
            problems.append(f"image {image.target} has no caption starting with 'Figure N.' below it")
    for variant in sorted({match.group() for match in _THEMED_VARIANT.finditer(text)}):
        problems.append(f"themed figure variant {variant}")
    return problems


def banned_terms_in(text: str) -> list[str]:
    """The banned infrastructure terms that occur in ``text``."""
    return [label for label, pattern in BANNED_TERMS if pattern.search(text)]


def _model_family(model_id: str) -> str:
    """``google/gemma-3-4b-it`` gives ``gemma``; ``gpt-oss:20b`` gives ``gpt-oss``."""
    name = model_id.split("/")[-1].split(":")[0]
    parts: list[str] = []
    for segment in re.split(r"[-_]", name):
        stem = re.match(r"[A-Za-z]*", segment).group()
        if stem:
            parts.append(stem)
        if stem != segment:
            break
    return "-".join(parts)


def _lower_names(names: set[str]) -> frozenset[str]:
    return frozenset(name.lower() for name in names if name and name.lower() not in _STOP_TOKENS)


def _route_and_family_names() -> tuple[frozenset[str], frozenset[str]]:
    """Provider and route names, and model family names, of the shipped data."""
    model_ids: set[str] = set()
    routes: set[str] = set()
    for entry in recall_guard.load_catalogue().entries:
        model_ids.add(entry.model)
        routes.update(entry.route.split(",")[0].split())
    for profile_id in profiles.available_profiles():
        profile = profiles.load_profile(profile_id)
        model_ids.add(profile.model_config.model)
        if profile.model_config.provider:
            routes.add(profile.model_config.provider)
        display = profile.display_name.split("(")[0]
        routes.update(word for word in display.split() if not re.search(r"[/:]", word))
    # The organisation prefix of a model id ("meta/", "google/") is not taken: those
    # are ordinary words. An organisation that is also a provider or a route is in
    # ``routes`` already.
    families: set[str] = set()
    for model_id in model_ids:
        family = _model_family(model_id)
        families.add(family)
        families.update(part for part in family.split("-") if len(part) >= 4)
    return _lower_names(routes), _lower_names(families)


def name_tokens() -> frozenset[str]:
    """Model, provider and route names of the shipped catalogue and profiles, lower case."""
    routes, families = _route_and_family_names()
    return routes | families


def versioned_tokens() -> frozenset[str]:
    """Family names under four letters ("phi"): a name only when a version follows."""
    routes, families = _route_and_family_names()
    return frozenset(name for name in families - routes if len(re.sub(r"[^a-z]", "", name)) < 4)


def heads_with_names(
    text: str, names: frozenset[str] | None = None, versioned: frozenset[str] | None = None
) -> list[str]:
    """Section heads that contain a model, provider or route name.

    A name counts when it stands alone or carries a version ("Gemma-3", "Phi4"), not
    as the first part of a hyphenated word ("Llama-based"). A name in ``versioned``
    counts only with a version, so "Phi coefficient" is an ordinary head.
    """
    names = name_tokens() if names is None else names
    versioned = versioned_tokens() if versioned is None else versioned
    if not names:
        return []
    alternatives = "|".join(
        re.escape(name) + (r"-?\d" if name in versioned else r"(?![A-Za-z]|-[A-Za-z])")
        for name in sorted(names, key=len, reverse=True)
    )
    pattern = re.compile(rf"(?<![A-Za-z0-9])(?:{alternatives})", _I)
    return [title for _, title in section_heads(text) if pattern.search(title)]


def content_problems(text: str) -> list[str]:
    """Every Scope B finding in ``text``; empty when the text is in form."""
    problems = [f"banned term: {term}" for term in banned_terms_in(text)]
    if EM_DASH in text:
        problems.append(f"{text.count(EM_DASH)} em dash(es)")
    problems.extend(f"name in section head: {head}" for head in heads_with_names(text))
    problems.extend(figure_problems(text))
    return problems


def register_front(text: str) -> str:
    """The register above its entries list: purpose, state of evidence and index."""
    return re.split(r"^## Entries\s*$", text, maxsplit=1, flags=re.MULTILINE)[0]


def without_exclusion_item(text: str) -> str:
    """``text`` without the list item that names what does not belong in the record."""
    kept: list[str] = []
    skipping = False
    for line in text.splitlines():
        if line.startswith("- "):
            skipping = line.startswith(f"- {EXCLUSION_MARKER}")
        elif not line.strip():
            skipping = False
        if not skipping:
            kept.append(line)
    return "\n".join(kept)


def class_list_in(text: str) -> tuple[str, ...] | None:
    """The class list a governing file states after ``Classes:``, up to the full stop."""
    match = re.search(r"Classes:\s*([^.]+)\.", text)
    if match is None:
        return None
    return tuple(" ".join(item.split()) for item in match.group(1).split(","))


def _read(relative: str) -> str:
    return (_ROOT / relative).read_text(encoding="utf-8")


def _dated_reports() -> list[Path]:
    return sorted(path for path in _RESULTS.glob("*.md") if path.name not in {"template.md", "index.md"})


def _scope_b_text(relative: str) -> str:
    text = _read(relative)
    return register_front(text) if relative == REGISTER else text


# --- helper unit tests --------------------------------------------------------

_GOOD_REPORT = """# Recall estimate inside and after the cutoff, 2026-11-11

> Register: R44 · Status: measured

| Summary | |
|---|---|
| Result | The difference is 0.096 (95% interval 0.01 to 0.18, n = 17 months). |
| Catalogue effect | none |
| Not shown | Whether the difference holds for p \\| q on another corpus. |
| Class | mode comparison |

## Question

One question.
"""


def test_parse_summary_block_reads_the_four_lines_in_order() -> None:
    block = parse_summary_block(_GOOD_REPORT)
    assert block is not None
    assert tuple(block) == SUMMARY_LABELS
    assert block["Catalogue effect"] == "none"
    assert block["Not shown"].endswith("p | q on another corpus.")
    assert summary_block_problems(_GOOD_REPORT) == []


def test_parse_summary_block_rejects_other_forms() -> None:
    prose = "# Title\n\nResult: 0.5. Catalogue effect: none. Not shown: x. Class: foundation.\n"
    assert parse_summary_block(prose) is None
    assert summary_block_problems(prose) == ["no summary block"]
    other_table = "| Result | 0.5 |\n|---|---|\n| Class | foundation |\n"
    assert parse_summary_block(other_table) is None
    fenced = "```markdown\n" + _GOOD_REPORT + "```\n"
    assert parse_summary_block(fenced) is None
    no_separator = "| Summary | |\n| Result | 0.5 |\n"
    assert parse_summary_block(no_separator) is None


def test_summary_block_problems_names_order_class_and_empty_lines() -> None:
    swapped = _GOOD_REPORT.replace("| Catalogue effect | none |\n", "").replace(
        "| Class | mode comparison |", "| Class | mode comparison |\n| Catalogue effect | none |"
    )
    assert any("expected" in problem for problem in summary_block_problems(swapped))
    wrong_class = _GOOD_REPORT.replace("| Class | mode comparison |", "| Class | phase B |")
    assert summary_block_problems(wrong_class) == ["class 'phase B' is not in the fixed list"]
    empty = _GOOD_REPORT.replace("| Catalogue effect | none |", "| Catalogue effect | |")
    assert summary_block_problems(empty) == ["summary line 'Catalogue effect' is empty"]


def test_section_heads_lists_heads_outside_code_fences() -> None:
    text = "# Title\n\n## Question\n\n```markdown\n## Not a head\n```\n\n### Sub head ##\n\n#hashtag\n"
    assert section_heads(text) == [(1, "Title"), (2, "Question"), (3, "Sub head")]
    assert section_heads("plain text\n\n    indented\n") == []


def test_embedded_images_pairs_each_image_with_the_line_below() -> None:
    text = (
        "Text.\n\n![Gap](../assets/gap.svg)\n\nFigure 1. The gap, n = 17 months.\n\n"
        '<img src="assets/b.png" alt="b">\nNo caption here.\n\n```\n![x](in-code.svg)\n```\n'
    )
    assert embedded_images(text) == [
        Image("../assets/gap.svg", "Figure 1. The gap, n = 17 months."),
        Image("assets/b.png", "No caption here."),
    ]
    assert embedded_images("No image, only a [link](page.md).\n") == []
    assert embedded_images("![last](a.svg)\n") == [Image("a.svg", None)]


def test_figure_problems_wants_a_numbered_caption_and_one_variant() -> None:
    good = "![Gap](assets/gap.svg)\n\nFigure 2. The gap with 95% intervals, n = 17.\n"
    assert figure_problems(good) == []
    assert figure_problems(good.replace("Figure 2. ", "*Figure 2.* ")) == []
    assert figure_problems("![Gap](assets/gap.svg)\n\nThe gap.\n") != []
    assert figure_problems("![Figure F1. Gap](assets/gap.svg)\n\nFigure F1. Gap.\n") != []
    themed = "![Gap](assets/gap-light.svg#only-light)\n\nFigure 1. Gap.\n"
    assert figure_problems(themed) == ["themed figure variant assets/gap-light.svg#only-light"]


@pytest.mark.parametrize(
    "target",
    [
        "assets/gap-light.svg",
        "assets/gap-dark.svg",
        "assets/gap-light.png",
        "assets/gap-dark.png",
        "assets/gap_dark.jpeg",
        "assets/gap-light.webp",
        "assets/gap.svg#only-light",
        "assets/gap.png#only-dark",
        "assets/gap-dark.png#only-dark",
    ],
)
def test_figure_problems_flags_every_themed_variant(target: str) -> None:
    assert figure_problems(f"![Gap]({target})\n\nFigure 1. Gap.\n") == [f"themed figure variant {target}"]


@pytest.mark.parametrize(
    "target", ["assets/highlight.svg", "assets/darkness.png", "assets/gap.svg", "assets/light-curve.png"]
)
def test_figure_problems_spares_ordinary_file_names(target: str) -> None:
    assert figure_problems(f"![Gap]({target})\n\nFigure 1. Gap.\n") == []


def test_banned_terms_catch_infrastructure_and_spare_research_vocabulary() -> None:
    assert banned_terms_in("The run stopped with exit code 1 on Windows after an OOM.") == [
        "Windows",
        "exit code",
        "OOM",
    ]
    assert banned_terms_in("The server on port 11434 ran in WSL2 with 32 GB RAM.") == ["WSL", "RAM", "port number"]
    spared = (
        "Memorization is measured over rolling windows of 12 months. The memorized share "
        "is reported per window, the program is unchanged and 11 months support each cell."
    )
    assert banned_terms_in(spared) == []


@pytest.mark.parametrize(
    ("text", "label"),
    [
        ("The server listened on port 11570.", "port number"),
        ("The server listened on port 8080.", "port number"),
        ("Ports 8000 and 8001 were taken.", "port number"),
        ("The endpoint was localhost:11434.", "local address"),
        ("Requests went to 127.0.0.1:8000.", "local address"),
        ("The server exited with code 3.", "exit code"),
        ("Exit status 137 ended the run.", "exit code"),
        ("Exit codes were not recorded.", "exit code"),
    ],
)
def test_banned_terms_catch_port_and_exit_wording(text: str, label: str) -> None:
    assert banned_terms_in(text) == [label]


@pytest.mark.parametrize(
    "text",
    [
        "Transport of the estimate to another corpus is not shown.",
        "The cells support 11 months each.",
        "The synthesis follows the report 2026 structure.",
        "The exit from the position is priced at the close.",
        "The memorization-adjusted estimate uses the memorized share.",
        "A rolling window, in windows of 12 months.",
        "The program, the parameter and the serving route are unchanged.",
    ],
)
def test_banned_terms_spare_quant_and_research_wording(text: str) -> None:
    assert banned_terms_in(text) == []


def test_model_family_strips_versions_and_organisations() -> None:
    assert _model_family("google/gemma-3-4b-it") == "gemma"
    assert _model_family("gpt-oss:20b") == "gpt-oss"
    assert _model_family("llava-llama3:ctx2048") == "llava-llama"
    assert _model_family("phi4-mini:3.8b") == "phi"


def test_name_tokens_come_from_the_catalogue_and_the_profiles() -> None:
    names = name_tokens()
    assert {"gemma", "llama", "gpt-oss", "ollama", "llama.cpp", "vllm", "modal", "fastflowlm", "nim"} <= names
    assert {"ryzen", "oga", "nvidia", "llava"} <= names
    assert not names & _STOP_TOKENS
    assert not any(re.search(r"\d", name) for name in names)
    # An organisation prefix of a model id is a name only when it is also a provider
    # or a route name in the data.
    assert not names & {"meta", "google", "openai", "amd"}
    assert "phi" in names and "phi" in versioned_tokens()
    assert not versioned_tokens() & {"nim", "oga"}, "short route names match without a version"


@pytest.mark.parametrize(
    "head",
    ["Meta-analysis", "Phi coefficient", "Methods", "Alpha level", "Graphical summary", "Mode comparison"],
)
def test_heads_with_names_spares_statistics_heads(head: str) -> None:
    assert heads_with_names(f"## {head}\n") == []


@pytest.mark.parametrize(
    "head",
    ["Phi-4 results", "Phi4 mini", "Gemma-3 on Ollama", "Results on llama.cpp", "Modal run", "NIM results"],
)
def test_heads_with_names_flags_model_provider_and_route_names(head: str) -> None:
    assert heads_with_names(f"## {head}\n") == [head]


def test_heads_with_names_flags_names_and_spares_ordinary_heads() -> None:
    text = (
        "# Recall estimate by evidence mode\n\n## Gemma3 on Ollama\n\n## Results\n\n"
        "### llama.cpp full distribution\n\n## Multimodal and nimble heads\n\nGemma in prose.\n"
    )
    assert heads_with_names(text) == ["Gemma3 on Ollama", "llama.cpp full distribution"]
    assert heads_with_names("## Results\n\n## Limits\n") == []


def test_content_problems_reports_each_kind_and_passes_clean_text() -> None:
    dirty = f"## Modal pilot\n\nA dash {EM_DASH} and Docker.\n\n![x](a-dark.svg)\n"
    problems = content_problems(dirty)
    assert "banned term: Docker" in problems
    assert "1 em dash(es)" in problems
    assert "name in section head: Modal pilot" in problems
    assert "themed figure variant a-dark.svg" in problems
    assert content_problems(_GOOD_REPORT) == []


def test_register_front_stops_at_the_entries_list() -> None:
    text = "# Register\n\n## Purpose\n\nText.\n\n## Index\n\n## Entries\n\n### R01 on Docker\n"
    assert register_front(text).rstrip().endswith("## Index")
    assert register_front("# Register\n\nNo entries head.\n") == "# Register\n\nNo entries head.\n"


def test_without_exclusion_item_removes_only_the_marked_item() -> None:
    text = f"- Lead with the result.\n- {EXCLUSION_MARKER} antivirus,\n  exit codes.\n- Plain prose.\n\nAntivirus.\n"
    kept = without_exclusion_item(text)
    assert "exit codes" not in kept
    assert "Lead with the result." in kept and "Plain prose." in kept
    assert banned_terms_in(kept) == ["antivirus"]
    assert without_exclusion_item("- one\n- two\n") == "- one\n- two"


def test_class_list_in_reads_the_list_after_the_label() -> None:
    assert class_list_in("Classes: qualification evidence, mode\n  comparison, foundation. More.") == (
        "qualification evidence",
        "mode comparison",
        "foundation",
    )
    assert class_list_in("One class per entry.") is None


# --- Scope A: every dated report ----------------------------------------------


def test_pending_lists_name_existing_reports_and_known_documents() -> None:
    assert SUMMARY_BLOCK_PENDING <= {path.name for path in _dated_reports()}
    assert CONTENT_PENDING <= set(CONTENT_DOCUMENTS)


@pytest.mark.parametrize("report", _dated_reports(), ids=lambda path: path.name)
def test_dated_report_has_the_summary_block_and_a_class(report: Path) -> None:
    text = report.read_text(encoding="utf-8")
    if report.name in SUMMARY_BLOCK_PENDING:
        assert parse_summary_block(text) is None, (
            f"{report.name} has a summary block now: remove it from SUMMARY_BLOCK_PENDING"
        )
        return
    assert summary_block_problems(text) == []


# --- Scope B: content of the restructured documents ---------------------------


@pytest.mark.parametrize("relative", CONTENT_DOCUMENTS)
def test_restructured_document_is_in_form(relative: str) -> None:
    exists = (_ROOT / relative).exists()
    if relative in CONTENT_PENDING:
        assert not exists or content_problems(_scope_b_text(relative)), (
            f"{relative} passes every content check now: remove it from CONTENT_PENDING"
        )
        return
    assert exists, f"{relative} is missing"
    assert content_problems(_scope_b_text(relative)) == []


def test_summary_blocks_carry_no_infrastructure_terms_or_em_dashes() -> None:
    for report in _dated_reports():
        block = parse_summary_block(report.read_text(encoding="utf-8"))
        if block is None:
            continue
        rows = "\n".join(f"{label}: {content}" for label, content in block.items())
        assert content_problems(rows) == [], report.name


# --- Scope C: the template and the reporting rules ----------------------------


def _template() -> str:
    return _TEMPLATE.read_text(encoding="utf-8")


def _flat(text: str) -> str:
    """`text` with every run of white space as one space, so phrases survive rewrapping."""
    return " ".join(text.split())


def _template_skeleton() -> str:
    match = re.search(r"```markdown\n(.*?)\n```", _template(), flags=re.DOTALL)
    assert match, "the template shows the report skeleton in a markdown code fence"
    return match.group(1)


def _steering() -> str:
    if not _STEERING.exists():
        pytest.skip(".kiro is not part of this build context")
    return _STEERING.read_text(encoding="utf-8")


def test_template_skeleton_has_the_summary_block_and_the_six_sections() -> None:
    skeleton = _template_skeleton()
    block = parse_summary_block(skeleton)
    assert block is not None and tuple(block) == SUMMARY_LABELS
    heads = section_heads(skeleton)
    assert heads[0][0] == 1
    assert tuple(title for level, title in heads if level == 2) == REPORT_SECTIONS
    assert skeleton.index("> Register:") < skeleton.index("| Summary | |") < skeleton.index("## Question")


def test_template_register_line_has_no_track_and_no_names() -> None:
    """Requirement 7.8: runs are not sorted by route or provider."""
    skeleton = _template_skeleton()
    assert not re.search(r"\btrack\b", skeleton, _I)
    front = skeleton.split("| Summary | |", 1)[0]
    register_line = " ".join(line.lstrip("> ") for line in front.splitlines() if line.startswith(">"))
    assert register_line.startswith("Register:") and "Status:" in register_line
    assert "Class" not in register_line, "the class is stated once, in the summary block"
    assert heads_with_names(f"# {register_line}\n", versioned=frozenset()) == []


def test_template_methods_guidance_covers_the_statistical_statements() -> None:
    """Requirements 6.1 to 6.6 and 6.11, as guidance in the skeleton."""
    skeleton = _template_skeleton()
    methods = _flat(skeleton.split("## Methods", 1)[1].split("## Data", 1)[0])
    for phrase in (
        "the estimand in words",  # 6.1
        "the calculation, with its inputs and their counts",  # 6.2
        "the unit of analysis, the sample size and any dependence between units",  # 6.3
        "the interval, with its method and level",  # 6.4
        "the null hypothesis, the test, the statistic, the p-value and the adjustment for its family",  # 6.5
        "an effect size with its interval",  # 6.6
    ):
        assert phrase in methods, phrase
    limits = _flat(skeleton.split("## Limits", 1)[1].split("## References", 1)[0])
    assert DETECTABLE_EFFECT in limits  # 6.11
    assert "small sample" not in limits
    references = _flat(skeleton.split("## References", 1)[1])
    assert "the operational notes when an incident removed data" in references  # 7.12


def test_template_requires_class_catalogue_entry_and_methods() -> None:
    text = _template()
    assert class_list_in(text) == CLASSES
    assert "docs/research-methods.md" in text and "methods page" in text
    skeleton = _template_skeleton()
    methods = skeleton.split("## Methods", 1)[1].split("## Data", 1)[0]
    assert "research-methods.md" in methods
    assert "in the order they appear in the results" in _flat(methods)
    catalogue_line = parse_summary_block(skeleton)["Catalogue effect"]
    assert "catalogue entry" in catalogue_line and "none" in catalogue_line


def test_template_links_specs_in_references_only() -> None:
    skeleton = _template_skeleton()
    body, references = skeleton.split("## References", 1)
    assert re.search(r"\bspec\b", references)
    assert not re.search(r"\bspecs?\b", body)
    text = _flat(_template())
    assert "Link the spec in References and nowhere else" in text
    assert "does not narrate how a spec was executed" in text


@pytest.mark.parametrize(
    "phrase",
    [
        "Lead with the result",
        "exactly as measured",
        "Each fact once per report",
        "never section heads or phase names",
        "an effect size with its interval",
        "never a p-value alone",
        "equal by construction",
        "names the missing input and shows no value",
        "numbered caption below",
        "states the sample size and what the interval is",  # 8.5
        "cannot differ by construction",  # 8.7
        "no light and dark variants",
        "Later evidence (YYYY-MM-DD)",
        "no em dashes, no marketing language",
        DETECTABLE_EFFECT,  # 6.11
        EXCLUSION_MARKER,
    ],
)
def test_template_states_the_rule(phrase: str) -> None:
    assert phrase in _flat(_template())


def test_template_is_itself_in_form() -> None:
    text = _template()
    assert EM_DASH not in text
    # The template names what does not belong in the record; that one list item is
    # exempt, the rest of the page is checked.
    assert banned_terms_in(without_exclusion_item(text)) == []
    assert banned_terms_in(text) != [], "the exclusion item names what is kept out"


@pytest.mark.parametrize(
    "phrase",
    [
        "docs/catalogue.md",
        "recall_guard/core/catalogue_data.json",
        "or states that it informs none",  # 9.3
        "a catalogue entry, not a research phase",  # 9.4
        "docs/research-methods.md",  # 9.5
        "the estimand in words",  # 6.1
        "the calculation with its inputs and their counts",  # 6.2
        "any dependence between units",  # 6.3
        "an interval with its method and level",  # 6.4
        "the adjustment for its family",  # 6.5
        "an effect size with its interval",  # 6.6
        DETECTABLE_EFFECT,  # 6.11
        "states the sample size and what the interval is",  # 8.5
        "cannot differ by construction",  # 8.7
        "References links the operational notes",  # 7.12
        "not in a track",  # 7.8
        "quant results",
        EXCLUSION_MARKER,
        ".kiro/handover.md",
        "Commit and push before and after",
        "docs/results/template.md",
        "Later evidence (YYYY-MM-DD)",
        "humanizer",
        "Chance-level forecast accuracy",
    ],
)
def test_steering_states_the_rule(phrase: str) -> None:
    assert phrase in _flat(_steering())


def test_steering_is_itself_in_form() -> None:
    text = _steering()
    assert EM_DASH not in text
    # Same exemption as for the template: the item that names what is kept out.
    assert banned_terms_in(without_exclusion_item(text)) == []


def test_class_list_is_the_same_in_the_test_and_the_template() -> None:
    assert class_list_in(_template()) == CLASSES


def test_class_list_is_the_same_in_the_test_and_the_steering() -> None:
    assert class_list_in(_steering()) == CLASSES
