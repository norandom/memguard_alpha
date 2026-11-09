"""The EDGAR outlook decision and its prompt (Requirement 7.1).

RELEASE is an illustrative press release, not EDGAR data. One test renders the
first stored record of data/edgar/held_out/oos_control.jsonl, read only.
"""

import ast
import hashlib
import json
import re
import sys
from pathlib import Path

import pytest

from recall_guard.core.decision import DecisionDefinition, DecisionParseError, adjust, parse_json_components
from recall_guard.dataset import edgar_outlook
from recall_guard.dataset.edgar_outlook import outlook_definition, prompt_sha256, render_outlook_prompt

ROOT = Path(__file__).resolve().parents[2]
HELD_OUT_CONTROL = ROOT / "data" / "edgar" / "held_out" / "oos_control.jsonl"

RELEASE = (
    "Example Corp Reports Third Quarter Results\n\n"
    "Example Corp today reported quarterly revenue of $10.0 million, up 4 percent from the prior "
    "quarter, and reaffirmed its full-year guidance."
)

# The fixed wording around the release, pinned line by line.
INSTRUCTION_LINES = [
    "Rate the business outlook that the press release below conveys as one number from -1 "
    "(clearly negative) to 1 (clearly positive).",
    "Use only the text of the release.",
    "",
    "Press release:",
]
ANSWER_LINES = [
    "",
    'Output a JSON object with the single key "outlook", e.g. {"outlook": 0.2}.',
]
PREFIX = "\n".join(INSTRUCTION_LINES) + "\n"
SUFFIX = "\n" + "\n".join(ANSWER_LINES)
EXPECTED_PROMPT = "\n".join([*INSTRUCTION_LINES, RELEASE, *ANSWER_LINES])
# hashlib.sha256(EXPECTED_PROMPT.encode("utf-8")).hexdigest(), computed once and pinned.
EXPECTED_SHA256 = "8b0f15b8ecc64fd3d51ae3c14c95098d831b5ff49451b30103b76c59508aa27a"


# -- the outlook decision definition (7.1) ------------------------------------


def test_outlook_definition_declares_one_outlook_in_unit_bounds_on_a_tenth_grid():
    assert isinstance(outlook_definition, DecisionDefinition)
    assert outlook_definition.name == "edgar_outlook"
    assert [(spec.name, spec.lower, spec.upper, spec.grid) for spec in outlook_definition.components] == [
        ("outlook", -1.0, 1.0, 0.1)
    ]
    assert outlook_definition.parse is parse_json_components


def test_outlook_definition_parses_the_json_example_into_the_outlook_and_its_sign():
    vector = outlook_definition.parse_response('{"outlook": 0.2}')

    assert vector == {"outlook": 0.2} and list(vector) == ["outlook"]
    assert outlook_definition.summary(vector) == 0.2
    assert outlook_definition.decide(vector) == 1


@pytest.mark.parametrize("text", ['{"outlook": -1}', '{"outlook": 1}', '{"outlook": 0}', '{"outlook": -0.35}'])
def test_outlook_definition_accepts_the_closed_bounds_and_off_grid_values(text):
    vector = outlook_definition.parse_response(text)

    assert vector == {"outlook": json.loads(text)["outlook"]}
    assert type(vector["outlook"]) is float


@pytest.mark.parametrize(("text", "message"), [
    ('{"outlook": 1.2}', "outside"),
    ('{"outlook": -1.01}', "outside"),
    ("{}", "missing"),
    ('{"sentiment": 0.2}', "missing"),
    ('{"outlook": 0.2, "confidence": 0.9}', "undeclared"),
    ('{"outlook": 0.2, "outlook": -0.2}', "duplicate"),
    ('{"outlook": NaN}', "non-finite"),
    ('{"outlook": Infinity}', "non-finite"),
    ('{"outlook": true}', "must be a number"),
    ('{"outlook": "0.2"}', "must be a number"),
    ("[0.2]", "must be a JSON object"),
    ("outlook: 0.2", "not valid JSON"),
    ("", "not valid JSON"),
])
def test_outlook_definition_rejects_out_of_range_missing_extra_duplicate_and_malformed_answers(text, message):
    with pytest.raises(DecisionParseError, match=message):
        outlook_definition.parse_response(text)


@pytest.mark.parametrize(("outlook", "expected"), [
    (0.2, 1),
    (1.0, 1),
    (0.1, 1),
    (1e-9, 1),               # survives the nine-decimal rounding
    (0.0, -1),               # zero is -1
    (-0.0, -1),
    (-1e-12, -1),            # a tiny negative residue
    (1e-12, -1),             # a tiny positive residue rounds to zero, as in the macro definition
    (-0.1, -1),
    (-1.0, -1),
])
def test_decide_is_one_above_zero_and_minus_one_at_or_below_zero(outlook, expected):
    assert outlook_definition.decide({"outlook": outlook}) == expected


def test_summary_is_the_outlook_itself():
    for value in (-1.0, -0.3, 0.0, 0.45, 1.0):
        assert outlook_definition.summary({"outlook": value}) == value


def test_adjusting_scales_the_outlook_and_keeps_its_sign_until_it_reaches_zero():
    raw = outlook_definition.parse_response('{"outlook": 0.3}')

    half = adjust(raw, 0.5)
    assert outlook_definition.summary(half) == pytest.approx(0.15, abs=1e-15)
    assert outlook_definition.decide(half) == 1
    assert outlook_definition.decide(adjust(raw, 0.9)) == 1
    assert outlook_definition.decide(adjust(raw, 1.0)) == -1      # fully discounted: zero, decided as -1
    assert outlook_definition.decide(adjust({"outlook": -0.3}, 0.5)) == -1


@pytest.mark.parametrize(("vector", "error", "message"), [
    ({}, ValueError, "exactly the key"),
    ({"outlook": 0.2, "confidence": 0.9}, ValueError, "exactly the key"),
    ({"sentiment": 0.2}, ValueError, "exactly the key"),
    ({"outlook": float("nan")}, ValueError, "finite"),
    ({"outlook": float("inf")}, ValueError, "finite"),
    ({"outlook": 10**400}, ValueError, "finite"),
    ({"outlook": True}, TypeError, "finite number"),
    ({"outlook": "0.2"}, TypeError, "finite number"),
    ([("outlook", 0.2)], TypeError, "mapping"),
])
def test_summary_and_decide_refuse_anything_but_one_finite_outlook(vector, error, message):
    with pytest.raises(error, match=message):
        outlook_definition.summary(vector)
    with pytest.raises(error, match=message):
        outlook_definition.decide(vector)


# -- the outlook prompt (7.1) --------------------------------------------------


def test_prompt_is_the_pinned_instruction_the_release_and_the_pinned_answer_format():
    assert render_outlook_prompt(RELEASE) == EXPECTED_PROMPT
    assert EXPECTED_PROMPT == PREFIX + RELEASE + SUFFIX


def test_rendering_is_deterministic():
    first = render_outlook_prompt(RELEASE)

    assert render_outlook_prompt(RELEASE) == first
    assert render_outlook_prompt("".join(list(RELEASE))) == first   # an equal string from another object


def test_prompt_contains_the_release_text_verbatim_exactly_once():
    prompt = render_outlook_prompt(RELEASE)

    assert prompt.count(RELEASE) == 1
    assert prompt[len(PREFIX):len(prompt) - len(SUFFIX)] == RELEASE


@pytest.mark.parametrize("release", [
    "  Title with edge spaces  \n\nBody ends with a newline.\n",
    "Title\r\n\r\nWindows line endings\tand a tab",
    "KCG\ufffds title\n\nA stored replacement character and non-ASCII text: caf\u00e9, \u00fcber.",
    "Title only",
])
def test_release_text_is_embedded_unchanged_without_trimming_or_normalisation(release):
    prompt = render_outlook_prompt(release)

    assert prompt == PREFIX + release + SUFFIX
    assert prompt_sha256(prompt) == hashlib.sha256((PREFIX + release + SUFFIX).encode("utf-8")).hexdigest()


@pytest.mark.parametrize(("release", "error"), [
    ("", ValueError),
    ("   ", ValueError),
    ("\n\n\t", ValueError),
    (None, TypeError),
    (b"Title\n\nBody", TypeError),
    (1500, TypeError),
    (["Title", "Body"], TypeError),
])
def test_rendering_refuses_an_empty_or_non_string_release(release, error):
    with pytest.raises(error, match="release_text"):
        render_outlook_prompt(release)


def test_instruction_names_no_date_trade_or_return_and_uses_no_dash_glyphs():
    wording = PREFIX + SUFFIX

    assert render_outlook_prompt(RELEASE).replace(RELEASE, "") == wording
    assert not re.search(r"\b(19|20)\d\d\b", wording)
    assert not re.search(r"\b(dates?|dated|filed|filing|trade|trading|returns?|buy|sell|position|price)\b",
                         wording, flags=re.IGNORECASE)
    assert "\u2014" not in wording and "\u2013" not in wording
    assert '{"outlook": 0.2}' in wording
    assert outlook_definition.parse_response('{"outlook": 0.2}') == {"outlook": 0.2}


# -- prompt hash ---------------------------------------------------------------


def test_prompt_sha256_of_the_fixed_release_is_pinned():
    assert hashlib.sha256(EXPECTED_PROMPT.encode("utf-8")).hexdigest() == EXPECTED_SHA256
    assert prompt_sha256(render_outlook_prompt(RELEASE)) == EXPECTED_SHA256
    assert prompt_sha256(render_outlook_prompt(RELEASE)) == EXPECTED_SHA256


def test_prompt_sha256_changes_when_one_character_of_the_release_changes():
    changed = RELEASE.replace("4 percent", "5 percent")
    assert changed != RELEASE and len(changed) == len(RELEASE)

    assert prompt_sha256(render_outlook_prompt(changed)) != EXPECTED_SHA256
    assert prompt_sha256(render_outlook_prompt(RELEASE + " ")) != EXPECTED_SHA256


def test_prompt_sha256_is_lowercase_hex_of_the_utf8_bytes():
    digest = prompt_sha256("caf\u00e9")

    assert re.fullmatch(r"[0-9a-f]{64}", digest)
    assert digest == hashlib.sha256(b"caf\xc3\xa9").hexdigest()
    assert digest != hashlib.sha256(b"caf\xe9").hexdigest()


@pytest.mark.parametrize("text", [None, b"prompt", 1, ["prompt"]])
def test_prompt_sha256_refuses_a_non_string(text):
    with pytest.raises(TypeError, match="text"):
        prompt_sha256(text)


def test_prompt_sha256_refuses_text_without_a_utf8_encoding():
    with pytest.raises(ValueError, match="surrogates"):
        prompt_sha256("lone surrogate \ud800")


# -- a stored EDGAR record -------------------------------------------------------


def test_a_stored_held_out_control_release_renders_and_hashes():
    with HELD_OUT_CONTROL.open("r", encoding="utf-8") as handle:
        record = json.loads(handle.readline())
    release = record["prompt"]

    prompt = render_outlook_prompt(release)

    assert isinstance(release, str) and 0 < len(release) <= 1500
    assert prompt == PREFIX + release + SUFFIX
    assert prompt.count(release) == 1
    assert prompt_sha256(prompt) == hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    assert prompt_sha256(render_outlook_prompt(release)) == prompt_sha256(prompt)


# -- imports -------------------------------------------------------------------------


def test_module_imports_only_core_and_the_standard_library_and_does_no_io():
    tree = ast.parse(Path(edgar_outlook.__file__).read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0
            imported.add(node.module)
    called = {node.func.id for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}

    for name in imported:
        top = name.split(".")[0]
        assert top in sys.stdlib_module_names or name.startswith("recall_guard.core."), name
    assert {name for name in imported if name.startswith("recall_guard.")} == {"recall_guard.core.decision"}
    assert not called & {"open", "print", "input", "exec", "eval", "__import__"}
    assert not {"requests", "Path", "os", "json"} & set(vars(edgar_outlook))
