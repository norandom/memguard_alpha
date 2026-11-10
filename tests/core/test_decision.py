"""Decision definitions: construction validation, strict parsing, adjustment and summary."""

from __future__ import annotations

import math

import pytest

from recall_guard.core.decision import (
    ComponentSpec,
    DecisionDefinition,
    DecisionDefinitionError,
    DecisionParseError,
    adjust,
    parse_json_components,
)


def _spread(vector):
    return vector["gold"] - vector["equity"]


def _definition(**overrides) -> DecisionDefinition:
    fields = {
        "name": "toy",
        "components": (
            ComponentSpec("equity", -1.0, 1.0, grid=0.1),
            ComponentSpec("gold", -1.0, 1.0, grid=0.1),
        ),
        "parse": parse_json_components,
        "summary": _spread,
        "decide": lambda vector: _spread(vector) > 0,
    }
    fields.update(overrides)
    return DecisionDefinition(**fields)


# -- construction ------------------------------------------------------------


@pytest.mark.parametrize(
    ("lower", "upper", "fragment"),
    [
        (1.0, 1.0, "lower must be < upper"),
        (1.0, -1.0, "lower must be < upper"),
        (float("nan"), 1.0, "lower bound must be a finite number"),
        (-1.0, float("inf"), "upper bound must be a finite number"),
        (True, 2.0, "lower bound must be a finite number"),
    ],
)
def test_bad_bounds_are_rejected_and_named(lower, upper, fragment):
    with pytest.raises(DecisionDefinitionError, match=fragment) as caught:
        ComponentSpec("growth", lower, upper)
    assert "'growth'" in str(caught.value)


@pytest.mark.parametrize("grid", [0.0, -0.1, float("nan"), float("inf")])
def test_bad_grid_is_rejected(grid):
    with pytest.raises(DecisionDefinitionError, match="grid must be a positive finite number"):
        ComponentSpec("growth", -1.0, 1.0, grid=grid)


def test_duplicate_component_names_are_rejected_and_named():
    with pytest.raises(DecisionDefinitionError, match="duplicate component name 'gold'"):
        _definition(components=(ComponentSpec("gold", -1, 1), ComponentSpec("gold", -1, 1)))


@pytest.mark.parametrize(
    ("overrides", "fragment"),
    [
        ({"components": ()}, "non-empty tuple"),
        ({"components": [ComponentSpec("gold", -1, 1)]}, "non-empty tuple"),
        ({"components": ("gold",)}, "must be a ComponentSpec"),
        ({"name": ""}, "definition name must be a non-empty string"),
        ({"parse": None}, "parse must be callable"),
        ({"summary": 3.0}, "summary must be callable or None"),
        ({"decide": "sign"}, "decide must be callable or None"),
    ],
)
def test_invalid_definitions_are_rejected_at_construction(overrides, fragment):
    with pytest.raises(DecisionDefinitionError, match=fragment):
        _definition(**overrides)


def test_invalid_definition_is_rejected_before_parse_is_ever_called():
    calls: list[str] = []

    def parse(text):
        calls.append(text)
        return {}

    with pytest.raises(DecisionDefinitionError):
        _definition(components=(ComponentSpec("gold", 1, 0),), parse=parse)
    assert calls == []


def test_parse_is_stored_unchanged():
    assert _definition().parse is parse_json_components


# -- strict parsing ----------------------------------------------------------


def test_parse_response_returns_declared_components_in_order():
    vector = _definition().parse_response('{"gold": 0.3, "equity": -0.2}')
    assert list(vector) == ["equity", "gold"]
    assert vector == {"equity": -0.2, "gold": 0.3}
    assert all(type(value) is float for value in vector.values())


def test_bounds_are_inclusive():
    assert _definition().parse_response('{"equity": -1, "gold": 1}') == {"equity": -1.0, "gold": 1.0}


def test_duplicate_key_in_response_is_rejected():
    with pytest.raises(DecisionParseError, match=r"duplicate keys \['gold'\]"):
        _definition().parse_response('{"equity": 0.1, "gold": 0.2, "gold": -0.9}')


def test_missing_component_is_rejected_and_named():
    with pytest.raises(DecisionParseError, match=r"missing components \['gold'\]"):
        _definition().parse_response('{"equity": 0.1}')


def test_extra_key_is_rejected_and_named():
    with pytest.raises(DecisionParseError, match=r"undeclared components \['cash'\]"):
        _definition().parse_response('{"equity": 0.1, "gold": 0.2, "cash": 0.0}')


def test_out_of_bounds_value_is_rejected_not_clipped():
    with pytest.raises(DecisionParseError, match=r"'gold'=1.2 outside \[-1.0, 1.0\]"):
        _definition().parse_response('{"equity": 0.1, "gold": 1.2}')


@pytest.mark.parametrize(
    "value",
    [True, None, "0.2", float("nan"), float("inf"), [0.2]],
)
def test_non_finite_or_non_numeric_values_from_the_extractor_are_rejected(value):
    definition = _definition(parse=lambda text: {"equity": 0.1, "gold": value})
    with pytest.raises(DecisionParseError, match="'gold' must be a finite number"):
        definition.parse_response("ignored")


def test_extractor_value_error_becomes_parse_error():
    def parse(text):
        raise ValueError("no object found")

    with pytest.raises(DecisionParseError, match="parse failed: no object found"):
        _definition(parse=parse).parse_response("prose")


def test_extractor_must_return_a_mapping():
    with pytest.raises(DecisionParseError, match="parse must return a mapping; got list"):
        _definition(parse=lambda text: [("equity", 0.1)]).parse_response("x")


@pytest.mark.parametrize(
    ("text", "fragment"),
    [
        ("[0.1, 0.2]", "must be a JSON object; got list"),
        ("0.5", "must be a JSON object; got float"),
        ("```json\n{}", "one JSON object inside one closed code fence"),
        ('{"gold": NaN}', "non-finite JSON constant NaN"),
        ('{"gold": 1e999}', "'gold' must be finite"),
        ('{"gold": true}', "'gold' must be a number"),
        ('{"gold": {"value": 0.1}}', "'gold' must be a number"),
        ('{"a": {"b": 1, "b": 2}}', r"duplicate keys \['b'\]"),
    ],
)
def test_parse_json_components_is_strict(text, fragment):
    with pytest.raises(DecisionParseError, match=fragment):
        parse_json_components(text)


def test_parse_json_components_returns_floats():
    assert parse_json_components('{"a": 1, "b": -0.25}') == {"a": 1.0, "b": -0.25}
    assert all(type(value) is float for value in parse_json_components('{"a": 1}').values())


# -- one optional code fence around the object (fix 9.2a) ---------------------

# The reply Gemma 3 4B gave through llama-server in the first live calibration stage, byte for byte:
# a correct object inside one markdown fence. Every one of the 944 draws was recorded as parse_error.
LIVE_FENCED_REPLY = '```json\n{\n  "equity": 0.3,\n  "gold": 0.15,\n  "bonds": 0.45,\n  "cash": 0.1\n}\n```\n'
LIVE_COMPONENTS = {"equity": 0.3, "gold": 0.15, "bonds": 0.45, "cash": 0.1}
OBJECT = '{"equity": 0.3, "gold": 0.15}'


def test_live_fenced_reply_is_the_recorded_shape():
    assert len(LIVE_FENCED_REPLY) == 80
    assert LIVE_FENCED_REPLY.startswith("```json\n{\n") and LIVE_FENCED_REPLY.endswith("\n}\n```\n")


def test_parse_json_components_accepts_the_live_fenced_reply():
    parsed = parse_json_components(LIVE_FENCED_REPLY)
    assert parsed == LIVE_COMPONENTS
    assert list(parsed) == ["equity", "gold", "bonds", "cash"]
    assert all(type(value) is float for value in parsed.values())


@pytest.mark.parametrize(
    "text",
    [
        f"```json\n{OBJECT}\n```",
        f"```JSON\n{OBJECT}\n```\n",
        f"```\n{OBJECT}\n```",
        f"\n  ```json  \n{OBJECT}\n  ```  \n\n",
        f"```json\r\n{OBJECT}\r\n```\r\n",
        '```json\n{\n  "equity": 0.3,\n\n  "gold": 0.15\n}\n```',
    ],
)
def test_one_fence_with_optional_tag_and_surrounding_whitespace_is_unwrapped(text):
    assert parse_json_components(text) == {"equity": 0.3, "gold": 0.15}


def test_fenced_reply_parses_through_the_definition_like_the_bare_object():
    definition = _definition()
    bare = '{"gold": 0.3, "equity": -0.2}'
    assert definition.parse_response(f"```json\n{bare}\n```\n") == definition.parse_response(bare)


@pytest.mark.parametrize(
    ("text", "fragment"),
    [
        ('```json\n{"equity": 0.1, "gold": 0.2, "gold": -0.9}\n```', r"duplicate keys \['gold'\]"),
        ('```json\n{"gold": NaN}\n```', "non-finite JSON constant NaN"),
        ('```json\n{"gold": Infinity}\n```', "non-finite JSON constant Infinity"),
        ("```json\n[0.1, 0.2]\n```", "must be a JSON object; got list"),
        ("```json\n0.5\n```", "must be a JSON object; got float"),
        ('```json\n{"gold": true}\n```', "'gold' must be a number"),
        ('```json\n{"gold": "0.2"}\n```', "'gold' must be a number"),
        ('```json\n{"gold": 1e999}\n```', "'gold' must be finite"),
        ("```json\n\n```", "not valid JSON"),
    ],
)
def test_strictness_is_unchanged_inside_the_fence(text, fragment):
    with pytest.raises(DecisionParseError, match=fragment):
        parse_json_components(text)


@pytest.mark.parametrize(
    ("text", "fragment"),
    [
        ("```json\n{}\n```", r"missing components \['equity', 'gold'\]"),
        ('```json\n{"equity": 0.1}\n```', r"missing components \['gold'\]"),
        ('```json\n{"equity": 0.1, "gold": 0.2, "cash": 0.0}\n```', r"undeclared components \['cash'\]"),
        ('```json\n{"equity": 0.1, "gold": 1.2}\n```', r"'gold'=1.2 outside \[-1.0, 1.0\]"),
    ],
)
def test_definition_strictness_is_unchanged_inside_the_fence(text, fragment):
    with pytest.raises(DecisionParseError, match=fragment):
        _definition().parse_response(text)


@pytest.mark.parametrize(
    "text",
    [
        f"Here are the loadings: {OBJECT}",
        f"{OBJECT}\nThese sum to less than one.",
        f"{OBJECT} {OBJECT}",
        f"Here you go:\n```json\n{OBJECT}\n```",
        f"```json\n{OBJECT}\n```\nValues sum to 0.45.",
        f"```json\n{OBJECT}\n```\n```json\n{OBJECT}\n```",
        f"```json\n{OBJECT}\n{OBJECT}\n```",
        f"```json\n{OBJECT}\n",
        f"```json\n{OBJECT}",
        f"{OBJECT}\n```",
        f"```json\n```json\n{OBJECT}\n```\n```",
        f"```json\n```\n{OBJECT}\n```",
        f"```json {OBJECT}```",
        f"```json\n{OBJECT}```",
    ],
    ids=[
        "prose-before-object",
        "prose-after-object",
        "two-objects",
        "prose-before-fence",
        "text-after-closing-fence",
        "two-fenced-objects",
        "two-objects-in-one-fence",
        "unclosed-fence",
        "unclosed-fence-no-newline",
        "closing-fence-only",
        "nested-fences",
        "fence-inside-fence",
        "fence-markers-on-the-object-line",
        "closing-fence-not-on-its-own-line",
    ],
)
def test_everything_but_one_object_in_one_fence_is_refused(text):
    with pytest.raises(DecisionParseError) as caught:
        parse_json_components(text)
    # The message names the problem and never echoes the reply.
    assert "equity" not in str(caught.value)


def test_fence_refusal_names_the_shape_without_echoing_the_reply():
    with pytest.raises(DecisionParseError, match="not one JSON object inside one closed code fence") as caught:
        parse_json_components(f"```json\n{OBJECT}\n```\nValues sum to 0.45.")
    assert "0.45" not in str(caught.value)


# -- adjustment ---------------------------------------------------------------


@pytest.mark.parametrize("p", [0.0, 0.211, 0.5, 0.76, 1.0])
def test_adjust_scales_magnitude_and_preserves_sign(p):
    vector = {"inflation": 0.6, "growth": -0.4, "policy": 0.0}
    adjusted = adjust(vector, p)
    assert list(adjusted) == list(vector)
    for name, raw in vector.items():
        assert adjusted[name] == pytest.approx(raw * (1 - p))
        assert math.copysign(1.0, adjusted[name]) == math.copysign(1.0, raw)


@pytest.mark.parametrize(("p", "expected"), [(-0.3, 1.0), (0.0, 1.0), (1.0, 0.0), (1.7, 0.0)])
def test_adjust_clips_estimate_to_unit_interval(p, expected):
    adjusted = adjust({"growth": -0.5, "inflation": 0.8}, p)
    assert adjusted == {"growth": pytest.approx(-0.5 * expected), "inflation": pytest.approx(0.8 * expected)}
    # Full memorization zeroes the magnitude but keeps the sign bit.
    assert math.copysign(1.0, adjusted["growth"]) == -1.0


@pytest.mark.parametrize("p", [float("nan"), float("inf"), float("-inf")])
def test_adjust_rejects_non_finite_estimate(p):
    with pytest.raises(ValueError, match="p_memorized must be a finite number"):
        adjust({"growth": 0.5}, p)


def test_adjust_rejects_unresolved_component():
    with pytest.raises(ValueError, match="component 'growth' must be a finite number"):
        adjust({"growth": None}, 0.2)


def test_adjust_does_not_mutate_input():
    vector = {"growth": 0.5}
    adjust(vector, 0.5)
    assert vector == {"growth": 0.5}


# -- summary for raw and adjusted ---------------------------------------------


def test_summary_value_for_raw_and_adjusted_decisions():
    definition = _definition()
    raw = definition.parse_response('{"equity": -0.4, "gold": 0.6}')
    adjusted = adjust(raw, 0.25)
    assert definition.summary(raw) == pytest.approx(1.0)
    assert definition.summary(adjusted) == pytest.approx(0.75)
    # A uniform scale keeps the categorical decision of a linear summary.
    assert definition.decide(raw) is definition.decide(adjusted) is True


def test_definition_without_summary_or_decide_is_valid():
    definition = _definition(summary=None, decide=None)
    assert definition.parse_response('{"equity": 0.0, "gold": 0.0}') == {"equity": 0.0, "gold": 0.0}
