"""The macro regime worked example: prompts, decision and accuracy rule (Requirements 2.4, 6.1, 6.6).

Macro rows and realized returns are illustrative values, not FRED or FMP data.
Hand-computed spreads use the frozen exposure matrix and are worked out in the
comments next to each vector.
"""

import ast
import json
import math
import re
from dataclasses import FrozenInstanceError, replace
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path

import pytest

from recall_guard.core.decision import DecisionParseError, adjust, parse_json_components
from recall_guard.dataset import macro_regime
from recall_guard.dataset.macro_regime import (
    ASSETS,
    EXPOSURE,
    MACRO_AXES,
    RAW_KEYS,
    ZSCORE_KEYS,
    MacroRow,
    RealizedReturns,
    defensive_spread,
    is_correct,
    regime_definition,
    render_anonymised,
    render_identifying,
    tilts,
)

ROW = MacroRow(
    decision_date=date(2019, 3, 29),
    raw={"cpi_yoy": 1.86, "t10y2y": 0.14, "baa10y": 2.38},
    zscores={"t10y2y_z": -1.236, "cpi_yoy_z": -0.4, "baa10y_z": 0.517},
    revised={"CPIAUCSL": False, "T10Y2Y": True, "BAA10Y": True},
)

# The adopter's base prompt for ROW, line by line (factor_scoring.render_regime_loadings_prompt).
ANONYMISED_LINES = [
    "You are a macroeconomic game theorist. Characterize the current macro regime as continuous factor loadings.",
    "",
    "You are given the macro state (z-scored against a rolling window) and an anonymized list of assets "
    "identified only by letter and category. You do not know what year it is; reason only from the numeric "
    "state in front of you.",
    "",
    "Macro state (z-scored):",
    "  - baa10y_z: +0.52",
    "  - cpi_yoy_z: -0.40",
    "  - t10y2y_z: -1.24",
    "",
    "Assets (anonymized):",
    "  - Asset_A: world_equity",
    "  - Asset_B: tech_sector",
    "  - Asset_C: gold_commodity",
    "  - Asset_D: short_treasury_cash",
    "",
    "Characterize the regime as a continuous loading on each of these five macro axes:",
    "  - inflation",
    "  - growth",
    "  - credit_stress",
    "  - policy",
    "  - risk_appetite",
    "",
    "Rules:",
    "  - Each loading is a continuous number in [-1, +1] (use -1 and +1 as the bounds).",
    "  - A loading describes how strongly the regime sits on that axis \u2014 it is NOT a trade, a bet, "
    "or a prediction.",
    "  - Do NOT output a long/short/neutral position and do NOT output a return for any asset; "
    "characterize the regime only.",
    "",
    'Output a JSON object mapping each axis name to its loading, e.g. {"inflation": 0.4, "growth": -0.2, '
    '"credit_stress": 0.6, "policy": -0.1, "risk_appetite": -0.3}.',
]

# Exactly the three appended blocks: as-of date, real identities, raw levels.
IDENTIFYING_ADDITION = "\n".join([
    "",
    "",
    "As-of date: 2019-03-29",
    "",
    "Real asset identities:",
    "  - Asset_A: SWDA.L",
    "  - Asset_B: XLK",
    "  - Asset_C: IAU",
    "  - Asset_D: DTB3 (3-month US Treasury bill)",
    "",
    "Raw (non-normalized) macro levels:",
    "  - baa10y: 2.38",
    "  - cpi_yoy: 1.86",
    "  - t10y2y: 0.14",
])

REAL_IDENTITIES = ("SWDA.L", "XLK", "IAU", "DTB3 (3-month US Treasury bill)")
TICKERS = ("SWDA", "XLK", "IAU", "DTB3", "BIL", "Treasury bill")
YEAR = re.compile(r"(?<!\d)(?:19|20)\d{2}(?!\d)")

# Defensive regime, the prompt's own example answer.
#   world_equity   -0.12 - 0.20 - 0.48 + 0.03 - 0.18 = -0.95
#   tech_sector    -0.24 - 0.20 - 0.42 + 0.05 - 0.24 = -1.05
#   gold_commodity  0.32 + 0.08 + 0.42 - 0.01 + 0.15 =  0.96
#   short_treasury  0.12 + 0.12 + 0.36 - 0.08 + 0.18 =  0.70
#   spread = (0.96 + 0.70) - (-0.95 - 1.05) = 3.66
DEFENSIVE = {"inflation": 0.4, "growth": -0.2, "credit_stress": 0.6, "policy": -0.1, "risk_appetite": -0.3}
DEFENSIVE_TILTS = {"world_equity": -0.95, "tech_sector": -1.05, "gold_commodity": 0.96, "short_treasury_cash": 0.70}
DEFENSIVE_SPREAD = 3.66

# Risk-on regime.
#   world_equity    0.06 + 0.70 + 0.40 - 0.09 + 0.48 =  1.55
#   tech_sector     0.12 + 0.70 + 0.35 - 0.15 + 0.64 =  1.66
#   gold_commodity -0.16 - 0.28 - 0.35 + 0.03 - 0.40 = -1.16
#   short_treasury -0.06 - 0.42 - 0.30 + 0.24 - 0.48 = -1.02
#   spread = (-1.16 - 1.02) - (1.55 + 1.66) = -5.39
RISK_ON = {"inflation": -0.2, "growth": 0.7, "credit_stress": -0.5, "policy": 0.3, "risk_appetite": 0.8}
RISK_ON_TILTS = {"world_equity": 1.55, "tech_sector": 1.66, "gold_commodity": -1.16, "short_treasury_cash": -1.02}
RISK_ON_SPREAD = -5.39

ZEROS = dict.fromkeys(MACRO_AXES, 0.0)

# Defensive minus risk column sums of the frozen exposure matrix (the adopter's appendix I).
SPREAD_WEIGHTS = {"inflation": 2.0, "growth": -3.0, "credit_stress": 2.8, "policy": 1.7, "risk_appetite": -2.5}

# Realized next-month simple returns; defensive minus risk is (gold + cash) - (equity + tech).
DEFENSIVE_MONTH = RealizedReturns(world_equity=-0.042, tech_sector=-0.061, gold_commodity=0.027,
                                  short_treasury_cash=0.0019)   # +0.1319
RISK_MONTH = RealizedReturns(world_equity=0.035, tech_sector=0.052, gold_commodity=-0.012,
                             short_treasury_cash=0.0017)        # -0.0973
FLAT_MONTH = RealizedReturns(world_equity=0.005, tech_sector=0.005, gold_commodity=0.01,
                             short_treasury_cash=0.0)           # exactly 0.0
MONTHS = (DEFENSIVE_MONTH, RISK_MONTH, FLAT_MONTH)

# Grid vectors whose spread is zero in decimal arithmetic but not in floats.
POSITIVE_RESIDUE = [
    {**ZEROS, "inflation": -0.3, "growth": -0.2},                          # -0.6 + 0.6, float +5.6e-17
    {**ZEROS, "growth": 0.3, "credit_stress": 0.2, "policy": 0.2},         # -0.9 + 0.56 + 0.34, float +5.6e-17
    {**ZEROS, "inflation": -0.2, "credit_stress": -0.1, "policy": 0.4},    # -0.4 - 0.28 + 0.68, float +6.9e-17
]
NEGATIVE_RESIDUE = [
    {**ZEROS, "inflation": 0.3, "growth": 0.2},                            # 0.6 - 0.6, float -5.6e-17
    {**ZEROS, "credit_stress": 0.3, "policy": -0.2, "risk_appetite": 0.2},  # 0.84 - 0.34 - 0.5, float -9.7e-17
]
# Months whose defensive minus risk return is zero in decimal arithmetic but not in floats.
POSITIVE_RESIDUE_MONTH = RealizedReturns(world_equity=0.04, tech_sector=0.02, gold_commodity=0.05,
                                         short_treasury_cash=0.01)   # 0.06 - 0.06, float +6.9e-18
NEGATIVE_RESIDUE_MONTH = RealizedReturns(world_equity=0.05, tech_sector=0.01, gold_commodity=0.04,
                                         short_treasury_cash=0.02)   # 0.06 - 0.06, float -6.9e-18


def decimal_spread(loadings) -> Decimal:
    return sum(Decimal(str(SPREAD_WEIGHTS[axis])) * Decimal(str(loadings[axis])) for axis in MACRO_AXES)


def realized_difference(realized: RealizedReturns) -> float:
    return (realized.gold_commodity + realized.short_treasury_cash) - (realized.world_equity + realized.tech_sector)


def answer(loadings, **changes) -> str:
    return json.dumps({**loadings, **changes})


def block(prompt: str, heading: str) -> list[str]:
    """The indented lines that follow ``heading`` up to the next blank line."""
    lines = prompt.split("\n")
    start = lines.index(heading) + 1
    end = lines.index("", start) if "" in lines[start:] else len(lines)
    return lines[start:end]


# -- renderings (6.1) --------------------------------------------------------


def test_anonymised_prompt_mirrors_the_adopter_base_lines():
    assert render_anonymised(ROW) == "\n".join(ANONYMISED_LINES)


def test_identifying_prompt_is_the_anonymised_prompt_plus_only_date_identities_and_raw_levels():
    anonymised = render_anonymised(ROW)
    identifying = render_identifying(ROW)

    assert identifying.startswith(anonymised)
    assert identifying.split("\n")[: len(ANONYMISED_LINES)] == ANONYMISED_LINES
    assert identifying[len(anonymised):] == IDENTIFYING_ADDITION


def test_anonymised_prompt_carries_no_year_ticker_date_or_raw_level():
    anonymised = render_anonymised(ROW)

    assert YEAR.search(anonymised) is None
    assert ROW.decision_date.isoformat() not in anonymised
    for ticker in TICKERS:
        assert ticker not in anonymised
    for identity in REAL_IDENTITIES:
        assert identity not in anonymised
    for level in ("2.38", "1.86", "0.14"):
        assert level not in anonymised
    assert "As-of date" not in anonymised and "Raw (non-normalized)" not in anonymised


def test_identifying_prompt_discloses_the_date_every_real_identity_and_every_raw_level():
    identifying = render_identifying(ROW)

    assert YEAR.search(identifying).group() == "2019"
    assert block(identifying, "Real asset identities:") == [
        f"  - {pseudo}: {real}" for pseudo, _, real in ASSETS
    ]
    assert [real for _, _, real in ASSETS] == list(REAL_IDENTITIES)
    assert block(identifying, "Raw (non-normalized) macro levels:") == [
        "  - baa10y: 2.38", "  - cpi_yoy: 1.86", "  - t10y2y: 0.14",
    ]


def test_rows_that_differ_only_in_identifying_fields_render_the_same_anonymised_prompt():
    other = replace(ROW, decision_date=date(2008, 10, 31), raw={"cpi_yoy": 4.9, "t10y2y": 1.73, "baa10y": 4.92},
                    revised={"CPIAUCSL": True})

    assert render_anonymised(other) == render_anonymised(ROW)
    assert render_identifying(other) != render_identifying(ROW)
    assert render_identifying(other).startswith(render_anonymised(ROW))


def test_rendering_is_deterministic_and_independent_of_insertion_order():
    reordered = MacroRow(
        decision_date=date(2019, 3, 29),
        raw=dict(reversed(list(ROW.raw.items()))),
        zscores=dict(reversed(list(ROW.zscores.items()))),
        revised=dict(ROW.revised),
    )

    assert render_anonymised(ROW) == render_anonymised(ROW) == render_anonymised(reordered)
    assert render_identifying(ROW) == render_identifying(ROW) == render_identifying(reordered)


def test_macro_state_and_raw_levels_are_sorted_by_key_with_the_adopter_number_formats():
    row = MacroRow(
        decision_date=date(2023, 1, 31),
        raw={"t10y2y": -0.03, "cpi_yoy": 6.4123456, "baa10y": 2},
        zscores={"t10y2y_z": 1.5, "cpi_yoy_z": 0, "baa10y_z": -2.004},
        revised={},
    )
    identifying = render_identifying(row)

    assert block(identifying, "Macro state (z-scored):") == [
        "  - baa10y_z: -2.00", "  - cpi_yoy_z: +0.00", "  - t10y2y_z: +1.50",
    ]
    assert block(identifying, "Raw (non-normalized) macro levels:") == [
        "  - baa10y: 2", "  - cpi_yoy: 6.41235", "  - t10y2y: -0.03",
    ]
    assert block(render_anonymised(row), "Macro state (z-scored):") == block(identifying, "Macro state (z-scored):")


def test_assets_and_their_real_identities_are_frozen():
    assert ASSETS == (
        ("Asset_A", "world_equity", "SWDA.L"),
        ("Asset_B", "tech_sector", "XLK"),
        ("Asset_C", "gold_commodity", "IAU"),
        ("Asset_D", "short_treasury_cash", "DTB3 (3-month US Treasury bill)"),
    )


@pytest.mark.parametrize(("changes", "error"), [
    ({"raw": {"cpi_yoy": 1.86, "t10y2y": 0.14}}, ValueError),
    ({"raw": {"cpi_yoy": 1.86, "t10y2y": 0.14, "baa10y": 2.38, "dtb3": 2.4}}, ValueError),
    ({"zscores": {"cpi_yoy_z": -0.4, "t10y2y_z": -1.236, "hy_oas_z": 0.517}}, ValueError),
    ({"zscores": {"cpi_yoy_z": math.nan, "t10y2y_z": -1.236, "baa10y_z": 0.517}}, ValueError),
    ({"raw": {"cpi_yoy": math.inf, "t10y2y": 0.14, "baa10y": 2.38}}, ValueError),
    ({"raw": {"cpi_yoy": True, "t10y2y": 0.14, "baa10y": 2.38}}, TypeError),
    ({"raw": {"cpi_yoy": "1.86", "t10y2y": 0.14, "baa10y": 2.38}}, TypeError),
    ({"raw": [("cpi_yoy", 1.86)]}, TypeError),
    ({"revised": {"CPIAUCSL": 1}}, TypeError),
    ({"revised": None}, TypeError),
    ({"decision_date": datetime(2019, 3, 29, 12)}, TypeError),
    ({"decision_date": "2019-03-29"}, TypeError),
])
def test_macro_row_fields_are_validated(changes, error):
    fields = {"decision_date": ROW.decision_date, "raw": ROW.raw, "zscores": ROW.zscores, "revised": ROW.revised}
    with pytest.raises(error):
        MacroRow(**{**fields, **changes})


def test_macro_row_keys_are_the_three_series_and_values_become_floats():
    row = MacroRow(decision_date=date(2019, 3, 29), raw={"cpi_yoy": 2, "t10y2y": 0, "baa10y": 3},
                   zscores={"cpi_yoy_z": 1, "t10y2y_z": 0, "baa10y_z": -1}, revised={})

    assert tuple(sorted(row.raw)) == RAW_KEYS and tuple(sorted(row.zscores)) == ZSCORE_KEYS
    assert all(type(value) is float for value in [*row.raw.values(), *row.zscores.values()])
    with pytest.raises(FrozenInstanceError):
        row.decision_date = date(2020, 1, 31)


# -- exposure, tilts and spread (2.4) ----------------------------------------


def test_exposure_matrix_is_the_frozen_adopter_matrix():
    assert EXPOSURE == {
        "world_equity": {"inflation": -0.3, "growth": 1.0, "credit_stress": -0.8, "policy": -0.3,
                         "risk_appetite": 0.6},
        "tech_sector": {"inflation": -0.6, "growth": 1.0, "credit_stress": -0.7, "policy": -0.5,
                        "risk_appetite": 0.8},
        "gold_commodity": {"inflation": 0.8, "growth": -0.4, "credit_stress": 0.7, "policy": 0.1,
                           "risk_appetite": -0.5},
        "short_treasury_cash": {"inflation": 0.3, "growth": -0.6, "credit_stress": 0.6, "policy": 0.8,
                                "risk_appetite": -0.6},
    }
    assert list(EXPOSURE) == [category for _, category, _ in ASSETS]
    assert all(tuple(row) == MACRO_AXES for row in EXPOSURE.values())
    with pytest.raises(TypeError):
        EXPOSURE["world_equity"]["growth"] = 0.0
    with pytest.raises(TypeError):
        EXPOSURE["cash"] = {}


@pytest.mark.parametrize(("loadings", "expected_tilts", "expected_spread"), [
    (DEFENSIVE, DEFENSIVE_TILTS, DEFENSIVE_SPREAD),
    (RISK_ON, RISK_ON_TILTS, RISK_ON_SPREAD),
])
def test_tilts_and_defensive_spread_match_hand_computation(loadings, expected_tilts, expected_spread):
    computed = tilts(loadings)

    assert list(computed) == [category for _, category, _ in ASSETS]
    assert computed == pytest.approx(expected_tilts, abs=1e-12)
    assert defensive_spread(loadings) == pytest.approx(expected_spread, abs=1e-12)
    assert defensive_spread(loadings) == pytest.approx(
        sum(SPREAD_WEIGHTS[axis] * loadings[axis] for axis in MACRO_AXES), abs=1e-12)


@pytest.mark.parametrize("axis", MACRO_AXES)
def test_spread_weight_per_axis_is_pinned(axis):
    one_hot = {name: float(name == axis) for name in MACRO_AXES}

    assert defensive_spread(one_hot) == pytest.approx(SPREAD_WEIGHTS[axis], abs=1e-12)


@pytest.mark.parametrize(("loadings", "error"), [
    ({key: value for key, value in DEFENSIVE.items() if key != "policy"}, ValueError),
    ({**DEFENSIVE, "liquidity": 0.2}, ValueError),
    ({**DEFENSIVE, "policy": None}, TypeError),
    ({**DEFENSIVE, "policy": True}, TypeError),
    ({**DEFENSIVE, "policy": math.nan}, ValueError),
    (list(DEFENSIVE.items()), TypeError),
])
def test_tilts_refuse_an_incomplete_or_non_numeric_vector(loadings, error):
    with pytest.raises(error):
        tilts(loadings)
    with pytest.raises(error):
        defensive_spread(loadings)


# -- the regime decision definition (2.4) ------------------------------------


def test_regime_definition_declares_five_loadings_in_unit_bounds_on_a_tenth_grid():
    assert regime_definition.name == "macro_regime"
    assert [(spec.name, spec.lower, spec.upper, spec.grid) for spec in regime_definition.components] == [
        (axis, -1.0, 1.0, 0.1) for axis in MACRO_AXES
    ]
    assert regime_definition.parse is parse_json_components
    assert regime_definition.summary is defensive_spread


def test_regime_definition_parses_a_json_answer_into_the_spread_and_its_sign():
    vector = regime_definition.parse_response(answer(DEFENSIVE))

    assert vector == DEFENSIVE and list(vector) == list(MACRO_AXES)
    assert regime_definition.summary(vector) == pytest.approx(DEFENSIVE_SPREAD, abs=1e-12)
    assert regime_definition.decide(vector) == 1


@pytest.mark.parametrize(("text", "message"), [
    (answer(DEFENSIVE, inflation=1.2), "outside"),
    (answer(DEFENSIVE, risk_appetite=-1.01), "outside"),
    (json.dumps({key: value for key, value in DEFENSIVE.items() if key != "risk_appetite"}), "missing"),
    (answer(DEFENSIVE, liquidity=0.1), "undeclared"),
    ('{"inflation": 0.4, "inflation": -0.4, "growth": -0.2, "credit_stress": 0.6, "policy": -0.1, '
     '"risk_appetite": -0.3}', "duplicate"),
    ('{"inflation": NaN, "growth": -0.2, "credit_stress": 0.6, "policy": -0.1, "risk_appetite": -0.3}',
     "non-finite"),
    ("inflation 0.4, growth -0.2", "not valid JSON"),
])
def test_regime_definition_rejects_out_of_range_missing_and_malformed_answers(text, message):
    with pytest.raises(DecisionParseError, match=message):
        regime_definition.parse_response(text)


@pytest.mark.parametrize(("loadings", "expected"), [
    (DEFENSIVE, 1),
    (RISK_ON, -1),
    (ZEROS, -1),
    ({**ZEROS, "inflation": 0.1}, 1),         # spread +0.2
    ({**ZEROS, "growth": 0.1}, -1),           # spread -0.3
])
def test_decide_is_one_above_zero_and_minus_one_at_or_below_zero(loadings, expected):
    assert regime_definition.decide(loadings) == expected
    assert regime_definition.decide(loadings) == (1 if defensive_spread(loadings) > 0 else -1)


# -- accuracy rule (6.6) -----------------------------------------------------


@pytest.mark.parametrize(("spread", "realized", "expected"), [
    (DEFENSIVE_SPREAD, DEFENSIVE_MONTH, True),
    (RISK_ON_SPREAD, DEFENSIVE_MONTH, False),
    (0.0, DEFENSIVE_MONTH, False),            # a zero spread is risk-on
    (DEFENSIVE_SPREAD, RISK_MONTH, False),
    (RISK_ON_SPREAD, RISK_MONTH, True),
    (0.0, RISK_MONTH, True),
    (DEFENSIVE_SPREAD, FLAT_MONTH, False),    # a zero realized difference is a risk-on outcome
    (RISK_ON_SPREAD, FLAT_MONTH, True),
    (0.0, FLAT_MONTH, True),
    (1e-8, FLAT_MONTH, False),                # beyond the nine-decimal rounding: defensive
    (1e-300, FLAT_MONTH, True),               # residue: rounds to zero, so risk-on
    (-0.0, DEFENSIVE_MONTH, False),
])
def test_is_correct_compares_the_spread_sign_with_defensive_minus_risk(spread, realized, expected):
    assert is_correct(spread, realized) is expected


def test_flat_month_has_exactly_zero_defensive_minus_risk():
    assert realized_difference(FLAT_MONTH) == 0.0


# -- floating-point residue never decides a sign ------------------------------


@pytest.mark.parametrize(("loadings", "residue_sign"), [
    *((loadings, 1) for loadings in POSITIVE_RESIDUE),
    *((loadings, -1) for loadings in NEGATIVE_RESIDUE),
])
def test_a_spread_that_is_zero_in_decimal_arithmetic_counts_as_risk_on_whatever_its_residue(loadings, residue_sign):
    spread = defensive_spread(loadings)

    assert decimal_spread(loadings) == 0
    assert 0 < abs(spread) < 1e-15 and math.copysign(1, spread) == residue_sign
    assert regime_definition.decide(loadings) == -1
    assert regime_definition.decide(regime_definition.parse_response(answer(loadings))) == -1
    assert is_correct(spread, DEFENSIVE_MONTH) is False
    for realized in MONTHS:
        assert is_correct(spread, realized) is is_correct(0.0, realized)


@pytest.mark.parametrize("p_memorized", [0.3, 0.35, 0.7])
@pytest.mark.parametrize("loadings", [*POSITIVE_RESIDUE, *NEGATIVE_RESIDUE])
def test_adjusting_a_zero_spread_vector_keeps_it_risk_on(loadings, p_memorized):
    adjusted = adjust(loadings, p_memorized)

    assert regime_definition.decide(adjusted) == regime_definition.decide(loadings) == -1
    for realized in MONTHS:
        assert is_correct(defensive_spread(adjusted), realized) is is_correct(defensive_spread(loadings), realized)


@pytest.mark.parametrize("realized", [POSITIVE_RESIDUE_MONTH, NEGATIVE_RESIDUE_MONTH])
def test_a_realized_difference_that_is_zero_in_decimal_arithmetic_is_a_risk_on_outcome(realized):
    assert 0 < abs(realized_difference(realized)) < 1e-15

    assert is_correct(DEFENSIVE_SPREAD, realized) is False
    assert is_correct(RISK_ON_SPREAD, realized) is True
    assert is_correct(0.0, realized) is True
    assert is_correct(defensive_spread(POSITIVE_RESIDUE[0]), realized) is True


def test_spreads_and_differences_beyond_the_rounding_keep_their_sign():
    assert regime_definition.decide({**ZEROS, "policy": 0.1}) == 1        # spread +0.17
    assert is_correct(1e-8, DEFENSIVE_MONTH) is True
    assert is_correct(-1e-8, DEFENSIVE_MONTH) is False
    small_month = RealizedReturns(world_equity=0.0, tech_sector=0.0, gold_commodity=1e-8, short_treasury_cash=0.0)
    assert is_correct(DEFENSIVE_SPREAD, small_month) is True


@pytest.mark.parametrize("loadings", [DEFENSIVE, RISK_ON])
@pytest.mark.parametrize("p_memorized", [-0.2, 0.0, 0.35, 0.9])
def test_raw_and_adjusted_vectors_score_the_same_on_the_same_returns(loadings, p_memorized):
    raw_spread = defensive_spread(loadings)
    adjusted_spread = defensive_spread(adjust(loadings, p_memorized))

    assert (raw_spread > 0) == (adjusted_spread > 0)
    for realized in MONTHS:
        assert is_correct(raw_spread, realized) is is_correct(adjusted_spread, realized)
        assert regime_definition.decide(adjust(loadings, p_memorized)) == regime_definition.decide(loadings)


@pytest.mark.parametrize("p_memorized", [1.0, 1.5])
def test_full_adjustment_gives_a_zero_spread_that_scores_as_risk_on(p_memorized):
    adjusted_spread = defensive_spread(adjust(DEFENSIVE, p_memorized))

    assert adjusted_spread == 0.0
    for realized in MONTHS:
        assert is_correct(adjusted_spread, realized) is is_correct(0.0, realized)


def test_is_correct_is_pure_and_leaves_the_returns_unchanged():
    before = replace(DEFENSIVE_MONTH)

    results = {is_correct(DEFENSIVE_SPREAD, DEFENSIVE_MONTH) for _ in range(3)}

    assert results == {True}
    assert DEFENSIVE_MONTH == before
    with pytest.raises(FrozenInstanceError):
        DEFENSIVE_MONTH.gold_commodity = 0.5


@pytest.mark.parametrize(("spread", "realized", "error"), [
    (math.nan, DEFENSIVE_MONTH, ValueError),
    (math.inf, DEFENSIVE_MONTH, ValueError),
    (True, DEFENSIVE_MONTH, TypeError),
    ("3.66", DEFENSIVE_MONTH, TypeError),
    (DEFENSIVE_SPREAD, {"world_equity": 0.0, "tech_sector": 0.0, "gold_commodity": 0.0,
                        "short_treasury_cash": 0.0}, TypeError),
])
def test_is_correct_refuses_a_non_finite_spread_or_untyped_returns(spread, realized, error):
    with pytest.raises(error):
        is_correct(spread, realized)


@pytest.mark.parametrize(("field", "value", "error"), [
    ("gold_commodity", math.nan, ValueError),
    ("tech_sector", -math.inf, ValueError),
    ("world_equity", True, TypeError),
    ("short_treasury_cash", "0.002", TypeError),
    ("short_treasury_cash", None, TypeError),
])
def test_realized_returns_are_finite_numbers(field, value, error):
    fields = {"world_equity": 0.01, "tech_sector": 0.02, "gold_commodity": -0.01, "short_treasury_cash": 0.002}
    with pytest.raises(error):
        RealizedReturns(**{**fields, field: value})


def test_realized_returns_become_floats():
    realized = RealizedReturns(world_equity=0, tech_sector=1, gold_commodity=-1, short_treasury_cash=0)

    assert all(type(getattr(realized, name)) is float
               for name in ("world_equity", "tech_sector", "gold_commodity", "short_treasury_cash"))


# -- no I/O ------------------------------------------------------------------

ALLOWED_IMPORTS = {
    "__future__", "collections.abc", "dataclasses", "datetime", "math", "numbers", "types",
    "recall_guard.core.decision",
}


def test_module_does_no_io_and_imports_no_fetching_code():
    tree = ast.parse(Path(macro_regime.__file__).read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module)
    called = {node.func.id for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}

    assert imported <= ALLOWED_IMPORTS
    assert not called & {"open", "print", "input", "exec", "eval", "__import__"}
    assert not {"requests", "fred", "fetch_point_in_time", "Path"} & set(vars(macro_regime))
