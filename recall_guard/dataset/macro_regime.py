"""The macro regime worked example: prompts, decision and accuracy rule (Requirements 2.4, 6.1, 6.6).

Each month of the worked example is one :class:`MacroRow`: three macro series as
they stood on the decision date (CPI year-on-year, the 10-year minus 2-year
Treasury spread T10Y2Y, and the Moody's Baa minus 10-year spread BAA10Y, which
replaces the discontinued high-yield OAS), as raw levels and as rolling z-scores.
The row is rendered twice. The anonymised prompt is the point-in-time form: the
z-scored state and four assets known only by letter and category, with no
date, no year and no ticker. The identifying prompt is that same text followed by
three blocks, the as-of date, the real asset identities and the raw levels, so a
contrast between the two attributes any difference to those additions alone.

The renderer mirrors ``render_regime_loadings_prompt`` in the adopter project
(Global_Macro_AI_Factors ``macro_framework/factor_scoring.py``) line for line.
Two things differ, both fixed by the design: the macro state keys
(``baa10y_z`` instead of ``hy_oas_z``) and the cash asset's real identity, which is
DTB3, the 3-month Treasury bill rate from FRED, rather than the BIL fund.

The model answers with five regime loadings in [-1, 1] (:data:`regime_definition`).
The frozen :data:`EXPOSURE` matrix turns them into one tilt per asset, and the
defensive spread, ``(gold + cash) - (equity + tech)`` in tilts, is the summary
value; its sign is the categorical decision. The spread is linear in the
loadings, so the memorization adjustment, which scales every loading by the same
non-negative factor, scales the spread too and keeps its sign.

Accuracy (6.6) is :func:`is_correct`, a pure function of a spread and the
realized next-month returns it is given. Nothing here fetches data: the study's
item assembly (7.1) builds the rows from the FRED loader and the returns from
FMP adjusted closes and DTB3.
"""

from __future__ import annotations

import math
import numbers
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, datetime
from types import MappingProxyType

from recall_guard.core.decision import ComponentSpec, DecisionDefinition, parse_json_components

#: The five regime axes, in the order the prompt lists them and the definition declares them.
MACRO_AXES: tuple[str, ...] = ("inflation", "growth", "credit_stress", "policy", "risk_appetite")
#: Macro state keys: 60-month rolling z-scores (ddof 1, at least 30 months) of the raw levels.
ZSCORE_KEYS: tuple[str, ...] = ("baa10y_z", "cpi_yoy_z", "t10y2y_z")
#: Raw level keys, in percent: CPI year-on-year, T10Y2Y and BAA10Y.
RAW_KEYS: tuple[str, ...] = ("baa10y", "cpi_yoy", "t10y2y")

#: (pseudo letter, category, real identity) per asset; the identity appears only in the identifying prompt.
ASSETS: tuple[tuple[str, str, str], ...] = (
    ("Asset_A", "world_equity", "SWDA.L"),
    ("Asset_B", "tech_sector", "XLK"),
    ("Asset_C", "gold_commodity", "IAU"),
    ("Asset_D", "short_treasury_cash", "DTB3 (3-month US Treasury bill)"),
)

#: Tilt per unit loading, per asset category and axis; dimensionless and read-only.
#: A copy of the adopter's ``REGIME_ASSET_EXPOSURE``.
EXPOSURE: Mapping[str, Mapping[str, float]] = MappingProxyType({
    "world_equity": MappingProxyType(
        {"inflation": -0.3, "growth": 1.0, "credit_stress": -0.8, "policy": -0.3, "risk_appetite": 0.6}),
    "tech_sector": MappingProxyType(
        {"inflation": -0.6, "growth": 1.0, "credit_stress": -0.7, "policy": -0.5, "risk_appetite": 0.8}),
    "gold_commodity": MappingProxyType(
        {"inflation": 0.8, "growth": -0.4, "credit_stress": 0.7, "policy": 0.1, "risk_appetite": -0.5}),
    "short_treasury_cash": MappingProxyType(
        {"inflation": 0.3, "growth": -0.6, "credit_stress": 0.6, "policy": 0.8, "risk_appetite": -0.6}),
})

#: The step the loadings are declared on.
LOADING_GRID = 0.1
# Decimals a value keeps before its sign is tested (design: ``round(spread, 9)``).
_SIGN_DECIMALS = 9


def _finite(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(f"{name} must be a finite number; got {value!r}")
    try:
        number = float(value)
    except OverflowError:
        raise ValueError(f"{name} must be a finite number") from None
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite number; got {value!r}")
    return number


def _numbers(mapping, keys: tuple[str, ...], name: str) -> dict[str, float]:
    """Exactly ``keys`` with finite numbers, as floats in key order."""
    if not isinstance(mapping, Mapping):
        raise TypeError(f"{name} must be a mapping; got {type(mapping).__name__}")
    missing = [key for key in keys if key not in mapping]
    extra = sorted(str(key) for key in mapping if key not in keys)
    if missing or extra:
        raise ValueError(f"{name} must have exactly the keys {list(keys)}; missing {missing}, undeclared {extra}")
    return {key: _finite(mapping[key], f"{name}[{key!r}]") for key in keys}


@dataclass(frozen=True)
class MacroRow:
    """One decision month: the macro series as available on ``decision_date``.

    ``raw`` holds the levels in percent under :data:`RAW_KEYS`; ``zscores`` holds
    their rolling z-scores under :data:`ZSCORE_KEYS`. ``revised`` maps each FRED
    series id to its ``Observation.revised_data`` flag, so a series used without
    a vintage stays labelled as revised data (6.5); it is not rendered. Values
    are stored as floats, keys sorted.
    """

    decision_date: date
    raw: Mapping[str, float]
    zscores: Mapping[str, float]
    revised: Mapping[str, bool]

    def __post_init__(self) -> None:
        if not isinstance(self.decision_date, date) or isinstance(self.decision_date, datetime):
            raise TypeError("decision_date must be a date, not a datetime")
        object.__setattr__(self, "raw", _numbers(self.raw, RAW_KEYS, "raw"))
        object.__setattr__(self, "zscores", _numbers(self.zscores, ZSCORE_KEYS, "zscores"))
        if not isinstance(self.revised, Mapping):
            raise TypeError(f"revised must be a mapping; got {type(self.revised).__name__}")
        for series_id, flag in self.revised.items():
            if not isinstance(series_id, str) or not isinstance(flag, bool):
                raise TypeError(f"revised must map series ids to bool; got {series_id!r}: {flag!r}")
        object.__setattr__(self, "revised", dict(sorted(self.revised.items())))


@dataclass(frozen=True)
class RealizedReturns:
    """Simple next-month returns of the four assets, as decimals (0.01 is one percent)."""

    world_equity: float
    tech_sector: float
    gold_commodity: float
    short_treasury_cash: float

    def __post_init__(self) -> None:
        for name in ("world_equity", "tech_sector", "gold_commodity", "short_treasury_cash"):
            object.__setattr__(self, name, _finite(getattr(self, name), name))


def _base_lines(row: MacroRow) -> list[str]:
    """The anonymised prompt, line by line, as in the adopter's renderer."""
    return [
        "You are a macroeconomic game theorist. Characterize the current macro regime "
        "as continuous factor loadings.",
        "",
        "You are given the macro state (z-scored against a rolling window) and an "
        "anonymized list of assets identified only by letter and category. You do not "
        "know what year it is; reason only from the numeric state in front of you.",
        "",
        "Macro state (z-scored):",
        *(f"  - {key}: {value:+.2f}" for key, value in sorted(row.zscores.items())),
        "",
        "Assets (anonymized):",
        *(f"  - {pseudo}: {category}" for pseudo, category, _ in ASSETS),
        "",
        "Characterize the regime as a continuous loading on each of these five macro axes:",
        *(f"  - {axis}" for axis in MACRO_AXES),
        "",
        "Rules:",
        "  - Each loading is a continuous number in [-1, +1] (use -1 and +1 as the bounds).",
        # The adopter's wording, kept verbatim; the escape keeps the glyph out of this source file.
        "  - A loading describes how strongly the regime sits on that axis \u2014 it is NOT a "
        "trade, a bet, or a prediction.",
        "  - Do NOT output a long/short/neutral position and do NOT output a return for any "
        "asset; characterize the regime only.",
        "",
        'Output a JSON object mapping each axis name to its loading, e.g. '
        '{"inflation": 0.4, "growth": -0.2, "credit_stress": 0.6, "policy": -0.1, '
        '"risk_appetite": -0.3}.',
    ]


def render_anonymised(row: MacroRow) -> str:
    """The point-in-time prompt: z-scored state and anonymised assets, no date, ticker or raw level."""
    return "\n".join(_base_lines(row))


def render_identifying(row: MacroRow) -> str:
    """The anonymised prompt verbatim, then the as-of date, the real identities and the raw levels.

    The date is the row's decision date and the identities are those of
    :data:`ASSETS`, so the identifying prompt cannot disagree with the row it
    renders.
    """
    identifying_block = [
        "",
        f"As-of date: {row.decision_date.isoformat()}",
        "",
        "Real asset identities:",
        *(f"  - {pseudo}: {real}" for pseudo, _, real in ASSETS),
        "",
        "Raw (non-normalized) macro levels:",
        *(f"  - {key}: {value:g}" for key, value in sorted(row.raw.items())),
    ]
    return "\n".join(_base_lines(row) + identifying_block)


def tilts(loadings: Mapping[str, float]) -> dict[str, float]:
    """Tilt per asset category: the sum over axes of loading times exposure, in :data:`ASSETS` order.

    ``loadings`` must hold exactly the five :data:`MACRO_AXES` as finite numbers.
    """
    vector = _numbers(loadings, MACRO_AXES, "loadings")
    return {
        category: math.fsum(vector[axis] * EXPOSURE[category][axis] for axis in MACRO_AXES)
        for _, category, _ in ASSETS
    }


def defensive_spread(loadings: Mapping[str, float]) -> float:
    """``(tilt_gold + tilt_cash) - (tilt_equity + tilt_tech)``; above zero leans defensive.

    Per axis the weight is the defensive minus risk column sum of :data:`EXPOSURE`:
    inflation 2.0, growth -3.0, credit_stress 2.8, policy 1.7, risk_appetite -2.5.
    """
    tilt = tilts(loadings)
    return (tilt["gold_commodity"] + tilt["short_treasury_cash"]) - (tilt["world_equity"] + tilt["tech_sector"])


def _defensive_minus_risk(realized: RealizedReturns) -> float:
    return (realized.gold_commodity + realized.short_treasury_cash) - (realized.world_equity + realized.tech_sector)


def _positive(value: float) -> bool:
    return round(value, _SIGN_DECIMALS) > 0


def is_correct(spread: float, realized: RealizedReturns) -> bool:
    """Whether the spread's sign matches the realized defensive minus risk return.

    ``(round(spread, 9) > 0) == (round((gold + cash) - (equity + tech), 9) > 0)``.
    Both values are rounded to nine decimals before their signs are tested, so a
    value that is zero in decimal arithmetic stays zero whatever its floating-point
    residue (below 1e-15 for loadings on the 0.1 grid). Zero on either side counts
    as risk-on, as in the ``decide`` of :data:`regime_definition`. The answer
    depends only on the spread and the returns given, so every variant is scored
    by the same rule.
    """
    value = _finite(spread, "spread")
    if not isinstance(realized, RealizedReturns):
        raise TypeError(f"realized must be RealizedReturns; got {type(realized).__name__}")
    return _positive(value) == _positive(_defensive_minus_risk(realized))


def _decide(loadings: Mapping[str, float]) -> int:
    """1 (defensive) when ``round(spread, 9)`` is above zero, otherwise -1 (risk-on, zero included).

    The rounding removes floating-point residue: a spread that is zero in decimal
    arithmetic on the 0.1 grid can come out as about 1e-16 of either sign, and it
    still counts as risk-on. It departs from the adopter, which tests the raw float.
    """
    return 1 if _positive(defensive_spread(loadings)) else -1


#: The worked example's decision (2.4): five loadings in [-1, 1] on a 0.1 grid,
#: summarised by the defensive spread and decided by its sign.
regime_definition = DecisionDefinition(
    name="macro_regime",
    components=tuple(ComponentSpec(axis, -1.0, 1.0, grid=LOADING_GRID) for axis in MACRO_AXES),
    parse=parse_json_components,
    summary=defensive_spread,
    decide=_decide,
)
