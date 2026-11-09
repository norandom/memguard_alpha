"""Decision definitions and the memorization adjustment rule.

A finance simulation asks a model for a decision, not for prose: a handful of
named numbers (regime loadings, a sentiment tilt) that the simulation turns into
positions. A :class:`DecisionDefinition` states which numbers those are, the
range each may take, how a reply is turned into them and, optionally, how they
collapse into one summary value and one categorical decision.

Validation runs when the definition is built, not when the first reply comes
back. A bad bound or a repeated name found after the ensemble has been paid for
costs every draw; found at construction it costs nothing.

Parsing is split in two on purpose. ``parse`` is the caller's raw extractor and
is stored exactly as given; :meth:`DecisionDefinition.parse_response` is the one
validating entry point and is the only path the guarded-decision API uses. The
extractor can therefore stay a one-liner (usually :func:`parse_json_components`)
while the strictness -- exact names, finite numbers, declared bounds -- lives in
one place and cannot be skipped by a definition that forgot it.

:func:`adjust` is the rule already used by the adopter project
(``guarded_tilt = raw_tilt * (1 - p_memorized)``): every component shrinks by the
same factor, so magnitudes fall as the memorization estimate rises and signs
never flip. Out-of-bounds replies are rejected rather than clipped, because a
silently clipped value is a decision the model never made.
"""

from __future__ import annotations

import json
import math
import numbers
from collections.abc import Callable, Hashable, Mapping
from dataclasses import dataclass

DecisionVector = Mapping[str, float]


class DecisionDefinitionError(ValueError):
    """A decision definition is invalid; raised at construction, before any request."""


class DecisionParseError(ValueError):
    """One response does not parse into the decision definition."""


def _is_real(value: object) -> bool:
    # bool is an int subclass; True as a regime loading is a parser bug, not 1.0.
    return isinstance(value, numbers.Real) and not isinstance(value, bool)


@dataclass(frozen=True)
class ComponentSpec:
    """One named numeric component of a decision, with closed bounds and an optional grid.

    ``grid`` is the step the prompt asks the model to answer on (0.1 for the
    regime loadings). Replies off the grid are accepted; the ensemble reduction
    measures adherence and snaps the located value, as in
    :class:`~recall_guard.core.ensemble.EnsembleSpec`.
    """

    name: str
    lower: float
    upper: float
    grid: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise DecisionDefinitionError(f"component name must be a non-empty string; got {self.name!r}")
        for bound in ("lower", "upper"):
            value = getattr(self, bound)
            if not _is_real(value) or not math.isfinite(value):
                raise DecisionDefinitionError(
                    f"component {self.name!r}: {bound} bound must be a finite number; got {value!r}"
                )
        if not self.lower < self.upper:
            raise DecisionDefinitionError(
                f"component {self.name!r}: lower must be < upper; got lower={self.lower!r}, upper={self.upper!r}"
            )
        if self.grid is not None and not (_is_real(self.grid) and self.grid > 0 and math.isfinite(self.grid)):
            raise DecisionDefinitionError(
                f"component {self.name!r}: grid must be a positive finite number; got {self.grid!r}"
            )


@dataclass(frozen=True)
class DecisionDefinition:
    """What the model is asked for and how its reply becomes numbers.

    ``summary`` maps a component vector to one value (the defensive spread in
    the worked example) and is applied alike to raw and adjusted vectors.
    ``decide`` maps a vector to a hashable category and drives agreement. Both
    must be pure, since replay recomputes them from recorded draws.
    """

    name: str
    components: tuple[ComponentSpec, ...]
    parse: Callable[[str], Mapping[str, float]]
    summary: Callable[[Mapping[str, float]], float] | None = None
    decide: Callable[[Mapping[str, float]], Hashable] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise DecisionDefinitionError(f"definition name must be a non-empty string; got {self.name!r}")
        if not isinstance(self.components, tuple) or not self.components:
            raise DecisionDefinitionError(
                f"definition {self.name!r}: components must be a non-empty tuple of ComponentSpec"
            )
        seen: set[str] = set()
        for spec in self.components:
            if not isinstance(spec, ComponentSpec):
                raise DecisionDefinitionError(
                    f"definition {self.name!r}: every component must be a ComponentSpec; got {spec!r}"
                )
            if spec.name in seen:
                raise DecisionDefinitionError(f"definition {self.name!r}: duplicate component name {spec.name!r}")
            seen.add(spec.name)
        if not callable(self.parse):
            raise DecisionDefinitionError(f"definition {self.name!r}: parse must be callable")
        for hook in ("summary", "decide"):
            value = getattr(self, hook)
            if value is not None and not callable(value):
                raise DecisionDefinitionError(f"definition {self.name!r}: {hook} must be callable or None")

    def parse_response(self, text: str) -> DecisionVector:
        """Parse one reply strictly into the declared components, in declared order.

        Raises :class:`DecisionParseError` naming the problem when the extractor
        fails, a declared component is missing, an undeclared key is present, or
        a value is not a finite, non-boolean number inside its bounds.
        """
        try:
            raw = self.parse(text)
        except DecisionParseError:
            raise
        except (ValueError, TypeError, KeyError) as exc:
            raise DecisionParseError(f"definition {self.name!r}: parse failed: {exc}") from exc
        if not isinstance(raw, Mapping):
            raise DecisionParseError(
                f"definition {self.name!r}: parse must return a mapping; got {type(raw).__name__}"
            )
        declared = [spec.name for spec in self.components]
        missing = [name for name in declared if name not in raw]
        if missing:
            raise DecisionParseError(f"definition {self.name!r}: missing components {missing}")
        declared_names = set(declared)
        extra = sorted(str(key) for key in raw if key not in declared_names)
        if extra:
            raise DecisionParseError(f"definition {self.name!r}: undeclared components {extra}")
        vector: dict[str, float] = {}
        for spec in self.components:
            value = raw[spec.name]
            if not _is_real(value) or not math.isfinite(value):
                raise DecisionParseError(
                    f"definition {self.name!r}: component {spec.name!r} must be a finite number; got {value!r}"
                )
            if not spec.lower <= value <= spec.upper:
                raise DecisionParseError(
                    f"definition {self.name!r}: component {spec.name!r}={value!r} outside "
                    f"[{spec.lower!r}, {spec.upper!r}]"
                )
            vector[spec.name] = float(value)
        return vector


def parse_json_components(text: str) -> dict[str, float]:
    """Strict JSON object of numbers; duplicate keys, non-objects and non-numbers are refused.

    ``json.loads`` keeps the last of two equal keys, which would let a reply that
    states two different loadings for one component pass as either. The pairs
    hook sees every key before the dict collapses them.
    """

    def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        keys = [key for key, _ in pairs]
        duplicates = sorted({key for key in keys if keys.count(key) > 1})
        if duplicates:
            raise DecisionParseError(f"duplicate keys {duplicates}")
        return dict(pairs)

    def no_constant(token: str) -> object:
        raise DecisionParseError(f"non-finite JSON constant {token}")

    if not isinstance(text, str):
        raise DecisionParseError(f"response must be a string; got {type(text).__name__}")
    try:
        parsed = json.loads(text, object_pairs_hook=unique_object, parse_constant=no_constant)
    except DecisionParseError:
        raise
    except ValueError as exc:
        raise DecisionParseError(f"response is not valid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise DecisionParseError(f"response must be a JSON object; got {type(parsed).__name__}")
    components: dict[str, float] = {}
    for key, value in parsed.items():
        if not _is_real(value):
            raise DecisionParseError(f"key {key!r} must be a number; got {value!r}")
        try:
            number = float(value)
        except OverflowError as exc:
            raise DecisionParseError(f"key {key!r} is outside the float range") from exc
        if not math.isfinite(number):
            raise DecisionParseError(f"key {key!r} must be finite; got {value!r}")
        components[key] = number
    return components


def adjust(vector: DecisionVector, p_memorized: float) -> dict[str, float]:
    """Scale every component by ``1 - clip(p_memorized, 0, 1)``; signs never change.

    Floats only: unresolved components (``None``) are the caller's to keep out.
    A non-finite estimate raises ``ValueError`` -- NaN would otherwise clip to
    NaN and erase the decision without saying so.
    """
    if not _is_real(p_memorized) or not math.isfinite(p_memorized):
        raise ValueError(f"p_memorized must be a finite number; got {p_memorized!r}")
    scale = 1.0 - min(max(float(p_memorized), 0.0), 1.0)
    adjusted: dict[str, float] = {}
    for name, value in vector.items():
        if not _is_real(value) or not math.isfinite(value):
            raise ValueError(f"component {name!r} must be a finite number; got {value!r}")
        adjusted[name] = float(value) * scale
    return adjusted
