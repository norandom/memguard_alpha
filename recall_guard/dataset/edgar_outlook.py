"""The EDGAR outlook decision and its prompt (Requirement 7.1).

The negative control asks a model for one number per SEC EDGAR press release:
the business outlook the release conveys, from -1 (clearly negative) to 1
(clearly positive). The EDGAR calibrator is fitted on prompts in this format and
the negative-control task asks for one outlook per held-out release, so both use
:func:`render_outlook_prompt` and :data:`outlook_definition` and cannot drift
apart.

The prompt is a short fixed instruction, the release text exactly as stored in
the ``prompt`` field of the EDGAR corpus (title, a blank line and the body, at
most 1500 characters; see :mod:`recall_guard.dataset.edgar_corpora`) and a JSON
example. The release is embedded unchanged, with no trimming and no
normalisation, so :func:`prompt_sha256` of a prompt depends only on the stored
record and the fixed wording. The instruction names no date, no later result of
the company and nothing else beyond the release, and it asks for no trade and no
return.

``decide`` follows the macro regime definition: the outlook is rounded to nine
decimals before its sign is tested, so a value that is zero up to floating-point
residue is decided as -1, like zero itself. The two definitions therefore treat a
fully discounted answer alike.
"""

from __future__ import annotations

import hashlib
import math
import numbers
from collections.abc import Mapping

from recall_guard.core.decision import ComponentSpec, DecisionDefinition, parse_json_components

#: The one component of the outlook decision.
OUTLOOK = "outlook"
#: The step the outlook is declared on.
OUTLOOK_GRID = 0.1
# Decimals a value keeps before its sign is tested, as in macro_regime.
_SIGN_DECIMALS = 9

#: The fixed instruction before the release text, line by line.
_INSTRUCTION_LINES: tuple[str, ...] = (
    "Rate the business outlook that the press release below conveys as one number "
    "from -1 (clearly negative) to 1 (clearly positive).",
    "Use only the text of the release.",
    "",
    "Press release:",
)
#: The fixed answer format after the release text, line by line.
_ANSWER_LINES: tuple[str, ...] = (
    "",
    'Output a JSON object with the single key "outlook", e.g. {"outlook": 0.2}.',
)


def render_outlook_prompt(release_text: str) -> str:
    """The outlook prompt for one release: the instruction, the release unchanged, the answer format.

    ``release_text`` is the ``prompt`` field of an EDGAR corpus record and is
    embedded exactly as given, so equal records render equal prompts. A
    non-string raises ``TypeError``; an empty or whitespace-only text raises
    ``ValueError``.
    """
    if not isinstance(release_text, str):
        raise TypeError(f"release_text must be a string; got {type(release_text).__name__}")
    if not release_text.strip():
        raise ValueError("release_text must not be empty")
    return "\n".join((*_INSTRUCTION_LINES, release_text, *_ANSWER_LINES))


def prompt_sha256(text: str) -> str:
    """The SHA-256 of the UTF-8 encoding of ``text``, as 64 lowercase hex digits.

    A non-string raises ``TypeError``. A string with no UTF-8 encoding (a lone
    surrogate) raises ``UnicodeEncodeError``, which is a ``ValueError``.
    """
    if not isinstance(text, str):
        raise TypeError(f"text must be a string; got {type(text).__name__}")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _outlook(vector: Mapping[str, float]) -> float:
    """The outlook itself, the summary value; ``vector`` must hold exactly one finite ``outlook``."""
    if not isinstance(vector, Mapping):
        raise TypeError(f"vector must be a mapping; got {type(vector).__name__}")
    if list(vector) != [OUTLOOK]:
        raise ValueError(f"vector must have exactly the key {[OUTLOOK]}; got {sorted(str(key) for key in vector)}")
    value = vector[OUTLOOK]
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(f"{OUTLOOK} must be a finite number; got {value!r}")
    try:
        number = float(value)
    except OverflowError:
        raise ValueError(f"{OUTLOOK} must be a finite number") from None
    if not math.isfinite(number):
        raise ValueError(f"{OUTLOOK} must be a finite number; got {value!r}")
    return number


def _decide(vector: Mapping[str, float]) -> int:
    """1 when ``round(outlook, 9)`` is above zero, otherwise -1 (zero included).

    The rounding removes floating-point residue, as in the macro regime
    definition: an outlook within about 5e-10 of zero, of either sign, is -1.
    """
    return 1 if round(_outlook(vector), _SIGN_DECIMALS) > 0 else -1


#: The negative-control decision (7.1): one outlook in [-1, 1] on a 0.1 grid,
#: summarised by the outlook itself and decided by its sign.
outlook_definition = DecisionDefinition(
    name="edgar_outlook",
    components=(ComponentSpec(OUTLOOK, -1.0, 1.0, grid=OUTLOOK_GRID),),
    parse=parse_json_components,
    summary=_outlook,
    decide=_decide,
)
