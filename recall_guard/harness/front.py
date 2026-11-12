"""The front: run a caller's prompt for a catalogue entry and return the best answer available.

:func:`ask` takes any prompt, a decision definition
(:class:`~recall_guard.core.decision.DecisionDefinition`) and the id of an
entry of the qualification catalogue, and returns one :class:`FrontAnswer`:
the answer, the recall estimate, the corrected answer where one applies, and
the entry, mode, status and reason the answer ran under. The prompt is the
caller's; it is sent as given.

The front adds no estimate and no rule of its own. Its back is the selection
interface (:mod:`recall_guard.harness.selection`):

* With a calibrator for the mode in use, :func:`ask` calls
  :func:`~recall_guard.harness.selection.select` and
  :func:`~recall_guard.harness.selection.guarded_select` unchanged. The status,
  the reason, the answer, the recall estimate and the correction are that
  result's values, and the result itself is returned as ``selected``.
* Without one, the same ensemble is drawn and reduced by the guard itself
  (:func:`~recall_guard.harness.guarded_decision.decide_without_estimate`),
  with no estimate on any draw. The answer comes back with the catalogue status
  of the mode, without a recall estimate and without a correction.
* With a calibrator fitted for another model, estimate source or settings than
  the client's, the guard refuses the calibrator before it sends a request.
  The answer is then drawn once, as without a calibrator, and ``limits`` says
  that the supplied calibrator was fitted for another model or settings.

An entry whose provider profile does not state the alternatives per token
needs ``k`` for a run with a calibrator; see :func:`ask`.

When the caller names no mode, :func:`choose_mode` picks one among the modes
the entry offers, in this order, and ``mode_choice`` states the rule that
applied: a qualified mode; otherwise a mode the caller supplied a calibrator
for; otherwise logprobs; otherwise logits. Logprobs is preferred when a rule
leaves both modes.

A status never refuses an answer: a not qualified or not tested entry and mode
still answers, uncorrected and marked. ``limits`` states what limited the
answer, in fixed phrases (the ``LIMIT_*`` constants); it is empty when nothing
did. An answer that rests on fewer draws than were asked for, because a
request failed or a reply could not be read, says so there, and
``draws_asked`` and ``draws_used`` give the counts.

When no answer can be produced, the :class:`FrontAnswer` carries a
:class:`FrontFailure` with the step that failed (``entry``, ``mode``,
``client``, ``request`` or ``reply``) and what the caller can change, and no
answer, estimate or correction. Arguments of the wrong type raise.

For offline tests and demos the module ships a client that replays given
replies (:class:`FixedReplies`, with :func:`fixed_reply` for replies that carry
token evidence), the matching full-distribution host for the logits mode
(:class:`FixedEvidence`), and one worked example (:func:`worked_example`): the
fenced point-in-time macro prompt with its decision definition, and an invented
reply to it (:data:`WORKED_EXAMPLE_REPLY`). None of them opens a connection.

Importing this module reads no file. The shipped catalogue is read only by an
:func:`ask` call that passes no catalogue, and a provider profile only by a
call that passes no client.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from types import MappingProxyType
from typing import TYPE_CHECKING

from recall_guard.core.catalogue import Catalogue, CatalogueEntry, CatalogueError, Mode, Status, load_catalogue
from recall_guard.core.completion import (
    CompletionClient,
    ModelConfig,
    ModelResponse,
    ParameterValue,
    TokenAlternative,
    TokenEvidence,
)
from recall_guard.core.decision import DecisionDefinition
from recall_guard.core.full_distribution import FullDistributionEvidence
from recall_guard.core.profiles import ProfileError, connect, load_profile
from recall_guard.core.request_control import BudgetExhaustedError, CallBudget, RequestError
from recall_guard.dataset.macro_regime import MacroRow, regime_definition, render_anonymised
from recall_guard.harness.guarded_decision import (
    REQUEST_ERROR_PREFIX,
    GuardConfig,
    SourceCalibrator,
    decide_without_estimate,
    full_distribution_client,
)
from recall_guard.harness.scorer_client import CalibrationMismatchError
from recall_guard.harness.selection import SelectedDecision, SelectionError, guarded_select, select
from recall_guard.mia.estimate_source import EstimateSource, SourceLabel

if TYPE_CHECKING:
    from recall_guard.core.full_distribution import FullDistributionClient

#: The steps a front call can fail at, in the order they are taken.
FAILURE_STEPS = ("entry", "mode", "client", "request", "reply")

#: ``mode_choice`` phrases: which rule gave the mode.
CHOICE_NAMED = "named by the caller"
CHOICE_QUALIFIED = "the entry's qualified mode"
CHOICE_QUALIFIED_TIE = "both modes are qualified; logprobs preferred"
CHOICE_CALIBRATED = "no qualified mode; the mode a calibrator was supplied for"
CHOICE_CALIBRATED_TIE = "no qualified mode; calibrators supplied for both modes; logprobs preferred"
CHOICE_LOGPROBS = "no qualified mode and no calibrator; logprobs is offered"
CHOICE_LOGITS = "no qualified mode and no calibrator; logprobs is not offered, so logits"
CHOICE_NONE = "no mode was chosen"

#: ``limits`` phrases: what limited an answer.
LIMIT_NO_QUALIFIED_MODE = "no qualified mode for the entry"
LIMIT_NAMED_MODE_NOT_QUALIFIED = "the named mode is not qualified"
LIMIT_NO_CALIBRATOR = "no calibrator supplied"
LIMIT_CALIBRATOR_NOT_RECORDED = "the supplied calibrator is not the recorded one"
LIMIT_CALIBRATOR_MISMATCH = "the supplied calibrator was fitted for another model or settings"
LIMIT_ESTIMATE_UNAVAILABLE = "the recall estimate could not be computed"
LIMIT_WEAK_CALIBRATOR = "the adjustment was skipped for a weak calibrator"
LIMIT_DRAWS_MISSING = "some draws failed or could not be read"

#: Every ``mode_choice`` phrase, in the order of the rules; a result's ``mode_choice`` is one of them.
MODE_CHOICES = (
    CHOICE_NAMED,
    CHOICE_QUALIFIED,
    CHOICE_QUALIFIED_TIE,
    CHOICE_CALIBRATED,
    CHOICE_CALIBRATED_TIE,
    CHOICE_LOGPROBS,
    CHOICE_LOGITS,
    CHOICE_NONE,
)

#: Every ``limits`` phrase, in a fixed order; each item of a result's ``limits`` is one of them.
LIMITS = (
    LIMIT_NO_QUALIFIED_MODE,
    LIMIT_NAMED_MODE_NOT_QUALIFIED,
    LIMIT_NO_CALIBRATOR,
    LIMIT_CALIBRATOR_NOT_RECORDED,
    LIMIT_CALIBRATOR_MISMATCH,
    LIMIT_ESTIMATE_UNAVAILABLE,
    LIMIT_WEAK_CALIBRATOR,
    LIMIT_DRAWS_MISSING,
)

_MODE_OF_SOURCE = MappingProxyType(
    {EstimateSource.TOP_K: Mode.LOGPROBS, EstimateSource.FULL_DISTRIBUTION: Mode.LOGITS}
)

_REMEDY_REQUEST = (
    "check that the model host is running and reachable at the endpoint in use, that the credential the "
    "profile names is set and accepted, and that the request budget covers the draws; then ask again"
)
_REMEDY_REPLY = (
    "make the prompt ask for exactly the components of the decision definition in the form its parse reads, "
    "or change the definition, or lower min_parsed"
)
_REMEDY_K_MISSING = (
    "pass k to ask: the number of alternatives per token the evidence carries (top_logprobs of the requests in "
    "the logprobs mode, the alternatives the full-distribution host keeps in the logits mode); it is needed "
    "when the entry's provider profile does not state it. Or pass no calibrator to get the answer without a "
    "recall estimate"
)
_REMEDY_K_GIVEN = (
    "leave k out when the entry's provider profile states the alternatives per token, or pass the number the "
    "detail names; k is a positive integer"
)
_REMEDY_HOST_BEARER = (
    "set {name} to the full-distribution host's bearer in the environment that host's client reads (for the "
    "client the front builds: the environment passed to ask, or else the process environment), or pass "
    "full_distribution= with a client whose credential variable is set"
)
#: The text the generation clients refuse a set but malformed credential with (recall_guard.core.providers).
_MALFORMED_CREDENTIAL = "Referenced credential contains whitespace, control or non-ASCII characters"


@dataclass(frozen=True)
class FrontFailure:
    """Why a front call produced no answer: the step that failed and what the caller can change."""

    step: str  # "entry" | "mode" | "client" | "request" | "reply"
    detail: str
    remedy: str

    def __post_init__(self) -> None:
        if self.step not in FAILURE_STEPS:
            raise ValueError(f"step must be one of {', '.join(FAILURE_STEPS)}; got {self.step!r}")


@dataclass(frozen=True)
class FrontAnswer:
    """One answer of the front, or its failure.

    ``answer`` is the uncorrected decision. ``p_memorized`` is the guard's
    recall estimate, None when no calibrator was supplied for the mode or the
    guard could compute none. ``corrected`` is present only for a qualified
    status (see :class:`~recall_guard.harness.selection.SelectedDecision`).
    ``mode_choice`` says why this mode was used and ``limits`` what limited the
    answer. ``selected`` is the selection interface's result, untouched; it is
    None when no calibrator was used.

    ``draws_asked`` is the size of the ensemble the answer was asked with and
    ``draws_used`` the number of draws it rests on: those whose request
    returned a reply that read into the definition. When ``draws_used`` is
    smaller, ``limits`` holds :data:`LIMIT_DRAWS_MISSING`.

    With ``failure`` set there is no answer, estimate, correction or
    ``selected`` and no draw count; ``mode`` and ``status`` are what was known
    when the step failed.
    """

    entry_id: str
    mode: Mode | None
    mode_choice: str
    status: Status | None
    reason: str
    answer: Mapping[str, float | None] | None
    p_memorized: float | None
    corrected: Mapping[str, float | None] | None
    limits: tuple[str, ...]
    failure: FrontFailure | None
    selected: SelectedDecision | None
    draws_asked: int | None = None
    draws_used: int | None = None


def _listing(entry: CatalogueEntry, catalogue: Catalogue) -> str:
    """Each mode of the entry with its status, and whether it is offered."""
    return "; ".join(
        f"{mode.value} is {catalogue.qualification(entry.entry_id, mode).status.value}"
        f"{'' if entry.modes[mode].available else ' (not offered)'}"
        for mode in Mode
    )


def _qualified_modes(entry: CatalogueEntry, catalogue: Catalogue) -> list[Mode]:
    return [mode for mode in Mode if entry.modes[mode].available
            and catalogue.qualification(entry.entry_id, mode).status is Status.QUALIFIED]


def choose_mode(
    entry: CatalogueEntry, *, catalogue: Catalogue, calibrators: Mapping[Mode, SourceCalibrator]
) -> tuple[Mode, str]:
    """The mode the front uses for ``entry`` when the caller names none, and the rule that gave it.

    Among the modes the entry offers: a qualified mode; otherwise a mode with
    a calibrator in ``calibrators``; otherwise logprobs; otherwise logits.
    When a rule leaves both modes, logprobs is used and the phrase says so.
    The second value is one of the ``CHOICE_*`` phrases.

    Raises :class:`~recall_guard.harness.selection.SelectionError` when the
    entry offers no mode.
    """
    offered = [mode for mode in Mode if entry.modes[mode].available]
    if not offered:
        raise SelectionError(
            f"Catalogue entry {entry.entry_id!r} offers no mode. Modes: {_listing(entry, catalogue)}"
        )
    for candidates, one, tie in (
        (_qualified_modes(entry, catalogue), CHOICE_QUALIFIED, CHOICE_QUALIFIED_TIE),
        ([mode for mode in offered if mode in calibrators], CHOICE_CALIBRATED, CHOICE_CALIBRATED_TIE),
    ):
        if len(candidates) > 1:
            return Mode.LOGPROBS, tie
        if candidates:
            return candidates[0], one
    if Mode.LOGPROBS in offered:
        return Mode.LOGPROBS, CHOICE_LOGPROBS
    return Mode.LOGITS, CHOICE_LOGITS


def _calibrators(
    calibrator: SourceCalibrator | Mapping[Mode, SourceCalibrator] | None,
) -> dict[Mode, SourceCalibrator]:
    """The supplied calibrators by mode; one calibrator serves the mode of its estimate source."""
    if calibrator is None:
        return {}
    if isinstance(calibrator, SourceCalibrator):
        return {_MODE_OF_SOURCE[EstimateSource(calibrator.identity.estimate_source)]: calibrator}
    if not isinstance(calibrator, Mapping):
        raise TypeError(
            "calibrator must be a SourceCalibrator, a mapping from Mode to SourceCalibrator, or None; "
            f"got {type(calibrator).__name__}"
        )
    by_mode: dict[Mode, SourceCalibrator] = {}
    for key, value in calibrator.items():
        if not isinstance(key, str):
            raise TypeError(f"calibrator keys must be modes; got {key!r}")
        if not isinstance(value, SourceCalibrator):
            raise TypeError(f"calibrator for mode {Mode(key).value} must be a SourceCalibrator; "
                            f"got {type(value).__name__}")
        by_mode[Mode(key)] = value
    return by_mode


def ask(
    prompt: str,
    definition: DecisionDefinition,
    *,
    entry: str,
    mode: Mode | None = None,
    client: CompletionClient | None = None,
    calibrator: SourceCalibrator | Mapping[Mode, SourceCalibrator] | None = None,
    full_distribution: FullDistributionClient | None = None,
    catalogue: Catalogue | None = None,
    k: int | None = None,
    draws: int = 32,
    min_parsed: int = 24,
    environment: Mapping[str, str] | None = None,
    temperature: float | None = None,
) -> FrontAnswer:
    """Run ``prompt`` for the catalogue entry ``entry`` and return the best answer available.

    ``mode`` None lets :func:`choose_mode` pick the mode. ``calibrator`` is one
    :class:`~recall_guard.harness.guarded_decision.SourceCalibrator`, used for
    the mode of the estimate source it was fitted for, or a mapping from mode
    to calibrator; the recall estimate and the correction need the one for the
    mode in use. ``client`` None builds the client from the entry's provider
    profile through :func:`~recall_guard.core.profiles.connect`, and a
    calibrated logits run without ``full_distribution`` builds that host's
    client from the same profile; endpoint and credential references are read
    by name from ``environment`` (default: the process environment). A
    calibrated logits run also asks the full-distribution client it will use,
    passed or built, before any request, whether its bearer is usable
    (``credential_problem()``, which reads the client's own environment). A
    passed host object without that method is not checked before the run.
    ``catalogue`` None loads the shipped catalogue. ``draws``,
    ``min_parsed`` and ``temperature`` are the guard's. ``temperature`` is the
    sampling temperature of every draw; None uses the client's own. A
    calibrator is used only when the draws are made at the temperature it was
    fitted at, so pass that temperature with it.

    ``k`` is the number of alternatives per token the evidence carries in the
    mode in use, as :func:`~recall_guard.harness.selection.select` defines it:
    a run with a calibrator needs it for an entry whose provider profile does
    not state it (an entry without a profile, or a profile without
    ``top_logprobs``), and a value that contradicts the profile is refused.
    A run without a calibrator for the mode in use reads no evidence and does
    not look at ``k``.

    A calibrator fitted for another model, estimate source or settings than
    the client's is not used: the guard refuses it before any request, and the
    answer is then drawn once, without a recall estimate, and says so in
    ``limits``.

    Never refuses for a status. Returns a failure (see :class:`FrontFailure`)
    for an unknown entry, an entry or a named mode that is not offered, a
    missing or contradicting ``k``, a client that cannot be built, a
    generation credential that is not a usable bearer, a full-distribution
    client whose bearer is unset, empty or not a usable bearer, a model that
    cannot be reached, and replies that do not read into the definition.
    Raises for arguments of the wrong type.
    """
    if not isinstance(prompt, str):
        raise TypeError("prompt must be a string")
    if not isinstance(entry, str):
        raise TypeError("entry must be a catalogue entry id")
    if mode is not None and not isinstance(mode, str):
        raise TypeError(f"mode must be a Mode or None; got {mode!r}")
    if catalogue is not None and not isinstance(catalogue, Catalogue):
        raise TypeError("catalogue must be a Catalogue, as load_catalogue() returns it, or None")
    if environment is not None and not isinstance(environment, Mapping):
        raise TypeError("environment must be a mapping or None")
    calibrators = _calibrators(calibrator)
    # The guard's own validation of the definition and the ensemble size, before anything is read or sent.
    GuardConfig(definition, SourceLabel(EstimateSource.TOP_K, None), draws=draws, min_parsed=min_parsed,
                temperature=temperature)
    if catalogue is None:
        catalogue = load_catalogue()

    chosen: Mode | None = None
    choice = CHOICE_NONE if mode is None else CHOICE_NAMED
    status: Status | None = None
    reason = ""

    def failed(step: str, detail: str, remedy: str) -> FrontAnswer:
        return FrontAnswer(entry_id=entry, mode=chosen, mode_choice=choice, status=status, reason=reason,
                           answer=None, p_memorized=None, corrected=None, limits=(),
                           failure=FrontFailure(step, detail, remedy), selected=None)

    try:
        record = catalogue.entry(entry)
    except CatalogueError as error:
        known = ", ".join(item.entry_id for item in catalogue.entries) or "none"
        return failed("entry", str(error), f"pass the id of a catalogue entry: {known}")

    modes = _listing(record, catalogue)
    if mode is None:
        try:
            chosen, choice = choose_mode(record, catalogue=catalogue, calibrators=calibrators)
        except SelectionError as error:
            return failed("mode", str(error), f"use an entry that offers a mode; this entry's modes: {modes}")
    else:
        remedy = f"name a mode the entry offers, or name none and the front chooses; this entry's modes: {modes}"
        try:
            chosen = Mode(mode)
        except ValueError:
            return failed("mode", f"{mode!r} is not a mode; the modes are logprobs and logits", remedy)
        if not record.modes[chosen].available:
            why = record.modes[chosen].unavailable_reason
            return failed("mode", f"catalogue entry {entry!r} does not offer the {chosen.value} mode"
                                  f"{f': {why}' if why else ''}", remedy)
    qualification = catalogue.qualification(entry, chosen)
    status, reason = qualification.status, qualification.reason

    used = calibrators.get(chosen)
    selection = None
    if used is not None:
        try:
            selection = select(entry, chosen, catalogue=catalogue, k=k)
        except SelectionError as error:
            return failed("mode", str(error), _REMEDY_K_MISSING if k is None else _REMEDY_K_GIVEN)

    calibrated_logits = used is not None and chosen is Mode.LOGITS
    needs_host = calibrated_logits and full_distribution is None
    profile = None
    if client is None or needs_host:
        argument = "client=" if client is None else "full_distribution="
        if record.profile_id is None:
            return failed("client", f"catalogue entry {entry!r} has no provider profile to reach the model through",
                          f"pass {argument} for this entry")
        try:
            profile = load_profile(record.profile_id)
            if client is None:
                # Room for every draw with the retries the profile allows.
                budget = CallBudget(draws * (profile.model_config.max_retries + 1))
                try:
                    client = connect(profile, budget, environment=environment)
                except ValueError as error:
                    # Only the client constructor's refusal of a set but malformed credential, recognised by
                    # its fixed text, which holds no value; a ProfileError goes to the handler below.
                    if isinstance(error, ProfileError) or str(error) != _MALFORMED_CREDENTIAL:
                        raise
                    name = (profile.model_config.credential_ref or "").removeprefix("env:")
                    return failed("client", f"the credential {name} of provider profile {record.profile_id!r} "
                                            "contains whitespace, control or non-ASCII characters",
                                  f"set {name} to the bare key, without surrounding or embedded whitespace, or "
                                  "pass client=")
                argument = "full_distribution="
            if needs_host:
                full_distribution = full_distribution_client(profile, environment=environment)
        except ProfileError as error:
            return failed("client", str(error),
                          f"set what provider profile {record.profile_id!r} needs (the detail names it) and start "
                          f"its model host, or pass {argument}")
    if calibrated_logits:
        # The host's client is asked here, before the first generation request: it reads its bearer only with
        # its own first request, when the ensemble has already been drawn. It reads its own environment, which
        # for a passed client need not be the front's. A host object without the check is left unchecked.
        check = getattr(full_distribution, "credential_problem", None)
        problem = check() if callable(check) else None
        if problem is not None:
            name = full_distribution.credential_ref.removeprefix("env:")
            return failed("client", f"the full-distribution host's {problem}", _REMEDY_HOST_BEARER.format(name=name))

    selected: SelectedDecision | None = None
    mismatched = False
    try:
        if used is not None:
            try:
                selected = guarded_select(prompt, selection, definition, client=client, calibrator=used,
                                          full_distribution=full_distribution, draws=draws, min_parsed=min_parsed,
                                          temperature=temperature)
            except CalibrationMismatchError:
                # The guard compares the calibrator's identity with the client's before it sends anything, so
                # no request has been made; the ensemble below is the only one drawn.
                mismatched = True
        if selected is None:
            plain = decide_without_estimate(prompt, definition, client=client, draws=draws, min_parsed=min_parsed,
                                            temperature=temperature)
            answer, why, replies, parsed = plain.decision, plain.reason, plain.n_replies, plain.n_parsed
        else:
            decision = selected.decision
            answer = None if decision.status == "failed" else decision.decision
            why = decision.reason
            replies = sum(1 for draw in decision.draws
                          if not (draw.failure_code or "").startswith(REQUEST_ERROR_PREFIX))
            parsed = sum(1 for draw in decision.draws if draw.parsed is not None)
    except (RequestError, BudgetExhaustedError) as error:
        return failed("request", f"{type(error).__name__}: {error}", _REMEDY_REQUEST)
    if answer is None:
        # Too few replies arrived for min_parsed to be reachable, or enough arrived and too few parsed.
        step = "request" if replies < min_parsed else "reply"
        return failed(step, why or "the guard reported failure",
                      _REMEDY_REQUEST if step == "request" else _REMEDY_REPLY)

    limits: list[str] = []
    if qualification.status is not Status.QUALIFIED:
        limits.append(LIMIT_NAMED_MODE_NOT_QUALIFIED if _qualified_modes(record, catalogue)
                      else LIMIT_NO_QUALIFIED_MODE)
    # A draw whose request failed or whose reply did not read into the definition is missing from the answer.
    missing = [LIMIT_DRAWS_MISSING] if parsed < draws else []
    if selected is None:
        limits.append(LIMIT_CALIBRATOR_MISMATCH if mismatched else LIMIT_NO_CALIBRATOR)
        return FrontAnswer(entry_id=entry, mode=chosen, mode_choice=choice, status=status, reason=reason,
                           answer=answer, p_memorized=None, corrected=None, limits=(*limits, *missing),
                           failure=None, selected=None, draws_asked=draws, draws_used=parsed)
    decision = selected.decision
    if qualification.status is Status.QUALIFIED and selected.status is not Status.QUALIFIED:
        limits.append(LIMIT_CALIBRATOR_NOT_RECORDED)
    if decision.status == "estimate_unavailable":
        limits.append(LIMIT_ESTIMATE_UNAVAILABLE)
    elif decision.status == "adjustment_skipped_weak_calibrator":
        limits.append(LIMIT_WEAK_CALIBRATOR)
    return FrontAnswer(entry_id=selected.entry_id, mode=selected.mode, mode_choice=choice, status=selected.status,
                       reason=selected.reason, answer=selected.answer, p_memorized=selected.p_memorized,
                       corrected=selected.corrected, limits=(*limits, *missing), failure=None, selected=selected,
                       draws_asked=draws, draws_used=parsed)


# -- offline use: fixed replies, fixed evidence and the worked example ---------------


@dataclass(frozen=True)
class FixedReply:
    """One reply for :class:`FixedReplies`: the text and what a response may carry with it.

    ``token_evidence`` is the per-token evidence in the shape a real response
    carries it (:class:`~recall_guard.core.completion.TokenEvidence`, one per
    output token, each with its top alternatives); None means the reply carries
    none, and a run with a top-k calibrator then has no estimate for that draw.
    :func:`fixed_reply` builds a record from plain pairs.
    """

    text: str
    token_evidence: tuple[TokenEvidence, ...] | None = None
    finish_reason: str | None = None
    usage: Mapping[str, int] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.text, str):
            raise TypeError("text must be a string")
        # ModelResponse validates and freezes the fields; a wrong reply is refused here, not on its request.
        checked = ModelResponse(self.text, self.token_evidence, self.usage, self.finish_reason)
        object.__setattr__(self, "token_evidence", checked.token_evidence)
        object.__setattr__(self, "usage", checked.usage)


def fixed_reply(
    text: str,
    tokens: Sequence[tuple[str, Mapping[str, float] | Sequence[tuple[str, float]]]],
    *,
    finish_reason: str | None = None,
    usage: Mapping[str, int] | None = None,
) -> FixedReply:
    """A :class:`FixedReply` from a text and, per output token, the token and its top alternatives.

    Each item of ``tokens`` is ``(token, alternatives)``; ``alternatives`` maps
    an alternative token to its natural-log probability, or lists
    ``(token, logprob)`` pairs. The realized token must be among its
    alternatives and takes its logprob from there; the alternatives are stored
    by descending logprob. For a realized token outside its alternatives, build
    the :class:`~recall_guard.core.completion.TokenEvidence` values directly.

    Examples
    --------
    >>> reply = fixed_reply('{"a": 1}', [
    ...     ('{"a"', {'{"a"': -0.1, '{"b"': -2.5}),
    ...     (': 1}', [(': 1}', -0.3), (': 0}', -1.4)]),
    ... ])
    >>> [(token.token, token.logprob) for token in reply.token_evidence]
    [('{"a"', -0.1), (': 1}', -0.3)]
    """
    evidence = []
    for token, alternatives in tokens:
        pairs = list(alternatives.items() if isinstance(alternatives, Mapping) else alternatives)
        ranked = tuple(TokenAlternative(name, logprob)
                       for name, logprob in sorted(pairs, key=lambda pair: pair[1], reverse=True))
        realized = [alternative.logprob for alternative in ranked if alternative.token == token]
        if not realized:
            raise ValueError(f"token {token!r} is not among its alternatives; build TokenEvidence directly for a "
                             "realized token outside the top alternatives")
        evidence.append(TokenEvidence(token, realized[0], ranked))
    return FixedReply(text, tuple(evidence), finish_reason, usage)


class _Fixed:
    """Given items handed out in order, one or several per request, optionally repeating."""

    def __init__(self, items: tuple, *, cycle: bool, what: str, code: str) -> None:
        if not isinstance(cycle, bool):
            raise TypeError("cycle must be a bool")
        if cycle and not items:
            raise ValueError(f"cycle=True needs at least one {what}")
        self._items = items
        self._cycle = cycle
        self._what = what
        self._code = code
        self._served = 0
        self._lock = threading.Lock()

    def take(self, count: int, note: Callable[[], None]) -> tuple:
        """The next ``count`` items, or a RequestError that serves none; ``note`` records the request."""
        with self._lock:
            given = len(self._items)
            if not self._cycle and self._served + count > given:
                raise RequestError(
                    f"The fixed {self._what} are used up: {given} were given and {self._served} served, and this "
                    f"request asks for {count} more. Give more, or pass cycle=True to repeat them.",
                    code=self._code,
                )
            taken = tuple(self._items[(self._served + offset) % given] for offset in range(count))
            self._served += count
            note()
            return taken

    @property
    def served(self) -> int:
        return self._served


class FixedReplies(CompletionClient):
    """A client that replays given replies in order, one per request, for offline tests and demos.

    It opens no connection and reads no file and no environment variable.
    ``replies`` holds plain texts (returned without token evidence) and
    :class:`FixedReply` records (returned with their token evidence, finish
    reason and usage). ``config`` is the configuration the client stands for:
    the guard compares its protocol, endpoint, model and parameters with a
    calibrator's before it uses that calibrator, so pass the configuration the
    calibrator was fitted for. The default names no endpoint (protocol
    ``callback``, model ``fixed-replies``, no parameters).

    A request after the last reply raises
    :class:`~recall_guard.core.request_control.RequestError` with code
    ``fixed_replies_exhausted``, as a host that stops answering would; the
    guard records it on that draw. ``cycle=True`` repeats the replies instead,
    so one reply serves a whole ensemble.

    ``requests`` is the number of requests served and ``prompts`` the prompts
    of those requests, in order.
    """

    def __init__(self, replies: Sequence[str | FixedReply], *, config: ModelConfig | None = None,
                 cycle: bool = False) -> None:
        if isinstance(replies, str | bytes) or not isinstance(replies, Sequence):
            raise TypeError("replies must be a sequence of texts or FixedReply records")
        if config is None:
            config = ModelConfig(protocol="callback", endpoint=None, model="fixed-replies")
        elif not isinstance(config, ModelConfig):
            raise TypeError("config must be a ModelConfig or None")
        records = []
        for reply in replies:
            if not isinstance(reply, str | FixedReply):
                raise TypeError(f"replies must hold texts or FixedReply records; got {type(reply).__name__}")
            records.append(FixedReply(reply) if isinstance(reply, str) else reply)
        self.config = config
        self._fixed = _Fixed(tuple(records), cycle=cycle, what="replies", code="fixed_replies_exhausted")
        self._prompts: list[str] = []

    @property
    def requests(self) -> int:
        """How many requests were served."""
        return self._fixed.served

    @property
    def prompts(self) -> tuple[str, ...]:
        """The prompts of the served requests, in order, as they arrived."""
        return tuple(self._prompts)

    def generate(self, prompt: str, **parameters: ParameterValue) -> ModelResponse:
        """The next reply. ``parameters`` are reported as requested settings and change nothing else."""
        if not isinstance(prompt, str):
            raise TypeError("prompt must be a string")
        (reply,) = self._fixed.take(1, lambda: self._prompts.append(prompt))
        return ModelResponse(reply.text, reply.token_evidence, reply.usage, reply.finish_reason,
                             requested_settings={**self.config.parameters, **parameters})


class FixedEvidence:
    """A full-distribution host that returns given evidence in order, for the logits mode offline.

    The companion of :class:`FixedReplies`: pass it to :func:`ask` as
    ``full_distribution``. It has the interface the guard and the front use of
    :class:`~recall_guard.core.full_distribution.FullDistributionClient`
    (``evidence``, ``evidence_batch`` and ``credential_problem``), opens no
    connection and needs no bearer. The guard asks for evidence once per
    parsed reply, so give one
    :class:`~recall_guard.core.full_distribution.FullDistributionEvidence` per
    reply that parses, or pass ``cycle=True`` to repeat what is given. The
    evidence is returned as given; ``top_k_keep`` and ``timeout_s`` are
    accepted and not applied.

    A request after the last evidence raises
    :class:`~recall_guard.core.request_control.RequestError` with code
    ``fixed_evidence_exhausted``; the guard then records that draw without an
    estimate. ``requests`` is the number of evidence objects served and
    ``calls`` the ``(prompt_text, response_text)`` pairs they were served for.
    """

    def __init__(self, evidence: Sequence[FullDistributionEvidence], *, cycle: bool = False) -> None:
        if isinstance(evidence, str | bytes) or not isinstance(evidence, Sequence):
            raise TypeError("evidence must be a sequence of FullDistributionEvidence")
        items = tuple(evidence)
        if not all(isinstance(item, FullDistributionEvidence) for item in items):
            raise TypeError("evidence must hold FullDistributionEvidence values")
        self._fixed = _Fixed(items, cycle=cycle, what="evidence objects", code="fixed_evidence_exhausted")
        self._calls: list[tuple[str, str]] = []

    @property
    def requests(self) -> int:
        """How many evidence objects were served."""
        return self._fixed.served

    @property
    def calls(self) -> tuple[tuple[str, str], ...]:
        """The ``(prompt_text, response_text)`` pair of each evidence object served, in order."""
        return tuple(self._calls)

    def credential_problem(self) -> None:
        """None: the fixed host needs no bearer."""
        return None

    def evidence(self, prompt_text: str, response_text: str, *, top_k_keep: int = 100) -> FullDistributionEvidence:
        """The next evidence object."""
        return self.evidence_batch(prompt_text, [response_text], top_k_keep=top_k_keep)[0]

    def evidence_batch(self, prompt_text: str, response_texts: Sequence[str], *, top_k_keep: int = 100,
                       timeout_s: float | None = None) -> tuple[FullDistributionEvidence, ...]:
        """The next evidence objects, one per response, in order; a batch that cannot be served whole serves none."""
        if not isinstance(prompt_text, str):
            raise TypeError("prompt_text must be a string")
        if not isinstance(response_texts, list | tuple) or not all(isinstance(text, str) for text in response_texts):
            raise TypeError("response_texts must be a list or tuple of strings")
        return self._fixed.take(len(response_texts),
                                lambda: self._calls.extend((prompt_text, text) for text in response_texts))


#: The macro state of the worked example. Synthetic: the values are invented to be plausible z-scores and
#: levels, taken from no month and from no data series; the date is a placeholder that no prompt of the
#: worked example shows.
WORKED_EXAMPLE_ROW = MacroRow(
    decision_date=date.min,
    raw={"baa10y": 2.47, "cpi_yoy": 3.61, "t10y2y": 0.38},
    zscores={"baa10y_z": 0.85, "cpi_yoy_z": 1.2, "t10y2y_z": -0.65},
    revised={},
)

#: A reply to the worked example's prompt that reads into its definition. It is an invented reply, written
#: by hand for offline tests and demos; it is not a model output and says nothing about any model.
WORKED_EXAMPLE_REPLY = (
    '{"inflation": 0.4, "growth": -0.2, "credit_stress": 0.3, "policy": 0.1, "risk_appetite": -0.3}'
)


def worked_example() -> tuple[str, DecisionDefinition]:
    """The worked example: the fenced point-in-time macro prompt and its decision definition.

    The prompt is :func:`~recall_guard.dataset.macro_regime.render_anonymised`
    of :data:`WORKED_EXAMPLE_ROW`: the macro state as rolling z-scores and four
    assets known by letter and category, with no date, no ticker and no raw
    level, asking for the regime as loadings on five macro axes and for no
    return forecast. The row is synthetic: its values are invented, not those
    of any month. The definition is
    :data:`~recall_guard.dataset.macro_regime.regime_definition`, the one the
    study uses: five loadings in [-1, 1], decided by the sign of the defensive
    spread. :data:`WORKED_EXAMPLE_REPLY` is an invented reply that reads into it.
    """
    return render_anonymised(WORKED_EXAMPLE_ROW), regime_definition


__all__ = [
    "CHOICE_CALIBRATED",
    "CHOICE_CALIBRATED_TIE",
    "CHOICE_LOGITS",
    "CHOICE_LOGPROBS",
    "CHOICE_NAMED",
    "CHOICE_NONE",
    "CHOICE_QUALIFIED",
    "CHOICE_QUALIFIED_TIE",
    "FAILURE_STEPS",
    "LIMIT_CALIBRATOR_MISMATCH",
    "LIMIT_CALIBRATOR_NOT_RECORDED",
    "LIMIT_DRAWS_MISSING",
    "LIMIT_ESTIMATE_UNAVAILABLE",
    "LIMIT_NAMED_MODE_NOT_QUALIFIED",
    "LIMIT_NO_CALIBRATOR",
    "LIMIT_NO_QUALIFIED_MODE",
    "LIMIT_WEAK_CALIBRATOR",
    "LIMITS",
    "MODE_CHOICES",
    "WORKED_EXAMPLE_REPLY",
    "WORKED_EXAMPLE_ROW",
    "FixedEvidence",
    "FixedReplies",
    "FixedReply",
    "FrontAnswer",
    "FrontFailure",
    "ask",
    "choose_mode",
    "fixed_reply",
    "worked_example",
]
