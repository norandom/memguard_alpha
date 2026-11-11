"""Selection interface: choose a catalogue entry and mode, run the guard under its status.

:func:`select` resolves an entry of the qualification catalogue
(:mod:`recall_guard.core.catalogue`) and one of its evidence modes into a
:class:`Selection`: the entry, the mode, the outcome of the qualification rule
and the estimate source the guard runs with. ``Mode.LOGPROBS`` is the top-k
source and ``Mode.LOGITS`` the full distribution. A mode the entry does not
offer is refused with the modes it does offer. Without a mode, the entry's
single qualified mode is chosen; otherwise the caller is asked to name one.

:func:`guarded_select` calls
:func:`~recall_guard.harness.guarded_decision.guarded_decide` unchanged and
returns a :class:`SelectedDecision` that states the entry, the mode and the
status the answer ran under. The correction is exposed only when all three
hold:

1. the catalogue status of the entry and mode is ``qualified``;
2. the guard status is ``ok``;
3. the supplied calibrator's fingerprint equals the one the catalogue records
   for that entry and mode.

A qualified record run with another calibrator, or one that records no
fingerprint, says nothing about the calibrator in use: the run is
``not_tested`` and the reason says so. For ``not_qualified`` and
``not_tested`` the answer is returned without the correction and carries the
status and its reason. The recall estimate is always the guard's own value.

The shipped catalogue is read only by a :func:`select` call that passes no
catalogue; importing this module reads no file.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from recall_guard.core.catalogue import (
    Catalogue,
    CatalogueEntry,
    CatalogueError,
    Mode,
    Qualification,
    Status,
    load_catalogue,
    qualify,
)
from recall_guard.core.completion import CompletionClient
from recall_guard.core.decision import DecisionDefinition
from recall_guard.core.full_distribution import FullDistributionClient
from recall_guard.core.profiles import ProfileError, load_profile
from recall_guard.harness.guarded_decision import GuardConfig, GuardedDecision, SourceCalibrator, guarded_decide
from recall_guard.mia.estimate_source import EstimateSource, SourceLabel

_SOURCE_OF_MODE = MappingProxyType(
    {Mode.LOGPROBS: EstimateSource.TOP_K, Mode.LOGITS: EstimateSource.FULL_DISTRIBUTION}
)


class SelectionError(ValueError):
    """The entry is unknown, the mode is not offered, or no single mode is qualified."""


@dataclass(frozen=True)
class Selection:
    """A catalogue entry and mode, the rule's outcome for them, and the guard's estimate source."""

    entry: CatalogueEntry
    mode: Mode
    qualification: Qualification
    source: SourceLabel


@dataclass(frozen=True)
class SelectedDecision:
    """One guarded answer with the catalogue entry, mode and status it ran under.

    ``answer`` is the uncorrected decision and ``p_memorized`` the guard's
    recall estimate, both taken from ``decision`` without change. ``corrected``
    is the guard's adjusted decision, present only when ``status`` is
    ``qualified``, the guard status is ``ok`` and the calibrator is the one the
    catalogue records; otherwise it is None and ``reason`` says why.
    ``decision`` is the guard's result, untouched.
    """

    entry_id: str
    mode: Mode
    status: Status
    reason: str
    answer: Mapping[str, float | None] | None
    p_memorized: float | None
    corrected: Mapping[str, float | None] | None
    decision: GuardedDecision


def _offered(entry: CatalogueEntry) -> str:
    return ", ".join(mode.value for mode in Mode if entry.modes[mode].available) or "no mode"


def _is_k(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 1


def _profile_k(entry: CatalogueEntry, mode: Mode) -> int | None:
    """Alternatives per position that the entry's provider profile declares for ``mode``; None without one."""
    if entry.profile_id is None:
        return None
    try:
        profile = load_profile(entry.profile_id)
    except ProfileError as error:
        raise SelectionError(
            f"Catalogue entry {entry.entry_id!r}: cannot load provider profile {entry.profile_id!r}: {error}"
        ) from None
    if mode is Mode.LOGPROBS:
        value = profile.model_config.parameters.get("top_logprobs")
    else:
        value = None if profile.full_distribution is None else profile.full_distribution.top_k_keep
    return value if _is_k(value) else None


def _source(entry: CatalogueEntry, mode: Mode, k: int | None) -> SourceLabel:
    if k is not None and not _is_k(k):
        raise SelectionError(f"k must be a positive integer or None; got {k!r}")
    declared = _profile_k(entry, mode)
    if declared is None:
        if k is None:
            raise SelectionError(
                f"Catalogue entry {entry.entry_id!r} has no provider profile that states the alternatives kept "
                f"per token in the {mode.value} mode; pass k"
            )
        declared = k
    elif k is not None and k != declared:
        raise SelectionError(
            f"Catalogue entry {entry.entry_id!r}: k={k} differs from the {declared} alternatives per token "
            f"that provider profile {entry.profile_id!r} declares for the {mode.value} mode"
        )
    return SourceLabel(_SOURCE_OF_MODE[mode], declared)


def _default_mode(entry: CatalogueEntry) -> Mode:
    """The entry's qualified mode when exactly one offered mode is qualified (Requirement 2.7)."""
    statuses = {mode: qualify(entry.modes[mode]).status for mode in Mode}
    qualified = [mode for mode in Mode if entry.modes[mode].available and statuses[mode] is Status.QUALIFIED]
    if len(qualified) == 1:
        return qualified[0]
    listing = "; ".join(
        f"{mode.value} is {statuses[mode].value}{'' if entry.modes[mode].available else ' (not offered)'}"
        for mode in Mode
    )
    raise SelectionError(
        f"Catalogue entry {entry.entry_id!r} has no single qualified mode; name one. Modes: {listing}"
    )


def select(
    entry_id: str, mode: Mode | str | None = None, *, catalogue: Catalogue | None = None, k: int | None = None
) -> Selection:
    """Select a catalogue entry and an evidence mode.

    ``catalogue`` None loads the catalogue shipped in the package. ``mode``
    None chooses the entry's qualified mode when exactly one is qualified.
    ``k`` is the number of alternatives per token in the selected mode: it is
    read from the entry's provider profile (``top_logprobs`` for logprobs, the
    full-distribution host's ``top_k_keep`` for logits) and must be passed for
    an entry whose profile does not state it; a value that contradicts the
    profile is refused.

    Raises :class:`SelectionError` for an unknown entry, for a mode the entry
    does not offer (the message names the offered modes) and, with ``mode``
    None, when no single mode is qualified (the message lists each mode with
    its status).
    """
    if catalogue is None:
        catalogue = load_catalogue()
    try:
        entry = catalogue.entry(entry_id)
    except CatalogueError as error:
        raise SelectionError(str(error)) from None
    if mode is None:
        chosen = _default_mode(entry)
    else:
        try:
            chosen = Mode(mode)
        except ValueError:
            raise SelectionError(
                f"Catalogue entry {entry.entry_id!r}: unknown mode {mode!r}; it offers: {_offered(entry)}"
            ) from None
    record = entry.modes[chosen]
    if not record.available:
        why = f" ({record.unavailable_reason})" if record.unavailable_reason else ""
        raise SelectionError(
            f"Catalogue entry {entry.entry_id!r} does not offer the {chosen.value} mode{why}; "
            f"it offers: {_offered(entry)}"
        )
    return Selection(entry=entry, mode=chosen, qualification=qualify(record), source=_source(entry, chosen, k))


def _status(selection: Selection, calibrator: SourceCalibrator) -> tuple[Status, str]:
    """The status a run has with this calibrator, and its reason.

    A qualified record qualifies one calibrator, the one whose fingerprint it
    records; any other status is carried as the catalogue gives it.
    """
    qualification = selection.qualification
    if qualification.status is not Status.QUALIFIED:
        return qualification.status, qualification.reason
    recorded = selection.entry.modes[selection.mode].calibrator_fingerprint
    if recorded is None:
        return Status.NOT_TESTED, (
            "the catalogue records no calibrator fingerprint for this entry and mode, so the supplied "
            "calibrator cannot be identified as the qualified one"
        )
    if calibrator.fingerprint != recorded:
        return Status.NOT_TESTED, (
            f"calibrator fingerprint mismatch: the supplied calibrator is {calibrator.fingerprint}, "
            f"the catalogue qualified {recorded}"
        )
    return Status.QUALIFIED, qualification.reason


def guarded_select(
    prompt: str,
    selection: Selection,
    definition: DecisionDefinition,
    *,
    client: CompletionClient,
    calibrator: SourceCalibrator,
    full_distribution: FullDistributionClient | None = None,
    draws: int = 32,
    min_parsed: int = 24,
) -> SelectedDecision:
    """Run the guard in the selected mode and mark the answer with the entry, mode and status.

    The guard is :func:`~recall_guard.harness.guarded_decision.guarded_decide`
    with the selection's estimate source; its result is returned untouched as
    ``decision``, and ``answer`` and ``p_memorized`` are its values. The
    correction is exposed as ``corrected`` only for a qualified status, a guard
    status of ``ok`` and the calibrator the catalogue records (see the module
    docstring); a calibrator the catalogue does not record makes the run
    ``not_tested``.
    """
    if not isinstance(selection, Selection):
        raise TypeError(f"selection must be a Selection, as select() returns it; got {type(selection).__name__}")
    config = GuardConfig(definition, selection.source, draws=draws, min_parsed=min_parsed)
    decision = guarded_decide(prompt, config, client=client, calibrator=calibrator,
                              full_distribution=full_distribution)
    status, reason = _status(selection, calibrator)
    corrected = None
    if status is Status.QUALIFIED:
        if decision.status == "ok":
            corrected = decision.adjusted
        else:
            reason = f"{reason}; no correction: the guard status is {decision.status}"
    return SelectedDecision(
        entry_id=selection.entry.entry_id,
        mode=selection.mode,
        status=status,
        reason=reason,
        answer=decision.decision,
        p_memorized=decision.p_memorized,
        corrected=corrected,
        decision=decision,
    )
