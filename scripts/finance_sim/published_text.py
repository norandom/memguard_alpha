"""The wording of a reason in a published table, note or caption (Requirements 7.5, 7.8 and 6.9).

The private analyses state why a value is missing in the study's working
vocabulary: its arms and renderings, the names of its estimate sources, private
paths, identifiers and the numbers of the requirements. A published table is
read without the study at hand, so every reason is published through ONE step,
:func:`published_reason`: it maps each stored reason this module knows to a
plain sentence (the serving route as data, "logprobs" and "logits" for the two
evidence modes, "identifying prompt" and "anonymised prompt", counts kept as
numbers) and refuses, with :class:`~scripts.finance_sim.figure_style.FigureError`,
a reason it does not recognise. A refusal is deliberate: an unknown reason is
never published as stored and never dropped.

:func:`check_published_text` is the vocabulary rule itself. It refuses a text
that names a requirement or a task, holds a path separator, a backtick, an
identifier with an underscore, an em dash or the study's arm vocabulary. Every
sentence :func:`published_reason` returns has passed it.

A stored reason may be a composition: a prefix with an inner reason ("no
consensus: ..."), or several reasons joined with "; ". The parts are translated
one by one; a composition with one unknown part is refused as a whole.

The private files keep their stored text; this module changes nothing there.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from types import MappingProxyType

from scripts.finance_sim.figure_style import FigureError

__all__ = ["FORBIDDEN_PUBLISHED", "check_published_text", "published_reason"]

#: What a published reason must not contain, as (what it is, pattern).
FORBIDDEN_PUBLISHED: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("a requirement number", re.compile(r"requirement", re.IGNORECASE)),
    ("a task number", re.compile(r"\btasks? \d", re.IGNORECASE)),
    ("a path separator", re.compile(r"[/\\]")),
    ("a backtick", re.compile(r"`")),
    ("an identifier with an underscore", re.compile(r"[A-Za-z0-9]_[A-Za-z0-9]")),
    ("an em dash", re.compile("—")),
    ("the study's arm vocabulary", re.compile(
        r"\barms?\b|\bprimary\b|\bprobe\b|\brenderings?\b|\btop[-_ ]k\b|\bfull[-_ ]distribution\b"
        r"|\bsampling\b|\bserving\b(?! routes?\b)", re.IGNORECASE)),
)

#: The published word of a stored name that appears inside a reason.
_WORDS: Mapping[str, str] = MappingProxyType({
    "top_k": "logprobs", "full_distribution": "logits", "pre": "before", "post": "after",
    "macro": "macro prompts", "edgar": "EDGAR releases",
    "holdout_auc_logprobs": "the held-out AUC of logprobs", "holdout_auc_logits": "the held-out AUC of logits",
    "paired_auc_difference": "the paired AUC difference",
})
_INT = r"\d+"
_NUMBER = r"[-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?"
_MODE = r"logprobs|logits"
_SOURCE = r"top_k|full_distribution"
_NAME = r"[A-Za-z][A-Za-z0-9_]*"
_PARSE_ERROR, _REQUEST_ERROR = "parse_error", "request_error:"
#: Group names whose value is itself a stored reason and is translated before it is placed.
_INNER = ("inner",)


def check_published_text(text: str, what: str = "a published reason") -> str:
    """``text`` unchanged when it holds nothing a published reason may not hold; refused otherwise."""
    for name, pattern in FORBIDDEN_PUBLISHED:
        found = pattern.search(text)
        if found is not None:
            raise FigureError(f"{what} holds {name} ({found.group()!r}) and is not published: {text!r}")
    return text


def _pattern(template: str, **groups: str) -> re.Pattern[str]:
    """``template`` as a pattern: its text literally, each ``{name}`` a group (digits unless ``groups`` says)."""
    parts = re.split(r"\{(\w+)\}", template)
    return re.compile("".join(re.escape(part) if index % 2 == 0 else f"(?P<{part}>{groups.get(part, _INT)})"
                              for index, part in enumerate(parts)))


def _failure(code: str, *, first: bool = False) -> str:
    """Why one draw has no parsed reply, from its failure code."""
    which = "first " if first else ""
    if code == _PARSE_ERROR:
        return f"the {which}reply could not be read"
    if code.startswith(_REQUEST_ERROR):
        return f"the {which}request failed"
    raise FigureError(f"no published wording for the failure code {code!r}")


def _too_few_parsed(match: re.Match[str]) -> str:
    """The guard's reason for a failed ensemble: the parsed count, the minimum and the failures by kind."""
    unread = failed = 0
    for entry in filter(None, (part.strip() for part in match["detail"].split(","))):
        code, _, count = entry.rpartition("=")
        if not count.isdigit():
            raise FigureError(f"no published wording for the failed draws {match['detail']!r}")
        _failure(code)
        if code == _PARSE_ERROR:
            unread += int(count)
        else:
            failed += int(count)
    parts = [f"{match['parsed']} of {match['total']} replies could be read, fewer than the {match['minimum']} needed"]
    if unread:
        parts.append(f"{unread} {'reply' if unread == 1 else 'replies'} could not be read")
    if failed:
        parts.append(f"{failed} {'request' if failed == 1 else 'requests'} failed")
    return "; ".join(parts)


def _not_computed(match: re.Match[str]) -> str:
    """The comparison's reason when a calibrator statistic has no value: the statistics by name, then the inputs."""
    names = [name.strip() for name in match["names"].split(",")]
    unknown = [name for name in names if name not in _WORDS]
    if unknown:
        raise FigureError(f"no published wording for the statistic {unknown[0]!r}")
    listed = [_WORDS[name] for name in names]
    subject = listed[0] if len(listed) == 1 else f"{', '.join(listed[:-1])} and {listed[-1]}"
    return f"{subject} cannot be computed; missing: {_translate(match['inner'])}"


def _template(published: str) -> Callable[[re.Match[str]], str]:
    def build(match: re.Match[str]) -> str:
        values = {name: _translate(value) if name in _INNER else _WORDS.get(value, value)
                  for name, value in match.groupdict().items() if value is not None}
        return published.format(**values)

    return build


def _rule(stored: str, published: str | None = None, **groups: str
          ) -> tuple[re.Pattern[str], Callable[[re.Match[str]], str]]:
    """One known reason: the stored text with ``{name}`` holes and its published sentence (the same when None)."""
    return _pattern(stored, **groups), _template(stored if published is None else published)


_FAVOURS = "the interval {lo} to {hi} of the paired AUC difference excludes 0 in favour of logits"
_LIMITS = ("the limits of agreement of the prompt estimates, {a} to {b}, checked on the outer bounds of their "
           "intervals (interval low of the lower limit {c}, interval high of the upper limit {d})")
_NUMBERS = {name: _NUMBER for name in ("lo", "hi", "a", "b", "c", "d", "m", "m2")}
_WEAK = ("The calibrator is weak (held-out discrimination below its threshold), so the adjustment was skipped and "
         "the decision is passed through unadjusted")
_WEAK_PUBLISHED = ("the calibrator is weak (its held-out AUC is below the minimum), so the adjustment was skipped "
                   "and the decision is left unadjusted")

#: Every stored reason this module publishes, in the order they are tried. A reason matches as a whole.
_RULES: tuple[tuple[re.Pattern[str], Callable[[re.Match[str]], str]], ...] = (
    # ----- the frozen plan: an evidence mode a route cannot have -----
    _rule("missing for the serving arm: the full-distribution host serves the local GGUF, which are different "
          "weights from the served model, and values from different builds are not comparable (Requirement 9.4)",
          "logits are not available on this serving route: they would come from other weights than the served "
          "model, and values from different weights are not comparable"),
    # ----- the accuracy gap -----
    _rule("no planned month carries the value in both renderings",
          "no planned month has the value under both the identifying and the anonymised prompt"),
    _rule("no {side}-cutoff month carries the value in both renderings, so the {again}-cutoff cell is empty",
          "no month {side} the training cutoff has the value under both the identifying and the anonymised prompt",
          side="pre|post", again="pre|post"),
    _rule("{n} months carry the value in both renderings, fewer than the frozen block length of {block} months, so "
          "the moving-block bootstrap is not defined",
          "{n} months have the value under both the identifying and the anonymised prompt, fewer than the block "
          "length of {block} months, so the moving-block bootstrap is not defined"),
    (re.compile(r"the interval cannot be computed: .+"), lambda match: "the interval cannot be computed"),
    # ----- the calibrators -----
    _rule("insufficient: {memorized} memorized and {control} control held-out draws; the local report's held-out "
          "calibration needs at least {minimum} per class",
          "too few held-out draws: {memorized} memorized and {control} control, and at least {minimum} are needed "
          "in each class"),
    _rule("the calibrator carries no retained held-out pairs", "the calibrator kept no held-out draws"),
    _rule("no calibrator file at {where}/{source}/{kind}.json under the calibrators directory: the calibration has "
          "not been fitted", "no {source} calibrator has been fitted on {kind} for this serving route",
          where=_NAME, source=_SOURCE, kind="macro|edgar"),
    # ----- one draw, one ensemble -----
    (_pattern("draw did not parse: {code}", code=r"\S+"), lambda match: _failure(match["code"])),
    (_pattern("the first draw did not parse: {code}", code=r"\S+"), lambda match: _failure(match["code"], first=True)),
    _rule("no estimate for this draw", "this draw has no recall estimate"),
    (_pattern("{parsed} of {total} draws parsed, below min_parsed={minimum}; failed draws:{detail}", detail=r".*"),
     _too_few_parsed),
    _rule("No parsed draw carries a memorization estimate (estimates on 0 of {n} parsed draws), so the decision is "
          "passed through unadjusted",
          "none of the {n} replies that could be read has a recall estimate, so the decision is left unadjusted"),
    _rule(f"{_WEAK}. The estimate uses {{used}} of {{parsed}} parsed draws; the other {{rest}} carry no estimate",
          f"{_WEAK_PUBLISHED}; the estimate uses {{used}} of {{parsed}} replies that could be read, the other "
          "{rest} have no estimate"),
    _rule(_WEAK, _WEAK_PUBLISHED),
    _rule("a component is unresolved (split into separated clusters across draws), so the decision has no summary "
          "value", "the draws split into separate groups on one component, so the ensemble decision has no summary "
                   "value"),
    _rule("no realized next-month returns for this month", "no realized next-month return is available for this "
                                                           "month"),
    _rule("the stored {variant} variant carries no {value} (status {status})",
          "the stored ensemble result has no value here", variant=_NAME, value=_NAME, status=_NAME),
    _rule("the consensus has no category (a component is unresolved) and the first draw lies inside the band, so "
          "whether it is off cannot be decided",
          "the ensemble has no category because the draws split on one component, and the first draw lies inside "
          "the band, so whether it differs cannot be decided"),
    _rule("the consensus has no category: a component is unresolved (split into separated clusters across draws)",
          "the ensemble has no category: the draws split into separate groups on one component"),
    _rule("no stored result: the stage has not executed this item and rendering",
          "no result is stored for this prompt"),
    _rule("no item carries the values this statistic needs (each missing item is listed with its reason)",
          "no prompt has the values this statistic needs"),
    _rule("the arm's ensemble has an odd number of draws, so it has no two disjoint halves of equal size",
          "the ensemble has an odd number of draws, so it cannot be split into two halves of equal size"),
    _rule("the two single draws agree on every item, so the single-draw variance is zero and the variance ratio is "
          "undefined", "the two single draws agree on every prompt, so the single-draw variance is zero and the "
                       "variance ratio is undefined"),
    # ----- the convergence analysis -----
    _rule("k {k} is above the {kept} alternatives the evidence keeps", "k {k} is above the {kept} alternatives kept"),
    _rule("the reply is not in the draw recording", "the reply was not recorded"),
    _rule("the full-distribution estimate is unavailable for this draw",
          "the recall estimate from logits is unavailable for this draw"),
    _rule("the estimate is unavailable: the calibrator cannot use these features",
          "the recall estimate is unavailable: the calibrator cannot use these values"),
    _rule("the full-distribution evidence {reference} is not recorded", "the logits of this draw were not recorded",
          reference=r"\S+"),
    _rule("no decision: fewer parsed draws than the frozen minimum (the stored result gives the counts)",
          "no ensemble decision: fewer replies could be read than the minimum"),
    _rule("a parsed draw lacks its reply or full-distribution evidence (listed in missing_draws), so the decision at "
          "this k is not compared",
          "a draw lacks its reply or its logits, so the decision at this k is not compared"),
    _rule("the {where} arm records no full-distribution evidence", "this serving route records no logits",
          where=_NAME),
    _rule("no convergence table at {name} under the analysis directory: the analysis has not been run",
          "the convergence analysis has not been run for these prompts", name=r"\S+"),
    # ----- the comparison of logprobs with logits -----
    _rule("the calibrator record of the {mode} mode ({source})", "the {mode} calibrator", mode=_MODE, source=_SOURCE),
    _rule("the held-out labels and probabilities of the {mode} calibrator", mode=_MODE),
    _rule("calibration draws that give the held-out rows of the {mode} calibrator: the calibration draws do not "
          "give the held-out rows ({usable} usable draws on its {prompts} held-out prompts, {rows} held-out rows in "
          "the record)",
          "calibration draws that match the held-out draws of the {mode} calibrator ({usable} usable draws on its "
          "{prompts} held-out prompts, {rows} held-out draws recorded)", mode=_MODE),
    _rule("the calibration draws: the {rows} held-out rows of the {mode} calibrator are not {each} draws on each of "
          "its {prompts} held-out prompts, so the rows of a prompt cannot be told apart from the record alone",
          "the calibration draws: the {rows} held-out draws of the {mode} calibrator are not {each} draws on each "
          "of its {prompts} held-out prompts, so the draws of one prompt cannot be told apart", mode=_MODE),
    _rule("held-out rows of the {mode} calibrator in the order of its held-out prompts: the calibration draws do "
          "not give the held-out rows (the held-out labels are not those of the held-out prompts in order)",
          "held-out draws of the {mode} calibrator in the order of its held-out prompts", mode=_MODE),
    _rule("held-out prompts of both classes"),
    _rule("held-out rows on the same prompts and draws in both modes: the logprobs calibrator holds out {a} prompts "
          "with {b} rows, the logits calibrator {c} prompts with {d} rows, and they are not the same draws",
          "held-out draws on the same prompts in both modes: the logprobs calibrator holds out {a} prompts with "
          "{b} draws, the logits calibrator {c} prompts with {d} draws, and they are not the same draws"),
    _rule("the stored item results of the run", "the stored results of the run"),
    _rule("stored item results with a draw that carries both estimates",
          "stored results with a draw that has both recall estimates"),
    _rule("stored item results with an ensemble decision that carries a recall estimate and applied the adjustment "
          "in both modes (of {stored} stored items, {none} have no ensemble decision in a mode, {bare} an ensemble "
          "decision without an estimate and {skipped} an ensemble decision that skipped the adjustment)",
          "stored results with an ensemble decision that has a recall estimate and applied the adjustment in both "
          "modes (of {stored} stored prompts, {none} have no ensemble decision in a mode, {bare} an ensemble "
          "decision without an estimate and {skipped} an ensemble decision that skipped the adjustment)"),
    _rule("at least two clusters", "at least two months"),
    _rule("the {mode} calibrator carries no weights digest, so the two modes are not shown to be measured on the "
          "same weights; their values are not compared",
          "the {mode} calibrator carries no weights identity, so the two modes are not shown to be measured on the "
          "same weights; their values are not compared", mode=_MODE),
    _rule("the two modes were measured on different weights (logprobs {first}, logits {second}); values compare "
          "only within one set of weights", first=r"[A-Za-z0-9:]+", second=r"[A-Za-z0-9:]+"),
    (_pattern("{names} cannot be computed; missing: {inner}", names=r"[a-z_]+(?:, [a-z_]+)*", inner=r".+"),
     _not_computed),
    _rule(f"{_FAVOURS}, but improve is concluded on the financial time-series boundary (macro) only",
          f"{_FAVOURS}, but improve is concluded on financial time-series prompts only", **_NUMBERS),
    _rule(f"{_FAVOURS} on the macro boundary and the cutoff check passed for logits",
          f"{_FAVOURS} on financial time-series prompts and the cutoff check passed for logits", **_NUMBERS),
    _rule(f"{_FAVOURS}, but the cutoff check for logits was not supplied, so improve cannot be concluded", **_NUMBERS),
    _rule(f"{_FAVOURS}, but the cutoff check for logits did not pass, so improve cannot be concluded", **_NUMBERS),
    _rule("it is not within {m} either side of 0, so match is not concluded either", **_NUMBERS),
    _rule("the interval {lo} to {hi} of the paired AUC difference is not within {m} either side of 0 and does not "
          "exclude 0 in favour of logits", **_NUMBERS),
    _rule("the limits of agreement cannot be computed; missing: {inner}", inner=r".+"),
    _rule("the intervals of the limits of agreement cannot be computed (no month resample gave a defined limit), so "
          "the limits cannot be checked on their interval bounds"),
    _rule(f"{_LIMITS} are not within {{m}} either side of 0", **_NUMBERS),
    _rule(f"the interval {{lo}} to {{hi}} of the paired AUC difference is within {{m}} either side of 0 and "
          f"{_LIMITS} are within {{m2}} either side of 0", **_NUMBERS),
    # ----- the cutoff check -----
    _rule("the {where} arm records no {source} estimate, so no month carries one",
          "this serving route records no {source} recall estimate, so no month has one", where=_NAME, source=_SOURCE),
    _rule("months with a {source} estimate under the {kind} prompt: {inside} inside the training range and {after} "
          "after the cutoff; at least {minimum} are needed in each group",
          "months with a {source} recall estimate under the {kind} prompt: {inside} inside the training range and "
          "{after} after the cutoff; at least {minimum} are needed in each group",
          source=_SOURCE, kind="identifying|anonymised"),
    # ----- the repeat baseline -----
    _rule("recorded draws of the arm's full-run stage: none are recorded",
          "no draw of the full run is recorded for this serving route"),
    _rule("prompts with a fully recorded ensemble and at least {minimum} parsed draws: the recorded draws hold none",
          "no prompt has a fully recorded ensemble with at least {minimum} replies that could be read"),
    # ----- compositions: a prefix with an inner reason, then a reason with its count -----
    _rule("no consensus: {inner}", "no ensemble decision: {inner}", inner=r".+"),
    _rule("no decision: {inner}", "no ensemble decision: {inner}", inner=r".+"),
    _rule("no estimate: {inner}", "no recall estimate: {inner}", inner=r".+"),
    _rule("missing: {inner}", "not in the half-ensemble statistic: {inner}", inner=r".+"),
    _rule("first half (draws {start} to {stop}): {inner}", inner=r".+"),
    _rule("second half (draws {start} to {stop}): {inner}", inner=r".+"),
    _rule("single draw {index}: {inner}", inner=r".+"),
    _rule("{kind}: {inner}", "{kind} prompt: {inner}", kind="identifying|anonymised", inner=r".+"),
    (_pattern("{inner}: {count}", inner=r".+"),
     lambda match: f"{_translate(match['inner'])} ({match['count']} {'draw' if match['count'] == '1' else 'draws'})"),
)


def _translate(stored: str) -> str:
    """The published sentence of one stored reason, a prefixed one or several joined with "; "; else refused."""
    text = " ".join(stored.split()).rstrip(".")
    for pattern, build in _RULES:
        match = pattern.fullmatch(text)
        if match is not None:
            try:
                return build(match)
            except FigureError:
                continue  # a composition whose inner part is unknown: another rule or a split may still fit
    for cut in (found.start() for found in re.finditer("; ", text)):
        try:
            return f"{_translate(text[:cut])}; {_translate(text[cut + 2:])}"
        except FigureError:
            continue
    raise FigureError(f"no published wording for the stored reason {stored!r}; add it to the published reasons "
                      "before it is published")


def published_reason(stored: object) -> str:
    """The plain sentence a published table, note or caption gives for the stored reason ``stored``.

    An empty reason stays empty. The sentence has no closing full stop, so a
    caption can place it inside one of its own. Counts are kept as numbers.

    Raises:
        FigureError: a reason this module does not recognise, or a sentence that still holds what a published
            reason may not hold (:data:`FORBIDDEN_PUBLISHED`).
    """
    if stored is None:
        return ""
    text = " ".join(str(stored).split())
    if not text:
        return ""
    return check_published_text(_translate(text))
