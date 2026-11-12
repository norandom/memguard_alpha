"""The one translation step of published reasons (Requirements 7.5, 7.8 and 6.9).

Every stored reason the analyses and the figure modules can hand to a published
table is listed here with the sentence that is published for it: the literals
are taken from the modules that write them, the formatted ones are built the
way those modules build them, so a change of a stored reason fails here before
it reaches a published file.
"""

from __future__ import annotations

import re

import pytest

from scripts.finance_sim import convergence, ensemble_metrics, execute, repeat_baseline
from scripts.finance_sim.figure_style import FigureError
from scripts.finance_sim.plan import SERVING_FULL_DISTRIBUTION_REASON
from scripts.finance_sim.published_text import FORBIDDEN_PUBLISHED, check_published_text, published_reason

NO_LOGITS = ("logits are not available on this serving route: they would come from other weights than the served "
             "model, and values from different weights are not comparable")
TOO_FEW = "23 of 32 replies could be read, fewer than the 24 needed; 9 replies could not be read"
UNRESOLVED = "the draws split into separate groups on one component, so the ensemble decision has no summary value"
FAVOURS = "the interval 0.1403 to 0.4106 of the paired AUC difference excludes 0 in favour of logits"
LIMITS = ("the limits of agreement of the prompt estimates, -0.7763 to 0.6693, checked on the outer bounds of their "
          "intervals (interval low of the lower limit -0.9, interval high of the upper limit 0.81)")
WEAK = ("The calibrator is weak (held-out discrimination below its threshold), so the adjustment was skipped and the "
        "decision is passed through unadjusted.")
WEAK_PUBLISHED = ("the calibrator is weak (its held-out AUC is below the minimum), so the adjustment was skipped and "
                  "the decision is left unadjusted")

#: (stored reason, published sentence), by the module that writes the stored one.
REASONS: tuple[tuple[str, str], ...] = (
    # plan.py: the frozen reason of the variants a route cannot compute
    (SERVING_FULL_DISTRIBUTION_REASON, NO_LOGITS),
    # analysis.py: the accuracy gap
    ("no planned month carries the value in both renderings",
     "no planned month has the value under both the identifying and the anonymised prompt"),
    ("no pre-cutoff month carries the value in both renderings, so the pre-cutoff cell is empty",
     "no month before the training cutoff has the value under both the identifying and the anonymised prompt"),
    ("no post-cutoff month carries the value in both renderings, so the post-cutoff cell is empty",
     "no month after the training cutoff has the value under both the identifying and the anonymised prompt"),
    ("5 months carry the value in both renderings, fewer than the frozen block length of 6 months, so the "
     "moving-block bootstrap is not defined",
     "5 months have the value under both the identifying and the anonymised prompt, fewer than the block length of "
     "6 months, so the moving-block bootstrap is not defined"),
    ("the interval cannot be computed: every resample was dropped", "the interval cannot be computed"),
    # figures.py: the calibrators of the reliability figure
    ("insufficient: 3 memorized and 12 control held-out draws; the local report's held-out calibration needs at "
     "least 10 per class",
     "too few held-out draws: 3 memorized and 12 control, and at least 10 are needed in each class"),
    ("the calibrator carries no retained held-out pairs", "the calibrator kept no held-out draws"),
    ("no calibrator file at serving/top_k/macro.json under the calibrators directory: the calibration has not been "
     "fitted", "no logprobs calibrator has been fitted on macro prompts for this serving route"),
    ("no calibrator file at sampling/full_distribution/edgar.json under the calibrators directory: the calibration "
     "has not been fitted", "no logits calibrator has been fitted on EDGAR releases for this serving route"),
    # figures_draws.py, ensemble_metrics.py, execute.py and the guard: one draw, one ensemble
    ("draw did not parse: parse_error", "the reply could not be read"),
    ("draw did not parse: request_error:interrupted", "the request failed"),
    (f"{ensemble_metrics.FIRST_DRAW_UNPARSED}: parse_error", "the first reply could not be read"),
    (f"{ensemble_metrics.FIRST_DRAW_UNPARSED}: request_error:TimeoutError", "the first request failed"),
    ("no estimate for this draw", "this draw has no recall estimate"),
    ("23 of 32 draws parsed, below min_parsed=24; failed draws: parse_error=9.", TOO_FEW),
    ("5 of 8 draws parsed, below min_parsed=6; failed draws: parse_error=2, request_error:interrupted=1.",
     "5 of 8 replies could be read, fewer than the 6 needed; 2 replies could not be read; 1 request failed"),
    ("no consensus: 23 of 32 draws parsed, below min_parsed=24; failed draws: parse_error=9.",
     f"no ensemble decision: {TOO_FEW}"),
    ("no decision: 23 of 32 draws parsed, below min_parsed=24; failed draws: parse_error=9.",
     f"no ensemble decision: {TOO_FEW}"),
    ("no estimate: No parsed draw carries a memorization estimate (estimates on 0 of 7 parsed draws), so the "
     "decision is passed through unadjusted.",
     "no recall estimate: none of the 7 replies that could be read has a recall estimate, so the decision is left "
     "unadjusted"),
    (WEAK, WEAK_PUBLISHED),
    (f"{WEAK} The estimate uses 5 of 7 parsed draws; the other 2 carry no estimate.",
     f"{WEAK_PUBLISHED}; the estimate uses 5 of 7 replies that could be read, the other 2 have no estimate"),
    (execute.UNRESOLVED_SUMMARY, UNRESOLVED),
    (execute.REALIZED_MISSING, "no realized next-month return is available for this month"),
    ("the stored ensemble_top_k variant carries no p_memorized (status ok)",
     "the stored ensemble result has no value here"),
    (ensemble_metrics.UNDECIDED_FIRST_DRAW,
     "the ensemble has no category because the draws split on one component, and the first draw lies inside the "
     "band, so whether it differs cannot be decided"),
    (ensemble_metrics.NO_CATEGORY,
     "the ensemble has no category: the draws split into separate groups on one component"),
    (ensemble_metrics.NO_RESULT, "no result is stored for this prompt"),
    (ensemble_metrics.NO_ITEMS, "no prompt has the values this statistic needs"),
    (ensemble_metrics.ODD_ENSEMBLE,
     "the ensemble has an odd number of draws, so it cannot be split into two halves of equal size"),
    (ensemble_metrics.ZERO_SINGLE_VARIANCE,
     "the two single draws agree on every prompt, so the single-draw variance is zero and the variance ratio is "
     "undefined"),
    (f"missing: {execute.UNRESOLVED_SUMMARY}", f"not in the half-ensemble statistic: {UNRESOLVED}"),
    ("missing: first half (draws 0 to 3): no decision: 2 of 4 draws parsed, below min_parsed=3; failed draws: "
     f"parse_error=2.; single draw 4: {execute.UNRESOLVED_SUMMARY}",
     "not in the half-ensemble statistic: first half (draws 0 to 3): no ensemble decision: 2 of 4 replies could be "
     f"read, fewer than the 3 needed; 2 replies could not be read; single draw 4: {UNRESOLVED}"),
    (f"second half (draws 4 to 7): {execute.UNRESOLVED_SUMMARY}", f"second half (draws 4 to 7): {UNRESOLVED}"),
    # convergence.py and figures_convergence.py
    ("k 100 is above the 20 alternatives the evidence keeps", "k 100 is above the 20 alternatives kept"),
    (f"{convergence.NO_REPLY}: 3", "the reply was not recorded (3 draws)"),
    (f"{convergence.FULL_ESTIMATE_UNAVAILABLE}: 2; {convergence.ESTIMATE_UNAVAILABLE}: 1",
     "the recall estimate from logits is unavailable for this draw (2 draws); the recall estimate is unavailable: "
     "the calibrator cannot use these values (1 draw)"),
    (f"the full-distribution evidence {'e' * 64} is not recorded: 4",
     "the logits of this draw were not recorded (4 draws)"),
    (convergence.NO_RESULT, "no result is stored for this prompt"),
    (convergence.NO_DECISION, "no ensemble decision: fewer replies could be read than the minimum"),
    (convergence.DRAWS_LACKING, "a draw lacks its reply or its logits, so the decision at this k is not compared"),
    ("the serving arm records no full-distribution evidence", "this serving route records no logits"),
    ("no convergence table at convergence-sampling-full_run.json under the analysis directory: the analysis has not "
     "been run", "the convergence analysis has not been run for these prompts"),
    # mode_comparison.py: a missing statistic, a refusal and the reason of the answer
    ("the calibrator record of the logits mode (full_distribution)", "the logits calibrator"),
    ("the calibrator record of the logprobs mode (top_k)", "the logprobs calibrator"),
    ("the held-out labels and probabilities of the logprobs calibrator",
     "the held-out labels and probabilities of the logprobs calibrator"),
    ("calibration draws that give the held-out rows of the logits calibrator: the calibration draws do not give "
     "the held-out rows (20 usable draws on its 11 held-out prompts, 22 held-out rows in the record)",
     "calibration draws that match the held-out draws of the logits calibrator (20 usable draws on its 11 held-out "
     "prompts, 22 held-out draws recorded)"),
    ("the calibration draws: the 21 held-out rows of the logprobs calibrator are not 2 draws on each of its 11 "
     "held-out prompts, so the rows of a prompt cannot be told apart from the record alone",
     "the calibration draws: the 21 held-out draws of the logprobs calibrator are not 2 draws on each of its 11 "
     "held-out prompts, so the draws of one prompt cannot be told apart"),
    ("held-out rows of the logits calibrator in the order of its held-out prompts: the calibration draws do not "
     "give the held-out rows (the held-out labels are not those of the held-out prompts in order)",
     "held-out draws of the logits calibrator in the order of its held-out prompts"),
    ("held-out prompts of both classes", "held-out prompts of both classes"),
    ("held-out rows on the same prompts and draws in both modes: the logprobs calibrator holds out 11 prompts with "
     "22 rows, the logits calibrator 12 prompts with 24 rows, and they are not the same draws",
     "held-out draws on the same prompts in both modes: the logprobs calibrator holds out 11 prompts with 22 draws, "
     "the logits calibrator 12 prompts with 24 draws, and they are not the same draws"),
    ("the stored item results of the run", "the stored results of the run"),
    ("stored item results with a draw that carries both estimates",
     "stored results with a draw that has both recall estimates"),
    ("stored item results with an ensemble decision that carries a recall estimate and applied the adjustment in "
     "both modes (of 40 stored items, 3 have no ensemble decision in a mode, 2 an ensemble decision without an "
     "estimate and 35 an ensemble decision that skipped the adjustment)",
     "stored results with an ensemble decision that has a recall estimate and applied the adjustment in both modes "
     "(of 40 stored prompts, 3 have no ensemble decision in a mode, 2 an ensemble decision without an estimate and "
     "35 an ensemble decision that skipped the adjustment)"),
    ("at least two clusters", "at least two months"),
    ("the logits calibrator carries no weights digest, so the two modes are not shown to be measured on the same "
     "weights; their values are not compared",
     "the logits calibrator carries no weights identity, so the two modes are not shown to be measured on the same "
     "weights; their values are not compared"),
    ("the two modes were measured on different weights (logprobs sha256:aa11, logits sha256:bb22); values compare "
     "only within one set of weights",
     "the two modes were measured on different weights (logprobs sha256:aa11, logits sha256:bb22); values compare "
     "only within one set of weights"),
    ("holdout_auc_logits, paired_auc_difference cannot be computed; missing: the calibrator record of the logits "
     "mode (full_distribution)",
     "the held-out AUC of logits and the paired AUC difference cannot be computed; missing: the logits calibrator"),
    ("paired_auc_difference cannot be computed; missing: held-out prompts of both classes",
     "the paired AUC difference cannot be computed; missing: held-out prompts of both classes"),
    (f"{FAVOURS}, but improve is concluded on the financial time-series boundary (macro) only",
     f"{FAVOURS}, but improve is concluded on financial time-series prompts only"),
    (f"{FAVOURS} on the macro boundary and the cutoff check passed for logits",
     f"{FAVOURS} on financial time-series prompts and the cutoff check passed for logits"),
    (f"{FAVOURS}, but the cutoff check for logits was not supplied, so improve cannot be concluded; it is not "
     "within 0.05 either side of 0, so match is not concluded either",
     f"{FAVOURS}, but the cutoff check for logits was not supplied, so improve cannot be concluded; it is not "
     "within 0.05 either side of 0, so match is not concluded either"),
    (f"{FAVOURS}, but the cutoff check for logits did not pass, so improve cannot be concluded; it is not within "
     "0.05 either side of 0, so match is not concluded either",
     f"{FAVOURS}, but the cutoff check for logits did not pass, so improve cannot be concluded; it is not within "
     "0.05 either side of 0, so match is not concluded either"),
    ("the interval -0.2 to 0.31 of the paired AUC difference is not within 0.05 either side of 0 and does not "
     "exclude 0 in favour of logits",
     "the interval -0.2 to 0.31 of the paired AUC difference is not within 0.05 either side of 0 and does not "
     "exclude 0 in favour of logits"),
    ("the limits of agreement cannot be computed; missing: the stored item results of the run",
     "the limits of agreement cannot be computed; missing: the stored results of the run"),
    ("the intervals of the limits of agreement cannot be computed (no month resample gave a defined limit), so the "
     "limits cannot be checked on their interval bounds",
     "the intervals of the limits of agreement cannot be computed (no month resample gave a defined limit), so the "
     "limits cannot be checked on their interval bounds"),
    (f"{LIMITS} are not within 0.1 either side of 0", f"{LIMITS} are not within 0.1 either side of 0"),
    (f"the interval -0.01 to 0.02 of the paired AUC difference is within 0.05 either side of 0 and {LIMITS} are "
     "within 0.1 either side of 0",
     f"the interval -0.01 to 0.02 of the paired AUC difference is within 0.05 either side of 0 and {LIMITS} are "
     "within 0.1 either side of 0"),
    # cutoff_check.py
    ("the serving arm records no full_distribution estimate, so no month carries one",
     "this serving route records no logits recall estimate, so no month has one"),
    ("months with a top_k estimate under the identifying prompt: 12 inside the training range and 1 after the "
     "cutoff; at least 2 are needed in each group",
     "months with a logprobs recall estimate under the identifying prompt: 12 inside the training range and 1 after "
     "the cutoff; at least 2 are needed in each group"),
    # repeat_baseline.py
    (repeat_baseline.NO_DRAWS, "no draw of the full run is recorded for this serving route"),
    (repeat_baseline.NO_PROMPTS,
     "no prompt has a fully recorded ensemble with at least 2 replies that could be read"),
)

#: A count in a text: a number standing on its own, not a part of a name or of a requirement number.
COUNT = re.compile(r"(?<![\w.:=])\d+(?![\w.])")


@pytest.mark.parametrize(("stored", "published"), REASONS, ids=[stored[:60] for stored, _ in REASONS])
def test_each_stored_reason_is_published_as_a_plain_sentence_with_its_numbers(stored: str, published: str) -> None:
    found = published_reason(stored)
    assert found == published
    assert not found.endswith(".") and "\n" not in found and "  " not in found
    for what, pattern in FORBIDDEN_PUBLISHED:
        assert not pattern.search(found), (what, found)
    assert check_published_text(found) == found


@pytest.mark.parametrize("stored", [
    "23 of 32 draws parsed, below min_parsed=24; failed draws: parse_error=9.",
    "5 months carry the value in both renderings, fewer than the frozen block length of 6 months, so the "
    "moving-block bootstrap is not defined",
    "insufficient: 3 memorized and 12 control held-out draws; the local report's held-out calibration needs at "
    "least 10 per class",
    "k 100 is above the 20 alternatives the evidence keeps",
    "months with a top_k estimate under the identifying prompt: 12 inside the training range and 1 after the "
    "cutoff; at least 2 are needed in each group",
])
def test_the_counts_of_a_stored_reason_are_kept_as_numbers(stored: str) -> None:
    published = published_reason(stored)
    for count in COUNT.findall(stored):
        assert re.search(rf"(?<!\d){count}(?!\d)", published), (count, published)


def test_an_empty_reason_stays_empty_and_whitespace_is_flattened() -> None:
    assert published_reason(None) == "" and published_reason("") == "" and published_reason("  \n ") == ""
    assert published_reason("no estimate\nfor   this draw.") == "this draw has no recall estimate"


@pytest.mark.parametrize("stored", [
    "the moon was in the wrong phase",
    "no consensus: the moon was in the wrong phase",
    "draw did not parse: cosmic_ray",
    "no estimate for this draw; the moon was in the wrong phase",
    "holdout_auc_logits, moon_phase cannot be computed; missing: held-out prompts of both classes",
    "23 of 32 draws parsed, below min_parsed=24; failed draws: cosmic_ray=9.",
])
def test_an_unknown_reason_is_refused_and_never_published_as_stored(stored: str) -> None:
    with pytest.raises(FigureError, match="no published wording"):
        published_reason(stored)


@pytest.mark.parametrize(("text", "what"), [
    ("values compare only within one build (Requirement 9.4)", "a requirement number"),
    ("see task 3.5 for the figure", "a task number"),
    ("no calibrator file at serving/macro", "a path separator"),
    ("read `gap.csv`", "a backtick"),
    ("below min_parsed", "an identifier with an underscore"),
    ("missing for the serving arm", "arm vocabulary"),
    ("the sampling arm draws again", "arm vocabulary"),
    ("the primary route", "arm vocabulary"),
    ("the probe draws", "arm vocabulary"),
    ("in both renderings", "arm vocabulary"),
    ("the top-k estimate", "arm vocabulary"),
    ("the full-distribution host", "arm vocabulary"),
    ("a value — with a dash", "an em dash"),
])
def test_a_text_in_the_working_vocabulary_is_refused(text: str, what: str) -> None:
    with pytest.raises(FigureError, match=what):
        check_published_text(text)


def test_the_serving_route_is_data_and_is_allowed() -> None:
    assert check_published_text("this serving route records no logits") == "this serving route records no logits"
    assert check_published_text("on the serving routes of the plan")
