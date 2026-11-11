"""Repeat-the-prompt baseline (qualified-model-catalogue task 3.3; its Requirements 5.6 and 6.1 to 6.4).

The question: when the same prompt is asked again on one serving route at one
temperature, how many different answers come back. It is the baseline any
ensemble result is read against.

The analysis reads recorded data only and sends no request: the frozen plan and
the recorded draws of each arm's full-run stage (the stage's versioned draw log,
``draws.jsonl``, read through :class:`~scripts.finance_sim.recording.DrawLog` as
:class:`~scripts.finance_sim.execute.ExecutionDraw` values, the latest version
of each draw). It reads draws, not stored item results, so an open stage with
draws and no results is analysed too.

Definitions. A prompt is one item id and rendering. Its ensemble is fully
recorded when every planned draw index of the arm (0 to ``draws - 1`` of the
frozen arm) is recorded, whatever the draw's outcome; an ensemble recorded only
in part is excluded and counted. Within a fully recorded ensemble, the number
of different answers is the number of distinct parsed answers among the draws
that parsed: the parsed component values, compared exactly after canonical
serialisation (sorted keys, values as floats). A fully recorded ensemble with
fewer than 2 parsed draws cannot show variation and is excluded and counted
separately.

Per arm, labelled by its serving route (the arm's provider profile and that
profile's provider, with the served model) and its frozen temperature, the
analysis reports the two exclusion counts, the number of prompts analysed, the
distribution (the number of prompts with 1, 2, 3, ... different answers), the
median, and the share of prompts with exactly one answer with a two-sided
Wilson score interval at 95 percent. Every reported value is a
:class:`~recall_guard.core.significance.Statistic`: estimand, formula, inputs
with their counts, the unit of analysis (prompt), n and whether the units are
independent (two renderings of one month share the month, so they are not). An
arm without recorded draws gives records with no value and ``missing`` set,
never a value.

Draws the plan does not size (another item or rendering, a draw index beyond
the arm's ensemble, a temperature other than the arm's) are refused, and draws
recorded under another plan are refused by the draw log.

Outputs: ``repeat-baseline.json`` (canonical: sorted keys, LF newlines) and its
markdown table ``repeat-baseline.md`` under ``runs/finance-sim/analysis/``, one
file for both arms, written once and never replaced by a different one. Derived
statistics only: no reply text, no parsed answer value and no per-prompt row.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median
from typing import Any

from recall_guard.core.consensus import Tail, wilson_interval
from recall_guard.core.profiles import load_profile
from recall_guard.core.significance import Interval, Statistic
from scripts.finance_sim.execute import ExecutionDraw, planned_keys
from scripts.finance_sim.limits import ARMS, RECORDS_DIR, RUNS_ROOT
from scripts.finance_sim.pilot import CONFIRMATORY_STAGE
from scripts.finance_sim.plan import StudyPlan, load_plan, require_complete
from scripts.finance_sim.recording import DrawLog, draw_log_path

#: Analysis outputs (design: ``runs/finance-sim/analysis/``).
ANALYSIS_DIR = RUNS_ROOT / "analysis"
BASELINE_SCHEMA = "finance-sim-repeat-baseline/1"
#: The stage whose recorded draws are analysed: each arm's full run.
STAGE = CONFIRMATORY_STAGE
#: Two-sided coverage of the Wilson interval.
CONFIDENCE = 0.95
#: Parsed draws a fully recorded ensemble needs to show variation.
MIN_PARSED_DRAWS = 2
UNIT = "prompt"
WILSON_METHOD = "Wilson score interval, two-sided"

NOT_FULLY_RECORDED = "ensembles_not_fully_recorded"
TOO_FEW_PARSED = f"prompts_with_fewer_than_{MIN_PARSED_DRAWS}_parsed_draws"
ANALYSED = "prompts_analysed"
MEDIAN = "median_different_answers"
ONE_ANSWER_SHARE = "share_of_prompts_with_one_answer"

NO_DRAWS = "recorded draws of the arm's full-run stage: none are recorded"
NO_PROMPTS = (f"prompts with a fully recorded ensemble and at least {MIN_PARSED_DRAWS} parsed draws: the recorded "
              "draws hold none")
RULE = (
    "a prompt is one item id and rendering; its ensemble is fully recorded when every planned draw index of the arm "
    "is recorded, and an ensemble recorded only in part is excluded and counted; the number of different answers of "
    "a prompt is the number of distinct parsed answers among its parsed draws, the parsed component values compared "
    f"exactly after canonical serialisation; a fully recorded ensemble with fewer than {MIN_PARSED_DRAWS} parsed "
    "draws cannot show variation and is excluded and counted separately"
)
INDEPENDENCE = (
    "the unit of analysis is the prompt; the two renderings of one month share the month, so prompts are not "
    "independent where both renderings of a month are counted, and the Wilson interval, which assumes independent "
    "prompts, may then be too narrow"
)
EXIT_WRITTEN, EXIT_REFUSED = 0, 2
_SHA256_LENGTH = 64


class RepeatBaselineError(ValueError):
    """Recorded draws or outputs of the repeat baseline are refused."""


def distribution_name(answers: int) -> str:
    """Name of the statistic counting the prompts with ``answers`` different answers."""
    return f"prompts_with_{answers}_different_answers"


# ---------- one arm ----------

def _provider(profile_id: str) -> str:
    """The provider the arm's profile names (a local package file read)."""
    return load_profile(profile_id).model_config.provider


def _answer(draw: ExecutionDraw) -> str | None:
    """The canonical serialisation of a parsed answer, None for a draw that did not parse."""
    parsed = draw.draw.parsed
    if parsed is None:
        return None
    return json.dumps({name: float(value) for name, value in parsed.items()}, sort_keys=True, separators=(",", ":"),
                      allow_nan=False)


def _by_prompt(arm: str, planned: Sequence[tuple[str, str]], size: int, temperature: float,
               draws: Iterable[ExecutionDraw]) -> dict[tuple[str, str], dict[int, ExecutionDraw]]:
    """The recorded draws of each planned prompt by draw index, in plan order; anything unplanned is refused."""
    if isinstance(draws, str | bytes | Mapping):
        raise TypeError("draws must be a sequence of ExecutionDraw values")
    found: dict[tuple[str, str], dict[int, ExecutionDraw]] = {key: {} for key in planned}
    for draw in draws:
        if not isinstance(draw, ExecutionDraw):
            raise TypeError(f"draws must be ExecutionDraw values; got {type(draw).__name__}")
        key, index = (draw.item_id, draw.rendering), draw.draw.draw_index
        if key not in found:
            raise RepeatBaselineError(f"a recorded draw is not a planned item and rendering of the {arm} arm's "
                                      f"{STAGE} stage; refused")
        if index >= size:
            raise RepeatBaselineError(f"a recorded draw has draw index {index}; the frozen {arm} arm draws {size} "
                                      "per ensemble; refused")
        if index in found[key]:
            raise RepeatBaselineError(f"draw index {index} of one prompt of the {arm} arm is given twice; refused")
        if draw.draw.temperature is not None and float(draw.draw.temperature) != temperature:
            raise RepeatBaselineError(f"a recorded draw was made at temperature {draw.draw.temperature:g}; the "
                                      f"frozen {arm} arm draws at temperature {temperature:g}; refused")
        found[key][index] = draw
    return {key: indexed for key, indexed in found.items() if indexed}


def _independent(keys: Sequence[tuple[str, str]], planned: Sequence[tuple[str, str]]) -> bool:
    """True when no two of the prompts share an item (a month); the planned prompts stand in for an empty set."""
    units = keys or planned
    return len({item_id for item_id, _ in units}) == len(units)


def _arm(plan: StudyPlan, arm: str, draws: Iterable[ExecutionDraw], provider: str
         ) -> tuple[dict[str, Any], tuple[Statistic, ...]]:
    """The header and the statistics of one arm."""
    arm_plan = plan.arms.arms[arm]
    size, temperature = arm_plan.draws, float(arm_plan.temperature)
    planned = planned_keys(plan, arm, STAGE)
    recorded = _by_prompt(arm, planned, size, temperature, draws)
    n_draws = sum(len(indexed) for indexed in recorded.values())
    route = f"{arm_plan.profile_id} ({provider}, model {arm_plan.served_model})"
    where = (f"the {arm} arm's {STAGE} stage on the serving route {route} at temperature {temperature:g}, "
             f"{size} planned draws per prompt")
    header = {"arm": arm, "route": {"profile_id": arm_plan.profile_id, "provider": provider,
                                    "served_model": arm_plan.served_model},
              "temperature": temperature, "planned_draws": size, "planned_prompts": len(planned),
              "recorded_draws": n_draws, "missing": None if n_draws else NO_DRAWS}

    def statistic(name: str, estimand: str, formula: str, inputs: Mapping[str, float | int],
                  keys: Sequence[tuple[str, str]], value: float | None, *, interval: Interval | None = None,
                  missing: str | None = None) -> Statistic:
        return Statistic(name=name, estimand=f"{estimand}, for {where}", formula=formula, inputs=inputs, unit=UNIT,
                         n=len(keys), independent_units=_independent(keys, planned), value=value, interval=interval,
                         test=None, effect_size=None, detectable_effect=None, by_construction=False,
                         missing=None if value is not None else missing)

    complete = [key for key, indexed in recorded.items() if len(indexed) == size]
    counts: dict[tuple[str, str], int] = {}
    for key in complete:
        answers = [answer for answer in (_answer(recorded[key][index]) for index in range(size)) if answer is not None]
        if len(answers) >= MIN_PARSED_DRAWS:
            counts[key] = len(set(answers))
    analysed = list(counts)
    n = len(analysed)
    distribution = Counter(counts.values())
    largest = max(distribution, default=1)
    spread = {distribution_name(k): distribution.get(k, 0) for k in range(1, largest + 1)}
    one = distribution.get(1, 0)
    absent = NO_DRAWS if not n_draws else NO_PROMPTS
    has_draws, has_prompts = n_draws > 0, n > 0

    estimands = {
        NOT_FULLY_RECORDED: "number of prompts with at least one recorded draw whose ensemble is not fully recorded "
                            "(excluded)",
        TOO_FEW_PARSED: f"number of prompts with a fully recorded ensemble and fewer than {MIN_PARSED_DRAWS} parsed "
                        "draws (excluded: they cannot show variation)",
        ANALYSED: f"number of prompts with a fully recorded ensemble and at least {MIN_PARSED_DRAWS} parsed draws "
                  "(analysed)",
        MEDIAN: "median over the analysed prompts of the number of different parsed answers the same prompt returned",
        ONE_ANSWER_SHARE: "share of the analysed prompts whose parsed draws all returned the same answer",
    }
    statistics = [
        statistic(NOT_FULLY_RECORDED, estimands[NOT_FULLY_RECORDED],
                  "prompts_with_recorded_draws - fully_recorded_ensembles",
                  {"prompts_with_recorded_draws": len(recorded), "fully_recorded_ensembles": len(complete),
                   "planned_draws_per_ensemble": size, "planned_prompts": len(planned),
                   "planned_prompts_without_recorded_draws": len(planned) - len(recorded)},
                  list(recorded), len(recorded) - len(complete) if has_draws else None, missing=NO_DRAWS),
        statistic(TOO_FEW_PARSED, estimands[TOO_FEW_PARSED], f"fully_recorded_ensembles - {ANALYSED}",
                  {"fully_recorded_ensembles": len(complete), ANALYSED: n}, complete,
                  len(complete) - n if has_draws else None, missing=NO_DRAWS),
        statistic(ANALYSED, estimands[ANALYSED], f"fully_recorded_ensembles - {TOO_FEW_PARSED}",
                  {"fully_recorded_ensembles": len(complete), TOO_FEW_PARSED: len(complete) - n}, complete,
                  n if has_draws else None, missing=NO_DRAWS),
    ]
    for answers in range(1, largest + 1):
        name = distribution_name(answers)
        statistics.append(statistic(
            name, f"number of analysed prompts whose parsed draws returned exactly {answers} different "
                  f"answer{'s' if answers > 1 else ''}",
            f"count of the {ANALYSED} prompts with exactly {answers} distinct parsed answer"
            f"{'s' if answers > 1 else ''} among their parsed draws",
            {ANALYSED: n}, analysed, spread[name] if has_prompts else None, missing=absent))
    bounds = wilson_interval(one, n, confidence=CONFIDENCE, tail=Tail.TWO_SIDED)
    statistics += [
        statistic(MEDIAN, estimands[MEDIAN],
                  f"median of the number of different answers over the {ANALYSED} prompts (the mean of the two middle "
                  "values when their number is even), from the distribution counts",
                  {**spread, ANALYSED: n}, analysed, float(median(counts.values())) if has_prompts else None,
                  missing=absent),
        statistic(ONE_ANSWER_SHARE, estimands[ONE_ANSWER_SHARE], f"{distribution_name(1)} / {ANALYSED}",
                  {distribution_name(1): one, ANALYSED: n}, analysed, one / n if has_prompts else None,
                  interval=None if bounds is None else Interval(lo=bounds[0], hi=bounds[1], confidence=CONFIDENCE,
                                                                method=WILSON_METHOD),
                  missing=absent),
    ]
    return header, tuple(statistics)


# ---------- the analysis ----------

def _statistic_dict(statistic: Statistic) -> dict[str, Any]:
    interval = statistic.interval
    return {"name": statistic.name, "estimand": statistic.estimand, "formula": statistic.formula,
            "inputs": dict(statistic.inputs), "unit": statistic.unit, "n": statistic.n,
            "independent_units": statistic.independent_units, "value": statistic.value,
            "interval": None if interval is None else {"lo": interval.lo, "hi": interval.hi,
                                                       "confidence": interval.confidence, "method": interval.method},
            "test": None, "effect_size": statistic.effect_size, "detectable_effect": statistic.detectable_effect,
            "by_construction": statistic.by_construction, "missing": statistic.missing}


@dataclass(frozen=True)
class RepeatBaseline:
    """The repeat baseline of both arms: the canonical JSON document, its statistics and its markdown table."""

    plan_sha256: str
    document: str
    statistics: Mapping[str, tuple[Statistic, ...]] = field(compare=False, repr=False)

    def statistic(self, arm: str, name: str) -> Statistic:
        """The statistic ``name`` of ``arm``."""
        for found in self.statistics[arm]:
            if found.name == name:
                return found
        raise KeyError(f"the {arm} arm reports no statistic {name!r}")

    def to_dict(self) -> dict[str, Any]:
        return json.loads(self.document)

    def to_markdown(self) -> str:
        return _markdown(self.to_dict())


def repeat_baseline(plan: StudyPlan, draws: Mapping[str, Iterable[ExecutionDraw]], *,
                    providers: Mapping[str, str] | None = None) -> RepeatBaseline:
    """The repeat baseline of both arms from the recorded draws of their full runs; no request is sent.

    ``draws`` gives the recorded draws per arm (the latest version of each); an
    arm it does not name, or names without draws, is reported missing.
    ``providers`` names each arm's provider; by default it is read from the
    arm's provider profile.
    """
    if not isinstance(plan, StudyPlan):
        raise TypeError(f"plan must be a StudyPlan; got {type(plan).__name__}")
    require_complete(plan)
    if not isinstance(draws, Mapping):
        raise TypeError("draws must map each arm to its recorded draws")
    unknown = [arm for arm in draws if arm not in ARMS]
    if unknown:
        raise RepeatBaselineError(f"draws name the unknown arm {unknown}; the arms are {list(ARMS)}")
    arms: dict[str, dict[str, Any]] = {}
    statistics: dict[str, tuple[Statistic, ...]] = {}
    for arm in ARMS:
        profile_id = plan.arms.arms[arm].profile_id
        provider = providers[arm] if providers is not None else _provider(profile_id)
        if not isinstance(provider, str) or not provider:
            raise RepeatBaselineError(f"the provider of the {arm} arm must be a non-empty string")
        header, found = _arm(plan, arm, draws.get(arm, ()), provider)
        arms[arm] = {**header, "statistics": [_statistic_dict(statistic) for statistic in found]}
        statistics[arm] = found
    document = {"schema": BASELINE_SCHEMA, "plan_sha256": plan.plan_sha256, "stage": STAGE, "confidence": CONFIDENCE,
                "rule": RULE, "independence": INDEPENDENCE, "arms": arms}
    text = json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n"
    return RepeatBaseline(plan_sha256=plan.plan_sha256, document=text, statistics=statistics)


def repeat_baseline_from_records(plan: StudyPlan, *, records_dir: str | Path = RECORDS_DIR) -> RepeatBaseline:
    """The repeat baseline from each arm's recorded full-run draws (``draws.jsonl``); no request is sent."""
    if not isinstance(plan, StudyPlan):
        raise TypeError(f"plan must be a StudyPlan; got {type(plan).__name__}")
    draws = {arm: DrawLog(draw_log_path(arm, STAGE, records_dir), plan=plan, loads=ExecutionDraw.from_dict).latest()
             for arm in ARMS}
    return repeat_baseline(plan, draws)


# ---------- the markdown table ----------

def _number(value: float | None) -> str:
    return "" if value is None else f"{value:.4g}"


def _cell_text(text: str) -> str:
    return text.replace("|", "/").replace("\n", " ")


def _row(arm: Mapping[str, Any], statistic: Mapping[str, Any]) -> list[str]:
    route = arm["route"]
    interval = statistic["interval"]
    inputs = ", ".join(f"{name} = {value:g}" for name, value in statistic["inputs"].items())
    calculation = f"{statistic['formula']}; inputs: {inputs}"
    if statistic["value"] is None:
        calculation = f"missing input: {statistic['missing']}"
    elif not statistic["independent_units"]:
        calculation += "; prompts not independent"
    bounds = "" if interval is None else (f"{interval['lo']:.4g} to {interval['hi']:.4g} "
                                          f"({interval['confidence'] * 100:g} percent)")
    return [arm["arm"], f"{route['profile_id']} ({route['provider']})", f"{arm['temperature']:g}", statistic["name"],
            "missing" if statistic["value"] is None else _number(statistic["value"]), bounds, str(statistic["n"]),
            calculation]


def _markdown(data: Mapping[str, Any]) -> str:
    lines = [
        f"# Repeat-the-prompt baseline: {data['stage']} stage",
        "",
        "How many different answers the same prompt returned when it was asked again, per serving route and "
        f"temperature. Frozen plan {data['plan_sha256']}.",
        "",
    ]
    for arm in data["arms"].values():
        route = arm["route"]
        state = "no recorded draws" if arm["missing"] is not None else f"{arm['recorded_draws']} recorded draws"
        lines.append(f"- {arm['arm']} arm: serving route {route['profile_id']} (provider {route['provider']}, model "
                     f"{route['served_model']}), temperature {arm['temperature']:g}, {arm['planned_draws']} planned "
                     f"draws per prompt, {arm['planned_prompts']} planned prompts, {state}.")
    lines += [
        "",
        "| Arm | Serving route | Temperature | Statistic | Value | Interval | n (prompts) | Calculation |",
        "|---|---|---|---|---|---|---|---|",
    ]
    lines.extend("| " + " | ".join(_cell_text(cell) for cell in _row(arm, statistic)) + " |"
                 for arm in data["arms"].values() for statistic in arm["statistics"])
    lines += [
        "",
        f"Method: {data['rule']}.",
        f"Interval: {WILSON_METHOD}, at {data['confidence'] * 100:g} percent, on the share of prompts with one answer; "
        "the counts and the median carry no interval.",
        f"Independence: {data['independence']}.",
        "A value that cannot be computed is shown as missing with the input that is missing. Derived statistics "
        "only: no reply, parsed answer or prompt is listed.",
        "",
    ]
    return "\n".join(lines)


# ---------- the written outputs ----------

def baseline_path(out_dir: str | Path = ANALYSIS_DIR) -> Path:
    """``<out_dir>/repeat-baseline.json``; its markdown table sits beside it with ``.md``."""
    return Path(out_dir) / "repeat-baseline.json"


def _write_text(path: Path, text: str) -> None:
    """Write atomically: a crash leaves the previous file or the new one, never half of one."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def write_baseline(baseline: RepeatBaseline, out_dir: str | Path = ANALYSIS_DIR) -> tuple[Path, Path]:
    """Write the JSON document and the markdown table once; the same outputs are a no-op, different ones refused."""
    if not isinstance(baseline, RepeatBaseline):
        raise TypeError(f"baseline must be a RepeatBaseline; got {type(baseline).__name__}")
    json_path = baseline_path(out_dir)
    markdown_path = json_path.with_suffix(".md")
    texts = {json_path: baseline.document, markdown_path: baseline.to_markdown()}
    for path, text in texts.items():
        if path.exists() and path.read_bytes() != text.encode("utf-8"):
            raise RepeatBaselineError(f"refusing to replace the different repeat baseline at {path}")
    for path, text in texts.items():
        if not path.exists():
            _write_text(path, text)
    return json_path, markdown_path


def main(argv: Sequence[str] | None = None) -> int:
    """Write the repeat baseline of both arms from the private recorded draws of the frozen plan.

    Exit codes: 0 written (or already written identically), 2 refused (another
    plan, draws the plan does not size, or different outputs at the path).
    """
    parser = argparse.ArgumentParser(description="Repeat-the-prompt baseline from recorded draws; sends no request.")
    parser.add_argument("--plan", type=Path, required=True, help="the frozen plan.json")
    parser.add_argument("--plan-sha256", required=True, help="the plan hash the recorded draws carry")
    parser.add_argument("--records-dir", type=Path, default=RECORDS_DIR)
    parser.add_argument("--out-dir", type=Path, default=ANALYSIS_DIR)
    args = parser.parse_args(argv)
    if len(args.plan_sha256) != _SHA256_LENGTH:
        print("repeat baseline: refused: --plan-sha256 must be 64 hex characters", file=sys.stderr)
        return EXIT_REFUSED
    try:
        plan = load_plan(args.plan, expected_sha256=args.plan_sha256)
        paths = write_baseline(repeat_baseline_from_records(plan, records_dir=args.records_dir), args.out_dir)
    except (ValueError, TypeError, OSError) as exc:
        print(f"repeat baseline: refused: {exc}", file=sys.stderr)
        return EXIT_REFUSED
    for path in paths:
        print(path)
    return EXIT_WRITTEN


__all__ = [
    "ANALYSED",
    "ANALYSIS_DIR",
    "BASELINE_SCHEMA",
    "CONFIDENCE",
    "INDEPENDENCE",
    "MEDIAN",
    "MIN_PARSED_DRAWS",
    "NOT_FULLY_RECORDED",
    "NO_DRAWS",
    "NO_PROMPTS",
    "ONE_ANSWER_SHARE",
    "RULE",
    "STAGE",
    "TOO_FEW_PARSED",
    "WILSON_METHOD",
    "RepeatBaseline",
    "RepeatBaselineError",
    "baseline_path",
    "distribution_name",
    "main",
    "repeat_baseline",
    "repeat_baseline_from_records",
    "write_baseline",
]


if __name__ == "__main__":
    raise SystemExit(main())
