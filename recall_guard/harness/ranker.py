"""Composite ranking + ``top3.md`` writer.

Implements the ``harness.ranker`` component from the honest-model-ranking
design (see design.md → Components and Interfaces → harness.ranker).
Satisfies Requirements 5.3, 6.4, 7.4, 8.1, 8.2, 8.3, 8.4.

Pipeline per ``ModelEvalResult``:

1. Pull the three composite components — MemGuard accuracy lower CI bound,
   MCS-AUC point estimate, and parse-success rate — into the
   ``CompositeScore.components`` dict (Req 8.1).
2. Apply the four gating warnings:

   * ``weak-calibration`` when ``mcs_auc.point < gates["mcs_auc_min"]`` (Req 5.3).
   * ``parse-unreliable`` when ``parse_success_rate < gates["parse_min"]``
     (Req 7.4).
   * ``not-better-than-baseline`` when ``memguard_accuracy.lo`` does not
     strictly exceed the majority-class upper CI bound (Req 6.4).
   * ``uncalibrated`` when the upstream evaluator/runner already flagged the
     result (Req 3.4 surfaced via the runner's ``ControlBaseline.is_calibrated``
     check).
3. Pass through ``temperature-not-honoured`` (Req 10.3) without treating it as
   a gate — it is purely informational (design.md "harness.ranker" — gates
   table makes it explicit that only the four listed warnings block).
4. Multiplicative gate: any blocking warning sets
   ``survives_gates=False`` and ``score=0.0`` (Req 8.1, design Invariants).
5. Surviving models score
   ``memguard_acc_lo * mcs_auc_point * parse_success_rate``.

``write_top3`` (Req 8.2, 8.3, 8.4):

* Sorts by ``score`` descending, stable on ties (input order preserved).
* Top section lists at most three *surviving* models — gate-failed scores
  never appear in the top-3 ledger because they would be misleading even
  with a zero score.
* Whenever fewer than three models survive, a ``Why fewer than three models``
  section enumerates each non-survivor and the gate(s) it failed (Req 8.3).
* A ``Composite score formula`` footer always shows the formula string and
  the gate thresholds so the reader can reproduce the ranking (Req 8.4).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from recall_guard.harness.evaluator import CIBound, ModelEvalResult

# --- Constants ----------------------------------------------------------------

#: Human-readable composite-score formula. Persisted in ``top3.md`` and the run
#: manifest so a reader can reproduce the ranking from per-model metrics
#: (Req 8.4).
COMPOSITE_FORMULA = "memguard_acc_lo * mcs_auc_point * parse_success_rate"

#: Default gate thresholds. ``parse_min`` is the minimum parse-success rate
#: (Req 7.4); ``mcs_auc_min`` is the minimum MCS-AUC point estimate (Req 5.3).
GATES: dict[str, float] = {"parse_min": 0.8, "mcs_auc_min": 0.6}


# --- Warning vocabulary -------------------------------------------------------

WARNING_WEAK_CALIBRATION = "weak-calibration"
WARNING_PARSE_UNRELIABLE = "parse-unreliable"
WARNING_NOT_BETTER_THAN_BASELINE = "not-better-than-baseline"
WARNING_UNCALIBRATED = "uncalibrated"
WARNING_TEMPERATURE_NOT_HONOURED = "temperature-not-honoured"

#: Warnings that block ``survives_gates``. ``temperature-not-honoured`` is
#: deliberately excluded — it is informational only (see design.md harness.ranker).
_BLOCKING_WARNINGS: frozenset[str] = frozenset(
    {
        WARNING_WEAK_CALIBRATION,
        WARNING_PARSE_UNRELIABLE,
        WARNING_NOT_BETTER_THAN_BASELINE,
        WARNING_UNCALIBRATED,
    }
)


# --- Public dataclass --------------------------------------------------------


@dataclass(frozen=True)
class CompositeScore:
    """Composite rank score for one model with gate verdict + warnings.

    Attributes
    ----------
    model:
        NVIDIA model ID this score belongs to.
    score:
        Multiplicative composite ``memguard_acc_lo * mcs_auc_point *
        parse_success_rate`` if all gates pass; ``0.0`` otherwise (Req 8.1).
    components:
        The three component values that fed the formula. Keys are stable:
        ``"memguard_acc_lo"``, ``"mcs_auc_point"``, ``"parse_success_rate"``.
    survives_gates:
        ``True`` iff none of ``weak-calibration``, ``parse-unreliable``,
        ``not-better-than-baseline``, ``uncalibrated`` are in ``warnings``.
        ``temperature-not-honoured`` does not affect this flag.
    warnings:
        Subset of the ranker warning vocabulary, including any
        informational warnings (``temperature-not-honoured``) passed through
        from the evaluator.
    """

    model: str
    score: float
    components: dict[str, float]
    survives_gates: bool
    warnings: list[str]


# --- Composite scoring -------------------------------------------------------


def _gate_warnings(
    result: ModelEvalResult,
    majority_baseline: CIBound,
    gates: dict[str, float],
) -> list[str]:
    """Compute the warning list for one model in stable, deterministic order.

    Order is fixed so the rendered ``top3.md`` does not flap between runs
    when more than one warning fires.
    """
    warnings: list[str] = []

    if result.mcs_auc.point < gates["mcs_auc_min"]:
        warnings.append(WARNING_WEAK_CALIBRATION)

    if result.parse_success_rate < gates["parse_min"]:
        warnings.append(WARNING_PARSE_UNRELIABLE)

    if result.memguard_accuracy.lo <= majority_baseline.hi:
        warnings.append(WARNING_NOT_BETTER_THAN_BASELINE)

    # Pass-through warnings from the evaluator / runner. ``uncalibrated`` is
    # set by the runner when the control baseline could not be built; the
    # evaluator never sets it directly. ``temperature-not-honoured`` is
    # informational only.
    upstream = result.warnings or []
    if WARNING_UNCALIBRATED in upstream:
        warnings.append(WARNING_UNCALIBRATED)
    if WARNING_TEMPERATURE_NOT_HONOURED in upstream:
        warnings.append(WARNING_TEMPERATURE_NOT_HONOURED)

    return warnings


def composite_score(
    results: list[ModelEvalResult],
    majority_baseline: CIBound,
    formula: str = COMPOSITE_FORMULA,
    gates: dict[str, float] = GATES,
) -> list[CompositeScore]:
    """Convert per-model evaluation results into composite scores.

    The ``formula`` argument is persisted alongside the gates in ``top3.md``
    (Req 8.4) but is not parsed at runtime — the multiplicative formula is the
    only one defined in this spec. A future variant would change both the
    component dict keys and the formula string in lockstep.

    Returns the scores in the *input order* of ``results``; ordering is
    deferred to ``write_top3`` (or other downstream consumers) so the
    canonical record stream stays aligned with the eval-set order (Req 9.3).
    """
    del formula  # Persisted via write_top3; not interpreted here.

    scores: list[CompositeScore] = []
    for result in results:
        components: dict[str, float] = {
            "memguard_acc_lo": float(result.memguard_accuracy.lo),
            "mcs_auc_point": float(result.mcs_auc.point),
            "parse_success_rate": float(result.parse_success_rate),
        }

        warnings = _gate_warnings(result, majority_baseline, gates)
        survives = not any(w in _BLOCKING_WARNINGS for w in warnings)

        if survives:
            score_value = (
                components["memguard_acc_lo"]
                * components["mcs_auc_point"]
                * components["parse_success_rate"]
            )
        else:
            score_value = 0.0

        scores.append(
            CompositeScore(
                model=result.model,
                score=float(score_value),
                components=components,
                survives_gates=survives,
                warnings=warnings,
            )
        )

    return scores


# --- top3.md writer ----------------------------------------------------------


def _format_components(components: dict[str, float]) -> str:
    """Render the component dict as a stable, copy-pasteable inline string."""
    parts = [
        f"memguard_acc_lo={components['memguard_acc_lo']:.4f}",
        f"mcs_auc_point={components['mcs_auc_point']:.4f}",
        f"parse_success_rate={components['parse_success_rate']:.4f}",
    ]
    return ", ".join(parts)


def _failed_gates(score: CompositeScore) -> list[str]:
    """Return the blocking warnings on a non-survivor in deterministic order."""
    return [w for w in score.warnings if w in _BLOCKING_WARNINGS]


def _render_survivor_block(rank: int, score: CompositeScore) -> list[str]:
    """Render one ranked surviving model as a numbered Markdown entry."""
    info_warnings = [w for w in score.warnings if w not in _BLOCKING_WARNINGS]
    lines = [
        f"{rank}. **{score.model}** — score = {score.score:.4f}",
        f"   - Components: {_format_components(score.components)}",
    ]
    if info_warnings:
        lines.append(f"   - Warnings: {', '.join(info_warnings)}")
    return lines


def _render_short_list_explanation(
    total_evaluated: int,
    nonsurvivors: list[CompositeScore],
) -> list[str]:
    """Render the 'Why fewer than three models' section (Req 8.3)."""
    lines = [
        "## Why fewer than three models",
        "",
        (
            f"Of {total_evaluated} model(s) evaluated, "
            f"{len(nonsurvivors)} did not pass all gates "
            "(parse-success ≥ 0.8, MCS-AUC ≥ 0.6, accuracy lower CI > "
            "majority upper CI, control baseline calibrated)."
        ),
        "",
    ]
    for ns in nonsurvivors:
        failed = _failed_gates(ns)
        failed_str = ", ".join(failed) if failed else "unknown"
        lines.append(f"- **{ns.model}** failed: {failed_str}")
    lines.append("")
    return lines


def _render_formula_footer(formula: str, gates: dict[str, float]) -> list[str]:
    """Render the composite-score formula footer (Req 8.4)."""
    gate_strs = ", ".join(f"{k}={v}" for k, v in gates.items())
    return [
        "## Composite score formula",
        "",
        f"Formula: `{formula}`",
        "",
        f"Gates: `{gate_strs}` plus accuracy lower CI > majority upper CI.",
        "",
        (
            "Any blocking warning (weak-calibration, parse-unreliable, "
            "not-better-than-baseline, uncalibrated) sets the score to 0 and "
            "removes the model from the top-3 list."
        ),
        "",
    ]


def write_top3(
    scores: list[CompositeScore],
    path: Path,
    formula: str = COMPOSITE_FORMULA,
    gates: dict[str, float] = GATES,
) -> None:
    """Write ``top3.md`` to ``path``.

    The file always contains the ``# Top 3 Models`` heading and the
    ``## Composite score formula`` footer; the explanatory section is
    included whenever fewer than three models survive (Req 8.3). The parent
    directory is created if missing so callers do not have to mkdir first.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Stable sort: descending by score, ties broken by input order.
    # ``sorted`` in CPython is stable, so reversing the indexed pairs is not
    # needed — we sort by negated score directly.
    indexed: list[tuple[int, CompositeScore]] = list(enumerate(scores))
    indexed.sort(key=lambda pair: -pair[1].score)
    sorted_scores = [s for _, s in indexed]

    survivors = [s for s in sorted_scores if s.survives_gates]
    nonsurvivors_in_input_order = [s for s in scores if not s.survives_gates]
    top_survivors = survivors[:3]

    lines: list[str] = ["# Top 3 Models", ""]

    if not top_survivors:
        lines.append("_No models passed all gates._")
        lines.append("")
    else:
        for rank, score in enumerate(top_survivors, start=1):
            lines.extend(_render_survivor_block(rank, score))
            lines.append("")

    if len(top_survivors) < 3:
        lines.extend(
            _render_short_list_explanation(
                total_evaluated=len(scores),
                nonsurvivors=nonsurvivors_in_input_order,
            )
        )

    lines.extend(_render_formula_footer(formula, gates))

    path.write_text("\n".join(lines), encoding="utf-8")


__all__ = [
    "COMPOSITE_FORMULA",
    "GATES",
    "CompositeScore",
    "composite_score",
    "write_top3",
]
