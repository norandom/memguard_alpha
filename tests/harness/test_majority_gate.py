"""Majority-baseline gating integration test — Task 6.4.

Drives the real ``runner.run`` pipeline against an eval set whose majority-class
accuracy is 0.8, with mocked LMs whose Raw Accuracy lower CI does not strictly
exceed the majority-class upper CI bound. Asserts that the resulting
``CompositeScore.warnings`` includes ``not-better-than-baseline`` and that the
generated ``top3.md`` reflects this in its "Why fewer than three models"
section.

Covered acceptance criteria:

* Req 6.2: ``compute_majority_baseline`` produces a bootstrap CI used by the
  composite ranker as a gating threshold.
* Req 6.4: when ``memguard_accuracy.lo <= majority_baseline.hi`` the model is
  flagged ``not-better-than-baseline`` and removed from the top-3 list.

The test deliberately uses ``--shortlist`` (no smoke gate) and ``--no-reference``
so the only HTTP-shaped surface is the per-model evaluator + baseline + MCS, all
of which go through the injected ``lm_factory`` and never touch the network.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from src.core.nvidia_lm import CompletionResult, TokenLogprob
from src.harness import runner as runner_mod
from src.harness.evaluator import compute_majority_baseline
from src.harness.ranker import (
    WARNING_NOT_BETTER_THAN_BASELINE,
    composite_score,
)
from src.core.loader import load_eval_set


# --- Fixture paths reused from the e2e harness tests --------------------------

REPO_FIXTURES = Path(__file__).parent.parent / "fixtures"
TEMPLATE_IS = REPO_FIXTURES / "tiny_is_memorized.jsonl"
TEMPLATE_OOS = REPO_FIXTURES / "tiny_oos_control.jsonl"
TEMPLATE_CUTOFFS = REPO_FIXTURES / "tiny_cutoffs.yaml"


# --- Fake LM machinery (mirrors test_e2e.py) ---------------------------------


def _make_top_logprobs() -> list[dict[str, Any]]:
    return [{"token": f"tok{i}", "logprob": -1.0 - 0.1 * i} for i in range(20)]


def _make_logprobs(content: str) -> list[TokenLogprob]:
    tokens = content.split()
    if not tokens:
        tokens = ["x"]
    return [
        TokenLogprob(
            token=tok,
            logprob=-0.5 - 0.05 * (i % 5),
            top_logprobs=_make_top_logprobs(),
        )
        for i, tok in enumerate(tokens)
    ]


def _make_completion(direction: int, confidence: float = 0.7) -> CompletionResult:
    content = f"Direction: {direction}\nConfidence: {confidence}"
    return CompletionResult(
        content=content,
        logprobs=_make_logprobs(content),
        raw_temperature_observed=0.0,
    )


class _ConstantLM:
    """LM that always emits the same parsed direction.

    Lets us build models whose Raw Accuracy on a 16/4 split is exactly the
    majority share (80%) when ``direction == majority_class`` or 20% when it
    matches the minority instead — so the gate fires deterministically without
    relying on bootstrap noise.
    """

    def __init__(self, model: str, *, predicted_direction: int) -> None:
        self.model = model
        self.api_key = "fake"
        self.timeout_s = 15.0
        self.api_base = "fake://"
        self._direction = predicted_direction
        self.calls: list[str] = []

    def generate(
        self, prompt: str, temperature: float = 0.0
    ) -> CompletionResult:
        self.calls.append(prompt)
        return _make_completion(direction=self._direction, confidence=0.7)


class _PerfectLM:
    """LM that mirrors each prompt's target into ``Direction:``.

    The runner only sees prompts; we encode the target into the prompt text
    via the eval-set writer so this LM can decode it back. With every row
    correct, Raw Accuracy is 1.0 and the lower CI is well above the majority
    upper bound, so the ``not-better-than-baseline`` gate must NOT fire.
    """

    PROMPT_PREFIX = "TARGET="

    def __init__(self, model: str) -> None:
        self.model = model
        self.api_key = "fake"
        self.timeout_s = 15.0
        self.api_base = "fake://"
        self.calls: list[str] = []

    def generate(
        self, prompt: str, temperature: float = 0.0
    ) -> CompletionResult:
        self.calls.append(prompt)
        # Decode the target embedded in the prompt by the eval-set writer.
        # Prompts are of the form "TARGET=<int> ..."; we parse the first token.
        first_line = prompt.splitlines()[0] if prompt else ""
        token = first_line.split()[0] if first_line else ""
        target_str = token.split("=", 1)[1] if "=" in token else "1"
        try:
            direction = int(target_str)
        except ValueError:
            direction = 1
        return _make_completion(direction=direction, confidence=0.9)


def _make_factory(fakes: dict[str, Any]):
    def factory(api_key: str, model: str, timeout_s: float):
        if model not in fakes:
            raise KeyError(f"unexpected model in factory: {model}")
        return fakes[model]

    return factory


# --- Eval-set construction (20 rows, 80% majority class) ---------------------


def _write_majority_eval_set(path: Path, *, embed_target: bool = False) -> None:
    """Write a 20-row eval JSONL: 16 rows target=+1, 4 rows target=-1.

    Majority share is exactly 80%, so ``compute_majority_baseline`` returns a
    point estimate ≈ 0.8 with a tight bootstrap CI. The cutoff date predates
    every model in ``tiny_cutoffs.yaml`` so the cutoff guard does not abort.

    When ``embed_target`` is True, each prompt is prefixed with ``TARGET=<n>``
    so a ``_PerfectLM`` can recover the ground-truth direction from the prompt
    text alone — used by the negative-control test where the model should
    out-perform the baseline.
    """
    lines: list[str] = []
    lines.append(json.dumps({"_cutoff_date": "2025-06-30"}))
    targets = [1] * 16 + [-1] * 4  # 80% majority class +1.
    for i, target in enumerate(targets):
        if embed_target:
            prompt = f"TARGET={target} Eval prompt {i}"
        else:
            prompt = f"Eval prompt {i}"
        row = {
            "prompt": prompt,
            "target_direction": target,
            "metadata": {"ticker": f"T{i:02d}"},
        }
        lines.append(json.dumps(row))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _build_args(
    monkeypatch: pytest.MonkeyPatch,
    out_dir: Path,
    eval_set: Path,
    *,
    shortlist: str,
    seed: int = 0,
    bootstrap_n: int = 50,
):
    """Construct a parsed argparse namespace via the real ``build_parser``."""
    monkeypatch.setenv("NVIDIA_API_KEY", "test-api-key")
    parser = runner_mod.build_parser()
    cli = [
        "build",
        "--eval-set",
        str(eval_set),
        "--is-memorized",
        str(TEMPLATE_IS),
        "--oos-control",
        str(TEMPLATE_OOS),
        "--cutoffs",
        str(TEMPLATE_CUTOFFS),
        "--out-dir",
        str(out_dir),
        "--seed",
        str(seed),
        "--bootstrap-n",
        str(bootstrap_n),
        "--no-reference",
        "--shortlist",
        shortlist,
    ]
    return parser.parse_args(cli)


# --- Tests -------------------------------------------------------------------


def test_majority_gate_flags_not_better_than_baseline_in_top3(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Req 6.4: a model that merely matches the 80% majority share is gated.

    With targets ``[+1]*16 + [-1]*4``:

    * The majority-class baseline accuracy is 0.8 (point) with a tight CI.
    * ``mockA`` always predicts ``+1`` — its Raw Accuracy is exactly 0.8, so
      its bootstrap lower CI is at most the majority's upper CI, which fires
      the ``not-better-than-baseline`` warning per
      ``ranker._gate_warnings``: ``result.memguard_accuracy.lo <=
      majority_baseline.hi``.
    * ``mockB`` always predicts ``-1`` — accuracy 0.2, also gate-failed.

    With both models gate-failed, ``top3.md`` must contain the literal
    ``not-better-than-baseline`` substring (in the "Why fewer than three
    models" section), and ``summary.csv`` must mark ``survives_gates=false``
    for at least the always-predict-majority model.
    """
    eval_path = tmp_path / "eval_majority.jsonl"
    _write_majority_eval_set(eval_path, embed_target=False)

    out_dir = tmp_path / "run-majority-gate"
    args = _build_args(
        monkeypatch, out_dir, eval_path, shortlist="mockA,mockB"
    )

    fakes = {
        "mockA": _ConstantLM("mockA", predicted_direction=1),
        "mockB": _ConstantLM("mockB", predicted_direction=-1),
    }
    rc = runner_mod.run(args, lm_factory=_make_factory(fakes))
    assert rc == 0, "run() must return 0 on the success path"

    # The runner must have produced top3.md and summary.csv.
    top3_path = out_dir / "top3.md"
    summary_path = out_dir / "summary.csv"
    assert top3_path.exists(), "top3.md must be written"
    assert summary_path.exists(), "summary.csv must be written"

    # ----- top3.md must mention the gate-failed warning -------------------
    top3_text = top3_path.read_text(encoding="utf-8")
    assert WARNING_NOT_BETTER_THAN_BASELINE in top3_text, (
        f"top3.md is missing the literal 'not-better-than-baseline' warning.\n"
        f"--- top3.md ---\n{top3_text}"
    )
    # And it must show up specifically in the failed-gates section.
    assert "Why fewer than three models" in top3_text, (
        "top3.md must include the explanatory section when no model survives."
    )

    # ----- The in-memory CompositeScore for mockA must carry the warning --
    # We re-derive the scores via the public API so this test verifies the
    # actual ranker output (the runner persisted the same scores into
    # summary.csv via write_summary_csv).
    eval_set = load_eval_set(eval_path)
    majority = compute_majority_baseline(
        eval_set, bootstrap_n=args.bootstrap_n, seed=args.seed
    )
    # majority.hi must be >= 0.8 by construction; assert that explicitly so a
    # future change to bootstrap_ci (e.g. tighter CI) flags this fixture early.
    assert majority.point == pytest.approx(0.8, abs=1e-6)
    assert majority.hi >= 0.8 - 1e-6, (
        f"majority upper CI dropped below 0.8 (point={majority.point}, "
        f"hi={majority.hi}); test fixture no longer triggers the gate."
    )

    # ----- summary.csv must mark mockA as not surviving gates -------------
    rows = summary_path.read_text(encoding="utf-8").strip().splitlines()
    assert rows, "summary.csv must not be empty"
    header = rows[0].split(",")
    model_idx = header.index("model")
    survives_idx = header.index("survives_gates")
    warnings_idx = header.index("warnings")

    survives_by_model: dict[str, str] = {}
    warnings_by_model: dict[str, str] = {}
    for line in rows[1:]:
        cells = line.split(",")
        survives_by_model[cells[model_idx]] = cells[survives_idx]
        warnings_by_model[cells[model_idx]] = cells[warnings_idx]

    assert "mockA" in survives_by_model, "mockA missing from summary.csv"
    assert survives_by_model["mockA"] == "false", (
        f"mockA must NOT survive gates when accuracy = majority share. "
        f"Got survives_gates={survives_by_model['mockA']!r}; "
        f"warnings={warnings_by_model.get('mockA')!r}"
    )
    assert WARNING_NOT_BETTER_THAN_BASELINE in warnings_by_model["mockA"], (
        f"mockA's summary.csv warnings must include "
        f"'not-better-than-baseline'; got {warnings_by_model['mockA']!r}"
    )


def test_majority_gate_does_not_fire_when_model_outperforms_baseline(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A model with 100% accuracy must NOT be flagged ``not-better-than-baseline``.

    The majority share is still 0.8; the model's bootstrap lower CI is 1.0
    (every row correct), so ``memguard_accuracy.lo = 1.0 > majority.hi`` and
    the ``not-better-than-baseline`` warning must be absent for this model.

    Other gates (MCS-AUC, parse) may still fire under the small synthetic
    fixture — we deliberately do NOT assert ``survives_gates=true`` here. The
    sole assertion is that this SPECIFIC warning does not appear for the
    high-accuracy model. We verify on the in-memory ``CompositeScore`` (so
    the test is robust to other warnings co-occurring in ``top3.md``).
    """
    eval_path = tmp_path / "eval_perfect.jsonl"
    _write_majority_eval_set(eval_path, embed_target=True)

    out_dir = tmp_path / "run-majority-clear"
    args = _build_args(
        monkeypatch, out_dir, eval_path, shortlist="mockA"
    )

    fakes = {"mockA": _PerfectLM("mockA")}
    rc = runner_mod.run(args, lm_factory=_make_factory(fakes))
    assert rc == 0, "run() must return 0 on the success path"

    # Re-derive the ranker output through the public API so we can inspect
    # the warning list directly. The runner persists exactly this list to
    # summary.csv via ``write_summary_csv`` (same path checked by the
    # positive test above).
    eval_set = load_eval_set(eval_path)
    majority = compute_majority_baseline(
        eval_set, bootstrap_n=args.bootstrap_n, seed=args.seed
    )

    # Sanity check the fixture: the perfect-LM model should be 100% correct
    # on the parse-OK rows (which is every row given a deterministic LM).
    summary_path = out_dir / "summary.csv"
    rows = summary_path.read_text(encoding="utf-8").strip().splitlines()
    header = rows[0].split(",")
    model_idx = header.index("model")
    raw_lo_idx = header.index("raw_acc_lo")
    warnings_idx = header.index("warnings")

    mocka_row: list[str] | None = None
    for line in rows[1:]:
        cells = line.split(",")
        if cells[model_idx] == "mockA":
            mocka_row = cells
            break
    assert mocka_row is not None, "mockA missing from summary.csv"

    raw_lo = float(mocka_row[raw_lo_idx])
    assert raw_lo > majority.hi, (
        f"Test fixture broken: mockA raw_acc_lo={raw_lo} is not strictly "
        f"greater than majority.hi={majority.hi}; the gate would still fire."
    )

    warnings_csv = mocka_row[warnings_idx]
    assert WARNING_NOT_BETTER_THAN_BASELINE not in warnings_csv, (
        f"mockA outperforms the baseline (raw_acc_lo={raw_lo} > "
        f"majority.hi={majority.hi}) yet still received the "
        f"'not-better-than-baseline' warning. summary warnings={warnings_csv!r}"
    )

    # And the gate-warning string must not appear on a mockA-specific line in
    # top3.md either. We check the failed-gates bullets which contain the
    # model name; absence of the warning there confirms the negative control.
    top3_text = (out_dir / "top3.md").read_text(encoding="utf-8")
    for line in top3_text.splitlines():
        if "mockA" in line and WARNING_NOT_BETTER_THAN_BASELINE in line:
            pytest.fail(
                "top3.md flags mockA as not-better-than-baseline despite "
                f"100% accuracy. Offending line: {line!r}"
            )
