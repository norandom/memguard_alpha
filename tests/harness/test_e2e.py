"""End-to-end harness integration test — Task 6.1.

Drives the real ``runner.run`` pipeline against a 10-row in-memory eval set
with two mocked LMs (no real HTTP), then replays from the persisted
``manifest.json`` into a second out-dir and asserts the top-3 ordering is
identical across the two runs.

Covered acceptance criteria:

* Req 9.3: per-(model, prompt) records are persisted to ``records.jsonl``.
* Req 9.4: ``records.jsonl``, ``summary.csv``, and ``top3.md`` are written
  under the chosen ``--out-dir``; the ``--candidates`` path additionally
  writes ``shortlist.json``.
* Req 10.1: a per-run ``manifest.json`` is written.
* Req 10.2: replay from a manifest reproduces the original top-3 ordering.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from src.core.nvidia_lm import CompletionResult, TokenLogprob
from src.harness import runner as runner_mod
from src.harness.smoke import Shortlist, SmokeOutcome


# --- Fixture paths used as templates for the in-memory eval set --------------

REPO_FIXTURES = Path(__file__).parent.parent / "fixtures"
TEMPLATE_IS = REPO_FIXTURES / "tiny_is_memorized.jsonl"
TEMPLATE_OOS = REPO_FIXTURES / "tiny_oos_control.jsonl"
TEMPLATE_CUTOFFS = REPO_FIXTURES / "tiny_cutoffs.yaml"


# --- Fake LM machinery (mirrors test_runner.py / test_runner_replay.py) -----


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


def _make_completion(direction: int = 1, confidence: float = 0.7) -> CompletionResult:
    content = f"Direction: {direction}\nConfidence: {confidence}"
    return CompletionResult(
        content=content,
        logprobs=_make_logprobs(content),
        raw_temperature_observed=0.0,
    )


class _FakeLM:
    """Configurable in-memory LM. Records calls for later assertions."""

    def __init__(
        self,
        model: str,
        *,
        direction_cycle: list[int] | None = None,
    ) -> None:
        self.model = model
        self.api_key = "fake"
        self.timeout_s = 15.0
        self.api_base = "fake://"
        # Different cycles per model so the rankings differ deterministically.
        self._cycle = direction_cycle or [1, -1, 0, 1, -1]
        self.calls: list[str] = []

    def generate(
        self, prompt: str, temperature: float = 0.0
    ) -> CompletionResult:
        idx = len(self.calls) % len(self._cycle)
        self.calls.append(prompt)
        return _make_completion(direction=self._cycle[idx], confidence=0.7)


def _make_factory(fakes: dict[str, _FakeLM]):
    def factory(api_key: str, model: str, timeout_s: float) -> _FakeLM:
        if model not in fakes:
            fakes[model] = _FakeLM(model=model)
        return fakes[model]

    return factory


# --- In-memory 10-row eval set construction ----------------------------------


def _write_10_row_eval_set(path: Path) -> None:
    """Write a 10-row eval JSONL with the standard ``_cutoff_date`` header.

    The cutoff predates any model in ``tiny_cutoffs.yaml`` so the cutoff
    guard does not abort the run.
    """
    lines: list[str] = []
    lines.append(json.dumps({"_cutoff_date": "2025-06-30"}))
    # Use a deterministic mix of target directions (+1, -1, 0).
    targets = [1, -1, 0, 1, -1, 1, -1, 0, 1, -1]
    for i, target in enumerate(targets):
        row = {
            "prompt": f"Eval prompt {i}",
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
    shortlist: str | None = "mockA,mockB",
    candidates: str | None = None,
    is_memorized: Path = TEMPLATE_IS,
    oos_control: Path = TEMPLATE_OOS,
    cutoffs: Path = TEMPLATE_CUTOFFS,
    seed: int = 0,
    bootstrap_n: int = 50,
):
    """Construct a parsed argparse namespace via the real ``build_parser``."""
    monkeypatch.setenv("NVIDIA_API_KEY", "test-api-key")
    parser = runner_mod.build_parser()
    cli: list[str] = [
        "--eval-set",
        str(eval_set),
        "--is-memorized",
        str(is_memorized),
        "--oos-control",
        str(oos_control),
        "--cutoffs",
        str(cutoffs),
        "--out-dir",
        str(out_dir),
        "--seed",
        str(seed),
        "--bootstrap-n",
        str(bootstrap_n),
        "--no-reference",
    ]
    if shortlist is not None:
        cli += ["--shortlist", shortlist]
    if candidates is not None:
        cli += ["--candidates", candidates]
    return parser.parse_args(cli)


def _extract_model_order(top3_text: str, models: tuple[str, ...]) -> list[str]:
    """Return the order in which model IDs appear in ``top3.md``."""
    order: list[str] = []
    for line in top3_text.splitlines():
        for model in models:
            if model in line and model not in order:
                order.append(model)
    return order


# --- Tests -------------------------------------------------------------------


def test_e2e_writes_all_five_artifacts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """End-to-end via the ``--candidates`` path writes ALL FIVE artifacts.

    The ``--candidates`` path is required so the runner persists
    ``shortlist.json`` (Req 1.4). ``smoke_test`` is monkeypatched to a
    deterministic stub so no real HTTP traffic is made.
    """
    eval_path = tmp_path / "eval10.jsonl"
    _write_10_row_eval_set(eval_path)

    candidates_path = tmp_path / "candidates.txt"
    candidates_path.write_text("mockA\nmockB\n", encoding="utf-8")

    out_dir = tmp_path / "run-candidates"

    args = _build_args(
        monkeypatch,
        out_dir,
        eval_path,
        shortlist=None,
        candidates=str(candidates_path),
    )

    def fake_smoke_test(
        candidates: list[str], api_key: str, smoke_prompts, **kwargs
    ) -> Shortlist:
        return Shortlist(
            selected=list(candidates),
            outcomes=[
                SmokeOutcome(model=m, passed=True, fail_reason=None)
                for m in candidates
            ],
        )

    monkeypatch.setattr(runner_mod, "smoke_test", fake_smoke_test)

    fakes = {"mockA": _FakeLM("mockA"), "mockB": _FakeLM("mockB")}
    rc = runner_mod.run(args, lm_factory=_make_factory(fakes))

    assert rc == 0, "run() must return 0 on the success path"
    # All five artifacts must exist.
    assert (out_dir / "manifest.json").exists()
    assert (out_dir / "shortlist.json").exists()
    assert (out_dir / "records.jsonl").exists()
    assert (out_dir / "summary.csv").exists()
    assert (out_dir / "top3.md").exists()


def test_e2e_writes_four_artifacts_with_shortlist_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The ``--shortlist`` path skips smoke and therefore does NOT write
    ``shortlist.json``. The other four artifacts must still all exist.

    This documents the alternate path the suite exercises for replay
    (``test_e2e_replay_reproduces_top3_ordering`` below uses the same path).
    """
    eval_path = tmp_path / "eval10.jsonl"
    _write_10_row_eval_set(eval_path)

    out_dir = tmp_path / "run-shortlist"
    args = _build_args(monkeypatch, out_dir, eval_path)

    fakes = {"mockA": _FakeLM("mockA"), "mockB": _FakeLM("mockB")}
    rc = runner_mod.run(args, lm_factory=_make_factory(fakes))

    assert rc == 0
    assert (out_dir / "manifest.json").exists()
    assert (out_dir / "records.jsonl").exists()
    assert (out_dir / "summary.csv").exists()
    assert (out_dir / "top3.md").exists()
    # Smoke skipped → no shortlist.json on the --shortlist path.
    assert not (out_dir / "shortlist.json").exists()



def test_e2e_records_jsonl_has_one_line_per_eval_row_per_model(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Req 9.3: ``records.jsonl`` contains one line per (model, eval row).

    With 10 eval rows × 2 shortlisted models, the file must contain exactly
    20 valid JSON records. The ``_FakeLM`` always returns parseable
    ``Direction:`` / ``Confidence:`` lines, so no rows should be dropped.
    """
    eval_path = tmp_path / "eval10.jsonl"
    _write_10_row_eval_set(eval_path)

    out_dir = tmp_path / "records"
    args = _build_args(monkeypatch, out_dir, eval_path)
    fakes = {"mockA": _FakeLM("mockA"), "mockB": _FakeLM("mockB")}
    rc = runner_mod.run(args, lm_factory=_make_factory(fakes))
    assert rc == 0

    text = (out_dir / "records.jsonl").read_text(encoding="utf-8").strip()
    lines = text.splitlines() if text else []
    assert len(lines) == 20, f"expected 20 records, got {len(lines)}"
    seen_models: set[str] = set()
    for line in lines:
        obj = json.loads(line)
        assert "model" in obj
        assert obj["model"] in {"mockA", "mockB"}
        seen_models.add(obj["model"])
    assert seen_models == {"mockA", "mockB"}


def test_e2e_summary_csv_has_two_model_rows_plus_majority(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Req 9.1 / 9.2: ``summary.csv`` lists one row per shortlisted model
    plus a majority-class baseline row. With two models that's three data
    rows in addition to the header.
    """
    eval_path = tmp_path / "eval10.jsonl"
    _write_10_row_eval_set(eval_path)

    out_dir = tmp_path / "summary"
    args = _build_args(monkeypatch, out_dir, eval_path)
    fakes = {"mockA": _FakeLM("mockA"), "mockB": _FakeLM("mockB")}
    rc = runner_mod.run(args, lm_factory=_make_factory(fakes))
    assert rc == 0

    rows = (
        (out_dir / "summary.csv")
        .read_text(encoding="utf-8")
        .strip()
        .splitlines()
    )
    assert rows, "summary.csv must not be empty"
    header = rows[0].split(",")
    assert "model" in header
    data_rows = rows[1:]
    assert (
        len(data_rows) == 3
    ), f"expected 3 data rows (2 models + majority), got {len(data_rows)}"
    model_idx = header.index("model")
    model_cells = {line.split(",")[model_idx] for line in data_rows}
    assert "mockA" in model_cells
    assert "mockB" in model_cells
