"""Tests for the harness.runner module — Task 5.1.

The runner orchestrates the full end-to-end run. These tests inject a fake
``lm_factory`` so no real HTTP traffic is made; they exercise:

* Req 1.5: ``--shortlist`` override skips smoke (no shortlist.json written,
  smoke_test never called).
* Req 2.5: ``assert_cutoff_safe`` aborts before any HTTP call when the eval
  set's cutoff_date precedes a model's cutoff.
* Req 3.4: ``ControlBaseline.is_calibrated == False`` results in a stub
  ``ModelEvalResult`` with the ``uncalibrated`` warning, surfaced through the
  ranker into ``summary.csv`` (``survives_gates=false``).
* Req 9.4: artifact paths printed at the end of a run.
* Req 10.1: the per-run manifest contains correct sha256 hashes for every
  input file.
* Req 10.3: the runner records the seed in the manifest.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from recall_guard.core.manifest import compute_file_hash, read_manifest
from recall_guard.core.nvidia_lm import CompletionResult, TokenLogprob

# Import target-under-test. The factory contract is documented in the task
# brief: ``lm_factory(api_key, model, timeout_s) -> NvidiaLM-like``.
from recall_guard.harness import runner as runner_mod  # noqa: E402

# --- Fixture paths -----------------------------------------------------------

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"
TINY_EVAL = FIXTURES_DIR / "tiny_eval.jsonl"
TINY_IS = FIXTURES_DIR / "tiny_is_memorized.jsonl"
TINY_OOS = FIXTURES_DIR / "tiny_oos_control.jsonl"
TINY_CUTOFFS = FIXTURES_DIR / "tiny_cutoffs.yaml"


# --- Fake LM machinery -------------------------------------------------------


def _make_top_logprobs() -> list[dict[str, Any]]:
    """Return a 20-element ``top_logprobs`` list as returned by NVIDIA."""
    return [{"token": f"tok{i}", "logprob": -1.0 - 0.1 * i} for i in range(20)]


def _make_logprobs(content: str) -> list[TokenLogprob]:
    """Build a deterministic logprobs list of length len(tokens) ≈ 5 for ``content``."""
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
        # NvidiaLM exposes these attributes; the runner reads ``model`` only.
        self.api_key = "fake"
        self.timeout_s = 15.0
        self.api_base = "fake://"
        self._cycle = direction_cycle or [1, -1, 0, 1, -1]
        self.calls: list[str] = []

    def generate(
        self, prompt: str, temperature: float = 0.0
    ) -> CompletionResult:
        idx = len(self.calls) % len(self._cycle)
        self.calls.append(prompt)
        # Vary direction with deterministic confidence so accuracy is non-trivial.
        return _make_completion(direction=self._cycle[idx], confidence=0.7)


def _make_factory(
    fakes: dict[str, _FakeLM],
    *,
    forbid_models: list[str] | None = None,
):
    """Return an ``lm_factory`` that yields the configured fake LMs."""
    forbidden = set(forbid_models or [])

    def factory(api_key: str, model: str, timeout_s: float) -> _FakeLM:
        if model in forbidden:
            raise AssertionError(
                f"factory was asked to construct forbidden model {model!r}"
            )
        if model not in fakes:
            # Lazily mint a fake so the runner can spin up reference / extra
            # models in tests that don't pre-register every ID.
            fakes[model] = _FakeLM(model=model)
        return fakes[model]

    return factory


# --- argparse parser test ----------------------------------------------------


def test_build_parser_has_required_flags() -> None:
    args = runner_mod.parse_argv(
        [
            "--eval-set",
            "x.jsonl",
            "--shortlist",
            "a,b",
            "--cutoffs",
            "c.yaml",
            "--out-dir",
            "/tmp/out",
        ]
    )
    assert args.eval_set == "x.jsonl"
    assert args.shortlist == "a,b"
    assert args.cutoffs == "c.yaml"
    assert args.out_dir == "/tmp/out"
    # Defaults should be exposed:
    assert args.seed == 0
    assert args.bootstrap_n == 1000


def test_parse_argv_rejects_nonpositive_bootstrap_n() -> None:
    with pytest.raises(SystemExit):
        runner_mod.parse_argv([
            "--eval-set", "x.jsonl",
            "--shortlist", "a",
            "--cutoffs", "c.yaml",
            "--bootstrap-n", "0",
        ])
    with pytest.raises(SystemExit):
        runner_mod.parse_argv([
            "--eval-set", "x.jsonl",
            "--shortlist", "a",
            "--cutoffs", "c.yaml",
            "--bootstrap-n", "-1",
        ])


# --- Run-success path --------------------------------------------------------


def _build_args(
    monkeypatch: pytest.MonkeyPatch,
    out_dir: Path,
    *,
    shortlist: str | None = "mockA,mockB",
    candidates: str | None = None,
    eval_set: Path = TINY_EVAL,
    is_memorized: Path = TINY_IS,
    oos_control: Path = TINY_OOS,
    cutoffs: Path = TINY_CUTOFFS,
    no_reference: bool = True,
    seed: int = 0,
    bootstrap_n: int = 50,
):
    """Build a parsed-argparse Namespace by going through ``build_parser``."""
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
    ]
    if shortlist is not None:
        cli += ["--shortlist", shortlist]
    if candidates is not None:
        cli += ["--candidates", candidates]
    if no_reference:
        cli += ["--no-reference"]
    return parser.parse_args(cli)


def test_run_writes_all_four_artifacts_with_shortlist(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Req 9.4 / 10.1: end-to-end run with --shortlist writes records, summary,
    top3, and manifest under out_dir.

    No shortlist.json is expected because smoke_test was skipped.
    """
    out_dir = tmp_path / "run1"
    args = _build_args(monkeypatch, out_dir)
    fakes = {"mockA": _FakeLM("mockA"), "mockB": _FakeLM("mockB")}
    factory = _make_factory(fakes)

    rc = runner_mod.run(args, lm_factory=factory)

    assert rc == 0
    assert (out_dir / "records.jsonl").exists()
    assert (out_dir / "summary.csv").exists()
    assert (out_dir / "top3.md").exists()
    assert (out_dir / "manifest.json").exists()
    # No smoke run -> no shortlist.json.
    assert not (out_dir / "shortlist.json").exists()


def test_run_returns_zero_on_success(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    out_dir = tmp_path / "rc0"
    args = _build_args(monkeypatch, out_dir)
    fakes = {"mockA": _FakeLM("mockA"), "mockB": _FakeLM("mockB")}
    rc = runner_mod.run(args, lm_factory=_make_factory(fakes))
    assert rc == 0


def test_run_returns_nonzero_on_missing_eval_set(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    out_dir = tmp_path / "missing"
    bogus = tmp_path / "does-not-exist.jsonl"
    args = _build_args(monkeypatch, out_dir, eval_set=bogus)
    fakes: dict[str, _FakeLM] = {}
    rc = runner_mod.run(args, lm_factory=_make_factory(fakes))
    assert rc != 0


def test_run_returns_nonzero_on_empty_shortlist(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    out_dir = tmp_path / "empty-shortlist"
    candidates = tmp_path / "candidates.txt"
    candidates.write_text("mockA\nmockB\n", encoding="utf-8")
    args = _build_args(monkeypatch, out_dir, shortlist=None, candidates=str(candidates))

    def fake_smoke_test(*args, **kwargs):
        return runner_mod.Shortlist(selected=[], outcomes=[])

    monkeypatch.setattr(runner_mod, "smoke_test", fake_smoke_test)
    rc = runner_mod.run(args, lm_factory=_make_factory({}))
    assert rc != 0
    assert not (out_dir / "summary.csv").exists()
    assert not (out_dir / "records.jsonl").exists()


def test_run_returns_nonzero_on_unexpected_model_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    out_dir = tmp_path / "unexpected-error"
    args = _build_args(monkeypatch, out_dir, shortlist="mockA")

    def boom(*args, **kwargs):
        raise TypeError("boom")

    monkeypatch.setattr(runner_mod, "_evaluate_one_model", boom)
    rc = runner_mod.run(args, lm_factory=_make_factory({"mockA": _FakeLM("mockA")}))
    assert rc != 0
    assert not (out_dir / "summary.csv").exists()


def test_run_returns_nonzero_on_missing_api_key(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """If NVIDIA_API_KEY is absent the runner exits non-zero before any work."""
    out_dir = tmp_path / "no-key"
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    # Disable .env auto-loading so the project's .env does not silently
    # restore the key the test is trying to prove must be missing.
    monkeypatch.setattr(runner_mod, "load_dotenv", lambda *a, **k: False)
    parser = runner_mod.build_parser()
    args = parser.parse_args(
        [
            "--eval-set",
            str(TINY_EVAL),
            "--shortlist",
            "mockA",
            "--cutoffs",
            str(TINY_CUTOFFS),
            "--out-dir",
            str(out_dir),
            "--no-reference",
        ]
    )
    fakes: dict[str, _FakeLM] = {}
    rc = runner_mod.run(args, lm_factory=_make_factory(fakes))
    assert rc != 0


# --- Cutoff guard ------------------------------------------------------------


def test_run_aborts_on_cutoff_violation_before_any_http_call(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Req 2.5: --shortlist contains a model whose cutoff post-dates the eval
    cutoff → ``assert_cutoff_safe`` raises ``CutoffViolation`` and the runner
    aborts BEFORE any HTTP call is made.
    """
    out_dir = tmp_path / "violation"
    args = _build_args(monkeypatch, out_dir, shortlist="late-cutoff-model")
    fake = _FakeLM("late-cutoff-model")
    rc = runner_mod.run(args, lm_factory=_make_factory({"late-cutoff-model": fake}))

    assert rc != 0
    # No HTTP call to the candidate model.
    assert fake.calls == []


# --- Uncalibrated handling ---------------------------------------------------


def test_run_skips_uncalibrated_models(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Req 3.4: when ``build_baseline`` returns ``is_calibrated=False`` for a
    model, the runner emits a stub ``ModelEvalResult`` with the
    ``uncalibrated`` warning. The model still appears in summary.csv with
    ``survives_gates=false``.
    """
    out_dir = tmp_path / "uncal"
    args = _build_args(monkeypatch, out_dir)
    fakes = {"mockA": _FakeLM("mockA"), "mockB": _FakeLM("mockB")}

    # Patch build_baseline to always report uncalibrated for mockA and
    # calibrated for mockB.
    from recall_guard.mia.control import ControlBaseline

    def fake_build_baseline(model_lm, control_rows, ref_lm, min_valid=50, max_workers=1):
        return ControlBaseline(
            model=model_lm.model,
            n_valid=0 if model_lm.model == "mockA" else 60,
            feature_means={
                "loss": None if model_lm.model == "mockA" else 0.5,
                "min_k": None if model_lm.model == "mockA" else 0.5,
                "min_k_pp": None if model_lm.model == "mockA" else 0.5,
                "zlib_ratio": None if model_lm.model == "mockA" else 1.0,
                "ref_delta": None,
            },
            feature_stds={
                "loss": None if model_lm.model == "mockA" else 0.1,
                "min_k": None if model_lm.model == "mockA" else 0.1,
                "min_k_pp": None if model_lm.model == "mockA" else 0.1,
                "zlib_ratio": None if model_lm.model == "mockA" else 0.1,
                "ref_delta": None,
            },
            is_calibrated=(model_lm.model != "mockA"),
            min_valid=min_valid,
        )

    monkeypatch.setattr(runner_mod, "build_baseline", fake_build_baseline)

    rc = runner_mod.run(args, lm_factory=_make_factory(fakes))
    assert rc == 0

    # Summary must include mockA with survives_gates=false.
    rows = (out_dir / "summary.csv").read_text(encoding="utf-8").strip().splitlines()
    assert rows, "summary.csv should not be empty"
    header = rows[0].split(",")
    survives_idx = header.index("survives_gates")
    warnings_idx = header.index("warnings")
    model_idx = header.index("model")
    mock_a_row: list[str] | None = None
    for line in rows[1:]:
        cells = line.split(",")
        if cells[model_idx] == "mockA":
            mock_a_row = cells
            break
    assert mock_a_row is not None, "mockA must appear in summary.csv"
    assert mock_a_row[survives_idx] == "false"
    assert "uncalibrated" in mock_a_row[warnings_idx]


# --- Smoke skip on shortlist override ---------------------------------------


def test_run_honors_shortlist_override_skips_smoke(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Req 1.5: --shortlist makes the runner skip smoke_test entirely."""
    out_dir = tmp_path / "skip-smoke"
    args = _build_args(monkeypatch, out_dir)

    smoke_called = {"count": 0}

    def fake_smoke_test(*pargs, **kwargs):  # pragma: no cover - assertion
        smoke_called["count"] += 1
        raise AssertionError("smoke_test must not be called when --shortlist is set")

    monkeypatch.setattr(runner_mod, "smoke_test", fake_smoke_test)

    fakes = {"mockA": _FakeLM("mockA"), "mockB": _FakeLM("mockB")}
    rc = runner_mod.run(args, lm_factory=_make_factory(fakes))
    assert rc == 0
    assert smoke_called["count"] == 0
    assert not (out_dir / "shortlist.json").exists()


# --- Reference-model wiring --------------------------------------------------


def test_run_with_no_reference_does_not_construct_ref_model(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """With --no-reference the lm_factory is called for every shortlisted
    model BUT not for the reference model.
    """
    out_dir = tmp_path / "no-ref"
    args = _build_args(monkeypatch, out_dir, no_reference=True)
    fakes = {"mockA": _FakeLM("mockA"), "mockB": _FakeLM("mockB")}
    # Mark "meta/llama-3.2-1b-instruct" as forbidden — the factory will raise
    # if the runner tries to construct it.
    factory = _make_factory(fakes, forbid_models=["meta/llama-3.2-1b-instruct"])

    rc = runner_mod.run(args, lm_factory=factory)
    assert rc == 0
    # Each candidate was used at least once.
    assert fakes["mockA"].calls
    assert fakes["mockB"].calls


# --- Manifest contents -------------------------------------------------------


def test_run_writes_manifest_with_correct_hashes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    out_dir = tmp_path / "manifest"
    args = _build_args(monkeypatch, out_dir, seed=42, bootstrap_n=25)
    fakes = {"mockA": _FakeLM("mockA"), "mockB": _FakeLM("mockB")}

    rc = runner_mod.run(args, lm_factory=_make_factory(fakes))
    assert rc == 0

    manifest = read_manifest(out_dir / "manifest.json")
    assert manifest.seed == 42
    assert manifest.bootstrap_n == 25
    assert manifest.eval_set_hash == compute_file_hash(TINY_EVAL)
    assert manifest.is_memorized_hash == compute_file_hash(TINY_IS)
    assert manifest.control_corpus_hash == compute_file_hash(TINY_OOS)
    assert manifest.cutoffs_hash == compute_file_hash(TINY_CUTOFFS)
    assert sorted(manifest.shortlist) == ["mockA", "mockB"]
    # Composite-score formula recorded for reproducibility (Req 8.4).
    assert "formula" in manifest.composite_score
    # Artifact paths recorded.
    assert "records" in manifest.artifacts
    assert "summary" in manifest.artifacts
    assert "top3" in manifest.artifacts


# --- Records.jsonl integrity ------------------------------------------------


def test_run_writes_one_record_per_eval_row_per_model(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Records.jsonl must contain one line per (model, eval_row).

    The tiny eval file has 5 rows; with two shortlisted models that's 10 lines.
    """
    out_dir = tmp_path / "records"
    args = _build_args(monkeypatch, out_dir)
    fakes = {"mockA": _FakeLM("mockA"), "mockB": _FakeLM("mockB")}

    rc = runner_mod.run(args, lm_factory=_make_factory(fakes))
    assert rc == 0

    text = (out_dir / "records.jsonl").read_text(encoding="utf-8").strip()
    if not text:
        lines: list[str] = []
    else:
        lines = text.splitlines()
    # 5 eval rows * 2 models = 10 records.
    assert len(lines) == 10
    # Every line must be valid JSON with a model field.
    for line in lines:
        obj = json.loads(line)
        assert "model" in obj
        assert obj["model"] in {"mockA", "mockB"}


# --- Smoke help-only path ----------------------------------------------------


def test_build_parser_help_does_not_raise(capsys: pytest.CaptureFixture) -> None:
    """``--help`` must succeed (argparse exits with SystemExit code 0)."""
    parser = runner_mod.build_parser()
    with pytest.raises(SystemExit) as exc_info:
        parser.parse_args(["--help"])
    assert exc_info.value.code == 0
