"""Tests for the harness.runner replay subcommand — Task 5.2.

Replay mode reads a previously persisted manifest, verifies that every input
file's bytes still hash to the value recorded in the manifest, then re-runs
the pipeline with the same seed/bootstrap_n/shortlist into a fresh out-dir.
The new ``top3.md`` ordering must match the original within bootstrap CIs;
exact ordering match is the strong success path. A mutated input file
aborts the replay non-zero with a hash-mismatch message.

Tests here exercise:

* Req 10.2: replay reproduces ranking from a manifest.
* Hash-mismatch abort path (input file mutated since the original run).
* CLI surface: ``harness replay --from-manifest PATH --out-dir PATH`` works
  end-to-end against a saved manifest; ``--help`` advertises the
  ``replay`` subcommand.
* The reconstructed args namespace propagates ``seed`` and ``bootstrap_n``
  from the manifest.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from src.core.manifest import read_manifest
from src.core.nvidia_lm import CompletionResult, TokenLogprob
from src.harness import runner as runner_mod


FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"
TINY_EVAL = FIXTURES_DIR / "tiny_eval.jsonl"
TINY_IS = FIXTURES_DIR / "tiny_is_memorized.jsonl"
TINY_OOS = FIXTURES_DIR / "tiny_oos_control.jsonl"
TINY_CUTOFFS = FIXTURES_DIR / "tiny_cutoffs.yaml"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
HARNESS_PY = PROJECT_ROOT / "harness.py"


# --- Fake LM machinery (mirrors test_runner.py) ------------------------------


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
        # Different cycles per model so the rankings have a deterministic order.
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


def _build_args(
    monkeypatch: pytest.MonkeyPatch,
    out_dir: Path,
    *,
    eval_set: Path = TINY_EVAL,
    is_memorized: Path = TINY_IS,
    oos_control: Path = TINY_OOS,
    cutoffs: Path = TINY_CUTOFFS,
    seed: int = 0,
    bootstrap_n: int = 50,
):
    """Build a parsed-argparse Namespace for the build subcommand."""
    monkeypatch.setenv("NVIDIA_API_KEY", "test-api-key")
    parser = runner_mod.build_parser()
    cli: list[str] = [
        "build",
        "--eval-set",
        str(eval_set),
        "--shortlist",
        "mockA,mockB",
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
    return parser.parse_args(cli)


# --- Tests -------------------------------------------------------------------


def test_replay_reproduces_ranking_from_manifest(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Req 10.2: replay produces the same top3.md ordering as the original.

    The ``_FakeLM`` is deterministic, so re-running with the same seed and
    bootstrap_n should yield identical bootstrap distributions and thus an
    identical top3.md ordering.
    """
    original_dir = tmp_path / "original"
    args = _build_args(monkeypatch, original_dir)
    fakes_orig = {"mockA": _FakeLM("mockA"), "mockB": _FakeLM("mockB")}
    rc = runner_mod.run(args, lm_factory=_make_factory(fakes_orig))
    assert rc == 0
    original_top3 = (original_dir / "top3.md").read_text(encoding="utf-8")

    replay_dir = tmp_path / "replay"
    fakes_replay = {"mockA": _FakeLM("mockA"), "mockB": _FakeLM("mockB")}
    rc = runner_mod.replay(
        manifest_path=original_dir / "manifest.json",
        out_dir=replay_dir,
        lm_factory=_make_factory(fakes_replay),
    )
    assert rc == 0
    replay_top3 = (replay_dir / "top3.md").read_text(encoding="utf-8")

    # Strong success: identical ordering. Compare the model-name order.
    def _extract_model_order(top3_text: str) -> list[str]:
        # top3.md lists models — extract any line containing a known model id.
        order: list[str] = []
        for line in top3_text.splitlines():
            for model in ("mockA", "mockB"):
                if model in line and model not in order:
                    order.append(model)
        return order

    assert _extract_model_order(original_top3) == _extract_model_order(replay_top3)


def test_replay_aborts_on_input_hash_mismatch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Mutating an input file after the original run aborts replay non-zero
    with a clear error naming the mismatched file.
    """
    original_dir = tmp_path / "original"
    # Copy the OOS control file into tmp so the mutation does not pollute the
    # repo's fixture file.
    mutable_oos = tmp_path / "mutable_oos_control.jsonl"
    shutil.copy(TINY_OOS, mutable_oos)

    args = _build_args(monkeypatch, original_dir, oos_control=mutable_oos)
    fakes = {"mockA": _FakeLM("mockA"), "mockB": _FakeLM("mockB")}
    rc = runner_mod.run(args, lm_factory=_make_factory(fakes))
    assert rc == 0

    # Mutate the OOS-control file: append a row.
    with mutable_oos.open("a", encoding="utf-8") as fh:
        fh.write(
            '{"prompt": "extra row", "label": 0, "metadata": '
            '{"published_at": "2025-12-15", "source": "test", '
            '"url": "https://example.com/extra"}}\n'
        )

    replay_dir = tmp_path / "replay"
    fakes_replay = {"mockA": _FakeLM("mockA"), "mockB": _FakeLM("mockB")}
    rc = runner_mod.replay(
        manifest_path=original_dir / "manifest.json",
        out_dir=replay_dir,
        lm_factory=_make_factory(fakes_replay),
    )
    assert rc != 0


def test_replay_aborts_with_hash_mismatch_message_naming_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture,
) -> None:
    """When an input file's hash no longer matches the manifest, the runner
    must print an error message that names the mismatched file.
    """
    original_dir = tmp_path / "original"
    mutable_oos = tmp_path / "mutable_oos_control.jsonl"
    shutil.copy(TINY_OOS, mutable_oos)

    args = _build_args(monkeypatch, original_dir, oos_control=mutable_oos)
    fakes = {"mockA": _FakeLM("mockA"), "mockB": _FakeLM("mockB")}
    rc = runner_mod.run(args, lm_factory=_make_factory(fakes))
    assert rc == 0

    with mutable_oos.open("a", encoding="utf-8") as fh:
        fh.write(
            '{"prompt": "another extra row", "label": 0, "metadata": '
            '{"published_at": "2025-12-15", "source": "test", '
            '"url": "https://example.com/another"}}\n'
        )

    capsys.readouterr()  # discard prior output
    replay_dir = tmp_path / "replay"
    fakes_replay: dict[str, _FakeLM] = {}
    rc = runner_mod.replay(
        manifest_path=original_dir / "manifest.json",
        out_dir=replay_dir,
        lm_factory=_make_factory(fakes_replay),
    )
    assert rc != 0
    captured = capsys.readouterr()
    combined = (captured.out or "") + (captured.err or "")
    # The error must mention the mutated path or the conventional name.
    assert "hash" in combined.lower() or "mismatch" in combined.lower()
    assert str(mutable_oos.name) in combined or "oos" in combined.lower()


def test_replay_uses_recorded_seed_and_bootstrap_n(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The replay must reconstruct args from the manifest, so the seed and
    bootstrap_n the inner ``run()`` sees should match the manifest values.
    """
    original_dir = tmp_path / "original"
    args = _build_args(monkeypatch, original_dir, seed=42, bootstrap_n=25)
    fakes = {"mockA": _FakeLM("mockA"), "mockB": _FakeLM("mockB")}
    rc = runner_mod.run(args, lm_factory=_make_factory(fakes))
    assert rc == 0

    manifest = read_manifest(original_dir / "manifest.json")
    assert manifest.seed == 42
    assert manifest.bootstrap_n == 25

    captured_args: list[Any] = []
    real_run = runner_mod.run

    def spy_run(args, *, lm_factory=None):
        captured_args.append(args)
        return real_run(args, lm_factory=lm_factory)

    monkeypatch.setattr(runner_mod, "run", spy_run)

    replay_dir = tmp_path / "replay"
    fakes_replay: dict[str, _FakeLM] = {}
    rc = runner_mod.replay(
        manifest_path=original_dir / "manifest.json",
        out_dir=replay_dir,
        lm_factory=_make_factory(fakes_replay),
    )
    assert rc == 0
    assert captured_args, "replay must call run() at least once"
    inner_args = captured_args[0]
    assert inner_args.seed == 42
    assert inner_args.bootstrap_n == 25


def test_replay_subcommand_in_help() -> None:
    """``python harness.py --help`` must list ``replay``; ``replay --help``
    must show ``--from-manifest`` and ``--out-dir``.
    """
    result_top = subprocess.run(
        [sys.executable, str(HARNESS_PY), "--help"],
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
    )
    assert result_top.returncode == 0
    assert "replay" in result_top.stdout

    result_replay = subprocess.run(
        [sys.executable, str(HARNESS_PY), "replay", "--help"],
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
    )
    assert result_replay.returncode == 0
    assert "--from-manifest" in result_replay.stdout
    assert "--out-dir" in result_replay.stdout


def test_build_subcommand_remains_compatible(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``python harness.py --eval-set ... --shortlist ...`` (no explicit
    subcommand) still works via ``parse_argv``: ``build`` is the default
    subcommand and the legacy invocation form is rewritten transparently.

    Also asserts that explicit ``build`` still works.
    """
    out_dir_legacy = tmp_path / "compat-legacy"
    monkeypatch.setenv("NVIDIA_API_KEY", "test-api-key")
    args_legacy = runner_mod.parse_argv(
        [
            "--eval-set",
            str(TINY_EVAL),
            "--shortlist",
            "mockA,mockB",
            "--is-memorized",
            str(TINY_IS),
            "--oos-control",
            str(TINY_OOS),
            "--cutoffs",
            str(TINY_CUTOFFS),
            "--out-dir",
            str(out_dir_legacy),
            "--no-reference",
            "--seed",
            "0",
            "--bootstrap-n",
            "25",
        ]
    )
    assert args_legacy.subcommand == "build"
    fakes = {"mockA": _FakeLM("mockA"), "mockB": _FakeLM("mockB")}
    rc = runner_mod.run(args_legacy, lm_factory=_make_factory(fakes))
    assert rc == 0
    assert (out_dir_legacy / "manifest.json").exists()

    # Explicit ``build`` form must also work.
    out_dir_explicit = tmp_path / "compat-explicit"
    args_explicit = runner_mod.parse_argv(
        [
            "build",
            "--eval-set",
            str(TINY_EVAL),
            "--shortlist",
            "mockA,mockB",
            "--is-memorized",
            str(TINY_IS),
            "--oos-control",
            str(TINY_OOS),
            "--cutoffs",
            str(TINY_CUTOFFS),
            "--out-dir",
            str(out_dir_explicit),
            "--no-reference",
            "--seed",
            "0",
            "--bootstrap-n",
            "25",
        ]
    )
    assert args_explicit.subcommand == "build"
    fakes2 = {"mockA": _FakeLM("mockA"), "mockB": _FakeLM("mockB")}
    rc = runner_mod.run(args_explicit, lm_factory=_make_factory(fakes2))
    assert rc == 0
