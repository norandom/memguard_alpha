"""Operator script modes run offline: docker and uv are stubs that only record their arguments.

The stubs never read the root .env; the tests check only that its path is passed.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/modal_operator.sh"
pytestmark = pytest.mark.skipif(
    sys.platform == "win32" or shutil.which("sh") is None,
    reason="the operator script runs in a Linux container and is tested on POSIX hosts",
)

# Records the docker arguments, then runs the container command line locally.
DOCKER = """#!/bin/sh
printf '%s\\n' "$@" > "$STUB_LOG.docker"
while [ "$#" -gt 0 ]; do
  if [ "$1" = sh ] && [ "$2" = -c ]; then shift 2; script=$1; shift; exec sh -c "$script" "$@"; fi
  shift
done
exit 97
"""
# Records each uv call and the HF_TOKEN value the container command exported.
UV = """#!/bin/sh
printf '%s\\n' "$*" >> "$STUB_LOG.uv"
printf 'hf=[%s]\\n' "${HF_TOKEN:-}" >> "$STUB_LOG.uv"
"""


def operator(tmp_path, *args, env=None):
    stubs = tmp_path / "bin"
    stubs.mkdir(exist_ok=True)
    for name, body in (("docker", DOCKER), ("uv", UV)):
        path = stubs / name
        path.write_text(body, encoding="utf-8", newline="\n")
        path.chmod(0o755)
    log = tmp_path / "stub"
    docker = Path(f"{log}.docker")
    uv = Path(f"{log}.uv")
    docker.unlink(missing_ok=True)
    uv.unlink(missing_ok=True)
    environment = {"PATH": f"{stubs}{os.pathsep}{os.environ['PATH']}", "STUB_LOG": str(log)} | (env or {})
    result = subprocess.run(["sh", str(SCRIPT), *args], capture_output=True, text=True, env=environment, timeout=60)
    return (result, docker.read_text().splitlines() if docker.exists() else [],
            [line for line in uv.read_text().splitlines() if not line.startswith("hf=")] if uv.exists() else [],
            uv.read_text() if uv.exists() else "")


def run_line(*arguments):
    return " ".join(("run", "--project", "deploy/modal", "--no-sync", "python", *arguments))


def env_file(docker):
    return docker[docker.index("--env-file") + 1] if "--env-file" in docker else None


def test_script_parses():
    result = subprocess.run(["sh", "-n", str(SCRIPT)], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("mode", ["stage", "serve"])
def test_stage_and_serve_pass_the_root_env_file_and_their_limits(tmp_path, mode):
    result, docker, uv, _ = operator(tmp_path, mode, "configs/modal/gemma-3-4b-measurement.json",
                                     "runs/finance-sim/modal/limits.json", "runs/finance-sim/modal/receipt.json")
    assert result.returncode == 0, result.stderr
    assert env_file(docker) == f"{ROOT}/.env"
    assert uv == ["sync --project deploy/modal --locked --quiet", run_line(
        "scripts/serve_modal_model.py", mode, "--manifest", "configs/modal/gemma-3-4b-measurement.json",
        "--stage-limits", "runs/finance-sim/modal/limits.json", "--receipt", "runs/finance-sim/modal/receipt.json")]


def test_stop_serve_passes_the_root_env_file_and_the_serve_receipt(tmp_path):
    result, docker, uv, _ = operator(tmp_path, "stop-serve", "configs/modal/gemma-3-4b-measurement.json",
                                     "runs/finance-sim/modal/receipt.json")
    assert result.returncode == 0, result.stderr
    assert env_file(docker) == f"{ROOT}/.env"
    assert uv[1] == run_line("scripts/serve_modal_model.py", "stop-serve", "--manifest",
                             "configs/modal/gemma-3-4b-measurement.json", "--receipt", "runs/finance-sim/modal/receipt.json")


@pytest.mark.parametrize("args", [
    ("stage", "m.json", "l.json"), ("serve", "m.json", "l.json", "r.json", "extra"), ("stop-serve", "m.json"),
    ("stop-serve", "m.json", "l.json", "r.json"),
])
def test_wrong_arity_exits_before_running_python(tmp_path, args):
    result, _, uv, _ = operator(tmp_path, *args)
    assert result.returncode == 2 and "needs" in result.stderr
    assert uv == ["sync --project deploy/modal --locked --quiet"]


@pytest.mark.parametrize("extra,expected", [
    ((), ()),
    (("approval.json",), ("--approval", "approval.json")),
    (("--modal-manifest", "configs/modal/gemma-3-4b-measurement.json"),
     ("--modal-manifest", "configs/modal/gemma-3-4b-measurement.json")),
    (("approval.json", "--modal-manifest", "configs/modal/gemma-3-4b-measurement.json"),
     ("--approval", "approval.json", "--modal-manifest", "configs/modal/gemma-3-4b-measurement.json")),
])
def test_measure_plan_passes_an_optional_modal_manifest_and_reads_no_credentials(tmp_path, extra, expected):
    result, docker, uv, _ = operator(tmp_path, "measure-plan", "plan.json", "manifest.json", "corpus.json", *extra)
    assert result.returncode == 0, result.stderr
    assert env_file(docker) is None
    assert uv[1] == run_line("scripts/measure_modal_model.py", "--plan", "plan.json", "--manifest", "manifest.json",
                             "--corpus", "corpus.json", *expected)


@pytest.mark.parametrize("extra", [("--modal-manifest",), ("approval.json", "--modal-manifest"),
                                   ("approval.json", "extra"), ("--other", "path")])
def test_measure_plan_refuses_anything_but_a_trailing_modal_manifest(tmp_path, extra):
    result, _, uv, _ = operator(tmp_path, "measure-plan", "plan.json", "manifest.json", "corpus.json", *extra)
    assert result.returncode == 2 and "--modal-manifest" in result.stderr
    assert len(uv) == 1


def test_credentials_are_whitespace_trimmed_by_name_including_hf_token(tmp_path):
    result, _, _, log = operator(tmp_path, "stage", "m.json", "l.json", "r.json",
                                 env={"HF_TOKEN": " not-a-real-token\n\t"})
    assert result.returncode == 0, result.stderr
    assert "hf=[not-a-real-token]" in log and "hf=[ not-a-real-token" not in log


def test_existing_modes_are_unchanged(tmp_path):
    result, docker, uv, _ = operator(tmp_path, "execute-measurement", "p", "m", "c", "a", "j", "r")
    assert result.returncode == 0 and env_file(docker) == f"{ROOT}/.env"
    assert uv[1].split() == run_line("scripts/measure_modal_model.py", "--execute", "--plan", "p", "--manifest", "m",
                                     "--corpus", "c", "--approval", "a", "--journal", "j", "--receipt", "r").split()
    result, docker, uv, _ = operator(tmp_path, "stop", "ap-" + "a" * 22)
    assert result.returncode == 0 and uv[1] == run_line("-m", "modal", "app", "stop", "ap-" + "a" * 22, "--yes")
