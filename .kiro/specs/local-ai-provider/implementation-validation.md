# Local-provider integration validation

Validation date: 2026-11-05. Task: 6.2. Language: English.

## Scope and source

Validated implementation commit: `74d0f2b7934d6491f480f3fbce40d94243bfe3c0`.
The working tree was clean before validation. After the initial full regression
exposed a CI fixture portability defect, the parent authorized the minimal test
repair described below. Runtime source, docs, task checkboxes and spec metadata
remain unchanged.
This is nonbehavioral integration validation, so no feature flag or artificial
RED test was introduced. Parent-reviewed focused receipts remain source-matched;
the coherent full regression below subsumes those tests.

The task claim is limited to available Ollama/provider integration and an explicit
assessment of the blocked NPU branch. It is not an overall NPU feature GO.

## Commands and fresh results

All repository commands ran through PowerShell. WSL was used only to invoke
Dagger CI. Dagger v0.21.8 used the existing `ci/src/ci/main.py` and root Dockerfile,
locked dependencies and Python 3.12.12. The Dagger command prefix below is:

```powershell
wsl -d Debian --cd /mnt/d/Source/memguard_alpha --exec /home/mc/.local/bin/dagger
```

| Check | Command after prefix (or PowerShell command) | Exit and result | Log under `runs/local-provider/validation/` |
| --- | --- | --- | --- |
| Full offline regression | `-m ci call test --source=. --python=3.12` | 1; 2710 passed, 12 setup errors in 313.62 seconds | `dagger-test.log` |
| Post-repair full offline regression | `-m ci call test --source=. --python=3.12` | 0; 2722 passed in 306.08 seconds, no skips/errors/failures | `dagger-test-repaired.log` |
| Strict documentation | `-m ci call docs --source=. export --path=runs/local-provider/validation/site` | 0; MkDocs strict build, documentation built in 8.97 seconds | `dagger-docs.log` |
| Wheel and sdist | `-m ci call build --source=. export --path=runs/local-provider/validation/dist` | 0; wheel and sdist built/exported | `dagger-build.log` |
| Focused Ruff | `uv run --no-sync python -m ruff check` with the file set below | 0; All checks passed! | `ruff.log` |
| Architecture | `uv run --no-sync python scripts/check_architecture.py` | 0; architecture check OK | `architecture.log` |
| Installed-wheel smoke | Dagger chain below | 0; installed wheel imported, lean imports passed, fixture completed | `dagger-smoke-wheel.log` |

Ruff file set: `recall_guard/core/{local_evidence,completion,providers,request_control}.py`,
`recall_guard/harness/{scorer_client,local_evaluation,local_execution,local_reporting}.py`,
`scripts/{probe_local_model,measure_local_models}.py`, `examples/local_provider.py`,
`tests/core/{test_local_evidence,test_providers}.py`,
`tests/harness/{test_scorer_client,test_local_evaluation,test_local_execution,test_local_reporting}.py`,
`tests/scripts/test_probe_local_model.py`, `tests/test_local_provider_example.py`.
These braces describe the explicit file list supplied, not PowerShell expansion.

The exported wheel is `recall_guard-0.5.0-py3-none-any.whl`, SHA256
`709a6fd1eb4b1639aa15fe74478a641e0ab57201519bfe9cdbd671d6820fff60`.
The sdist is `recall_guard-0.5.0.tar.gz`. No release or publication is implied.

### Installed artifact liveness

Existing packaging tests inspect dependency metadata, so an additional smoke was
necessary to verify an actual installed wheel. The successful Dagger chain was:

```text
-m ci call base --source=.
with-exec --args=uv,build,--out-dir,/outputs/dist
with-user --name=root
with-exec --args=uv,pip,install,--python,/opt/venv/bin/python,--no-deps,--reinstall,/outputs/dist/recall_guard-0.5.0-py3-none-any.whl
with-user --name=app
with-exec --args=cp,/app/examples/local_provider.py,/outputs/local_provider.py
with-workdir --path=/outputs
with-exec --args=python,- --stdin=<script below>
stdout
```

The lines form one CLI call. `--stdin` received this script as one PowerShell
here-string argument:

```python
import pathlib
import sys
import runpy
import recall_guard
from recall_guard.core.local_evidence import LocalModelIdentity
from recall_guard.harness.local_execution import execute_plan
location = pathlib.Path(recall_guard.__file__).resolve()
assert str(location).startswith('/opt/venv/lib/')
heavy = {'torch', 'onnxruntime', 'onnxruntime_genai', 'matplotlib', 'vectorbt'}
assert not heavy.intersection(sys.modules)
print('Installed wheel:', location)
print('Lean local-provider imports: PASS')
sys.argv = ['/outputs/local_provider.py', '--fixture']
runpy.run_path('/outputs/local_provider.py', run_name='__main__')
```

Observed output:

```text
Installed wheel: /opt/venv/lib/python3.12/site-packages/recall_guard/__init__.py
Lean local-provider imports: PASS
status: completed
network_requests: 0
fixture_attempts: 29
scored_held_out_rows: 4
evidence: synthetic mechanics only; no model efficacy measurement
```

The example was copied outside `/app`, and the wheel import path was asserted,
so this smoke does not mistake checkout imports for an installed artifact.
The initial attempt (`dagger-smoke.log`, exit 2) could not replace the root-owned
editable `.pth` as the app user. Only the disposable container installation step
was changed to root; smoke execution still ran as app. The successful receipt is
`dagger-smoke-wheel.log`. No host installation or infrastructure change occurred.

## Requirements and integration coverage

| Requirement | Implementation seam and concrete evidence | Assessment |
| --- | --- | --- |
| 1.3 | Existing `OpenAICompatibleClient`, explicit `ModelConfig`, `LocalAttemptBudget.client`; provider rejection/anonymous-local tests, local preflight model/context rejection and probe timeout/no-replay tests | Explicit failures retained; no hosted/model substitution |
| 2.2 | `validate_local_evidence` invoked by scorer response validation; `test_actual_probe_requires_complete_aligned_evidence`, `test_invalid_evidence_after_successful_probe_is_not_skipped`, local alignment/censoring fixtures | Invalid probe and later evidence rejected; no completion-only scoring fallback |
| 4.1 | `execute_plan` invokes `MemoryGuardedScorer.calibrate_client`, `score`, `score_ensemble`; existing example runs real provider parsing and scorer math | Existing APIs reused; fixture smoke reaches calibrated single and ensemble rows |
| 4.2 | `CalibrationIdentity` includes optional local identity; client and response checks preserve provenance; before/after sidecars and whole-batch invalidation | Drift/loss/unknown provenance tests cover both interfaces; legacy fingerprint omission preserved; sidecars do not establish atomic identity guarantees |
| 5.2 | Journaled `LocalAttemptBudget` with existing request admission controls; cancellation, unknown outcomes, elapsed and physical limits, persistence-before-send and reopened-journal tests | Stops new work without replay; retains prior/unknown attempts; does not claim remote generation stopped |
| 6.1 | Unchanged lean runtime dependency set in packaging tests; architecture gate and installed-wheel import assertion | No accelerator runtime import/dependency required; optional experiment packages remain checkout tooling |
| 6.2 | Explicit endpoint passed unchanged; real loopback HTTP fixture in `test_provider_readiness.py` runs in Dagger; `docs/local-provider.md` external-host networking guidance | Offline configured HTTP path verified; actual Windows-host/Dagger network route to Ollama not exercised and must be checked for the deployment network |

The seven task-boundary requirements have automated or inspection evidence, with
the deployment-network qualification stated explicitly for 6.2. The container
fixture demonstrates actual socket exchange; it does not demonstrate a physical
host route or accelerator availability. The fixture example substitutes HTTP
only and retains real parsing, evidence admission, calibration and scoring.

Cross-task contracts agree on generated-token positions, natural-log units,
partial alternative distributions, explicit unknown identity/resource values,
and item/arm/draw keys. Reporting pairs common usable items and excludes
condition-mismatched batches from calibrated claims. Unknown provenance permits
labeled within-run exploration, not proven cross-run reuse.

Dependency direction remains core contracts -> harness -> scripts/examples.
The architecture checker passed. No runtime source defect was observed; the CI fixture portability repair is described below. The optional
OGA probe/server and fake-runtime tests are absent because their prerequisite
branch never ran; these are conditional omissions, not verified implementations.
The new local evidence/evaluation/execution/reporting, probe/measurement scripts
and example were scanned for TBD/TODO/FIXME/HACK/XXX and simple hardcoded
password/api_key/secret/token assignments: zero matches. This narrow scan is
not a comprehensive security audit.

## Live evidence and conditional gaps

Existing dated evidence is preserved in `ollama-feasibility-receipt.json`,
`npu-setup-receipt.json` and `docs/results/2026-11-05-local-provider-feasibility.md`.
This task made no additional model calls. Historical live feasibility used two
physical attempts total and zero retries: Gemma passed exact aligned token
logprobs plus alternatives on CPU (8.547 seconds); GPT-OSS-20B responded on CPU
(24.953 seconds) but failed token-text alignment and remains ineligible.
Those observations verify only the tested configurations, not GPU/NPU support.

Task 1.2 completed a failed-prerequisite assessment. Tasks 2.2, 4.1 and 4.2 remain
blocked/conditionally skipped: no accessible pinned Gemma artifact revision and
footprint/license, unresolved compatible OGA release, unverified driver
compatibility and no known compatible isolated runtime. NPU inference count is
zero. No direct OGA full-vocabulary score, prefix alignment, text-only execution,
observed NPU placement, existing NPU server or optional bridge was verified.
Pure full-logit normalization fixtures are not substitutes for those gates.
Requirements 3.1/3.2 and NPU-specific portions of 2.4/3.4/3.6/3.7 therefore remain
unverified; the serving portions of 1.1/1.2/1.3/2.1/2.2/5.2/6.1 have no NPU evidence.

Unblocking requires exact artifact access/revision/license/footprint, authoritative
runtime/OGA matching, installed-driver compatibility evidence, and a concrete
isolated setup plan. Then run bounded direct text/evidence/placement validation
before any server verification or bridge build-out. No budget transfers from the
blocked NPU branch, driver/BIOS/WSL changes, or model substitution are authorized
by this validation report.

No live local development calibration, held-out discrimination measurement,
calibration-quality assessment, or mitigation experiment was performed. Synthetic
fixture fitting proves mechanics only. AUC is discrimination, not mitigation.
Historical NIM attempts remain labeled **NIM unavailable** with HTTP 504 and
unresolved cause; frontier completion-only work remains separate.

## Verification result

- Task claim: verify available integration and publish source-matched evidence,
  including the conditional NPU gap assessment.
- Task verification: VERIFIED for the bounded task claim; ready for independent review.
- Integration decision: available offline integration passed after the fixture repair;
  no remaining integration blocker was found within task 6.2.
- Overall NPU-inclusive feature decision: no feature GO; NPU prerequisites still require reassessment.
- Ownership: LOCAL test fixture portability defect repaired; external NPU prerequisites remain unresolved. No upstream repository fix is identified.
- Parent owns independent review, task state and commits.

## Integration finding and repair

The initial full Dagger suite completed with **2710 passed and 12 errors** (exit 1).
All 12 errors are setup failures in `tests/scripts/test_probe_local_model.py`:
its fixture loads `.kiro/specs/local-ai-provider/candidate-manifest.json`.
The canonical Dagger source filter and `.dockerignore` exclude `.kiro`, so the
file is unavailable in the intended CI container. This is a LOCAL feature test
fixture portability defect, not an NPU environment failure. This initial receipt
remains a failed result; it is not relabeled as passing.

Repair: provided a sanitized synthetic candidate manifest in the test fixture
boundary (`tests/fixtures/local_provider/candidate-manifest.json`), resolving it
relative to the test file. The fixture preserves the two frozen Ollama models,
settings and budget, with synthetic digests and no machine inventory metadata.
The parent authorized this repair after reviewing the concrete failure. The
`.kiro` exclusion remains intact; there are no production or infrastructure edits.

The fresh failed full-suite result is the RED receipt. The repaired test module
passed in the actual Dagger context: **18 passed in 4.24 seconds**, exit 0,
`dagger-probe-tests-green.log`. Command after the Dagger prefix:
`-m ci call base --source=. with-exec --args=uv,run,--no-sync,pytest,-q,tests/scripts/test_probe_local_model.py stdout`.
Changed-test Ruff also passed (`ruff-repair.log`, exit 0).
A single post-fix full regression ran because source changed after a concrete
integration failure: **2722 passed in 306.08 seconds**, exit 0, with no skipped
tests, errors or failures (`dagger-test-repaired.log`). Its result is recorded
separately from the RED run.

Source match is base commit above plus these SHA256 file hashes:

| File | SHA256 at repaired validation |
| --- | --- |
| `tests/scripts/test_probe_local_model.py` | `453e8b03af0c7094734f72b2b032e3db56839d5ed7952aa3534e705ecb4ee6fb` |
| `tests/fixtures/local_provider/candidate-manifest.json` | `1c6f54701cd1395f89c009818fdef1f7e37d45d760992c8d429dbe405005fe18` |

Documentation, application and packaging source are unchanged by this fixture
repair. Their successful strict-docs/build/installed-wheel receipts above remain
applicable; those checks were not unnecessarily repeated. The built artifacts
belong to the base commit before the test-only repair, as stated above.
