# Task 7.2 integration and public-contract validation

Validation date: 2026-10-31. Baseline: `a826d7e1acaa76872a9e2dc388b8b7a05d1ba3ec`.
This report covers offline integration and readability. It does not complete
the live measurement or platform matrix.

The fresh Windows 3.12 offline gate passed: 1919 tests passed, one skipped and
one warning. Ruff and architecture passed. Task-local changes are ready for
independent review; this is not feature-wide GO.

## Task brief

- Acceptance: retain legacy scorer/reducer behavior and lightweight imports;
  disabling rewriting makes no auxiliary calls; credentials stay outside
  persisted metadata; names, units, record grain and pairing remain explicit;
  tests use offline fixtures; live and platform gaps remain visible.
- Completion evidence: review public contracts and numerical alignment, run a
  focused regression subset, one full current native Windows suite, all-repo
  Ruff, the architecture checker and the diff whitespace check.
- Constraints: preserve existing algorithms, entry points, dependencies,
  calibration/budget/identity checks and frozen study artifacts. No feature
  flag, source-data change or live provider call belongs to this task.
- References: requirements 1.2, 3.4, 5.6, 6.1-6.5, 7.8, 7.9 and 8.2; design
  `Contracts`, `Scoring and ensemble compatibility`, `Evaluation protocol`,
  `File structure and responsibility plan`, and `Validation and requirement
  traceability`.

## Source review and small corrections

The declaration inventory in `runs/task-7-2-contract-inventory.json` covers
88 public declarations in 24 feature and integration modules. Declarations, fields and
docstrings were checked alongside the source paths below. This is a focused
integration review, not a claim of exhaustive behavioral proof.

The first inventory console print hit a CP1252 `UnicodeEncodeError` after the
UTF-8 JSON file was written. Its ignored diagnostic script now prints an
ASCII JSON summary and reran with exit 0. Production encoding was not changed.

| Boundary and source | Semantics checked |
| --- | --- |
| `core/completion.py`: `ModelConfig`, capability types, `ModelResponse`, token types and `CompletionClient` | Explicit protocol/endpoint/model; immutable requested settings; omitted versus explicit null; runtime credential reference excluded from export; token order is output position; logprobs use natural logs and retain censored markers without making them MIA evidence. Missing usage/evidence stays unknown. |
| `core/request_control.py`: `CallBudget`, `RequestEvent`, `RequestControl`, request failures; `core/providers.py`, `provider_anthropic.py`, `provider_callback.py` | Physical attempts include probes/retries; logical callback invocations are separate. Times are monotonic seconds with UTC event timestamps. Stock transports disable hidden retries/redirects/ambient authentication. Opaque callbacks cannot claim strict accounting; synchronous callback runtime errors differ from rejected parameters. Probe acceptance does not confirm unechoed settings. |
| `core/ensemble.py`: result/spec, generation and reduction functions | Existing reducer and original response objects are retained. Usable responses, requested/attempted logical draws and physical calls differ. Input waves follow the same canonical permutation as draws; component locations retain caller-defined units and grid spacing uses those units. Added this alignment/units clarification to `reduce_draws`. |
| `harness/scorer_client.py`; `harness/scorer.py`: calibration identity, configured fit/response scoring, `ScoredDraw`, scored replay and legacy methods | Fit identity binds protocol/endpoint/model/settings/intervention. Supplied responses are scored without regeneration. Scores, response content and executor waves stay associated through sorting; per-draw scores feed the existing mean reduction. Added binary coefficient shape/feature-column/class-axis documentation to the fit fingerprint. |
| `mia/mcs.py`: extended calibrator metadata and optional `train(before_fit=...)`; package-root exports; `harness/request_failures.py` | Existing matrix columns remain `feature_order`, row labels are IS=1/OOS=0, and retained holdout tuples come from the same holdout rows and actual class-1 predictions used for AUC. The optional hook runs after collection and before split/fit; the unchanged default remains `None`. Root exports add workflow types without plotting/backtest imports. Shared stop categories inspect structured sanitized failures and separate transient limits from fatal quota/auth/configuration errors. |
| `harness/workflow_calibration.py`: calibration item/result and preparation | Rejected/pending input counts differ from valid producer rows. Accepted prompts are materialized for the existing baseline/training stages. Added explicit holdout index pairing, class labels and probability units; holdout order is independent of preparation-item order. |
| `harness/rewriting.py`, `rewrite_fidelity.py`: inputs, renderers, generation, policy and check functions | Instructions, evidence and as-of date remain distinct. Candidates are unverified until both required check stages pass. Failure retains completed stages and schedules no target fallback. Pattern checks disclose limited coverage and possible false rejection; automated acceptance is not a guarantee. |
| `harness/workflow.py`, `workflow_types.py`: workflow, draw, preflight and result | Disabled rewriting does not validate/probe/invoke auxiliary roles. Logical draw indices precede outcomes and differ from canonical retained-ensemble order. Shared rewrite identity/evidence hashes bind draws. Single guarded fields, per-draw ensemble evidence and mean exposure multiplier are distinguished. |
| `harness/intervention.py`, `intervention_types.py`, `intervention_journal.py`: runner, protocol, item/draw/rewrite/request/checkpoint/resume/run/condition contracts and snapshots | Labels stay outside rewrite inputs. Draw keys are `(item_id, arm, draw_index)` with draw zero preselected; physical keys are `(execution_id, call_id)`. Ordered inputs freeze counterbalancing. Intent is saved before provider operations; ambiguous restarts refuse replay. Snapshots whitelist data rather than traverse live clients/scorers. Added the absolute accuracy-difference unit for `utility_tolerance`. |
| `harness/intervention_config.py`: strict JSON reader, validated config, loading and plan | Exact input bytes bind hashes; offline plans resolve no credentials and claim no observed provider readiness. Nominal/retry upper bounds remain subordinate to remaining physical cap. Historical eligibility and ordinary post-cutoff guards remain distinct. |
| `harness/intervention_metrics.py`, `intervention_usage.py`: metric and usage summaries | Analyzer rows are item/condition in fixed condition order, aligned by lexicographic item ID. One-dimensional `item_deltas` has one eligible paired item per element; `bootstrap_means` has one item-resampled replicate per element. Arms/draws stay together; neither condition rows nor target draws become independent resampling units. Accuracy/confidence deltas use probability units. Usage rows are unique physical requests; overlapping single/ensemble views cannot be added. Latency sums request work in seconds, not concurrent wall time. Missing cost/counters remain unknown. |
| `harness/intervention_report.py`; private `harness/_execution_metadata.py` | UTF-8/LF bytes and relative artifact paths support exact offline replay without fitting. Provider validation is caller-asserted, separate from integrity hashes. Structured credential/header keys are rejected while exact task labels/answers and raw text remain data. Corrected `ExperimentReport`'s docstring to distinguish retained draw grain from analyzer item/condition grain. |

These feature modules use typed records, mappings and lists rather than pandas
DataFrames. No new DataFrame index/column alignment is introduced. Existing MIA
training uses rows of valid source items and columns in `feature_order`
(`mia/mcs.py:_gather_train_xy`); the scorer reuses that producer and estimator.
No array or reducer mathematics changed in this task.

The stale Sentrux enforcement comment in `pyproject.toml` now describes review,
Ruff, tests and architecture guidance. The requirements review tail and
`spec.json` now reflect the actual approvals and partial measurement while
retaining `phase=implementing` and all approval values.

## Offline isolation and regression evidence

The runner tests already inject fake model factories, but
`harness/runner.py:run` intentionally loads dotenv in production. A test-module
autouse fixture now replaces only `runner_mod.load_dotenv`, preventing these
offline tests from reading the checkout `.env`. Existing synthetic-key,
missing-key and rejected-key assertions remain intact. Production loading is
unchanged. No new test duplicates this fixture.

Provider adapter tests intercept `requests.adapters.HTTPAdapter.send` and pass
synthetic credential mappings (`tests/core/test_providers.py`,
`test_provider_anthropic.py`). Pilot helper tests likewise intercept transport
and supply fixture mappings. The completion application/kernel tests block
outbound HTTP; the moved-report replay subprocess blocks socket connections
and DNS. Legacy scorer/runner tests inject fake clients. No measurement script,
real provider access, `.env` content or frozen study data was used by this task.

Both pytest runs use the ignored controller
`runs/task-7-2-validation-controller.py`. It passes only OS/process essentials
to the child, omits provider credentials, sets child `PYTHONIOENCODING=utf-8`,
merges stdout/stderr into a durable log and records HEAD, tracked-diff hash,
versions, UTC timestamps, duration and raw/hex exit. Heartbeats do not alter
test selection. Native Windows 11 build 26200, Python 3.12.10, pytest 9.0.3,
NumPy 2.4.4, scikit-learn 1.8.0, nbclient 0.10.4 and jupyter-client 8.8.0.

| Check | Exact child command | Result and evidence |
| --- | --- | --- |
| Focused integration | `.venv/Scripts/python.exe -m pytest -q tests/harness/test_scorer.py tests/core/test_ensemble.py tests/core/test_ensemble_completion.py tests/harness/test_workflow.py tests/harness/test_workflow_scoring.py tests/harness/test_runner.py tests/harness/test_intervention.py tests/harness/test_intervention_report.py tests/test_packaging.py` | 422 passed, 1 skipped in 60.56 s; raw exit `0` / `0x00000000`; `runs/task-7-2-focused.log` and `.json`. Includes scorer/evaluator parity, lightweight root-import subprocess, reducer replay, configured scoring, workflow, journal/report and packaging checks. This preceded final docstring-only clarifications. |
| Full offline suite | `.venv/Scripts/python.exe -m pytest -q -ra` | 1919 passed, 1 skipped, 1 warning in 168.78 s; controller elapsed 172.437 s; raw exit `0` / `0x00000000`. `runs/task-7-2-full-py312.log` and `.json`. `-ra` adds reporting only; no test selection or ignore arguments. |
| Ruff | `uv run --no-sync ruff check .` | Exit 0; `All checks passed!`; `runs/task-7-2-ruff.log` and `.json`. |
| Architecture | `uv run --no-sync python scripts/check_architecture.py` | Exit 0; `architecture check OK`; `runs/task-7-2-architecture.log` and `.json`. |
| Whitespace | `git diff --check` | Exit 0; no whitespace errors; `runs/task-7-2-diff-check.log` and `.json`. |
| Change boundary | `.venv/Scripts/python.exe runs/task-7-2-source-audit.py` | Exit 0; executable AST unchanged in all five edited production modules; parsed TOML unchanged; spec approvals/phase unchanged; runner test diff is only the isolation fixture; no tracked data changes. `runs/task-7-2-source-audit.json`. |

The sole existing skip is
`tests/harness/test_intervention_report.py::test_symlink_escape_is_rejected`,
where Windows denied symlink creation with `WinError 1314`. The sole warning
comes from `tests/harness/test_notebook.py::test_notebook_executes_in_mock_mode`:
pyzmq reports that the Proactor loop lacks `add_reader` and uses tornado's
additional selector thread. The notebook test passed. No warning suppression,
test skip, event-loop policy or system setting was added.

The full attempt ran from 19:18:43.275333 to 19:21:35.703574 UTC. Its log SHA-256
is `d7526f9b0e8e2e10ad0dcb6e4fee39a4bcfa77d8cc8d43f1e61053f60983b8cb`;
the tracked-diff SHA-256 at launch is
`6eb59740e1db5d99cb2dad9021d10a3d844f8bcd3ca1745335fb964a2f3913a2`.
The report was written after launch; implementation source did not change
during or after this attempt. One full suite was run; no repeat was needed.

Kiro fresh-evidence assessment: `VERIFIED` for these `TEST_OR_BUILD` results
at the current source state. The requirements/design checks above support
submission for independent review. No feature-completion claim is made.

## Evidence limits

## Independent review and parent verification, 2026-11-01

The fresh independent review returned `APPROVED` with no blocking findings.
Its full native Windows Python 3.12.10 run completed with 1919 passed, one
symlink-privilege skip and one existing notebook warning in 183.30 seconds,
raw exit `0 / 0x00000000`. The tracked source diff matched the implementation
run. Primary evidence is in `runs/task-7-2-review-20261101.{log,json}` and
`runs/task-7-2-review-20261101-verdict.md`. The log SHA-256 is
`f5400a096bfc82061616ddf944d8d3afc00edb2bd0a1c410cee320cf652a4b23`.

Parent verification inspected the actual command, output, raw exit and current
diff; a fresh isolated runner/packaging test run also exited zero. Evidence is
in `runs/task-7-2-parent-20261101.{log,json}`. Ruff, architecture and whitespace
checks passed in the independent review. This verifies task 7.2 within its
integration boundary; the platform and live-measurement limits below remain.

## Remaining evidence limits

The earlier intermittent native Windows `0xE0000001` abrupt exits, and the
separate task 7.1 traceback `MemoryError`, have no established source cause.
The history remains in `runs/full-regression-debug-20261031.md`. An old
1373-test diagnostic pass excluded report tests and is not the current full
gate. This fresh pass establishes only this invocation's result; it does not
prove either historical failure fixed.

Task 7.3 owns the remaining Windows 3.14/Linux matrix, distribution checks and
strict documentation build. This Windows 3.12 check cannot establish all of
requirement 6.1. Notebook/kernel tests in the full suite are offline runtime
evidence, not live provider validation.

Tasks 6.2, 6.3 and full 6.4 remain incomplete. NIM inference eligibility is
unproven after timeouts and HTTP 504; all four allowed eligibility requests
are used. No fresh matched logprob calibration or second capability-mode live
measurement is available. The saved Opus portion rejected all 16 rewrites and
has zero valid rewrite pairs, so efficacy is unavailable. See
[the dated partial pilot report](../../../docs/results/2026-10-31-completion-api-pilot.md).
These offline gates do not establish two-mode coverage or feature-wide GO.
