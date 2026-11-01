# Task 7.3 platform validation — 2026-11-01

Task 7.3 is **APPROVED AND VERIFIED** after staging remediation. Native Windows and Linux Python
3.12/3.14 have passing current-source full-suite evidence, including actual
notebook kernels. The wheel, source distribution, installed callback workflows,
saved metric replay and strict documentation build pass. Linux module discovery
was repaired by extracting the unchanged verified archive outside every Git
checkout. Independent review then exposed duplicate pytest discovery from the
remaining Windows staging tree; moving that owned tree outside the checkout
restored the canonical Windows full suite. No production or CI change was
required. This is not feature-wide GO.

## Task brief and boundary

Acceptance: exercise the supported Windows/Linux Python matrix and actual
notebook kernels; preserve explicit model/feature choices, UTF-8 artifacts and
runtime-versus-parameter errors; build usable distribution artifacts and strict
documentation; report offline evidence separately from live-provider coverage.
The deliverable is this report plus durable ignored `runs/task-7-3-*` evidence.
No production, CI, tests, task checkboxes or specification metadata changed.
RED is not applicable to this verification/report-only task; no artificial
behavioral test was added. Failed diagnostic assertions are disclosed below.

Requirements checked: **5.6, 6.1, 6.2, 6.5, 7.8, 7.9**. Design checked:
**Contracts**, **File structure and responsibility plan**, and **Validation and
requirement traceability**. The source-state and request-control boundaries
remain those approved for completion-api-mitigation.

## Source and execution controls

The archived/tested implementation commit is
`b1e923ca9682ca3eb80e4e85b8147305a055b358`. The post-review Windows rerun records
HEAD `273a28864497a1e8cb98753e9b1c2db4ab5f97a4`, whose only intervening changes
are the root-owned readiness specification/proposal Markdown and metadata.
Source, tests, CI and build/dependency files are identical to b1e923c; the
changed-path list is retained in `runs/task-7-3-staging-move-audit.json`.
The Windows 3.14 launch had an empty tracked diff. A separate tracked-only
snapshot was created by `git archive --format=tar --output=<archive> <commit>`
and extracted into `runs/task-7-3-source-b1e923ca`. The 315-file manifest is
`runs/task-7-3-source-manifest.json`; archive SHA-256 is
`534e2d2e8ca8b226a021f850defd1ddb9a5a4f5fc4ddd01aba7824e7fa50e22e`.
Before any Dagger invocation, archive entries and extracted paths were checked
for traversal, symlinks, `.env`, private-key extensions, `.git`, `.venv`, `runs`
and `HANDOFF.md`. None was present. Root-owned untracked readiness proposals
were excluded. Every explicit Dagger `--source` argument selected the archive,
not the live checkout. The initial staging location nevertheless let Dagger
infer the parent Git checkout as its module context; the isolation repair below
removes that unintended parent context.

`runs/task-7-3-validation-controller.py` records each exact command, working
directory, UTC timestamps, raw exit, hexadecimal exit, log hash and source
identity. It constructs an OS/process-essential environment without provider
credentials or inherited Python overrides. Python children are interrogated
directly for their executable, interpreter, OS and full installed dependency
inventory; controller versions are separately labeled. UTF-8 output is explicit.
The docs invocation also sets `DISABLE_MKDOCS_2_WARNING=true`; Material still
printed its advisory, but MkDocs emitted no build warning/error and exited zero.

In the historical commands below, `SOURCE` was the absolute snapshot path above.
That exact owned directory has since moved outside the checkout; its current
path and preservation audit are recorded under **Review remediation** below.
Original logs/manifests retain their execution-time paths. `PY312` is
`D:/Source/memguard_alpha/.venv/Scripts/python.exe`, and `PY314` is
`C:/Users/mariu/AppData/Local/Temp/recall-guard-ci-py314/Scripts/python.exe`.
`UV` is the installed executable at
`C:/Users/mariu/AppData/Local/Microsoft/WinGet/Packages/astral-sh.uv_Microsoft.Winget.Source_8wekyb3d8bbwe/uv.exe`.
`LEAN` is
`C:/Users/mariu/AppData/Local/Temp/recall-guard-task-7-3-lean-b1e923ca/Scripts/python.exe`.
Every abbreviated command is also stored expanded in its JSON evidence file.

## Platform and notebook results

| Environment | Command and result | Evidence |
| --- | --- | --- |
| Native Windows 11 build 26200, CPython 3.12.10 | `PY312 -m pytest -q -ra`: 1919 passed, 1 skipped, 1 warning, 183.30 seconds, raw exit 0 | Current Nov 1 independent task 7.2 run, `runs/task-7-2-review-20261101.{log,json}` and verdict; reviewed source diff subsequently committed as b1e923c |
| Native Windows 11 build 26200, CPython 3.12.10, after review remediation | Canonical root `PY312 -m pytest -q -ra`: 1919 passed, 1 skipped, 1 warning, 154.22 seconds, raw exit 0 | `runs/task-7-3-win312-post-staging-remediation.{log,json}`; HEAD 273a288 contains documentation-only additions to the tested implementation |
| Native Windows 11 build 26200, CPython 3.14.9 | `PY314 -m pytest -q -ra`: 1919 passed, 1 skipped, 1 warning, 324.16 seconds, raw exit 0 | `runs/task-7-3-win314-full.{log,json}` |
| Linux CPython 3.12.12, Debian 12 bookworm, canonical Dagger module | `dagger -m ./ci call test --source . --python 3.12`: 1920 passed, no skips/warnings, 186.62 seconds, raw exit 0 | `runs/task-7-3-linux312-isolated-full.{log,json}`; initial failed invocation retained separately |
| Linux CPython 3.14.2, Debian 12 bookworm, canonical Dagger module | `dagger -m ./ci call test --source . --python 3.14`: 1920 passed, no skips/warnings, 169.35 seconds, raw exit 0 | `runs/task-7-3-linux314-isolated-full.{log,json}` |

The initial Windows 3.12 evidence reused the independent Task 7.2 run. A later
Windows 3.12 full rerun was necessary to verify the staging cleanup identified
by review, as recorded below. The fresh Windows 3.14 run started
at 07:51:04.216625 UTC and finished at 07:56:41.053796 UTC; its full log hash is
`72268e0d2d1df3c1201d57c2d665f722008d149b8c44658aed042e4a11a78c97`.
Both Windows environments use pytest 9.0.3, NumPy 2.4.4, scikit-learn 1.8.0,
nbclient 0.10.4, jupyter-client 8.8.0 and ipykernel 7.2.0. The 3.14 environment
has stale editable *distribution metadata* reporting recall-guard 0.4.1; its
actual import resolves to this checkout's `recall_guard/__init__.py`, confirmed
separately in `runs/task-7-3-win314-source.{log,json}`. It is a source-suite gate,
not evidence for that old distribution. The new installed wheel is tested below.

The Windows skip is `test_symlink_escape_is_rejected`: WinError 1314 means this
user lacks symlink privilege. A targeted Linux 3.12 verbose run explicitly
reports this test **PASSED**, 1 passed in 7.70 seconds, raw exit 0
(`runs/task-7-3-linux312-symlink.{log,json}`); the full Linux suite has no skips.
The Windows warning is the existing pyzmq/Tornado Proactor-loop selector fallback
in the legacy notebook test. Neither was hidden or filtered.

`tests/harness/test_completion_examples.py` executes the complete application
and four fresh notebook kernels for rewriting off/on crossed with one/three
draws. Each notebook installs a temporary kernel pointing at `sys.executable`,
asserts the running kernel uses that interpreter, runs from the notebook
directory, blocks outbound HTTP and cleans up its owned kernel. Assertions
cover preserved choices, unavailable MIA/MCS, exact admission counts, rejected
rewrites and distinct runtime/parameter failures. The full suites also execute
the legacy mock-mode notebook. Thus these are actual kernel smoke results,
not notebook JSON parsing alone.

The actual canonical Linux containers use CPython **3.12.12** and **3.14.2**,
Debian GNU/Linux 12 (bookworm), glibc 2.36, on the WSL2 Linux kernel. The child
executable is `/src/.venv/bin/python3`; each has pytest 9.0.3, NumPy 2.4.4,
scikit-learn 1.8.0, nbclient 0.10.4, jupyter-client 8.8.0 and ipykernel 7.2.0.
The full inventory and OS-release content are in
`runs/task-7-3-linux312-container-metadata-final.{log,json}` and
`runs/task-7-3-linux314-container-metadata-final.{log,json}`. The latter's merged
stdout/stderr has a release advisory inserted in its final stdout copy; the
prefixed execution record contains the complete JSON. No SDK/CLI upgrade was
performed. uv warned that hardlinks were unavailable and used file copies;
this is dependency-setup output, not a pytest warning or test skip.

## Linux staging failure and repair

Read-only preflight found `/home/mc/.local/bin/dagger`, CLI v0.21.8, and Docker
server 29.9.2 in the user's Debian WSL distribution. The unchanged module
`ci/dagger.json` specifies engineVersion v0.21.4. The invocation was:

```sh
cd /mnt/d/Source/memguard_alpha/runs/task-7-3-source-b1e923ca
env -i HOME=/home/mc PATH=/home/mc/.local/bin:/usr/local/bin:/usr/bin:/bin /home/mc/.local/bin/dagger -m ./ci call test --source . --python 3.12
```

The CLI automatically provisioned its v0.21.8 engine. Its own old-engine cleanup
took 1m38s; provisioning then completed, with client/server v0.21.8 reported.
No manual cleanup, pruning, upgrade or module edit was attempted. At 07:54:34
UTC the command exited 1 after 171.687 seconds, before test execution. The
diagnostic says `unknown command "test" for "dagger call"`, with main object
`Query` and only `container-echo`/`grep-dir` functions, rather than `Ci`.
Log hash: `7a912d6826480a85ce3f9485293eea777aa7cdcff612c4a86697bf9067af6cef`.

The CLI's read-only help shows `-m, --mod` is supported. A single explicitly
selected help command, `dagger call -m ./ci --help` in the same clean snapshot,
still exposed the two unrelated functions; it exited zero because help itself
succeeded. Evidence is `runs/task-7-3-linux-module-help.{log,json}`. The executable
was identified as a native ELF binary, not a shell wrapper. These initial
observations prompted the independent investigation retained in
`runs/task-7-3-dagger-debug.md`.

The investigation established a staging configuration gap: Dagger 0.21.8 walks
up to the parent Git root and applies its ignore rules to the module context.
The archive was under ignored `runs/`, so its Python project files were filtered
out and the Python runtime scaffolded the starter `Query` module. An unchanged
isolated CI copy exposed all six real `Ci` functions with the same engine,
excluding the CLI/module version difference as the cause of this symptom.

The retry extracted the **entire original archive** into the new directory
`/tmp/task-7-3-source-b1e923ca-bp2vemju`, outside any Git checkout. All 315 file
hashes exactly match the original manifest and the archive hash above; path,
secret-file and symlink checks passed, and `git rev-parse --show-toplevel`
returned 128 with no root. Evidence is `runs/task-7-3-isolated-source.json`
and `runs/task-7-3-isolate-source-final.{log,json}`. No source, module, engine
pin or ignore-rule edits were necessary. No deletion, pruning or manual engine
cleanup was performed.

From that isolated full root, the one module-help confirmation exposed `base`,
`build`, `docs`, `lint`, `test`, and `test-matrix` and exited zero
(`runs/task-7-3-linux-isolated-module-help.{log,json}`). Canonical suites use
the same sanitized command above with this new working directory and Python
3.12 or 3.14. The module's `test` function runs `uv run pytest -q` after
`uv sync --frozen`. Container metadata is obtained through unchanged `Ci.base`
with an additional read-only Python probe, not inferred from image tags.
Linux 3.12 ran from 08:09:17 to 08:13:28 UTC (251.062 seconds including setup),
with log SHA-256 `6057f9e00d7f8d6830d854e6f377f3872b672325ac8ef31dda985a52619c51fd`.
Linux 3.14 ran from 08:13:33 to 08:17:23 UTC (230.250 seconds including setup),
with log SHA-256 `8a2901da4284553ed32da26e77f30aa2a131875827cc2afaa17ea82d6fc1f4b4`.
These were one full invocation per supported Linux version after isolation.
The post-staging source audit still found exactly 315 identical files and all
56 unchanged dated JSON artifacts, with no ancestor Git root
(`runs/task-7-3-isolated-source-final-audit.{log,json}`).

Two retry diagnostics failed before running their intended operations: the staging command
initially requested absent Debian `python3` (exit 127), then used the existing
uv-managed Python 3.12.14 successfully; the first metadata probe hit WSL's
shell parsing of Python parentheses (exit 2), then used WSL `--exec` to pass
arguments directly. Original logs remain at `task-7-3-isolate-source` and
`task-7-3-linux312-container-metadata`; corrected runs have `-final` labels.

## Review remediation: remove staging from test discovery

The first independent review rejected the workspace state: the canonical
`D:/Source/memguard_alpha/.venv/Scripts/python.exe -m pytest -q -ra` recursively
collected both `runs/task-7-3-source-b1e923ca/tests/` and `tests/`, causing
**62 import-file-mismatch collection errors in 14.53 seconds**, raw exit 2.
The copied `tests.core.test_bootstrap` was the first reported mismatch. Evidence
is preserved unchanged in `runs/task-7-3-review-20261101.{log,json}` and
`runs/task-7-3-review-20261101-verdict.md`; failed-log SHA-256 is
`b0c07c74c40409fc1acabc38e13f5544a3b7534a84a03f782e33d29a0c18ea21`.
This local validation-staging regression does not invalidate the earlier
passing receipts or establish a production-code failure.

The exact owned Windows staging directory was moved with native PowerShell
`Move-Item -LiteralPath` to:

```text
C:/Users/mariu/AppData/Local/Temp/recall-guard-task-7-3-staging-remediation-5e9f92fc9a6540ecb437107f00ad9c7e/task-7-3-source-b1e923ca
```

Before moving, the resolved absolute source was checked against the exact
owned path under `D:/Source/memguard_alpha/runs/`; the unique destination was
checked against the explicit Temp parent, with no ancestor Git checkout.
Reparse points and excluded file paths were rejected. The source is now absent
from the checkout. All **441 files** present immediately before the move were
preserved byte for byte, including the **315 original archived files** and
126 generated `.pyc` files retained for inspection. The original archive hash is
unchanged. No pytest settings, tests, production code or CI were changed.
No archive, log or manifest was removed or rewritten.

Move evidence is `runs/task-7-3-staging-move-command.{log,json}` and
`runs/task-7-3-staging-move.json`; independent post-move hash verification is
`runs/task-7-3-staging-move-audit-command.{log,json}` and
`runs/task-7-3-staging-move-audit.json`. Both commands exited zero. The latter
records the current destination and all additional retained filenames, so
future verification can use the relocated tree while preserving historical
receipt paths. The isolated Linux tree remains unchanged.

After the move, the exact canonical root suite was run once through the same
credential-free validation controller: `PY312 -m pytest -q -ra`. It passed
**1919 tests, with 1 skip and 1 warning, in 154.22 seconds**, raw exit 0.
The skip remains the Windows symlink-privilege limitation already covered by
the Linux tests; the warning remains the legacy notebook selector fallback.
The command ran from 08:23:39.712424 to 08:26:18.632183 UTC, 158.922 seconds
including controller/probe overhead. Receipt and full output are
`runs/task-7-3-win312-post-staging-remediation.{log,json}`; log SHA-256 is
`056931d4bc04ae5ad95e06969fdcd386ed2a9f108bdb48f8a635350346841802`.
`runs/task-7-3-remediation-evidence-check.json` verifies the preserved failed
review receipt and all three successful remediation receipt/log hashes.
The collection regression is resolved without changing test discovery rules.

## Distribution, first usable state and documentation

All following commands exited zero; their logs and expanded metadata are
`runs/task-7-3-<label>.{log,json}`.

| Label | Command / check | Result |
| --- | --- | --- |
| `build` | `UV --directory SOURCE build --out-dir D:/Source/memguard_alpha/runs/task-7-3-dist` | Wheel plus sdist; uv explicitly built the wheel from the source distribution |
| `lean-create` | `UV venv --python PY312 <external lean environment>` | Clean Python 3.12.10 environment outside the source tree |
| `lean-install` | `UV pip install --python LEAN <built wheel>` | recall-guard 0.5.0 and only runtime dependencies installed |
| `installed-smoke-final` | `LEAN <external installed-smoke.py>` from an external temporary directory | Import, all four callback feature combinations and saved-bundle metric replay pass with outbound HTTP blocked |
| `cli-help` | `LEAN scripts/run_intervention.py --help` from SOURCE | Documented CLI modes/options available |
| `cli-replay` | `LEAN scripts/run_intervention.py --replay docs/results/2026-10-31-completion-api/anthropic-evaluation-live` from SOURCE | Saved metrics recomputed successfully, no live mode |
| `docs-strict` | `PY312 -m mkdocs build --strict --site-dir D:/Source/memguard_alpha/runs/task-7-3-site` from SOURCE | Strict build successful; MkDocs reports 7.67 seconds |
| `artifact-audit-corrected` | `PY312 runs/task-7-3-artifact-audit.py` | Package contents/metadata, committed study bytes and downloadable copies verified |
| `ruff` | `PY312 -m ruff check .` | All checks passed |
| `architecture` | `PY312 scripts/check_architecture.py` | Architecture checks OK |

Installed smoke verifies that no checkout path is on `sys.path`, the import is
inside the clean environment, and neither matplotlib nor vectorbt is installed
or imported. Runtime dependencies are fresh consumer resolution rather than the
development lock: notably NumPy 2.5.4, scikit-learn 1.9.1 and SciPy 1.18.1;
the exact inventory is in child metadata. It then executes the actual configured
callback workflow through preflight to completed answers for original single,
original ensemble, rewritten single and rewritten ensemble, checking explicit
unavailable MCS and request counts. The callback responses are fixed offline
fixtures and establish wiring, not provider compatibility or effectiveness.

The 12 saved JSON artifacts were copied to an external `saved-bundles-München-東京`
path and all hashes checked before the installed package loaded/recomputed the
report. `metrics.json`'s saved `metrics` payload equals the recomputation. This
exercises Unicode paths and offline portability; Unicode text round-trips are
also covered by the full-suite report tests. The copied `.gitattributes` is a
13th ancillary file, not a JSON artifact.

Artifact SHA-256 values:

| Artifact | Bytes | SHA-256 |
| --- | ---: | --- |
| `recall_guard-0.5.0-py3-none-any.whl` | 218353 | `a612de03e18611445e453c616e3c0837c6a94ee2f754a0b9ee2be3d255001fbb` |
| `recall_guard-0.5.0.tar.gz` | 2461907 | `88b8c45621a6632edba5c99775f8fec08beee9a4cc0fe00499e2979aa4b9b155` |

The wheel has 53 members; the sdist has 316. Archive path inspection excludes
ignored credentials, local environments, run outputs, private-key extensions
and traversal. Wheel metadata requires Python >=3.12 and exposes only `backtest`
and `docs` extras; development/pipeline tooling remain dependency groups. The
sdist contains the adopter application, notebook and experiment CLI. Detailed
artifact inventory is `runs/task-7-3-artifact-inventory.json`.

Exactly **56 dated JSON files** were verified unchanged against committed bytes:
44 study originals under `data/experiments/2026-10-31-completion-api/`, plus 12
published copies under `docs/results/2026-10-31-completion-api/`. All 12 generated
site downloads equal their documentation sources and study originals byte for
byte. Strict MkDocs verifies its configured documentation/link build; no claim
is made that external websites were crawled for availability.

Three initial diagnostic failures were in the new ignored verification scripts,
not production: `installed-smoke` counted `.gitattributes` as a 13th JSON file;
`installed-smoke-corrected` compared the metrics payload to its wrapper document;
and `artifact-audit` incorrectly assumed 56 study originals instead of 44 study
originals plus 12 copies. Their exit-1 evidence is preserved. The final smoke
and corrected audit passed after those assertions were corrected to the observed
artifact schema/count. No product code or tests were changed to obtain a pass.

The staging retry also verified the same wheel on Linux using the existing
uv-managed **CPython 3.12.14** in a newly created external environment
`/tmp/task-7-3-linux-lean-a0sgju6d/venv`. Its wheel SHA-256 was checked before
installation. The adapted existing smoke ran outside both source checkouts,
verified the import comes from the installed environment, retained lean imports,
completed all four callback feature combinations and recomputed the saved
`metrics` payload from an exact copy of the 12 JSON files in a Unicode path.
Outbound application HTTP was blocked; dependency installation used public
package downloads. Venv creation, wheel installation and smoke each exited
zero. Actual interpreter, OS, full runtime dependency inventory, subcommand
exits and output are retained in `runs/task-7-3-linux-installed-smoke.{log,json}`
(log SHA-256 `bc8a3418df1ee80eb9eb59046fe3e3ab4dbc25322bfd09d3c30869d1c9b97a39`).
This Linux installed-artifact runtime is distinct from the canonical container
matrix; the earlier Windows build/docs/artifact gates were not rerun solely
because of the staging repair.

## Requirement assessment and remaining limits

| Requirement | Evidence and limitation |
| --- | --- |
| 5.6 | All checks above are offline; no credentials or provider calls used. Live gaps remain explicitly unmeasured. |
| 6.1 | Native Windows 3.12/3.14 pass without WSL/GPU; canonical Linux 3.12/3.14 suites pass. Linux was hosted through Debian WSL/Docker for validation, which is not a consumer runtime requirement. |
| 6.2 | Existing report UTF-8 round-trip tests pass; Unicode-path installed replay and exact JSON/download bytes verified. |
| 6.5 | Strict guide/API build, complete application examples, CLI help/replay and unchanged dated downloadable report verified. |
| 7.8 | Four real current-interpreter notebook combinations preserve the configuration and feature choices. |
| 7.9 | Complete application/kernel assertions distinguish callback runtime errors from unsupported native parameters. |

The historical intermittent Windows `0xE0000001` exits and separate traceback
`MemoryError` still have no established cause; this passing invocation does not
prove them fixed. The separate Dagger module-selection staging fault is resolved
by isolation and does not explain those historical Windows failures.

Tasks 6.2, 6.3 and full 6.4 remain incomplete: NIM returned HTTP 504 after about
302 seconds under the last 600-second timeout, exhausting the frozen four-request
eligibility allowance. No logprob-capable live calibration/measurement is newly
available. The saved Opus experiment rejected all 16 rewrites and has zero valid
rewrite pairs; no mitigation-effectiveness conclusion follows. Adaptive provider
readiness is separate work and no readiness/provider request was made here.

## Verification result

- STATUS: VERIFIED
- CLAIM_TYPE: TEST_OR_BUILD
- CLAIM: The current-source supported platform matrix, actual notebook kernels, installed-artifact/replay, strict documentation and artifact-integrity gates passed.
- EVIDENCE: Windows 3.12/3.14 each passed 1919 tests with one documented symlink skip; Linux 3.12/3.14 each passed all 1920 tests. The Linux targeted symlink test, four installed callback combinations and Unicode saved-metric replay passed. Earlier Windows build, installed-artifact, strict docs and artifact audits remain valid. `runs/task-7-3-evidence-summary.json` verifies all nine new successful receipt/log hashes, extracts the actual runtime inventories, confirms an empty tracked diff at b1e923c and checks all 56 live dated JSON files against the archived hashes.
- GAPS: Fresh independent task-completion review is pending after the rejected collection check was remediated and the canonical suite passed. Historical Windows intermittent failures are not explained by these passes. Required live measurement coverage remains incomplete as documented above.
- NOTES: Controller owns the fresh independent review and Task 7.3 checkbox/commit. No feature-wide GO or mitigation-effectiveness claim.

## Independent acceptance and parent verification

The fresh remediation reviewer returned APPROVED with no blocking findings.
The canonical native Windows suite passed 1919 tests, with the documented
skip and warning, in 235.06 seconds (raw exit 0). Evidence is
`runs/task-7-3-remediation-review-20261101.{log,json}` and its verdict. Log
SHA-256: `559e6771638871b41f943f4cd88885c8632d2d70921fe0b5cc34081f48688819`.
Ruff, architecture and whitespace gates passed. The independent audit verified
22 previous receipt/log hashes, all 441 relocated staging files, both built
distributions, 56 dated JSON files and 12 downloadable copies.

Parent verification checked the actual final log, complete test count, zero
exit and exact hash, together with the successful earlier matrix receipts and
archive identity. Task completion is VERIFIED. Earlier statements of pending
review describe the implementer submission and are superseded by this section.
The live-measurement limitations above remain; feature-wide completion is not
verified. No readiness implementation or new provider request is covered here.
