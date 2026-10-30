> Updated matrix evidence: [Local CI matrix](local-ci-matrix.md) verifies both Python 3.12 and 3.14 on native Windows and Linux through Dagger. Earlier pending-version notes below describe the initial validation.

# Windows and Adoption Validation

Date: 2026-10-30  
Decision: **GO for the verified Windows Python 3.12 environment**  
This report supersedes the prior implementation-validation.md acceptance blockers.

## Policy and scope

The user explicitly removed Sentrux as a requirement going forward. Steering and current task instructions now use tests, Ruff and the native import-boundary checker. Complexity and dependency cycles remain code-review concerns; no new analysis tool is required.

The user also authorized the upstream Windows portability repair, readable code and separate adopter/engineering documentation. No model-training or hosted inference capability was added.

## Changes

- Architecture errors display portable forward-slash paths.
- Gap-analysis inputs and Markdown/CSV outputs use explicit UTF-8. Dataset builder outputs use UTF-8 as well.
- Pacing waits under its admission lock and records actual wakeup time, preventing late wakeups from compressing later starts. Network POSTs remain outside the lock.
- Regression tests reproduce a legacy cp1252 default and a delayed sleep; both failed before the corresponding fixes.
- Touched report code uses named records, prompt hashes and file handles rather than ambiguous single-letter values.
- README explains choosing and trying a workflow. MkDocs covers contracts, inputs, failure handling and verification. Both cite Jane Street; the decoder page links the original article.
- Native Windows CI targets Python 3.12 and 3.14 with pinned action revisions.

COBESY framing/composition and an embedded humanizer pass informed the documentation. The audience assumptions, reading paths and writing checks are recorded in adoption-notes.md.

## Evidence

Environment: Windows, Python 3.12.10, UTF-8 mode 0, default locale encoding cp1252.

| Check | Result |
| --- | --- |
| Full default-environment suite: uv run --no-sync pytest -q --tb=short | 600 passed, 1 non-blocking Jupyter event-loop warning, 117.65 seconds |
| Targeted encoding, pacing, architecture, docs and portfolio smoke tests | 53 passed |
| Independent reviewer encoding/pacing/architecture checks | 29 passed |
| uv run --no-sync ruff check . | Pass |
| uv run --no-sync python scripts/check_architecture.py | Pass |
| uv run --no-sync mkdocs build --strict | Pass |
| uv build | Wheel and source distribution built |
| Built-wheel import and greedy decoder smoke | Pass |
| Exact documentation callback example | (1, 2) eos |
| uv run --no-sync python harness.py --help | Pass |
| Windows CI matrix and pinned action syntax | Locally checked; hosted jobs not run |
| Independent task-4 review | APPROVED |

The existing event-loop warning is emitted by Jupyter/ZeroMQ during the offline notebook test; the notebook completes successfully.

## Coverage and limits

All 21 narrowed-MVP criteria retain implementation evidence and task-local approvals. Final task 4 is now satisfied under the user's updated gates. The architecture graph remains downward-only; no default dependency or scorer behavior was repurposed for divergence calibration.

No known Windows blocker remains for the verified core/decoder setup. Python 3.14 on Windows is configured for CI, not claimed locally validated. Live NVIDIA/FMP calls require credentials/network and were not exercised by the offline suite. Any custom local-model callback has its own runtime, hardware and tokenizer compatibility requirements.

Mathematical correctness and Windows compatibility do not establish reduced memorization. This remains an experimental callback utility.

