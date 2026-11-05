# Reassessment — 2026-11-01

The user requested reassessment because Recall Guard is a portable Python library for API calls and calculations, not an OS-specific native application. Implementation and tests are stopped. This assessment does not mark unfinished work complete or discard existing changes.

## Findings

- No OS-specific application dependency has been demonstrated by this work. The unexplained 0xE0000001 test-process exits are execution observations, not a diagnosed Windows, uv, provider or library defect.
- The workflow overemphasized repeated platform validation and task-by-task full-suite reviews. The readiness implementation became too broad before delivering the end-user fallback/reporting flow.
- Since commit 0f232ad, committed readiness work added 2,964 library lines and 2,941 test lines across 16 library files and 10 test files. Current uncommitted execution work is additional.
- Tasks through 3.1 are committed at f9dfccb. Task 3.2 is unfinished. Its reviewed snapshot passed 2,380 Linux tests but had two independently reproduced defects: unsanitized failed response evidence and a race replacing a fatal cause with caller cancellation. Current repair edits are partial and unvalidated.
- All workers and tests have stopped. No provider inference calls were made during this continuation. Existing study artifacts and live allowances remain unchanged.

## Narrower way forward

1. Inspect the current changes for duplication and unnecessary abstractions before adding more code. Reuse existing clients, request accounting, workflow and ensemble calculations.
2. Finish the two concrete execution defects with focused regressions. Preserve successful response behavior, exact request counts and truthful failure reasons.
3. Demonstrate one coherent path: explicit model/settings, capability observation, caller-authorized completion-only fallback, existing rewriting/ensemble execution, and a compact report of what actually ran. MCS remains unavailable without the required token evidence.
4. Use uv run --no-sync from the project root. Run targeted checks during changes and one full regression at a meaningful integration boundary. Use an application/notebook smoke for portability; do not use unexplained process exits to invent an OS-specific development problem.
5. Reconcile the remaining specification tasks around this delivery path before expanding persistence/reporting infrastructure. Keep implemented optional transport facilities separate from the minimal demonstration; do not silently remove approved behavior or change old artifacts.
6. Only then execute the separately approved bounded measurement under its frozen limits. Operational availability and mitigation effectiveness remain separate measurements. Do not replenish old budgets or add calibration/paid auxiliary calls implicitly.

The original requirements and approvals remain recorded in
archive/pre-reassessment-2026-11-01. Revision2 requirements/design/tasks and the
[infrastructure assessment](infrastructure-assessment.md) make this narrower
delivery concrete. The [Docker spec](../docker-execution/design.md) proposes one
shared locked environment. These documents do not claim the feature is finished
or diagnose the unexplained host process exits. Implementation remains stopped;
the revised tasks and Docker implementation are not auto-approved by this
reassessment. The user subsequently approved proceeding with Docker.

## Upstream test-contract correction during Docker acceptance

The actual pinned-Dagger container regression completed with2381 passing and
four failing tests, all in the uncommitted task3.2 regression case for failure
sanitization. The assertion referenced ModelResponse.complete, which is not part
of that contract. Complete versus partial retained evidence is represented by
ModelResponse versus PartialResponse types. Corrected that assertion to require
ModelResponse while preserving content redaction, numeric evidence, usage,
physical counts and failure-kind assertions. No runtime contract or library
implementation was changed by this correction. It belongs to readiness3.2;
the rest of its uncommitted repair remains preserved and unaccepted.
