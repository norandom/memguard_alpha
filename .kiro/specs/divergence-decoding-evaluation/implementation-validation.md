# Implementation evidence

## Task 1.1

Status: verified complete after independent review on 2026-11-01.

Contracts and local-only planning are implemented under
`experiments/divergence_evaluation/`. Independent Docker validation passed
26 tests in 2.17 seconds and scoped Ruff, both exit 0. Review verdict APPROVED,
with no findings. The implementation test pass was 26 tests in 4.14 seconds.
RED before implementation was a missing `experiments` module collection error.

Coverage includes immutable versioned records, invalid target/settings/budget
rejection, nested serialization, explicit unknown cost/MCS, unresolved execution
prerequisites and a CLI fixture denying inference imports, credential reads and
network access. This evidence verifies task 1.1 only, not model execution,
auxiliary fitting or mitigation efficacy.

## Task 2.1

Status: verified complete after independent review on 2026-11-01.

Corpus admission and deterministic whole-document matched-size controls are
implemented in `experiments/divergence_evaluation/corpora.py`. Final focused
Docker corpus/contracts tests passed 41 tests in 2.40s. Source-matched Docker
Ruff passed. Independent review APPROVED after one evidence repair.

The original test-first RED attempt failed before pytest with a WSL connection
timeout. Subsequent verification remediation disabled the two public functions:
15 tests failed in 2.67s; enabling them yielded 15 passes in 1.02s. The temporary
flag was removed before the final 41-test pass. This later negative-control
check is not represented as an original preimplementation RED receipt.

Coverage includes temporal/revision/rights/hash/path checks, partition overlap,
supplied semantic audit attestations, declared intentional fact exposure and
control source/period proportions with explicit whole-document discrepancies.
Token counts remain declarations until checked by the later tokenizer fitter.
No real study corpus has been admitted or fitted by these tests.
