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

## Task 1.2

Status: verified complete after independent review on 2026-11-01.

Optional Docker image built successfully and was independently inspected:
`sha256:acdf5deaffe08298a0ada7d99918eaae181140029deff35433ee0ee7e3b28c47`.
Network-disabled inventory and pinned dependency/GPT-OSS class imports passed.
The reviewer ran 71 focused packaging/experiment tests, scoped Ruff and the
offline uv lock check, all exit 0. The lean installed wheel passed independent
checks outside the checkout with experimental and inference packages absent.

The optional dependency group and Docker target do not alter library runtime
dependencies. Slow dependency download/export and transient WSL connection
errors resolved without configuration changes. Image imports establish software
availability, not GPU capacity or model feasibility. No model artifacts or
inference were used. See `environment-validation.md` for versions and receipts.

## Task 2.2

Status: verified complete after independent review on 2026-11-01.

The sparse estimator implements fixed interpolation, document resets, finite
normalized full-vocabulary log probabilities and identity-checked JSON reload.
Actual tokenizer callback counts must match admitted declarations. Fitting
enforces document/token limits and conservative owned-allocation admission;
this is not a process-RSS or tokenizer-callback memory guarantee.

Independent final Docker validation passed 73 experiment tests in 2.10s and
scoped Ruff, exit 0. Review APPROVED after one substantive repair: variable
fact/audit/partition metadata was missing from the original pre-admission memory
estimate. Seven regressions failed before repair. The reviewer independently
confirmed that the original 5,000-fact-ID / 20,000-byte-cap bypass is now rejected
before admission allocations. Earlier behavioral negative-control verification
disabled the fitter (23 failures), enabled it (23 passes), then removed the flag.

All fitted data used for these tests is synthetic and tiny. No actual study
auxiliary artifact or mitigation result has been produced.
