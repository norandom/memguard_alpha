# Adoption writing notes — Windows and divergence decoding

## Intake and framing

- Target behavior: adopters choose the scorer or experimental decoder and run a small example; engineers integrate the chosen interface and run the repository checks.
- Hidden assumption (inferred): readers need reproducible evidence before trusting a contamination claim. Organizational safety and network topology are unknown; no culture diagnosis is asserted.
- Anchor (inferred): an engineer who can reproduce the example and explain its failure modes to the next integrator.
- Framing protocol: describe a measurement problem and explicit limits. Avoid accuracy/alpha promises and guaranteed unlearning. The framing is wrong if an adopter concludes that synthetic correctness proves mitigation, or cannot identify the credentials/runtime required.
- Distribution: README at repository entry; engineering index at documentation entry; exact callback contract at the point of implementation. No broadcast or organizational rollout is needed.

## Composition blueprint

| Audience | Governing idea | Reading path | Detail budget |
| --- | --- | --- | --- |
| Adopter / README | Choose the smallest useful entry point and try it | Purpose, workflow choice, installation, offline trial, Windows, interpretation, references | Short examples; link contracts instead of reproducing them |
| Engineering adopter / MkDocs | Integrate one explicit contract and verify it | Interface choice, revision, scorer inputs/failures, decoder contract, ownership, workflows, Windows checks, evidence | Exact types/semantics in linked API and decoder page |

Keywords: calibration, logit, token ID, declared vocabulary, failure, Windows, UTF-8.
Each term appears next to an action or contract rather than in an introductory glossary. Older portfolio-performance detail stays in linked benchmark/consumer material.

## Validation and humanizer pass

Applied COBESY's systemic framing, cognitive compression and composition guidance, then its writing linter and an embedded humanizer pass. Removed the former shell-first opening, undefined example inputs, stale fixed test count, generic claims and mixed adopter/developer chronology. Preserved software/applied-paper citations and release identity. Added the Jane Street citation at the divergence entry points.

The reader can choose an interface before learning internals. Offline and live examples are identified. PowerShell uses ordinary Python commands. Claimed model effectiveness remains falsifiable and unproven for the helper. The prose uses concrete subjects and actions, ordinary headings, and no decorative emphasis or AI-style conclusions. Org rollout artifacts are not relevant to these repository documents.

## Windows scope

User instruction on 2026-10-30 supersedes older Sentrux requirements. Current gates are tests, Ruff, import-boundary validation, build and strict docs. Historical reports remain intact; a new validation result will supersede their blockers. Native Windows CI covers Python 3.12 and 3.14, but remote execution is not claimed until those jobs run.

