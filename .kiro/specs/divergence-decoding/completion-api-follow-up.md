# Completion-only API requirement

Recorded 2026-10-30 after the v0.5.0 local measurement.

The user requires optional memorization mitigation for ordinary completion
APIs, explicitly including Anthropic, OpenAI and other providers, without
requiring response logprobs. The approved and shipped divergence MVP accepts
three caller-controlled next-token-logit callbacks. It does not meet this
completion-only requirement.

The exact divergence equation needs compatible next-token vectors and control
of token selection somewhere. A server can apply it internally without
returning logprobs. A text-only completion interface with no equivalent
generation controls does not permit a faithful client-side implementation.
Three completed answers cannot be substituted for those vectors.

An API-compatible alternative must be specified separately, with explicit
provenance for the method and no claim of equivalence to divergence decoding.
Jane Street's prompt-rewriting experiment is a research candidate, not an
approved implementation decision. It would need factual-fidelity and
point-in-time preservation checks as well as recall, task-utility, failure-rate
and cost measurements. The ordinary hosted scorer's MIA features also require
logprobs; completion-only mitigation must not imply that those scores are
available.

Status: requirement recorded; method, requirements approval, design and tasks
remain open. Existing divergence requirements and implementation status are
unchanged. The 2026-10-30 hosted measurement did not exercise mitigation.
