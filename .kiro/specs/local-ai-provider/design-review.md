# Design review — 2026-11-05

Decision: PASS, self-review of the feasibility-first design; not hardware validation.

- All 32 numbered acceptance criteria map to concrete design contracts.
- Scope separates local hosting from frontier completion-only work and NIM diagnosis.
- Existing client/scorer integration avoids a second scoring system.
- The optional bridge is gated by actual Gemma NPU score access and placement.
- Logit pruning, special-token alignment and runtime-version ambiguity are explicit
  failure gates rather than assumed compatibility.
- Legacy serialization and calibration identity remain unchanged when local
  provenance is absent. Response reconstruction must preserve new provenance.
- PowerShell is the Windows execution shell; WSL is only for Dagger CI.

Next phase: user review of design, then tasks. Paused at user request after this
documentation task; no task generation, installation or live measurement started.
