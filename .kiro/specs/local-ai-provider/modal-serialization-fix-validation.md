# Modal serving serialization repair: local validation

Date: 2026-11-06. Scope: task 10.2 deployment diagnostic repair only.
No cloud calls, deployment, GPU startup, inference, upload, download or credential
file reads were performed for this repair. Existing receipts remain unchanged.

## Root cause and red evidence

The nested serialized `Serving` class captured the factory's SDK `App` in
`start`, `record` and `readiness`. That App's local registry contains its own
unhydrated `Serving` class and service function. Recipe construction does not
serialize the class, so the previous construction-only smoke missed the cycle.

Offline Python 3.12 and the actual installed Modal 1.6.3 serializer reproduced
the failure from the pre-repair `HEAD:deploy/modal/serve.py` source, executed in
an in-memory module. `Cls._get_user_cls()` returned the actual raw user class.
The App, Cls and Function were never hydrated. The following first diagnostic
also reproduced the original App capture before the Dict capture was addressed.

`modal._serialization.serialize(raw_class)` failed:

```text
InvalidError: Can't serialize object Function(Serving.*) which hasn't been hydrated.
```

Removing only the captured App's local Function registry entries isolated the
remaining class edge and reproduced the exact live diagnostic:

```text
InvalidError: Can't serialize object Cls(Serving) which hasn't been hydrated.
```

The initial repair removed App capture but left an unhydrated captured Dict.
Independent review rejected its test preparation, which manually hydrated that
Dict: the SDK does not discover this closure-only Dict in its dependency graph.
A fresh network-blocked diagnostic on the untouched raw class reproduced:

```text
InvalidError: Can't serialize object modal.Dict.from_name('recall-guard-gpt-oss-20b-pilot-provenance', environment_name='recall-guard-pilot') which hasn't been hydrated.
```

The corrected regression uses the untouched raw class without synthetic
hydration or registry edits. The diagnostic subprocess blocked socket connect,
connect_ex, getaddrinfo, sendto and bind through a Python audit hook.

## Repair and identity contract

`Serving` captures neither an App nor a Dict. Its enter hook obtains the observed
running App ID once, before checking remote expiry or launching vLLM, and uses
that same ID for expiry admission, provenance keys/snapshots and readiness.
An enter failure without identity does not publish a closing snapshot.

Modal 1.6.3's installed public API has no running-App getter. The narrow helper
uses its private `modal.app._App._get_container_app()` compatibility seam.
Source inspection confirmed `_App._init_container` copies
`running_app.app_id` before assigning `_App._container_app`; the getter returns
None outside a container. The helper requires exactly SDK 1.6.3 and validates
the observed ID format. Version checks use the SDK module's `__version__`,
including in the remote SDK loader, without requiring distribution metadata.
Missing context, invalid identity or another SDK version
fails closed. Recheck this private seam before any SDK upgrade. No configured
App ID or lookup by reusable App name substitutes for container identity.

The parent lifecycle explicitly creates/hydrates the named provenance Dict with
`create_if_missing=True` before spawning the expiry controller, within a
five-second control-call timeout and the existing total arming timeout. That
operation increments the arming control-call count. Only after it succeeds may
the CPU controller be spawned; creation failure closes admission and cleans up.
Parent acknowledgement reads and remote runtime lookups use
`create_if_missing=False`. The serving enter hook creates its Dict handle at
runtime and stores it on the instance. Expiry ownership, nonce/deadline checks,
stop ownership and resource/request limits retain their existing behavior.
This aligns with requirements 7.3, 7.5, 7.6 and 7.10 and `modal-design.md`.

## Fresh green verification

- Actual SDK serialization regression: 8 passed under the optional deployment
  Python 3.12 environment, using the existing root pytest module path. No
  dependencies, lock files or interpreter pins changed. The actual serializer
  accepts the untouched repaired raw user class with no synthetic hydration or
  registry modifications; the Cls and service Function remain unhydrated before
  and after serialization. Closure checks confirm start/record capture only the
  plain Plan and readiness has no nonlocal captures.
- That SDK module also checks observed context identity, absent context, invalid
  IDs and the SDK version guard. All tests block socket network operations.
- `uv run --no-sync python -m pytest tests/deploy tests/scripts/test_probe_modal_model.py -q`:
  **99 passed, 1 skipped**. The skip is the optional actual-SDK module in the
  root environment without Modal; it was separately executed above.
- Ruff over `serve.py` and both changed/new test modules: **passed**.
- `uv run --no-sync python scripts/check_architecture.py`: **passed**.

Exact optional-SDK test command, from the repository root in PowerShell, using
the installed root pytest module without adding deployment dependencies:

```powershell
@'
import sys
sys.path.append(r'D:\Source\memguard_alpha\.venv\Lib\site-packages')
import pytest
raise SystemExit(pytest.main(['tests/deploy/test_modal_sdk_serialization.py', '-q']))
'@ | uv run --project deploy/modal --no-sync python -
```

Live deployment, remote container identity, GPU readiness and token evidence
still require the separately controlled bounded pilot. This local result does
not establish live task 10.2 acceptance or mark any task checkbox complete.
