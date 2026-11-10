"""Request recording, resume and host checks of the study runner (task 7.4; Requirements 5.8, 11.1).

Every study request and its outcome is recorded privately, per arm and stage,
beside the stage's ``budget.json`` (task 7.5), under
``runs/finance-sim/records/<arm>/<stage>/``:

- ``<item_id>.jsonl``: append-only, one JSON object per line. A ``request`` line
  is written and flushed to disk before each generation request is sent and an
  ``outcome`` line after it: ``plan_sha256``, ``arm``, ``stage``, ``item_id``,
  ``rendering``, ``draw_index``, ``requested_at``, ``prompt_sha256``,
  ``settings_json`` (``identity_settings`` of the generation settings), ``seed``
  and ``host_digest`` on the request; ``response_sha256`` or null,
  ``failure_code`` or null, ``usage``, ``latency_s``, ``evidence_ref``, the
  recorded draw (``RecordedDraw.to_dict``) and the reply itself
  (``ModelResponse.to_dict``, private) on the outcome. ``evidence_request`` and
  ``evidence_outcome`` lines record the full-distribution evidence batches the
  same way.
- ``<item_id>.<rendering>.result.json``: the serialised ``GuardedRun`` of one
  complete ensemble (``GuardedRun.to_dict``), with the hashes it was recorded
  under; ``<item_id>.<rendering>.evidence.json``: one evidence batch.
- ``receipt.json``: the stage's host checks, its status, stop reason and exit
  code, and the teardown evidence of the hosts' Windows-side scripts.

Resume. Before any request for an item and rendering, a complete result whose
plan hash, prompt hash, settings, calibrator and host digest match, and whose
decision recomputes from its recorded draws, is reused and nothing is sent. A
partial record is completed draw by draw through ``guarded_run(recorded=...)``:
a recorded request is never issued again. A request line without an outcome
(the process died while that request was in flight, or just before it was sent)
is closed on resume as the failed draw ``request_error:interrupted`` and never
resent, so a request that may have reached the host is never replayed and no
recorded outcome is lost; the ledger has charged it its full reservation
(7.5). A request the ledger or the gateway refused before generation (admission
closed, HTTP 401 or 403) gets an outcome with ``sent: false``: it was never
generated, so its draw index stays open. A record under another plan hash is
refused (``PlanChangedError``), never merged; one under another prompt,
settings, calibrator or host digest is refused (``RecordingError``).

Admission. A :class:`StageSession` opens the stage's ``BudgetLedger`` and enters
it, so every admission runs inside ``with ledger:``. The generation client must
use a ``LedgerCallBudget`` on that ledger, send no retry (``max_retries`` 0) and
carry the allowance's ``max_tokens`` and timeout, so ``RequestAllowance``
reflects the real request (:func:`request_allowance`). Full-distribution
evidence is a separate pass after all generation, in an :class:`EvidenceSession`
that never opens the ledger, so evidence calls count against no request, token
or time limit (the plan's ``EVIDENCE_RULE``); each batch has its own timeout
(:class:`EvidencePolicy`, 600 s).

Hosts (11.1). The runner never starts or restarts a host; the Windows-side
scripts do. Before admitting requests the session requires a passed host check
whose expectation is the frozen plan's: llama-server through its gateway with
the bearer (``GET /v1/models``, then ``GET /props`` whose ``model_path``
basename, read with ``PureWindowsPath``, must be the pinned GGUF blob), the
full-distribution host (the client's digest must be the pinned one, then a
minimal evidence batch; the host answers 409 for other weights), and the
serving arm (``GET /v1/models`` on the Modal endpoint with the proxy bearer must
list the arm's registry id). The pin is the declared artifact digest of the
arm's frozen provider profile. A check that finds other weights, or a host that
is not ready, refuses the session before any request (exit 2); nothing changes
and the stage stays resumable.

Failures during a stage. Only a frozen limit closing admission makes a stage
incomplete (exit 3, the ledger's rule of 7.5). A host failure leaves the stage
open and resumable (exit 1) with everything recorded kept, and the host is not
restarted: once the operator restarts it with its script, a new session resumes
at the next unrecorded draw or the next missing evidence batch, never repeating
a generation request. A failed generation request is recorded on its draw; when
the host is then unreachable, or serves other weights, the session ends
(:class:`HostFailure`). For evidence batches, a ``host_busy`` refusal (503
``memory_low``) is retried a bounded number of times with a wait (5 times, 60 s
apart), then the session ends; a client timeout or transport failure is never
resent blindly: the batch is sent again only after the listener accepts
connections and answers a minimal probe (alive and idle), at most ``max_sends``
times per session. A dead host ends the session. A batch the host rejects for
its content is recorded with its reason as missing evidence; it is not sent
again in that session, and a rerun repeats only the missing batches. A
settled batch records its evidence reference per draw; a request whose outcome
was lost is settled by the later attempt that succeeds.

Serving arm. Admission is also gated on the serve receipt's deadline: a request
is admitted only while it can finish before the app's armed stop. One serve
lifecycle per stage per approval: the first serve receipt is bound to the stage
(``serve.json``) and another lifecycle's receipt is refused, so a crash and a
second serve cannot exceed the approved total.

Versioned draws. :class:`DrawLog` (``draws.jsonl``) records each version of a
draw that a caller reports through ``on_draw``, such as a calibration draw first
with its full-distribution evidence pending and again once the evidence pass
settles it with its evidence reference; the latest version of each draw wins
and is what a resume passes back as ``recorded``.

Teardown. :func:`record_teardown` records each host script's stop result in the
stage receipt with ``verified_stopped``: a Modal serve receipt's ``cleanup``, the
captured output of a lifecycle script's ``stop`` command, or a gateway log's
last leftover check.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import socket
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path, PureWindowsPath
from time import monotonic
from types import TracebackType
from typing import Any
from urllib.parse import urlsplit

import requests

from recall_guard.core.completion import ModelConfig, ModelResponse
from recall_guard.core.distribution import RecordedDraw
from recall_guard.core.full_distribution import FullDistributionEvidence
from recall_guard.core.profiles import load_profile
from recall_guard.core.request_control import BudgetExhaustedError, RequestError
from recall_guard.harness.guarded_decision import (
    REQUEST_ERROR_PREFIX,
    GuardConfig,
    GuardedRun,
    SourceCalibrator,
    decide_from_draws,
    guarded_run,
)
from recall_guard.harness.scorer_client import identity_settings
from scripts.finance_sim.limits import (
    ARMS,
    EXIT_COMPLETE,
    EXIT_INCOMPLETE,
    EXIT_REFUSED,
    EXIT_UNFINISHED,
    RECORDS_DIR,
    SAMPLING_ARM,
    SERVING_ARM,
    AdmissionClosedError,
    ApprovalError,
    BudgetLedger,
    LedgerCallBudget,
    RequestAllowance,
    ServingApproval,
    StudyLimits,
    budget_path,
)
from scripts.finance_sim.plan import PlanChangedError, StudyPlan, ensure_unchanged, require_complete, sha256_of

RECORD_SCHEMA = "finance-sim-request/1"
RESULT_SCHEMA = "finance-sim-result/1"
EVIDENCE_SCHEMA = "finance-sim-evidence/1"
RECEIPT_SCHEMA = "finance-sim-stage-receipt/1"
SERVE_BINDING_SCHEMA = "finance-sim-serve-binding/1"

REQUEST, OUTCOME = "request", "outcome"
EVIDENCE_REQUEST, EVIDENCE_OUTCOME = "evidence_request", "evidence_outcome"
LINE_KINDS: tuple[str, ...] = (REQUEST, OUTCOME, EVIDENCE_REQUEST, EVIDENCE_OUTCOME)
#: Failure code of a sent request whose outcome was lost; closed on resume, never resent.
INTERRUPTED_FAILURE = REQUEST_ERROR_PREFIX + "interrupted"
#: Prefix of the failure code of a request refused before generation (``sent: false``).
NOT_SENT_PREFIX = "not_sent:"
CLOSED_BY_RESUME = "resume"
#: Failure code of an evidence batch whose outcome was lost.
EVIDENCE_INTERRUPTED = "interrupted"

LLAMA_SERVER, FULL_DISTRIBUTION_HOST, SERVING_HOST = "llama_server", "full_distribution", "serving"
#: The hosts each arm checks, and the one that generates.
ARM_HOSTS: Mapping[str, tuple[str, ...]] = {SAMPLING_ARM: (LLAMA_SERVER, FULL_DISTRIBUTION_HOST),
                                            SERVING_ARM: (SERVING_HOST,)}
GENERATION_HOST: Mapping[str, str] = {SAMPLING_ARM: LLAMA_SERVER, SERVING_ARM: SERVING_HOST}

OK, UNREACHABLE, BUSY, UNAUTHORIZED, DIGEST_MISMATCH, INVALID = (
    "ok", "unreachable", "busy", "unauthorized", "digest_mismatch", "invalid")
CHECK_STATUSES: tuple[str, ...] = (OK, UNREACHABLE, BUSY, UNAUTHORIZED, DIGEST_MISMATCH, INVALID)

#: Stop reasons of a stage that a host ended; the frozen limits' reasons are the ledger's.
HOST_UNAVAILABLE = "host_unavailable"
HOST_BUSY = "host_busy"
EVIDENCE_UNRESOLVED = "evidence_timeout_unresolved"

#: Stage receipt statuses.
OPEN, COMPLETE, INCOMPLETE, REFUSED = "open", "complete", "incomplete", "refused"

#: The full-distribution gateway's upstream timeout; an evidence batch must finish well inside it.
GATEWAY_TIMEOUT_S = 1800.0
#: About one prompt evaluation plus every response's decodes of a 32-draw batch on the iGPU.
EVIDENCE_TIMEOUT_S = 600.0
CHECK_TIMEOUT_S = 10.0
PROBE_TIMEOUT_S = 30.0
#: The minimal evidence batch of the full-distribution check: a few tokens, about a second on an idle host.
PROBE_PROMPT = "Reply with one word."
PROBE_RESPONSE = "Ready."

_ITEM =re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}")
_RENDERING = re.compile(r"[a-z][a-z0-9_]{0,31}")
_WINDOWS_DEVICES = frozenset({"con", "prn", "aux", "nul", *(f"com{i}" for i in range(1, 10)),
                              *(f"lpt{i}" for i in range(1, 10))})
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")
_ENV_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_REGISTERED_ENDPOINT = re.compile(r"https://[a-zA-Z0-9-]+(?:\.[a-zA-Z0-9-]+)?\.modal\.run/v1")
_CHAT_PATH = "/chat/completions"
_RECEIPT_KEYS = frozenset({"schema", "plan_sha256", "arm", "stage", "status", "stop_reason", "exit_code", "sessions",
                           "last_session", "host_checks", "teardown", "teardown_verified", "created_at", "updated_at"})
_RESULT_KEYS = frozenset({"schema", "plan_sha256", "arm", "stage", "item_id", "rendering", "prompt_sha256",
                          "settings_json", "config", "source", "calibrator_fingerprint", "host_digest", "recorded_at",
                          "run"})
_EVIDENCE_KEYS = frozenset({"schema", "plan_sha256", "arm", "stage", "item_id", "rendering", "prompt_sha256",
                            "batch_sha256", "model_digest", "top_k_keep", "draw_indices", "evidence", "recorded_at"})
_BINDING_IDENTITY = ("plan_sha256", "stage", "app_id", "gpu_started_at_unix_s", "deadline_unix_s")
#: The fields of a recorded ensemble that every line and result must share with the run that resumes it.
_ENSEMBLE_CHECKS = (("prompt_sha256", "prompt"), ("settings_json", "generation settings"),
                    ("config", "ensemble settings"), ("source", "estimate source"),
                    ("calibrator_fingerprint", "calibrator"), ("host_digest", "host digest"))


# ---------- errors ----------

class RecordingError(ValueError):
    """A record that cannot be merged: another prompt, settings, calibrator or host digest, or a corrupt line."""


class HostCheckError(RuntimeError):
    """A host is not ready before admission; nothing was sent, and the stage is refused (exit 2)."""

    def __init__(self, message: str, *, check: HostCheck | None = None, reason: str | None = None) -> None:
        super().__init__(message)
        self.check = check
        self.reason = reason or (check.status if check is not None else "host_check")


class HostDigestError(HostCheckError):
    """The host serves weights other than the frozen plan pins; nothing was sent."""

    def __init__(self, message: str, *, check: HostCheck | None = None) -> None:
        super().__init__(message, check=check, reason=DIGEST_MISMATCH)


class HostFailure(RuntimeError):
    """A host failed during the stage: the session ends, the stage stays open (exit 1), the host is not restarted.

    No frozen limit was reached, so the ledger stays open and nothing recorded
    is removed: once the operator has restarted the host with its script, a new
    session resumes the recording. ``reason`` is the receipt's stop reason.
    """

    def __init__(self, reason: str, message: str) -> None:
        super().__init__(message)
        self.reason = reason
        self.exit_code = EXIT_UNFINISHED


class HttpTransportError(Exception):
    """A host GET failed in transport: ``unreachable`` (no listener) or ``timeout`` (no reply in time)."""

    def __init__(self, kind: str) -> None:
        if kind not in (UNREACHABLE, "timeout"):
            raise ValueError(f"kind must be unreachable or timeout; got {kind!r}")
        super().__init__(f"GET failed in transport ({kind})")
        self.kind = kind


# ---------- small helpers ----------

def _utc_now() -> datetime:
    return datetime.now(UTC)


def _iso(moment: datetime) -> str:
    return moment.astimezone(UTC).isoformat(timespec="seconds")


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    keys = [key for key, _ in pairs]
    if len(set(keys)) != len(keys):
        raise ValueError(f"duplicate keys {sorted({key for key in keys if keys.count(key) > 1})}")
    return dict(pairs)


def _no_constant(name: str) -> Any:
    raise ValueError(f"non-finite number {name}")


def _strict_json(text: str) -> Any:
    """Parse JSON refusing duplicate keys and NaN or infinity."""
    return json.loads(text, object_pairs_hook=_no_duplicate_keys, parse_constant=_no_constant)


def _read_json(path: Path, what: str) -> dict[str, Any]:
    try:
        data = _strict_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RecordingError(f"{path} is not a readable {what} ({exc}); refused, never merged") from None
    if not isinstance(data, dict):
        raise RecordingError(f"{path} does not hold a {what}; refused, never merged")
    return data


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write atomically: a crash leaves the previous file or the new one, never half of one."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _append_line(path: Path, payload: Mapping[str, Any]) -> None:
    """Append one JSON line and flush it to disk before returning."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")
    with path.open("ab") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def _repaired_lines(path: Path) -> list[str]:
    """The lines of an append-only record, after repairing a last line a crash cut short.

    A last line without its newline that is a complete JSON object gets its
    newline; one that is not was never a complete record and is dropped, so
    later appends stay line-aligned.
    """
    data = path.read_bytes()
    if data and not data.endswith(b"\n"):
        cut = data.rfind(b"\n") + 1
        try:
            complete = isinstance(_strict_json(data[cut:].decode("utf-8")), dict)
        except (UnicodeDecodeError, ValueError):
            complete = False
        with path.open("r+b") as stream:
            if complete:
                stream.seek(0, os.SEEK_END)
                stream.write(b"\n")
            else:
                stream.truncate(cut)
            stream.flush()
            os.fsync(stream.fileno())
        data = path.read_bytes()
    return data.decode("utf-8").splitlines()


def _item(value: object) -> str:
    if not isinstance(value, str) or not _ITEM.fullmatch(value) or value.lower() in _WINDOWS_DEVICES:
        raise ValueError(f"item_id must be letters, digits, '-' or '_' (at most 128, no device name); got {value!r}")
    return value


def _rendering(value: object) -> str:
    if not isinstance(value, str) or not _RENDERING.fullmatch(value):
        raise ValueError(f"rendering must be a lowercase identifier; got {value!r}")
    return value


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number; got {value!r}")
    return float(value)


def _digest(value: object) -> str:
    if not isinstance(value, str) or not _DIGEST.fullmatch(value):
        raise ValueError(f"a model digest must be sha256:<64 lowercase hex>; got {value!r}")
    return value


# ---------- paths ----------

def stage_directory(arm: str, stage: str, records_dir: str | Path = RECORDS_DIR) -> Path:
    """``<records_dir>/<arm>/<stage>/``: the stage's records, results, receipt and ``budget.json``."""
    return budget_path(arm, stage, records_dir).parent


def requests_path(arm: str, stage: str, item_id: str, records_dir: str | Path = RECORDS_DIR) -> Path:
    """The item's append-only request record, every rendering in one file."""
    return stage_directory(arm, stage, records_dir) / f"{_item(item_id)}.jsonl"


def result_path(arm: str, stage: str, item_id: str, rendering: str, records_dir: str | Path = RECORDS_DIR) -> Path:
    return stage_directory(arm, stage, records_dir) / f"{_item(item_id)}.{_rendering(rendering)}.result.json"


def evidence_path(arm: str, stage: str, item_id: str, rendering: str, records_dir: str | Path = RECORDS_DIR) -> Path:
    return stage_directory(arm, stage, records_dir) / f"{_item(item_id)}.{_rendering(rendering)}.evidence.json"


def receipt_path(arm: str, stage: str, records_dir: str | Path = RECORDS_DIR) -> Path:
    return stage_directory(arm, stage, records_dir) / "receipt.json"


def serve_binding_path(stage: str, records_dir: str | Path = RECORDS_DIR) -> Path:
    """The serve lifecycle bound to a serving-arm stage: one serve per stage per approval."""
    return stage_directory(SERVING_ARM, stage, records_dir) / "serve.json"


def read_records(arm: str, stage: str, item_id: str, records_dir: str | Path = RECORDS_DIR) -> list[dict[str, Any]]:
    """The item's record lines as written (read only; nothing is repaired or validated against a plan)."""
    path = requests_path(arm, stage, item_id, records_dir)
    if not path.exists():
        return []
    return [_strict_json(line) for line in path.read_bytes().decode("utf-8").splitlines() if line.strip()]


# ---------- the plan's pins and the allowance ----------

def pinned_digest(plan: StudyPlan, arm: str = SAMPLING_ARM) -> str:
    """The GGUF digest the sampling arm's frozen profile declares; both local hosts must serve it.

    The plan freezes the arm's ``profile_id``; the profile declares the artifact
    digest, and its full-distribution host must name the same digest.
    """
    require_complete(plan)
    if arm != SAMPLING_ARM:
        raise ValueError("only the sampling arm runs a pinned local GGUF; the serving arm is checked by its registry id")
    profile = load_profile(plan.arms.arms[arm].profile_id)
    identity = profile.model_config.local_identity
    digest = _digest(None if identity is None else identity.declared.artifact_digest)
    host = profile.full_distribution
    if host is not None and host.model_digest != digest:
        raise ValueError(f"profile {profile.profile_id}: the full-distribution host names {host.model_digest}, the "
                         f"generation model {digest}")
    return digest


def request_allowance(plan: StudyPlan, arm: str, config: ModelConfig, *,
                      usd_per_second: float | None = None) -> RequestAllowance:
    """The allowance of one generation request: the arm's frozen ``max_tokens`` and the client's real timeout.

    The client must send exactly the arm's ``max_tokens`` and no retry, so the
    reservation is what one admitted request can use.
    """
    require_complete(plan)
    if arm not in ARMS:
        raise ValueError(f"arm must be one of {list(ARMS)}; got {arm!r}")
    if not isinstance(config, ModelConfig):
        raise TypeError(f"config must be a ModelConfig; got {type(config).__name__}")
    max_tokens = plan.arms.arms[arm].max_tokens
    if config.parameters.get("max_tokens") != max_tokens:
        raise ValueError(f"the client sends max_tokens {config.parameters.get('max_tokens')!r}; the {arm} arm's "
                         f"frozen max_tokens is {max_tokens}")
    if config.max_retries != 0:
        raise ValueError(f"the client retries (max_retries={config.max_retries}); a retry would resend a study "
                         "request, so max_retries must be 0")
    return RequestAllowance(output_tokens=max_tokens, timeout_s=config.timeout_s, usd_per_second=usd_per_second)


# ---------- host checks ----------

@dataclass(frozen=True)
class HttpReply:
    status: int
    body: Any


HttpGet = Callable[[str, Mapping[str, str], float], HttpReply]


def http_get(url: str, headers: Mapping[str, str], timeout_s: float) -> HttpReply:
    """One GET with retries and redirects disabled and ambient proxies or netrc ignored.

    Raises :class:`HttpTransportError` without the original exception, which
    could carry a header value.
    """
    kind = UNREACHABLE
    try:
        with requests.Session() as session:
            session.trust_env = False
            session.mount("http://", requests.adapters.HTTPAdapter(max_retries=0))
            session.mount("https://", requests.adapters.HTTPAdapter(max_retries=0))
            reply = session.get(url, headers=dict(headers), timeout=timeout_s, allow_redirects=False)
            try:
                body = reply.json()
            except ValueError:
                body = None
            return HttpReply(reply.status_code, body)
    except requests.ConnectTimeout:
        kind = UNREACHABLE
    except requests.Timeout:
        kind = "timeout"
    except (requests.RequestException, UnicodeError):
        kind = UNREACHABLE
    raise HttpTransportError(kind)


@dataclass(frozen=True)
class HostCheck:
    """One host check: what was expected, what the host reported, and when.

    ``url`` is the credential-free endpoint the check used (the generation
    client's endpoint, or the full-distribution gateway's base URL); ``expected``
    is the pin it was checked against and ``observed`` what the host reported.
    """

    host: str
    url: str
    status: str
    expected: str
    observed: str | None
    detail: str | None
    checked_at: str

    def __post_init__(self) -> None:
        if self.host not in (LLAMA_SERVER, FULL_DISTRIBUTION_HOST, SERVING_HOST):
            raise ValueError(f"host must be one of {[LLAMA_SERVER, FULL_DISTRIBUTION_HOST, SERVING_HOST]}; "
                             f"got {self.host!r}")
        if self.status not in CHECK_STATUSES:
            raise ValueError(f"status must be one of {list(CHECK_STATUSES)}; got {self.status!r}")

    @property
    def ok(self) -> bool:
        return self.status == OK

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _bearer(credential_ref: object, environment: Mapping[str, str] | None) -> str:
    """The referenced bearer, refused before any request when missing or not a visible-ASCII token."""
    environment = os.environ if environment is None else environment
    name = credential_ref.removeprefix("env:") if isinstance(credential_ref, str) else None
    if not name or not _ENV_NAME.fullmatch(name):
        raise ValueError("the client's credential_ref must name an environment variable")
    value = environment.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"the bearer {name} is missing or empty; nothing was sent")
    if value != value.strip() or not value.isascii() or any(char.isspace() or not char.isprintable() for char in value):
        raise ValueError(f"the bearer {name} contains whitespace, control or non-ASCII characters; nothing was sent")
    return value


def _fetch(get: HttpGet, url: str, path: str, headers: Mapping[str, str],
           timeout_s: float) -> tuple[str | None, HttpReply | None, str | None]:
    """``(failed status, reply, detail)``; the status is None when the GET returned 200."""
    try:
        reply = get(url, headers, timeout_s)
    except HttpTransportError as error:
        return (UNREACHABLE if error.kind == UNREACHABLE else BUSY), None, f"GET {path} failed in transport ({error.kind})"
    if reply.status in (401, 403):
        return UNAUTHORIZED, reply, f"GET {path}: HTTP {reply.status}"
    if reply.status == 503:
        return BUSY, reply, f"GET {path}: HTTP 503"
    if reply.status != 200:
        return INVALID, reply, f"GET {path}: HTTP {reply.status}"
    return None, reply, None


def _chat_base(config: object) -> str:
    """The ``.../v1`` base of an OpenAI-compatible chat-completions endpoint."""
    if not isinstance(config, ModelConfig):
        raise TypeError(f"config must be a ModelConfig; got {type(config).__name__}")
    endpoint = config.endpoint
    if not isinstance(endpoint, str) or not endpoint.endswith("/v1" + _CHAT_PATH):
        raise ValueError(f"the generation endpoint must end with /v1{_CHAT_PATH}; got {endpoint!r}")
    return endpoint.removesuffix(_CHAT_PATH)


def check_llama_server(config: ModelConfig, *, expected_digest: str, environment: Mapping[str, str] | None = None,
                       get: HttpGet = http_get, timeout_s: float = CHECK_TIMEOUT_S,
                       now: Callable[[], datetime] = _utc_now) -> HostCheck:
    """llama-server through its gateway: ``GET /v1/models`` with the bearer, then the GGUF named by ``GET /props``.

    ``model_path`` is an absolute Windows path; its basename (``PureWindowsPath``)
    must be the pinned blob ``sha256-<hex>``. The bearer is the client's
    ``credential_ref``; it is sent, never recorded.
    """
    expected = _digest(expected_digest)
    base = _chat_base(config)
    root = base.removesuffix("/v1")
    headers = {"Authorization": f"Bearer {_bearer(config.credential_ref, environment)}"}

    def result(status: str, observed: str | None = None, detail: str | None = None) -> HostCheck:
        return HostCheck(LLAMA_SERVER, config.endpoint, status, expected, observed, detail, _iso(now()))

    status, reply, detail = _fetch(get, base + "/models", "/v1/models", headers, timeout_s)
    if status is not None:
        return result(status, detail=detail)
    data = reply.body.get("data") if isinstance(reply.body, Mapping) else None
    if not isinstance(data, list) or not data:
        return result(INVALID, detail="GET /v1/models listed no model")
    status, reply, detail = _fetch(get, root + "/props", "/props", headers, timeout_s)
    if status is not None:
        return result(status, detail=detail)
    model_path = reply.body.get("model_path") if isinstance(reply.body, Mapping) else None
    if not isinstance(model_path, str) or not model_path.strip():
        return result(INVALID, detail="GET /props named no model_path")
    blob = PureWindowsPath(model_path).name
    pinned = "sha256-" + expected.removeprefix("sha256:")
    if blob != pinned:
        return result(DIGEST_MISMATCH, observed=blob, detail=f"/props model_path names {blob}; the pin is {pinned}")
    return result(OK, observed=expected)


def check_serving_host(config: ModelConfig, *, expected_model: str, environment: Mapping[str, str] | None = None,
                       get: HttpGet = http_get, timeout_s: float = CHECK_TIMEOUT_S,
                       now: Callable[[], datetime] = _utc_now) -> HostCheck:
    """The Modal endpoint: ``GET /v1/models`` with the proxy bearer must list ``expected_model``."""
    if not isinstance(expected_model, str) or not expected_model:
        raise ValueError("expected_model must name the served model")
    base = _chat_base(config)
    headers = {"Authorization": f"Bearer {_bearer(config.credential_ref, environment)}"}

    def result(status: str, observed: str | None = None, detail: str | None = None) -> HostCheck:
        return HostCheck(SERVING_HOST, config.endpoint, status, expected_model, observed, detail, _iso(now()))

    status, reply, detail = _fetch(get, base + "/models", "/v1/models", headers, timeout_s)
    if status is not None:
        return result(status, detail=detail)
    data = reply.body.get("data") if isinstance(reply.body, Mapping) else None
    if not isinstance(data, list):
        return result(INVALID, detail="GET /v1/models returned no model list")
    ids = [entry["id"] for entry in data if isinstance(entry, Mapping) and isinstance(entry.get("id"), str)]
    if expected_model not in ids:
        return result(DIGEST_MISMATCH, observed=", ".join(ids) or None,
                      detail=f"GET /v1/models lists {ids}, not {expected_model}")
    return result(OK, observed=expected_model)


def check_full_distribution_host(client: Any, *, expected_digest: str, timeout_s: float = PROBE_TIMEOUT_S,
                                 connect: Callable[..., Any] | None = None,
                                 now: Callable[[], datetime] = _utc_now) -> HostCheck:
    """The full-distribution host through its gateway: alive, idle and serving the pinned digest.

    A client for other weights is refused without a request. Otherwise the
    gateway port must accept a connection (alive) and a minimal evidence batch
    must come back within ``timeout_s`` (idle; the host is serial, so a host
    still computing an earlier batch answers late and reads as ``busy``). The
    host answers 409 for other weights and 503 below its memory floor.
    """
    expected = _digest(expected_digest)
    base = getattr(client, "base_url", None)
    if not isinstance(base, str):
        raise TypeError("client must be a full-distribution client with a base_url")
    observed = getattr(client, "model_digest", None)

    def result(status: str, observed: str | None = None, detail: str | None = None) -> HostCheck:
        return HostCheck(FULL_DISTRIBUTION_HOST, base, status, expected, observed, detail, _iso(now()))

    if observed != expected:
        return result(DIGEST_MISMATCH, observed=observed,
                      detail="the client requests other weights than the pin; nothing was sent")
    parts = urlsplit(base)
    address = (parts.hostname, parts.port or (443 if parts.scheme == "https" else 80))
    try:
        (connect or socket.create_connection)(address, timeout_s).close()
    except OSError:
        return result(UNREACHABLE, detail=f"nothing accepts connections on {address[0]}:{address[1]}")
    try:
        batch = client.evidence_batch(PROBE_PROMPT, [PROBE_RESPONSE], top_k_keep=1, timeout_s=timeout_s)
    except RequestError as error:
        status = {"digest_mismatch": DIGEST_MISMATCH, "auth_failed": UNAUTHORIZED, "host_busy": BUSY,
                  "transport_error": BUSY}.get(error.code, INVALID)
        return result(status, detail=f"probe batch refused ({error.code})")
    if len(batch) != 1 or batch[0].model_digest != expected:
        return result(DIGEST_MISMATCH, detail="the probe batch came back for other weights")
    return result(OK, observed=expected)


# ---------- the serve lease (serving arm) ----------

def _refused(detail: str) -> ApprovalError:
    return ApprovalError(f"Serving arm refused before any request: {detail}")


@dataclass(frozen=True)
class ServeLease:
    """The serve lifecycle of one serving-arm stage, from its serve receipt.

    A request is admitted only while it can finish before ``deadline_unix_s``,
    the expiry controller's armed stop (GPU start plus the stage's
    ``max_seconds``). One lifecycle per stage per approval: the first receipt is
    bound to the stage and another lifecycle's receipt is refused.
    """

    stage: str
    receipt: str
    app_id: str
    endpoint_url: str
    gpu_started_at_unix_s: float
    deadline_unix_s: float

    @classmethod
    def from_receipt(cls, path: str | Path, *, plan: StudyPlan, approval: ServingApproval, stage: str) -> ServeLease:
        """Validate a serve receipt against the approved stage limits; local files only."""
        if not isinstance(approval, ServingApproval):
            raise _refused(f"stage {stage} needs the validated approval record of plan {plan.plan_sha256}")
        ensure_unchanged(plan, approval.plan_sha256)
        target = Path(path)
        try:
            receipt = _strict_json(target.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise _refused(f"the serve receipt {target} is unreadable ({exc})") from None
        if not isinstance(receipt, dict) or receipt.get("mode") != "serve":
            raise _refused(f"{target} is not a serve receipt")
        if receipt.get("status") != "serving":
            raise _refused(f"the serve receipt {target} has status {receipt.get('status')!r}, not 'serving'; the app "
                           "is not reachable for study requests")
        ensure_unchanged(plan, receipt.get("plan_sha256"))
        limits = approval.limits_for(stage)
        expected = {"plan_sha256": limits.plan_sha256, "stage": stage, "max_requests": limits.max_requests,
                    "max_output_tokens": limits.max_output_tokens, "max_seconds": limits.max_seconds,
                    "max_usd": limits.max_usd}
        if receipt.get("stage_limits") != expected:
            raise _refused(f"the serve receipt's stage limits differ from the approved {stage} stage limits")
        stage_sha256 = next(item.sha256 for item in approval.presentation.stages if item.stage == stage)
        if receipt.get("stage_limits_sha256") != stage_sha256:
            raise _refused(f"the serve receipt names stage limits file sha256 {receipt.get('stage_limits_sha256')!r}, "
                           f"not the approved {stage} stage limits file {stage_sha256}")
        if receipt.get("manifest_sha256") != approval.presentation.manifest_sha256:
            raise _refused("the serve receipt's manifest_sha256 is not the approved manifest's")
        app_id = receipt.get("app_id")
        if not isinstance(app_id, str) or not app_id:
            raise _refused("the serve receipt names no app_id")
        try:
            started = _finite(receipt.get("gpu_started_at_unix_s"), "gpu_started_at_unix_s")
            deadline = _finite(receipt.get("deadline_unix_s"), "deadline_unix_s")
        except ValueError as exc:
            raise _refused(f"the serve receipt has no armed deadline ({exc})") from None
        if not started < deadline <= started + limits.max_seconds + 1e-6:
            raise _refused(f"the serve receipt's deadline {deadline} is not within the approved {limits.max_seconds} s "
                           f"of the GPU start {started}")
        endpoint = receipt.get("endpoint_url")
        if not isinstance(endpoint, str) or not _REGISTERED_ENDPOINT.fullmatch(endpoint) or (
                approval.presentation.endpoint is not None and endpoint != approval.presentation.endpoint):
            raise _refused(f"the serve receipt's endpoint_url {endpoint!r} is not the approved endpoint "
                           f"{approval.presentation.endpoint!r}")
        return cls(stage, str(target), app_id, endpoint, started, deadline)

    def admits(self, now_unix_s: float, timeout_s: float) -> bool:
        """True when a request started now can finish before the armed stop."""
        return now_unix_s + timeout_s <= self.deadline_unix_s

    def _binding(self, plan_sha256: str) -> dict[str, Any]:
        return {"plan_sha256": plan_sha256, "stage": self.stage, "app_id": self.app_id,
                "gpu_started_at_unix_s": self.gpu_started_at_unix_s, "deadline_unix_s": self.deadline_unix_s}

    def check_binding(self, path: Path, plan_sha256: str) -> None:
        """Refuse a receipt of another lifecycle than the one already bound to this stage."""
        if not path.exists():
            return
        bound = _read_json(path, "serve binding")
        if bound.get("schema") != SERVE_BINDING_SCHEMA or any(
                bound.get(name) != value for name, value in self._binding(plan_sha256).items()):
            raise _refused(f"one serve per stage per approval: stage {self.stage} is bound to app "
                           f"{bound.get('app_id')!r} (deadline {bound.get('deadline_unix_s')}); the serve receipt "
                           f"{self.receipt} is another lifecycle, which could exceed the approved total")

    def bind(self, path: Path, plan_sha256: str, now: datetime) -> None:
        self.check_binding(path, plan_sha256)
        if not path.exists():
            _write_json(path, {"schema": SERVE_BINDING_SCHEMA, **self._binding(plan_sha256), "receipt": self.receipt,
                               "bound_at": _iso(now)})


# ---------- evidence policy ----------

@dataclass(frozen=True)
class EvidencePolicy:
    """How evidence batches are sent: their own timeout, the busy retries and the idle check before a resend.

    ``max_sends`` bounds, per session, the batches sent for one item and
    rendering whose outcome was not a ``host_busy`` refusal (a refused batch was
    never computed).
    """

    timeout_s: float = EVIDENCE_TIMEOUT_S
    top_k_keep: int = 100
    max_busy_retries: int = 5
    busy_wait_s: float = 60.0
    max_sends: int = 2
    idle_wait_s: float = 1200.0
    idle_poll_s: float = 30.0

    def __post_init__(self) -> None:
        timeout = _finite(self.timeout_s, "timeout_s")
        if not 0 < timeout < GATEWAY_TIMEOUT_S:
            raise ValueError(f"timeout_s must be above 0 and below the gateway's {GATEWAY_TIMEOUT_S} s; got {timeout}")
        for name, minimum in (("top_k_keep", 1), ("max_busy_retries", 0), ("max_sends", 1)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer of at least {minimum}; got {value!r}")
        for name in ("busy_wait_s", "idle_wait_s", "idle_poll_s"):
            if _finite(getattr(self, name), name) < 0:
                raise ValueError(f"{name} must not be negative")


# ---------- the stage receipt ----------

def _new_receipt(plan: StudyPlan, arm: str, stage: str, created: datetime) -> dict[str, Any]:
    stamp = _iso(created)
    return {"schema": RECEIPT_SCHEMA, "plan_sha256": plan.plan_sha256, "arm": arm, "stage": stage, "status": OPEN,
            "stop_reason": None, "exit_code": EXIT_UNFINISHED, "sessions": 0, "last_session": None,
            "host_checks": [], "teardown": [], "teardown_verified": None, "created_at": stamp, "updated_at": stamp}


def _read_receipt(path: Path, plan: StudyPlan, arm: str, stage: str) -> dict[str, Any]:
    data = _read_json(path, "stage receipt")
    if set(data) != _RECEIPT_KEYS or data["schema"] != RECEIPT_SCHEMA:
        raise RecordingError(f"{path} is not a {RECEIPT_SCHEMA} receipt; refused, never merged")
    if data["plan_sha256"] != plan.plan_sha256:
        raise PlanChangedError(f"{path} was written under plan_sha256 {data['plan_sha256']}; this plan is "
                               f"{plan.plan_sha256}; refused, never merged")
    if (data["arm"], data["stage"]) != (arm, stage):
        raise RecordingError(f"{path} is the receipt of {data['arm']}/{data['stage']}, not {arm}/{stage}")
    if not isinstance(data["host_checks"], list) or not isinstance(data["teardown"], list):
        raise RecordingError(f"{path}: host_checks and teardown must be lists")
    return data


def _load_receipt(plan: StudyPlan, arm: str, stage: str, records_dir: Path, now: datetime) -> tuple[Path, dict]:
    path = receipt_path(arm, stage, records_dir)
    return path, (_read_receipt(path, plan, arm, stage) if path.exists() else _new_receipt(plan, arm, stage, now))


# ---------- teardown evidence ----------

_STOPPED = "Stopped: no host or gateway process remains"
_LEFTOVER = " leftover check: "
_NONE_REMAINS = "no host or gateway process remains"


def _text_of(data: bytes) -> str:
    if data[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return data.decode("utf-16", errors="replace")
    return data.decode("utf-8-sig", errors="replace")


def teardown_evidence(path: str | Path, *, now: Callable[[], datetime] = _utc_now) -> dict[str, Any]:
    """Read one host script's stop result as teardown evidence with ``verified_stopped``.

    - A Modal serve receipt (JSON, ``mode`` serve): verified when ``cleanup`` is
      ``verified_stopped`` (``stop-serve`` found no app and no controller left).
    - A gateway log with leftover checks (the gateway's own idle or host-exit
      stop): verified when the last check found that no host or gateway process
      remains.
    - Otherwise the captured output of a lifecycle script's ``stop`` command:
      verified when it reports ``Stopped: no host or gateway process remains``
      and names nothing still running or listening.
    """
    target = Path(path)
    data = target.read_bytes()
    text = _text_of(data)
    entry = {"path": str(target), "sha256": hashlib.sha256(data).hexdigest(), "recorded_at": _iso(now())}
    if target.suffix.lower() == ".json":
        try:
            receipt = _strict_json(text)
        except ValueError as exc:
            raise ValueError(f"{target} is not a JSON serve receipt ({exc})") from None
        if not isinstance(receipt, dict) or receipt.get("mode") != "serve":
            raise ValueError(f"{target} is not a Modal serve receipt")
        cleanup = receipt.get("cleanup")
        detail = f"status {receipt.get('status')!r}, cleanup {cleanup!r}"
        if receipt.get("recovery_command"):
            detail += ", recovery command recorded"
        return entry | {"kind": "modal_serve_receipt", "verified_stopped": cleanup == "verified_stopped",
                        "detail": detail}
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    leftover = [line for line in lines if _LEFTOVER in line]
    if leftover:
        stamp = leftover[-1].split(" ", 1)[0]
        messages = [line.split(_LEFTOVER, 1)[1] for line in leftover if line.split(" ", 1)[0] == stamp]
        verified = len(messages) == 1 and messages[0].startswith(_NONE_REMAINS)
        return entry | {"kind": "gateway_log", "verified_stopped": verified,
                        "detail": f"leftover check at {stamp}: " + "; ".join(messages)[:500]}
    still = [line for line in lines if line.startswith("STILL ")]
    stopped = any(line.startswith(_STOPPED) for line in lines)
    detail = "; ".join(still)[:500] if still else (lines[-1][:500] if lines else "empty stop output")
    return entry | {"kind": "stop_output", "verified_stopped": stopped and not still, "detail": detail}


def record_teardown(plan: StudyPlan, arm: str, stage: str, paths: Sequence[str | Path], *,
                    records_dir: str | Path = RECORDS_DIR, now: Callable[[], datetime] = _utc_now) -> list[dict]:
    """Record the hosts' stop results in the stage receipt, after the stage's session has ended."""
    if isinstance(paths, str | Path) or not isinstance(paths, Sequence) or not paths:
        raise ValueError("paths must list at least one stop result")
    path, receipt = _load_receipt(plan, arm, stage, Path(records_dir), now())
    entries = [teardown_evidence(item, now=now) for item in paths]
    receipt["teardown"].extend(entries)
    receipt["teardown_verified"] = all(entry["verified_stopped"] for entry in entries)
    receipt["updated_at"] = _iso(now())
    _write_json(path, receipt)
    return entries


# ---------- versioned draw records (the calibration seam) ----------

DRAW_LOG_SCHEMA = "finance-sim-draw-version/1"


def draw_log_path(arm: str, stage: str, records_dir: str | Path = RECORDS_DIR) -> Path:
    """The stage's versioned draw log, ``draws.jsonl``, beside its request records."""
    return stage_directory(arm, stage, records_dir) / "draws.jsonl"


class DrawLog:
    """Append-only private log of draw versions in which the latest version of each draw wins.

    A draw is any record with a ``key`` (a tuple of JSON scalars naming one draw
    of one prompt), ``to_dict`` and a ``from_dict`` (``loads``), such as the
    calibration draws of task 7.6. A draw made before its full-distribution
    evidence is recorded as a pending version; when the evidence pass settles
    it, the settled version (with its per-source evidence reference) is
    appended and replaces it. Nothing is rewritten, so no recorded version is
    lost. Use the log itself as the ``on_draw`` callback and pass
    :meth:`latest` as ``recorded`` on resume, so no recorded draw is requested
    again. Lines under another plan hash are refused, never merged.
    """

    def __init__(self, path: str | Path, *, plan: StudyPlan, loads: Callable[[Mapping[str, Any]], Any],
                 now: Callable[[], datetime] = _utc_now) -> None:
        if not isinstance(plan, StudyPlan):
            raise TypeError(f"plan must be a StudyPlan; got {type(plan).__name__}")
        if not callable(loads):
            raise TypeError("loads must turn a recorded payload back into a draw (its from_dict)")
        self.path = Path(path)
        self.plan_sha256 = plan.plan_sha256
        self._loads = loads
        self._now = now

    @staticmethod
    def _key(draw: Any) -> list[Any]:
        key = getattr(draw, "key", None)
        if not isinstance(key, tuple) or not key or not all(
                value is None or isinstance(value, bool | int | str) for value in key):
            raise TypeError("a recorded draw needs a key: a non-empty tuple of JSON scalars")
        return list(key)

    def __call__(self, draw: Any) -> None:
        """Record one version of ``draw``; flushed to disk before returning."""
        payload = draw.to_dict()
        if self._key(self._loads(payload)) != self._key(draw):
            raise RecordingError("a draw must read back from its payload unchanged to be recorded")
        _append_line(self.path, {"schema": DRAW_LOG_SCHEMA, "plan_sha256": self.plan_sha256, "key": self._key(draw),
                                 "recorded_at": _iso(self._now()), "draw": payload})

    def latest(self) -> list[Any]:
        """The latest recorded version of each draw, in the order the draws were first recorded."""
        if not self.path.exists():
            return []
        versions: dict[tuple, Any] = {}
        for number, raw in enumerate(_repaired_lines(self.path), start=1):
            where = f"{self.path} line {number}"
            try:
                line = _strict_json(raw)
            except ValueError as exc:
                raise RecordingError(f"{where} is not valid JSON ({exc}); refused") from None
            if not isinstance(line, dict) or line.get("schema") != DRAW_LOG_SCHEMA:
                raise RecordingError(f"{where} is not a {DRAW_LOG_SCHEMA} line; refused")
            if line.get("plan_sha256") != self.plan_sha256:
                raise PlanChangedError(f"{where} was recorded under plan_sha256 {line.get('plan_sha256')}; this plan "
                                       f"is {self.plan_sha256}; refused, never merged")
            try:
                draw = self._loads(line["draw"])
            except (KeyError, TypeError, ValueError) as exc:
                raise RecordingError(f"{where}: the recorded draw cannot be read ({exc}); refused") from None
            if self._key(draw) != line.get("key"):
                raise RecordingError(f"{where}: the recorded key does not match its draw; refused")
            versions[tuple(line["key"])] = draw  # A later version replaces an earlier one in place.
        return list(versions.values())


# ---------- the stage session ----------

@dataclass(frozen=True)
class _Ensemble:
    """What every line and the result of one item and rendering are recorded under."""

    item_id: str
    rendering: str
    prompt_sha256: str
    settings_json: str
    config: dict[str, Any]
    source: dict[str, Any]
    calibrator_fingerprint: str
    host_digest: str
    temperature: float | None


def _not_sent_reason(error: BaseException) -> str | None:
    """Why a request was refused before generation, or None when it may have been generated."""
    if isinstance(error, BudgetExhaustedError):
        return "admission_closed"
    if getattr(error, "status_code", None) in (401, 403):
        return "auth_failed"
    return None


def _ledger_state(path: Path) -> tuple[str, str | None]:
    """The stage ledger's status and stop reason as persisted (read only); open before its first session."""
    if not path.exists():
        return "open", None
    data = _read_json(path, "budget ledger")
    if data.get("status") not in ("open", "complete", "incomplete"):
        raise RecordingError(f"{path} records no ledger status; refused")
    return data["status"], data.get("stop_reason")


def _stage_outcome(status: str, stop_reason: str | None) -> tuple[str, str | None, int]:
    """The receipt's status, stop reason and exit code from the ledger's; only a frozen limit is incomplete."""
    if status == "complete":
        return COMPLETE, None, EXIT_COMPLETE
    if status == "incomplete":
        return INCOMPLETE, stop_reason, EXIT_INCOMPLETE
    return OPEN, None, EXIT_UNFINISHED


class _StageRecords:
    """What both sessions of one arm and stage share: the records, the stage receipt and the host checks.

    A session is a context manager. On exit the stage receipt records its
    status and exit code: complete (0); open (1: an interruption, or a host
    failure whose reason ``stop_reason`` names; a new session resumes); refused
    before any request (2: a host check or the approval; nothing changed); or
    incomplete (3: a frozen limit closed admission, the only incomplete stage).
    """

    _KIND = "stage"
    _COUNTERS: tuple[str, ...] = ()

    def __init__(self, plan: StudyPlan, arm: str, stage: str, *, records_dir: Path, receipt: tuple[Path, dict],
                 clock: Callable[[], float], now: Callable[[], datetime], sleep: Callable[[float], None]) -> None:
        self.plan, self.arm, self.stage = plan, arm, stage
        self.records_dir = records_dir
        self._receipt_path, self._receipt = receipt
        self._clock, self._now, self._sleep = clock, now, sleep
        self._active = False
        self._ended = False
        self._hosts: dict[str, HostCheck] = {}
        self._pins: dict[str, str] = {}

    # ----- state -----

    @property
    def directory(self) -> Path:
        return stage_directory(self.arm, self.stage, self.records_dir)

    @property
    def status(self) -> str:
        return self._receipt["status"]

    @property
    def exit_code(self) -> int:
        return self._receipt["exit_code"]

    @property
    def receipt(self) -> dict[str, Any]:
        """A detached copy of the stage receipt as it stands now."""
        return json.loads(json.dumps(self._receipt))

    def __enter__(self) -> _StageRecords:
        if self._ended:
            raise RuntimeError("this session has ended; open a new one to resume")
        self._enter()
        self._active = True
        self._receipt["sessions"] += 1
        self._receipt["last_session"] = {"session": self._receipt["sessions"], "kind": self._KIND,
                                         "opened_at": _iso(self._now()), "closed_at": None,
                                         **dict.fromkeys(self._COUNTERS, 0)}
        self._write_receipt()
        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc: BaseException | None,
                 traceback: TracebackType | None) -> None:
        try:
            self._exit(exc_type, exc, traceback)
        finally:
            self._active = False
            self._ended = True
            self._receipt["last_session"]["closed_at"] = _iso(self._now())
            if isinstance(exc, HostFailure):
                outcome = (OPEN, exc.reason, EXIT_UNFINISHED)
            elif isinstance(exc, HostCheckError):
                outcome = (REFUSED, exc.reason, EXIT_REFUSED)
            elif isinstance(exc, ApprovalError):
                outcome = (REFUSED, "approval", EXIT_REFUSED)
            else:
                outcome = self._outcome()
            self._receipt["status"], self._receipt["stop_reason"], self._receipt["exit_code"] = outcome
            self._write_receipt()

    def _enter(self) -> None:
        """Enter what the session holds open (the generation session's ledger)."""

    def _exit(self, exc_type: type[BaseException] | None, exc: BaseException | None,
              traceback: TracebackType | None) -> None:
        """Leave what the session holds open."""

    def _outcome(self) -> tuple[str, str | None, int]:
        return _stage_outcome(*_ledger_state(budget_path(self.arm, self.stage, self.records_dir)))

    def _write_receipt(self) -> None:
        self._receipt["updated_at"] = _iso(self._now())
        _write_json(self._receipt_path, self._receipt)

    def _count(self, name: str) -> None:
        self._receipt["last_session"][name] += 1

    def _require_active(self) -> None:
        if not self._active:
            raise RuntimeError("use the session inside `with session:`; a stage session then admits every request "
                               "inside the ledger's session (with ledger:)")

    # ----- records -----

    def _header(self, kind: str, item_id: str, rendering: str) -> dict[str, Any]:
        return {"schema": RECORD_SCHEMA, "kind": kind, "plan_sha256": self.plan.plan_sha256, "arm": self.arm,
                "stage": self.stage, "item_id": item_id, "rendering": rendering}

    def _read_lines(self, item_id: str) -> tuple[Path, list[dict[str, Any]]]:
        """The item's record lines, checked against this plan, arm and stage.

        A last line cut short by a crash (no newline, not a JSON object) was never
        a complete record; it is dropped so appends stay line-aligned. Any other
        unreadable line refuses the record.
        """
        path = requests_path(self.arm, self.stage, item_id, self.records_dir)
        if not path.exists():
            return path, []
        lines = []
        for number, raw in enumerate(_repaired_lines(path), start=1):
            where = f"{path} line {number}"
            try:
                line = _strict_json(raw)
            except ValueError as exc:
                raise RecordingError(f"{where} is not valid JSON ({exc}); refused") from None
            if not isinstance(line, dict) or line.get("schema") != RECORD_SCHEMA or line.get("kind") not in LINE_KINDS:
                raise RecordingError(f"{where} is not a {RECORD_SCHEMA} record line; refused")
            if line.get("plan_sha256") != self.plan.plan_sha256:
                raise PlanChangedError(f"{where} was recorded under plan_sha256 {line.get('plan_sha256')}; this plan "
                                       f"is {self.plan.plan_sha256}; refused, never merged")
            if (line.get("arm"), line.get("stage"), line.get("item_id")) != (self.arm, self.stage, item_id):
                raise RecordingError(f"{where} belongs to {line.get('arm')}/{line.get('stage')}/{line.get('item_id')}; "
                                     "refused, never merged")
            lines.append(line)
        return path, lines

    # ----- teardown -----

    def record_teardown(self, paths: Sequence[str | Path]) -> list[dict]:
        """Record the hosts' stop results in this stage's receipt (see :func:`teardown_evidence`)."""
        if isinstance(paths, str | Path) or not isinstance(paths, Sequence) or not paths:
            raise ValueError("paths must list at least one stop result")
        entries = [teardown_evidence(item, now=self._now) for item in paths]
        self._receipt["teardown"].extend(entries)
        self._receipt["teardown_verified"] = all(entry["verified_stopped"] for entry in entries)
        self._write_receipt()
        return entries

    # ----- hosts -----

    def _pin(self, host: str) -> str:
        if host not in self._pins:
            self._pins[host] = (self.plan.arms.registry_ids[SERVING_ARM] if host == SERVING_HOST
                                else pinned_digest(self.plan, SAMPLING_ARM))
        return self._pins[host]

    def _note_check(self, check: HostCheck) -> None:
        self._receipt["host_checks"].append(check.to_dict())
        self._write_receipt()

    def require_host(self, check: HostCheck) -> HostCheck:
        """Admit a host for this stage: the check must have passed against the frozen plan's pin.

        Records the check in the stage receipt. Raises :class:`HostDigestError`
        when the host serves other weights, or when the check was made against
        another expectation than the plan's, and :class:`HostCheckError` when the
        host is unreachable, unauthorized or not ready. Nothing is sent either way.
        """
        self._require_active()
        if not isinstance(check, HostCheck):
            raise TypeError(f"check must be a HostCheck; got {type(check).__name__}")
        if check.host not in ARM_HOSTS[self.arm]:
            raise ValueError(f"the {self.arm} arm checks {list(ARM_HOSTS[self.arm])}; got {check.host}")
        pin = self._pin(check.host)
        self._note_check(check)
        if check.expected != pin:
            raise HostDigestError(f"the {check.host} check expected {check.expected}, but the frozen plan pins {pin}; "
                                  "nothing was sent", check=check)
        if check.status == DIGEST_MISMATCH or (check.ok and check.observed != pin):
            raise HostDigestError(f"the {check.host} host at {check.url} serves {check.observed}, but the frozen plan "
                                  f"pins {pin} ({check.detail}); nothing was sent", check=check)
        if not check.ok:
            raise HostCheckError(f"the {check.host} host at {check.url} is not ready ({check.status}: "
                                 f"{check.detail}); nothing was sent", check=check)
        self._hosts[check.host] = check
        return check

    def _host(self, host: str) -> HostCheck:
        check = self._hosts.get(host)
        if check is None:
            raise RuntimeError(f"the {host} host check must pass in this session (require_host) before its requests "
                               "are admitted; nothing was sent")
        return check


class StageSession(_StageRecords):
    """One generation session of one arm and stage: the recording, the ledger session and the host checks.

    Open it with :meth:`open` and use it as a context manager; it enters the
    stage's ledger, so every request is admitted inside the ledger's session.
    See :class:`_StageRecords` for the exit codes.
    """

    _KIND = "generation"
    _COUNTERS = ("requests_sent", "results_written", "results_reused")

    def __init__(self, plan: StudyPlan, arm: str, stage: str, ledger: BudgetLedger, *, records_dir: Path,
                 lease: ServeLease | None, receipt: tuple[Path, dict], clock: Callable[[], float],
                 wall: Callable[[], float], now: Callable[[], datetime], sleep: Callable[[float], None]) -> None:
        super().__init__(plan, arm, stage, records_dir=records_dir, receipt=receipt, clock=clock, now=now,
                         sleep=sleep)
        self.ledger = ledger
        self._lease = lease
        self._wall = wall

    @classmethod
    def open(cls, plan: StudyPlan, *, arm: str, stage: str, limits: StudyLimits, allowance: RequestAllowance,
             approval: ServingApproval | None = None, serve_receipt: str | Path | None = None,
             records_dir: str | Path = RECORDS_DIR, clock: Callable[[], float] = monotonic,
             wall: Callable[[], float] = time.time, now: Callable[[], datetime] = _utc_now,
             sleep: Callable[[float], None] = time.sleep) -> StageSession:
        """Open the stage: validate the plan, limits and allowance, then the serve receipt, then the ledger.

        Everything that can refuse is checked before the ledger is opened, so a
        refused stage leaves its files unchanged.
        """
        require_complete(plan)
        if arm not in ARMS:
            raise ValueError(f"arm must be one of {list(ARMS)}; got {arm!r}")
        if not isinstance(limits, StudyLimits):
            raise TypeError(f"limits must be StudyLimits; got {type(limits).__name__}")
        if not isinstance(allowance, RequestAllowance):
            raise TypeError(f"allowance must be a RequestAllowance; got {type(allowance).__name__}")
        ensure_unchanged(plan, limits.plan_sha256)
        if (limits.arm, limits.stage) != (arm, stage):
            raise ValueError(f"the limits are for the {limits.arm} arm, stage {limits.stage}; this session is the "
                             f"{arm} arm, stage {stage}")
        max_tokens = plan.arms.arms[arm].max_tokens
        if allowance.output_tokens != max_tokens:
            raise ValueError(f"the allowance reserves {allowance.output_tokens} output tokens per request; the {arm} "
                             f"arm's frozen max_tokens is {max_tokens}")
        if allowance.timeout_s > limits.max_seconds:
            raise ValueError(f"the request timeout {allowance.timeout_s} s exceeds the stage's max_seconds "
                             f"{limits.max_seconds}")
        records = Path(records_dir)
        receipt = _load_receipt(plan, arm, stage, records, now())
        lease = None
        if arm == SERVING_ARM:
            if serve_receipt is None:
                raise _refused(f"the serving arm needs the serve receipt of stage {stage} (serve_receipt=...)")
            lease = ServeLease.from_receipt(serve_receipt, plan=plan, approval=approval, stage=stage)
            lease.check_binding(serve_binding_path(stage, records), plan.plan_sha256)
        elif serve_receipt is not None:
            raise ValueError("only the serving arm runs under a serve receipt")
        ledger = BudgetLedger.open(budget_path(arm, stage, records), plan, limits, allowance, approval=approval,
                                   clock=clock, now=now)
        if lease is not None:
            lease.bind(serve_binding_path(stage, records), plan.plan_sha256, now())
        return cls(plan, arm, stage, ledger, records_dir=records, lease=lease, receipt=receipt, clock=clock,
                   wall=wall, now=now, sleep=sleep)

    def _enter(self) -> None:
        self.ledger.__enter__()

    def _exit(self, exc_type: type[BaseException] | None, exc: BaseException | None,
              traceback: TracebackType | None) -> None:
        self.ledger.__exit__(exc_type, exc, traceback)

    def _outcome(self) -> tuple[str, str | None, int]:
        return _stage_outcome(self.ledger.status, self.ledger.stop_reason)

    def complete(self) -> None:
        """Mark the stage complete once every planned request has finished (the ledger refuses otherwise)."""
        self._require_active()
        self.ledger.complete()

    def _probe_after_failure(self, probe: Callable[[], HostCheck]) -> None:
        """After a failed generation request: a dead host, or one serving other weights, ends the session."""
        check = probe()
        if not isinstance(check, HostCheck):
            raise TypeError(f"probe must return a HostCheck; got {type(check).__name__}")
        self._note_check(check)
        if (check.ok and check.observed == self._pin(check.host)) or check.status == BUSY:
            return
        reason = {DIGEST_MISMATCH: DIGEST_MISMATCH, UNAUTHORIZED: UNAUTHORIZED}.get(
            check.status, DIGEST_MISMATCH if check.ok else HOST_UNAVAILABLE)
        raise HostFailure(reason, f"the {check.host} host at {check.url} is {check.status} after a failed request "
                                  f"({check.detail}); the host is not restarted and the stage stays open. Restart it "
                                  "with its script and resume: only the unrecorded draws are issued")

    def _gate_serving(self) -> None:
        if self._lease is None or self._lease.admits(self._wall(), self.ledger.allowance.timeout_s):
            return
        reason = self.ledger.stop_admission("max_seconds")
        raise AdmissionClosedError(
            f"Admission closed for the serving arm, stage {self.stage}: a request now could run past the serve "
            f"deadline {self._lease.deadline_unix_s} (the stage's max_seconds lifetime, one serve per stage per "
            "approval); the stage is incomplete and its partial results are kept",
            reason=reason, status=self.ledger.status)

    # ----- records -----

    @staticmethod
    def _same_ensemble(record: Mapping[str, Any], ensemble: _Ensemble, where: str) -> None:
        for key, what in _ENSEMBLE_CHECKS:
            if record.get(key) != getattr(ensemble, key):
                raise RecordingError(f"{where} was recorded under another {what} ({key} {record.get(key)!r}; this run "
                                     f"has {getattr(ensemble, key)!r}); refused, never merged")

    def _slots(self, path: Path, lines: Sequence[Mapping[str, Any]], ensemble: _Ensemble) -> dict[int, dict]:
        """Per draw index of this rendering: its open request line and its sent outcome, if any."""
        slots: dict[int, dict] = {}
        for number, line in enumerate(lines, start=1):
            if line["kind"] not in (REQUEST, OUTCOME) or line.get("rendering") != ensemble.rendering:
                continue
            where, index = f"{path} line {number}", line.get("draw_index")
            if isinstance(index, bool) or not isinstance(index, int) or index < 0:
                raise RecordingError(f"{where} has no valid draw_index; refused")
            slot = slots.setdefault(index, {"request": None, "outcome": None})
            if line["kind"] == REQUEST:
                if slot["request"] is not None:
                    raise RecordingError(f"{where} requests draw {index} again although it was already sent; a "
                                         "recorded request is never issued again; refused")
                self._same_ensemble(line, ensemble, where)
                slot["request"] = line
            elif slot["request"] is None or slot["outcome"] is not None:
                raise RecordingError(f"{where} records an outcome of draw {index} without its request; refused")
            elif line.get("sent") is True:
                slot["outcome"] = line
            elif line.get("sent") is False:
                slot["request"] = None  # Refused before generation: the draw index stays open.
            else:
                raise RecordingError(f"{where} does not say whether draw {index} was sent; refused")
        return slots

    def _close_interrupted(self, path: Path, ensemble: _Ensemble, slots: dict[int, dict]) -> None:
        """Close each request whose outcome was lost as the failed draw ``request_error:interrupted``."""
        for index, slot in sorted(slots.items()):
            request = slot["request"]
            if request is None or slot["outcome"] is not None:
                continue
            draw = RecordedDraw(draw_index=index, temperature=request.get("temperature"), seed=request.get("seed"),
                                response_sha256=_sha256_text(""), parsed=None, failure_code=INTERRUPTED_FAILURE,
                                estimate=None, evidence_ref=None)
            line = self._outcome_line(ensemble, draw, None, None, None) | {"closed_by": CLOSED_BY_RESUME}
            _append_line(path, line)
            slot["outcome"] = line

    @staticmethod
    def _recorded(path: Path, slots: Mapping[int, dict]) -> tuple[list[RecordedDraw], dict, dict]:
        draws, responses, evidence = [], {}, {}
        for index, slot in sorted(slots.items()):
            outcome = slot["outcome"]
            if outcome is None:
                continue
            try:
                draw = RecordedDraw.from_dict(outcome["draw"])
                response = None if outcome.get("response") is None else ModelResponse.from_dict(outcome["response"])
                full = (None if outcome.get("full_evidence") is None
                        else FullDistributionEvidence.from_dict(outcome["full_evidence"]))
            except (KeyError, TypeError, ValueError) as exc:
                raise RecordingError(f"{path}: the outcome of draw {index} cannot be read ({exc}); refused") from None
            if draw.draw_index != index or (
                    response is not None and _sha256_text(response.content) != outcome.get("response_sha256")):
                raise RecordingError(f"{path}: the outcome of draw {index} does not match its draw or reply; refused")
            draws.append(draw)
            if response is not None:
                responses[index] = response
            if full is not None:
                evidence[index] = full
        return draws, responses, evidence

    def _request_line(self, ensemble: _Ensemble, index: int, seed: int | None) -> dict[str, Any]:
        return self._header(REQUEST, ensemble.item_id, ensemble.rendering) | {
            "draw_index": index, "seed": seed, "temperature": ensemble.temperature,
            "requested_at": _iso(self._now()), "prompt_sha256": ensemble.prompt_sha256,
            "settings_json": ensemble.settings_json, "config": ensemble.config, "source": ensemble.source,
            "calibrator_fingerprint": ensemble.calibrator_fingerprint, "host_digest": ensemble.host_digest,
            "session": self._receipt["sessions"]}

    def _outcome_line(self, ensemble: _Ensemble, draw: RecordedDraw, response: ModelResponse | None,
                      full: FullDistributionEvidence | None, latency_s: float | None) -> dict[str, Any]:
        return self._header(OUTCOME, ensemble.item_id, ensemble.rendering) | {
            "draw_index": draw.draw_index, "sent": True, "recorded_at": _iso(self._now()),
            "response_sha256": None if response is None else _sha256_text(response.content),
            "failure_code": draw.failure_code,
            "usage": None if response is None or response.usage is None else dict(response.usage),
            "latency_s": latency_s, "evidence_ref": draw.evidence_ref, "draw": draw.to_dict(),
            "response": None if response is None else response.to_dict(),
            "full_evidence": None if full is None else full.to_dict(), "closed_by": None}

    def _not_sent_line(self, ensemble: _Ensemble, index: int, reason: str) -> dict[str, Any]:
        return self._header(OUTCOME, ensemble.item_id, ensemble.rendering) | {
            "draw_index": index, "sent": False, "recorded_at": _iso(self._now()), "response_sha256": None,
            "failure_code": NOT_SENT_PREFIX + reason, "usage": None, "latency_s": None, "evidence_ref": None,
            "draw": None, "response": None, "full_evidence": None, "closed_by": None}

    # ----- generation -----

    def _check_client(self, client: Any, check: HostCheck) -> ModelConfig:
        config = getattr(client, "config", None)
        if not isinstance(config, ModelConfig):
            raise TypeError("client must be a configured completion client with a ModelConfig")
        budget = getattr(client, "budget", None)
        if not isinstance(budget, LedgerCallBudget) or budget.ledger is not self.ledger:
            raise ValueError("the client's budget must be a LedgerCallBudget on this stage's ledger, so that every "
                             "request is admitted by it; nothing was sent")
        allowance = self.ledger.allowance
        if config.max_retries != 0:
            raise ValueError(f"the client retries (max_retries={config.max_retries}); a retry would resend a study "
                             "request, so max_retries must be 0")
        if config.parameters.get("max_tokens") != allowance.output_tokens:
            raise ValueError(f"the client sends max_tokens {config.parameters.get('max_tokens')!r}; the allowance "
                             f"reserves {allowance.output_tokens}")
        if config.timeout_s != allowance.timeout_s:
            raise ValueError(f"the client's timeout_s {config.timeout_s} is not the allowance's request timeout "
                             f"{allowance.timeout_s}")
        if config.endpoint != check.url:
            raise ValueError(f"the client's endpoint {config.endpoint} is not the checked host endpoint {check.url}")
        if self._lease is not None and config.endpoint != self._lease.endpoint_url + _CHAT_PATH:
            raise ValueError(f"the client's endpoint {config.endpoint} is not the serve receipt's endpoint "
                             f"{self._lease.endpoint_url}")
        return config

    def _ensemble(self, item_id: str, rendering: str, prompt: str, config: GuardConfig, client_config: ModelConfig,
                  calibrator: SourceCalibrator, check: HostCheck) -> _Ensemble:
        arm = self.plan.arms.arms[self.arm]
        temperature = config.temperature if config.temperature is not None else client_config.parameters.get(
            "temperature")
        if temperature != arm.temperature:
            raise ValueError(f"the ensemble draws at temperature {temperature!r}; the {self.arm} arm's frozen "
                             f"temperature is {arm.temperature}")
        if config.seed_base != arm.seed_base:
            raise ValueError(f"the ensemble's seed_base {config.seed_base!r} is not the {self.arm} arm's frozen "
                             f"seed_base {arm.seed_base!r}")
        parameters = json.loads(json.dumps(client_config.to_dict()["parameters"]))
        if config.temperature is not None:
            parameters["temperature"] = config.temperature
        return _Ensemble(
            item_id=item_id, rendering=rendering, prompt_sha256=_sha256_text(prompt),
            settings_json=identity_settings(parameters),
            config={"definition": config.definition.name, "draws": config.draws, "min_parsed": config.min_parsed,
                    "mode": config.mode.value, "temperature": config.temperature, "seed_base": config.seed_base},
            source={"source": config.source.source.value, "k": config.source.k},
            calibrator_fingerprint=calibrator.fingerprint, host_digest=check.observed,
            temperature=float(temperature) if temperature is not None else None)

    def _stored_result(self, ensemble: _Ensemble, config: GuardConfig) -> GuardedRun | None:
        path = result_path(self.arm, self.stage, ensemble.item_id, ensemble.rendering, self.records_dir)
        if not path.exists():
            return None
        data = _read_json(path, "recorded result")
        if set(data) != _RESULT_KEYS or data["schema"] != RESULT_SCHEMA:
            raise RecordingError(f"{path} is not a {RESULT_SCHEMA} result; refused, never merged")
        if data["plan_sha256"] != self.plan.plan_sha256:
            raise PlanChangedError(f"{path} was recorded under plan_sha256 {data['plan_sha256']}; this plan is "
                                   f"{self.plan.plan_sha256}; refused, never merged")
        if (data["arm"], data["stage"], data["item_id"], data["rendering"]) != (
                self.arm, self.stage, ensemble.item_id, ensemble.rendering):
            raise RecordingError(f"{path} belongs to another arm, stage, item or rendering; refused")
        self._same_ensemble(data, ensemble, str(path))
        try:
            run = GuardedRun.from_dict(data["run"])
            identity = run.decision.identity
            recomputed = decide_from_draws(run.decision.draws, config, identity)
        except (TypeError, ValueError) as exc:
            raise RecordingError(f"{path} does not hold a guarded run that recomputes ({exc}); refused") from None
        if (identity.prompt_sha256, identity.settings_json, identity.calibrator_fingerprint) != (
                ensemble.prompt_sha256, ensemble.settings_json, ensemble.calibrator_fingerprint):
            raise RecordingError(f"{path}: the stored identity differs from this run's; refused, never merged")
        if recomputed.to_dict() != run.decision.to_dict():
            raise RecordingError(f"{path}: the stored decision does not recompute from its recorded draws; refused")
        return run

    def run_ensemble(self, item_id: str, rendering: str, prompt: str, config: GuardConfig, *, client: Any,
                     calibrator: SourceCalibrator, full_distribution: Any = None,
                     probe: Callable[[], HostCheck] | None = None) -> GuardedRun:
        """One recorded ensemble of ``prompt`` for an item and rendering, resumed from its record.

        A matching complete result is returned and nothing is sent. Otherwise the
        recorded draws pre-fill :func:`guarded_run` and only the unrecorded draws
        are issued, each recorded before it is sent and after it returns. After a
        failed request ``probe`` (a host check) decides whether the host is still
        there; a dead host raises :class:`HostFailure`.
        """
        self._require_active()
        item_id, rendering = _item(item_id), _rendering(rendering)
        if not isinstance(prompt, str) or not prompt:
            raise ValueError("prompt must be a non-empty string")
        if not isinstance(config, GuardConfig):
            raise TypeError(f"config must be a GuardConfig; got {type(config).__name__}")
        if not isinstance(calibrator, SourceCalibrator):
            raise TypeError(f"calibrator must be a SourceCalibrator; got {type(calibrator).__name__}")
        check = self._host(GENERATION_HOST[self.arm])
        client_config = self._check_client(client, check)
        ensemble = self._ensemble(item_id, rendering, prompt, config, client_config, calibrator, check)
        stored = self._stored_result(ensemble, config)
        if stored is not None:
            self._count("results_reused")
            return stored
        path, lines = self._read_lines(item_id)
        slots = self._slots(path, lines, ensemble)
        self._close_interrupted(path, ensemble, slots)
        recorded, responses, evidence = self._recorded(path, slots)
        if len(recorded) < config.draws and self.ledger.status != "open":
            self.ledger.admit()  # Raises AdmissionClosedError; nothing is recorded or sent.
        pending: dict[str, Any] = {}

        def on_request(index: int, seed: int | None) -> None:
            self._gate_serving()
            _append_line(path, self._request_line(ensemble, index, seed))
            pending.update(index=index, started=self._clock())
            self._count("requests_sent")

        def on_draw(draw: RecordedDraw, response: ModelResponse | None, full: FullDistributionEvidence | None) -> None:
            latency = max(0.0, self._clock() - pending["started"])
            _append_line(path, self._outcome_line(ensemble, draw, response, full, latency))
            pending.clear()
            if draw.failure_code is not None and draw.failure_code.startswith(REQUEST_ERROR_PREFIX) and probe:
                self._probe_after_failure(probe)

        try:
            run = guarded_run(prompt, config, client=client, calibrator=calibrator,
                              full_distribution=full_distribution, recorded=recorded, on_request=on_request,
                              on_draw=on_draw)
        except BaseException as error:
            reason = _not_sent_reason(error) if pending else None
            if reason is not None:
                _append_line(path, self._not_sent_line(ensemble, pending["index"], reason))
            raise
        identity = run.decision.identity
        if (identity.prompt_sha256, identity.settings_json) != (ensemble.prompt_sha256, ensemble.settings_json):
            raise RecordingError("the guarded run's identity differs from the recorded prompt or settings")
        merged = GuardedRun(run.decision,
                            tuple(responses.get(index, reply) for index, reply in enumerate(run.responses)),
                            tuple(evidence.get(index, item) for index, item in enumerate(run.evidence)))
        _write_json(result_path(self.arm, self.stage, item_id, rendering, self.records_dir), {
            "schema": RESULT_SCHEMA, "plan_sha256": self.plan.plan_sha256, "arm": self.arm, "stage": self.stage,
            "item_id": item_id, "rendering": rendering, "prompt_sha256": ensemble.prompt_sha256,
            "settings_json": ensemble.settings_json, "config": ensemble.config, "source": ensemble.source,
            "calibrator_fingerprint": ensemble.calibrator_fingerprint, "host_digest": ensemble.host_digest,
            "recorded_at": _iso(self._now()), "run": merged.to_dict()})
        self._count("results_written")
        return merged


class EvidenceSession(_StageRecords):
    """The full-distribution evidence pass of a sampling-arm stage, after all of its generation.

    It never opens the stage's ledger: evidence calls count against no request,
    token or time limit (the plan's ``EVIDENCE_RULE``), and the ledger's elapsed
    time does not run while evidence is computed. No generation request is ever
    sent from here. The stage stays open while a batch is missing (exit 1, stop
    reason ``evidence_missing`` or the host failure), and a rerun repeats only
    the missing batches.
    """

    _KIND = "evidence"
    _COUNTERS = ("evidence_batches", "evidence_reused", "evidence_missing")

    def __init__(self, plan: StudyPlan, arm: str, stage: str, *, records_dir: Path, receipt: tuple[Path, dict],
                 clock: Callable[[], float], now: Callable[[], datetime], sleep: Callable[[float], None]) -> None:
        super().__init__(plan, arm, stage, records_dir=records_dir, receipt=receipt, clock=clock, now=now,
                         sleep=sleep)
        self._missing: set[str] = set()

    @classmethod
    def open(cls, plan: StudyPlan, *, stage: str, arm: str = SAMPLING_ARM, records_dir: str | Path = RECORDS_DIR,
             clock: Callable[[], float] = monotonic, now: Callable[[], datetime] = _utc_now,
             sleep: Callable[[float], None] = time.sleep) -> EvidenceSession:
        """Open the evidence pass of a stage; the ledger is read only, never opened."""
        require_complete(plan)
        if arm != SAMPLING_ARM:
            raise ValueError("only the sampling arm has a full-distribution host; the serving arm's "
                             "full-distribution variants are missing by design")
        records = Path(records_dir)
        stage_directory(arm, stage, records)  # Refuses a stage name that is not a lowercase identifier.
        receipt = _load_receipt(plan, arm, stage, records, now())
        return cls(plan, arm, stage, records_dir=records, receipt=receipt, clock=clock, now=now, sleep=sleep)

    def _outcome(self) -> tuple[str, str | None, int]:
        if self._receipt["last_session"]["evidence_missing"]:
            return OPEN, "evidence_missing", EXIT_UNFINISHED
        return super()._outcome()

    # ----- full-distribution evidence -----

    def _stored_evidence(self, item_id: str, rendering: str, batch: str, digest: str,
                         indices: Sequence[int]) -> dict[int, FullDistributionEvidence] | None:
        path = evidence_path(self.arm, self.stage, item_id, rendering, self.records_dir)
        if not path.exists():
            return None
        data = _read_json(path, "recorded evidence batch")
        if set(data) != _EVIDENCE_KEYS or data["schema"] != EVIDENCE_SCHEMA:
            raise RecordingError(f"{path} is not a {EVIDENCE_SCHEMA} batch; refused, never merged")
        if data["plan_sha256"] != self.plan.plan_sha256:
            raise PlanChangedError(f"{path} was recorded under plan_sha256 {data['plan_sha256']}; this plan is "
                                   f"{self.plan.plan_sha256}; refused, never merged")
        if (data["batch_sha256"], data["model_digest"], data["draw_indices"]) != (batch, digest, list(indices)):
            raise RecordingError(f"{path} holds evidence of another batch or digest; refused, never merged")
        try:
            evidence = [FullDistributionEvidence.from_dict(item) for item in data["evidence"]]
        except (TypeError, ValueError) as exc:
            raise RecordingError(f"{path}: the recorded evidence cannot be read ({exc}); refused") from None
        if len(evidence) != len(indices) or any(item.model_digest != digest for item in evidence):
            raise RecordingError(f"{path}: the recorded evidence does not match its batch; refused")
        return dict(zip(indices, evidence, strict=True))

    @staticmethod
    def _evidence_state(path: Path, lines: Sequence[Mapping[str, Any]], batch: str) -> dict[str, Any]:
        """The batch's attempts so far: the last attempt number, an attempt without outcome, the last outcome."""
        state = {"attempts": 0, "pending": None, "last": None, "succeeded": False}
        for line in lines:
            if line["kind"] not in (EVIDENCE_REQUEST, EVIDENCE_OUTCOME) or line.get("batch_sha256") != batch:
                continue
            if line["kind"] == EVIDENCE_REQUEST:
                if state["pending"] is not None:
                    raise RecordingError(f"{path}: evidence batch {batch} was sent again before its outcome; refused")
                state["pending"] = line
                state["attempts"] = max(state["attempts"], line.get("attempt", 0))
                continue
            if state["pending"] is None or line.get("attempt") != state["pending"].get("attempt"):
                raise RecordingError(f"{path}: an evidence outcome of batch {batch} has no request; refused")
            state["pending"] = None
            state["last"] = line.get("failure_code")
            state["succeeded"] = state["succeeded"] or line.get("failure_code") is None
        return state

    def _wait_idle(self, probe: Callable[[], HostCheck] | None, policy: EvidencePolicy) -> None:
        """Before an evidence batch is sent again: the listener must be alive and answer a minimal probe (idle)."""
        if probe is None:
            raise ValueError("an evidence batch is resent only after an idle check; pass probe=...")
        deadline = self._clock() + policy.idle_wait_s
        while True:
            check = probe()
            if not isinstance(check, HostCheck):
                raise TypeError(f"probe must return a HostCheck; got {type(check).__name__}")
            self._note_check(check)
            if check.ok and check.observed == self._pin(check.host):
                return
            if check.status == UNREACHABLE:
                raise HostFailure(HOST_UNAVAILABLE, f"the full-distribution host at {check.url} is unreachable "
                                  f"({check.detail}); the host is not restarted and the stage stays open. Restart it "
                                  "with its script and rerun: only the missing evidence batches are sent")
            if check.status != BUSY:
                reason = UNAUTHORIZED if check.status == UNAUTHORIZED else DIGEST_MISMATCH if check.status in (
                    DIGEST_MISMATCH, OK) else HOST_UNAVAILABLE
                raise HostFailure(reason, f"the full-distribution host at {check.url} is {check.status} "
                                          f"({check.detail}); the stage stays open")
            if self._clock() + policy.idle_poll_s > deadline:
                raise HostFailure(EVIDENCE_UNRESOLVED, f"the full-distribution host at {check.url} stayed busy for "
                                  f"{policy.idle_wait_s} s after a batch timed out; it may still be computing, so the "
                                  "batch is not resent now and the stage stays open")
            self._sleep(policy.idle_poll_s)

    def record_evidence(self, item_id: str, rendering: str, prompt: str, run: GuardedRun, *, client: Any,
                        probe: Callable[[], HostCheck] | None, policy: EvidencePolicy = EvidencePolicy()
                        ) -> tuple[FullDistributionEvidence | None, ...]:
        """Full-distribution evidence for the parsed draws of one recorded ensemble, in one recorded batch.

        Evidence calls never touch the generation ledger. A settled batch is
        reused without a request; its outcome line records the evidence
        reference of each draw (``evidence_refs``, the SHA-256 of the canonical
        evidence payload, the convention of ``RecordedDraw.evidence_ref``).
        ``host_busy`` is retried at most ``policy.max_busy_retries`` times with
        ``policy.busy_wait_s`` between; then :class:`HostFailure` ends the
        session and the stage stays open. A transport failure or client timeout
        is resent only after ``probe`` finds the host alive and idle, at most
        ``policy.max_sends`` sends per session; a dead host ends the session.
        A batch the host rejects for its content is recorded with its reason
        and returned as missing (None per draw); it is not sent again in this
        session, and a rerun repeats it. Returns one evidence object per draw
        index, None for a draw without a parsed reply.
        """
        self._require_active()
        item_id, rendering = _item(item_id), _rendering(rendering)
        if not isinstance(run, GuardedRun):
            raise TypeError(f"run must be a GuardedRun; got {type(run).__name__}")
        if not isinstance(policy, EvidencePolicy):
            raise TypeError(f"policy must be an EvidencePolicy; got {type(policy).__name__}")
        if not isinstance(prompt, str) or _sha256_text(prompt) != run.decision.identity.prompt_sha256:
            raise RecordingError("the prompt is not the one the ensemble was drawn for")
        check = self._host(FULL_DISTRIBUTION_HOST)
        digest = getattr(client, "model_digest", None)
        if digest != check.observed:
            raise HostDigestError(f"the full-distribution client requests {digest}; the checked host serves the pin "
                                  f"{check.observed}; nothing was sent", check=check)
        empty = (None,) * len(run.decision.draws)
        pairs = [(draw.draw_index, reply.content) for draw, reply in zip(run.decision.draws, run.responses, strict=True)
                 if draw.parsed is not None and reply is not None and reply.content]
        if not pairs:
            return empty
        indices = [index for index, _ in pairs]
        texts = [text for _, text in pairs]
        hashes = [_sha256_text(text) for text in texts]
        prompt_sha256 = run.decision.identity.prompt_sha256
        batch = sha256_of({"rendering": rendering, "prompt_sha256": prompt_sha256,
                           "responses": [[index, value] for index, value in zip(indices, hashes, strict=True)],
                           "top_k_keep": policy.top_k_keep, "model_digest": digest})
        path, lines = self._read_lines(item_id)
        state = self._evidence_state(path, lines, batch)
        header = {"batch_sha256": batch}

        def outcome(attempt: int, failure: str | None, latency: float | None,
                    found: Mapping[int, FullDistributionEvidence] | None) -> None:
            refs = None if found is None else {str(index): sha256_of(found[index].to_dict()) for index in indices}
            _append_line(path, self._header(EVIDENCE_OUTCOME, item_id, rendering) | header | {
                "attempt": attempt, "recorded_at": _iso(self._now()), "latency_s": latency, "failure_code": failure,
                "evidence_refs": refs, "evidence_sha256": None if refs is None else [refs[str(i)] for i in indices],
                "evidence_file": None if found is None else evidence_path(
                    self.arm, self.stage, item_id, rendering, self.records_dir).name})

        def by_index(found: Mapping[int, FullDistributionEvidence]) -> tuple[FullDistributionEvidence | None, ...]:
            return tuple(found.get(draw.draw_index) for draw in run.decision.draws)

        stored = self._stored_evidence(item_id, rendering, batch, digest, indices)
        if stored is not None:
            if state["pending"] is not None:  # Written before its outcome line: the outcome is recovered, not resent.
                outcome(state["pending"]["attempt"], None, None, stored)
            self._count("evidence_reused")
            return by_index(stored)
        if state["succeeded"]:
            raise RecordingError(f"{path}: evidence batch {batch} was recorded but its evidence file is missing; "
                                 "refused, never resent")
        if batch in self._missing:
            self._count("evidence_missing")
            return empty  # Rejected for its content in this session: a rerun repeats it.
        need_idle = state["last"] in ("transport_error", EVIDENCE_INTERRUPTED)
        if state["pending"] is not None:
            outcome(state["pending"]["attempt"], EVIDENCE_INTERRUPTED, None, None)
            need_idle = True
        attempt, sends, busy = state["attempts"], 0, 0
        while True:
            if need_idle:
                self._wait_idle(probe, policy)
                need_idle = False
            if sends >= policy.max_sends:
                raise HostFailure(EVIDENCE_UNRESOLVED, f"evidence batch {batch} of {item_id}/{rendering} was sent "
                                  f"{sends} times in this session without an outcome; it is not sent again now and "
                                  "the stage stays open")
            attempt += 1
            _append_line(path, self._header(EVIDENCE_REQUEST, item_id, rendering) | header | {
                "attempt": attempt, "draw_indices": indices, "response_sha256": hashes,
                "prompt_sha256": prompt_sha256, "model_digest": digest, "host_digest": check.observed,
                "top_k_keep": policy.top_k_keep, "timeout_s": policy.timeout_s, "requested_at": _iso(self._now())})
            started = self._clock()
            try:
                batch_evidence = client.evidence_batch(prompt, list(texts), top_k_keep=policy.top_k_keep,
                                                       timeout_s=policy.timeout_s)
            except RequestError as error:
                code = error.code or "request_error"
                outcome(attempt, code, max(0.0, self._clock() - started), None)
                if code == "host_busy":
                    busy += 1
                    if busy > policy.max_busy_retries:
                        raise HostFailure(HOST_BUSY, f"the full-distribution host refused batch {batch} as busy "
                                          f"{busy} times (memory below its floor); the stage stays open") from None
                    self._sleep(policy.busy_wait_s)
                    continue
                sends += 1
                if code == "transport_error":
                    need_idle = True  # It may still be computing: never resend blindly.
                    continue
                if code in ("digest_mismatch", "auth_failed"):
                    raise HostFailure(DIGEST_MISMATCH if code == "digest_mismatch" else UNAUTHORIZED,
                                      f"the full-distribution host refused batch {batch} ({code}); the stage stays "
                                      "open") from None
                return self._missing_batch(batch, empty)
            sends += 1
            if len(batch_evidence) != len(texts) or any(item.model_digest != digest for item in batch_evidence):
                outcome(attempt, "invalid_evidence", max(0.0, self._clock() - started), None)
                return self._missing_batch(batch, empty)
            found = dict(zip(indices, batch_evidence, strict=True))
            _write_json(evidence_path(self.arm, self.stage, item_id, rendering, self.records_dir), {
                "schema": EVIDENCE_SCHEMA, "plan_sha256": self.plan.plan_sha256, "arm": self.arm, "stage": self.stage,
                "item_id": item_id, "rendering": rendering, "prompt_sha256": prompt_sha256, "batch_sha256": batch,
                "model_digest": digest, "top_k_keep": policy.top_k_keep, "draw_indices": indices,
                "evidence": [item.to_dict() for item in batch_evidence], "recorded_at": _iso(self._now())})
            outcome(attempt, None, max(0.0, self._clock() - started), found)
            self._count("evidence_batches")
            return by_index(found)

    def _missing_batch(self, batch: str, empty: tuple[None, ...]) -> tuple[None, ...]:
        self._missing.add(batch)
        self._count("evidence_missing")
        return empty


__all__ = [
    "BUSY",
    "CHECK_STATUSES",
    "DIGEST_MISMATCH",
    "EVIDENCE_TIMEOUT_S",
    "EVIDENCE_UNRESOLVED",
    "HOST_BUSY",
    "HOST_UNAVAILABLE",
    "INTERRUPTED_FAILURE",
    "INVALID",
    "NOT_SENT_PREFIX",
    "OK",
    "UNAUTHORIZED",
    "UNREACHABLE",
    "DrawLog",
    "EvidencePolicy",
    "EvidenceSession",
    "HostCheck",
    "HostCheckError",
    "HostDigestError",
    "HostFailure",
    "HttpReply",
    "HttpTransportError",
    "RecordingError",
    "ServeLease",
    "StageSession",
    "check_full_distribution_host",
    "check_llama_server",
    "check_serving_host",
    "draw_log_path",
    "evidence_path",
    "http_get",
    "pinned_digest",
    "read_records",
    "receipt_path",
    "record_teardown",
    "request_allowance",
    "requests_path",
    "result_path",
    "serve_binding_path",
    "stage_directory",
    "teardown_evidence",
]
