"""Bounded incremental UTF-8/SSE assembly for an OpenAI-chat response.

The caller owns transport, cancellation and secret sanitation. Feed bounded byte
chunks, then call finish at EOF; on transport interruption retain partial(). No
requests, retries, clocks, background workers or provider capability inference
live here. SSE retry/id fields never authorize another request.
"""

import codecs
import json
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace

from recall_guard.core.completion import ModelResponse, ParameterValue, TokenAlternative, TokenEvidence
from recall_guard.core.readiness_evidence import PartialResponse, PartialResponseReason
from recall_guard.core.readiness_types import _integer
from recall_guard.core.request_control import RequestError

_MIN_RETAINED_BYTES = 1024
_USAGE_COUNTERS = frozenset({"prompt_tokens", "completion_tokens", "total_tokens"})


@dataclass(frozen=True)
class StreamProgress:
    """Cumulative wire bytes, Unicode characters and nonempty semantic deltas.

    One snapshot replaces the previous snapshot; no event history is retained.
    Keepalive events count comment-only SSE blocks. Only answer/reasoning deltas
    trigger the callback; byte traffic, roles and comments are not progress.
    """

    transport_bytes: int = 0
    keepalive_events: int = 0
    answer_characters: int = 0
    reasoning_characters: int = 0
    answer_deltas: int = 0
    reasoning_deltas: int = 0


class StreamAssembler:
    """One response, ordered by explicit choice index zero and token position.

    Wire event/response limits count bytes, including SSE fields and delimiters.
    The evidence limit counts serialized UTF-8 JSON, including escaped text and
    metadata. It defaults to max(response limit, 1024); this independent envelope
    floor permits valid failure evidence even for a one-byte transport limit.
    Explicit evidence limits must also be at least 1024 bytes.

    Token rows are usable only when every nonempty answer delta has exactly
    aligned rows in that same event. Missing or mismatched rows invalidate the
    whole token sequence; later rows cannot fill or silently shift a gap. Raw
    censoring markers are retained for ModelResponse's availability checks.
    """

    def __init__(
        self, *, max_event_bytes: int = 1024 * 1024, max_response_bytes: int = 8 * 1024 * 1024,
        max_retained_bytes: int | None = None,
        requested_settings: Mapping[str, ParameterValue] | None = None,
        request_id: str | None = None, on_progress: Callable[[StreamProgress], None] | None = None,
    ) -> None:
        _integer(max_event_bytes, "max_event_bytes", minimum=1)
        _integer(max_response_bytes, "max_response_bytes", minimum=1)
        if max_event_bytes > max_response_bytes:
            raise ValueError("max_event_bytes must not exceed max_response_bytes")
        if max_retained_bytes is None:
            max_retained_bytes = max(max_response_bytes, _MIN_RETAINED_BYTES)
        _integer(max_retained_bytes, "max_retained_bytes", minimum=_MIN_RETAINED_BYTES)
        if on_progress is not None and not callable(on_progress):
            raise TypeError("on_progress must be callable")
        initial = ModelResponse("", requested_settings=requested_settings or {}, request_id=request_id)
        invalid_metadata = False
        try:
            json.dumps(initial.to_dict(), ensure_ascii=False, allow_nan=False).encode("utf-8")
        except UnicodeError:
            invalid_metadata = True
        if invalid_metadata:
            raise ValueError("Stream metadata must be UTF-8 encodable") from None
        self._settings = initial.requested_settings
        self._request_id = initial.request_id
        self._max_event_bytes = max_event_bytes
        self._max_response_bytes = max_response_bytes
        self._max_retained_bytes = max_retained_bytes
        self._on_progress = on_progress
        self._progress = StreamProgress()
        self._decoder = codecs.getincrementaldecoder("utf-8-sig")("strict")
        self._line: list[str] = []
        self._data: list[str] = []
        self._event_bytes = 0
        self._processed_bytes = 0
        self._comment = False
        self._pending_cr = False
        self._content: list[str] = []
        self._tokens: list[TokenEvidence] = []
        self._tokens_aligned = True
        self._usage = None
        self._finish_reason = None
        self._done = False
        self._closed = False
        self._failure: RequestError | None = None

    @property
    def progress(self) -> StreamProgress:
        return self._progress

    def feed(self, chunk: bytes) -> None:
        """Consume arbitrary chunk boundaries, retaining at most configured limits."""
        self._check_open()
        if not isinstance(chunk, bytes):
            raise TypeError("stream chunks must be bytes")
        # Count all supplied bytes even if parsing rejects an earlier prefix.
        # Processed bytes below enforce limits before any event is committed.
        self._progress = replace(self._progress, transport_bytes=self._progress.transport_bytes + len(chunk))
        # Scan delimiters without duplicating an arbitrarily large supplied chunk.
        start = 0
        for ending in re.finditer(b"[\r\n]", chunk):
            self._segment(chunk, start, ending.start())
            delimiter = chunk[ending.start():ending.end()]
            if self._pending_cr and delimiter != b"\n":
                self._dispatch_pending_cr()
            self._count_bytes(1)
            self._decode(delimiter)  # rejects UTF-8 split by an illegal newline
            if self._pending_cr:
                self._dispatch_pending_cr()
            elif delimiter == b"\r":
                # Wait for one-byte lookahead (or EOF) so a CRLF's two bytes
                # count toward this event before its evidence is committed.
                self._pending_cr = True
            else:
                self._dispatch_line()
            start = ending.end()
        self._segment(chunk, start, len(chunk))

    def _segment(self, chunk: bytes, start: int, end: int) -> None:
        if start == end:
            return
        if self._pending_cr:
            self._dispatch_pending_cr()
        # Bounds are checked before slicing/decoding, including unterminated lines.
        self._count_bytes(end - start)
        self._line.append(self._decode(chunk[start:end]))

    def _dispatch_pending_cr(self) -> None:
        self._pending_cr = False
        self._dispatch_line()

    def _count_bytes(self, count: int) -> None:
        if self._processed_bytes + count > self._max_response_bytes:
            self._fail(PartialResponseReason.RESPONSE_LIMIT)
        if self._event_bytes + count > self._max_event_bytes:
            self._fail(PartialResponseReason.EVENT_LIMIT)
        self._processed_bytes += count
        self._event_bytes += count

    def _decode(self, raw: bytes, *, final: bool = False) -> str:
        failed = False
        try:
            decoded = self._decoder.decode(raw, final=final)
        except UnicodeError:
            failed = True
        if failed:
            self._fail(PartialResponseReason.MALFORMED)
        return decoded

    def _dispatch_line(self) -> None:
        line = "".join(self._line)
        self._line.clear()
        if not line:
            data, comment = self._data, self._comment
            self._data = []
            self._comment = False
            self._event_bytes = 0
            if data:
                self._event("\n".join(data))
            elif comment:
                self._progress = replace(self._progress, keepalive_events=self._progress.keepalive_events + 1)
        elif line.startswith(":"):
            self._comment = True
        else:
            field, _, value = line.partition(":")
            if field == "data":
                self._data.append(value.removeprefix(" "))

    def _event(self, data: str) -> None:
        if self._done:
            self._fail(PartialResponseReason.MALFORMED)
        if data == "[DONE]":
            self._done = True
            return
        failed = False
        try:
            payload = json.loads(data)
            if not isinstance(payload, dict) or "error" in payload:
                raise ValueError("Expected a completion chunk")
            choices = payload["choices"]
            if not isinstance(choices, list):
                raise ValueError("Expected choices")
            selected = []
            for choice in choices:
                if not isinstance(choice, dict):
                    raise ValueError("Expected a choice")
                _integer(choice.get("index"), "choice index")
                if choice["index"] == 0:
                    selected.append(choice)
            if len(selected) > 1:
                raise ValueError("Repeated choice zero")
            usage = self._parse_usage(payload.get("usage"))
            parsed = self._parse_choice(selected[0]) if selected else None
        except (ValueError, TypeError, KeyError, OverflowError, RecursionError):
            failed = True
        if failed:
            self._fail(PartialResponseReason.MALFORMED)
        # Commit one validated event atomically; malformed later fields cannot
        # retroactively corrupt previously retained text, usage or token order.
        if usage is not None:
            self._usage = usage
        if parsed is not None:
            self._accept_choice(*parsed)

    @staticmethod
    def _parse_usage(usage):
        if usage is None:
            return None
        if not isinstance(usage, dict):
            raise ValueError("Expected usage mapping")
        counters = {name: count for name, count in usage.items() if name in _USAGE_COUNTERS}
        for count in counters.values():
            _integer(count, "usage count")
        return counters or None

    def _parse_choice(self, choice):
        delta = choice.get("delta")
        if not isinstance(delta, dict):
            raise ValueError("Expected delta")
        content = delta.get("content")
        reasoning = delta.get("reasoning_content")
        for value in (content, reasoning):
            if value is not None and not isinstance(value, str):
                raise ValueError("Expected text delta")
            if value is not None:
                value.encode("utf-8")  # reject escaped lone surrogates
        finish = choice.get("finish_reason")
        if finish is not None and (not isinstance(finish, str) or not finish.strip()):
            raise ValueError("Expected terminal finish reason")
        if finish is not None:
            finish.encode("utf-8")
        if self._finish_reason is not None and (
            content or reasoning or finish is not None or choice.get("logprobs") is not None
        ):
            raise ValueError("Choice continued after terminal finish reason")
        rows, aligned = self._parse_tokens(choice.get("logprobs"), content or "")
        return content or "", reasoning or "", finish, rows, aligned

    @staticmethod
    def _parse_tokens(logprobs, content):
        if logprobs is None:
            return [], not content
        if not isinstance(logprobs, dict):
            raise ValueError("Expected logprobs mapping")
        raw = logprobs.get("content")
        if raw is None:
            return [], not content
        if not isinstance(raw, list):
            raise ValueError("Expected token rows")
        rows = []
        for entry in raw:
            if not isinstance(entry, dict):
                raise ValueError("Expected token row")
            alternatives = entry.get("top_logprobs")
            if alternatives is None:
                alternatives = []
            if not isinstance(alternatives, list):
                raise ValueError("Expected token alternatives")
            if any(not isinstance(alt, dict) for alt in alternatives):
                raise ValueError("Expected token alternative row")
            row = TokenEvidence(entry["token"], entry["logprob"], tuple(
                TokenAlternative(alt["token"], alt["logprob"]) for alt in alternatives))
            for candidate in (row, *row.top_logprobs):
                candidate.token.encode("utf-8")
            rows.append(row)
        return rows, "".join(row.token for row in rows) == content

    def _accept_choice(self, content, reasoning, finish, rows, aligned) -> None:
        if content:
            self._content.append(content)
        self._tokens_aligned = self._tokens_aligned and aligned
        if self._tokens_aligned:
            self._tokens.extend(rows)
        else:
            self._tokens.clear()
        if finish is not None:
            self._finish_reason = finish
        if content or reasoning:
            self._progress = replace(
                self._progress, answer_characters=self._progress.answer_characters + len(content),
                reasoning_characters=self._progress.reasoning_characters + len(reasoning),
                answer_deltas=self._progress.answer_deltas + bool(content),
                reasoning_deltas=self._progress.reasoning_deltas + bool(reasoning),
            )
            failed = False
            if self._on_progress is not None:
                try:
                    self._on_progress(self._progress)
                except Exception:
                    failed = True
            if failed:
                self._fail(PartialResponseReason.INTERRUPTED, code="callback_failed")

    def finish(self) -> ModelResponse:
        """Return complete evidence only after framed DONE and choice-zero finish."""
        self._check_open()
        self._decode(b"", final=True)
        if self._pending_cr:
            self._dispatch_pending_cr()
        if not self._done or self._finish_reason is None or self._line or self._data:
            self._fail(PartialResponseReason.PREMATURE_EOF)
        self._closed = True
        return self._response()

    def _response(self) -> ModelResponse:
        return ModelResponse("".join(self._content), tuple(self._tokens) if self._tokens_aligned else None,
                             self._usage, self._finish_reason, self._settings, request_id=self._request_id)

    def partial(self, reason: PartialResponseReason = PartialResponseReason.INTERRUPTED) -> PartialResponse:
        """Snapshot incomplete evidence; shrink tokens/metadata/text to its JSON cap.

        Content retains a Unicode prefix. If truncation is necessary token rows
        are dropped as a unit, so no token position falsely describes the prefix.
        Oversized settings/IDs are omitted, never partially rewritten.
        """
        reason = PartialResponseReason(reason)
        data = self._response().to_dict() | {
            "reason": reason.value, "reasoning_characters": self._progress.reasoning_characters,
            "max_retained_bytes": self._max_retained_bytes, "complete": False,
        }

        def size():
            return len(json.dumps(data, ensure_ascii=False, allow_nan=False).encode("utf-8"))

        if size() > self._max_retained_bytes:
            data["token_evidence"] = None
        for key, empty in (("requested_settings", {}), ("confirmed_settings", {}), ("request_id", None),
                           ("finish_reason", None), ("usage", None)):
            if size() > self._max_retained_bytes:
                data[key] = empty
        if size() > self._max_retained_bytes:
            content = data["content"]
            low, high = 0, len(content)
            while low < high:
                middle = (low + high + 1) // 2
                data["content"] = content[:middle]
                if size() <= self._max_retained_bytes:
                    low = middle
                else:
                    high = middle - 1
            data["content"] = content[:low]
        return PartialResponse.from_dict(data)

    def _check_open(self) -> None:
        if self._failure is not None:
            raise self._failure from None
        if self._closed:
            raise ValueError("Stream assembler is already finished")

    def _fail(self, reason: PartialResponseReason, *, code: str | None = None) -> None:
        self._failure = RequestError("Provider stream did not complete safely", code=code or reason.value,
                                     response_evidence=self.partial(reason))
        raise self._failure from None
