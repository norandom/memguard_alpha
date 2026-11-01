"""Offline wire-level stream contracts, independent of transport admission."""

import json
from dataclasses import FrozenInstanceError

import pytest

from recall_guard.core.provider_stream import StreamAssembler
from recall_guard.core.readiness_evidence import PartialResponse, PartialResponseReason
from recall_guard.core.request_control import RequestError


def event(payload, newline="\n"):
    return ("data: " + json.dumps(payload, ensure_ascii=False) + newline * 2).encode()


def delta(content=None, *, index=0, reasoning=None, tokens=None, finish=None):
    fields = {}
    if content is not None:
        fields["content"] = content
    if reasoning is not None:
        fields["reasoning_content"] = reasoning
    choice = {"index": index, "delta": fields, "finish_reason": finish}
    if tokens is not None:
        choice["logprobs"] = {"content": tokens}
    return {"choices": [choice]}


def token(text, *, logprob=-0.2, alternatives=True):
    return {"token": text, "logprob": logprob,
            "top_logprobs": [{"token": text, "logprob": logprob}] if alternatives else []}


def complete(*events):
    return b"".join(events) + event(delta(finish="stop")) + b"data: [DONE]\n\n"


def assemble(wire, **kwargs):
    parser = StreamAssembler(**kwargs)
    parser.feed(wire)
    return parser.finish()


@pytest.mark.parametrize("newline", ["\n", "\r", "\r\n"])
def test_every_split_preserves_utf8_bom_sse_and_token_order(newline):
    wire = b"\xef\xbb\xbf" + event(delta("A", tokens=[token("A")]), newline)
    wire += event(delta("\U0001f642", tokens=[token("\U0001f642")]), newline)
    wire += event(delta(finish="length"), newline) + ("data: [DONE]" + newline * 2).encode()
    for split in range(len(wire) + 1):
        parser = StreamAssembler()
        parser.feed(wire[:split])
        parser.feed(wire[split:])
        result = parser.finish()
        assert result.content == "A\U0001f642"
        assert [t.token for t in result.token_evidence] == ["A", "\U0001f642"]
        assert result.finish_reason == "length"
        assert result.mia_unavailable_reason is None
        assert result.usage is None


def test_multiline_data_comments_and_unknown_fields_do_not_become_progress():
    observations = []
    parser = StreamAssembler(on_progress=observations.append)
    parser.feed(b": ping\n\nretry: 0\nid: untrusted\nevent: message\n\n")
    assert observations == []
    parser.feed(b'data: {"choices":\ndata: [{"index": 0, "delta": {"content": "yes"}}]}\n\n')
    parser.feed(complete())
    assert parser.finish().content == "yes"
    assert len(observations) == 1
    assert observations[0].answer_characters == 3
    assert observations[0].keepalive_events == 1
    with pytest.raises(FrozenInstanceError):
        observations[0].answer_characters = 99


def test_choice_zero_is_explicit_and_reasoning_never_enters_answer_or_tokens():
    observations = []
    parser = StreamAssembler(on_progress=observations.append, requested_settings={"temperature": 0.2})
    parser.feed(event(delta("WRONG", index=1, tokens=[token("WRONG")], finish="stop")))
    parser.feed(event(delta(reasoning="think")))
    parser.feed(complete(event(delta("ok", tokens=[token("ok")]))))
    result = parser.finish()
    assert result.content == "ok"
    assert [t.token for t in result.token_evidence] == ["ok"]
    assert result.requested_settings == {"temperature": 0.2}
    assert result.confirmed_settings == {}
    assert parser.progress.reasoning_characters == 5
    assert parser.progress.answer_characters == 2
    assert parser.progress.reasoning_deltas == parser.progress.answer_deltas == 1
    assert len(observations) == 2


@pytest.mark.parametrize("suffix", [b"", b"data: [DONE]", b"data: [DONE]\n"])
def test_eof_does_not_dispatch_incomplete_event(suffix):
    parser = StreamAssembler()
    parser.feed(event(delta("partial", finish="stop")) + suffix)
    with pytest.raises(RequestError) as caught:
        parser.finish()
    evidence = caught.value.response_evidence
    assert evidence.content == "partial"
    assert evidence.reason == PartialResponseReason.PREMATURE_EOF
    assert not evidence.complete and not evidence.usable_answer
    assert evidence.mia_unavailable_reason


@pytest.mark.parametrize("wire", [event(delta("text")) + b"data: [DONE]\n\n",
                                  event(delta("wrong", index=1, finish="stop")) + b"data: [DONE]\n\n"])
def test_done_without_choice_zero_finish_is_incomplete(wire):
    with pytest.raises(RequestError) as caught:
        assemble(wire)
    assert isinstance(caught.value.response_evidence, PartialResponse)


@pytest.mark.parametrize("bad", [b"data: not-json\n\n", b"data: \xff\n\n",
                                  event({"error": {"message": "secret-provider-detail"}}),
                                  event({"choices": [{"delta": {"content": "missing index"}}]}),
                                  event(delta(123)), event({"choices": "wrong"})])
def test_malformed_preserves_prior_answer_without_raw_diagnostics(bad):
    parser = StreamAssembler()
    parser.feed(event(delta("safe", tokens=[token("safe")])) + event(delta(reasoning="reason")))
    with pytest.raises(RequestError) as caught:
        parser.feed(bad)
    error = caught.value
    assert error.response_evidence.content == "safe"
    assert error.response_evidence.reasoning_characters == 6
    assert error.response_evidence.reason == PartialResponseReason.MALFORMED
    assert "secret-provider-detail" not in str(error)
    assert error.__cause__ is None


@pytest.mark.parametrize("tail", [b"data: " + b"x" * 1000, b":" + b"x" * 1000,
                                   b"data: x\n" * 200])
def test_event_limit_bounds_unterminated_lines_comments_and_multiline_events(tail):
    parser = StreamAssembler(max_event_bytes=256, max_response_bytes=4096)
    parser.feed(event(delta("ok")))
    with pytest.raises(RequestError) as caught:
        parser.feed(tail)
    evidence = caught.value.response_evidence
    assert evidence.reason == PartialResponseReason.EVENT_LIMIT
    assert evidence.content == "ok"
    assert len(json.dumps(evidence.to_dict(), ensure_ascii=False).encode()) <= evidence.max_retained_bytes


def test_response_limit_counts_keepalives_and_tiny_limits_fail_with_valid_evidence():
    parser = StreamAssembler(max_event_bytes=32, max_response_bytes=32)
    with pytest.raises(RequestError) as caught:
        parser.feed(b": ping\n\n" * 100)
    assert caught.value.response_evidence.reason == PartialResponseReason.RESPONSE_LIMIT
    with pytest.raises(RequestError) as caught:
        assemble(b"data: x", max_event_bytes=1, max_response_bytes=1)
    evidence = caught.value.response_evidence
    assert isinstance(evidence, PartialResponse)
    assert len(json.dumps(evidence.to_dict()).encode()) <= evidence.max_retained_bytes


@pytest.mark.parametrize("split_crlf", [False, True])
def test_event_limit_counts_all_crlf_bytes_before_accepting_event(split_crlf):
    wire = event(delta("oversized", tokens=[token("oversized")]), "\r\n")
    parser = StreamAssembler(max_event_bytes=len(wire) - 1, max_response_bytes=4096)
    parser.feed(event(delta("safe", tokens=[token("safe")]), "\r\n"))
    chunks = [wire[:-1], wire[-1:]] if split_crlf else [wire]
    with pytest.raises(RequestError) as caught:
        for chunk in chunks:
            parser.feed(chunk)
    evidence = caught.value.response_evidence
    assert evidence.reason == PartialResponseReason.EVENT_LIMIT
    assert evidence.content == "safe"
    assert [row.token for row in evidence.token_evidence] == ["safe"]
    assert parser.progress.transport_bytes == len(event(delta("safe", tokens=[token("safe")]), "\r\n")) + len(wire)


@pytest.mark.parametrize("newline", ["\n", "\r", "\r\n"])
def test_event_at_exact_wire_limit_accepts_each_newline_byte_by_byte(newline):
    wire = event(delta("ok", tokens=[token("ok")]), newline)
    parser = StreamAssembler(max_event_bytes=len(wire), max_response_bytes=4096)
    wire += event(delta(finish="stop"), newline) + ("data: [DONE]" + newline * 2).encode()
    for byte in wire:
        parser.feed(bytes([byte]))
    assert parser.finish().content == "ok"
    assert parser.progress.transport_bytes == len(wire)


@pytest.mark.parametrize("first,second", [([token("a")], None), (None, [token("b")]),
                                          ([token("wrong")], [token("b")]),
                                          ([token("a", logprob=-9999)], [token("b")]),
                                          ([token("a") | {"top_logprobs": None}], [token("b")]),
                                          ([token("a", alternatives=False)], [token("b")])])
def test_missing_misaligned_censored_or_incomplete_tokens_never_make_mcs_available(first, second):
    result = assemble(complete(event(delta("a", tokens=first)), event(delta("b", tokens=second))))
    assert result.content == "ab"
    assert result.mia_unavailable_reason is not None


def test_usage_only_event_retains_valid_reported_counts():
    result = assemble(complete(event(delta("ok")), event({"choices": [], "usage": {
        "prompt_tokens": 4, "completion_tokens": 1, "total_tokens": 5,
        "completion_tokens_details": {"reasoning_tokens": 0}}})))
    assert result.usage == {"prompt_tokens": 4, "completion_tokens": 1, "total_tokens": 5}


@pytest.mark.parametrize("usage", [{"total_tokens": True}, {"prompt_tokens": -1},
                                    {"completion_tokens": 1.2}, {"total_tokens": "8"}])
def test_invalid_usage_is_never_repaired_or_assumed(usage):
    with pytest.raises(RequestError) as caught:
        assemble(event(delta("ok")) + event({"choices": [], "usage": usage}))
    assert caught.value.response_evidence.usage is None


def test_interrupted_partial_is_bounded_including_escaped_text_and_metadata():
    parser = StreamAssembler(max_event_bytes=20000, max_response_bytes=20000, max_retained_bytes=1024,
                             requested_settings={"vendor_label": "x" * 4000})
    parser.feed(event(delta("\x01\U0001f642" * 600, tokens=[token("\x01\U0001f642" * 600)])))
    evidence = parser.partial(PartialResponseReason.INTERRUPTED)
    assert evidence.content
    assert evidence.reason == PartialResponseReason.INTERRUPTED
    assert not evidence.complete
    assert len(json.dumps(evidence.to_dict(), ensure_ascii=False).encode()) <= 1024
    assert evidence.token_evidence is None


def test_callback_failure_is_static_and_preserves_content():
    def broken(progress):
        raise ValueError("sensitive callback details")
    parser = StreamAssembler(on_progress=broken)
    with pytest.raises(RequestError) as caught:
        parser.feed(event(delta("answer")))
    assert caught.value.code == "callback_failed"
    assert caught.value.response_evidence.content == "answer"
    assert "sensitive" not in str(caught.value)


@pytest.mark.parametrize("kwargs", [{"max_event_bytes": 0}, {"max_response_bytes": True},
                                     {"max_retained_bytes": 1}, {"max_event_bytes": 9, "max_response_bytes": 8}])
def test_invalid_local_bounds_rejected(kwargs):
    with pytest.raises((ValueError, TypeError)):
        StreamAssembler(**kwargs)


MALFORMED_TOKEN_ROWS = [
    [None], [False], ["token"], [[]],
] + [[token("bad") | {"top_logprobs": value}] for value in [{}, False, "", 0, [None], [False], [[]]]]


@pytest.mark.parametrize("rows", MALFORMED_TOKEN_ROWS)
def test_malformed_token_shapes_retain_prior_evidence_atomically(rows):
    parser = StreamAssembler(max_retained_bytes=1024)
    prior = delta("safe", reasoning="think", tokens=[token("safe")])
    prior["usage"] = {"completion_tokens": 1}
    parser.feed(event(prior))
    malformed = delta("bad", reasoning="discard", tokens=rows)
    malformed["usage"] = {"completion_tokens": 99}
    with pytest.raises(RequestError) as caught:
        parser.feed(event(malformed))
    evidence = caught.value.response_evidence
    assert isinstance(evidence, PartialResponse)
    assert evidence.reason == PartialResponseReason.MALFORMED
    assert evidence.content == "safe"
    assert [row.token for row in evidence.token_evidence] == ["safe"]
    assert evidence.usage == {"completion_tokens": 1}
    assert evidence.reasoning_characters == 5
    assert not evidence.usable_answer and evidence.mia_unavailable_reason
    assert len(json.dumps(evidence.to_dict(), ensure_ascii=False).encode()) <= 1024


@pytest.mark.parametrize("malformed", [
    delta(finish="\ud800"),
    delta("bad", tokens=[token("\ud800")]),
    delta("bad", tokens=[token("bad") | {"top_logprobs": [token("\ud800")]}]),
])
def test_escaped_surrogates_in_provider_metadata_fail_before_evidence_mutation(malformed):
    parser = StreamAssembler()
    parser.feed(event(delta("safe", tokens=[token("safe")])))
    wire = ("data: " + json.dumps(malformed, ensure_ascii=True) + "\n\n").encode()
    with pytest.raises(RequestError) as caught:
        parser.feed(wire)
    evidence = caught.value.response_evidence
    assert evidence.reason == PartialResponseReason.MALFORMED
    assert evidence.content == "safe" and evidence.finish_reason is None
    assert [row.token for row in evidence.token_evidence] == ["safe"]
    assert len(json.dumps(evidence.to_dict(), ensure_ascii=False).encode()) <= evidence.max_retained_bytes


@pytest.mark.parametrize("late", [delta("late"), delta(reasoning="late"),
                                  delta(tokens=[token("late")]), delta(finish="length")])
def test_choice_finish_rejects_late_evidence_without_mutating_prior_answer(late):
    parser = StreamAssembler()
    parser.feed(event(delta("safe", tokens=[token("safe")], finish="stop")))
    with pytest.raises(RequestError) as caught:
        parser.feed(event(late))
    evidence = caught.value.response_evidence
    assert evidence.reason == PartialResponseReason.MALFORMED
    assert evidence.content == "safe" and evidence.finish_reason == "stop"
    assert [row.token for row in evidence.token_evidence] == ["safe"]
    assert evidence.reasoning_characters == 0


def test_done_rejects_later_events_and_failed_parser_retains_same_evidence():
    parser = StreamAssembler()
    parser.feed(complete(event(delta("safe", tokens=[token("safe")]))))
    with pytest.raises(RequestError) as caught:
        parser.feed(event(delta("late")))
    error = caught.value
    snapshot = parser.progress
    assert error.response_evidence.content == "safe"
    assert [row.token for row in error.response_evidence.token_evidence] == ["safe"]
    for action in (lambda: parser.feed(event(delta("more"))), parser.finish):
        with pytest.raises(RequestError) as repeated:
            action()
        assert repeated.value is error
        assert parser.progress == snapshot


def test_finished_parser_rejects_mutation_and_preserves_returned_response():
    parser = StreamAssembler()
    parser.feed(complete(event(delta("safe", tokens=[token("safe")]))))
    result = parser.finish()
    snapshot = parser.progress
    for action in (lambda: parser.feed(event(delta("late"))), parser.finish):
        with pytest.raises(ValueError, match="already finished"):
            action()
    assert result.content == "safe" and result.finish_reason == "stop"
    assert [row.token for row in result.token_evidence] == ["safe"]
    assert parser.progress == snapshot


@pytest.mark.parametrize("chunk_sizes", [(33,), (10, 23), (1,) * 33])
@pytest.mark.parametrize("response_limit,reason", [(32, PartialResponseReason.RESPONSE_LIMIT),
                                                  (64, PartialResponseReason.EVENT_LIMIT)])
def test_transport_bytes_count_all_supplied_oversized_input(chunk_sizes, response_limit, reason):
    parser = StreamAssembler(max_event_bytes=32, max_response_bytes=response_limit)
    with pytest.raises(RequestError) as caught:
        for size in chunk_sizes:
            parser.feed(b"x" * size)
    assert caught.value.response_evidence.reason == reason
    assert parser.progress.transport_bytes == 33
    assert parser.progress.answer_characters == 0


def test_transport_bytes_include_unprocessed_tail_after_malformed_event():
    parser = StreamAssembler()
    prior = event(delta("safe", tokens=[token("safe")]))
    malformed = b"data: not-json\n\n" + b"unprocessed tail" * 100
    parser.feed(prior)
    with pytest.raises(RequestError) as caught:
        parser.feed(malformed)
    assert parser.progress.transport_bytes == len(prior) + len(malformed)
    assert caught.value.response_evidence.content == "safe"
    assert [row.token for row in caught.value.response_evidence.token_evidence] == ["safe"]


@pytest.mark.parametrize("kwargs", [
    {"request_id": "private-\ud800"},
    {"requested_settings": {"vendor_label": "private-\ud800"}},
    {"requested_settings": {"private-\ud800": "value"}},
    {"requested_settings": {"vendor_label": [{"nested": ("private-\ud800",)}]}},
    {"requested_settings": {"vendor_label": [{"private-\ud800": "value"}]}},
])
def test_unencodable_local_metadata_is_rejected_at_construction(kwargs):
    with pytest.raises(ValueError, match="^Stream metadata must be UTF-8 encodable$") as caught:
        StreamAssembler(**kwargs)
    assert "private" not in str(caught.value)
    assert caught.value.__cause__ is None


def test_valid_unicode_local_metadata_survives_partial_evidence():
    parser = StreamAssembler(request_id="请求🙂", requested_settings={"vendor_label": {"标签": ["值", ("🙂",)]}})
    evidence = parser.partial()
    assert evidence.request_id == "请求🙂"
    assert evidence.to_dict()["requested_settings"] == {"vendor_label": {"标签": ["值", ["🙂"]]}}
    with pytest.raises(RequestError) as caught:
        parser.finish()
    assert caught.value.response_evidence.to_dict() == evidence.to_dict() | {"reason": "premature_eof"}


def test_oversized_unicode_local_metadata_is_omitted_within_partial_limit():
    parser = StreamAssembler(max_retained_bytes=1024, request_id="🙂" * 4000,
                             requested_settings={"vendor_label": {"标签": ["🙂" * 4000]}})
    parser.feed(event(delta("safe")))
    evidence = parser.partial()
    assert evidence.content == "safe"
    assert evidence.request_id is None and evidence.requested_settings == {}
    assert len(json.dumps(evidence.to_dict(), ensure_ascii=False).encode()) <= 1024
