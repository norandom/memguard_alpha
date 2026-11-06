"""Explicit metadata for mapping a GPT-OSS Harmony generation to its final answer."""

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

MARKERS = ("<|channel|>", "<|start|>", "<|end|>", "<|message|>", "<|return|>", "<|call|>")


@dataclass(frozen=True)
class HarmonyTokenMap:
    """Pinned identity and special IDs; ``marker_ids`` follows ``MARKERS`` order."""

    tokenizer_revision: str
    tokenizer_sha256: str
    marker_ids: tuple[int, ...]
    special_token_ids: frozenset[int]

    def __post_init__(self) -> None:
        if not isinstance(self.tokenizer_revision, str) or not re.fullmatch(r"[0-9a-f]{40}", self.tokenizer_revision):
            raise ValueError("Harmony mapping requires a pinned tokenizer revision")
        if not isinstance(self.tokenizer_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", self.tokenizer_sha256):
            raise ValueError("Harmony mapping requires the tokenizer SHA-256")
        if (not isinstance(self.marker_ids, tuple) or len(self.marker_ids) != len(MARKERS)
                or any(type(value) is not int or value < 0 for value in self.marker_ids)
                or len(set(self.marker_ids)) != len(MARKERS)):
            raise ValueError("Harmony control IDs must be distinct nonnegative integers")
        if (not isinstance(self.special_token_ids, frozenset)
                or any(type(value) is not int or value < 0 for value in self.special_token_ids)
                or not set(self.marker_ids) <= self.special_token_ids):
            raise ValueError("Harmony special token IDs must include every control ID")

    @classmethod
    def from_tokenizer_json(cls, path: str | Path, *, tokenizer_revision: str) -> "HarmonyTokenMap":
        if not isinstance(tokenizer_revision, str) or not re.fullmatch(r"[0-9a-f]{40}", tokenizer_revision):
            raise ValueError("Harmony mapping requires a pinned tokenizer revision")
        raw = Path(path).read_bytes()
        entries = json.loads(raw)["added_tokens"]
        if not isinstance(entries, list) or not all(isinstance(entry, dict) for entry in entries):
            raise ValueError("Harmony tokenizer special token metadata is malformed")
        labels = {}
        for entry in entries:
            if entry.get("special") is True:
                label, token_id = entry["content"], entry["id"]
                if not isinstance(label, str) or type(token_id) is not int or token_id < 0:
                    raise ValueError("Harmony tokenizer special token metadata is malformed")
                if label in labels or token_id in labels.values():
                    raise ValueError("Harmony tokenizer contains duplicate special token metadata")
                labels[label] = token_id
        try:
            ids = tuple(labels[label] for label in MARKERS)
        except KeyError:
            raise ValueError("Harmony tokenizer is missing control token metadata") from None
        return cls(tokenizer_revision, hashlib.sha256(raw).hexdigest(), ids, frozenset(labels.values()))


@dataclass(frozen=True)
class HarmonyFinalSpan:
    """Half-open positions in the original token_ids/logprobs arrays, not retokens."""

    start: int
    stop: int
    generated_positions: int


def _token_text(entry: object) -> str:
    if not isinstance(entry, dict) or not isinstance(entry.get("token"), str):
        raise ValueError("Harmony token text is missing or malformed")
    text = entry["token"]
    values = entry.get("bytes")
    if values is not None:
        if not isinstance(values, list) or any(type(value) is not int or not 0 <= value <= 255 for value in values):
            raise ValueError("Harmony token bytes are malformed")
        try:
            decoded = bytes(values).decode("utf-8")
        except UnicodeDecodeError:
            raise ValueError("Harmony split-byte decoding is unsupported") from None
        if decoded != text:
            raise ValueError("Harmony token byte and text representations disagree")
    return text


def final_answer_span(choice: dict, token_map: HarmonyTokenMap) -> HarmonyFinalSpan:
    """Map one complete final message using authoritative control IDs.

    The generated stream starts in the assistant header, as configured by the
    GPT-OSS template's ``<|start|>assistant`` generation prefix. Later messages
    must declare their role. Exactly one final channel must end with return;
    analysis is excluded. Tools, other channels, partial generations and multiple
    visible messages are unsupported. No content search or tokenization occurs.
    """
    if choice.get("finish_reason") != "stop":
        raise ValueError("Harmony mapping requires a complete stopped generation")
    message = choice.get("message")
    if (not isinstance(message, dict) or message.get("role") != "assistant"
            or message.get("tool_calls") or message.get("function_call")):
        raise ValueError("Harmony tool or missing messages are unsupported")
    content = message.get("content")
    if not isinstance(content, str):
        raise ValueError("Harmony final content must be text")
    ids = choice.get("token_ids")
    section = choice.get("logprobs")
    entries = section.get("content") if isinstance(section, dict) else None
    if (not isinstance(ids, list) or not isinstance(entries, list) or not ids or len(ids) != len(entries)
            or any(type(token_id) is not int or token_id < 0 for token_id in ids)):
        raise ValueError("Harmony original token IDs must match probability positions")
    texts = [_token_text(entry) for entry in entries]
    channel_id, start_id, end_id, message_id, return_id, call_id = token_map.marker_ids
    controls = dict(zip(token_map.marker_ids, MARKERS, strict=True))
    state, role_parts, channel_parts = "header", ["assistant"], []
    in_channel = False
    channel = None
    body_start = 0
    span = None
    for position, (token_id, text) in enumerate(zip(ids, texts, strict=True)):
        if token_id in controls and text != controls[token_id]:
            raise ValueError("Harmony control ID and rendered marker disagree")
        if state == "finished":
            raise ValueError("Harmony tokens follow the final return")
        if state == "next":
            if token_id != start_id:
                raise ValueError("Harmony message start is missing")
            state, role_parts, channel_parts, in_channel = "header", [], [], False
            continue
        if state == "header":
            if token_id == start_id and position == 0:
                role_parts = []
            elif token_id == channel_id and not in_channel:
                in_channel = True
            elif token_id == message_id:
                role, channel = "".join(role_parts).strip(), "".join(channel_parts).strip()
                if role != "assistant" or channel not in {"analysis", "final"}:
                    raise ValueError("Harmony role, recipient or channel is unsupported")
                state, body_start = "body", position + 1
            elif token_id in token_map.special_token_ids:
                raise ValueError("Harmony header contains an unexpected control token")
            else:
                (channel_parts if in_channel else role_parts).append(text)
            continue
        if token_id in {end_id, return_id}:
            if channel == "final":
                if span is not None or token_id != return_id:
                    raise ValueError("Harmony final message is ambiguous or incomplete")
                span = HarmonyFinalSpan(body_start, position, len(ids))
            elif token_id != end_id:
                raise ValueError("Harmony analysis terminated without a final message")
            state = "finished" if token_id == return_id else "next"
        elif token_id == call_id or token_id in token_map.special_token_ids:
            raise ValueError("Harmony body contains an unsupported control token")
    if state != "finished" or span is None:
        raise ValueError("Harmony final message is missing or truncated")
    if "".join(texts[span.start:span.stop]) != content:
        raise ValueError("Harmony final channel does not match message content")
    return span


def map_final_response(body: object, token_map: HarmonyTokenMap) -> dict:
    """Return a detached envelope selecting original entries; never change scores."""
    if not isinstance(body, dict) or not isinstance(body.get("choices"), list) or len(body["choices"]) != 1:
        raise ValueError("Harmony mapping requires exactly one response choice")
    choice = body["choices"][0]
    if not isinstance(choice, dict):
        raise ValueError("Harmony response choice is malformed")
    span = final_answer_span(choice, token_map)
    mapped_choice = choice | {"token_ids": choice["token_ids"][span.start:span.stop], "logprobs": choice["logprobs"] | {
        "content": choice["logprobs"]["content"][span.start:span.stop]}}
    return body | {"choices": [mapped_choice]}
