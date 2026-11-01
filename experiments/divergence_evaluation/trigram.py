"""Locally fitted sparse trigrams, with document resets and fixed beta=1.

The memory limit bounds a conservative estimate of owned fitting allocations,
not process RSS. Supplied tokenizer callback allocations are outside that bound;
process resource controls remain the responsibility of the execution worker.
"""

import hashlib
import json
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass, fields, is_dataclass
from pathlib import Path

import numpy as np

from .contracts import AuxiliaryManifest, FittingLimits, JsonContract
from .corpora import CorpusManifest, _counts, validate_corpora

# Allowance covers the estimator containers/metadata and temporary text bytes.
# Each sparse entry reserves substantially more than tuple/int/dict storage.
_BASE_BYTES = 8192
_ENTRY_BYTES = 1024


@dataclass(frozen=True)
class FittingReceipt(JsonContract):
    documents: int
    tokens: int
    measured_count_bytes: int
    estimated_peak_bytes: int
    elapsed_seconds: float
    estimate_kind: str = "conservative-owned-fitting-allocations-v1"
    excluded_memory: str = "tokenizer callback allocations and process/runtime memory"
    peak_process_memory_bytes: int | None = None


def _token_id(token, vocabulary_size):
    # NumPy integer token arrays are legitimate callback inputs; booleans are not.
    if isinstance(token, (bool, np.bool_)) or not isinstance(token, (int, np.integer)):
        raise ValueError("token ID must be an integer")
    token = int(token)
    if not 0 <= token < vocabulary_size:
        raise ValueError("token ID outside vocabulary")
    return token


def _memory_check(estimated_bytes, limits):
    if estimated_bytes > limits.max_memory_bytes:
        raise ValueError("owned fitting memory estimate exceeds max_memory_bytes")


def _metadata_bytes(manifest, control_document_ids, limits):
    """Reserve admission metadata without duplicating its variable collections.

    Count every occurrence, including repeated IDs: admission creates per-fact
    sets, partition/audit sets and exposure maps. A KiB per node conservatively
    covers these containers and references; string lengths additionally reserve
    temporary parsing/path/message copies without encoding or copying the text.
    Walk all typed fields so nested audit/exposure metadata cannot escape the cap.
    """
    estimated_bytes = _BASE_BYTES

    def reserve(value):
        nonlocal estimated_bytes
        estimated_bytes += 1024 + (16 * len(value) if isinstance(value, str) else 0)
        _memory_check(estimated_bytes, limits)
        if is_dataclass(value):
            for field in fields(value):
                reserve(getattr(value, field.name))
        elif isinstance(value, tuple):
            for item in value:
                reserve(item)

    reserve(manifest)
    reserve(control_document_ids)
    return estimated_bytes


def _measured_bytes(counts):
    """Measured shallow container sizes plus key/int sizes (shared ints overcount)."""
    return sum(sys.getsizeof(table) + sum(sys.getsizeof(key) + sys.getsizeof(count)
                                         + sum(sys.getsizeof(token) for token in key)
                                         for key, count in table.items()) for table in counts)


class SparseTrigram:
    """Sparse counts produce float64 log probabilities, axis 0 = tokenizer token ID."""

    def __init__(self, manifest, vocabulary_size, counts, receipt):
        self.manifest = manifest
        self.vocabulary_size = vocabulary_size
        self.receipt = receipt
        self._unigrams, self._bigrams, self._trigrams = counts
        self._bigram_outgoing = Counter()
        self._trigram_outgoing = Counter()
        self._bigram_next = defaultdict(dict)
        self._trigram_next = defaultdict(dict)
        for (previous, token), count in self._bigrams.items():
            self._bigram_outgoing[previous] += count
            self._bigram_next[previous][token] = count
        for (first, previous, token), count in self._trigrams.items():
            self._trigram_outgoing[first, previous] += count
            self._trigram_next[first, previous][token] = count

    def log_probabilities(self, prefix: Sequence[int]) -> np.ndarray:
        """Use the last two IDs; empty/one-token prefixes back off to p1/p2."""
        # Check the complete prefix rather than silently accepting an invalid ID.
        tail = []
        for token in prefix:
            tail.append(_token_id(token, self.vocabulary_size))
            if len(tail) > 2:
                del tail[0]
        probabilities = np.ones(self.vocabulary_size, dtype=np.float64)
        for (token,), count in self._unigrams.items():
            probabilities[token] += count
        probabilities /= self.receipt.tokens + self.vocabulary_size
        if tail:
            previous = tail[-1]
            for token, count in self._bigram_next.get(previous, {}).items():
                probabilities[token] += count
            probabilities /= self._bigram_outgoing[previous] + 1
        if len(tail) == 2:
            first, previous = tail
            for token, count in self._trigram_next.get((first, previous), {}).items():
                probabilities[token] += count
            probabilities /= self._trigram_outgoing[first, previous] + 1
        return np.log(probabilities)

    def save(self, path: Path) -> None:
        """Portable numeric JSON rows; exclusive creation prevents silent overwrite."""
        payload = {"manifest": asdict(self.manifest), "vocabulary_size": self.vocabulary_size,
                   "receipt": asdict(self.receipt)}
        for name, table in zip(("unigrams", "bigrams", "trigrams"),
                               (self._unigrams, self._bigrams, self._trigrams), strict=True):
            payload[name] = [list(key) + [count] for key, count in sorted(table.items())]
        with Path(path).open("x", encoding="utf-8") as output:
            json.dump(payload, output, sort_keys=True, allow_nan=False)

    @classmethod
    def load(cls, path: Path, *, tokenizer_mapping_hash: str, vocabulary_size: int):
        """Load JSON only, rejecting incompatible identity and malformed count tables."""
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or set(payload) != {
            "manifest", "vocabulary_size", "receipt", "unigrams", "bigrams", "trigrams"
        }:
            raise ValueError("invalid sparse trigram artifact fields")
        manifest = AuxiliaryManifest.from_dict(payload["manifest"])
        if manifest.tokenizer_mapping_hash != tokenizer_mapping_hash:
            raise ValueError("tokenizer mapping mismatch")
        if (type(vocabulary_size) is not int or vocabulary_size < 1
                or type(payload["vocabulary_size"]) is not int or payload["vocabulary_size"] != vocabulary_size):
            raise ValueError("vocabulary size mismatch")
        receipt = FittingReceipt.from_dict(payload["receipt"])
        counts = []
        for order, name in enumerate(("unigrams", "bigrams", "trigrams"), 1):
            rows = payload[name]
            if not isinstance(rows, list):
                raise ValueError("sparse counts must be numeric rows")
            table = {}
            for row in rows:
                if not isinstance(row, list) or len(row) != order + 1:
                    raise ValueError("invalid sparse count row")
                key = tuple(_token_id(token, vocabulary_size) for token in row[:-1])
                count = row[-1]
                if type(count) is not int or count <= 0 or key in table:
                    raise ValueError("invalid or duplicate sparse count")
                table[key] = count
            counts.append(table)
        _validate_counts(manifest, receipt, counts)
        return cls(manifest, vocabulary_size, counts, receipt)


def _validate_counts(manifest, receipt, counts):
    unigrams, bigrams, trigrams = counts
    documents = sum(row[2] for row in manifest.counts_by_source_window)
    tokens = sum(row[3] for row in manifest.counts_by_source_window)
    if (documents != receipt.documents or tokens != receipt.tokens or tokens != sum(unigrams.values())
            or len(manifest.source_hashes) != documents
            or any(not 0 < row[2] <= row[3] for row in manifest.counts_by_source_window)
            or not 0 < documents <= tokens or documents > manifest.fitting_limits.max_documents
            or tokens > manifest.fitting_limits.max_tokens):
        raise ValueError("artifact document/token count mismatch")
    if (sum(bigrams.values()) != tokens - documents
            or not max(0, tokens - 2 * documents) <= sum(trigrams.values()) <= sum(bigrams.values())):
        raise ValueError("invalid document-reset transition counts")
    outgoing = defaultdict(int)
    incoming = defaultdict(int)
    for (previous, token), count in bigrams.items():
        outgoing[previous] += count
        incoming[token] += count
    if any(count > unigrams.get((token,), 0) for token, count in outgoing.items()) or any(
        count > unigrams.get((token,), 0) for token, count in incoming.items()
    ):
        raise ValueError("bigram counts exceed unigram counts")
    starts = defaultdict(int)
    ends = defaultdict(int)
    for (first, previous, token), count in trigrams.items():
        starts[first, previous] += count
        ends[previous, token] += count
    if any(count > bigrams.get(key, 0) for table in (starts, ends) for key, count in table.items()):
        raise ValueError("trigram counts exceed bigram counts")
    if (receipt.measured_count_bytes <= 0 or receipt.estimated_peak_bytes < receipt.measured_count_bytes
            or receipt.estimated_peak_bytes > manifest.fitting_limits.max_memory_bytes
            or receipt.elapsed_seconds < 0 or not np.isfinite(receipt.elapsed_seconds)
            or receipt.peak_process_memory_bytes is not None
            or receipt.estimate_kind != "conservative-owned-fitting-allocations-v1"
            or receipt.excluded_memory != "tokenizer callback allocations and process/runtime memory"):
        raise ValueError("invalid owned fitting memory receipt")


def fit_trigram(manifest: CorpusManifest, content_root: Path, *, role: str,
                tokenizer_mapping_hash: str, vocabulary_size: int,
                tokenize: Callable[[str], Iterable[int]], limits: FittingLimits,
                control_document_ids: tuple[str, ...] | None = None) -> SparseTrigram:
    """Fit one role using the supplied pinned-tokenizer callback on exact UTF-8 text.

    Validate provenance and declared counts, never clip documents. The tokenizer
    must preserve the pinned policy (including any special tokens); its mapping
    identity is supplied explicitly and recorded. Its own allocations are excluded
    from the owned-memory estimate; no process peak measurement is claimed.
    """
    started = time.monotonic()
    if type(vocabulary_size) is not int or vocabulary_size <= 0:
        raise ValueError("vocabulary size must be positive")
    # Validate role/hash/frozen policy using the shared contract before any work.
    AuxiliaryManifest(role, tokenizer_mapping_hash, (), (), limits)
    # Admit metadata allocation before constructing our ID and entry containers.
    metadata_bytes = _metadata_bytes(manifest, control_document_ids, limits)
    if role == "matched_size_control":
        ids = control_document_ids
        if not ids or len(ids) != len(set(ids)) or not set(ids) <= set(manifest.newer_document_ids):
            raise ValueError("control document IDs must select unique newer fitting documents")
    else:
        if control_document_ids is not None:
            raise ValueError("control document IDs require matched_size_control role")
        ids = manifest.older_document_ids if role == "older" else manifest.newer_document_ids
    by_id = {entry.document.document_id: entry for entry in manifest.entries}
    if not ids or not set(ids) <= set(by_id):
        raise ValueError("unknown or empty fitting document IDs")
    entries = tuple(by_id[document_id] for document_id in ids)
    if len(entries) > limits.max_documents:
        raise ValueError("fitting document limit exceeded")
    if sum(entry.token_count for entry in entries) > limits.max_tokens:
        raise ValueError("fitting token limit exceeded")
    root = Path(content_root).resolve()
    # Metadata estimates include source/window aggregation and admission's maps.
    workspace_bytes = metadata_bytes
    # Admission also hashes held-out files one at a time. Account for its largest
    # file buffer before invoking admission, including non-fitting documents.
    for entry in manifest.entries:
        path = (root / entry.content_path).resolve()
        if not path.is_relative_to(root) or Path(entry.content_path).is_absolute():
            raise ValueError("corpus admission failed: local path")
        workspace_bytes = max(workspace_bytes, metadata_bytes + 5 * path.stat().st_size)
    _memory_check(workspace_bytes, limits)
    admission = validate_corpora(manifest, root)
    if not admission.admitted:
        raise ValueError("corpus admission failed: " + ", ".join(issue.code for issue in admission.issues))
    counts = ({}, {}, {})
    total_tokens = 0
    distinct_counts = 0
    for entry in entries:
        content = (root / entry.content_path).read_bytes()
        _memory_check(metadata_bytes + 5 * len(content) + distinct_counts * _ENTRY_BYTES, limits)
        if hashlib.sha256(content).hexdigest() != entry.document.content_hash:
            raise ValueError("corpus content changed after admission")
        document_tokens = 0
        tail = ()
        for raw_token in tokenize(content.decode("utf-8")):
            token = _token_id(raw_token, vocabulary_size)
            document_tokens += 1
            total_tokens += 1
            if total_tokens > limits.max_tokens:
                raise ValueError("actual fitting token limit exceeded")
            keys = ((token,),) + tuple(tail[-size:] + (token,) for size in range(1, len(tail) + 1))
            growth = sum(key not in table for key, table in zip(keys, counts, strict=False))
            _memory_check(workspace_bytes + (distinct_counts + growth) * _ENTRY_BYTES, limits)
            distinct_counts += growth
            for key, table in zip(keys, counts, strict=False):
                table[key] = table.get(key, 0) + 1
            tail = (tail + (token,))[-2:]
        if document_tokens != entry.token_count:
            raise ValueError(f"declared token count differs from pinned tokenizer for {entry.document.document_id}")
    fitted_manifest = AuxiliaryManifest(role, tokenizer_mapping_hash,
                                        tuple(entry.document.content_hash for entry in entries),
                                        _counts(entries), limits)
    receipt = FittingReceipt(len(entries), total_tokens, _measured_bytes(counts),
                             workspace_bytes + distinct_counts * _ENTRY_BYTES, time.monotonic() - started)
    _validate_counts(fitted_manifest, receipt, counts)
    return SparseTrigram(fitted_manifest, vocabulary_size, counts, receipt)
