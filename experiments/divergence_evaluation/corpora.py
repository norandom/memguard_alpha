"""Local corpus provenance and whole-document controls; no fetching or tokenization.

Token counts are operator declarations for the frozen tokenizer. Fitting must
verify them against actual tokenization before treating them as measured counts.
"""

import hashlib
import random
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from .contracts import CorpusDocument, JsonContract, Split

OLDER_END = datetime(2020, 1, 1, tzinfo=UTC)
NEWER_END = datetime(2024, 7, 1, tzinfo=UTC)


@dataclass(frozen=True)
class CorpusEntry(JsonContract):
    document: CorpusDocument
    content_path: str
    token_count: int


@dataclass(frozen=True)
class SemanticAudit(JsonContract):
    reviewer: str
    evidence_reference: str
    audited_at: str
    document_ids: tuple[str, ...]
    near_duplicates_clear: bool
    factual_exposure_reviewed: bool


@dataclass(frozen=True)
class FactExposure(JsonContract):
    fact_id: str
    document_ids: tuple[str, ...]
    rationale: str
    evidence_reference: str


@dataclass(frozen=True)
class CorpusManifest(JsonContract):
    entries: tuple[CorpusEntry, ...]
    older_document_ids: tuple[str, ...]
    newer_document_ids: tuple[str, ...]
    semantic_audit: SemanticAudit | None
    intentional_fact_exposures: tuple[FactExposure, ...] = ()


@dataclass(frozen=True)
class AdmissionIssue(JsonContract):
    code: str
    document_ids: tuple[str, ...]
    reason: str


@dataclass(frozen=True)
class CorpusAdmission(JsonContract):
    admitted: bool
    issues: tuple[AdmissionIssue, ...]
    # Row grain: source ID, exposure period, documents, declared tokenizer tokens.
    older_counts: tuple[tuple[str, str, int, int], ...]
    newer_counts: tuple[tuple[str, str, int, int], ...]
    intentional_fact_exposures: tuple[FactExposure, ...]


@dataclass(frozen=True)
class MatchedSizeControl(JsonContract):
    seed: int
    document_ids: tuple[str, ...]
    target_tokens: int
    actual_tokens: int
    discrepancy_tokens: int
    counts_by_source_window: tuple[tuple[str, str, int, int], ...]
    # Row grain: source, period, target tokens, actual tokens, actual minus target.
    stratum_discrepancies: tuple[tuple[str, str, int, int, int], ...]


def _timestamp(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if "T" not in value or parsed.utcoffset() is None:
        raise ValueError("timestamp requires time and timezone")
    return parsed.astimezone(UTC)


def _window(entry: CorpusEntry) -> str:
    # Latest revision is the evidence boundary for the exact bytes being fitted.
    return "older" if _timestamp(entry.document.revised_at) < OLDER_END else "later"


def _counts(entries: tuple[CorpusEntry, ...]) -> tuple[tuple[str, str, int, int], ...]:
    totals = defaultdict(lambda: [0, 0])
    for entry in entries:
        try:
            window = _window(entry)
        except ValueError:
            window = "invalid_timestamp"
        row = totals[entry.document.source_id, window]
        row[0] += 1
        row[1] += entry.token_count
    return tuple((source, window, *totals[source, window]) for source, window in sorted(totals))


def validate_corpora(manifest: CorpusManifest, content_root: Path) -> CorpusAdmission:
    """Accumulate blocking evidence without rewriting dates, text or audit decisions."""
    issues = []

    def reject(code, ids, reason):
        issues.append(AdmissionIssue(code, tuple(ids), reason))

    root = Path(content_root).resolve()
    by_id = {}
    by_hash = defaultdict(list)
    fact_documents = defaultdict(set)
    for entry in manifest.entries:
        document = entry.document
        ids = (document.document_id,)
        if document.document_id in by_id:
            reject("duplicate_document_id", ids, "Document IDs must be unique")
        by_id[document.document_id] = entry
        by_hash[document.content_hash].append(document)
        for fact_id in document.fact_exposure_ids:
            fact_documents[fact_id].add(document.document_id)
        for field in ("document_id", "source_id", "rights_reference"):
            if not getattr(document, field).strip():
                reject("rights" if field == "rights_reference" else "identity", ids, f"Missing {field}")
        if type(entry.token_count) is not int or entry.token_count <= 0:
            reject("token_count", ids, "Declared tokenizer token count must be positive")
        try:
            published = _timestamp(document.published_at)
            revised = _timestamp(document.revised_at)
            if revised < published:
                reject("revision_before_publication", ids, "Revision predates publication")
        except ValueError:
            reject("timestamp", ids, "Publication and revision require timezone-aware ISO timestamps")
        try:
            path = (root / entry.content_path).resolve()
            if Path(entry.content_path).is_absolute() or not path.is_relative_to(root):
                reject("local_path", ids, "Content must be a relative path within the local corpus root")
            elif hashlib.sha256(path.read_bytes()).hexdigest() != document.content_hash:
                reject("content_hash", ids, "Local bytes differ from declared SHA-256")
        except (OSError, ValueError):
            reject("local_content", ids, "Local content cannot be read")

    older_ids = set(manifest.older_document_ids)
    newer_ids = set(manifest.newer_document_ids)
    if not older_ids or not newer_ids:
        reject("empty_fit", (), "Both declared fitting corpora must be nonempty")
    if not older_ids <= newer_ids:
        reject("older_not_in_newer", sorted(older_ids - newer_ids), "Newer must include every older document")
    for role, declared_ids, end in (("older", manifest.older_document_ids, OLDER_END),
                                    ("newer", manifest.newer_document_ids, NEWER_END)):
        if len(declared_ids) != len(set(declared_ids)):
            reject("duplicate_fit_id", declared_ids, f"Repeated IDs in {role} corpus")
        for document_id in declared_ids:
            if document_id not in by_id:
                reject("unknown_fit_id", (document_id,), f"Unknown {role} fitting document")
                continue
            document = by_id[document_id].document
            if document.split != Split.FITTING:
                reject("fit_partition", (document_id,), "Auxiliary exposure must use fitting documents only")
            try:
                if max(_timestamp(document.published_at), _timestamp(document.revised_at)) >= end:
                    reject(f"{role}_window", (document_id,), f"Exact revision falls outside {role} exposure window")
            except ValueError:
                pass  # Timestamp failure already retained above.
    for documents in by_hash.values():
        if len({document.split for document in documents}) > 1:
            reject("split_text_overlap", (document.document_id for document in documents),
                   "Identical text occurs across declared partitions")

    audit = manifest.semantic_audit
    if audit is None:
        reject("semantic_audit", (), "Human semantic near-duplicate and factual-exposure audit is required")
    else:
        try:
            _timestamp(audit.audited_at)
            valid_audit = (audit.reviewer.strip() and audit.evidence_reference.strip()
                           and set(audit.document_ids) == set(by_id) and audit.near_duplicates_clear
                           and audit.factual_exposure_reviewed)
        except ValueError:
            valid_audit = False
        if not valid_audit:
            reject("semantic_audit", audit.document_ids, "Human audit is incomplete or reports unresolved overlap")
    exposures = {}
    for exposure in manifest.intentional_fact_exposures:
        if (exposure.fact_id in exposures or not exposure.fact_id.strip() or not exposure.document_ids
                or len(exposure.document_ids) != len(set(exposure.document_ids)) or not exposure.rationale.strip()
                or not exposure.evidence_reference.strip()
                or set(exposure.document_ids) != fact_documents.get(exposure.fact_id, set())):
            reject("invalid_fact_exposure", exposure.document_ids, "Shared factual exposure needs exact IDs and audit evidence")
        exposures[exposure.fact_id] = exposure
    for fact_id, document_ids in fact_documents.items():
        if len({by_id[document_id].document.split for document_id in document_ids}) > 1 and fact_id not in exposures:
            reject("unattested_fact_exposure", sorted(document_ids), "Cross-partition fact exposure must be declared separately")

    older = tuple(entry for entry in manifest.entries if entry.document.document_id in older_ids)
    newer = tuple(entry for entry in manifest.entries if entry.document.document_id in newer_ids)
    return CorpusAdmission(not issues, tuple(issues), _counts(older), _counts(newer),
                           manifest.intentional_fact_exposures)


def matched_size_control(reference: tuple[CorpusEntry, ...], candidates: tuple[CorpusEntry, ...],
                         *, seed: int) -> MatchedSizeControl:
    """Downsample full newer fitting corpus to older size, preserving newer strata.

Use proportional integer token quotas (largest remainder), then the nearest
prefix of a seeded permutation per source/period. Whole-document granularity
can undershoot or overshoot; this is a reproducible heuristic, not an optimum.
Call only after admission, using the pinned tokenizer's declared token counts.
"""
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    for entries in (reference, candidates):
        if len({entry.document.document_id for entry in entries}) != len(entries):
            raise ValueError("duplicate control document IDs")
        for entry in entries:
            published = _timestamp(entry.document.published_at)
            revised = _timestamp(entry.document.revised_at)
            if (entry.document.split != Split.FITTING or entry.token_count <= 0
                    or revised < published or revised >= NEWER_END):
                raise ValueError("control inputs must be admitted fitting documents")
    target = sum(entry.token_count for entry in reference)
    strata = defaultdict(list)
    for entry in candidates:
        strata[entry.document.source_id, _window(entry)].append(entry)
    totals = {key: sum(entry.token_count for entry in entries) for key, entries in strata.items()}
    total = sum(totals.values())
    quotas = {key: target * tokens // total for key, tokens in totals.items()}
    remainder_order = sorted(totals, key=lambda key: (-(target * totals[key] % total), key))
    for key in remainder_order[:target - sum(quotas.values())]:
        quotas[key] += 1
    rng = random.Random(seed)
    selected = []
    discrepancies = []
    for key in sorted(strata):
        entries = sorted(strata[key], key=lambda entry: entry.document.document_id)
        rng.shuffle(entries)
        prefix_tokens = [0]
        for entry in entries:
            prefix_tokens.append(prefix_tokens[-1] + entry.token_count)
        count = min(range(len(prefix_tokens)), key=lambda index: (abs(prefix_tokens[index] - quotas[key]), index))
        selected.extend(entries[:count])
        discrepancies.append((*key, quotas[key], prefix_tokens[count], prefix_tokens[count] - quotas[key]))
    actual = sum(entry.token_count for entry in selected)
    return MatchedSizeControl(seed, tuple(sorted(entry.document.document_id for entry in selected)),
                              target, actual, actual - target, _counts(tuple(selected)), tuple(discrepancies))
