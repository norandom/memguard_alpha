"""Hand-calculated sparse estimators using only tiny synthetic token corpora."""

import hashlib
import json
from dataclasses import replace

import numpy as np
import pytest

from experiments.divergence_evaluation import trigram
from experiments.divergence_evaluation.contracts import Cohort, CorpusDocument, FittingLimits, Split
from experiments.divergence_evaluation.corpora import CorpusEntry, CorpusManifest, FactExposure, SemanticAudit
from experiments.divergence_evaluation.trigram import SparseTrigram, fit_trigram

TOKENIZER = "a" * 64
LIMITS = FittingLimits(20, 100, 1_000_000)


def corpus(tmp_path, documents=((0, 1, 0), (1,))):
    entries = []
    for index, tokens in enumerate(documents):
        content = " ".join(map(str, tokens)).encode()
        name = str(index)
        (tmp_path / name).write_bytes(content)
        document = CorpusDocument(name, "synthetic", hashlib.sha256(content).hexdigest(),
                                  "synthetic-license", "2019-01-01T00:00:00Z", "2019-01-01T00:00:00Z",
                                  Cohort.HISTORICAL, Split.FITTING)
        entries.append(CorpusEntry(document, name, len(tokens)))
    ids = tuple(entry.document.document_id for entry in entries)
    return CorpusManifest(tuple(entries), ids, ids,
                          SemanticAudit("tester", "synthetic-audit", "2026-11-01T00:00:00Z", ids, True, True))


def fit(tmp_path, data=None, **kwargs):
    options = dict(role="older", tokenizer_mapping_hash=TOKENIZER, vocabulary_size=3,
                   tokenize=lambda text: [int(token) for token in text.split()], limits=LIMITS)
    options.update(kwargs)
    return fit_trigram(data or corpus(tmp_path), tmp_path, **options)


def test_hand_calculated_interpolation_uses_outgoing_counts_and_document_resets(tmp_path):
    model = fit(tmp_path)
    p1 = np.array([3, 3, 1]) / 7
    p2 = (np.array([1, 0, 0]) + p1) / 2
    p3 = (np.array([1, 0, 0]) + p2) / 2
    np.testing.assert_allclose(np.exp(model.log_probabilities([])), p1)
    np.testing.assert_allclose(np.exp(model.log_probabilities([1])), p2)
    np.testing.assert_allclose(np.exp(model.log_probabilities([0, 1])), p3)
    # End-of-document 1 must not create a second outgoing bigram; no 1,1 bridge.
    np.testing.assert_allclose(np.exp(model.log_probabilities([1, 0])),
                               (np.array([0, 1, 0]) + p1) / 2)
    assert model.receipt.documents == 2
    assert model.receipt.tokens == 4
    assert model.manifest.counts_by_source_window == (("synthetic", "older", 2, 4),)


@pytest.mark.parametrize("prefix", [[], [0], [1], [2], [0, 1], [2, 1], [2, 2], [1, 2, 0, 1]])
def test_full_vocab_finite_normalized_and_unseen_backoff(tmp_path, prefix):
    model = fit(tmp_path)
    scores = model.log_probabilities(prefix)
    assert scores.shape == (3,)
    assert np.isfinite(scores).all()
    assert np.exp(scores).sum() == pytest.approx(1)
    np.testing.assert_array_equal(model.log_probabilities([2, 2]), model.log_probabilities([]))
    np.testing.assert_array_equal(model.log_probabilities([2, 1]), model.log_probabilities([1]))


def test_reload_reproduces_scores_and_requires_tokenizer_identity(tmp_path):
    model = fit(tmp_path)
    path = tmp_path / "model.json"
    model.save(path)
    loaded = SparseTrigram.load(path, tokenizer_mapping_hash=TOKENIZER, vocabulary_size=3)
    assert loaded.manifest == model.manifest
    assert loaded.receipt == model.receipt
    for prefix in ([], [0], [1, 0], [0, 1]):
        np.testing.assert_array_equal(loaded.log_probabilities(prefix), model.log_probabilities(prefix))
    with pytest.raises(FileExistsError):
        model.save(path)
    with pytest.raises(ValueError, match="tokenizer"):
        SparseTrigram.load(path, tokenizer_mapping_hash="b" * 64, vocabulary_size=3)
    with pytest.raises(ValueError, match="vocabulary"):
        SparseTrigram.load(path, tokenizer_mapping_hash=TOKENIZER, vocabulary_size=4)


def test_actual_pinned_tokenizer_counts_are_checked(tmp_path):
    data = corpus(tmp_path)
    with pytest.raises(ValueError, match="declared token count"):
        fit(tmp_path, data, tokenize=lambda text: [0])
    assert fit(tmp_path, data, role="newer").manifest.role == "newer"


@pytest.mark.parametrize("limits,reason", [
    (FittingLimits(1, 100, 1_000_000), "document"),
    (FittingLimits(20, 3, 1_000_000), "token"),
    (FittingLimits(20, 100, 1), "memory"),
])
def test_limits_fail_without_silent_clipping(tmp_path, limits, reason):
    called = []
    with pytest.raises(ValueError, match=reason):
        fit(tmp_path, limits=limits, tokenize=lambda text: called.append(text))
    assert not called


def test_memory_receipt_distinguishes_measured_owned_objects_from_process_peak(tmp_path):
    model = fit(tmp_path)
    assert model.receipt.measured_count_bytes > 0
    assert model.receipt.estimated_peak_bytes >= model.receipt.measured_count_bytes
    assert model.receipt.estimated_peak_bytes <= LIMITS.max_memory_bytes
    assert model.receipt.peak_process_memory_bytes is None
    assert model.receipt.elapsed_seconds >= 0
    assert model.receipt.estimate_kind == "conservative-owned-fitting-allocations-v1"
    assert "tokenizer callback" in model.receipt.excluded_memory


def test_sparse_growth_is_refused_before_next_token_is_admitted(tmp_path):
    full_estimate_bytes = fit(tmp_path).receipt.estimated_peak_bytes
    seen = []

    def tokenize(text):
        for token in text.split():
            seen.append(token)
            yield int(token)

    # A budget just below the complete estimate refuses the final new count.
    with pytest.raises(ValueError, match="memory"):
        fit(tmp_path, limits=FittingLimits(20, 100, full_estimate_bytes - 1), tokenize=tokenize)
    assert seen == ["0", "1", "0"]


def test_control_role_fits_only_selected_admitted_documents(tmp_path):
    data = corpus(tmp_path)
    model = fit(tmp_path, data, role="matched_size_control", control_document_ids=("1",))
    assert model.receipt.tokens == 1
    np.testing.assert_allclose(np.exp(model.log_probabilities([0, 1])), np.array([1, 2, 1]) / 4)
    assert model.manifest.source_hashes == (data.entries[1].document.content_hash,)
    with pytest.raises(ValueError, match="control"):
        fit(tmp_path, data, role="matched_size_control", control_document_ids=("absent",))


@pytest.mark.parametrize("tokens", [[-1, 0, 0], [3, 0, 0], [True, 0, 0], [0.0, 0, 0]])
def test_invalid_tokenizer_ids_rejected(tmp_path, tokens):
    with pytest.raises(ValueError, match="token ID"):
        fit(tmp_path, tokenize=lambda text: tokens)


def test_invalid_prefix_ids_rejected(tmp_path):
    model = fit(tmp_path)
    for prefix in ([3], [-1], [True], [0.0]):
        with pytest.raises(ValueError, match="token ID"):
            model.log_probabilities(prefix)


def test_provenance_failure_blocks_fitting(tmp_path):
    data = corpus(tmp_path)
    (tmp_path / "0").write_text("tampered", encoding="utf-8")
    with pytest.raises(ValueError, match="admission"):
        fit(tmp_path, data)


def test_corrupted_sparse_counts_and_policy_rejected(tmp_path):
    model = fit(tmp_path)
    path = tmp_path / "model.json"
    model.save(path)
    original = json.loads(path.read_text(encoding="utf-8"))
    for field, value in (("unigrams", [[3, 1]]), ("bigrams", [[0, 1, -1]]),
                         ("trigrams", [[0, 1, 0, 10]])):
        payload = dict(original, **{field: value})
        path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(ValueError):
            SparseTrigram.load(path, tokenizer_mapping_hash=TOKENIZER, vocabulary_size=3)
    payload = dict(original)
    payload["manifest"]["smoothing_beta"] = 2
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="policy"):
        SparseTrigram.load(path, tokenizer_mapping_hash=TOKENIZER, vocabulary_size=3)


def test_declaration_cannot_bypass_actual_fitting_token_cap(tmp_path):
    data = corpus(tmp_path)
    data = replace(data, entries=tuple(replace(entry, token_count=1) for entry in data.entries))
    with pytest.raises(ValueError, match="token"):
        fit(tmp_path, data, limits=FittingLimits(20, 3, 1_000_000))


@pytest.mark.parametrize("metadata_kind", ["facts", "audit", "older", "newer", "exposures", "text", "control"])
def test_variable_metadata_budget_is_checked_before_admission(tmp_path, monkeypatch, metadata_kind):
    data = corpus(tmp_path, ((0,),))
    repeated_ids = ("0",) * 5000
    options = {}
    if metadata_kind == "facts":
        document = replace(data.entries[0].document, fact_exposure_ids=tuple(f"fact-{index}" for index in range(5000)))
        data = replace(data, entries=(replace(data.entries[0], document=document),))
    elif metadata_kind == "audit":
        data = replace(data, semantic_audit=replace(data.semantic_audit, document_ids=repeated_ids))
    elif metadata_kind in ("older", "newer"):
        data = replace(data, **{f"{metadata_kind}_document_ids": repeated_ids})
    elif metadata_kind == "exposures":
        data = replace(data, intentional_fact_exposures=(FactExposure("fact", repeated_ids, "synthetic", "audit"),))
    elif metadata_kind == "text":
        data = replace(data, semantic_audit=replace(data.semantic_audit, evidence_reference="x" * 100_000))
    else:
        options = dict(role="matched_size_control", control_document_ids=repeated_ids)
    calls = []
    original_admission = trigram.validate_corpora

    def admission(*args):
        calls.append(True)
        return original_admission(*args)

    monkeypatch.setattr(trigram, "validate_corpora", admission)
    with pytest.raises(ValueError, match="memory"):
        fit(tmp_path, data, limits=FittingLimits(20, 100, 100_000), **options)
    assert not calls
