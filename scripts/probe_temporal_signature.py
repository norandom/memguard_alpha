"""Source-backed lexical signature diagnostic, without model API calls."""

import argparse
import hashlib
import json
import re
from collections import Counter
from datetime import UTC, datetime
from html.parser import HTMLParser
from pathlib import Path
from time import perf_counter

import numpy as np
import requests

from experiments.divergence_evaluation.contracts import AuxiliaryManifest, FittingLimits
from experiments.divergence_evaluation.trigram import FittingReceipt, SparseTrigram, _measured_bytes

DATES = ("20190130", "20190320", "20190501", "20190619",
         "20200315", "20200429", "20200610", "20200729")
TOKEN_PATTERN = r"[a-z]+(?:'[a-z]+)?|[0-9]+(?:-[0-9]+/[0-9]+|/[0-9]+)?|[^\w\s]"
# Row grain: diagnostic ID, category, prefix, candidate 1, candidate 2.
PROBES = (
    ("rate_at", "later_exposure", "the target range for the federal funds rate at",
     "0 to 1/4 percent", "2-1/4 to 2-1/2 percent"),
    ("rate_to", "later_exposure", "the target range for the federal funds rate to",
     "0 to 1/4 percent", "2-1/4 to 2-1/2 percent"),
    ("outbreak", "later_exposure", "the coronavirus outbreak has",
     "harmed communities", "strengthened communities"),
    ("risk", "later_exposure", "poses considerable",
     "risks", "opportunities"),
    ("labor", "earlier_exposure", "the labor market has",
     "continued to strengthen", "weakened"),
    ("spending", "earlier_exposure", "household spending has",
     "continued to grow", "collapsed"),
    ("mandate", "shared_control", "maximum employment and",
     "price stability", "asset speculation"),
    ("inflation", "shared_control", "symmetric 2 percent",
     "objective", "ceiling"),
    ("voting", "shared_control", "voting for the",
     "monetary policy action", "weather forecast"),
    ("rate_paraphrase", "paraphrase_stress", "the announced interest rate band was",
     "0 to 1/4 percent", "2-1/4 to 2-1/2 percent"),
    ("outbreak_paraphrase", "paraphrase_stress", "the public health crisis has",
     "harmed communities", "strengthened communities"),
    ("unseen", "unseen_control", "quasar nebula",
     "harmed communities", "strengthened communities"),
)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def digest(content):
    return hashlib.sha256(content).hexdigest()


class Paragraphs(HTMLParser):
    """Read paragraph text only; body selection is explicit and checked below."""

    def __init__(self):
        super().__init__()
        self.depth = 0
        self.current = []
        self.paragraphs = []

    def handle_starttag(self, tag, attrs):
        if tag == "p":
            # HTML permits an omitted </p>; the next <p> closes it implicitly.
            if self.depth:
                self.handle_endtag("p")
            self.depth = 1

    def handle_endtag(self, tag):
        if tag == "p" and self.depth:
            self.depth -= 1
            if not self.depth:
                self.paragraphs.append(" ".join("".join(self.current).split()))
                self.current = []

    def handle_data(self, data):
        if self.depth:
            self.current.append(data)


def extract_statement(html):
    parser = Paragraphs()
    parser.feed(html)
    start = next(index for index, paragraph in enumerate(parser.paragraphs)
                 if paragraph.startswith(("Information received since", "The coronavirus outbreak",
                                           "The Federal Reserve is committed")))
    end = next(index for index in range(start, len(parser.paragraphs))
               if parser.paragraphs[index].startswith("Voting for"))
    return "\n\n".join(parser.paragraphs[start:end + 1]) + "\n"


def words(text):
    return re.findall(TOKEN_PATTERN, text.lower())


def acquire(directory):
    directory.mkdir(parents=True, exist_ok=False)
    manifest = []
    for date in DATES:
        url = f"https://www.federalreserve.gov/newsevents/pressreleases/monetary{date}a.htm"
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        response.encoding = "utf-8"
        (directory / f"{date}.html").write_bytes(response.content)
        text = extract_statement(response.text)
        if not 150 < len(words(text)) < 2500:
            raise ValueError(f"Unexpected statement size: {date}")
        expected_date = datetime.strptime(date, "%Y%m%d").strftime("%B %d, %Y").replace(" 0", " ")
        if expected_date not in response.text:
            raise ValueError(f"Publication date not found: {date}")
        (directory / f"{date}.txt").write_text(text, encoding="utf-8")
        update = re.search(r"Last Update:.*?([A-Z][a-z]+\s+\d{1,2},\s+\d{4})",
                           response.text, re.DOTALL)
        manifest.append({
            "id": date, "url": url, "published_date": expected_date,
            "last_update_date": update.group(1) if update else None,
            "retrieved_at": datetime.now(UTC).isoformat(),
            "html_sha256": digest(response.content), "text_sha256": digest(text.encode()),
            "word_tokens": len(words(text)),
            "rights_reference": "https://www.federalreserve.gov/disclaimer.htm",
            "revision_evidence": "current page dates; not independently archived historical bytes",
        })
        write_json(directory / "sources.json", manifest)
        print(json.dumps({"downloaded": date, "tokens": len(words(text))}), flush=True)


def fit_counts(documents, role, vocabulary_size, tokenizer_hash):
    """Diagnostic-only count fit; does not assert production study admission."""
    started = perf_counter()
    tables = [Counter(), Counter(), Counter()]
    for document in documents:
        for order, table in enumerate(tables, start=1):
            table.update(tuple(document[index:index + order]) for index in range(len(document) - order + 1))
    token_count = sum(map(len, documents))
    source_hashes = tuple(digest(json.dumps(document).encode()) for document in documents)
    manifest = AuxiliaryManifest(
        role, tokenizer_hash, source_hashes,
        (("fomc-diagnostic-token-sequences", "mixed" if role == "newer" else "older",
          len(documents), token_count),),
        FittingLimits(32, 100_000, 128_000_000),
    )
    measured_bytes = _measured_bytes(tables)
    estimate_bytes = max(measured_bytes, 8192 + 1024 * sum(map(len, tables)))
    receipt = FittingReceipt(len(documents), token_count, measured_bytes, estimate_bytes,
                             perf_counter() - started)
    return SparseTrigram(manifest, vocabulary_size, tables, receipt)


def sequence_score(model, prefix_ids, candidate_ids):
    score = 0.0
    context = list(prefix_ids)
    for token_id in candidate_ids:
        score += float(model.log_probabilities(context)[token_id])
        context.append(token_id)
    return score


def measure(source_directory, output_directory):
    output_directory.mkdir(parents=True, exist_ok=False)
    sources = json.loads((source_directory / "sources.json").read_text(encoding="utf-8"))
    if [source["id"] for source in sources] != list(DATES):
        raise ValueError("Incomplete or reordered source corpus")
    documents = [(source_directory / f"{date}.txt").read_text(encoding="utf-8") for date in DATES]
    if any(digest(text.encode()) != source["text_sha256"] for text, source in zip(documents, sources, strict=True)):
        raise ValueError("Source hash mismatch")
    vocabulary = ["<unknown>"] + sorted({word for document in documents for word in words(document)})
    token_ids = {token: index for index, token in enumerate(vocabulary)}

    def encode(text):
        return [token_ids.get(word, 0) for word in words(text)]

    tokenizer_hash = digest(json.dumps({"pattern": TOKEN_PATTERN, "vocabulary": vocabulary}).encode())
    write_json(output_directory / "plan.json", {
        "sources": sources, "probes": PROBES, "tokenizer_hash": tokenizer_hash,
        "vocabulary": vocabulary, "pattern": TOKEN_PATTERN,
        "purpose": "source signature diagnostic; no target correction or efficacy evaluation",
        "admission": "exploratory counts only; no human semantic audit or study admission",
    })
    encoded = [encode(document) for document in documents]
    newer_token_count = sum(map(len, encoded))
    volume_documents = []
    remaining = newer_token_count
    index = 0
    while remaining:
        repeated = encoded[index % 4][:remaining]
        volume_documents.append(repeated)
        remaining -= len(repeated)
        index += 1
    models = {
        "older": fit_counts(encoded[:4], "older", len(vocabulary), tokenizer_hash),
        "newer": fit_counts(encoded, "newer", len(vocabulary), tokenizer_hash),
        "old_volume_control": fit_counts(volume_documents, "matched_size_control", len(vocabulary), tokenizer_hash),
    }
    for name, model in models.items():
        model.save(output_directory / f"{name}.json")
        restored = SparseTrigram.load(output_directory / f"{name}.json",
                                      tokenizer_mapping_hash=tokenizer_hash, vocabulary_size=len(vocabulary))
        np.testing.assert_array_equal(model.log_probabilities([]), restored.log_probabilities([]))
    rows = []
    for probe_id, category, prefix, candidate_one, candidate_two in PROBES:
        prefix_ids = encode(prefix)
        scores = {
            name: [sequence_score(model, prefix_ids, encode(candidate)) for candidate in (candidate_one, candidate_two)]
            for name, model in models.items()
        }
        old_preference = scores["older"][0] - scores["older"][1]
        new_preference = scores["newer"][0] - scores["newer"][1]
        volume_preference = scores["old_volume_control"][0] - scores["old_volume_control"][1]
        for model in models.values():
            assert np.isclose(np.exp(model.log_probabilities(prefix_ids)).sum(), 1)
            assert np.isfinite(model.log_probabilities(prefix_ids)).all()
        delta = models["older"].log_probabilities(prefix_ids) - models["newer"].log_probabilities(prefix_ids)
        sham = models["older"].log_probabilities(prefix_ids) - models["older"].log_probabilities(prefix_ids)
        assert np.array_equal(sham, np.zeros(len(vocabulary)))
        rows.append({
            "probe_id": probe_id, "category": category, "prefix": prefix,
            "candidate_one": candidate_one, "candidate_two": candidate_two,
            "sequence_log_probabilities": scores,
            "newer_minus_older_log_odds": new_preference - old_preference,
            "old_volume_minus_older_log_odds": volume_preference - old_preference,
            "newer_minus_old_volume_log_odds": new_preference - volume_preference,
            "prefix_unknown_tokens": prefix_ids.count(0),
            "candidate_unknown_tokens": [encode(text).count(0) for text in (candidate_one, candidate_two)],
            "distinct_final_context": prefix_ids[-2:],
            "full_next_token_delta_l2": float(np.linalg.norm(delta)),
            "sham_max_absolute_delta": float(np.max(np.abs(sham))),
        })
    write_json(output_directory / "probe-results.json", rows)
    later_rows = [row for row in rows if row["category"] == "later_exposure"]
    summary = {
        "documents_older": 4, "documents_newer": 8, "prompts": len(rows),
        "unique_final_contexts": len({tuple(row["distinct_final_context"]) for row in rows}),
        "tokens": {name: model.receipt.tokens for name, model in models.items()},
        "volume_control_documents": len(volume_documents),
        "later_probes_positive_beyond_volume": sum(row["newer_minus_old_volume_log_odds"] > 0 for row in later_rows),
        "later_probes": len(later_rows), "all_shams_zero": all(row["sham_max_absolute_delta"] == 0 for row in rows),
        "model_api_calls": 0, "mitigation_efficacy": "not_tested",
        "tokenizer_target_compatible": False, "human_study_audit": "not_performed",
    }
    write_json(output_directory / "summary.json", summary)
    print(json.dumps(summary), flush=True)
    for row in rows:
        print(json.dumps({key: row[key] for key in ("probe_id", "newer_minus_older_log_odds",
                                                   "newer_minus_old_volume_log_odds", "candidate_unknown_tokens")}),
              flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("acquire", "measure"))
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.action == "acquire":
        acquire(args.sources)
    else:
        if args.output is None:
            parser.error("--output is required for measure")
        measure(args.sources, args.output)
