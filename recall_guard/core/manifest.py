"""Run-manifest writer/reader and file hashing for the harness.

Implements the design's ``core.manifest`` interface (Req 6.5, 8.4, 10.1, 10.2):

- ``Manifest`` is a frozen dataclass holding the per-run reproducibility record
  (harness version, fixed seed, sha256 hashes of all input files, the resolved
  shortlist, the composite-score formula + weights, MCS hyperparameters, the
  bootstrap resample count, and an artifact-name -> path map).
- ``write_manifest(out_dir, manifest)`` serialises the dataclass to
  ``manifest.json`` (indented, sorted keys for stable diffs) and returns the
  path written. The output directory is created if missing.
- ``read_manifest(path)`` validates the JSON top-level shape and reconstructs
  the dataclass, so ``read_manifest(write_manifest(out_dir, m)) == m``.
- ``compute_file_hash(path)`` returns a sha256 hex digest of the file bytes,
  read in 8KB chunks so it tolerates files of arbitrary size without loading
  them fully into memory.

Design constraints honoured:
- ``shortlist`` keeps the design's ``list[str]`` typing; this means a
  ``Manifest`` instance is frozen-immutable but not ``hash()``-able, which is
  fine because we never use it as a dict key.
- Serialisation goes via ``dataclasses.asdict`` so nested dicts/lists round-trip
  cleanly without bespoke encoders.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

# Read in 8KB chunks: large enough to amortise the syscall cost, small enough
# to keep memory bounded for files of any size.
_HASH_CHUNK_SIZE = 8 * 1024


@dataclass(frozen=True)
class Manifest:
    """Per-run reproducibility manifest written to ``<out_dir>/manifest.json``.

    Fields mirror the design's ``core.manifest`` Service Interface verbatim.
    The dataclass is frozen so callers cannot mutate a manifest after it has
    been hashed/written, which keeps the persisted ``manifest.json`` faithful
    to whatever the runner actually saw.
    """

    harness_version: str
    seed: int
    eval_set_hash: str
    control_corpus_hash: str
    is_memorized_hash: str
    cutoffs_hash: str
    shortlist: list[str]
    composite_score: dict  # {"formula": str, "weights": dict[str, float] | None}
    mcs_hyperparams: dict
    bootstrap_n: int
    artifacts: dict[str, str]  # name -> path
    # Optional cmmd-backtest extension (Req 7.5, 8.2). When ``None`` the
    # manifest serialises to the pre-existing 11-key schema byte-identically;
    # ``write_manifest`` deliberately omits the key in that case so old runs
    # remain bit-stable. When a backtest block is supplied it must record the
    # fields listed in design.md § Manifest extension (signal_model, universe,
    # cash_ticker, cmmd_quantile, cmmd_threshold_value, fees_one_way,
    # init_cash, seed, bootstrap_n, n_is_rows, n_oos_rows, artifacts).
    backtest: dict | None = None


def compute_file_hash(path: Path | str) -> str:
    """Return the sha256 hex digest of the bytes at ``path``.

    Reads in 8KB chunks via ``hashlib.sha256().update(chunk)`` so the function
    can hash files larger than fit comfortably in memory. The chunked read is
    semantically equivalent to ``hashlib.sha256(path.read_bytes()).hexdigest()``
    for any file size; the dedicated test exercises a >16KB payload to make
    sure the chunk boundary does not corrupt the digest.
    """
    p = Path(path)
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(_HASH_CHUNK_SIZE), b""):
            h.update(chunk)
    return h.hexdigest()


def _expected_field_names() -> set[str]:
    """Return the set of Manifest field names (single source of truth)."""
    return {f.name for f in fields(Manifest)}


def write_manifest(out_dir: Path | str, manifest: Manifest) -> Path:
    """Serialise ``manifest`` to ``<out_dir>/manifest.json`` and return that path.

    ``out_dir`` is created (with parents) if it does not yet exist; this lets
    the runner pin the output directory at run start before any per-run
    artifact has been produced. JSON is indented and key-sorted so the file is
    diff-friendly across runs that differ only in metadata order.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = dataclasses.asdict(manifest)
    # Keep the on-disk schema byte-identical for legacy (no-backtest) runs:
    # absent the optional cmmd-backtest extension, ``backtest`` is omitted
    # rather than serialised as ``null``. This preserves the pre-existing
    # 11-key shape on which downstream tooling (and read_manifest's strict
    # key validator) is built.
    if payload.get("backtest") is None:
        payload.pop("backtest", None)

    target = out_dir / "manifest.json"
    with target.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
        # Trailing newline keeps the file POSIX-friendly for downstream tools.
        fh.write("\n")
    return target


def read_manifest(path: Path | str) -> Manifest:
    """Load ``manifest.json`` from ``path`` and reconstruct the dataclass.

    Validates the top-level shape: the JSON object must have exactly the same
    keys as ``Manifest``'s fields. Missing or extra keys raise ``ValueError``
    naming the offending key(s) so manifest drift is caught immediately rather
    than silently dropped on round-trip.
    """
    p = Path(path)
    with p.open("r", encoding="utf-8") as fh:
        decoded = json.load(fh)

    if not isinstance(decoded, dict):
        raise ValueError(
            f"Manifest at {p} must be a JSON object, got {type(decoded).__name__}."
        )

    expected = _expected_field_names()
    # ``backtest`` is the sole optional field (cmmd-backtest extension,
    # Req 7.5, 8.2): legacy manifests omit the key entirely, so it must not
    # count as missing or extraneous.
    optional = {"backtest"}
    required = expected - optional
    actual = set(decoded.keys())

    missing = required - actual
    if missing:
        raise ValueError(
            f"Manifest at {p} is missing required key(s): {sorted(missing)}."
        )
    extra = actual - expected
    if extra:
        raise ValueError(
            f"Manifest at {p} has unexpected key(s): {sorted(extra)}."
        )

    # Pass through field-by-field; types come straight from the JSON decode and
    # match the Manifest annotations (str/int for primitives, list/dict for
    # the structured fields).
    backtest_raw = decoded.get("backtest")
    backtest = dict(backtest_raw) if isinstance(backtest_raw, dict) else None
    return Manifest(
        harness_version=decoded["harness_version"],
        seed=decoded["seed"],
        eval_set_hash=decoded["eval_set_hash"],
        control_corpus_hash=decoded["control_corpus_hash"],
        is_memorized_hash=decoded["is_memorized_hash"],
        cutoffs_hash=decoded["cutoffs_hash"],
        shortlist=list(decoded["shortlist"]),
        composite_score=dict(decoded["composite_score"]),
        mcs_hyperparams=dict(decoded["mcs_hyperparams"]),
        bootstrap_n=decoded["bootstrap_n"],
        artifacts=dict(decoded["artifacts"]),
        backtest=backtest,
    )


# `field` is imported only so callers reading this module see the standard
# dataclass surface; it is not used directly here. Keep it referenced to avoid
# accidental future removal during refactors.
_ = field
