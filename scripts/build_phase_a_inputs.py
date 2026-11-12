"""Write the frozen Phase A measurement inputs (release-1.0 M4.1), offline.

For each selected profile this writes ``<output>/<profile>/{plan,manifest,corpus}.json``
in the local measurement format (``recall_guard.harness.local_evaluation.load_plan``)
and then rewrites ``<output>/index.json`` with every present file's SHA-256 and the
request totals. No credential, network, model or inference is used.

Phase A (measurement plan, approved 2026-11-08): per profile 1 probe + 40 memorized
+ 2 x 100 control + 80 held-out single answers, 0 ensemble draws = 321 logical
requests; temperature 0, top_p 1, top_logprobs 5, retries 0; seed 0, min_valid 50,
min_auc 0.6; local max_tokens 32 (context 2048), Modal 1024 with reasoning_effort low (context 4096;
rerun r2 after the first run spent its 512-token budget on reasoning).

Splits: calibration/is_memorized -> is_memorized, calibration/oos_control ->
oos_control, held_out/{is_memorized,oos_control} -> held_out. Every row carries its
label so reports can compute held-out AUC. ``item_id`` is sha256(metadata.url).

Input token declarations:

* Modal GPT-OSS-20B: exact counts of the one-user-message chat input with the
  pinned tokenizer and chat template (as ``probe_modal_model`` counts its pilot).
  Needs ``transformers`` and the pinned snapshot under ``models/hub``; run it in the
  Modal operator environment.
* Ollama: no tokenizer endpoint exists, so the declaration is a hard upper bound:
  UTF-8 bytes of the prompt (byte-level BPE and byte-fallback SentencePiece
  tokenizers never emit more than one token per byte) plus a fixed 128-token
  allowance for the Ollama chat template. A UTF-8 bytes / 2 estimate would not be
  a bound, and the journal stops a run whose reported prompt tokens exceed it.

Endpoint: the plan hash covers the manifest bytes, including the endpoint, so the
local endpoint is frozen at build time as the authenticated Windows gateway
(``http://<vEthernet (WSL) IPv4>:11435/v1/chat/completions``, the WSL default
gateway). Pass ``--local-endpoint`` explicitly; the run-time orchestrator refuses to
start when the current gateway address differs, and the inputs must then be rebuilt.

NPU (FastFlowLM) profiles: same EDGAR corpus, endpoint = the authenticated FLM
gateway (``http://<vEthernet (WSL) IPv4>:11541/v1/chat/completions``, pass
``--npu-endpoint``), requested execution mode ``NPU``, the profile's declared local
identity, and the same input-token upper bound (UTF-8 bytes + 128). Each NPU
profile also applies its own cutoff (``CUTOFF_KEYS`` -> data/cutoffs.yaml): control
rows (calibration and held-out) must be filed strictly AFTER the cutoff, memorized
rows strictly before it. Rows are dropped, never replaced; the counts and the
request total follow from what remains, and fewer calibration controls than
``min_valid`` (the baseline uses all of them) is refused. Only flm-gemma4-e4b
(cutoff 2025-01-31) loses rows; the corpus already lies after the other cutoffs.

Usage (dev container, repository checkout mounted at /phase-a):
  uv run --no-sync python scripts/build_phase_a_inputs.py --output /phase-a \
      --local-endpoint http://172.17.32.1:11435/v1/chat/completions --profiles local
Modal profile (operator venv with transformers and models/):
  uv run --project deploy/modal --no-sync python scripts/build_phase_a_inputs.py --profiles modal
NPU profiles:
  ... build_phase_a_inputs.py --profiles npu --npu-endpoint http://172.17.32.1:11541/v1/chat/completions
Modal rerun (r2; leaves the first run's inputs untouched):
  ... build_phase_a_inputs.py --profiles modal --output runs/phase-a/modal-gpt-oss-20b-r2
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path
from urllib.parse import urlsplit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from recall_guard.core.loader import load_cutoffs  # noqa: E402
from recall_guard.core.local_evidence import LocalModelIdentity  # noqa: E402
from recall_guard.core.profiles import load_profile  # noqa: E402
from recall_guard.harness.local_evaluation import MeasurementLimits, load_plan  # noqa: E402

LOCAL_PROFILES = ("ollama-phi4-mini", "ollama-llama3.2-3b", "ollama-gemma3-4b", "ollama-llama3.1-8b",
                  "ollama-llava-llama3-ctx2048")
MODAL_PROFILE = "modal-gpt-oss-20b"
#: Native Windows FastFlowLM (AMD NPU), reached through scripts/hosts/serve_local_flm.ps1.
NPU_PROFILES = ("flm-llama3.2-3b", "flm-gemma3-4b", "flm-gemma4-e4b")
PROFILES = (*LOCAL_PROFILES, MODAL_PROFILE, *NPU_PROFILES)
#: Profile -> cutoff registry key (data/cutoffs.yaml). FLM serves the same base models.
CUTOFF_KEYS = {"ollama-phi4-mini": "microsoft/phi-4-mini-instruct", "ollama-llama3.2-3b": "meta/llama-3.2-3b-instruct",
               "ollama-gemma3-4b": "google/gemma-3-4b-it", "ollama-llama3.1-8b": "meta/llama-3.1-8b-instruct",
               "ollama-llava-llama3-ctx2048": "xtuner/llava-llama-3-8b-v1_1", MODAL_PROFILE: "openai/gpt-oss-20b",
               "flm-llama3.2-3b": "meta/llama-3.2-3b-instruct", "flm-gemma3-4b": "google/gemma-3-4b-it",
               "flm-gemma4-e4b": "google/gemma-4-e4b-it"}
GATEWAY_PORTS = {"local": 11435, "npu": 11541}
SPLIT_FILES = {"is_memorized": ("calibration/is_memorized.jsonl",),
               "oos_control": ("calibration/oos_control.jsonl",),
               "held_out": ("held_out/is_memorized.jsonl", "held_out/oos_control.jsonl")}
LOGICAL_REQUESTS = 321  # 1 + 40 + 2 * 100 + 80 * (1 + 0)
TEMPLATE_ALLOWANCE_TOKENS = 128
LOCAL = {"context": 2048, "output": 32, "max_elapsed_s": 21600.0}  # 6 h per model: ~67 s per request
MODAL = {"context": 4096, "output": 1024, "max_elapsed_s": 2700.0}  # = manifest gpu_phase_seconds
MODAL_INPUT_MARGIN = 16  # Harmony system date text; exact counts plus a small fixed margin.
MODAL_MANIFEST = Path("configs/modal/gpt-oss-20b-measurement.json")
SETTINGS = {"temperature": 0, "top_p": 1, "logprobs": True, "top_logprobs": 5}
PLAN_FIELDS = {"prompt_template": "{prompt}", "selected_features": ["mia", "mcs"], "ensemble_draws": 0,
               "seed": 0, "min_valid": 50, "min_auc": 0.6}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def encode(data) -> bytes:
    """Byte-stable JSON (sorted keys, LF) so hashes match on Windows and Linux."""
    return (json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def read_rows(data_dir: Path) -> dict[str, list[dict]]:
    """Rows per split after checking each file against the EDGAR manifest hash and label."""
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    rows: dict[str, list[dict]] = {}
    for split, names in SPLIT_FILES.items():
        rows[split] = []
        for name in names:
            raw = (data_dir / name).read_bytes()
            entry = manifest["files"][name]
            if sha256(raw) != entry["sha256"]:
                raise ValueError(f"edgar_file_hash_mismatch:{name}")
            file_rows = [json.loads(line) for line in raw.decode("utf-8").splitlines()]
            if len(file_rows) != entry["rows"] or any(
                    row["label"] != entry["label"] or not row["prompt"] for row in file_rows):
                raise ValueError(f"edgar_rows_invalid:{name}")
            rows[split].extend(file_rows)
    counts = {split: len(items) for split, items in rows.items()}
    if counts != {"is_memorized": 40, "oos_control": 100, "held_out": 80}:
        raise ValueError(f"unexpected_split_sizes:{counts}")
    return rows


def logical_requests(rows) -> int:
    """1 probe + memorized + 2 x control (baseline and training) + held-out single answers (0 draws)."""
    return 1 + len(rows["is_memorized"]) + 2 * len(rows["oos_control"]) + len(rows["held_out"])


def cutoff_rows(rows, cutoff, min_valid: int):
    """Keep control rows filed strictly after ``cutoff``; memorized rows must predate it.

    Returns ``(rows, counts)``. Nothing is padded: too few calibration controls for
    ``min_valid`` raises ``ValueError('control_below_min_valid:...')``.
    """
    def after(row) -> bool:
        return row["metadata"]["published_at"] > cutoff.isoformat()

    if any(after(row) or row["metadata"]["published_at"] == cutoff.isoformat()
           for split in rows for row in rows[split] if row["label"] == 1):
        raise ValueError("memorized_row_not_before_cutoff")
    kept = {split: [row for row in items if row["label"] == 1 or after(row)] for split, items in rows.items()}
    counts = {"cutoff": cutoff.isoformat(),
              "calibration_control": {"before": len(rows["oos_control"]), "after": len(kept["oos_control"])},
              "held_out_control": {"before": sum(r["label"] == 0 for r in rows["held_out"]),
                                   "after": sum(r["label"] == 0 for r in kept["held_out"])}}
    if len(kept["oos_control"]) < min_valid:
        raise ValueError(f"control_below_min_valid:{len(kept['oos_control'])}<{min_valid}")
    return kept, counts


def local_token_bound(prompt: str) -> int:
    return len(prompt.encode("utf-8")) + TEMPLATE_ALLOWANCE_TOKENS


def modal_token_counter(tokenizer_root: Path):
    """Exact chat-input counts with the pinned GPT-OSS tokenizer and template (local files only)."""
    from deploy.modal.manifest import load_plan as load_modal_plan
    from scripts.probe_modal_model import tokenizer_directory, tokenizer_hashes

    modal_plan = load_modal_plan(MODAL_MANIFEST)
    hashes = tokenizer_hashes(modal_plan, tokenizer_root)
    profile = load_profile(MODAL_PROFILE)
    if hashes is None or hashes["tokenizer.json"] != profile.harmony.tokenizer_sha256:
        raise ValueError("pinned_tokenizer_unavailable_or_changed")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_directory(modal_plan, tokenizer_root)),
                                              local_files_only=True, trust_remote_code=False)

    # vLLM renders reasoning_effort into the Harmony system message, so count with it.
    effort = {key: value for key, value in profile.model_config.parameters.items() if key == "reasoning_effort"}

    def count(prompt: str) -> int:
        return len(tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=True,
                                                 add_generation_prompt=True, **effort))
    return count, modal_plan.target.endpoint, hashes


def build_profile(profile_id: str, rows, output: Path, *, local_endpoint=None, tokenizer_root=Path("models"),
                  npu_endpoint=None, cutoffs=None):
    profile = load_profile(profile_id)
    modal = profile_id == MODAL_PROFILE
    npu = profile_id in NPU_PROFILES
    corpus_filter = None
    if npu:
        rows, corpus_filter = cutoff_rows(rows, cutoffs[CUTOFF_KEYS[profile_id]], PLAN_FIELDS["min_valid"])
        corpus_filter = {"cutoff_key": CUTOFF_KEYS[profile_id], **corpus_filter}
    total = logical_requests(rows)
    shape = MODAL if modal else LOCAL
    if modal:
        count, modal_endpoint, tokenizer = modal_token_counter(tokenizer_root)
        endpoint, credential, mode = modal_endpoint + "/chat/completions", "env:MODAL_PROXY_BEARER", "GPU"
        config = profile.model_config.to_dict()
        counting = {"method": "pinned_tokenizer_chat_template_exact", "tokenizer_sha256": tokenizer,
                    "max_input_margin_tokens": MODAL_INPUT_MARGIN}
    else:
        count, endpoint, credential, mode = local_token_bound, local_endpoint, "env:LOCAL_MODEL_API_KEY", "CPU/GPU offload"
        if npu:
            endpoint, mode = npu_endpoint, "NPU"
        if profile.host_requirements.context_tokens != shape["context"]:
            raise ValueError(f"profile_context_mismatch:{profile_id}")
        config = profile.model_config.to_dict()
        config["parameters"] = {**config["parameters"], **SETTINGS, "max_tokens": shape["output"]}
        counting = {"method": "utf8_bytes_plus_template_allowance_upper_bound",
                    "template_allowance_tokens": TEMPLATE_ALLOWANCE_TOKENS}
    # NPU profiles declare their FLM build (runtime, backend, driver, weights digest); Ollama declares nothing.
    identity = profile.model_config.local_identity if npu else None
    config = config | {"endpoint": endpoint, "local_identity": (identity or LocalModelIdentity()).to_dict()}
    corpus = {split: [{"item_id": sha256(row["metadata"]["url"].encode("utf-8")), "prompt": row["prompt"],
                       "input_tokens": count(row["prompt"]), "label": row["label"]} for row in items]
              for split, items in rows.items()}
    declared = max(row["input_tokens"] for items in corpus.values() for row in items)
    max_input = declared + (MODAL_INPUT_MARGIN if modal else 0)
    if modal and max_input + shape["output"] > shape["context"]:
        raise ValueError(f"context_exceeded:{profile_id}:{max_input}+{shape['output']}>{shape['context']}")
    limits = MeasurementLimits(max_requests=total, context_tokens=shape["context"],
                               max_input_tokens=max_input, max_output_tokens=shape["output"],
                               max_total_tokens=total * (max_input + shape["output"]),
                               max_elapsed_s=shape["max_elapsed_s"])
    manifest = {"config": config, "authentication": "environment", "credential_ref": credential,
                "requested_execution_mode": mode, "profile_id": profile_id, "input_token_counting": counting}
    if corpus_filter is not None:
        manifest["corpus_filter"] = corpus_filter
    directory = output / profile_id
    directory.mkdir(parents=True, exist_ok=True)
    manifest_bytes, corpus_bytes = encode(manifest), encode(corpus)
    plan = {"schema_version": 1, "manifest_sha256": sha256(manifest_bytes), "corpus_sha256": sha256(corpus_bytes),
            **PLAN_FIELDS, "limits": asdict(limits)}
    for name, data in (("manifest.json", manifest_bytes), ("corpus.json", corpus_bytes), ("plan.json", encode(plan))):
        (directory / name).write_bytes(data)
    loaded = load_plan(directory / "plan.json", directory / "manifest.json", directory / "corpus.json")
    if loaded.logical_requests != total or loaded.physical_requests != total or (
            not npu and total != LOGICAL_REQUESTS):
        raise ValueError("phase_a_request_count_mismatch")
    built = {"profile": profile_id, "logical_requests": loaded.logical_requests,
             "max_input_tokens": max_input, "max_total_tokens": limits.max_total_tokens}
    return built if corpus_filter is None else built | {"corpus_filter": corpus_filter}


def write_index(output: Path) -> dict:
    """Hash every present profile file; totals count only complete profiles."""
    files, complete = {}, []
    for profile_id in PROFILES:
        names = ("plan.json", "manifest.json", "corpus.json")
        paths = [output / profile_id / name for name in names]
        if all(path.is_file() for path in paths):
            complete.append(profile_id)
            for path in paths:
                files[f"{profile_id}/{path.name}"] = sha256(path.read_bytes())
    local = [p for p in complete if p in LOCAL_PROFILES]
    # NPU totals follow each profile's own (cutoff-filtered) plan.
    npu = sum(json.loads((output / p / "plan.json").read_text(encoding="utf-8"))["limits"]["max_requests"]
              for p in complete if p in NPU_PROFILES)
    index = {"phase": "A", "files": files, "profiles": complete,
             "missing_profiles": [p for p in PROFILES if p not in complete],
             "logical_requests_per_profile": LOGICAL_REQUESTS, "ensemble_draws": 0,
             "request_totals": {"local": LOGICAL_REQUESTS * len(local),
                                "modal": LOGICAL_REQUESTS * (MODAL_PROFILE in complete), "npu": npu}}
    (output / "index.json").write_bytes(encode(index))
    return index


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=Path("runs/phase-a"))
    parser.add_argument("--data", type=Path, default=Path("data/edgar"))
    parser.add_argument("--profiles", default="all", help="all, local, modal, npu or a comma-separated profile list")
    parser.add_argument("--local-endpoint", help="full gateway chat URL, e.g. http://172.17.32.1:11435/v1/chat/completions")
    parser.add_argument("--npu-endpoint", help="full FLM gateway chat URL, e.g. http://172.17.32.1:11541/v1/chat/completions")
    parser.add_argument("--cutoffs", type=Path, default=Path("data/cutoffs.yaml"))
    parser.add_argument("--tokenizer-root", type=Path, default=Path("models"))
    args = parser.parse_args(argv)
    selected = {"all": PROFILES, "local": LOCAL_PROFILES, "modal": (MODAL_PROFILE,), "npu": NPU_PROFILES}.get(
        args.profiles, tuple(args.profiles.split(",")))
    if any(p not in PROFILES for p in selected):
        parser.error("unknown profile")
    for kind, group, value in (("local", LOCAL_PROFILES, args.local_endpoint), ("npu", NPU_PROFILES, args.npu_endpoint)):
        if any(p in group for p in selected):
            parts = urlsplit(value or "")
            port = GATEWAY_PORTS[kind]
            if parts.scheme != "http" or parts.port != port or parts.path != "/v1/chat/completions":
                parser.error(f"{kind} profiles need --{kind}-endpoint http://<gateway>:{port}/v1/chat/completions")
    rows = read_rows(args.data)
    cutoffs = load_cutoffs(args.cutoffs) if any(p in NPU_PROFILES for p in selected) else None
    built = [build_profile(p, rows, args.output, local_endpoint=args.local_endpoint,
                           tokenizer_root=args.tokenizer_root, npu_endpoint=args.npu_endpoint, cutoffs=cutoffs)
             for p in selected]
    index = write_index(args.output)
    print(json.dumps({"built": built, "index_profiles": index["profiles"],
                      "missing_profiles": index["missing_profiles"], "request_totals": index["request_totals"]},
                     indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
