# Optional experiment environment — task 1.2

The `experiment` dependency group is checkout tooling, outside library runtime
dependencies and consumer extras. The canonical Dockerfile exposes an explicit
`experiment` target; the default final target remains the lean `runtime` image.
The experiment group supports Python 3.12–3.13; this does not narrow the library's
Python requirement. This delivery uses Linux x86-64 and Python 3.12.12.

## Package selection

| Package | Pin | Selection evidence |
| --- | --- | --- |
| torch | 2.8.0 | Published stable release with Triton 3.4.0 dependency |
| transformers | 4.55.4 | Includes GPT-OSS causal-LM forward/cache interface |
| triton | 3.4.0 | GPT-OSS MXFP4 backend's minimum Triton release |
| kernels | 0.9.0 | Pinned kernel-loader package; remote kernel artifacts remain separate |
| accelerate | 1.10.1 | Pinned device-placement/loading dependency |

Primary sources checked on 2026-11-01:

- [OpenAI Transformers GPT-OSS recipe](https://developers.openai.com/cookbook/articles/gpt-oss/run-transformers)
  identifies Transformers, Torch, Accelerate, Triton 3.4 and kernels.
- [Transformers 4.55.4 GPT-OSS API](https://huggingface.co/docs/transformers/v4.55.4/en/model_doc/gpt_oss)
  documents the causal-LM forward pass.
- [Versioned MXFP4 quantizer source](https://github.com/huggingface/transformers/blob/v4.55.4/src/transformers/quantizers/quantizer_mxfp4.py)
  checks Triton >=3.4.0 and kernels and can silently dequantize on an unsupported
  environment. Task 3.1 must reject unsupported precision/hardware before loading.
- [PyTorch previous releases](https://pytorch.org/get-started/previous-versions/)
  provides the Torch 2.8.0 release installation matrix.

The complete transitive versions and wheel hashes are in `uv.lock`. Installing
`kernels` does not pin or acquire a Hub kernel repository. Actual kernel artifact,
model/tokenizer revisions, accelerator, driver and precision acceptance remain
runtime prerequisites. An import receipt does not establish inference feasibility.

## Reproduction

From the repository root:

```text
docker build --target experiment --tag recall-guard:experiment .
docker image inspect recall-guard:experiment --format '{{.Id}}'
docker run --rm --network none recall-guard:experiment
```

The default experiment command runs `uv run --no-sync python -m
experiments.divergence_evaluation.environment`. It prints the lock SHA-256,
Python/platform identity and installed dependency versions without importing
Torch or Transformers. Save that JSON with the Docker content ID (or registry
digest after publishing) in the execution plan. A mutable tag alone is insufficient.
The image defaults to offline Hugging Face access. A separately approved local
artifact acquisition process is required before any runtime loading.

## Verification receipts

- RED: Docker `pytest tests/test_packaging.py -k experiment -q` failed with
  `KeyError: 'experiment'` before the dependency group existed.
- GREEN: Docker `uv run --no-sync pytest tests/test_packaging.py
  tests/experiments -q --override-ini cache_dir=/tmp/pytest-cache`: 31 passed.
- Docker `uv run --no-sync ruff check --no-cache tests/test_packaging.py
  experiments/divergence_evaluation/environment.py`: passed.
- Docker `uv lock --check --offline`: 216 packages resolved; lock current.
- Canonical runtime image rebuilt and installed its wheel successfully.
  Build configuration digest: `sha256:ceb02398f17e16448533b06e6e8822dd00d35845ae9c6a7c7bde0ab16b1554c8`.
  With network disabled and working directory `/outputs`, `uv run --no-sync
  python` imported `/opt/venv/lib/python3.12/site-packages/recall_guard/__init__.py`.
  `find_spec` confirmed `experiments`, Torch, Transformers, Accelerate, kernels
  and Triton absent. Wheel distribution files contained no `experiments/` paths;
  importing Recall Guard did not import matplotlib.
- Credential-free inventory execution passed in the existing lean development
  image with network disabled. Lock SHA-256:
  `ac84c29f8ffbafddd5c766db5289cb3f5e6e3cb7493b08f22775a9e6d17a31a0`.

- Optional build session `98029` completed successfully after downloading only
  Python/CUDA package wheels. Final source refresh build `20474` reused the
  dependency layer and exited zero.
- Final `docker image inspect recall-guard:experiment --format '{{.Id}}'`:
  `sha256:acdf5deaffe08298a0ada7d99918eaae181140029deff35433ee0ee7e3b28c47`.
  Build configuration digest:
  `sha256:275130324f3de11f961cc3eac3a59b7ffce958bed541637c10f26517dd2fc94f`.
- Final image, `--network none`, `/app`, `uv run --no-sync python`: imported
  Torch 2.8.0, Transformers 4.55.4, Accelerate 1.10.1, kernels 0.9.0, Triton
  3.4.0, `GptOssForCausalLM` and experimental contracts; exited zero.
- Final image, network disabled: inventory execution exited zero and asserted
  that none of Torch, Transformers, Accelerate, kernels or Triton was imported.
  The inventory recorded Python 3.12.12, the lock hash above, and all 53 installed
  distribution versions. Version receipts use `importlib.metadata.version`:
  kernels does not expose a module `__version__` attribute.

No model weights, tokenizer artifacts or remote kernels were requested, and no
model was instantiated or run. A transient read-only Docker status attempt
returned `Wsl/Service/WSAETIMEDOUT` during the initial build; the preserved build
subsequently completed without package/index or host configuration changes.
Actual device memory, precision and kernel feasibility remain unvalidated.
