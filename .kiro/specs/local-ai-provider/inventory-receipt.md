# Local candidate inventory receipt — 2026-11-05

Task 1.1 inventory is complete. The NPU branch is blocked at prerequisites;
all three configurations remain unverified. No inference, model-weight download,
installation, driver change or hardware configuration change occurred.

## Task brief and result

Requirements 1.1, 1.2, 1.4, 3.5, 3.7, 6.1 and 6.3 require explicit independent
Gemma/GPT-OSS candidates, known and unknown provenance, separated memory categories,
and an optional host with honest failed prerequisites. The adjacent
[candidate manifest](candidate-manifest.json) freezes three configurations and a
maximum six generation attempts, including any later server checks. This inventory
uses zero attempts. No unused NPU allowance transfers to another configuration.
Design sections are named, not numbered: Technology Stack; Local evidence and
provenance; Local endpoint admission; Gemma NPU feasibility gate; Data Models and
System Flow; Error Handling; Testing Strategy.

## Read-only machine observations

| Field | Observation |
| --- | --- |
| CPU | AMD Ryzen AI 9 HX 370 w/ Radeon 890M |
| OS | Windows 11 Pro, build 26200 |
| OS-visible total physical RAM | 31,802,109,952 bytes |
| Free physical RAM snapshot | 16,530,744 KiB; transient, not a run allocation |
| Integrated GPU | AMD Radeon 890M, driver 32.0.31041.1004 |
| Reported GPU AdapterRAM | 2,147,483,648 bytes from Win32_VideoController |
| Shared GPU memory limit/usage | Unknown; not measured |
| Dedicated GPU memory usage | Unknown; AdapterRAM is not an allocation measurement |
| NPU | NPU Compute Accelerator Device, ComputeAccelerator class, status OK |
| NPU signed driver | 32.0.20102.3930 |
| Ryzen AI default installation directory | C:/Program Files/RyzenAI absent; other locations not exhaustively searched |
| Ollama binary | C:/Users/mariu/AppData/Local/AMD/AI_Bundle/Ollama/ollama.exe |
| Ollama client | 0.40.11 from --version; server version unknown |
| Ollama process | None observed |
| localhost:11434 | Connection refused for metadata requests; unavailable endpoint, not model incompatibility |
| Completed local model manifest | llava-llama3:latest only; SHA256 44c161b1f46523301da9c0cc505afa4a4a0cc62f580581d98a430bb21acd46de |
| Partial download | GPT-OSS template digest fa6710a93d78da62641e192361344be7a8c0a1c3737f139cf89f20ce1626b99c partial files observed; no pull process observed; remaining download size unknown |

The 2 GiB GPU field is not a universal model-size limit. Shared system memory and
CPU offload may be available, but allocation, throughput and exhaustion remain
unmeasured. No load time, generation latency or score capability was established.
Partial files do not establish a completed model; check existing state before any
later pull. This task did not repeat the earlier pull.

## Candidate selection and prerequisite gates

Ollama Gemma 3 4B Q4_K_M and GPT-OSS-20B MXFP4 are pinned by the full registry
manifest hashes and individual layer hashes in JSON. Total uncompressed registry
blob transfer sizes are respectively 3,350,272,403 and 13,793,441,566 bytes
(excludes protocol overhead and local runtime storage). Combined fresh blob
transfer is 17,143,713,969 bytes; reuse and remaining bytes are unknown. These
sizes do not estimate RAM or peak disk requirements. Model pages identify Gemma
Terms of Use and Apache-2.0 respectively. Template digests are known; separately
identified tokenizer revisions remain unknown until local GGUF inspection.

AMD's release-linked candidate is
`amd/gemma-3-4b-npu-basic-text-logit-pruning_rai_1.9.0_npu`, paired by name with
Ryzen AI 1.9.0. The public linked page returned HTTP 401, so its commit, artifact
size, quantization, tokenizer and artifact-specific licensing remain unverified.
The runtime ZIP HEAD returned Content-Length 6,615,496,792 bytes; expanded size
and checksum are unknown. No archive was downloaded.

The AMD documentation has conflicting OGA versions (0.18.0 change notes and
0.14.0 API paragraph). Its documented driver 32.0.201.448 differs from the
installed version. Numeric ordering alone cannot prove compatibility. Therefore
this is a release-selected candidate, not an executable installation lock.
The current driver exists; compatible driver support is unverified, not absent.
Resolve release compatibility, artifact access, exact commit, footprint and
license before setup. Driver installation is outside authorization. Task 1.2 can
retain this failed-prerequisite assessment and disable the NPU branch.

The artifact name explicitly signals logit pruning. Neither its name nor the OGA
API proves genuine full-vocabulary scores. A future admitted NPU probe must prove
text-only operation, same-prefix aligned probabilities and observed placement;
pruned-only scores cannot be normalized into a full distribution. No score or
NPU support claim is made here.

## Installation plan, not executed

Ollama is already installed. Once downloads are approved against the concrete
manifest, PowerShell commands are:

```powershell
$ollamaExe = 'C:/Users/mariu/AppData/Local/AMD/AI_Bundle/Ollama/ollama.exe'
Start-Process -FilePath $ollamaExe -ArgumentList 'serve' -WindowStyle Hidden
# Recheck any existing pull before admitting one serialized download.
& $ollamaExe pull gemma3:4b
& $ollamaExe pull gpt-oss:20b
# Verify installed manifest digests against candidate-manifest.json before inference.
```

Do not execute an NPU installation command while prerequisite gates remain open.
After compatibility and licensing review and a concrete verified artifact/runtime download plan,
the selected runtime archive can be acquired and expanded through PowerShell:

```powershell
Invoke-WebRequest 'https://download.amd.com/opendownload/RyzenAI/1.9.0/ryzen-ai-1.9.0-msi.zip' -OutFile './ryzen-ai-1.9.0-msi.zip'
Expand-Archive -LiteralPath './ryzen-ai-1.9.0-msi.zip' -DestinationPath './ryzen-ai-1.9.0-installer'
# Inspect archive contents and vendor instructions before selecting its installer.
# The internal installer filename/options and artifact commit are unverified;
# no guessed executable command or moving-revision model download is supplied.
```

AMD directs users to its installer wizard and bundled environment. No independent
PyPI OGA version is selected to paper over the version conflict. Keep the optional
vendor runtime outside the project environment and CI image. Installation remains
blocked rather than presenting unverified commands as reproducible setup.

## Evidence and verification

PowerShell read-only commands used: Get-CimInstance Win32_Processor,
Win32_ComputerSystem, Win32_OperatingSystem, Win32_VideoController and
Win32_PnPSignedDriver; Get-PnpDevice; Get-Process ollama; Get-ChildItem on the
local Ollama manifests/partial blobs; Get-FileHash; Ollama --version;
Invoke-RestMethod against /api/version, /api/tags and /api/ps; Invoke-WebRequest
registry manifests and AMD archive HEAD. No generation endpoint was called.
Registry byte content was decoded as UTF-8 before JSON parsing and SHA256 hashing.

Primary references accessed on 2026-11-05:

- [Ollama Gemma candidate](https://ollama.com/library/gemma3:4b)
- [Ollama GPT-OSS candidate](https://ollama.com/library/gpt-oss:20b)
- [Gemma registry manifest](https://registry.ollama.ai/v2/library/gemma3/manifests/4b)
- [GPT-OSS registry manifest](https://registry.ollama.ai/v2/library/gpt-oss/manifests/20b)
- [AMD OGA deployment](https://ryzenai.docs.amd.com/en/main/hybrid_oga.html)
- [AMD installation prerequisites](https://ryzenai.docs.amd.com/en/main/inst.html)
- [Release-linked AMD Gemma artifact, access failed](https://huggingface.co/amd/gemma-3-4b-npu-basic-text-logit-pruning_rai_1.9.0_npu)

Nonbehavioral evidence artifacts require JSON/schema-content checks and
`git diff --check`; application tests and artificial RED/GREEN tests are not
applicable. No package dependencies or application behavior changed.
