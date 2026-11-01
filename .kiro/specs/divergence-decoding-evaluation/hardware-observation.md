# Local hardware observation

Date: 2026-11-01. Read-only observation for the later compute proposal; this
does not complete task 6.1 or authorize provisioning, downloads or inference.

Windows CIM reported:

| Field | Observed value |
| --- | --- |
| Video controller name | AMD Radeon(TM) 890M Graphics |
| AdapterRAM | 2147483648 bytes |
| TotalPhysicalMemory | 31802109952 bytes, approximately 29.6 GiB |

Commands: `Get-CimInstance Win32_VideoController` and
`Get-CimInstance Win32_ComputerSystem`, selecting only the fields above.
No NVIDIA controller was listed. AdapterRAM from this query does not measure
available inference VRAM, especially for integrated graphics. Neither host RAM
nor that field establishes Docker GPU access or model feasibility.

The optional Torch/CUDA image can be validated for package imports without a
GPU. Its real-model smoke still needs a separately identified supported compute
environment, measured memory headroom and an approved resource plan. Do not
silently dequantize GPT-OSS or replace its backend based on this observation.
A different local backend would require an explicit design update and its own
parity/format validation. Remote GPU allocation has not been selected or approved.
