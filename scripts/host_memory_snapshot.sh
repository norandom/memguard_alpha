#!/bin/sh
# Write a timestamped free-RAM snapshot of THIS Linux or macOS machine (the Ollama
# model host); same JSON as host_memory_snapshot.ps1. Host-side tooling: it reads
# only local memory figures, needs no root and makes no network calls. Under WSL2 it
# records the WSL VM (host "NAME WSL VM", source wsl_vm_snapshot), which is the model
# host when Ollama runs in WSL Docker (compose profile "ollama").
# Usage: sh scripts/host_memory_snapshot.sh [PATH]  (default runs/container/host-memory.json,
# which the container sees as /outputs/host-memory.json). Refused after 10 minutes.
set -eu
out="${1:-$(dirname "$0")/../runs/container/host-memory.json}"
if [ -r /proc/meminfo ]; then
  available=$(awk '/^MemAvailable:/ {printf "%.0f", $2 * 1024}' /proc/meminfo)
  total=$(awk '/^MemTotal:/ {printf "%.0f", $2 * 1024}' /proc/meminfo)
  source=linux_host_snapshot
  host=$(hostname)
  # WSL2: /proc/meminfo is the shared WSL VM, the model host for Ollama in WSL Docker.
  if [ -n "${WSL_DISTRO_NAME:-}" ] || grep -qi microsoft /proc/sys/kernel/osrelease 2>/dev/null; then
    source=wsl_vm_snapshot
    host="$host WSL VM"
  fi
else
  # macOS: available = free + inactive pages (the psutil definition).
  page=$(sysctl -n hw.pagesize)
  available=$(vm_stat | awk -v p="$page" '/Pages free|Pages inactive/ {gsub("\\.", "", $NF); n += $NF} END {printf "%.0f", n * p}')
  total=$(sysctl -n hw.memsize)
  source=macos_host_snapshot
  host=$(hostname)
fi
mkdir -p "$(dirname "$out")"
printf '{"available_bytes": %s, "total_bytes": %s, "observed_at": "%s", "host": "%s", "source": "%s"}\n' \
  "$available" "$total" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$(printf %s "$host" | tr -d '"\\')" "$source" > "$out"
echo "Wrote $out"
