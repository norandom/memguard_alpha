#!/bin/sh
# Write a timestamped free-RAM snapshot of THIS Linux or macOS machine (the Ollama
# model host); same JSON as host_memory_snapshot.ps1. Host-side tooling: it reads
# only local memory figures, needs no root and makes no network calls.
# Usage: sh scripts/host_memory_snapshot.sh [PATH]  (default runs/container/host-memory.json,
# which the container sees as /outputs/host-memory.json). Refused after 10 minutes.
set -eu
out="${1:-$(dirname "$0")/../runs/container/host-memory.json}"
if [ -r /proc/meminfo ]; then
  available=$(awk '/^MemAvailable:/ {printf "%.0f", $2 * 1024}' /proc/meminfo)
  total=$(awk '/^MemTotal:/ {printf "%.0f", $2 * 1024}' /proc/meminfo)
  source=linux_host_snapshot
else
  # macOS: available = free + inactive pages (the psutil definition).
  page=$(sysctl -n hw.pagesize)
  available=$(vm_stat | awk -v p="$page" '/Pages free|Pages inactive/ {gsub("\\.", "", $NF); n += $NF} END {printf "%.0f", n * p}')
  total=$(sysctl -n hw.memsize)
  source=macos_host_snapshot
fi
mkdir -p "$(dirname "$out")"
printf '{"available_bytes": %s, "total_bytes": %s, "observed_at": "%s", "host": "%s", "source": "%s"}\n' \
  "$available" "$total" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$(hostname | tr -d '"\\')" "$source" > "$out"
echo "Wrote $out"
