#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "${SCRIPT_DIR}/run-probe.sh" long_context_probe.py \
  --salt "${1:-manual-cold-probe}" \
  --targets 32768 131072 261120
