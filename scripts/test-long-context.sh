#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SALT="${1:-manual-cold-probe}"
docker exec -i qwen38-agent-native python3 - \
  --salt "${SALT}" \
  --targets 32768 131072 261120 < "${SCRIPT_DIR}/long_context_probe.py"
