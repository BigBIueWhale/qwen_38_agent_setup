#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
docker exec -i qwen38-agent-native python3 - "$@" \
  < "${SCRIPT_DIR}/native_context_boundary_probe.py"
