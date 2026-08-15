#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
docker exec -i qwen38-agent-native python3 - \
  < "${SCRIPT_DIR}/agent_history_roundtrip_probe.py"
