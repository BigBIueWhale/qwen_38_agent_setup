#!/usr/bin/env bash
set -euo pipefail

if (( $# != 0 )); then
  printf 'Usage: %s\n' "$0" >&2
  exit 2
fi

docker exec -i qwen38-agent-native python3 - < "$(dirname -- "$0")/responses_protocol_probe.py"
