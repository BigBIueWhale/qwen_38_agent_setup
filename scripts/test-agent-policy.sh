#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=../config/runtime-v1.sh
source "${PROJECT_DIR}/config/runtime-v1.sh"

if (($# != 0)); then
  printf 'Usage: %s\n' "$0" >&2
  exit 2
fi

docker exec -i "${CONTAINER_NAME}" python3 - < "${SCRIPT_DIR}/agent_policy_probe.py"
