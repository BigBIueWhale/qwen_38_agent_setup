#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if (($# != 0)); then
  printf 'Usage: %s\n' "$0" >&2
  exit 2
fi
# The image contract as proved: fifteen distinct full-pixel images transcribed
# in one request, then a sixteenth refused before inference.
"${SCRIPT_DIR}/run-probe.sh" vision_quality_probe.py --images 15
exec "${SCRIPT_DIR}/run-probe.sh" vision_quality_probe.py --expect-count-rejection
