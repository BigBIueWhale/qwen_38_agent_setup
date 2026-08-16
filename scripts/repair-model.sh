#!/usr/bin/env bash
set -Eeuo pipefail
IFS=$'\n\t'

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)"

# shellcheck source=config/runtime-v1.sh
source "${PROJECT_DIR}/config/runtime-v1.sh"

actual_base_image_id="$(docker image inspect --format '{{.Id}}' "${BASE_IMAGE_TAG}" 2>/dev/null || true)"
if [[ "${actual_base_image_id}" != "${EXPECTED_BASE_IMAGE_ID}" ]]; then
  printf '%s\n' \
    "ERROR: pinned model-repair image is missing or has the wrong identity." \
    "Expected tag: ${BASE_IMAGE_TAG}" \
    "Expected ID:  ${EXPECTED_BASE_IMAGE_ID}" \
    "Found ID:     ${actual_base_image_id:-missing}" >&2
  exit 1
fi

exec docker run --rm \
  --network none \
  --read-only \
  --tmpfs /tmp:rw,noexec,nosuid,size=256m \
  --volume "${PROJECT_DIR}:/project:rw" \
  --entrypoint /usr/bin/python3 \
  "${BASE_IMAGE_TAG}" \
  /project/scripts/repair-offset-norms.py
