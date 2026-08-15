#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=runtime-common.sh
source "${SCRIPT_DIR}/runtime-common.sh"
install_unexpected_error_trap
require_no_arguments "./stop.sh" "$@"

if ! container_exists; then
  existing_listener="$(listener_output)"
  if [[ -n "${existing_listener}" ]]; then
    die "The project container is absent, but TCP port ${LISTEN_PORT} is still in use." \
      "Listener(s):" "${existing_listener}" \
      "It is not owned by this project and was not killed."
  fi
  printf 'ALREADY STOPPED — nothing needed to be changed.\n'
  printf 'To start the one supported profile: ./start.sh\n'
  exit 0
fi

assert_owned_container
container_image_id="$(docker inspect --format '{{.Image}}' "${CONTAINER_NAME}")"
if [[ "${container_image_id}" != "${EXPECTED_IMAGE_ID}" ]]; then
  die "The owned container has an unexpected image and was not removed." \
    "Expected: ${EXPECTED_IMAGE_ID}" \
    "Found:    ${container_image_id}"
fi

if [[ "$(docker inspect --format '{{.State.Running}}' "${CONTAINER_NAME}")" == "true" ]]; then
  printf 'Stopping the exact project container...\n'
  docker stop --timeout 30 "${CONTAINER_NAME}" >/dev/null
fi
printf 'Removing the stopped container...\n'
docker rm "${CONTAINER_NAME}" >/dev/null

if container_exists; then
  die "Docker still reports the project container after removal."
fi

remaining_listener="$(listener_output)"
if [[ -n "${remaining_listener}" ]]; then
  die "The project container stopped, but another process still uses TCP port ${LISTEN_PORT}." \
    "Listener(s):" "${remaining_listener}" \
    "The unrelated process was not killed."
fi

printf '\nSTOPPED — no Qwen service is listening on TCP port %s.\n' "${LISTEN_PORT}"
printf 'Preserved: model weights, pinned images, reviewed patch, and cache volume.\n'
printf 'To start it again: ./start.sh\n'
