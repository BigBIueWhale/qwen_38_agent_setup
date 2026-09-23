#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=runtime-common.sh
source "${SCRIPT_DIR}/runtime-common.sh"
install_unexpected_error_trap
require_no_arguments "./scripts/restore-images.sh" "$@"
require_command docker
require_command sha256sum

archive="${PROJECT_DIR}/artifacts/${IMAGE_ARCHIVE_NAME}"
if [[ ! -f "${archive}" ]]; then
  die "The pinned offline image archive is missing." \
    "Expected: ${archive}"
fi
printf '%s  %s\n' "${IMAGE_ARCHIVE_SHA256}" "${archive}" | \
  sha256sum --check --strict

printf 'Loading the exact base and runtime images from the verified local archive...\n'
docker load --input "${archive}"

base_id="$(docker image inspect --format '{{.Id}}' "${BASE_IMAGE_TAG}")"
runtime_id="$(docker image inspect --format '{{.Id}}' "${IMAGE_TAG}")"
if [[ "${base_id}" != "${EXPECTED_BASE_IMAGE_ID}" || \
      "${runtime_id}" != "${EXPECTED_IMAGE_ID}" ]]; then
  die "Docker loaded image IDs that do not match the version lock." \
    "Expected base:    ${EXPECTED_BASE_IMAGE_ID}" \
    "Loaded base:      ${base_id}" \
    "Expected runtime: ${EXPECTED_IMAGE_ID}" \
    "Loaded runtime:   ${runtime_id}"
fi

# The loaded runtime image carries only the version tag it was saved under,
# which a later re-pin of this version moves to another image; it takes its
# identity tag here, as it does where it was built, so it keeps a name.
identity_tag_id="$(docker image inspect --format '{{.Id}}' "${IMAGE_IDENTITY_TAG}" 2>/dev/null || true)"
if [[ -n "${identity_tag_id}" && "${identity_tag_id}" != "${EXPECTED_IMAGE_ID}" ]]; then
  die "The identity tag ${IMAGE_IDENTITY_TAG} already names ${identity_tag_id}; it was not moved."
fi
docker tag "${EXPECTED_IMAGE_ID}" "${IMAGE_IDENTITY_TAG}"

printf '\nRESTORED — exact pinned images are available without a network pull.\n'
printf 'Base:     %s\n' "${base_id}"
printf 'Runtime:  %s\n' "${runtime_id}"
printf 'Identity: %s\n' "${IMAGE_IDENTITY_TAG}"
