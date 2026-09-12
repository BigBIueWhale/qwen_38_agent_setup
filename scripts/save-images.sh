#!/usr/bin/env bash
# Produce the offline image archive that `./scripts/restore-images.sh` restores
# and that `./start.sh` requires to exist.
#
# Every other artifact this repository depends on is derived by something in it:
# the corrected checkpoint by `repair-model.sh`, the served chat template by
# `derive-chat-template.py`, the runtime image by `build-vllm.sh`. The archive
# was the exception -- it was produced by hand, which meant nobody but its author
# could recreate the one file a second machine needs in order to run this stack
# at all. This is that step, written down.
#
# The archive is an artifact of this act rather than a reproducible derivation:
# `docker save` is not byte-stable, so its SHA-256 is adopted from the file this
# produces and verified against that pin by every later consumer. What IS
# reproducible is what goes into it -- both images are refused unless they carry
# the exact IDs the version lock names.
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=runtime-common.sh
source "${SCRIPT_DIR}/runtime-common.sh"
install_unexpected_error_trap
require_no_arguments "./scripts/save-images.sh" "$@"
require_command docker
require_command sha256sum

base_id="$(docker image inspect --format '{{.Id}}' "${BASE_IMAGE_TAG}" 2>/dev/null || true)"
runtime_id="$(docker image inspect --format '{{.Id}}' "${IMAGE_TAG}" 2>/dev/null || true)"
if [[ "${base_id}" != "${EXPECTED_BASE_IMAGE_ID}" || \
      "${runtime_id}" != "${EXPECTED_IMAGE_ID}" ]]; then
  die "Refusing to archive images that are not the ones this version lock names." \
    "Expected base:    ${EXPECTED_BASE_IMAGE_ID}" \
    "Found base:       ${base_id:-missing}" \
    "Expected runtime: ${EXPECTED_IMAGE_ID}" \
    "Found runtime:    ${runtime_id:-missing}" \
    "Build the runtime with ./scripts/build-vllm.sh build and adopt the ID it" \
    "reports before archiving it."
fi

archive="${PROJECT_DIR}/artifacts/${IMAGE_ARCHIVE_NAME}"
staging="${archive}.saving"
install -d -m 0700 "${PROJECT_DIR}/artifacts"
rm -f -- "${staging}"
cleanup_staging() {
  rm -f -- "${staging}"
}
trap cleanup_staging EXIT

printf 'Archiving the pinned base and runtime images...\n'
docker save --output "${staging}" "${BASE_IMAGE_TAG}" "${IMAGE_TAG}"
chmod 0600 -- "${staging}"
sync -f -- "${staging}"
mv -- "${staging}" "${archive}"
trap - EXIT

observed="$(sha256sum -- "${archive}" | cut -d' ' -f1)"
printf '\nSAVED %s\n' "${archive}"
printf 'Bytes:  %s\n' "$(stat -c '%s' -- "${archive}")"
printf 'SHA256: %s\n' "${observed}"
if [[ "${observed}" == "${IMAGE_ARCHIVE_SHA256}" ]]; then
  printf 'This matches IMAGE_ARCHIVE_SHA256; nothing to adopt.\n'
  exit 0
fi
printf '\nIMAGE_ARCHIVE_SHA256 in config/runtime-v1.sh still reads:\n  %s\n' \
  "${IMAGE_ARCHIVE_SHA256}"
printf 'Adopt the hash above, commit it, and the restore path will accept this archive.\n'
