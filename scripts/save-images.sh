#!/usr/bin/env bash
# Produce the offline image archive that `./scripts/restore-images.sh` restores
# and that `./start.sh` requires to exist, and pin it.
#
# Every other artifact this repository depends on is derived by something in it:
# the corrected checkpoint by `repair-model.sh`, the served chat template by
# `derive-chat-template.py`, the runtime image by `build-vllm.sh`. This is the
# step that produces the archive, the one file a second machine needs in order
# to run this stack at all.
#
# The archive is pinned the way the image is (settle_produced_identity in
# runtime-common.sh): this step is the only witness of the bytes it wrote, so it
# writes IMAGE_ARCHIVE_SHA256 itself, together with IMAGE_ARCHIVE_INPUTS_SHA256,
# the digest of what it saved -- the two images the lock pins and the tags they
# are saved under. Saving the images an archive was pinned for must reproduce
# that archive, because the pinned archive is what restore verifies on every
# host; other bytes are refused rather than adopted. docker save usually
# reproduces its bytes but not always -- of four consecutive saves of the v27
# images on one host, the first differed and the other three matched the pin --
# so that refusal can be the save's doing rather than the images'.
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
    "Build the runtime with ./scripts/build-vllm.sh build, which pins the image" \
    "it builds, before archiving it."
fi

lock="${PROJECT_DIR}/config/runtime-v1.sh"
archive="${PROJECT_DIR}/artifacts/${IMAGE_ARCHIVE_NAME}"
archive_inputs="$(
  printf 'image %s %s\n' \
    "${BASE_IMAGE_TAG}" "${EXPECTED_BASE_IMAGE_ID}" \
    "${IMAGE_TAG}" "${EXPECTED_IMAGE_ID}" | sha256sum
)"
archive_inputs="${archive_inputs%% *}"

# The archive pinned for exactly these images is already here: there is
# nothing to save, and saving again could only reproduce it or be refused.
if [[ "${archive_inputs}" == "${IMAGE_ARCHIVE_INPUTS_SHA256}" && \
      -f "${archive}" && ! -L "${archive}" ]]; then
  printf 'Verifying the archive already at %s...\n' "${archive}"
  present="$(sha256sum -- "${archive}")"
  if [[ "${present%% *}" == "${IMAGE_ARCHIVE_SHA256}" ]]; then
    printf 'The pinned archive of the pinned images is already here; nothing was saved.\n'
    printf 'SHA256: %s\n' "${IMAGE_ARCHIVE_SHA256}"
    exit 0
  fi
fi

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
observed="$(sha256sum -- "${staging}")"
observed="${observed%% *}"

settle_produced_identity pin_outcome "${lock}" \
  IMAGE_ARCHIVE_SHA256 IMAGE_ARCHIVE_INPUTS_SHA256 \
  "${observed}" "${archive_inputs}" \
  "The images the archive pin was saved from saved to different bytes." \
  "Saved: ${BASE_IMAGE_TAG} ${EXPECTED_BASE_IMAGE_ID} and ${IMAGE_TAG} ${EXPECTED_IMAGE_ID}; the new bytes were discarded and ${archive} was not touched." \
  "Next: the pinned archive is the one restore verifies on every host, so keep it rather than replace it. Copy it to ${archive} from a host that holds it; if none does, run this again, since docker save does not always reproduce its bytes."

mv -- "${staging}" "${archive}"
trap - EXIT

printf '\nSAVED %s\n' "${archive}"
printf 'Bytes:  %s\n' "$(stat -c '%s' -- "${archive}")"
printf 'SHA256: %s\n' "${observed}"
if [[ "${pin_outcome}" == written ]]; then
  "${PROJECT_DIR}/scripts/generate-deployment-input-manifest.sh"
  printf 'Wrote the archive pin into config/runtime-v1.sh: IMAGE_ARCHIVE_SHA256=%s, IMAGE_ARCHIVE_INPUTS_SHA256=%s.\n' \
    "${observed}" "${archive_inputs}"
  printf 'Next: commit it together with the image pin the build wrote.\n'
else
  printf 'This is the pinned archive, saved again byte for byte.\n'
fi
