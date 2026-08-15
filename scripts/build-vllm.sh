#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-build}"
EXPECTED_STATUS=" M vllm/v1/attention/backends/turboquant_attn.py"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=../config/runtime-v1.sh
source "${PROJECT_DIR}/config/runtime-v1.sh"
VLLM_DIR="${PROJECT_DIR}/vllm"
PATCH_FILE="${PROJECT_DIR}/patches/vllm-turboquant-k8v4-direct-workspace.patch"
PATCHED_FILE="${VLLM_DIR}/vllm/v1/attention/backends/turboquant_attn.py"
DOCKERFILE="${PROJECT_DIR}/containers/Dockerfile.runtime"
DOCKERIGNORE="${PROJECT_DIR}/.dockerignore"

case "${MODE}" in
  build|check)
    ;;
  *)
    echo "Usage: $0 [build|check]" >&2
    exit 2
    ;;
esac

actual_commit="$(git -C "${VLLM_DIR}" rev-parse HEAD)"
if [[ "${actual_commit}" != "${VLLM_COMMIT}" ]]; then
  echo "Refusing unexpected vLLM commit: ${actual_commit}" >&2
  exit 1
fi

actual_base_image_id="$(docker image inspect --format '{{.Id}}' "${BASE_IMAGE_TAG}" 2>/dev/null || true)"
if [[ "${actual_base_image_id}" != "${EXPECTED_BASE_IMAGE_ID}" ]]; then
  echo "Required immutable base image is missing or incorrect." >&2
  echo "Expected: ${EXPECTED_BASE_IMAGE_ID}" >&2
  echo "Found:    ${actual_base_image_id:-nothing}" >&2
  echo "No mutable tag or network pull will be substituted." >&2
  echo "Restore the pinned bytes with: ./scripts/restore-images.sh" >&2
  exit 1
fi

actual_status="$(git -C "${VLLM_DIR}" status --short --untracked-files=all)"
if [[ "${actual_status}" != "${EXPECTED_STATUS}" ]]; then
  echo "Refusing unexpected vLLM worktree state:" >&2
  printf '%s\n' "${actual_status}" >&2
  exit 1
fi

printf '%s  %s\n' "${PATCH_DIFF_SHA256}" "${PATCH_FILE}" | \
  sha256sum --check --strict

read -r actual_patch_sha256 _ < <(
  git -C "${VLLM_DIR}" diff -- \
    vllm/v1/attention/backends/turboquant_attn.py | sha256sum
)
if [[ "${actual_patch_sha256}" != "${PATCH_DIFF_SHA256}" ]]; then
  echo "Refusing unreviewed TurboQuant diff: ${actual_patch_sha256}" >&2
  exit 1
fi

printf '%s  %s\n' "${PATCHED_FILE_SHA256}" "${PATCHED_FILE}" | \
  sha256sum --check --strict
printf '%s  %s\n%s  %s\n' \
  "${RUNTIME_DOCKERFILE_SHA256}" "${DOCKERFILE}" \
  "${DOCKERIGNORE_SHA256}" "${DOCKERIGNORE}" | \
  sha256sum --check --strict

git -C "${VLLM_DIR}" diff --check

if [[ "${MODE}" == "check" ]]; then
  echo "Pinned base image, vLLM commit, and reviewed TurboQuant patch are exact."
  exit 0
fi

DOCKER_BUILDKIT=1 docker build --progress=plain \
  --network none \
  --pull=false \
  --provenance=false \
  --target runtime \
  --build-arg "BASE_IMAGE=${BASE_IMAGE_TAG}" \
  --build-arg "UPSTREAM_FILE_SHA256=${UPSTREAM_FILE_SHA256}" \
  --build-arg "PATCHED_FILE_SHA256=${PATCHED_FILE_SHA256}" \
  --build-arg "SOURCE_DATE_EPOCH=${SOURCE_DATE_EPOCH}" \
  --tag "${IMAGE_TAG}" \
  --file "${DOCKERFILE}" \
  "${PROJECT_DIR}"

actual_image_id="$(docker image inspect --format '{{.Id}}' "${IMAGE_TAG}")"
actual_installed_sha256="$(
  docker run --rm --network none --entrypoint sha256sum "${IMAGE_TAG}" \
    /usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/turboquant_attn.py \
    | { read -r file_sha256 _; printf '%s' "${file_sha256}"; }
)"
if [[ "${actual_installed_sha256}" != "${PATCHED_FILE_SHA256}" ]]; then
  echo "Built image contains the wrong patched file: ${actual_installed_sha256}" >&2
  exit 1
fi
if [[ "${actual_image_id}" != "${EXPECTED_IMAGE_ID}" ]]; then
  echo "Reproducible build ID mismatch." >&2
  echo "Expected: ${EXPECTED_IMAGE_ID}" >&2
  echo "Found:    ${actual_image_id}" >&2
  exit 1
fi

echo "Built ${IMAGE_TAG} with no build-time network access."
echo "Verified reproducible image ID: ${actual_image_id}"
