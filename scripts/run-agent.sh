#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=runtime-common.sh
source "${SCRIPT_DIR}/runtime-common.sh"
install_unexpected_error_trap
require_no_arguments "./start.sh" "$@"

printf 'Starting the only supported Qwen3.8 serving profile.\n'
printf 'It will listen on %s:%s and nowhere else.\n\n' \
  "${LISTEN_HOST}" "${LISTEN_PORT}"

check_host_prerequisites
check_pinned_build_inputs
check_model_files

if container_exists; then
  assert_owned_container
  if [[ "$(docker inspect --format '{{.State.Running}}' "${CONTAINER_NAME}")" == "true" ]]; then
    printf '\nThe server is already running. Validating it instead of starting a duplicate...\n'
    assert_running_profile
    print_healthy_summary
    exit 0
  fi
  die "The owned container exists but is stopped or broken." \
    "Run ./stop.sh once, then run ./start.sh again."
fi

assert_port_is_free

if docker volume inspect "${CACHE_VOLUME}" >/dev/null 2>&1; then
  cache_label="$(
    docker volume inspect --format '{{index .Labels "qwen38.project"}}' \
      "${CACHE_VOLUME}"
  )"
  cache_profile_label="$(
    docker volume inspect --format '{{index .Labels "qwen38.runtime.profile"}}' \
      "${CACHE_VOLUME}"
  )"
  if [[ "${cache_label}" != "${CONTAINER_LABEL}" || \
        "${cache_profile_label}" != "${PROFILE_VERSION}" ]]; then
    die "The expected cache-volume name belongs to an unrecognized volume." \
      "Volume: ${CACHE_VOLUME}" \
      "Expected project label: ${CONTAINER_LABEL}" \
      "Found label: ${cache_label:-missing}" \
      "Expected profile label: ${PROFILE_VERSION}" \
      "Found profile label: ${cache_profile_label:-missing}" \
      "The volume was not modified or deleted."
  fi
else
  docker volume create \
    --label "qwen38.project=${CONTAINER_LABEL}" \
    --label "qwen38.runtime.profile=${PROFILE_VERSION}" \
    "${CACHE_VOLUME}" >/dev/null
fi

docker_environment_args=()
for runtime_environment in "${RUNTIME_ENV[@]}"; do
  docker_environment_args+=(--env "${runtime_environment}")
done

cleanup_needed=false
cleanup_failed_start() {
  local status=$?
  trap - EXIT
  if [[ "${cleanup_needed}" == "true" ]]; then
    printf '\nStartup failed. Recent container logs follow:\n' >&2
    docker logs --tail 120 "${CONTAINER_NAME}" >&2 2>/dev/null || true
    printf '\nRemoving only the failed project container...\n' >&2
    docker rm --force "${CONTAINER_NAME}" >/dev/null 2>&1 || true
  fi
  printf 'The model files, image, reviewed patches/template, and cache volume were preserved.\n' >&2
  exit "${status}"
}
trap cleanup_failed_start EXIT

cleanup_needed=true
docker run --detach \
  --name "${CONTAINER_NAME}" \
  --label "qwen38.project=${CONTAINER_LABEL}" \
  --label "qwen38.runtime.profile=${PROFILE_VERSION}" \
  --gpus all \
  --network host \
  --restart no \
  --shm-size 8g \
  --cap-drop ALL \
  --security-opt no-new-privileges:true \
  "${docker_environment_args[@]}" \
  --volume "${MODEL_DIR}:/model:ro" \
  --volume "${CACHE_VOLUME}:/root/.cache/vllm" \
  "${IMAGE_TAG}" \
  "${VLLM_ARGS[@]}" >/dev/null

printf '\nContainer created. Waiting for the pinned model to become healthy'
deadline=$((SECONDS + STARTUP_TIMEOUT_SECONDS))
while true; do
  if [[ "$(docker inspect --format '{{.State.Running}}' "${CONTAINER_NAME}")" != "true" ]]; then
    printf '\nERROR: The container exited during startup.\n' >&2
    false
  fi
  if docker exec "${CONTAINER_NAME}" curl --fail --silent \
      "${ENDPOINT}/health" >/dev/null 2>&1; then
    break
  fi
  if ((SECONDS >= deadline)); then
    printf '\nERROR: Startup exceeded %s seconds.\n' \
      "${STARTUP_TIMEOUT_SECONDS}" >&2
    false
  fi
  printf '.'
  sleep 2
done
printf ' ready.\n'

assert_running_profile
cleanup_needed=false
trap - EXIT
print_healthy_summary
