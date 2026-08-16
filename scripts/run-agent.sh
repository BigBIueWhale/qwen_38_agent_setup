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
  cache_model_revision_label="$(
    docker volume inspect --format '{{index .Labels "qwen38.model.revision"}}' \
      "${CACHE_VOLUME}"
  )"
  cache_model_correction_label="$(
    docker volume inspect --format '{{index .Labels "qwen38.model.correction"}}' \
      "${CACHE_VOLUME}"
  )"
  cache_model_sha256_label="$(
    docker volume inspect --format '{{index .Labels "qwen38.model.sha256"}}' \
      "${CACHE_VOLUME}"
  )"
  if [[ "${cache_label}" != "${CONTAINER_LABEL}" || \
        "${cache_profile_label}" != "${PROFILE_VERSION}" || \
        "${cache_model_revision_label}" != "${MODEL_REVISION}" || \
        "${cache_model_correction_label}" != "${MODEL_CORRECTION}" || \
        "${cache_model_sha256_label}" != "${MODEL_SHA256}" ]]; then
    die "The expected cache-volume name belongs to an unrecognized volume." \
      "Volume: ${CACHE_VOLUME}" \
      "Expected project label: ${CONTAINER_LABEL}" \
      "Found label: ${cache_label:-missing}" \
      "Expected profile label: ${PROFILE_VERSION}" \
      "Found profile label: ${cache_profile_label:-missing}" \
      "Expected model revision: ${MODEL_REVISION}" \
      "Found model revision: ${cache_model_revision_label:-missing}" \
      "Expected model correction: ${MODEL_CORRECTION}" \
      "Found model correction: ${cache_model_correction_label:-missing}" \
      "Expected model SHA-256: ${MODEL_SHA256}" \
      "Found model SHA-256: ${cache_model_sha256_label:-missing}" \
      "The volume was not modified or deleted."
  fi
else
  docker volume create \
    --label "qwen38.project=${CONTAINER_LABEL}" \
    --label "qwen38.runtime.profile=${PROFILE_VERSION}" \
    --label "qwen38.model.revision=${MODEL_REVISION}" \
    --label "qwen38.model.correction=${MODEL_CORRECTION}" \
    --label "qwen38.model.sha256=${MODEL_SHA256}" \
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
  --label "qwen38.model.revision=${MODEL_REVISION}" \
  --label "qwen38.model.official-revision=${OFFICIAL_MODEL_REVISION}" \
  --label "qwen38.model.correction=${MODEL_CORRECTION}" \
  --label "qwen38.model.sha256=${MODEL_SHA256}" \
  --label "qwen38.model.manifest.sha256=${MODEL_MANIFEST_SHA256}" \
  --gpus all \
  --network host \
  --restart no \
  --read-only \
  --tmpfs "/root:${ROOT_TMPFS_OPTIONS}" \
  --tmpfs "/tmp:${TMP_TMPFS_OPTIONS}" \
  --tmpfs "/run:${RUN_TMPFS_OPTIONS}" \
  --shm-size 8g \
  --cap-drop ALL \
  --security-opt no-new-privileges:true \
  "${docker_environment_args[@]}" \
  --volume "${MODEL_DIR}:/model:ro" \
  --volume "${CACHE_VOLUME}:/root/.cache/vllm" \
  "${IMAGE_TAG}" \
  "${VLLM_ARGS[@]}" >/dev/null

printf '\nContainer created. Blocking on the Docker log stream for readiness...\n'
readiness_pattern='Application startup complete|Engine core initialization failed|EngineCore encountered a fatal error|CUDA out of memory|ValueError:.*KV cache|Traceback \(most recent call last\)'
readiness_line=''
readiness_status=1
startup_deadline=$((SECONDS + STARTUP_TIMEOUT_SECONDS))
coproc STARTUP_LOG_STREAM {
  docker logs --follow "${CONTAINER_NAME}" 2>&1
}
startup_log_pid="${STARTUP_LOG_STREAM_PID}"
startup_log_fd="${STARTUP_LOG_STREAM[0]}"
while ((SECONDS < startup_deadline)); do
  remaining_seconds=$((startup_deadline - SECONDS))
  if ! IFS= read -r -t "${remaining_seconds}" startup_log_line \
      <&"${startup_log_fd}"; then
    break
  fi
  if [[ "${startup_log_line}" =~ ${readiness_pattern} ]]; then
    readiness_line="${startup_log_line}"
    readiness_status=0
    break
  fi
done

# A matched line or absolute timeout ends this one event stream explicitly.
# Do not leave `docker logs --follow` alive and do not wait for a second event.
kill "${startup_log_pid}" >/dev/null 2>&1 || true
wait "${startup_log_pid}" >/dev/null 2>&1 || true
exec {startup_log_fd}<&-

if ((readiness_status != 0)); then
  if [[ "$(docker inspect --format '{{.State.Running}}' "${CONTAINER_NAME}")" != "true" ]]; then
    die "The container exited before reporting application readiness."
  fi
  if ((SECONDS >= startup_deadline)); then
    die "Startup exceeded the pinned timeout." \
      "Timeout: ${STARTUP_TIMEOUT_SECONDS} seconds"
  fi
  die "The blocking Docker-log readiness wait failed." \
    "Exit status: ${readiness_status}"
fi
printf '%s\n' "${readiness_line}"
if [[ "${readiness_line}" != *'Application startup complete'* ]]; then
  die "vLLM reported a fatal startup condition before readiness." \
    "Matched log line: ${readiness_line}"
fi
if [[ "$(docker inspect --format '{{.State.Running}}' "${CONTAINER_NAME}")" != "true" ]]; then
  die "The container exited immediately after reporting readiness."
fi
if ! docker exec "${CONTAINER_NAME}" curl --fail --silent \
    "${ENDPOINT}/health" >/dev/null; then
  die "The one post-readiness health validation failed."
fi
printf 'Pinned vLLM application is ready.\n'

assert_running_profile
cleanup_needed=false
trap - EXIT
print_healthy_summary
