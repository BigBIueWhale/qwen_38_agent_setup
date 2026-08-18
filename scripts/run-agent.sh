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

for fixed_relay in "${MODEL_BRIDGE_NAME}" "${MODEL_INGRESS_NAME}"; do
  if relay_container_exists "${fixed_relay}"; then
    die "A fixed relay container already exists without the complete healthy backend." \
      "Container: ${fixed_relay}" \
      "Run ./status.sh for evidence, then ./stop.sh for ownership-checked teardown."
  fi
done

assert_port_is_free

if [[ -e "${AGENT_SERVICE_RUNTIME_ROOT}" ]]; then
  [[ -d "${AGENT_SERVICE_RUNTIME_ROOT}" && ! -L "${AGENT_SERVICE_RUNTIME_ROOT}" ]] || \
    die "The agent-service runtime root is not a real directory." \
      "Path: ${AGENT_SERVICE_RUNTIME_ROOT}"
  require_equal "agent-service runtime root owner" \
    "$(stat -c '%u:%g' "${AGENT_SERVICE_RUNTIME_ROOT}")" "1000:1000"
  require_equal "agent-service runtime root mode" \
    "$(stat -c '%a' "${AGENT_SERVICE_RUNTIME_ROOT}")" "700"
else
  install -d -m 0700 "${AGENT_SERVICE_RUNTIME_ROOT}"
fi
require_equal "agent-service runtime root owner/mode" \
  "$(stat -c '%u:%g:%a' "${AGENT_SERVICE_RUNTIME_ROOT}")" "1000:1000:700"

if [[ -e "${MODEL_SOCKET_DIR}" ]]; then
  [[ -d "${MODEL_SOCKET_DIR}" && ! -L "${MODEL_SOCKET_DIR}" ]] || \
    die "The central model socket path is not a real directory." \
      "Path: ${MODEL_SOCKET_DIR}"
else
  install -d -m 0700 "${MODEL_SOCKET_DIR}"
fi
require_equal "central model socket directory owner/mode" \
  "$(stat -c '%u:%g:%a' "${MODEL_SOCKET_DIR}")" "1000:1000:700"
[[ ! -e "${MODEL_SOCKET_DIR}/relay.sock" ]] || \
  die "A stale central model socket exists; refusing to replace it implicitly." \
    "Path: ${MODEL_SOCKET_DIR}/relay.sock" \
    "Run ./stop.sh for ownership-checked cleanup."

cache_volume_created=false
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
  cache_volume_created=true
fi

if [[ "${cache_volume_created}" == true ]]; then
  printf 'Initializing the newly created empty cache volume for non-root vLLM...\n'
  docker run --rm \
    --network none \
    --restart no \
    --user 0:0 \
    --read-only \
    --cap-drop ALL \
    --cap-add CHOWN \
    --security-opt no-new-privileges:true \
    --memory 64m \
    --memory-swap 64m \
    --pids-limit 16 \
    --volume "${CACHE_VOLUME}:/cache" \
    --entrypoint /bin/sh \
    "${EXPECTED_IMAGE_ID}" \
    -ceu '
      test -d /cache
      test ! -L /cache
      test -z "$(find /cache -mindepth 1 -maxdepth 1 -print -quit)"
      chmod 0770 /cache
      chown 2000:0 /cache
      test "$(stat -c "%u:%g:%a" /cache)" = "2000:0:770"
    '
fi

cache_owner_mode="$(
  docker run --rm \
    --network none \
    --restart no \
    --user 2000:0 \
    --read-only \
    --cap-drop ALL \
    --security-opt no-new-privileges:true \
    --memory 64m \
    --memory-swap 64m \
    --pids-limit 16 \
    --volume "${CACHE_VOLUME}:/home/vllm/.cache/vllm" \
    --entrypoint /usr/bin/stat \
    "${EXPECTED_IMAGE_ID}" \
    -c '%u:%g:%a' /home/vllm/.cache/vllm
)"
require_equal "non-root vLLM cache volume owner/mode" "${cache_owner_mode}" "2000:0:770"

docker_environment_args=()
for runtime_environment in "${RUNTIME_ENV[@]}"; do
  docker_environment_args+=(--env "${runtime_environment}")
done

cleanup_needed=false
backend_created=false
bridge_created=false
ingress_created=false

cleanup_backend_is_exact() {
  [[ "$(docker inspect --format '{{index .Config.Labels "qwen38.project"}}' "${CONTAINER_NAME}" 2>/dev/null)" == "${CONTAINER_LABEL}" && \
     "$(docker inspect --format '{{index .Config.Labels "qwen38.runtime.profile"}}' "${CONTAINER_NAME}" 2>/dev/null)" == "${PROFILE_VERSION}" && \
     "$(docker inspect --format '{{.Image}}/{{.Config.Image}}' "${CONTAINER_NAME}" 2>/dev/null)" == "${EXPECTED_IMAGE_ID}/${EXPECTED_IMAGE_ID}" ]]
}

cleanup_relay_is_exact() {
  local name="$1" component="$2"
  [[ "$(docker inspect --format '{{index .Config.Labels "agent_service.profile"}}' "${name}" 2>/dev/null)" == "${AGENT_SERVICE_PROFILE}" && \
     "$(docker inspect --format '{{index .Config.Labels "agent_service.component"}}' "${name}" 2>/dev/null)" == "${component}" && \
     "$(docker inspect --format '{{.Image}}/{{.Config.Image}}/{{json .Config.Cmd}}' "${name}" 2>/dev/null)" == "${EXPECTED_RELAY_IMAGE_ID}/${EXPECTED_RELAY_IMAGE_ID}/[\"${component}\"]" ]]
}

cleanup_failed_start() {
  local status=$?
  local name description running
  local cleanup_failures=0
  trap - EXIT
  if [[ "${cleanup_needed}" == "true" ]]; then
    if [[ "${backend_created}" == true ]] && container_exists && cleanup_backend_is_exact; then
      printf '\nStartup failed. Recent backend logs follow:\n' >&2
      if ! docker logs --tail 120 "${CONTAINER_NAME}" >&2; then
        printf 'CLEANUP DIAGNOSTIC: could not read failed backend logs.\n' >&2
        cleanup_failures=$((cleanup_failures + 1))
      fi
    fi
    printf '\nCleaning only components created by this failed start attempt...\n' >&2
    for name in "${MODEL_INGRESS_NAME}" "${MODEL_BRIDGE_NAME}" "${CONTAINER_NAME}"; do
      if [[ "${name}" == "${MODEL_INGRESS_NAME}" ]]; then
        [[ "${ingress_created}" == true ]] || continue
        description="model ingress"
        if relay_container_exists "${name}" && ! cleanup_relay_is_exact "${name}" model-ingress; then
          printf 'CLEANUP FAILURE: %s changed ownership/identity and was preserved: %s\n' "${description}" "${name}" >&2
          cleanup_failures=$((cleanup_failures + 1))
          continue
        fi
      elif [[ "${name}" == "${MODEL_BRIDGE_NAME}" ]]; then
        [[ "${bridge_created}" == true ]] || continue
        description="model bridge"
        if relay_container_exists "${name}" && ! cleanup_relay_is_exact "${name}" model-bridge; then
          printf 'CLEANUP FAILURE: %s changed ownership/identity and was preserved: %s\n' "${description}" "${name}" >&2
          cleanup_failures=$((cleanup_failures + 1))
          continue
        fi
      else
        [[ "${backend_created}" == true ]] || continue
        description="vLLM backend"
        if container_exists && ! cleanup_backend_is_exact; then
          printf 'CLEANUP FAILURE: %s changed ownership/identity and was preserved: %s\n' "${description}" "${name}" >&2
          cleanup_failures=$((cleanup_failures + 1))
          continue
        fi
      fi
      if docker container inspect "${name}" >/dev/null 2>&1; then
        running="$(docker inspect --format '{{.State.Running}}' "${name}")"
        if [[ "${running}" == true ]] && ! docker stop --timeout -1 "${name}" >/dev/null; then
          printf 'CLEANUP FAILURE: could not stop %s: %s\n' "${description}" "${name}" >&2
          cleanup_failures=$((cleanup_failures + 1))
          continue
        fi
        if ! docker rm "${name}" >/dev/null; then
          printf 'CLEANUP FAILURE: could not remove stopped %s: %s\n' "${description}" "${name}" >&2
          cleanup_failures=$((cleanup_failures + 1))
        fi
      else
        printf 'CLEANUP NOTE: created %s was already absent: %s\n' "${description}" "${name}" >&2
      fi
    done
    if [[ -e "${MODEL_SOCKET_DIR}/relay.sock" ]]; then
      if [[ -S "${MODEL_SOCKET_DIR}/relay.sock" && \
            "$(stat -c '%u:%g:%a' "${MODEL_SOCKET_DIR}/relay.sock")" == 1000:1000:660 ]]; then
        if ! rm -- "${MODEL_SOCKET_DIR}/relay.sock"; then
          printf 'CLEANUP FAILURE: could not remove the exact failed-start model socket.\n' >&2
          cleanup_failures=$((cleanup_failures + 1))
        fi
      else
        printf 'CLEANUP FAILURE: unrecognized model socket was preserved: %s\n' "${MODEL_SOCKET_DIR}/relay.sock" >&2
        cleanup_failures=$((cleanup_failures + 1))
      fi
    fi
    printf 'Failed-start cleanup completed with %d diagnostic failure(s).\n' "${cleanup_failures}" >&2
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
  --network none \
  --restart no \
  --user 2000:0 \
  --read-only \
  --tmpfs "/tmp:${TMP_TMPFS_OPTIONS}" \
  --tmpfs "/run:${RUN_TMPFS_OPTIONS}" \
  --shm-size 8g \
  --cap-drop ALL \
  --security-opt no-new-privileges:true \
  "${docker_environment_args[@]}" \
  --volume "${MODEL_DIR}:/model:ro" \
  --volume "${CACHE_VOLUME}:/home/vllm/.cache/vllm" \
  "${EXPECTED_IMAGE_ID}" \
  "${VLLM_ARGS[@]}" >/dev/null
backend_created=true

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
if kill -0 "${startup_log_pid}" >/dev/null 2>&1; then
  kill "${startup_log_pid}"
fi
capture_child_wait_status "${startup_log_pid}" startup_log_wait_status \
  >/dev/null 2>&1
# The log follower was deliberately terminated after the decisive event or
# timeout. Its status is not a readiness result; readiness_status above is.
: "${startup_log_wait_status}"
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

wait_for_relay_event() {
  local relay_name="$1" event="$2" grep_status
  local -a pipeline_status
  set +o pipefail
  # The pipeline must itself be conditional.  An inherited ERR trap fires for
  # a no-match inside grep even when pipefail is disabled, before PIPESTATUS can
  # otherwise be inspected and the relay logs reported.
  # Follow from the container's start, never from attach time: a relay that
  # prints readiness before this pipeline attaches must still be seen, or
  # the gate races against relay startup speed.
  if timeout --foreground 30s docker logs --follow \
      "${relay_name}" 2>&1 |
      grep --fixed-strings --line-regexp --max-count=1 \
        "${event}" >/dev/null; then
    pipeline_status=("${PIPESTATUS[@]}")
  else
    pipeline_status=("${PIPESTATUS[@]}")
  fi
  set -o pipefail
  grep_status="${pipeline_status[1]}"
  if [[ "${grep_status}" != 0 ]]; then
    if ! docker logs --tail 100 "${relay_name}" >&2; then
      printf 'Could not read diagnostic logs from fixed relay %s.\n' "${relay_name}" >&2
    fi
    die "Fixed relay did not publish its exact readiness event." \
      "Container: ${relay_name}" "Event: ${event}"
  fi
}

backend_id="$(docker inspect --format '{{.Id}}' "${CONTAINER_NAME}")"
printf 'Starting the fixed central model-socket bridge...\n'
docker run --detach \
  --name "${MODEL_BRIDGE_NAME}" \
  --label "agent_service.profile=${AGENT_SERVICE_PROFILE}" \
  --label agent_service.component=model-bridge \
  --label "qwen38.runtime.profile=${PROFILE_VERSION}" \
  --network "container:${backend_id}" \
  --restart no \
  --user 1000:1000 \
  --read-only \
  --cap-drop ALL \
  --security-opt no-new-privileges:true \
  --memory "${RELAY_MEMORY}" \
  --memory-swap "${RELAY_MEMORY}" \
  --pids-limit "${RELAY_PIDS_LIMIT}" \
  --mount "type=bind,src=${MODEL_SOCKET_DIR},dst=/sock" \
  "${EXPECTED_RELAY_IMAGE_ID}" model-bridge >/dev/null
bridge_created=true
wait_for_relay_event "${MODEL_BRIDGE_NAME}" \
  "RELAY_READY role=model-bridge sandbox=${RELAY_SANDBOX} listen=unix:/sock/relay.sock target=tcp:127.0.0.1:8000"
[[ -S "${MODEL_SOCKET_DIR}/relay.sock" ]] || \
  die "Model bridge reported readiness without creating the central Unix socket."

printf 'Starting the only host-network component for 127.0.0.1:8000 ingress...\n'
docker run --detach \
  --name "${MODEL_INGRESS_NAME}" \
  --label "agent_service.profile=${AGENT_SERVICE_PROFILE}" \
  --label agent_service.component=model-ingress \
  --label "qwen38.runtime.profile=${PROFILE_VERSION}" \
  --network host \
  --restart no \
  --user 1000:1000 \
  --read-only \
  --cap-drop ALL \
  --security-opt no-new-privileges:true \
  --memory "${RELAY_MEMORY}" \
  --memory-swap "${RELAY_MEMORY}" \
  --pids-limit "${RELAY_PIDS_LIMIT}" \
  --mount "type=bind,src=${MODEL_SOCKET_DIR},dst=/sock,readonly" \
  "${EXPECTED_RELAY_IMAGE_ID}" model-ingress >/dev/null
ingress_created=true
wait_for_relay_event "${MODEL_INGRESS_NAME}" \
  "RELAY_READY role=model-ingress sandbox=${RELAY_SANDBOX} listen=tcp:127.0.0.1:8000 target=unix:/sock/relay.sock"

if ! curl --fail --silent --show-error --max-time 30 "${ENDPOINT}/health" >/dev/null; then
  die "The fixed model ingress reported readiness but the exact backend health path failed."
fi

assert_running_profile
cleanup_needed=false
trap - EXIT
print_healthy_summary
