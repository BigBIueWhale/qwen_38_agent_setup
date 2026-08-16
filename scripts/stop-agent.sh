#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=runtime-common.sh
source "${SCRIPT_DIR}/runtime-common.sh"
install_unexpected_error_trap
require_no_arguments "./stop.sh" "$@"

any_component=false
container_exists && any_component=true
relay_container_exists "${MODEL_BRIDGE_NAME}" && any_component=true
relay_container_exists "${MODEL_INGRESS_NAME}" && any_component=true

# Resolve and validate every potentially mutated object before stopping or
# removing any of them. A late ownership contradiction must never leave a
# half-torn-down stack.
if container_exists; then
  assert_owned_container
fi
if relay_container_exists "${MODEL_BRIDGE_NAME}"; then
  assert_owned_model_relay "${MODEL_BRIDGE_NAME}" model-bridge
fi
if relay_container_exists "${MODEL_INGRESS_NAME}"; then
  assert_owned_model_relay "${MODEL_INGRESS_NAME}" model-ingress
fi
if [[ -e "${MODEL_SOCKET_DIR}/relay.sock" ]]; then
  [[ -S "${MODEL_SOCKET_DIR}/relay.sock" && \
     "$(stat -c '%u:%g:%a' "${MODEL_SOCKET_DIR}/relay.sock")" == 1000:1000:660 ]] || \
    die "Refusing teardown because the model socket path is unrecognized." \
      "Path: ${MODEL_SOCKET_DIR}/relay.sock"
fi

existing_listener="$(listener_output)"
if relay_container_exists "${MODEL_INGRESS_NAME}" && \
   [[ "$(docker inspect --format '{{.State.Running}}' "${MODEL_INGRESS_NAME}")" == true ]]; then
  assert_exact_loopback_listener
elif [[ -n "${existing_listener}" ]]; then
  die "TCP port ${LISTEN_PORT} is occupied without the exact running model ingress." \
    "Listener(s):" "${existing_listener}" \
    "No component was stopped or removed."
fi

if [[ "${any_component}" == false ]]; then
  existing_listener="$(listener_output)"
  if [[ -n "${existing_listener}" ]]; then
    die "The project containers are absent, but TCP port ${LISTEN_PORT} is still in use." \
      "Listener(s):" "${existing_listener}" \
      "It is not owned by this project and was not killed."
  fi
  if [[ -e "${MODEL_SOCKET_DIR}/relay.sock" ]]; then
    [[ -S "${MODEL_SOCKET_DIR}/relay.sock" && \
       "$(stat -c '%u:%g:%a' "${MODEL_SOCKET_DIR}/relay.sock")" == 1000:1000:660 ]] || \
      die "The component containers are absent but the model socket path is unrecognized." \
        "Path: ${MODEL_SOCKET_DIR}/relay.sock"
    rm -- "${MODEL_SOCKET_DIR}/relay.sock"
    printf 'Removed the exact stale project-owned model Unix socket.\n'
  fi
  printf 'ALREADY STOPPED — no model component or loopback listener remains.\n'
  printf 'To start the one supported profile: ./start.sh\n'
  exit 0
fi

stop_exact() {
  local name="$1" description="$2"
  if [[ "$(docker inspect --format '{{.State.Running}}' "${name}")" == true ]]; then
    printf 'Stopping the exact %s without an arbitrary deadline...\n' "${description}"
    docker stop --timeout -1 "${name}" >/dev/null
  fi
  require_equal "${description} stopped state" \
    "$(docker inspect --format '{{.State.Running}}' "${name}")" false
}

remove_stopped_exact() {
  local name="$1" description="$2"
  require_equal "${description} stopped state before removal" \
    "$(docker inspect --format '{{.State.Running}}' "${name}")" false
  printf 'Removing the stopped %s...\n' "${description}"
  docker rm "${name}" >/dev/null
}

# Keep both fixed relays alive while vLLM drains existing direct and agent
# requests. Docker namespace ownership requires the bridge to be removed
# before the stopped backend itself can be removed, so stop and removal are
# intentionally separate phases.
if container_exists; then
  stop_exact "${CONTAINER_NAME}" "network-none vLLM container"
fi
if relay_container_exists "${MODEL_INGRESS_NAME}"; then
  stop_exact "${MODEL_INGRESS_NAME}" "model ingress"
fi
if relay_container_exists "${MODEL_BRIDGE_NAME}"; then
  stop_exact "${MODEL_BRIDGE_NAME}" "model bridge"
fi

if relay_container_exists "${MODEL_INGRESS_NAME}"; then
  remove_stopped_exact "${MODEL_INGRESS_NAME}" "model ingress"
fi
if relay_container_exists "${MODEL_BRIDGE_NAME}"; then
  remove_stopped_exact "${MODEL_BRIDGE_NAME}" "model bridge"
fi
if container_exists; then
  remove_stopped_exact "${CONTAINER_NAME}" "network-none vLLM container"
fi

if [[ -e "${MODEL_SOCKET_DIR}/relay.sock" ]]; then
  [[ -S "${MODEL_SOCKET_DIR}/relay.sock" && \
     "$(stat -c '%u:%g:%a' "${MODEL_SOCKET_DIR}/relay.sock")" == 1000:1000:660 ]] || \
    die "Refusing to remove an unrecognized model socket path." \
      "Path: ${MODEL_SOCKET_DIR}/relay.sock"
  rm -- "${MODEL_SOCKET_DIR}/relay.sock"
fi

container_exists && die "Docker still reports the backend container after removal."
relay_container_exists "${MODEL_BRIDGE_NAME}" && die "Docker still reports the model bridge after removal."
relay_container_exists "${MODEL_INGRESS_NAME}" && die "Docker still reports the model ingress after removal."

remaining_listener="$(listener_output)"
if [[ -n "${remaining_listener}" ]]; then
  die "The project stopped, but another process still uses TCP port ${LISTEN_PORT}." \
    "Listener(s):" "${remaining_listener}" \
    "The unrelated process was not killed."
fi

printf '\nSTOPPED — backend and both fixed relays are absent; port %s is free.\n' "${LISTEN_PORT}"
printf 'Preserved: model weights, pinned images, reviewed patches/template, and cache volume.\n'
printf 'To start it again: ./start.sh\n'
