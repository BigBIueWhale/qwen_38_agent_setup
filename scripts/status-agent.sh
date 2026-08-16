#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=runtime-common.sh
source "${SCRIPT_DIR}/runtime-common.sh"
install_unexpected_error_trap
require_no_arguments "./status.sh" "$@"

printf 'Checking pinned host, image, source-patch, and checkpoint state...\n'
check_host_prerequisites
check_pinned_build_inputs
check_model_files

if [[ -e "${AGENT_SERVICE_RUNTIME_ROOT}" ]]; then
  [[ -d "${AGENT_SERVICE_RUNTIME_ROOT}" && ! -L "${AGENT_SERVICE_RUNTIME_ROOT}" ]] || \
    die "The agent-service runtime root is not a real directory." \
      "Path: ${AGENT_SERVICE_RUNTIME_ROOT}"
  require_equal "agent-service runtime root owner/mode" \
    "$(stat -c '%u:%g:%a' "${AGENT_SERVICE_RUNTIME_ROOT}")" "1000:1000:700"
fi

if ! container_exists; then
  if relay_container_exists "${MODEL_BRIDGE_NAME}" || \
     relay_container_exists "${MODEL_INGRESS_NAME}" || \
     [[ -e "${MODEL_SOCKET_DIR}/relay.sock" ]]; then
    die "The backend is absent but fixed-relay/socket state remains." \
      "Bridge present: $(relay_container_exists "${MODEL_BRIDGE_NAME}" && printf yes || printf no)" \
      "Ingress present: $(relay_container_exists "${MODEL_INGRESS_NAME}" && printf yes || printf no)" \
      "Socket present: $([[ -e "${MODEL_SOCKET_DIR}/relay.sock" ]] && printf yes || printf no)" \
      "Run ./stop.sh for exact ownership-checked cleanup."
  fi
  existing_listener="$(listener_output)"
  if [[ -n "${existing_listener}" ]]; then
    die "The project container is stopped, but TCP port ${LISTEN_PORT} is occupied." \
      "Listener(s):" "${existing_listener}" \
      "This is not a valid project state. The unknown listener was not modified."
  fi
  printf '\nSTOPPED — all pinned inputs are valid and TCP port %s is free.\n' \
    "${LISTEN_PORT}"
  printf 'To start the one supported profile: ./start.sh\n'
  exit 0
fi

assert_running_profile
print_healthy_summary
