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

if ! container_exists; then
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
