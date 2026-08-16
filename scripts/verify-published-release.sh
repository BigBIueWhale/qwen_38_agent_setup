#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/runtime-common.sh
source "${SCRIPT_DIR}/runtime-common.sh"

install_unexpected_error_trap
require_no_arguments "./scripts/verify-published-release.sh" "$@"
check_host_prerequisites
check_pinned_build_inputs
require_published_release

printf 'PUBLISHED RELEASE VERIFIED\n'
printf 'remote https://github.com/BigBIueWhale/qwen_38_agent_setup\n'
printf 'branch master\n'
printf 'commit %s\n' "$(git -C "${PROJECT_DIR}" rev-parse --verify HEAD)"
