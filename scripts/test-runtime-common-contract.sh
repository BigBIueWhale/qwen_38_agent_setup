#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/runtime-common.sh
source "${SCRIPT_DIR}/runtime-common.sh"

if (($# != 0)); then
  printf 'ERROR: runtime-common contract test accepts no arguments.\n' >&2
  exit 2
fi

required_functions=(
  die
  capture_child_wait_status
  require_command
  require_equal
  require_clean_committed_repository
  require_published_release
  check_host_prerequisites
  check_pinned_build_inputs
  assert_running_profile
)
for required_function in "${required_functions[@]}"; do
  declare -F "${required_function}" >/dev/null || {
    printf 'ERROR: required runtime helper is not defined: %s\n' \
      "${required_function}" >&2
    exit 1
  }
done

# A deliberately signalled child must be reaped without invoking the ERR trap.
# This is the exact lifecycle used to end `docker logs --follow` after one
# decisive readiness event.
sleep 60 &
wait_test_pid=$!
kill "${wait_test_pid}"
wait_test_err_trap_fired=false
trap 'wait_test_err_trap_fired=true' ERR
capture_child_wait_status "${wait_test_pid}" wait_test_status
trap - ERR
[[ "${wait_test_status}" == "143" ]] || {
  printf 'ERROR: deliberately terminated child returned unexpected wait status: %s\n' \
    "${wait_test_status}" >&2
  exit 1
}
[[ "${wait_test_err_trap_fired}" == false ]] || {
  printf 'ERROR: deliberately terminated child incorrectly invoked the ERR trap.\n' >&2
  exit 1
}

# The relay waiter deliberately inspects a no-match pipeline.  Its conditional
# form must retain grep's status without dispatching the global ERR trap.
pipeline_err_trap_fired=false
trap 'pipeline_err_trap_fired=true' ERR
set +o pipefail
if printf '%s\n' 'not-the-ready-event' |
    grep --fixed-strings --line-regexp 'the-ready-event' >/dev/null; then
  pipeline_status=("${PIPESTATUS[@]}")
else
  pipeline_status=("${PIPESTATUS[@]}")
fi
set -o pipefail
trap - ERR
[[ "${pipeline_status[1]}" == "1" ]] || {
  printf 'ERROR: no-match grep returned unexpected status: %s\n' \
    "${pipeline_status[1]}" >&2
  exit 1
}
[[ "${pipeline_err_trap_fired}" == false ]] || {
  printf 'ERROR: conditional no-match pipeline incorrectly invoked the ERR trap.\n' >&2
  exit 1
}

require_equal "runtime-common equality self-test" "exact" "exact"
if failure_output="$(
  (
    require_equal "runtime-common mismatch self-test" "observed" "expected"
  ) 2>&1
)"; then
  printf 'ERROR: require_equal accepted a deliberate mismatch.\n' >&2
  exit 1
fi
[[ "${failure_output}" == *'runtime-common mismatch self-test differs from the pinned contract.'* && \
   "${failure_output}" == *'Expected: expected'* && \
   "${failure_output}" == *'Found:    observed'* && \
   "${failure_output}" == *'Nothing was silently substituted.'* ]] || {
  printf 'ERROR: require_equal mismatch evidence changed unexpectedly:\n%s\n' \
    "${failure_output}" >&2
  exit 1
}

printf 'RUNTIME_COMMON_CONTRACT_OK functions=%s\n' "${#required_functions[@]}"
