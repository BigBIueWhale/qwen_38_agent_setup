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
  host_isolation_refusal
  host_isolation_refusals
  host_isolation_cases
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

# The host check as run-agent.sh runs it, with Docker's reported security
# options replaced by the given report. The other host facts it reads are
# faked to a host that satisfies them, and a command it only requires to exist
# refuses loudly if it is ever invoked. This runs inside the pinned base image,
# which carries none of these host tools.
host_isolation_check_on() (
  # Named apart from the check's own locals: bash scopes dynamically, so the
  # fake must not read a variable the check itself declares.
  local fake_security_options="$1"
  docker() {
    case "$*" in
      "version --format {{.Server.Version}}")
        printf '%s\n' 29.8.1
        ;;
      "info --format {{json .SecurityOptions}}")
        printf '%s\n' "${fake_security_options}"
        ;;
      "info --format {{json .Runtimes}}")
        printf '%s\n' '{"nvidia":{"path":"nvidia-container-runtime"},"runc":{"path":"runc"}}'
        ;;
      *)
        printf 'unexpected fake Docker invocation: %q' "$1" >&2
        printf ' %q' "${@:2}" >&2
        printf '\n' >&2
        return 97
        ;;
    esac
  }
  nvidia-smi() {
    [[ "$*" == '--query-gpu=memory.total --format=csv,noheader,nounits' ]] || {
      printf 'unexpected fake nvidia-smi invocation: %s\n' "$*" >&2
      return 97
    }
    printf '%s\n' "${MINIMUM_GPU_MEMORY_MIB}"
  }
  # git and ss only have to exist; these bodies run only if the check misuses
  # them.
  # shellcheck disable=SC2317
  git() {
    printf 'the host check must not invoke git: %s\n' "$*" >&2
    return 97
  }
  # shellcheck disable=SC2317
  ss() {
    printf 'the host check must not invoke ss: %s\n' "$*" >&2
    return 97
  }
  check_host_prerequisites
)

# Every case of the host-isolation rule's own corpus -- the file agent_service
# carries byte-identically -- goes through the real host check: each accepted
# report passes silently, and each refused one exits with a refusal naming the
# failed property, what is required, what was reported, and a next action.
host_isolation_accepted=0
host_isolation_refused=0
while IFS=$'\x1f' read -r verdict report fragment; do
  if isolation_output="$(host_isolation_check_on "${report}" 2>&1)"; then
    isolation_status=0
  else
    isolation_status=$?
  fi
  case "${verdict}" in
    accept)
      [[ "${isolation_status}" == 0 && -z "${isolation_output}" ]] || {
        printf 'ERROR: the host check refused an accepted report %s (exit %s):\n%s\n' \
          "${report}" "${isolation_status}" "${isolation_output}" >&2
        exit 1
      }
      host_isolation_accepted=$((host_isolation_accepted + 1))
      ;;
    refuse)
      [[ "${isolation_status}" == 1 && \
         "${isolation_output}" == *'ERROR: This host does not provide the container isolation the deployment requires.'* && \
         "${isolation_output}" == *"Docker reports these security options: ${report:-<nothing>}"* && \
         "${isolation_output}" == *"${fragment}"* && \
         "${isolation_output}" == *'  Required: '* && \
         "${isolation_output}" == *'  Reported: '* && \
         "${isolation_output}" == *'  Next:     '* && \
         "${isolation_output}" == *'Nothing was silently substituted.'* ]] || {
        printf 'ERROR: the host check did not refuse %s with "%s" (exit %s):\n%s\n' \
          "${report:-<nothing>}" "${fragment}" "${isolation_status}" "${isolation_output}" >&2
        exit 1
      }
      host_isolation_refused=$((host_isolation_refused + 1))
      ;;
    *)
      printf 'ERROR: host-isolation case has no verdict: %q\n' "${verdict}" >&2
      exit 1
      ;;
  esac
done < <(host_isolation_cases)
((host_isolation_accepted > 0 && host_isolation_refused > 0)) || {
  printf 'ERROR: the host-isolation corpus yielded %s accepted and %s refused cases.\n' \
    "${host_isolation_accepted}" "${host_isolation_refused}" >&2
  exit 1
}

printf 'RUNTIME_COMMON_CONTRACT_OK functions=%s host-isolation=%s-accepted-%s-refused\n' \
  "${#required_functions[@]}" "${host_isolation_accepted}" "${host_isolation_refused}"
