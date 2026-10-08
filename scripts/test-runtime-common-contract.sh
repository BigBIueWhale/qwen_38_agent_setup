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
  lock_value
  write_lock_values
  settle_produced_identity
  require_command
  require_equal
  require_clean_committed_repository
  require_published_release
  check_host_prerequisites
  check_pinned_build_inputs
  check_model_files
  run_gpu_release_units
  require_image_built_from_inputs
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
  local fake_security_options="$1" fake_gpu_capability="${2:-${VALIDATED_CUDA_CAPABILITY}}"
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
    [[ "$*" == '--query-gpu=compute_cap --format=csv,noheader,nounits' ]] || {
      printf 'unexpected fake nvidia-smi invocation: %s\n' "$*" >&2
      return 97
    }
    printf '%s\n' "${fake_gpu_capability}"
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

# A GPU of any other compute capability is refused before launch as outside the
# validated lock, naming what was validated, what was found and the next action;
# the engine's own refusal of a GPU without native FP4 is a separate property.
accepted_security_options="$(host_isolation_cases | awk -F $'\x1f' '$1 == "accept" {print $2; exit}')"
for capability in 10.0 9.0 12.1; do
  if capability_output="$(host_isolation_check_on "${accepted_security_options}" "${capability}" 2>&1)"; then
    printf 'ERROR: the host check accepted compute capability %s.\n' "${capability}" >&2
    exit 1
  fi
  [[ "${capability_output}" == *"ERROR: GPU compute capability ${capability} is outside the validated lock."* && \
     "${capability_output}" == *"Validated: ${VALIDATED_CUDA_CAPABILITY} -- the image's kernels are built for it"* && \
     "${capability_output}" == *"Found:     ${capability}"* && \
     "${capability_output}" == *"Next:      serve on a compute capability ${VALIDATED_CUDA_CAPABILITY} GPU"* ]] || {
    printf 'ERROR: the host check did not refuse compute capability %s by its own statement:\n%s\n' \
      "${capability}" "${capability_output}" >&2
    exit 1
  }
done

# The units that need the GPU run in the image a build has made: each shipped
# unit, on the GPU, with the verified model read-only at /model, as the server's
# user and in its environment, and every one must pass. A host without the
# validated GPU is refused before anything runs, and a unit that fails refuses
# by name and stops. Docker, the GPU and the model check are faked to the given
# host; the gate's Docker runs are recorded rather than executed.
gpu_units_log="$(mktemp)"
gpu_release_units_on() (
  local fake_runtimes="$1" fake_capability="$2" failing_unit="$3"
  docker() {
    case "$*" in
      "version --format {{.Server.Version}}") printf '%s\n' 29.8.1 ;;
      "info --format {{json .SecurityOptions}}") printf '%s\n' "${accepted_security_options}" ;;
      "info --format {{json .Runtimes}}") printf '%s\n' "${fake_runtimes}" ;;
      "run --rm "*)
        printf 'run %s\n' "${*:2}" >>"${gpu_units_log}"
        [[ -z "${failing_unit}" || "$*" != *" /opt/qwen38/${failing_unit}.py" ]]
        ;;
      *)
        printf 'unexpected fake Docker invocation: %s\n' "$*" >&2
        return 97
        ;;
    esac
  }
  nvidia-smi() {
    printf '%s\n' "${fake_capability}"
  }
  # shellcheck disable=SC2317
  git() { return 97; }
  # shellcheck disable=SC2317
  ss() { return 97; }
  check_model_files() {
    printf 'model verified\n' >>"${gpu_units_log}"
  }
  run_gpu_release_units sha256:0123abcd
)
nvidia_runtimes='{"nvidia":{"path":"nvidia-container-runtime"},"runc":{"path":"runc"}}'
gpu_unit_cases=0
for gpu_case in passes no-nvidia-runtime capability-9.0 turboquant-fails nvfp4-fails; do
  : >"${gpu_units_log}"
  runtimes="${nvidia_runtimes}" capability="${VALIDATED_CUDA_CAPABILITY}" failing=""
  case "${gpu_case}" in
    no-nvidia-runtime) runtimes='{"runc":{"path":"runc"}}' ;;
    capability-9.0) capability=9.0 ;;
    turboquant-fails) failing=turboquant_k8v4_unit ;;
    nvfp4-fails) failing=nvfp4_kernel_unit ;;
  esac
  if gpu_output="$(gpu_release_units_on "${runtimes}" "${capability}" "${failing}" 2>&1)"; then
    gpu_status=0
  else
    gpu_status=$?
  fi
  mapfile -t gpu_runs < <(grep '^run ' "${gpu_units_log}" || true)
  case "${gpu_case}" in
    passes)
      [[ "${gpu_status}" == 0 && "${#gpu_runs[@]}" == "${#GPU_RELEASE_UNITS[@]}" && \
         "$(head -n 1 "${gpu_units_log}")" == 'model verified' ]] || {
        printf 'ERROR: the GPU units did not all run after the model check (exit %s):\n%s\n' \
          "${gpu_status}" "$(cat "${gpu_units_log}")" >&2
        exit 1
      }
      for index in "${!GPU_RELEASE_UNITS[@]}"; do
        gpu_run="${gpu_runs[${index}]}"
        for fragment in '--gpus all' '--network none' '--user 2000:0' '--read-only' \
            "--volume ${MODEL_DIR}:/model:ro" '--env HOME=/home/vllm' \
            "--entrypoint python3 sha256:0123abcd /opt/qwen38/${GPU_RELEASE_UNITS[${index}]}.py"; do
          [[ "${gpu_run}" == *" ${fragment}"* ]] || {
            printf 'ERROR: GPU unit run %s lacks "%s": %s\n' "${index}" "${fragment}" "${gpu_run}" >&2
            exit 1
          }
        done
        [[ "${gpu_run}" == *" /opt/qwen38/${GPU_RELEASE_UNITS[${index}]}.py" ]] || {
          printf 'ERROR: GPU unit run %s does not end with its unit: %s\n' "${index}" "${gpu_run}" >&2
          exit 1
        }
      done
      ;;
    no-nvidia-runtime|capability-9.0)
      [[ "${gpu_status}" == 1 && "${#gpu_runs[@]}" == 0 && \
         ! -s "${gpu_units_log}" && \
         "${gpu_output}" == *'The image is pinned only once the units that need the GPU pass in it on this host.'* && \
         ( "${gpu_output}" == *"ERROR: Docker's NVIDIA runtime is not configured."* || \
           "${gpu_output}" == *'ERROR: GPU compute capability 9.0 is outside the validated lock.'* ) ]] || {
        printf 'ERROR: a host without the validated GPU was not refused before any unit ran (%s, exit %s):\n%s\n' \
          "${gpu_case}" "${gpu_status}" "${gpu_output}" >&2
        exit 1
      }
      ;;
    *-fails)
      expected_runs=1
      [[ "${failing}" == nvfp4_kernel_unit ]] && expected_runs=2
      [[ "${gpu_status}" == 1 && "${#gpu_runs[@]}" == "${expected_runs}" && \
         "${gpu_output}" == *"ERROR: The release unit ${failing} failed in sha256:0123abcd on this host's GPU; the image was not pinned."* && \
         "${gpu_output}" == *"It ran the shipped /opt/qwen38/${failing}.py with ${MODEL_DIR} at /model"* && \
         "${gpu_output}" == *'Next: '* ]] || {
        printf 'ERROR: a failing %s was not refused by name after %s run(s) (exit %s):\n%s\n' \
          "${failing}" "${expected_runs}" "${gpu_status}" "${gpu_output}" >&2
        exit 1
      }
      ;;
  esac
  gpu_unit_cases=$((gpu_unit_cases + 1))
done
rm -f -- "${gpu_units_log}"

# A produced identity is written by the step that produced it and verified by
# every later run of the same inputs, on a scratch copy of the real lock, for
# both pairs the lock holds: the image a build makes and the archive a save
# writes. A run of the same inputs that produces anything else is refused and
# changes nothing -- the reproducibility guarantee -- and a run of other inputs
# records its own, rewriting only those two lines in the shape the paired
# repository reads.
digest_of() {
  local digest
  digest="$(printf '%s' "$1" | sha256sum)"
  printf '%s\n' "${digest%% *}"
}
pin_scratch="$(mktemp -d)"
pin_lock="${pin_scratch}/runtime-v1.sh"
settled_pairs=0
for pair in EXPECTED_IMAGE_ID:IMAGE_BUILD_INPUTS_SHA256:sha256: \
            IMAGE_ARCHIVE_SHA256:IMAGE_ARCHIVE_INPUTS_SHA256:; do
  identity_name="${pair%%:*}"
  inputs_name="${pair#*:}"
  identity_prefix="${inputs_name#*:}"
  inputs_name="${inputs_name%%:*}"
  cp -- "${PROJECT_DIR}/config/runtime-v1.sh" "${pin_lock}"
  chmod 0640 "${pin_lock}"
  first_inputs="$(digest_of "first inputs of ${identity_name}")"
  other_inputs="$(digest_of "other inputs of ${identity_name}")"
  first_identity="${identity_prefix}$(digest_of "first ${identity_name}")"
  other_identity="${identity_prefix}$(digest_of "other ${identity_name}")"

  settle_produced_identity outcome "${pin_lock}" "${identity_name}" "${inputs_name}" \
    "${first_identity}" "${first_inputs}" "Synthetic refusal." "Next: synthetic."
  [[ "${outcome}" == written ]] || {
    printf 'ERROR: new inputs for %s were %s, not written.\n' "${identity_name}" "${outcome}" >&2
    exit 1
  }
  diff <(grep -v "^readonly \(${identity_name}\|${inputs_name}\)=" "${PROJECT_DIR}/config/runtime-v1.sh") \
       <(grep -v "^readonly \(${identity_name}\|${inputs_name}\)=" "${pin_lock}") >/dev/null || {
    printf 'ERROR: writing %s changed lines other than its own two.\n' "${identity_name}" >&2
    exit 1
  }
  [[ "$(grep -c "^readonly ${identity_name}=\"${first_identity}\"\$" "${pin_lock}")" == 1 && \
     "$(grep -c "^readonly ${inputs_name}=\"${first_inputs}\"\$" "${pin_lock}")" == 1 ]] || {
    printf 'ERROR: %s and %s were not written as readonly NAME="value" lines.\n' \
      "${identity_name}" "${inputs_name}" >&2
    exit 1
  }
  [[ "$(stat -c %a "${pin_lock}")" == 640 ]] || {
    printf 'ERROR: writing %s changed the lock'"'"'s mode.\n' "${identity_name}" >&2
    exit 1
  }
  cp -- "${pin_lock}" "${pin_scratch}/written"

  settle_produced_identity outcome "${pin_lock}" "${identity_name}" "${inputs_name}" \
    "${first_identity}" "${first_inputs}" "Synthetic refusal." "Next: synthetic."
  [[ "${outcome}" == verified ]] && cmp -s "${pin_lock}" "${pin_scratch}/written" || {
    printf 'ERROR: a second run of the same inputs reproducing %s was not verified unchanged.\n' \
      "${identity_name}" >&2
    exit 1
  }

  if refusal_output="$(
    (
      settle_produced_identity outcome "${pin_lock}" "${identity_name}" "${inputs_name}" \
        "${other_identity}" "${first_inputs}" \
        "Synthetic mismatch for ${identity_name}." "Next: synthetic next action."
    ) 2>&1
  )"; then
    printf 'ERROR: a second run of the same inputs producing another %s was accepted.\n' \
      "${identity_name}" >&2
    exit 1
  fi
  [[ "${refusal_output}" == *"ERROR: Synthetic mismatch for ${identity_name}."* && \
     "${refusal_output}" == *"Pinned:   ${first_identity} (${identity_name})"* && \
     "${refusal_output}" == *"Produced: ${other_identity}"* && \
     "${refusal_output}" == *"Inputs:   ${first_inputs} (${inputs_name})"* && \
     "${refusal_output}" == *"Next: synthetic next action."* && \
     "${refusal_output}" == *"${pin_lock} was not changed."* ]] && \
    cmp -s "${pin_lock}" "${pin_scratch}/written" || {
    printf 'ERROR: the same-inputs mismatch for %s was not refused unchanged by its own statement:\n%s\n' \
      "${identity_name}" "${refusal_output}" >&2
    exit 1
  }

  settle_produced_identity outcome "${pin_lock}" "${identity_name}" "${inputs_name}" \
    "${other_identity}" "${other_inputs}" "Synthetic refusal." "Next: synthetic."
  [[ "${outcome}" == written && \
     "$(grep -c "^readonly ${identity_name}=\"${other_identity}\"\$" "${pin_lock}")" == 1 ]] || {
    printf 'ERROR: other inputs did not record their own %s.\n' "${identity_name}" >&2
    exit 1
  }
  settled_pairs=$((settled_pairs + 1))
done
rm -rf -- "${pin_scratch}"

# Serving refuses, before anything loads, a commit whose image inputs are not
# the ones the pinned image was built from, naming both digests and the release
# that would serve it; the pinned inputs pass silently.
# The build verifier calls it with its own values readonly, as here.
readonly image_inputs_sha256="${IMAGE_BUILD_INPUTS_SHA256}" context_file_count=128
pinned_output="$(require_image_built_from_inputs "${IMAGE_BUILD_INPUTS_SHA256}" 128 2>&1)" && \
  [[ -z "${pinned_output}" ]] || {
  printf 'ERROR: the inputs the pinned image was built from were refused:\n%s\n' \
    "${pinned_output}" >&2
  exit 1
}
stale_inputs="$(digest_of 'image inputs of a later commit')"
if stale_output="$( (require_image_built_from_inputs "${stale_inputs}" 128) 2>&1 )"; then
  printf 'ERROR: the pinned image was accepted for inputs it was not built from.\n' >&2
  exit 1
fi
[[ "${stale_output}" == *"ERROR: The pinned runtime image was not built from this commit's image inputs."* && \
   "${stale_output}" == *"Pinned image: ${EXPECTED_IMAGE_ID}"* && \
   "${stale_output}" == *"  built from: ${IMAGE_BUILD_INPUTS_SHA256}"* && \
   "${stale_output}" == *"This commit:  ${stale_inputs} (128 context files)"* && \
   "${stale_output}" == *"Next: cut a runtime image release from this commit"* ]] || {
  printf 'ERROR: stale image inputs were not refused by their own statement:\n%s\n' \
    "${stale_output}" >&2
  exit 1
}
# start.sh, status.sh and the release audit reach the refusal through the
# serving verification, which runs before any container is created.
[[ "$(declare -f check_pinned_build_inputs)" == *'/build-vllm.sh" serve-check'* ]] || {
  printf 'ERROR: check_pinned_build_inputs does not run build-vllm.sh serve-check.\n' >&2
  exit 1
}

printf 'RUNTIME_COMMON_CONTRACT_OK functions=%s host-isolation=%s-accepted-%s-refused capability-refusals=3 gpu-unit-cases=%s produced-identities=%s stale-image-refusal=1\n' \
  "${#required_functions[@]}" "${host_isolation_accepted}" "${host_isolation_refused}" "${gpu_unit_cases}" "${settled_pairs}"
