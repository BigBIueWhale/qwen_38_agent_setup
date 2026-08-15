#!/usr/bin/env bash

COMMON_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${COMMON_SCRIPT_DIR}/.." && pwd)"
# shellcheck source=../config/runtime-v1.sh
source "${PROJECT_DIR}/config/runtime-v1.sh"
readonly COMMON_SCRIPT_DIR PROJECT_DIR
readonly MODEL_DIR="${PROJECT_DIR}/${MODEL_DIR_NAME}"
readonly PATCH_FILE="${PROJECT_DIR}/patches/vllm-turboquant-k8v4-direct-workspace.patch"
readonly MODEL_MANIFEST="${PROJECT_DIR}/manifests/${MODEL_MANIFEST_NAME}"

die() {
  printf '\nERROR: %s\n' "$1" >&2
  shift || true
  if (($#)); then
    printf '%s\n' "$@" >&2
  fi
  printf 'Nothing was silently substituted.\n' >&2
  exit 1
}

unexpected_error() {
  local status=$?
  printf '\nERROR: An unexpected command failed (exit %d).\n' "${status}" >&2
  printf 'Command: %s\n' "${BASH_COMMAND}" >&2
  printf 'The operation did not continue through the failure.\n' >&2
  exit "${status}"
}

install_unexpected_error_trap() {
  trap unexpected_error ERR
}

require_no_arguments() {
  local correct_command="$1"
  shift
  if (($# != 0)); then
    die "This command has no modes or options." \
      "Run it with no arguments: ${correct_command}"
  fi
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || \
    die "Required command is unavailable: $1" \
      "Do not install an arbitrary replacement. Restore the pinned host prerequisite."
}

check_host_prerequisites() {
  local command_name docker_client docker_server toolkit_line toolkit_version
  local gpu_report runtimes
  for command_name in docker git nvidia-container-cli nvidia-smi sha256sum ss; do
    require_command "${command_name}"
  done

  docker_client="$(docker version --format '{{.Client.Version}}')"
  docker_server="$(docker version --format '{{.Server.Version}}')"
  if [[ "${docker_client}" != "${EXPECTED_DOCKER_VERSION}" || \
        "${docker_server}" != "${EXPECTED_DOCKER_VERSION}" ]]; then
    die "Docker version does not match the validated profile." \
      "Expected client/server: ${EXPECTED_DOCKER_VERSION}/${EXPECTED_DOCKER_VERSION}" \
      "Found client/server:    ${docker_client}/${docker_server}"
  fi

  IFS= read -r toolkit_line < <(nvidia-container-cli --version)
  toolkit_version="${toolkit_line#cli-version: }"
  if [[ "${toolkit_version}" != "${EXPECTED_NVIDIA_CONTAINER_CLI_VERSION}" ]]; then
    die "NVIDIA Container Toolkit version does not match." \
      "Expected: ${EXPECTED_NVIDIA_CONTAINER_CLI_VERSION}" \
      "Found:    ${toolkit_version}"
  fi

  runtimes="$(docker info --format '{{json .Runtimes}}')"
  if [[ "${runtimes}" != *'"nvidia"'* ]]; then
    die "Docker's NVIDIA runtime is not configured."
  fi

  gpu_report="$(
    nvidia-smi \
      --query-gpu=name,memory.total,driver_version \
      --format=csv,noheader,nounits
  )"
  if [[ "${gpu_report}" != \
        "${EXPECTED_GPU_NAME}, ${EXPECTED_GPU_MEMORY_MIB}, ${EXPECTED_DRIVER_VERSION}" ]]; then
    die "GPU or NVIDIA driver does not match the validated profile." \
      "Expected: ${EXPECTED_GPU_NAME}, ${EXPECTED_GPU_MEMORY_MIB}, ${EXPECTED_DRIVER_VERSION}" \
      "Found:    ${gpu_report}"
  fi
}

check_pinned_build_inputs() {
  "${COMMON_SCRIPT_DIR}/build-vllm.sh" check

  local image_id
  image_id="$(docker image inspect --format '{{.Id}}' "${IMAGE_TAG}" 2>/dev/null || true)"
  if [[ "${image_id}" != "${EXPECTED_IMAGE_ID}" ]]; then
    die "The exact validated runtime image is missing or incorrect." \
      "Expected tag: ${IMAGE_TAG}" \
      "Expected ID:  ${EXPECTED_IMAGE_ID}" \
      "Found ID:     ${image_id:-nothing}" \
      "Run ./scripts/restore-images.sh to restore the pinned offline archive."
  fi

  local archive="${PROJECT_DIR}/artifacts/${IMAGE_ARCHIVE_NAME}"
  if [[ ! -f "${archive}" ]]; then
    die "The pinned offline image archive is missing." \
      "Expected: ${archive}" \
      "The current image may run, but disaster recovery is not complete."
  fi
  printf 'Verifying the pinned offline image archive...\n'
  printf '%s  %s\n' "${IMAGE_ARCHIVE_SHA256}" "${archive}" | \
    sha256sum --check --strict
}

check_model_files() {
  local expected_file path base_name
  local -A expected_files=()
  if [[ ! -d "${MODEL_DIR}" ]]; then
    die "The pinned model directory is missing." \
      "Expected: ${MODEL_DIR}"
  fi
  printf '%s  %s\n' "${MODEL_MANIFEST_SHA256}" "${MODEL_MANIFEST}" | \
    sha256sum --check --strict

  for expected_file in "${MODEL_FILES[@]}"; do
    expected_files["${expected_file}"]=1
    [[ -f "${MODEL_DIR}/${expected_file}" ]] || \
      die "Pinned model snapshot file is missing." \
        "Missing: ${MODEL_DIR}/${expected_file}"
  done
  shopt -s dotglob nullglob
  for path in "${MODEL_DIR}"/*; do
    [[ -f "${path}" ]] || continue
    base_name="${path##*/}"
    [[ -n "${expected_files[${base_name}]:-}" ]] || \
      die "Unexpected top-level file exists in the pinned model snapshot." \
        "Unexpected: ${path}" \
        "The file was not deleted or ignored."
  done
  shopt -u dotglob nullglob

  printf 'Verifying every pinned model, tokenizer, template, and configuration file; this intentionally takes several seconds...\n'
  (
    cd -- "${MODEL_DIR}"
    sha256sum --check --strict "${MODEL_MANIFEST}"
  )
}

listener_output() {
  ss -H -ltn "sport = :${LISTEN_PORT}"
}

assert_port_is_free() {
  local output
  output="$(listener_output)"
  if [[ -n "${output}" ]]; then
    die "TCP port ${LISTEN_PORT} is already in use; the server was not started." \
      "Existing listener(s):" "${output}" \
      "Run ./status.sh. Do not kill an unknown process automatically."
  fi
}

assert_exact_loopback_listener() {
  local output local_address
  local -a lines=()
  output="$(listener_output)"
  if [[ -n "${output}" ]]; then
    mapfile -t lines <<< "${output}"
  fi
  if ((${#lines[@]} != 1)); then
    die "Expected exactly one listener on TCP port ${LISTEN_PORT}." \
      "Found ${#lines[@]} listener(s):" "${output:-none}"
  fi
  read -r _ _ _ local_address _ <<< "${lines[0]}"
  if [[ "${local_address}" != "${LISTEN_HOST}:${LISTEN_PORT}" ]]; then
    die "The service is listening on the wrong address." \
      "Expected: ${LISTEN_HOST}:${LISTEN_PORT}" \
      "Found:    ${local_address}" \
      "The unsafe listener was not accepted as healthy."
  fi
}

container_exists() {
  docker container inspect "${CONTAINER_NAME}" >/dev/null 2>&1
}

assert_owned_container() {
  local project_label profile_label
  project_label="$(
    docker inspect --format '{{index .Config.Labels "qwen38.project"}}' \
      "${CONTAINER_NAME}"
  )"
  profile_label="$(
    docker inspect --format '{{index .Config.Labels "qwen38.runtime.profile"}}' \
      "${CONTAINER_NAME}"
  )"
  if [[ "${project_label}" != "${CONTAINER_LABEL}" || \
        "${profile_label}" != "${PROFILE_VERSION}" ]]; then
    die "A container named ${CONTAINER_NAME} exists but is not the exact owned profile." \
      "Project label: ${project_label:-missing}" \
      "Profile label: ${profile_label:-missing}" \
      "It was not modified or removed."
  fi
}

assert_runtime_versions() {
  local actual_report
  actual_report="$(
    docker exec "${CONTAINER_NAME}" python3 -c \
      'import importlib.metadata as m, platform, torch, transformers, vllm; names=["tokenizers","safetensors","compressed-tensors","flashinfer-python","triton","numpy","fastapi","uvicorn"]; print("python="+platform.python_version()); print("vllm="+vllm.__version__); print("torch="+torch.__version__); print("transformers="+transformers.__version__); [print(n+"="+m.version(n)) for n in names]; print("torch_cuda="+str(torch.version.cuda)); print("cuda_capability="+".".join(map(str,torch.cuda.get_device_capability())))'
  )"
  if [[ "${actual_report}" != "${EXPECTED_RUNTIME_REPORT}" ]]; then
    die "Container software versions differ from the validated lock." \
      "Expected:" "${EXPECTED_RUNTIME_REPORT}" \
      "Found:" "${actual_report}"
  fi
}

assert_running_profile() {
  local running image_id network port_bindings published_ports cap_drop security_opts
  local model_source model_rw cache_name cache_type cache_project_label
  local cache_profile_label actual_command expected_command
  local actual_environment wrapped_environment required_environment patched_sha api

  assert_owned_container
  running="$(docker inspect --format '{{.State.Running}}' "${CONTAINER_NAME}")"
  [[ "${running}" == "true" ]] || \
    die "The owned container exists but is not running." \
      "Run ./stop.sh to clean it up, then ./start.sh."

  image_id="$(docker inspect --format '{{.Image}}' "${CONTAINER_NAME}")"
  [[ "${image_id}" == "${EXPECTED_IMAGE_ID}" ]] || \
    die "Running container image does not match." \
      "Expected: ${EXPECTED_IMAGE_ID}" "Found: ${image_id}"

  network="$(docker inspect --format '{{.HostConfig.NetworkMode}}' "${CONTAINER_NAME}")"
  [[ "${network}" == "host" ]] || \
    die "Running container has the wrong network mode." \
      "Expected the single loopback profile's host network; found: ${network}"

  port_bindings="$(docker inspect --format '{{json .HostConfig.PortBindings}}' "${CONTAINER_NAME}")"
  [[ "${port_bindings}" == "{}" ]] || \
    die "Docker port publishing is unexpectedly configured." \
      "Expected: {}" "Found: ${port_bindings}"
  published_ports="$(docker port "${CONTAINER_NAME}")"
  [[ -z "${published_ports}" ]] || \
    die "Docker reports published ports, which are forbidden." \
      "Found:" "${published_ports}"

  cap_drop="$(docker inspect --format '{{json .HostConfig.CapDrop}}' "${CONTAINER_NAME}")"
  security_opts="$(docker inspect --format '{{json .HostConfig.SecurityOpt}}' "${CONTAINER_NAME}")"
  [[ "${cap_drop}" == '["ALL"]' ]] || \
    die "Container capabilities are not fully dropped." \
      "Expected: [\"ALL\"]" "Found: ${cap_drop}"
  [[ "${security_opts}" == '["no-new-privileges:true"]' ]] || \
    die "Container no-new-privileges protection is missing." \
      "Found: ${security_opts}"

  model_source="$(docker inspect --format '{{range .Mounts}}{{if eq .Destination "/model"}}{{.Source}}{{end}}{{end}}' "${CONTAINER_NAME}")"
  model_rw="$(docker inspect --format '{{range .Mounts}}{{if eq .Destination "/model"}}{{.RW}}{{end}}{{end}}' "${CONTAINER_NAME}")"
  [[ "${model_source}" == "${MODEL_DIR}" && "${model_rw}" == "false" ]] || \
    die "Model mount is not the exact read-only checkpoint mount." \
      "Expected source: ${MODEL_DIR}" "Found source: ${model_source:-missing}" \
      "Expected writable: false" "Found writable: ${model_rw:-missing}"

  cache_name="$(docker inspect --format '{{range .Mounts}}{{if eq .Destination "/root/.cache/vllm"}}{{.Name}}{{end}}{{end}}' "${CONTAINER_NAME}")"
  cache_type="$(docker inspect --format '{{range .Mounts}}{{if eq .Destination "/root/.cache/vllm"}}{{.Type}}{{end}}{{end}}' "${CONTAINER_NAME}")"
  [[ "${cache_name}" == "${CACHE_VOLUME}" && "${cache_type}" == "volume" ]] || \
    die "vLLM cache mount does not match the pinned named volume." \
      "Expected: volume ${CACHE_VOLUME}" \
      "Found: ${cache_type:-missing} ${cache_name:-missing}"
  cache_project_label="$(
    docker volume inspect --format '{{index .Labels "qwen38.project"}}' \
      "${CACHE_VOLUME}"
  )"
  cache_profile_label="$(
    docker volume inspect --format '{{index .Labels "qwen38.runtime.profile"}}' \
      "${CACHE_VOLUME}"
  )"
  [[ "${cache_project_label}" == "${CONTAINER_LABEL}" && \
     "${cache_profile_label}" == "${PROFILE_VERSION}" ]] || \
    die "Pinned cache-volume labels do not match." \
      "Expected project/profile: ${CONTAINER_LABEL}/${PROFILE_VERSION}" \
      "Found project/profile: ${cache_project_label:-missing}/${cache_profile_label:-missing}"

  actual_command="$(docker inspect --format '{{range .Config.Cmd}}{{println .}}{{end}}' "${CONTAINER_NAME}")"
  expected_command="$(printf '%s\n' "${VLLM_ARGS[@]}")"
  [[ "${actual_command}" == "${expected_command}" ]] || \
    die "Running vLLM arguments differ from the one validated profile." \
      "Expected:" "${expected_command}" "Found:" "${actual_command}"

  actual_environment="$(docker inspect --format '{{range .Config.Env}}{{println .}}{{end}}' "${CONTAINER_NAME}")"
  wrapped_environment=$'\n'"${actual_environment}"$'\n'
  for required_environment in "${RUNTIME_ENV[@]}"; do
    [[ "${wrapped_environment}" == *$'\n'"${required_environment}"$'\n'* ]] || \
      die "Required container environment setting is missing." \
        "Missing: ${required_environment}"
  done

  assert_exact_loopback_listener

  docker exec "${CONTAINER_NAME}" curl --fail --silent \
    "${ENDPOINT}/health" >/dev/null
  api="$(
    docker exec "${CONTAINER_NAME}" curl --fail --silent \
      "${ENDPOINT}/v1/models"
  )"
  [[ "${api}" == *"\"id\":\"${SERVED_MODEL}\""* && \
     "${api}" == *"\"max_model_len\":${MAX_MODEL_LEN}"* ]] || \
    die "The API reports the wrong model identity or context limit." \
      "Expected model: ${SERVED_MODEL}" \
      "Expected max length: ${MAX_MODEL_LEN}" \
      "Response: ${api}"

  patched_sha="$(
    docker exec "${CONTAINER_NAME}" sha256sum \
      /usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/turboquant_attn.py \
      | { read -r file_sha _; printf '%s' "${file_sha}"; }
  )"
  [[ "${patched_sha}" == "${PATCHED_FILE_SHA256}" ]] || \
    die "Running TurboQuant source does not match the reviewed patch." \
      "Expected: ${PATCHED_FILE_SHA256}" "Found: ${patched_sha}"

  assert_runtime_versions
}

print_healthy_summary() {
  printf '\nHEALTHY — the one supported serving profile is correct.\n'
  printf 'Endpoint:      %s\n' "${ENDPOINT}"
  printf 'Listener:      %s:%s only\n' "${LISTEN_HOST}" "${LISTEN_PORT}"
  printf 'Model:         %s\n' "${SERVED_MODEL}"
  printf 'Context limit: %s tokens\n' "${MAX_MODEL_LEN}"
  printf 'Runtime image: %s\n' "${EXPECTED_IMAGE_ID}"
  printf 'To stop it:    ./stop.sh\n'
}
