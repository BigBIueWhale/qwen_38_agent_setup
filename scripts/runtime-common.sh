#!/usr/bin/env bash

COMMON_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${COMMON_SCRIPT_DIR}/.." && pwd)"
# shellcheck source=../config/runtime-v1.sh
source "${PROJECT_DIR}/config/runtime-v1.sh"
readonly COMMON_SCRIPT_DIR PROJECT_DIR
readonly MODEL_DIR="${PROJECT_DIR}/${MODEL_DIR_NAME}"
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
  local gpu_report runtimes git_line sha256sum_line ss_line
  for command_name in docker git nvidia-container-cli nvidia-smi sha256sum ss; do
    require_command "${command_name}"
  done

  if [[ "${BASH_VERSION}" != "${EXPECTED_BASH_VERSION}" ]]; then
    die "Bash version does not match the validated host tools." \
      "Expected: ${EXPECTED_BASH_VERSION}" \
      "Found:    ${BASH_VERSION}"
  fi
  IFS= read -r git_line < <(git --version)
  IFS= read -r sha256sum_line < <(sha256sum --version)
  IFS= read -r ss_line < <(ss --version 2>&1)
  if [[ "${git_line}" != "${EXPECTED_GIT_VERSION_REPORT}" || \
        "${sha256sum_line}" != "${EXPECTED_SHA256SUM_VERSION_REPORT}" || \
        "${ss_line}" != "${EXPECTED_SS_VERSION_REPORT}" ]]; then
    die "Host command versions do not match the validated lock." \
      "Expected git:       ${EXPECTED_GIT_VERSION_REPORT}" \
      "Found git:          ${git_line}" \
      "Expected sha256sum: ${EXPECTED_SHA256SUM_VERSION_REPORT}" \
      "Found sha256sum:    ${sha256sum_line}" \
      "Expected ss:        ${EXPECTED_SS_VERSION_REPORT}" \
      "Found ss:           ${ss_line}"
  fi

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
  local model_source model_rw model_revision_label cache_name cache_type
  local cache_project_label cache_profile_label cache_model_revision_label
  local actual_command expected_command
  local actual_environment wrapped_environment required_environment api
  local installed_report expected_installed_report
  local additional_installed_report expected_additional_installed_report
  local image_profile_label

  assert_owned_container
  running="$(docker inspect --format '{{.State.Running}}' "${CONTAINER_NAME}")"
  [[ "${running}" == "true" ]] || \
    die "The owned container exists but is not running." \
      "Run ./stop.sh to clean it up, then ./start.sh."

  image_id="$(docker inspect --format '{{.Image}}' "${CONTAINER_NAME}")"
  [[ "${image_id}" == "${EXPECTED_IMAGE_ID}" ]] || \
    die "Running container image does not match." \
      "Expected: ${EXPECTED_IMAGE_ID}" "Found: ${image_id}"
  image_profile_label="$(
    docker image inspect --format '{{index .Config.Labels "qwen38.runtime.profile"}}' \
      "${IMAGE_TAG}"
  )"
  [[ "${image_profile_label}" == "${PROFILE_VERSION}" ]] || \
    die "Runtime image profile label does not match." \
      "Expected: ${PROFILE_VERSION}" "Found: ${image_profile_label:-missing}"

  model_revision_label="$(
    docker inspect --format '{{index .Config.Labels "qwen38.model.revision"}}' \
      "${CONTAINER_NAME}"
  )"
  [[ "${model_revision_label}" == "${MODEL_REVISION}" ]] || \
    die "Running container model-revision label does not match." \
      "Expected: ${MODEL_REVISION}" \
      "Found:    ${model_revision_label:-missing}"

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
  cache_model_revision_label="$(
    docker volume inspect --format '{{index .Labels "qwen38.model.revision"}}' \
      "${CACHE_VOLUME}"
  )"
  [[ "${cache_project_label}" == "${CONTAINER_LABEL}" && \
     "${cache_profile_label}" == "${PROFILE_VERSION}" && \
     "${cache_model_revision_label}" == "${MODEL_REVISION}" ]] || \
    die "Pinned cache-volume labels do not match." \
      "Expected project/profile/revision: ${CONTAINER_LABEL}/${PROFILE_VERSION}/${MODEL_REVISION}" \
      "Found project/profile/revision: ${cache_project_label:-missing}/${cache_profile_label:-missing}/${cache_model_revision_label:-missing}"

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

  installed_report="$(
    docker exec "${CONTAINER_NAME}" sha256sum \
      /usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/turboquant_attn.py \
      /usr/local/lib/python3.12/dist-packages/vllm/tool_parsers/structural_tag_registry.py \
      /usr/local/lib/python3.12/dist-packages/vllm/config/model.py \
      /usr/local/lib/python3.12/dist-packages/vllm/entrypoints/anthropic/protocol.py \
      /usr/local/lib/python3.12/dist-packages/vllm/entrypoints/anthropic/serving.py \
      /usr/local/lib/python3.12/dist-packages/vllm/entrypoints/openai/chat_completion/protocol.py \
      /usr/local/lib/python3.12/dist-packages/vllm/sampling_params.py \
      /usr/local/lib/python3.12/dist-packages/vllm/v1/core/sched/utils.py \
      /usr/local/lib/python3.12/dist-packages/vllm/v1/engine/input_processor.py \
      /usr/local/lib/python3.12/dist-packages/vllm/v1/request.py \
      /usr/local/lib/python3.12/dist-packages/vllm/parser/qwen3.py \
      /usr/local/lib/python3.12/dist-packages/vllm/v1/structured_output/__init__.py \
      /usr/local/lib/python3.12/dist-packages/vllm/entrypoints/anthropic/api_router.py \
      /usr/local/lib/python3.12/dist-packages/vllm/entrypoints/openai/chat_completion/serving.py \
      /usr/local/lib/python3.12/dist-packages/vllm/entrypoints/openai/responses/context.py \
      /usr/local/lib/python3.12/dist-packages/vllm/entrypoints/openai/responses/protocol.py \
      /usr/local/lib/python3.12/dist-packages/vllm/entrypoints/openai/responses/serving.py \
      /usr/local/lib/python3.12/dist-packages/vllm/entrypoints/openai/responses/streaming_events.py \
      /usr/local/lib/python3.12/dist-packages/vllm/entrypoints/openai/responses/utils.py \
      /usr/local/lib/python3.12/dist-packages/vllm/parser/engine/parser_engine.py \
      /opt/qwen38/chat_template.jinja \
      /opt/qwen38/phase_budget_unit.py
  )"
  expected_installed_report="$(printf '%s  %s\n%s  %s\n%s  %s\n%s  %s\n%s  %s\n%s  %s\n%s  %s\n%s  %s\n%s  %s\n%s  %s\n%s  %s\n%s  %s\n%s  %s\n%s  %s\n%s  %s\n%s  %s\n%s  %s\n%s  %s\n%s  %s\n%s  %s\n%s  %s\n%s  %s' \
    "${TURBOQUANT_PATCHED_FILE_SHA256}" /usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/turboquant_attn.py \
    "${TOOL_SCHEMA_PATCHED_FILE_SHA256}" /usr/local/lib/python3.12/dist-packages/vllm/tool_parsers/structural_tag_registry.py \
    "${MODEL_CONFIG_PATCHED_FILE_SHA256}" /usr/local/lib/python3.12/dist-packages/vllm/config/model.py \
    "${ANTHROPIC_PROTOCOL_PATCHED_FILE_SHA256}" /usr/local/lib/python3.12/dist-packages/vllm/entrypoints/anthropic/protocol.py \
    "${ANTHROPIC_SERVING_PATCHED_FILE_SHA256}" /usr/local/lib/python3.12/dist-packages/vllm/entrypoints/anthropic/serving.py \
    "${CHAT_PROTOCOL_PATCHED_FILE_SHA256}" /usr/local/lib/python3.12/dist-packages/vllm/entrypoints/openai/chat_completion/protocol.py \
    "${SAMPLING_PARAMS_PATCHED_FILE_SHA256}" /usr/local/lib/python3.12/dist-packages/vllm/sampling_params.py \
    "${SCHED_UTILS_PATCHED_FILE_SHA256}" /usr/local/lib/python3.12/dist-packages/vllm/v1/core/sched/utils.py \
    "${INPUT_PROCESSOR_PATCHED_FILE_SHA256}" /usr/local/lib/python3.12/dist-packages/vllm/v1/engine/input_processor.py \
    "${REQUEST_PATCHED_FILE_SHA256}" /usr/local/lib/python3.12/dist-packages/vllm/v1/request.py \
    "${QWEN3_PARSER_PATCHED_FILE_SHA256}" /usr/local/lib/python3.12/dist-packages/vllm/parser/qwen3.py \
    "${STRUCTURED_OUTPUT_PATCHED_FILE_SHA256}" /usr/local/lib/python3.12/dist-packages/vllm/v1/structured_output/__init__.py \
    "${ANTHROPIC_API_ROUTER_PATCHED_FILE_SHA256}" /usr/local/lib/python3.12/dist-packages/vllm/entrypoints/anthropic/api_router.py \
    "${CHAT_SERVING_PATCHED_FILE_SHA256}" /usr/local/lib/python3.12/dist-packages/vllm/entrypoints/openai/chat_completion/serving.py \
    "${RESPONSES_CONTEXT_PATCHED_FILE_SHA256}" /usr/local/lib/python3.12/dist-packages/vllm/entrypoints/openai/responses/context.py \
    "${RESPONSES_PROTOCOL_PATCHED_FILE_SHA256}" /usr/local/lib/python3.12/dist-packages/vllm/entrypoints/openai/responses/protocol.py \
    "${RESPONSES_SERVING_PATCHED_FILE_SHA256}" /usr/local/lib/python3.12/dist-packages/vllm/entrypoints/openai/responses/serving.py \
    "${RESPONSES_STREAMING_PATCHED_FILE_SHA256}" /usr/local/lib/python3.12/dist-packages/vllm/entrypoints/openai/responses/streaming_events.py \
    "${RESPONSES_UTILS_PATCHED_FILE_SHA256}" /usr/local/lib/python3.12/dist-packages/vllm/entrypoints/openai/responses/utils.py \
    "${PARSER_ENGINE_PATCHED_FILE_SHA256}" /usr/local/lib/python3.12/dist-packages/vllm/parser/engine/parser_engine.py \
    "${AGENT_CHAT_TEMPLATE_SHA256}" /opt/qwen38/chat_template.jinja \
    "${PHASE_BUDGET_UNIT_SHA256}" /opt/qwen38/phase_budget_unit.py)"
  [[ "${installed_report}" == "${expected_installed_report}" ]] || \
    die "Running source/template bytes do not match the reviewed profile." \
      "Expected:" "${expected_installed_report}" \
      "Found:" "${installed_report}"

  additional_installed_report="$(
    docker exec "${CONTAINER_NAME}" sha256sum \
      /usr/local/lib/python3.12/dist-packages/vllm/v1/worker/workspace.py \
      /usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu_model_runner.py \
      /usr/local/lib/python3.12/dist-packages/vllm/entrypoints/serve/utils/api_utils.py \
      /usr/local/lib/python3.12/dist-packages/vllm/envs.py \
      /usr/local/lib/python3.12/dist-packages/vllm/entrypoints/chat_utils.py \
      /usr/local/lib/python3.12/dist-packages/vllm/multimodal/media/connector.py \
      /usr/local/lib/python3.12/dist-packages/vllm/multimodal/media/image.py \
      /usr/local/lib/python3.12/dist-packages/vllm/renderers/params.py \
      /usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/qwen3_vl.py \
      /opt/qwen38/vision_workspace_unit.py \
      /opt/qwen38/vision_contract_unit.py \
      /opt/qwen38/vision_mlp_unit.py
  )"
  expected_additional_installed_report="$(printf '%s  %s\n%s  %s\n%s  %s\n%s  %s\n%s  %s\n%s  %s\n%s  %s\n%s  %s\n%s  %s\n%s  %s\n%s  %s\n%s  %s' \
    "${WORKSPACE_PATCHED_FILE_SHA256}" /usr/local/lib/python3.12/dist-packages/vllm/v1/worker/workspace.py \
    "${GPU_MODEL_RUNNER_PATCHED_FILE_SHA256}" /usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu_model_runner.py \
    "${API_UTILS_PATCHED_FILE_SHA256}" /usr/local/lib/python3.12/dist-packages/vllm/entrypoints/serve/utils/api_utils.py \
    "${ENVS_PATCHED_FILE_SHA256}" /usr/local/lib/python3.12/dist-packages/vllm/envs.py \
    "${CHAT_UTILS_PATCHED_FILE_SHA256}" /usr/local/lib/python3.12/dist-packages/vllm/entrypoints/chat_utils.py \
    "${MEDIA_CONNECTOR_PATCHED_FILE_SHA256}" /usr/local/lib/python3.12/dist-packages/vllm/multimodal/media/connector.py \
    "${IMAGE_MEDIA_PATCHED_FILE_SHA256}" /usr/local/lib/python3.12/dist-packages/vllm/multimodal/media/image.py \
    "${RENDER_PARAMS_PATCHED_FILE_SHA256}" /usr/local/lib/python3.12/dist-packages/vllm/renderers/params.py \
    "${QWEN3_VL_MODEL_PATCHED_FILE_SHA256}" /usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/qwen3_vl.py \
    "${VISION_WORKSPACE_UNIT_SHA256}" /opt/qwen38/vision_workspace_unit.py \
    "${VISION_CONTRACT_UNIT_SHA256}" /opt/qwen38/vision_contract_unit.py \
    "${VISION_MLP_UNIT_SHA256}" /opt/qwen38/vision_mlp_unit.py)"
  [[ "${additional_installed_report}" == "${expected_additional_installed_report}" ]] || \
    die "Running vision/runtime bytes do not match the reviewed profile." \
      "Expected:" "${expected_additional_installed_report}" \
      "Found:" "${additional_installed_report}"

  assert_runtime_versions
}

print_healthy_summary() {
  printf '\nHEALTHY — the one supported serving profile is correct.\n'
  printf 'Endpoint:      %s\n' "${ENDPOINT}"
  printf 'Listener:      %s:%s only\n' "${LISTEN_HOST}" "${LISTEN_PORT}"
  printf 'Model:         %s\n' "${SERVED_MODEL}"
  printf 'Model revision: %s\n' "${MODEL_REVISION}"
  printf 'Context limit: %s tokens\n' "${MAX_MODEL_LEN}"
  printf 'Vision:         15 inline static PNG images; 16,777,216 pixels each; videos disabled\n'
  printf 'Image quality:  BF16 vision tower; <=30:1 proven aspect ratio; no request overrides\n'
  printf 'Thinking:      xhigh; old traces omitted by default\n'
  printf 'Sampling:      explicit Qwen3.8 defaults; repetition penalty 1.0\n'
  printf 'Phase ceilings: reasoning 262144; final response 131072 tokens\n'
  printf 'Runtime image: %s\n' "${EXPECTED_IMAGE_ID}"
  printf 'To stop it:    ./stop.sh\n'
}
