#!/usr/bin/env bash

COMMON_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${COMMON_SCRIPT_DIR}/.." && pwd)"
# shellcheck source=../config/runtime-v1.sh
source "${PROJECT_DIR}/config/runtime-v1.sh"
readonly COMMON_SCRIPT_DIR PROJECT_DIR
readonly MODEL_DIR="${PROJECT_DIR}/${MODEL_DIR_NAME}"
readonly MODEL_MANIFEST="${PROJECT_DIR}/manifests/${MODEL_MANIFEST_NAME}"
readonly DEPLOYMENT_INPUT_MANIFEST="${PROJECT_DIR}/config/deployment-inputs.sha256"

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

require_equal() {
  local description="$1" observed="$2" expected="$3"
  if [[ "${observed}" != "${expected}" ]]; then
    die "${description} differs from the pinned contract." \
      "Expected: ${expected}" \
      "Found:    ${observed:-<empty>}"
  fi
}

capture_child_wait_status() {
  local child_pid="${1:-}" output_name="${2:-}" wait_status
  if (($# != 2)) || [[ ! "${child_pid}" =~ ^[1-9][0-9]*$ ]] || \
      [[ ! "${output_name}" =~ ^[a-zA-Z_][a-zA-Z0-9_]*$ ]]; then
    die "capture_child_wait_status requires one numeric child PID and one shell variable name."
  fi

  # `wait` commonly returns 128+signal after an intentional termination.  It
  # must be a conditional command: merely disabling `errexit` does not suppress
  # an installed ERR trap.  The caller owns the meaning of the captured status.
  if wait "${child_pid}"; then
    wait_status=0
  else
    wait_status=$?
  fi
  printf -v "${output_name}" '%s' "${wait_status}"
}

check_host_prerequisites() {
  # Functional requirements only. The tools this deployment actually invokes
  # must exist, Docker must respond with its NVIDIA runtime configured, the
  # container-isolation features the profile depends on must be active, and
  # exactly one GPU with at least the memory the locked KV/VRAM budget was
  # calibrated for must be present. Exact host software versions, binary
  # hashes, and GPU/driver identity are deliberately not asserted: they tie
  # the deployment to one specific computer without making inference any
  # more correct. Everything inside the pinned images remains exact.
  local command_name docker_server runtimes
  local gpu_report gpu_count gpu_memory
  for command_name in docker git nvidia-smi sha256sum ss; do
    require_command "${command_name}"
  done

  docker_server="$(docker version --format '{{.Server.Version}}')" ||
    die "Docker server is not responding."
  [[ -n "${docker_server}" ]] || die "Docker server reported an empty version."
  require_equal "Docker security options" \
    "$(docker info --format '{{json .SecurityOptions}}')" \
    "${EXPECTED_DOCKER_SECURITY_OPTIONS}"

  runtimes="$(docker info --format '{{json .Runtimes}}')"
  if [[ "${runtimes}" != *'"nvidia"'* ]]; then
    die "Docker's NVIDIA runtime is not configured."
  fi

  gpu_report="$(
    nvidia-smi \
      --query-gpu=memory.total \
      --format=csv,noheader,nounits
  )"
  gpu_count="$(wc -l <<<"${gpu_report}")"
  require_equal "GPU count" "${gpu_count}" "1"
  gpu_memory="${gpu_report//[[:space:]]/}"
  [[ "${gpu_memory}" =~ ^[0-9]+$ ]] || die "nvidia-smi reported a non-numeric GPU memory total: ${gpu_report}"
  if (( gpu_memory < MINIMUM_GPU_MEMORY_MIB )); then
    die "GPU memory is below the locked VRAM budget's calibration floor." \
      "Required: at least ${MINIMUM_GPU_MEMORY_MIB} MiB" \
      "Found:    ${gpu_memory} MiB"
  fi
}

require_clean_committed_repository() {
  local repository_status repository_branch
  repository_status="$(
    git -C "${PROJECT_DIR}" status \
      --porcelain=v1 \
      --untracked-files=all \
      --ignore-submodules=dirty
  )"
  if [[ -n "${repository_status}" ]]; then
    die "The backend deployment repository has uncommitted or untracked inputs." \
      "Repository: ${PROJECT_DIR}" \
      "Status:" \
      "${repository_status}" \
      "Commit the exact reviewed release inputs before operating the stack."
  fi
  git -C "${PROJECT_DIR}" diff --quiet --exit-code || \
    die "Tracked backend files differ from HEAD."
  git -C "${PROJECT_DIR}" diff --cached --quiet --exit-code || \
    die "The backend index differs from HEAD."
  repository_branch="$(git -C "${PROJECT_DIR}" symbolic-ref --quiet --short HEAD)" || \
    die "The backend deployment repository is detached." \
      "The only supported release branch is master."
  if [[ "${repository_branch}" != "master" ]]; then
    die "The backend deployment repository is on an unsupported branch." \
      "Expected: master" \
      "Found:    ${repository_branch}"
  fi
}

require_published_release() {
  local expected_remote actual_fetch_remote actual_push_remote repository_head published
  require_clean_committed_repository
  expected_remote="https://github.com/BigBIueWhale/qwen_38_agent_setup"
  actual_fetch_remote="$(git -C "${PROJECT_DIR}" remote get-url origin)" || \
    die "The exact origin fetch remote is unavailable."
  actual_push_remote="$(git -C "${PROJECT_DIR}" remote get-url --push origin)" || \
    die "The exact origin push remote is unavailable."
  require_equal "origin fetch remote" "${actual_fetch_remote}" "${expected_remote}"
  require_equal "origin push remote" "${actual_push_remote}" "${expected_remote}"
  repository_head="$(git -C "${PROJECT_DIR}" rev-parse --verify HEAD)"
  published="$(git -C "${PROJECT_DIR}" ls-remote --exit-code origin refs/heads/master | awk 'NR == 1 {print $1}')" || \
    die "Could not query the exact GitHub master ref for the publication audit."
  [[ "${published}" =~ ^[0-9a-f]{40}$ ]] || \
    die "The queried GitHub master ref is not one exact commit: ${published:-<empty>}"
  require_equal "published GitHub master release" "${published}" "${repository_head}"
}

check_pinned_build_inputs() {
  require_clean_committed_repository
  if [[ ! -f "${DEPLOYMENT_INPUT_MANIFEST}" || -L "${DEPLOYMENT_INPUT_MANIFEST}" ]]; then
    die "The deployment-input manifest is missing or is not a regular non-symlink file." \
      "Expected: ${DEPLOYMENT_INPUT_MANIFEST}"
  fi
  if [[ "$(wc -l <"${DEPLOYMENT_INPUT_MANIFEST}")" != "69" ]]; then
    die "The deployment-input manifest does not contain the exact 69-file allowlist." \
      "Manifest: ${DEPLOYMENT_INPUT_MANIFEST}"
  fi
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
  local relay_image_id relay_profile relay_component relay_source relay_sandbox
  relay_image_id="$(docker image inspect --format '{{.Id}}' "${RELAY_IMAGE_TAG}" 2>/dev/null || true)"
  [[ "${relay_image_id}" == "${EXPECTED_RELAY_IMAGE_ID}" ]] || \
    die "The exact fixed-relay image is missing or incorrect." \
      "Expected tag: ${RELAY_IMAGE_TAG}" \
      "Expected ID:  ${EXPECTED_RELAY_IMAGE_ID}" \
      "Found ID:     ${relay_image_id:-nothing}" \
      "Build the pinned paired agent_service repository before starting this component."
  relay_profile="$(docker image inspect --format '{{index .Config.Labels "agent_service.profile"}}' "${RELAY_IMAGE_TAG}")"
  relay_component="$(docker image inspect --format '{{index .Config.Labels "agent_service.component"}}' "${RELAY_IMAGE_TAG}")"
  relay_source="$(docker image inspect --format '{{index .Config.Labels "agent_service.relay.source.sha256"}}' "${RELAY_IMAGE_TAG}")"
  relay_sandbox="$(docker image inspect --format '{{index .Config.Labels "agent_service.relay.sandbox"}}' "${RELAY_IMAGE_TAG}")"
  [[ "${relay_profile}" == "${AGENT_SERVICE_PROFILE}" && \
     "${relay_component}" == "fixed-relay" && \
     "${relay_source}" == "${RELAY_SOURCE_SHA256}" && \
     "${relay_sandbox}" == "${RELAY_SANDBOX}" ]] || \
    die "The fixed-relay image labels differ from the paired contract." \
      "Expected profile/component/source/sandbox: ${AGENT_SERVICE_PROFILE}/fixed-relay/${RELAY_SOURCE_SHA256}/${RELAY_SANDBOX}" \
      "Found profile/component/source/sandbox: ${relay_profile:-missing}/${relay_component:-missing}/${relay_source:-missing}/${relay_sandbox:-missing}"

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
    cd -- "${MODEL_DIR}" || \
      die "Could not enter the pinned model directory for manifest verification." \
        "Directory: ${MODEL_DIR}"
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

relay_container_exists() {
  docker container inspect "$1" >/dev/null 2>&1
}

assert_owned_model_relay() {
  local name="$1" component="$2" profile observed_component image configured_image command
  profile="$(docker inspect --format '{{index .Config.Labels "agent_service.profile"}}' "${name}")"
  observed_component="$(docker inspect --format '{{index .Config.Labels "agent_service.component"}}' "${name}")"
  image="$(docker inspect --format '{{.Image}}' "${name}")"
  configured_image="$(docker inspect --format '{{.Config.Image}}' "${name}")"
  command="$(docker inspect --format '{{json .Config.Cmd}}' "${name}")"
  [[ "${profile}" == "${AGENT_SERVICE_PROFILE}" && \
     "${observed_component}" == "${component}" && \
     "${image}" == "${EXPECTED_RELAY_IMAGE_ID}" && \
     "${configured_image}" == "${EXPECTED_RELAY_IMAGE_ID}" && \
     "${command}" == "[\"${component}\"]" ]] || \
    die "Refusing to modify unrecognized fixed relay container ${name}." \
      "Expected profile/component/image/configured-image/command: ${AGENT_SERVICE_PROFILE}/${component}/${EXPECTED_RELAY_IMAGE_ID}/${EXPECTED_RELAY_IMAGE_ID}/[\"${component}\"]" \
      "Found profile/component/image/configured-image/command: ${profile:-missing}/${observed_component:-missing}/${image:-missing}/${configured_image:-missing}/${command:-missing}"
}

assert_owned_container() {
  local project_label profile_label image configured_image
  project_label="$(
    docker inspect --format '{{index .Config.Labels "qwen38.project"}}' \
      "${CONTAINER_NAME}"
  )"
  profile_label="$(
    docker inspect --format '{{index .Config.Labels "qwen38.runtime.profile"}}' \
      "${CONTAINER_NAME}"
  )"
  image="$(docker inspect --format '{{.Image}}' "${CONTAINER_NAME}")"
  configured_image="$(docker inspect --format '{{.Config.Image}}' "${CONTAINER_NAME}")"
  if [[ "${project_label}" != "${CONTAINER_LABEL}" || \
        "${profile_label}" != "${PROFILE_VERSION}" || \
        "${image}" != "${EXPECTED_IMAGE_ID}" || \
        "${configured_image}" != "${EXPECTED_IMAGE_ID}" ]]; then
    die "A container named ${CONTAINER_NAME} exists but is not the exact owned profile." \
      "Project label: ${project_label:-missing}" \
      "Profile label: ${profile_label:-missing}" \
      "Image/configured image: ${image:-missing}/${configured_image:-missing}" \
      "It was not modified or removed."
  fi
}

assert_model_relay_profile() {
  local backend_id name expected_role expected_network expected_rw expected_component expected_event
  local running image configured_image user network read_only command
  local cap_drop cap_add security memory memory_swap pids bindings mount_count
  local mount_source mount_destination mount_rw profile component
  local privileged restart apparmor devices device_requests pid_mode ipc_mode uts_mode pid status_file
  backend_id="$(docker inspect --format '{{.Id}}' "${CONTAINER_NAME}")"
  for name in "${MODEL_BRIDGE_NAME}" "${MODEL_INGRESS_NAME}"; do
    relay_container_exists "${name}" || \
      die "Required fixed model relay is absent: ${name}"
    if [[ "${name}" == "${MODEL_BRIDGE_NAME}" ]]; then
      expected_role=model-bridge
      expected_network="container:${backend_id}"
      expected_rw=true
      expected_component=model-bridge
      expected_event="RELAY_READY role=model-bridge sandbox=${RELAY_SANDBOX} listen=unix:/sock/relay.sock target=tcp:127.0.0.1:8000"
    else
      expected_role=model-ingress
      expected_network=host
      expected_rw=false
      expected_component=model-ingress
      expected_event="RELAY_READY role=model-ingress sandbox=${RELAY_SANDBOX} listen=tcp:127.0.0.1:8000 target=unix:/sock/relay.sock"
    fi
    running="$(docker inspect --format '{{.State.Running}}' "${name}")"
    image="$(docker inspect --format '{{.Image}}' "${name}")"
    configured_image="$(docker inspect --format '{{.Config.Image}}' "${name}")"
    user="$(docker inspect --format '{{.Config.User}}' "${name}")"
    network="$(docker inspect --format '{{.HostConfig.NetworkMode}}' "${name}")"
    read_only="$(docker inspect --format '{{.HostConfig.ReadonlyRootfs}}' "${name}")"
    command="$(docker inspect --format '{{json .Config.Cmd}}' "${name}")"
    privileged="$(docker inspect --format '{{.HostConfig.Privileged}}' "${name}")"
    restart="$(docker inspect --format '{{.HostConfig.RestartPolicy.Name}}' "${name}")"
    cap_drop="$(docker inspect --format '{{json .HostConfig.CapDrop}}' "${name}")"
    cap_add="$(docker inspect --format '{{json .HostConfig.CapAdd}}' "${name}")"
    security="$(docker inspect --format '{{json .HostConfig.SecurityOpt}}' "${name}")"
    apparmor="$(docker inspect --format '{{.AppArmorProfile}}' "${name}")"
    devices="$(docker inspect --format '{{json .HostConfig.Devices}}' "${name}")"
    device_requests="$(docker inspect --format '{{json .HostConfig.DeviceRequests}}' "${name}")"
    pid_mode="$(docker inspect --format '{{.HostConfig.PidMode}}' "${name}")"
    ipc_mode="$(docker inspect --format '{{.HostConfig.IpcMode}}' "${name}")"
    uts_mode="$(docker inspect --format '{{.HostConfig.UTSMode}}' "${name}")"
    memory="$(docker inspect --format '{{.HostConfig.Memory}}' "${name}")"
    memory_swap="$(docker inspect --format '{{.HostConfig.MemorySwap}}' "${name}")"
    pids="$(docker inspect --format '{{.HostConfig.PidsLimit}}' "${name}")"
    bindings="$(docker inspect --format '{{json .HostConfig.PortBindings}}' "${name}")"
    mount_count="$(docker inspect --format '{{len .Mounts}}' "${name}")"
    mount_source="$(docker inspect --format '{{(index .Mounts 0).Source}}' "${name}")"
    mount_destination="$(docker inspect --format '{{(index .Mounts 0).Destination}}' "${name}")"
    mount_rw="$(docker inspect --format '{{(index .Mounts 0).RW}}' "${name}")"
    profile="$(docker inspect --format '{{index .Config.Labels "agent_service.profile"}}' "${name}")"
    component="$(docker inspect --format '{{index .Config.Labels "agent_service.component"}}' "${name}")"
    [[ "${running}" == true && "${image}" == "${EXPECTED_RELAY_IMAGE_ID}" && \
       "${configured_image}" == "${EXPECTED_RELAY_IMAGE_ID}" && \
       "${user}" == 1000:1000 && "${network}" == "${expected_network}" && \
       "${read_only}" == true && "${command}" == "[\"${expected_role}\"]" && \
       "${privileged}" == false && "${restart}" == no && \
       "${cap_drop}" == '["ALL"]' && "${cap_add}" == null && \
       "${security}" == '["no-new-privileges:true"]' && \
       "${apparmor}" == "${EXPECTED_CONTAINER_APPARMOR_PROFILE}" && \
       "${devices}" == '[]' && "${device_requests}" == null && \
       -z "${pid_mode}" && "${ipc_mode}" == private && -z "${uts_mode}" && \
       "${memory}" == 33554432 && "${memory_swap}" == 33554432 && \
       "${pids}" == "${RELAY_PIDS_LIMIT}" && "${bindings}" == '{}' && \
       "${mount_count}" == 1 && "${mount_source}" == "${MODEL_SOCKET_DIR}" && \
       "${mount_destination}" == /sock && "${mount_rw}" == "${expected_rw}" && \
       "${profile}" == "${AGENT_SERVICE_PROFILE}" && \
       "${component}" == "${expected_component}" ]] || \
      die "Fixed relay ${name} differs from its exact ${expected_role} contract." \
        "running/image/configured-image: ${running}/${image}/${configured_image}" \
        "user/network/root/command: ${user}/${network}/${read_only}/${command}" \
        "cap/security/memory/swap/pids: ${cap_drop}/${security}/${memory}/${memory_swap}/${pids}" \
        "bindings/mount: ${bindings}/${mount_count}:${mount_source}:${mount_destination}:${mount_rw}" \
        "labels: ${profile}/${component}"
    pid="$(docker inspect --format '{{.State.Pid}}' "${name}")"
    [[ "${pid}" =~ ^[1-9][0-9]*$ ]] || \
      die "Fixed relay ${name} has an invalid host PID: ${pid}"
    status_file="/proc/${pid}/status"
    [[ -r "${status_file}" ]] || \
      die "Cannot inspect fixed relay kernel sandbox status: ${status_file}"
    require_equal "${name} kernel no_new_privs" \
      "$(awk '$1 == "NoNewPrivs:" {print $2}' "${status_file}")" 1
    require_equal "${name} kernel seccomp mode" \
      "$(awk '$1 == "Seccomp:" {print $2}' "${status_file}")" 2
    # One filter is Docker's pinned builtin profile; the relay stacks its
    # socket-domain and no-new-bind filters on top of it.
    require_equal "${name} stacked seccomp filter count" \
      "$(awk '$1 == "Seccomp_filters:" {print $2}' "${status_file}")" 3
    require_equal "${name} exact sandbox readiness count" \
      "$(docker logs "${name}" 2>&1 | grep --fixed-strings --line-regexp --count "${expected_event}" || true)" 1
  done
  [[ -d "${MODEL_SOCKET_DIR}" && ! -L "${MODEL_SOCKET_DIR}" && \
     -S "${MODEL_SOCKET_DIR}/relay.sock" ]] || \
    die "Central model Unix socket is absent or unsafe: ${MODEL_SOCKET_DIR}/relay.sock"
  require_equal "central model socket owner/mode" \
    "$(stat -c '%u:%g:%a' "${MODEL_SOCKET_DIR}/relay.sock")" "1000:1000:660"
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

assert_backend_network_none_proc() {
  local pid route_file ipv6_route_file dev_file interfaces namespace host_namespace
  pid="$(docker inspect --format '{{.State.Pid}}' "${CONTAINER_NAME}")"
  [[ "${pid}" =~ ^[1-9][0-9]*$ ]] || \
    die "The network-none backend has an invalid host PID: ${pid}"
  # Reading /proc/<other-uid>/ns/net is restricted on this host.  Coreutils is
  # pinned in the runtime, so read the same namespace inode from within the
  # container and compare it to the host namespace.  Route/interface contents
  # remain independently inspected through the host's /proc/<pid>/net view.
  namespace="$(docker exec "${CONTAINER_NAME}" \
    /usr/bin/readlink /proc/self/ns/net)"
  host_namespace="$(readlink /proc/self/ns/net)"
  [[ -n "${namespace}" && "${namespace}" != "${host_namespace}" ]] || \
    die "The backend does not have a distinct network namespace." \
      "Backend namespace: ${namespace:-unreadable}" \
      "Host namespace:    ${host_namespace:-unreadable}"
  route_file="/proc/${pid}/net/route"
  ipv6_route_file="/proc/${pid}/net/ipv6_route"
  dev_file="/proc/${pid}/net/dev"
  [[ -r "${route_file}" && "$(wc -l <"${route_file}")" == 1 ]] || \
    die "The backend network namespace has an unexpected IPv4 route table."
  if [[ -e "${ipv6_route_file}" && \
        ( ! -r "${ipv6_route_file}" || -s "${ipv6_route_file}" ) ]]; then
    die "The backend network namespace has an unexpected IPv6 route table."
  fi
  [[ -r "${dev_file}" ]] || \
    die "Cannot inspect backend network devices: ${dev_file}"
  interfaces="$(awk -F: 'NR > 2 {gsub(/[[:space:]]/, "", $1); if ($1 != "") print $1}' "${dev_file}")"
  require_equal "backend network-none interfaces" "${interfaces}" lo
}

assert_running_profile() {
  local running image_id network port_bindings published_ports cap_drop cap_add security_opts
  local privileged restart apparmor devices device_requests pid_mode ipc_mode uts_mode shm_size mount_count
  local read_only_root tmp_tmpfs run_tmpfs
  local model_source model_rw model_revision_label model_official_revision_label
  local model_correction_label model_sha256_label model_manifest_sha256_label
  local cache_name cache_type cache_rw cache_label_count cache_project_label cache_profile_label
  local cache_model_revision_label cache_model_correction_label cache_model_sha256_label
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
  require_equal "running backend configured immutable image ID" \
    "$(docker inspect --format '{{.Config.Image}}' "${CONTAINER_NAME}")" \
    "${EXPECTED_IMAGE_ID}"
  require_equal "running backend non-root user" \
    "$(docker inspect --format '{{.Config.User}}' "${CONTAINER_NAME}")" 2000:0
  image_profile_label="$(
    docker image inspect --format '{{index .Config.Labels "qwen38.runtime.profile"}}' \
      "${IMAGE_TAG}"
  )"
  [[ "${image_profile_label}" == "${IMAGE_PROFILE_VERSION}" ]] || \
    die "Runtime image profile label does not match." \
      "Expected: ${IMAGE_PROFILE_VERSION}" "Found: ${image_profile_label:-missing}"

  model_revision_label="$(
    docker inspect --format '{{index .Config.Labels "qwen38.model.revision"}}' \
      "${CONTAINER_NAME}"
  )"
  [[ "${model_revision_label}" == "${MODEL_REVISION}" ]] || \
    die "Running container model-revision label does not match." \
      "Expected: ${MODEL_REVISION}" \
      "Found:    ${model_revision_label:-missing}"
  model_official_revision_label="$(
    docker inspect --format '{{index .Config.Labels "qwen38.model.official-revision"}}' \
      "${CONTAINER_NAME}"
  )"
  model_correction_label="$(
    docker inspect --format '{{index .Config.Labels "qwen38.model.correction"}}' \
      "${CONTAINER_NAME}"
  )"
  model_sha256_label="$(
    docker inspect --format '{{index .Config.Labels "qwen38.model.sha256"}}' \
      "${CONTAINER_NAME}"
  )"
  model_manifest_sha256_label="$(
    docker inspect --format '{{index .Config.Labels "qwen38.model.manifest.sha256"}}' \
      "${CONTAINER_NAME}"
  )"
  [[ "${model_official_revision_label}" == "${OFFICIAL_MODEL_REVISION}" && \
     "${model_correction_label}" == "${MODEL_CORRECTION}" && \
     "${model_sha256_label}" == "${MODEL_SHA256}" && \
     "${model_manifest_sha256_label}" == "${MODEL_MANIFEST_SHA256}" ]] || \
    die "Running container corrected-model labels do not match." \
      "Expected official revision/correction/model/manifest: ${OFFICIAL_MODEL_REVISION}/${MODEL_CORRECTION}/${MODEL_SHA256}/${MODEL_MANIFEST_SHA256}" \
      "Found official revision/correction/model/manifest: ${model_official_revision_label:-missing}/${model_correction_label:-missing}/${model_sha256_label:-missing}/${model_manifest_sha256_label:-missing}"

  network="$(docker inspect --format '{{.HostConfig.NetworkMode}}' "${CONTAINER_NAME}")"
  [[ "${network}" == "none" ]] || \
    die "Running container has the wrong network mode." \
      "Expected network-none isolation; found: ${network}"
  assert_backend_network_none_proc

  port_bindings="$(docker inspect --format '{{json .HostConfig.PortBindings}}' "${CONTAINER_NAME}")"
  [[ "${port_bindings}" == "{}" ]] || \
    die "Docker port publishing is unexpectedly configured." \
      "Expected: {}" "Found: ${port_bindings}"
  published_ports="$(docker port "${CONTAINER_NAME}")"
  [[ -z "${published_ports}" ]] || \
    die "Docker reports published ports, which are forbidden." \
      "Found:" "${published_ports}"

  cap_drop="$(docker inspect --format '{{json .HostConfig.CapDrop}}' "${CONTAINER_NAME}")"
  cap_add="$(docker inspect --format '{{json .HostConfig.CapAdd}}' "${CONTAINER_NAME}")"
  security_opts="$(docker inspect --format '{{json .HostConfig.SecurityOpt}}' "${CONTAINER_NAME}")"
  privileged="$(docker inspect --format '{{.HostConfig.Privileged}}' "${CONTAINER_NAME}")"
  restart="$(docker inspect --format '{{.HostConfig.RestartPolicy.Name}}' "${CONTAINER_NAME}")"
  apparmor="$(docker inspect --format '{{.AppArmorProfile}}' "${CONTAINER_NAME}")"
  devices="$(docker inspect --format '{{json .HostConfig.Devices}}' "${CONTAINER_NAME}")"
  device_requests="$(docker inspect --format '{{json .HostConfig.DeviceRequests}}' "${CONTAINER_NAME}")"
  pid_mode="$(docker inspect --format '{{.HostConfig.PidMode}}' "${CONTAINER_NAME}")"
  ipc_mode="$(docker inspect --format '{{.HostConfig.IpcMode}}' "${CONTAINER_NAME}")"
  uts_mode="$(docker inspect --format '{{.HostConfig.UTSMode}}' "${CONTAINER_NAME}")"
  shm_size="$(docker inspect --format '{{.HostConfig.ShmSize}}' "${CONTAINER_NAME}")"
  [[ "${cap_drop}" == '["ALL"]' && "${cap_add}" == null && \
     "${security_opts}" == '["no-new-privileges:true"]' && \
     "${privileged}" == false && "${restart}" == no && \
     "${apparmor}" == "${EXPECTED_CONTAINER_APPARMOR_PROFILE}" && \
     "${devices}" == '[]' && \
     "${device_requests}" == '[{"Driver":"","Count":-1,"DeviceIDs":null,"Capabilities":[["gpu"]],"Options":{}}]' && \
     -z "${pid_mode}" && "${ipc_mode}" == private && -z "${uts_mode}" && \
     "${shm_size}" == 8589934592 ]] || \
    die "Container hardening, namespace, or GPU-device contract differs." \
      "cap-drop/cap-add/security: ${cap_drop}/${cap_add}/${security_opts}" \
      "privileged/restart/AppArmor: ${privileged}/${restart}/${apparmor}" \
      "devices/device-requests: ${devices}/${device_requests}" \
      "pid/ipc/uts/shm: ${pid_mode:-private}/${ipc_mode}/${uts_mode:-private}/${shm_size}"

  read_only_root="$(
    docker inspect --format '{{.HostConfig.ReadonlyRootfs}}' "${CONTAINER_NAME}"
  )"
  tmp_tmpfs="$(
    docker inspect --format '{{index .HostConfig.Tmpfs "/tmp"}}' "${CONTAINER_NAME}"
  )"
  run_tmpfs="$(
    docker inspect --format '{{index .HostConfig.Tmpfs "/run"}}' "${CONTAINER_NAME}"
  )"
  [[ "${read_only_root}" == "true" ]] || \
    die "Container root filesystem is writable." \
      "Expected read-only root; found: ${read_only_root:-missing}"
  [[ "$(docker inspect --format '{{len .HostConfig.Tmpfs}}' "${CONTAINER_NAME}")" == 2 && \
     "${tmp_tmpfs}" == "${TMP_TMPFS_OPTIONS}" && \
     "${run_tmpfs}" == "${RUN_TMPFS_OPTIONS}" ]] || \
    die "Container runtime tmpfs mounts differ from the exact profile." \
      "Expected /tmp: ${TMP_TMPFS_OPTIONS}" \
      "Found /tmp: ${tmp_tmpfs:-missing}" \
      "Expected /run: ${RUN_TMPFS_OPTIONS}" \
      "Found /run: ${run_tmpfs:-missing}"

  model_source="$(docker inspect --format '{{range .Mounts}}{{if eq .Destination "/model"}}{{.Source}}{{end}}{{end}}' "${CONTAINER_NAME}")"
  model_rw="$(docker inspect --format '{{range .Mounts}}{{if eq .Destination "/model"}}{{.RW}}{{end}}{{end}}' "${CONTAINER_NAME}")"
  mount_count="$(docker inspect --format '{{len .Mounts}}' "${CONTAINER_NAME}")"
  [[ "${mount_count}" == 2 && "${model_source}" == "${MODEL_DIR}" && "${model_rw}" == "false" ]] || \
    die "Model mount is not the exact read-only checkpoint mount." \
      "Expected mount count: 2" "Found mount count: ${mount_count}" \
      "Expected source: ${MODEL_DIR}" "Found source: ${model_source:-missing}" \
      "Expected writable: false" "Found writable: ${model_rw:-missing}"

  cache_name="$(docker inspect --format '{{range .Mounts}}{{if eq .Destination "/home/vllm/.cache/vllm"}}{{.Name}}{{end}}{{end}}' "${CONTAINER_NAME}")"
  cache_type="$(docker inspect --format '{{range .Mounts}}{{if eq .Destination "/home/vllm/.cache/vllm"}}{{.Type}}{{end}}{{end}}' "${CONTAINER_NAME}")"
  cache_rw="$(docker inspect --format '{{range .Mounts}}{{if eq .Destination "/home/vllm/.cache/vllm"}}{{.RW}}{{end}}{{end}}' "${CONTAINER_NAME}")"
  [[ "${cache_name}" == "${CACHE_VOLUME}" && "${cache_type}" == "volume" && \
     "${cache_rw}" == true ]] || \
    die "vLLM cache mount does not match the pinned named volume." \
      "Expected: writable volume ${CACHE_VOLUME}" \
      "Found: ${cache_type:-missing} ${cache_name:-missing}, writable=${cache_rw:-missing}"
  require_equal "mounted non-root vLLM cache owner/mode" \
    "$(docker exec "${CONTAINER_NAME}" stat -c '%u:%g:%a' /home/vllm/.cache/vllm)" \
    "2000:0:770"
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
  cache_model_correction_label="$(
    docker volume inspect --format '{{index .Labels "qwen38.model.correction"}}' \
      "${CACHE_VOLUME}"
  )"
  cache_model_sha256_label="$(
    docker volume inspect --format '{{index .Labels "qwen38.model.sha256"}}' \
      "${CACHE_VOLUME}"
  )"
  cache_label_count="$(docker volume inspect --format '{{len .Labels}}' "${CACHE_VOLUME}")"
  [[ "${cache_project_label}" == "${CONTAINER_LABEL}" && \
     "${cache_profile_label}" == "${PROFILE_VERSION}" && \
     "${cache_model_revision_label}" == "${MODEL_REVISION}" && \
     "${cache_model_correction_label}" == "${MODEL_CORRECTION}" && \
     "${cache_model_sha256_label}" == "${MODEL_SHA256}" && \
     "${cache_label_count}" == 5 ]] || \
    die "Pinned cache-volume labels do not match." \
      "Expected project/profile/revision/correction/model: ${CONTAINER_LABEL}/${PROFILE_VERSION}/${MODEL_REVISION}/${MODEL_CORRECTION}/${MODEL_SHA256}" \
      "Found project/profile/revision/correction/model/count: ${cache_project_label:-missing}/${cache_profile_label:-missing}/${cache_model_revision_label:-missing}/${cache_model_correction_label:-missing}/${cache_model_sha256_label:-missing}/${cache_label_count:-missing}"

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

  assert_model_relay_profile
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
