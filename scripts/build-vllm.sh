#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-build}"
EXPECTED_STATUS=$' M tests/entrypoints/anthropic/test_anthropic_messages_conversion.py\n M tests/entrypoints/serve/utils/test_api_utils.py\n M tests/entrypoints/unit_tests/test_chat_utils.py\n M tests/multimodal/media/test_connector.py\n M tests/multimodal/media/test_image.py\n M tests/quantization/test_turboquant.py\n M tests/v1/worker/test_gpu_model_runner_mm_gather.py\n M vllm/config/model.py\n M vllm/entrypoints/anthropic/api_router.py\n M vllm/entrypoints/anthropic/protocol.py\n M vllm/entrypoints/anthropic/serving.py\n M vllm/entrypoints/chat_utils.py\n M vllm/entrypoints/openai/chat_completion/protocol.py\n M vllm/entrypoints/openai/chat_completion/serving.py\n M vllm/entrypoints/openai/responses/context.py\n M vllm/entrypoints/openai/responses/protocol.py\n M vllm/entrypoints/openai/responses/serving.py\n M vllm/entrypoints/openai/responses/streaming_events.py\n M vllm/entrypoints/openai/responses/utils.py\n M vllm/entrypoints/serve/utils/api_utils.py\n M vllm/envs.py\n M vllm/model_executor/models/qwen3_vl.py\n M vllm/multimodal/media/connector.py\n M vllm/multimodal/media/image.py\n M vllm/parser/engine/parser_engine.py\n M vllm/parser/qwen3.py\n M vllm/renderers/params.py\n M vllm/sampling_params.py\n M vllm/tool_parsers/structural_tag_registry.py\n M vllm/v1/attention/backends/turboquant_attn.py\n M vllm/v1/attention/ops/triton_turboquant_decode.py\n M vllm/v1/attention/ops/triton_turboquant_store.py\n M vllm/v1/core/sched/utils.py\n M vllm/v1/engine/input_processor.py\n M vllm/v1/kv_offload/cpu/gpu_worker.py\n M vllm/v1/request.py\n M vllm/v1/structured_output/__init__.py\n M vllm/v1/worker/gpu_model_runner.py\n M vllm/v1/worker/workspace.py\n?? tests/v1/worker/test_workspace.py'
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=../config/runtime-v1.sh
source "${PROJECT_DIR}/config/runtime-v1.sh"
VLLM_DIR="${PROJECT_DIR}/vllm"
DOCKERFILE="${PROJECT_DIR}/containers/Dockerfile.runtime"
DOCKERIGNORE="${PROJECT_DIR}/.dockerignore"
TEMPLATE_FILE="${PROJECT_DIR}/chat_template.jinja"
PHASE_BUDGET_UNIT_FILE="${PROJECT_DIR}/scripts/phase_budget_unit.py"
VISION_WORKSPACE_UNIT_FILE="${PROJECT_DIR}/scripts/vision_workspace_unit.py"
VISION_CONTRACT_UNIT_FILE="${PROJECT_DIR}/scripts/vision_contract_unit.py"
VISION_MLP_UNIT_FILE="${PROJECT_DIR}/scripts/vision_mlp_unit.py"
TURBOQUANT_K8V4_UNIT_FILE="${PROJECT_DIR}/scripts/turboquant_k8v4_unit.py"
QWEN38_CONTEXT_UNIT_FILE="${PROJECT_DIR}/scripts/qwen38_context_unit.py"
NVFP4_KERNEL_UNIT_FILE="${PROJECT_DIR}/scripts/nvfp4_kernel_unit.py"
SOURCE_PATCH_DIR="${PROJECT_DIR}/patches/source_patch_v1"
SOURCE_PATCH_MANIFEST="${SOURCE_PATCH_DIR}/manifest.sha256"
DEPLOYMENT_INPUT_MANIFEST="${PROJECT_DIR}/config/deployment-inputs.sha256"
RUNTIME_COMMON_CONTRACT_TEST="${PROJECT_DIR}/scripts/test-runtime-common-contract.sh"

TURBOQUANT_PATCH_FILE="${PROJECT_DIR}/patches/vllm-turboquant-k8v4-direct-workspace.patch"
TOOL_SCHEMA_PATCH_FILE="${PROJECT_DIR}/patches/vllm-enforce-auto-tool-schema.patch"
AGENT_DEFAULTS_PATCH_FILE="${PROJECT_DIR}/patches/vllm-qwen38-agent-defaults-and-thinking.patch"
PHASE_BUDGET_PATCH_FILE="${PROJECT_DIR}/patches/vllm-qwen38-separate-final-response-budget.patch"
IMPLICIT_TOOL_GRAMMAR_PATCH_FILE="${PROJECT_DIR}/patches/vllm-qwen-implicit-tool-grammar-boundary.patch"
ANTHROPIC_VALIDATION_PATCH_FILE="${PROJECT_DIR}/patches/vllm-anthropic-validation-http400.patch"
TOOL_TRUNCATION_PATCH_FILE="${PROJECT_DIR}/patches/vllm-tool-truncation-finish-reason.patch"
VISION_RUNTIME_PATCH_FILE="${PROJECT_DIR}/patches/vllm-qwen38-vision-runtime.patch"
NUMERICAL_AUDITS_PATCH_FILE="${PROJECT_DIR}/patches/vllm-qwen38-numerical-audits.patch"
TURBOQUANT_GUARDS_PATCH_FILE="${PROJECT_DIR}/patches/vllm-turboquant-fail-closed-guards.patch"
KV_OFFLOAD_PINNING_PATCH_FILE="${PROJECT_DIR}/patches/vllm-kv-offload-pinning-fail-closed.patch"

TURBOQUANT_REL="vllm/v1/attention/backends/turboquant_attn.py"
TOOL_SCHEMA_REL="vllm/tool_parsers/structural_tag_registry.py"
MODEL_CONFIG_REL="vllm/config/model.py"
ANTHROPIC_PROTOCOL_REL="vllm/entrypoints/anthropic/protocol.py"
ANTHROPIC_SERVING_REL="vllm/entrypoints/anthropic/serving.py"
CHAT_PROTOCOL_REL="vllm/entrypoints/openai/chat_completion/protocol.py"
SAMPLING_PARAMS_REL="vllm/sampling_params.py"
SCHED_UTILS_REL="vllm/v1/core/sched/utils.py"
INPUT_PROCESSOR_REL="vllm/v1/engine/input_processor.py"
REQUEST_REL="vllm/v1/request.py"
QWEN3_PARSER_REL="vllm/parser/qwen3.py"
STRUCTURED_OUTPUT_REL="vllm/v1/structured_output/__init__.py"
ANTHROPIC_API_ROUTER_REL="vllm/entrypoints/anthropic/api_router.py"
CHAT_SERVING_REL="vllm/entrypoints/openai/chat_completion/serving.py"
RESPONSES_CONTEXT_REL="vllm/entrypoints/openai/responses/context.py"
RESPONSES_PROTOCOL_REL="vllm/entrypoints/openai/responses/protocol.py"
RESPONSES_SERVING_REL="vllm/entrypoints/openai/responses/serving.py"
RESPONSES_STREAMING_REL="vllm/entrypoints/openai/responses/streaming_events.py"
RESPONSES_UTILS_REL="vllm/entrypoints/openai/responses/utils.py"
PARSER_ENGINE_REL="vllm/parser/engine/parser_engine.py"
KV_OFFLOAD_WORKER_REL="vllm/v1/kv_offload/cpu/gpu_worker.py"
WORKSPACE_REL="vllm/v1/worker/workspace.py"
GPU_MODEL_RUNNER_REL="vllm/v1/worker/gpu_model_runner.py"
API_UTILS_REL="vllm/entrypoints/serve/utils/api_utils.py"
ENVS_REL="vllm/envs.py"
CHAT_UTILS_REL="vllm/entrypoints/chat_utils.py"
MEDIA_CONNECTOR_REL="vllm/multimodal/media/connector.py"
IMAGE_MEDIA_REL="vllm/multimodal/media/image.py"
RENDER_PARAMS_REL="vllm/renderers/params.py"
QWEN3_VL_MODEL_REL="vllm/model_executor/models/qwen3_vl.py"

case "${MODE}" in
  build|check)
    ;;
  *)
    echo "Usage: $0 [build|check]" >&2
    exit 2
    ;;
esac

if [[ ! -f "${DEPLOYMENT_INPUT_MANIFEST}" || -L "${DEPLOYMENT_INPUT_MANIFEST}" ]]; then
  echo "Deployment-input manifest is missing or is not a regular non-symlink file." >&2
  exit 1
fi
if [[ "$(wc -l <"${DEPLOYMENT_INPUT_MANIFEST}")" != "70" ]]; then
  echo "Deployment-input manifest must contain exactly 70 hashed files." >&2
  exit 1
fi
(
  cd "${PROJECT_DIR}"
  sha256sum --check --strict \
    "${DEPLOYMENT_INPUT_MANIFEST#"${PROJECT_DIR}"/}"
)
bash "${RUNTIME_COMMON_CONTRACT_TEST}"

actual_commit="$(git -C "${VLLM_DIR}" rev-parse HEAD)"
if [[ "${actual_commit}" != "${VLLM_COMMIT}" ]]; then
  echo "Refusing unexpected vLLM commit: ${actual_commit}" >&2
  exit 1
fi

actual_base_image_id="$(docker image inspect --format '{{.Id}}' "${BASE_IMAGE_TAG}" 2>/dev/null || true)"
if [[ "${actual_base_image_id}" != "${EXPECTED_BASE_IMAGE_ID}" ]]; then
  echo "Required immutable base image is missing or incorrect." >&2
  echo "Expected: ${EXPECTED_BASE_IMAGE_ID}" >&2
  echo "Found:    ${actual_base_image_id:-nothing}" >&2
  echo "No mutable tag or network pull will be substituted." >&2
  echo "Restore the pinned bytes with: ./scripts/restore-images.sh" >&2
  exit 1
fi

actual_status="$(git -C "${VLLM_DIR}" status --short --untracked-files=all)"
if [[ "${actual_status}" != "${EXPECTED_STATUS}" ]]; then
  echo "Refusing unexpected vLLM worktree state:" >&2
  printf '%s\n' "${actual_status}" >&2
  exit 1
fi

printf '%s  %s\n%s  %s\n%s  %s\n%s  %s\n%s  %s\n%s  %s\n%s  %s\n%s  %s\n%s  %s\n%s  %s\n%s  %s\n' \
  "${TURBOQUANT_PATCH_DIFF_SHA256}" "${TURBOQUANT_PATCH_FILE}" \
  "${TOOL_SCHEMA_PATCH_DIFF_SHA256}" "${TOOL_SCHEMA_PATCH_FILE}" \
  "${AGENT_DEFAULTS_PATCH_DIFF_SHA256}" "${AGENT_DEFAULTS_PATCH_FILE}" \
  "${PHASE_BUDGET_PATCH_DIFF_SHA256}" "${PHASE_BUDGET_PATCH_FILE}" \
  "${IMPLICIT_TOOL_GRAMMAR_PATCH_DIFF_SHA256}" "${IMPLICIT_TOOL_GRAMMAR_PATCH_FILE}" \
  "${ANTHROPIC_VALIDATION_PATCH_DIFF_SHA256}" "${ANTHROPIC_VALIDATION_PATCH_FILE}" \
  "${TOOL_TRUNCATION_PATCH_DIFF_SHA256}" "${TOOL_TRUNCATION_PATCH_FILE}" \
  "${VISION_RUNTIME_PATCH_DIFF_SHA256}" "${VISION_RUNTIME_PATCH_FILE}" \
  "${NUMERICAL_AUDITS_PATCH_DIFF_SHA256}" "${NUMERICAL_AUDITS_PATCH_FILE}" \
  "${TURBOQUANT_GUARDS_PATCH_DIFF_SHA256}" "${TURBOQUANT_GUARDS_PATCH_FILE}" \
  "${KV_OFFLOAD_PINNING_PATCH_DIFF_SHA256}" "${KV_OFFLOAD_PINNING_PATCH_FILE}" | \
  sha256sum --check --strict

printf '%s  %s\n' \
  "${SOURCE_PATCH_MANIFEST_SHA256}" "${SOURCE_PATCH_MANIFEST}" | \
  sha256sum --check --strict
(
  cd "${PROJECT_DIR}"
  sha256sum --check --strict \
    "${SOURCE_PATCH_MANIFEST#"${PROJECT_DIR}"/}"
)

# The patcher and its failure tests execute inside the exact immutable Python
# image boundary. Nothing is imported into or installed on the host.
docker run --rm \
  --network none \
  --read-only \
  --user "$(id -u):$(id -g)" \
  --tmpfs /tmp:rw,nodev,nosuid,size=512m \
  --env PYTHONPYCACHEPREFIX=/tmp/pycache \
  --entrypoint python3 \
  --volume "${PROJECT_DIR}:/project:ro" \
  --workdir /project \
  "${BASE_IMAGE_TAG}" \
  -m unittest -v patches.source_patch_v1.test_framework

# Prove that the landmark-aware transaction recreates this exact worktree from
# the pinned upstream commit. The reviewed diffs are independently hashed and
# parsed as review evidence, but they never select mutation locations. The
# private worktree is discarded on every failure and never becomes a runtime.
VERIFY_WORKTREE="$(mktemp -d /tmp/qwen38-vllm-verify.XXXXXX)"
remove_verify_worktree() {
  if ! git -C "${VLLM_DIR}" worktree remove --force "${VERIFY_WORKTREE}"; then
    printf 'ERROR: failed to remove the exact disposable verification worktree: %s\n' \
      "${VERIFY_WORKTREE}" >&2
    return 1
  fi
}
cleanup_verify_worktree() {
  local status=$?
  trap - EXIT
  if ! remove_verify_worktree; then
    status=1
  fi
  exit "${status}"
}
trap cleanup_verify_worktree EXIT
git -C "${VLLM_DIR}" worktree add --detach "${VERIFY_WORKTREE}" \
  "${VLLM_COMMIT}" >/dev/null
docker run --rm \
  --network none \
  --read-only \
  --user "$(id -u):$(id -g)" \
  --tmpfs /tmp:rw,nodev,nosuid,size=512m \
  --env PYTHONPYCACHEPREFIX=/tmp/pycache \
  --entrypoint python3 \
  --volume "${PROJECT_DIR}:/project:ro" \
  --volume "${VERIFY_WORKTREE}:/source:rw" \
  --workdir /project \
  "${BASE_IMAGE_TAG}" \
  -m patches.source_patch_v1.apply_vllm_patchset /source /project
reproduced_status="$(
  git -C "${VERIFY_WORKTREE}" status --short --untracked-files=all
)"
if [[ "${reproduced_status}" != "${EXPECTED_STATUS}" ]]; then
  echo "Reviewed patches produced an unexpected vLLM worktree state:" >&2
  printf '%s\n' "${reproduced_status}" >&2
  exit 1
fi
while IFS= read -r status_line; do
  relative_path="${status_line:3}"
  if ! cmp -s \
    "${VERIFY_WORKTREE}/${relative_path}" \
    "${VLLM_DIR}/${relative_path}"; then
    echo "Reviewed patches do not reproduce ${relative_path}." >&2
    exit 1
  fi
done <<<"${EXPECTED_STATUS}"
remove_verify_worktree
trap - EXIT

printf '%s  %s\n%s  %s\n%s  %s\n%s  %s\n%s  %s\n%s  %s\n%s  %s\n%s  %s\n%s  %s\n%s  %s\n%s  %s\n%s  %s\n%s  %s\n%s  %s\n%s  %s\n%s  %s\n%s  %s\n%s  %s\n%s  %s\n%s  %s\n%s  %s\n%s  %s\n%s  %s\n' \
  "${TURBOQUANT_PATCHED_FILE_SHA256}" "${VLLM_DIR}/${TURBOQUANT_REL}" \
  "${TOOL_SCHEMA_PATCHED_FILE_SHA256}" "${VLLM_DIR}/${TOOL_SCHEMA_REL}" \
  "${MODEL_CONFIG_PATCHED_FILE_SHA256}" "${VLLM_DIR}/${MODEL_CONFIG_REL}" \
  "${ANTHROPIC_PROTOCOL_PATCHED_FILE_SHA256}" "${VLLM_DIR}/${ANTHROPIC_PROTOCOL_REL}" \
  "${ANTHROPIC_SERVING_PATCHED_FILE_SHA256}" "${VLLM_DIR}/${ANTHROPIC_SERVING_REL}" \
  "${CHAT_PROTOCOL_PATCHED_FILE_SHA256}" "${VLLM_DIR}/${CHAT_PROTOCOL_REL}" \
  "${SAMPLING_PARAMS_PATCHED_FILE_SHA256}" "${VLLM_DIR}/${SAMPLING_PARAMS_REL}" \
  "${SCHED_UTILS_PATCHED_FILE_SHA256}" "${VLLM_DIR}/${SCHED_UTILS_REL}" \
  "${INPUT_PROCESSOR_PATCHED_FILE_SHA256}" "${VLLM_DIR}/${INPUT_PROCESSOR_REL}" \
  "${REQUEST_PATCHED_FILE_SHA256}" "${VLLM_DIR}/${REQUEST_REL}" \
  "${QWEN3_PARSER_PATCHED_FILE_SHA256}" "${VLLM_DIR}/${QWEN3_PARSER_REL}" \
  "${STRUCTURED_OUTPUT_PATCHED_FILE_SHA256}" "${VLLM_DIR}/${STRUCTURED_OUTPUT_REL}" \
  "${ANTHROPIC_API_ROUTER_PATCHED_FILE_SHA256}" "${VLLM_DIR}/${ANTHROPIC_API_ROUTER_REL}" \
  "${CHAT_SERVING_PATCHED_FILE_SHA256}" "${VLLM_DIR}/${CHAT_SERVING_REL}" \
  "${RESPONSES_CONTEXT_PATCHED_FILE_SHA256}" "${VLLM_DIR}/${RESPONSES_CONTEXT_REL}" \
  "${RESPONSES_PROTOCOL_PATCHED_FILE_SHA256}" "${VLLM_DIR}/${RESPONSES_PROTOCOL_REL}" \
  "${RESPONSES_SERVING_PATCHED_FILE_SHA256}" "${VLLM_DIR}/${RESPONSES_SERVING_REL}" \
  "${RESPONSES_STREAMING_PATCHED_FILE_SHA256}" "${VLLM_DIR}/${RESPONSES_STREAMING_REL}" \
  "${RESPONSES_UTILS_PATCHED_FILE_SHA256}" "${VLLM_DIR}/${RESPONSES_UTILS_REL}" \
  "${PARSER_ENGINE_PATCHED_FILE_SHA256}" "${VLLM_DIR}/${PARSER_ENGINE_REL}" \
  "${KV_OFFLOAD_WORKER_PATCHED_FILE_SHA256}" "${VLLM_DIR}/${KV_OFFLOAD_WORKER_REL}" \
  "${AGENT_CHAT_TEMPLATE_SHA256}" "${TEMPLATE_FILE}" \
  "${PHASE_BUDGET_UNIT_SHA256}" "${PHASE_BUDGET_UNIT_FILE}" | \
  sha256sum --check --strict

printf '%s  %s\n%s  %s\n%s  %s\n%s  %s\n%s  %s\n%s  %s\n%s  %s\n%s  %s\n%s  %s\n%s  %s\n%s  %s\n%s  %s\n' \
  "${WORKSPACE_PATCHED_FILE_SHA256}" "${VLLM_DIR}/${WORKSPACE_REL}" \
  "${GPU_MODEL_RUNNER_PATCHED_FILE_SHA256}" "${VLLM_DIR}/${GPU_MODEL_RUNNER_REL}" \
  "${API_UTILS_PATCHED_FILE_SHA256}" "${VLLM_DIR}/${API_UTILS_REL}" \
  "${ENVS_PATCHED_FILE_SHA256}" "${VLLM_DIR}/${ENVS_REL}" \
  "${CHAT_UTILS_PATCHED_FILE_SHA256}" "${VLLM_DIR}/${CHAT_UTILS_REL}" \
  "${MEDIA_CONNECTOR_PATCHED_FILE_SHA256}" "${VLLM_DIR}/${MEDIA_CONNECTOR_REL}" \
  "${IMAGE_MEDIA_PATCHED_FILE_SHA256}" "${VLLM_DIR}/${IMAGE_MEDIA_REL}" \
  "${RENDER_PARAMS_PATCHED_FILE_SHA256}" "${VLLM_DIR}/${RENDER_PARAMS_REL}" \
  "${QWEN3_VL_MODEL_PATCHED_FILE_SHA256}" "${VLLM_DIR}/${QWEN3_VL_MODEL_REL}" \
  "${VISION_WORKSPACE_UNIT_SHA256}" "${VISION_WORKSPACE_UNIT_FILE}" \
  "${VISION_CONTRACT_UNIT_SHA256}" "${VISION_CONTRACT_UNIT_FILE}" \
  "${VISION_MLP_UNIT_SHA256}" "${VISION_MLP_UNIT_FILE}" | \
  sha256sum --check --strict

printf '%s  %s\n%s  %s\n' \
  "${RUNTIME_DOCKERFILE_SHA256}" "${DOCKERFILE}" \
  "${DOCKERIGNORE_SHA256}" "${DOCKERIGNORE}" | \
  sha256sum --check --strict

printf '%s  %s\n%s  %s\n%s  %s\n' \
  "${TURBOQUANT_K8V4_UNIT_SHA256}" "${TURBOQUANT_K8V4_UNIT_FILE}" \
  "${QWEN38_CONTEXT_UNIT_SHA256}" "${QWEN38_CONTEXT_UNIT_FILE}" \
  "${NVFP4_KERNEL_UNIT_SHA256}" "${NVFP4_KERNEL_UNIT_FILE}" | \
  sha256sum --check --strict

git -C "${VLLM_DIR}" diff --check

if [[ "${MODE}" == "check" ]]; then
  echo "Pinned base image, vLLM commit, transactional landmark patcher, twenty-nine reviewed runtime source files, seven reviewed modified test files, one reviewed new test file, nine review diffs, agent template, three numerical audit units, and all build units are exact."
  exit 0
fi

DOCKER_BUILDKIT=1 docker build --progress=plain \
  --network none \
  --pull=false \
  --provenance=false \
  --target runtime \
  --build-arg "BASE_IMAGE=${BASE_IMAGE_TAG}" \
  --build-arg "TURBOQUANT_UPSTREAM_FILE_SHA256=${TURBOQUANT_UPSTREAM_FILE_SHA256}" \
  --build-arg "TOOL_SCHEMA_UPSTREAM_FILE_SHA256=${TOOL_SCHEMA_UPSTREAM_FILE_SHA256}" \
  --build-arg "MODEL_CONFIG_UPSTREAM_FILE_SHA256=${MODEL_CONFIG_UPSTREAM_FILE_SHA256}" \
  --build-arg "ANTHROPIC_PROTOCOL_UPSTREAM_FILE_SHA256=${ANTHROPIC_PROTOCOL_UPSTREAM_FILE_SHA256}" \
  --build-arg "ANTHROPIC_SERVING_UPSTREAM_FILE_SHA256=${ANTHROPIC_SERVING_UPSTREAM_FILE_SHA256}" \
  --build-arg "CHAT_PROTOCOL_UPSTREAM_FILE_SHA256=${CHAT_PROTOCOL_UPSTREAM_FILE_SHA256}" \
  --build-arg "SAMPLING_PARAMS_UPSTREAM_FILE_SHA256=${SAMPLING_PARAMS_UPSTREAM_FILE_SHA256}" \
  --build-arg "SCHED_UTILS_UPSTREAM_FILE_SHA256=${SCHED_UTILS_UPSTREAM_FILE_SHA256}" \
  --build-arg "INPUT_PROCESSOR_UPSTREAM_FILE_SHA256=${INPUT_PROCESSOR_UPSTREAM_FILE_SHA256}" \
  --build-arg "REQUEST_UPSTREAM_FILE_SHA256=${REQUEST_UPSTREAM_FILE_SHA256}" \
  --build-arg "QWEN3_PARSER_UPSTREAM_FILE_SHA256=${QWEN3_PARSER_UPSTREAM_FILE_SHA256}" \
  --build-arg "STRUCTURED_OUTPUT_UPSTREAM_FILE_SHA256=${STRUCTURED_OUTPUT_UPSTREAM_FILE_SHA256}" \
  --build-arg "ANTHROPIC_API_ROUTER_UPSTREAM_FILE_SHA256=${ANTHROPIC_API_ROUTER_UPSTREAM_FILE_SHA256}" \
  --build-arg "CHAT_SERVING_UPSTREAM_FILE_SHA256=${CHAT_SERVING_UPSTREAM_FILE_SHA256}" \
  --build-arg "RESPONSES_CONTEXT_UPSTREAM_FILE_SHA256=${RESPONSES_CONTEXT_UPSTREAM_FILE_SHA256}" \
  --build-arg "RESPONSES_PROTOCOL_UPSTREAM_FILE_SHA256=${RESPONSES_PROTOCOL_UPSTREAM_FILE_SHA256}" \
  --build-arg "RESPONSES_SERVING_UPSTREAM_FILE_SHA256=${RESPONSES_SERVING_UPSTREAM_FILE_SHA256}" \
  --build-arg "RESPONSES_STREAMING_UPSTREAM_FILE_SHA256=${RESPONSES_STREAMING_UPSTREAM_FILE_SHA256}" \
  --build-arg "RESPONSES_UTILS_UPSTREAM_FILE_SHA256=${RESPONSES_UTILS_UPSTREAM_FILE_SHA256}" \
  --build-arg "PARSER_ENGINE_UPSTREAM_FILE_SHA256=${PARSER_ENGINE_UPSTREAM_FILE_SHA256}" \
  --build-arg "KV_OFFLOAD_WORKER_UPSTREAM_FILE_SHA256=${KV_OFFLOAD_WORKER_UPSTREAM_FILE_SHA256}" \
  --build-arg "WORKSPACE_UPSTREAM_FILE_SHA256=${WORKSPACE_UPSTREAM_FILE_SHA256}" \
  --build-arg "GPU_MODEL_RUNNER_UPSTREAM_FILE_SHA256=${GPU_MODEL_RUNNER_UPSTREAM_FILE_SHA256}" \
  --build-arg "API_UTILS_UPSTREAM_FILE_SHA256=${API_UTILS_UPSTREAM_FILE_SHA256}" \
  --build-arg "ENVS_UPSTREAM_FILE_SHA256=${ENVS_UPSTREAM_FILE_SHA256}" \
  --build-arg "CHAT_UTILS_UPSTREAM_FILE_SHA256=${CHAT_UTILS_UPSTREAM_FILE_SHA256}" \
  --build-arg "MEDIA_CONNECTOR_UPSTREAM_FILE_SHA256=${MEDIA_CONNECTOR_UPSTREAM_FILE_SHA256}" \
  --build-arg "IMAGE_MEDIA_UPSTREAM_FILE_SHA256=${IMAGE_MEDIA_UPSTREAM_FILE_SHA256}" \
  --build-arg "RENDER_PARAMS_UPSTREAM_FILE_SHA256=${RENDER_PARAMS_UPSTREAM_FILE_SHA256}" \
  --build-arg "QWEN3_VL_MODEL_UPSTREAM_FILE_SHA256=${QWEN3_VL_MODEL_UPSTREAM_FILE_SHA256}" \
  --build-arg "TURBOQUANT_PATCHED_FILE_SHA256=${TURBOQUANT_PATCHED_FILE_SHA256}" \
  --build-arg "TOOL_SCHEMA_PATCHED_FILE_SHA256=${TOOL_SCHEMA_PATCHED_FILE_SHA256}" \
  --build-arg "MODEL_CONFIG_PATCHED_FILE_SHA256=${MODEL_CONFIG_PATCHED_FILE_SHA256}" \
  --build-arg "ANTHROPIC_PROTOCOL_PATCHED_FILE_SHA256=${ANTHROPIC_PROTOCOL_PATCHED_FILE_SHA256}" \
  --build-arg "ANTHROPIC_SERVING_PATCHED_FILE_SHA256=${ANTHROPIC_SERVING_PATCHED_FILE_SHA256}" \
  --build-arg "CHAT_PROTOCOL_PATCHED_FILE_SHA256=${CHAT_PROTOCOL_PATCHED_FILE_SHA256}" \
  --build-arg "SAMPLING_PARAMS_PATCHED_FILE_SHA256=${SAMPLING_PARAMS_PATCHED_FILE_SHA256}" \
  --build-arg "SCHED_UTILS_PATCHED_FILE_SHA256=${SCHED_UTILS_PATCHED_FILE_SHA256}" \
  --build-arg "INPUT_PROCESSOR_PATCHED_FILE_SHA256=${INPUT_PROCESSOR_PATCHED_FILE_SHA256}" \
  --build-arg "REQUEST_PATCHED_FILE_SHA256=${REQUEST_PATCHED_FILE_SHA256}" \
  --build-arg "QWEN3_PARSER_PATCHED_FILE_SHA256=${QWEN3_PARSER_PATCHED_FILE_SHA256}" \
  --build-arg "STRUCTURED_OUTPUT_PATCHED_FILE_SHA256=${STRUCTURED_OUTPUT_PATCHED_FILE_SHA256}" \
  --build-arg "ANTHROPIC_API_ROUTER_PATCHED_FILE_SHA256=${ANTHROPIC_API_ROUTER_PATCHED_FILE_SHA256}" \
  --build-arg "CHAT_SERVING_PATCHED_FILE_SHA256=${CHAT_SERVING_PATCHED_FILE_SHA256}" \
  --build-arg "RESPONSES_CONTEXT_PATCHED_FILE_SHA256=${RESPONSES_CONTEXT_PATCHED_FILE_SHA256}" \
  --build-arg "RESPONSES_PROTOCOL_PATCHED_FILE_SHA256=${RESPONSES_PROTOCOL_PATCHED_FILE_SHA256}" \
  --build-arg "RESPONSES_SERVING_PATCHED_FILE_SHA256=${RESPONSES_SERVING_PATCHED_FILE_SHA256}" \
  --build-arg "RESPONSES_STREAMING_PATCHED_FILE_SHA256=${RESPONSES_STREAMING_PATCHED_FILE_SHA256}" \
  --build-arg "RESPONSES_UTILS_PATCHED_FILE_SHA256=${RESPONSES_UTILS_PATCHED_FILE_SHA256}" \
  --build-arg "PARSER_ENGINE_PATCHED_FILE_SHA256=${PARSER_ENGINE_PATCHED_FILE_SHA256}" \
  --build-arg "KV_OFFLOAD_WORKER_PATCHED_FILE_SHA256=${KV_OFFLOAD_WORKER_PATCHED_FILE_SHA256}" \
  --build-arg "WORKSPACE_PATCHED_FILE_SHA256=${WORKSPACE_PATCHED_FILE_SHA256}" \
  --build-arg "GPU_MODEL_RUNNER_PATCHED_FILE_SHA256=${GPU_MODEL_RUNNER_PATCHED_FILE_SHA256}" \
  --build-arg "API_UTILS_PATCHED_FILE_SHA256=${API_UTILS_PATCHED_FILE_SHA256}" \
  --build-arg "ENVS_PATCHED_FILE_SHA256=${ENVS_PATCHED_FILE_SHA256}" \
  --build-arg "CHAT_UTILS_PATCHED_FILE_SHA256=${CHAT_UTILS_PATCHED_FILE_SHA256}" \
  --build-arg "MEDIA_CONNECTOR_PATCHED_FILE_SHA256=${MEDIA_CONNECTOR_PATCHED_FILE_SHA256}" \
  --build-arg "IMAGE_MEDIA_PATCHED_FILE_SHA256=${IMAGE_MEDIA_PATCHED_FILE_SHA256}" \
  --build-arg "RENDER_PARAMS_PATCHED_FILE_SHA256=${RENDER_PARAMS_PATCHED_FILE_SHA256}" \
  --build-arg "QWEN3_VL_MODEL_PATCHED_FILE_SHA256=${QWEN3_VL_MODEL_PATCHED_FILE_SHA256}" \
  --build-arg "AGENT_CHAT_TEMPLATE_SHA256=${AGENT_CHAT_TEMPLATE_SHA256}" \
  --build-arg "PHASE_BUDGET_UNIT_SHA256=${PHASE_BUDGET_UNIT_SHA256}" \
  --build-arg "VISION_WORKSPACE_UNIT_SHA256=${VISION_WORKSPACE_UNIT_SHA256}" \
  --build-arg "VISION_CONTRACT_UNIT_SHA256=${VISION_CONTRACT_UNIT_SHA256}" \
  --build-arg "VISION_MLP_UNIT_SHA256=${VISION_MLP_UNIT_SHA256}" \
  --build-arg "TURBOQUANT_K8V4_UNIT_SHA256=${TURBOQUANT_K8V4_UNIT_SHA256}" \
  --build-arg "QWEN38_CONTEXT_UNIT_SHA256=${QWEN38_CONTEXT_UNIT_SHA256}" \
  --build-arg "NVFP4_KERNEL_UNIT_SHA256=${NVFP4_KERNEL_UNIT_SHA256}" \
  --build-arg "NUMERICAL_AUDITS_PATCH_DIFF_SHA256=${NUMERICAL_AUDITS_PATCH_DIFF_SHA256}" \
  --build-arg "KV_OFFLOAD_PINNING_PATCH_DIFF_SHA256=${KV_OFFLOAD_PINNING_PATCH_DIFF_SHA256}" \
  --build-arg "SOURCE_DATE_EPOCH=${SOURCE_DATE_EPOCH}" \
  --tag "${IMAGE_TAG}" \
  --file "${DOCKERFILE}" \
  "${PROJECT_DIR}"

actual_image_id="$(docker image inspect --format '{{.Id}}' "${IMAGE_TAG}")"
actual_installed_report="$(
  docker run --rm --network none --entrypoint sha256sum "${IMAGE_TAG}" \
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
    /usr/local/lib/python3.12/dist-packages/vllm/v1/kv_offload/cpu/gpu_worker.py \
    /opt/qwen38/chat_template.jinja \
    /opt/qwen38/phase_budget_unit.py
)"
expected_installed_report="$(printf '%s  %s\n%s  %s\n%s  %s\n%s  %s\n%s  %s\n%s  %s\n%s  %s\n%s  %s\n%s  %s\n%s  %s\n%s  %s\n%s  %s\n%s  %s\n%s  %s\n%s  %s\n%s  %s\n%s  %s\n%s  %s\n%s  %s\n%s  %s\n%s  %s\n%s  %s\n%s  %s' \
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
  "${KV_OFFLOAD_WORKER_PATCHED_FILE_SHA256}" /usr/local/lib/python3.12/dist-packages/vllm/v1/kv_offload/cpu/gpu_worker.py \
  "${AGENT_CHAT_TEMPLATE_SHA256}" /opt/qwen38/chat_template.jinja \
  "${PHASE_BUDGET_UNIT_SHA256}" /opt/qwen38/phase_budget_unit.py)"
if [[ "${actual_installed_report}" != "${expected_installed_report}" ]]; then
  echo "Built image contains unexpected source/template bytes." >&2
  echo "Expected:" >&2
  printf '%s\n' "${expected_installed_report}" >&2
  echo "Found:" >&2
  printf '%s\n' "${actual_installed_report}" >&2
  exit 1
fi

additional_installed_report="$(
  docker run --rm --network none --entrypoint sha256sum "${IMAGE_TAG}" \
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
    /opt/qwen38/vision_mlp_unit.py \
    /opt/qwen38/turboquant_k8v4_unit.py \
    /opt/qwen38/qwen38_context_unit.py \
    /opt/qwen38/nvfp4_kernel_unit.py
)"
expected_additional_installed_report="$(printf '%s  %s\n%s  %s\n%s  %s\n%s  %s\n%s  %s\n%s  %s\n%s  %s\n%s  %s\n%s  %s\n%s  %s\n%s  %s\n%s  %s\n%s  %s\n%s  %s\n%s  %s' \
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
  "${VISION_MLP_UNIT_SHA256}" /opt/qwen38/vision_mlp_unit.py \
  "${TURBOQUANT_K8V4_UNIT_SHA256}" /opt/qwen38/turboquant_k8v4_unit.py \
  "${QWEN38_CONTEXT_UNIT_SHA256}" /opt/qwen38/qwen38_context_unit.py \
  "${NVFP4_KERNEL_UNIT_SHA256}" /opt/qwen38/nvfp4_kernel_unit.py)"
if [[ "${additional_installed_report}" != "${expected_additional_installed_report}" ]]; then
  echo "Built image contains unexpected vision/runtime bytes." >&2
  echo "Expected:" >&2
  printf '%s\n' "${expected_additional_installed_report}" >&2
  echo "Found:" >&2
  printf '%s\n' "${additional_installed_report}" >&2
  exit 1
fi
if [[ "${actual_image_id}" != "${EXPECTED_IMAGE_ID}" ]]; then
  echo "Reproducible build ID mismatch." >&2
  echo "Expected: ${EXPECTED_IMAGE_ID}" >&2
  echo "Found:    ${actual_image_id}" >&2
  exit 1
fi

echo "Built ${IMAGE_TAG} with no build-time network access."
echo "Verified reproducible image ID: ${actual_image_id}"
