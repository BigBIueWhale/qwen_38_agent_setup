#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-build}"
EXPECTED_STATUS=$' M vllm/config/model.py\n M vllm/entrypoints/anthropic/protocol.py\n M vllm/entrypoints/anthropic/serving.py\n M vllm/entrypoints/openai/chat_completion/protocol.py\n M vllm/sampling_params.py\n M vllm/tool_parsers/structural_tag_registry.py\n M vllm/v1/attention/backends/turboquant_attn.py\n M vllm/v1/core/sched/utils.py\n M vllm/v1/engine/input_processor.py\n M vllm/v1/request.py'
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=../config/runtime-v1.sh
source "${PROJECT_DIR}/config/runtime-v1.sh"
VLLM_DIR="${PROJECT_DIR}/vllm"
DOCKERFILE="${PROJECT_DIR}/containers/Dockerfile.runtime"
DOCKERIGNORE="${PROJECT_DIR}/.dockerignore"
TEMPLATE_FILE="${PROJECT_DIR}/chat_template.jinja"
PHASE_BUDGET_UNIT_FILE="${PROJECT_DIR}/scripts/phase_budget_unit.py"

TURBOQUANT_PATCH_FILE="${PROJECT_DIR}/patches/vllm-turboquant-k8v4-direct-workspace.patch"
TOOL_SCHEMA_PATCH_FILE="${PROJECT_DIR}/patches/vllm-enforce-auto-tool-schema.patch"
AGENT_DEFAULTS_PATCH_FILE="${PROJECT_DIR}/patches/vllm-qwen38-agent-defaults-and-thinking.patch"
PHASE_BUDGET_PATCH_FILE="${PROJECT_DIR}/patches/vllm-qwen38-separate-final-response-budget.patch"

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

case "${MODE}" in
  build|check)
    ;;
  *)
    echo "Usage: $0 [build|check]" >&2
    exit 2
    ;;
esac

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

printf '%s  %s\n%s  %s\n%s  %s\n%s  %s\n' \
  "${TURBOQUANT_PATCH_DIFF_SHA256}" "${TURBOQUANT_PATCH_FILE}" \
  "${TOOL_SCHEMA_PATCH_DIFF_SHA256}" "${TOOL_SCHEMA_PATCH_FILE}" \
  "${AGENT_DEFAULTS_PATCH_DIFF_SHA256}" "${AGENT_DEFAULTS_PATCH_FILE}" \
  "${PHASE_BUDGET_PATCH_DIFF_SHA256}" "${PHASE_BUDGET_PATCH_FILE}" | \
  sha256sum --check --strict

for reviewed_patch in \
  "${TURBOQUANT_PATCH_FILE}" \
  "${TOOL_SCHEMA_PATCH_FILE}" \
  "${AGENT_DEFAULTS_PATCH_FILE}" \
  "${PHASE_BUDGET_PATCH_FILE}"; do
  git -C "${VLLM_DIR}" apply --reverse --check "${reviewed_patch}"
done

read -r actual_turboquant_diff _ < <(
  git -C "${VLLM_DIR}" diff -- "${TURBOQUANT_REL}" | sha256sum
)
[[ "${actual_turboquant_diff}" == "${TURBOQUANT_PATCH_DIFF_SHA256}" ]] || {
  echo "Refusing unreviewed TurboQuant diff: ${actual_turboquant_diff}" >&2
  exit 1
}

read -r actual_tool_schema_diff _ < <(
  git -C "${VLLM_DIR}" diff -- "${TOOL_SCHEMA_REL}" | sha256sum
)
[[ "${actual_tool_schema_diff}" == "${TOOL_SCHEMA_PATCH_DIFF_SHA256}" ]] || {
  echo "Refusing unreviewed automatic-tool-schema diff: ${actual_tool_schema_diff}" >&2
  exit 1
}

read -r actual_agent_defaults_diff _ < <(
  git -C "${VLLM_DIR}" diff -- \
    "${MODEL_CONFIG_REL}" \
    "${ANTHROPIC_PROTOCOL_REL}" \
    "${ANTHROPIC_SERVING_REL}" \
    "${CHAT_PROTOCOL_REL}" | sha256sum
)
[[ "${actual_agent_defaults_diff}" == "${AGENT_DEFAULTS_PATCH_DIFF_SHA256}" ]] || {
  echo "Refusing unreviewed agent-defaults/Anthropic diff: ${actual_agent_defaults_diff}" >&2
  exit 1
}

read -r actual_phase_budget_diff _ < <(
  git -C "${VLLM_DIR}" diff -- \
    "${SAMPLING_PARAMS_REL}" \
    "${SCHED_UTILS_REL}" \
    "${INPUT_PROCESSOR_REL}" \
    "${REQUEST_REL}" | sha256sum
)
[[ "${actual_phase_budget_diff}" == "${PHASE_BUDGET_PATCH_DIFF_SHA256}" ]] || {
  echo "Refusing unreviewed separate-phase-budget diff: ${actual_phase_budget_diff}" >&2
  exit 1
}

printf '%s  %s\n%s  %s\n%s  %s\n%s  %s\n%s  %s\n%s  %s\n%s  %s\n%s  %s\n%s  %s\n%s  %s\n%s  %s\n%s  %s\n' \
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
  "${AGENT_CHAT_TEMPLATE_SHA256}" "${TEMPLATE_FILE}" \
  "${PHASE_BUDGET_UNIT_SHA256}" "${PHASE_BUDGET_UNIT_FILE}" | \
  sha256sum --check --strict

printf '%s  %s\n%s  %s\n' \
  "${RUNTIME_DOCKERFILE_SHA256}" "${DOCKERFILE}" \
  "${DOCKERIGNORE_SHA256}" "${DOCKERIGNORE}" | \
  sha256sum --check --strict

git -C "${VLLM_DIR}" diff --check

if [[ "${MODE}" == "check" ]]; then
  echo "Pinned base image, vLLM commit, ten reviewed source files, patches, agent template, and phase-budget unit are exact."
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
  --build-arg "AGENT_CHAT_TEMPLATE_SHA256=${AGENT_CHAT_TEMPLATE_SHA256}" \
  --build-arg "PHASE_BUDGET_UNIT_SHA256=${PHASE_BUDGET_UNIT_SHA256}" \
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
    /opt/qwen38/chat_template.jinja \
    /opt/qwen38/phase_budget_unit.py
)"
expected_installed_report="$(printf '%s  %s\n%s  %s\n%s  %s\n%s  %s\n%s  %s\n%s  %s\n%s  %s\n%s  %s\n%s  %s\n%s  %s\n%s  %s\n%s  %s' \
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
if [[ "${actual_image_id}" != "${EXPECTED_IMAGE_ID}" ]]; then
  echo "Reproducible build ID mismatch." >&2
  echo "Expected: ${EXPECTED_IMAGE_ID}" >&2
  echo "Found:    ${actual_image_id}" >&2
  exit 1
fi

echo "Built ${IMAGE_TAG} with no build-time network access."
echo "Verified reproducible image ID: ${actual_image_id}"
