#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'USAGE'
Usage: build-vllm.sh {build|check|materialise}
  build        verify every pinned input, then build and export the runtime image
  check        verify every pinned input and stop; nothing is written
  materialise  write the live vllm/ tree from the verified reconstruction
USAGE
  exit 2
}

# The mode is required. It defaulted to `build`, which made the longest and
# only image-producing action the one you got by typing nothing -- a second
# mode wearing the first one's clothes.
(($# == 1)) || usage
MODE="$1"
EXPECTED_STATUS=$' M tests/config/test_config_utils.py
 M tests/distributed/test_rocm_quick_reduce.py
 M tests/engine/test_arg_utils.py
 M tests/entrypoints/anthropic/test_anthropic_messages_conversion.py
 M tests/entrypoints/multimodal/openai/chat_completion/test_video.py
 M tests/entrypoints/multimodal/openai/chat_completion/test_vision.py
 M tests/entrypoints/openai/chat_completion/test_logprob_token_ids.py
 M tests/entrypoints/openai/completion/test_completion.py
 M tests/entrypoints/openai/responses/test_responses_utils.py
 M tests/entrypoints/openai/responses/test_serving_responses.py
 M tests/entrypoints/openai/test_render_token_offsets.py
 M tests/entrypoints/scale_out/derender/test_derender.py
 M tests/entrypoints/scale_out/render/test_render_multimodal.py
 M tests/entrypoints/scale_out/token_in_token_out/test_generate_stream.py
 D tests/entrypoints/scale_out/token_in_token_out/test_mm_serde.py
 M tests/entrypoints/scale_out/token_in_token_out/test_protocol.py
 M tests/entrypoints/scale_out/token_in_token_out/test_serving_multimodal_tokens.py
 M tests/entrypoints/serve/exception_handling/test_http_status_metrics.py
 M tests/entrypoints/serve/exception_handling/test_validation_exception_handler.py
 M tests/entrypoints/serve/lora/test_lora_adapters.py
 M tests/entrypoints/serve/utils/test_api_utils.py
 M tests/entrypoints/unit_tests/test_chat_utils.py
 M tests/evals/gsm8k/test_gsm8k_offloading.py
 M tests/models/language/pooling/test_reward.py
 M tests/multimodal/media/test_connector.py
 M tests/multimodal/media/test_image.py
 M tests/parser/engine/replay_harness.py
 M tests/parser/engine/streaming_helpers.py
 M tests/parser/engine/test_delegating_replay.py
 M tests/parser/engine/test_engine.py
 M tests/parser/engine/test_nemotron_v3.py
 M tests/parser/engine/test_parser_engine.py
 M tests/parser/engine/test_qwen3.py
 M tests/parser/engine/test_qwen3_reasoning.py
 M tests/parser/engine/test_replay.py
 M tests/parser/engine/test_seed_oss.py
 M tests/parser/engine/test_token_id_scanner.py
 M tests/parser/engine/trace_builder.py
 M tests/quantization/test_turboquant.py
 M tests/renderers/test_hf.py
 M tests/test_request_input_bounds.py
 M tests/test_sampling_params.py
 M tests/tool_parsers/test_structural_tag_registry.py
 M tests/v1/core/test_prefix_caching.py
 M tests/v1/core/test_single_type_kv_cache_manager.py
 M tests/v1/e2e/general/test_context_length.py
 M tests/v1/engine/test_output_processor.py
 M tests/v1/kv_connector/nixl_integration/run_multi_connector_accuracy_test.sh
 M tests/v1/kv_connector/nixl_integration/run_multi_connector_edge_case_test.sh
 M tests/v1/kv_connector/nixl_integration/spec_decode_acceptance_test.sh
 M tests/v1/kv_connector/unit/offloading_connector/test_config.py
 M tests/v1/kv_connector/unit/offloading_connector/test_events.py
 M tests/v1/kv_connector/unit/offloading_connector/test_scheduler.py
 M tests/v1/kv_connector/unit/offloading_connector/test_worker.py
 M tests/v1/kv_connector/unit/offloading_connector/utils.py
 M tests/v1/kv_connector/unit/test_config.py
 M tests/v1/kv_connector/unit/test_hma_auto_config.py
 M tests/v1/kv_connector/unit/test_offloading_connector.py
 D tests/v1/kv_offload/cpu/policies/__init__.py
 D tests/v1/kv_offload/cpu/policies/test_factory.py
 M tests/v1/kv_offload/cpu/test_manager.py
 M tests/v1/kv_offload/test_factory.py
 M tests/v1/kv_offload/test_file_mapper.py
 M tests/v1/kv_offload/tiering/p2p/run_accuracy_test.sh
 M tests/v1/kv_offload/tiering/test_fs_tier.py
 M tests/v1/kv_offload/tiering/test_obj_tier.py
 M tests/v1/kv_offload/tiering/test_tiering_offloading.py
 M tests/v1/logits_processors/test_correctness.py
 M tests/v1/simple_kv_offload/test_integration.py
 M tests/v1/simple_kv_offload/test_scheduler.py
 M tests/v1/streaming_input/test_async_llm_streaming.py
 M tests/v1/structured_output/test_backend_xgrammar_stop_tokens.py
 M tests/v1/worker/test_gpu_model_runner_mm_gather.py
 M tests/v1/worker/test_gpu_worker.py
 M vllm/config/cache.py
 M vllm/config/model.py
 M vllm/config/reasoning.py
 M vllm/config/vllm.py
 M vllm/distributed/kv_transfer/kv_connector/v1/offloading/config.py
 M vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py
 M vllm/distributed/kv_transfer/kv_connector/v1/offloading_connector.py
 M vllm/engine/arg_utils.py
 M vllm/entrypoints/anthropic/api_router.py
 M vllm/entrypoints/anthropic/protocol.py
 M vllm/entrypoints/anthropic/serving.py
 M vllm/entrypoints/chat_utils.py
 M vllm/entrypoints/generate/api_router.py
 M vllm/entrypoints/llm.py
 M vllm/entrypoints/openai/chat_completion/protocol.py
 M vllm/entrypoints/openai/chat_completion/serving.py
 M vllm/entrypoints/openai/cli_args.py
 M vllm/entrypoints/openai/completion/protocol.py
 M vllm/entrypoints/openai/completion/serving.py
 M vllm/entrypoints/openai/engine/protocol.py
 M vllm/entrypoints/openai/responses/context.py
 M vllm/entrypoints/openai/responses/protocol.py
 M vllm/entrypoints/openai/responses/serving.py
 M vllm/entrypoints/openai/responses/streaming_events.py
 M vllm/entrypoints/openai/responses/utils.py
 M vllm/entrypoints/scale_out/derender/serving.py
 M vllm/entrypoints/scale_out/render/serving.py
 D vllm/entrypoints/scale_out/token_in_token_out/mm_serde.py
 M vllm/entrypoints/scale_out/token_in_token_out/protocol.py
 M vllm/entrypoints/scale_out/token_in_token_out/serving.py
 M vllm/entrypoints/serve/exception_handling/error_response.py
 M vllm/entrypoints/serve/exception_handling/handlers/exception.py
 M vllm/entrypoints/serve/exception_handling/handlers/http.py
 M vllm/entrypoints/serve/exception_handling/handlers/validation.py
 M vllm/entrypoints/serve/exception_handling/handlers/vllm_error.py
 M vllm/entrypoints/serve/exception_handling/register.py
 M vllm/entrypoints/serve/utils/api_utils.py
 D vllm/entrypoints/serve/utils/tool_calls_utils.py
 M vllm/envs.py
 M vllm/model_executor/models/qwen3_vl.py
 M vllm/multimodal/media/connector.py
 M vllm/multimodal/media/image.py
 M vllm/multimodal/processing/inputs.py
 M vllm/multimodal/processing/processor.py
 M vllm/parser/abstract_parser.py
 M vllm/parser/deepseek_v32.py
 M vllm/parser/deepseek_v4.py
 M vllm/parser/engine/adapters.py
 M vllm/parser/engine/events.py
 M vllm/parser/engine/parser_engine.py
 M vllm/parser/engine/parser_engine_config.py
 M vllm/parser/engine/streaming_parser_engine.py
 M vllm/parser/engine/token_id_scanner.py
 M vllm/parser/gemma4.py
 M vllm/parser/glm47_moe.py
 M vllm/parser/inkling.py
 M vllm/parser/kimi_k2.py
 M vllm/parser/minimax_m2.py
 M vllm/parser/mistral.py
 M vllm/parser/qwen3.py
 M vllm/reasoning/abs_reasoning_parsers.py
 M vllm/renderers/base.py
 M vllm/renderers/hf.py
 M vllm/renderers/online_derenderer.py
 M vllm/renderers/params.py
 M vllm/sampling_params.py
 M vllm/tokenizers/detokenizer_utils.py
 M vllm/tool_parsers/abstract_tool_parser.py
 M vllm/tool_parsers/structural_tag_registry.py
 M vllm/tool_parsers/utils.py
 M vllm/v1/attention/backends/turboquant_attn.py
 M vllm/v1/attention/ops/triton_turboquant_decode.py
 M vllm/v1/attention/ops/triton_turboquant_store.py
 M vllm/v1/core/block_pool.py
 M vllm/v1/core/kv_cache_coordinator.py
 M vllm/v1/core/kv_cache_manager.py
 M vllm/v1/core/kv_cache_utils.py
 M vllm/v1/core/sched/scheduler.py
 M vllm/v1/core/sched/utils.py
 M vllm/v1/core/single_type_kv_cache_manager.py
 M vllm/v1/engine/async_llm.py
 M vllm/v1/engine/detokenizer.py
 M vllm/v1/engine/input_processor.py
 M vllm/v1/engine/output_processor.py
 M vllm/v1/kv_offload/base.py
 M vllm/v1/kv_offload/config.py
 M vllm/v1/kv_offload/cpu/common.py
 M vllm/v1/kv_offload/cpu/gpu_worker.py
 M vllm/v1/kv_offload/cpu/manager.py
 D vllm/v1/kv_offload/cpu/policies/__init__.py
 D vllm/v1/kv_offload/cpu/policies/arc.py
 D vllm/v1/kv_offload/cpu/policies/base.py
 D vllm/v1/kv_offload/cpu/policies/factory.py
 D vllm/v1/kv_offload/cpu/policies/lru.py
 M vllm/v1/kv_offload/cpu/spec.py
 M vllm/v1/kv_offload/tiering/manager.py
 M vllm/v1/kv_offload/tiering/spec.py
 M vllm/v1/request.py
 M vllm/v1/sample/thinking_budget_state.py
 M vllm/v1/simple_kv_offload/manager.py
 M vllm/v1/structured_output/__init__.py
 M vllm/v1/structured_output/backend_types.py
 M vllm/v1/structured_output/backend_xgrammar.py
 M vllm/v1/worker/gpu_model_runner.py
 M vllm/v1/worker/gpu_worker.py
 M vllm/v1/worker/startup_plan.py
 M vllm/v1/worker/workspace.py
?? tests/entrypoints/openai/chat_completion/test_parallel_tool_call_integrity.py
?? tests/entrypoints/openai/test_beam_search_boundary.py
?? tests/entrypoints/scale_out/derender/test_terminal_metadata.py
?? tests/entrypoints/scale_out/token_in_token_out/test_raw_images.py
?? tests/entrypoints/scale_out/token_in_token_out/test_raw_media_boundary.py
?? tests/entrypoints/test_kv_scope_protocol.py
?? tests/parser/engine/test_qwen_terminal_authority.py
?? tests/parser/engine/test_qwen_xml_fidelity.py
?? tests/parser/engine/test_reasoning_token_count.py
?? tests/v1/core/test_kv_cache_users_sizing.py
?? tests/v1/core/test_prefix_cache.py
?? tests/v1/worker/test_workspace.py
?? vllm/v1/core/prefix_cache.py
?? vllm/v1/structured_output/stop_checker.py'
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
CHAT_TEMPLATE_RETENTION_UNIT_FILE="${PROJECT_DIR}/scripts/chat_template_retention_unit.py"
NVFP4_KERNEL_UNIT_FILE="${PROJECT_DIR}/scripts/nvfp4_kernel_unit.py"
TOOL_OUTPUT_PARSER_UNIT_FILE="${PROJECT_DIR}/scripts/tool_output_parser_unit.py"
REASONING_USAGE_UNIT_FILE="${PROJECT_DIR}/scripts/reasoning_usage_unit.py"
QWEN_GRAMMAR_UNIT_FILE="${PROJECT_DIR}/scripts/qwen_grammar_unit.py"
SOURCE_PATCH_DIR="${PROJECT_DIR}/patches/source_patch_v1"
SOURCE_PATCH_MANIFEST="${SOURCE_PATCH_DIR}/manifest.sha256"
GENERATED_STAGES_REL="patches/source_patch_v1/generated_vllm_stages.py"
DEPLOYMENT_INPUT_MANIFEST="${PROJECT_DIR}/config/deployment-inputs.sha256"
RUNTIME_COMMON_CONTRACT_TEST="${PROJECT_DIR}/scripts/test-runtime-common-contract.sh"
README_FILE="${PROJECT_DIR}/README.md"
MODEL_MANIFEST_DIR="${PROJECT_DIR}/manifests"

# The image is exported to a tarball and loaded, rather than tagged directly,
# so that `rewrite-timestamp=true` can normalise every layer timestamp to
# SOURCE_DATE_EPOCH. Without it the COPY layers keep the build context's file
# mtimes, and two machines holding byte-identical trees produce different image
# IDs purely because their checkouts happened at different moments -- which is
# exactly what happened, and what made EXPECTED_IMAGE_ID unreachable anywhere
# except the machine that first wrote the tree.
BUILD_EXPORT_DIR="$(mktemp -d "${TMPDIR:-/tmp}/qwen38-vllm-build.XXXXXX")"
case "${BUILD_EXPORT_DIR}" in
  "${TMPDIR:-/tmp}"/qwen38-vllm-build.*) ;;
  *) echo "Unexpected temporary build-export directory: ${BUILD_EXPORT_DIR}" >&2; exit 1 ;;
esac
readonly BUILD_EXPORT_DIR
cleanup_build_export() {
  rm -rf -- "${BUILD_EXPORT_DIR}"
}
trap cleanup_build_export EXIT
RUNTIME_ARCHIVE="${BUILD_EXPORT_DIR}/runtime.tar"
readonly RUNTIME_ARCHIVE

TURBOQUANT_PATCH_FILE="${PROJECT_DIR}/patches/vllm-turboquant-k8v4-direct-workspace.patch"
TOOL_SCHEMA_PATCH_FILE="${PROJECT_DIR}/patches/vllm-enforce-auto-tool-schema.patch"
AGENT_DEFAULTS_PATCH_FILE="${PROJECT_DIR}/patches/vllm-qwen38-agent-defaults-and-thinking.patch"
PHASE_BUDGET_PATCH_FILE="${PROJECT_DIR}/patches/vllm-qwen38-separate-final-response-budget.patch"
IMPLICIT_TOOL_GRAMMAR_PATCH_FILE="${PROJECT_DIR}/patches/vllm-qwen-implicit-tool-grammar-boundary.patch"
QWEN_LANGUAGE_PATCH_FILE="${PROJECT_DIR}/patches/vllm-qwen-exact-tool-language.patch"
PNG_SOURCE_PATCH_FILE="${PROJECT_DIR}/patches/vllm-png-source-admission.patch"
SAMPLING_BOUNDARY_PATCH_FILE="${PROJECT_DIR}/patches/vllm-sampling-decoding-boundary.patch"
GENERATE_RESULT_PATCH_FILE="${PROJECT_DIR}/patches/vllm-token-generation-result-integrity.patch"
RAW_IMAGE_TRANSPORT_PATCH_FILE="${PROJECT_DIR}/patches/vllm-raw-image-token-transport.patch"
XML_TEXT_FIDELITY_PATCH_FILE="${PROJECT_DIR}/patches/vllm-xml-text-fidelity.patch"
INPUT_STREAM_AGENT_IDENTITY_PATCH_FILE="${PROJECT_DIR}/patches/vllm-input-stream-agent-identity.patch"
PRECISE_REQUEST_ERRORS_PATCH_FILE="${PROJECT_DIR}/patches/vllm-precise-request-errors.patch"
TOKEN_TEXT_PROVENANCE_PATCH_FILE="${PROJECT_DIR}/patches/vllm-token-text-provenance.patch"
SCHEMA_FAITHFUL_XML_PATCH_FILE="${PROJECT_DIR}/patches/vllm-schema-faithful-xml.patch"
ONE_WAY_THINKING_BOUNDARY_PATCH_FILE="${PROJECT_DIR}/patches/vllm-one-way-thinking-boundary.patch"
TOOL_OUTPUT_COMPLETION_PATCH_FILE="${PROJECT_DIR}/patches/vllm-tool-output-completion.patch"
PHASE_AWARE_PARSER_TERMINALS_PATCH_FILE="${PROJECT_DIR}/patches/vllm-phase-aware-parser-terminals.patch"
SAMPLING_RESOLUTION_PATCH_FILE="${PROJECT_DIR}/patches/vllm-generation-sampling-resolution.patch"
ANTHROPIC_TERMINAL_PATCH_FILE="${PROJECT_DIR}/patches/vllm-anthropic-terminal-metadata.patch"
RESPONSES_IDENTITY_PATCH_FILE="${PROJECT_DIR}/patches/vllm-responses-stream-identity.patch"
RESPONSES_HISTORY_PATCH_FILE="${PROJECT_DIR}/patches/vllm-responses-history-integrity.patch"
SINGLE_CALL_PATCH_FILE="${PROJECT_DIR}/patches/vllm-qwen-single-call-grammar.patch"
KV_PHYSICAL_PATCH_FILE="${PROJECT_DIR}/patches/vllm-kv-physical-free-memory.patch"
ANTHROPIC_INPUTS_PATCH_FILE="${PROJECT_DIR}/patches/vllm-anthropic-input-fidelity.patch"
ANTHROPIC_VALIDATION_PATCH_FILE="${PROJECT_DIR}/patches/vllm-anthropic-validation-http400.patch"
TOOL_TRUNCATION_PATCH_FILE="${PROJECT_DIR}/patches/vllm-tool-truncation-finish-reason.patch"
VISION_RUNTIME_PATCH_FILE="${PROJECT_DIR}/patches/vllm-qwen38-vision-runtime.patch"
NUMERICAL_AUDITS_PATCH_FILE="${PROJECT_DIR}/patches/vllm-qwen38-numerical-audits.patch"
TURBOQUANT_GUARDS_PATCH_FILE="${PROJECT_DIR}/patches/vllm-turboquant-fail-closed-guards.patch"
KV_OFFLOAD_PINNING_PATCH_FILE="${PROJECT_DIR}/patches/vllm-kv-offload-pinning-fail-closed.patch"
SHARED_PREFIX_CACHE_PATCH_FILE="${PROJECT_DIR}/patches/vllm-shared-prefix-cache-and-user-capacity.patch"
EXACT_REASONING_USAGE_PATCH_FILE="${PROJECT_DIR}/patches/vllm-exact-reasoning-usage.patch"
CANONICAL_FRAMING_PATCH_FILE="${PROJECT_DIR}/patches/vllm-qwen-canonical-parameter-framing.patch"
QWEN_OWNED_GRAMMAR_PATCH_FILE="${PROJECT_DIR}/patches/vllm-qwen-owned-tool-grammar.patch"

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
CACHE_CONFIG_REL="vllm/config/cache.py"
VLLM_CONFIG_REL="vllm/config/vllm.py"
ARG_UTILS_REL="vllm/engine/arg_utils.py"
LLM_ENTRYPOINT_REL="vllm/entrypoints/llm.py"
KV_CACHE_UTILS_REL="vllm/v1/core/kv_cache_utils.py"
GPU_WORKER_REL="vllm/v1/worker/gpu_worker.py"
STARTUP_PLAN_REL="vllm/v1/worker/startup_plan.py"
KV_OFFLOAD_CONFIG_REL="vllm/v1/kv_offload/config.py"
KV_OFFLOAD_BASE_REL="vllm/v1/kv_offload/base.py"
KV_OFFLOAD_CPU_SPEC_REL="vllm/v1/kv_offload/cpu/spec.py"
KV_OFFLOAD_CPU_MANAGER_REL="vllm/v1/kv_offload/cpu/manager.py"
KV_TIERING_SPEC_REL="vllm/v1/kv_offload/tiering/spec.py"
KV_TIERING_MANAGER_REL="vllm/v1/kv_offload/tiering/manager.py"
OFFLOAD_CONNECTOR_CONFIG_REL="vllm/distributed/kv_transfer/kv_connector/v1/offloading/config.py"
OFFLOAD_CONNECTOR_SCHEDULER_REL="vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py"
COMPLETION_PROTOCOL_REL="vllm/entrypoints/openai/completion/protocol.py"
GENERATE_API_ROUTER_REL="vllm/entrypoints/generate/api_router.py"
CLI_ARGS_REL="vllm/entrypoints/openai/cli_args.py"
TITOTO_PROTOCOL_REL="vllm/entrypoints/scale_out/token_in_token_out/protocol.py"
TITOTO_SERVING_REL="vllm/entrypoints/scale_out/token_in_token_out/serving.py"
ABSTRACT_PARSER_REL="vllm/parser/abstract_parser.py"
PARSER_ADAPTERS_REL="vllm/parser/engine/adapters.py"
ENGINE_PROTOCOL_REL="vllm/entrypoints/openai/engine/protocol.py"

case "${MODE}" in
  build|check|materialise)
    ;;
  *)
    usage
    ;;
esac

if [[ ! -f "${DEPLOYMENT_INPUT_MANIFEST}" || -L "${DEPLOYMENT_INPUT_MANIFEST}" ]]; then
  echo "Deployment-input manifest is missing or is not a regular non-symlink file." >&2
  exit 1
fi
if [[ "$(wc -l <"${DEPLOYMENT_INPUT_MANIFEST}")" != "${DEPLOYMENT_INPUT_FILE_COUNT}" ]]; then
  echo "Deployment-input manifest must contain exactly" \
    "${DEPLOYMENT_INPUT_FILE_COUNT} hashed files." >&2
  exit 1
fi
(
  cd "${PROJECT_DIR}"
  sha256sum --check --strict \
    "${DEPLOYMENT_INPUT_MANIFEST#"${PROJECT_DIR}"/}"
)
docker run --rm --network none --read-only \
  --user "$(id -u):$(id -g)" \
  --volume "${PROJECT_DIR}:/project:ro" \
  --entrypoint bash "${BASE_IMAGE_TAG}" /project/scripts/test-runtime-common-contract.sh

# Every SHA-256 the README states is one this repository derives. A hash in the
# README is provenance -- a claim that some file, image, or archive has exactly
# these bytes -- and the deployment-input manifest, the runtime lock, and the
# model manifests already fix every such value. A hash matching none of them
# describes an artifact nobody has, which is a build failure and not a reading
# error. It is checked here, above the manifest whose bytes it reads, so the
# claim is measured against verified pins rather than against whatever the
# working tree happens to hold.
#
# Identity this repository does not pin is named by its owning repository
# instead of restated as a hash: a value with no local source has nothing to be
# checked against, and an unowned copy is exactly what drifts.
readme_pins="$(grep -ohE '\b[0-9a-f]{64}\b' "${README_FILE}" | sort -u || true)"
if [[ -z "${readme_pins}" ]]; then
  echo "The README states no pinned values; the provenance extractor no longer matches it." >&2
  exit 1
fi
underived_pins="$(comm -23 <(printf '%s\n' "${readme_pins}") <(
  {
    cut -d' ' -f1 "${DEPLOYMENT_INPUT_MANIFEST}"
    grep -ohE '\b[0-9a-f]{64}\b' \
      "${PROJECT_DIR}/config/runtime-v1.sh" \
      "${MODEL_MANIFEST_DIR}"/*.sha256
  } | sort -u
))"
if [[ -n "${underived_pins}" ]]; then
  echo "The README states values this repository does not derive:" >&2
  printf '%s\n' "${underived_pins}" | sed 's/^/  /' >&2
  exit 1
fi

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
# A path the live tree changes that the reviewed patch set does not name is
# authored work or damage in every mode: no committed identity describes it,
# and nothing here may write over it.
unnamed_live_paths="$(
  sed -n 's/^...//p' <<<"${actual_status}" \
    | grep -Fxv -f <(sed -n 's/^...//p' <<<"${EXPECTED_STATUS}") || true
)"
if [[ -n "${unnamed_live_paths}" ]]; then
  echo "Refusing a vLLM worktree that changes paths the reviewed patch set" \
    "does not name:" >&2
  sed 's/^/  /' <<<"${unnamed_live_paths}" >&2
  echo "These are hand-authored edits or a damaged tree; nothing was written." >&2
  echo "Next: compile authored work into a reviewed stage with" \
    "patches/source_patch_v1/compile_review_diff.py and commit it, or remove" \
    "the paths deliberately; no mode here will decide that for you." >&2
  exit 1
fi
# A tree merely missing paths a pulled stage added is behind, not damaged, and
# that is precisely what `materialise` repairs; every other mode still
# requires the exact reviewed state.
if [[ "${MODE}" != "materialise" && "${actual_status}" != "${EXPECTED_STATUS}" ]]; then
  echo "Refusing unexpected vLLM worktree state:" >&2
  printf '%s\n' "${actual_status}" >&2
  echo "The live tree is behind the reviewed patch set." >&2
  echo "Next: ./scripts/build-vllm.sh materialise" >&2
  exit 1
fi

printf '%s  %s\n' \
  "${TURBOQUANT_PATCH_DIFF_SHA256}" "${TURBOQUANT_PATCH_FILE}" \
  "${TOOL_SCHEMA_PATCH_DIFF_SHA256}" "${TOOL_SCHEMA_PATCH_FILE}" \
  "${AGENT_DEFAULTS_PATCH_DIFF_SHA256}" "${AGENT_DEFAULTS_PATCH_FILE}" \
  "${PHASE_BUDGET_PATCH_DIFF_SHA256}" "${PHASE_BUDGET_PATCH_FILE}" \
  "${IMPLICIT_TOOL_GRAMMAR_PATCH_DIFF_SHA256}" "${IMPLICIT_TOOL_GRAMMAR_PATCH_FILE}" \
  "${ANTHROPIC_VALIDATION_PATCH_DIFF_SHA256}" "${ANTHROPIC_VALIDATION_PATCH_FILE}" \
  "${ANTHROPIC_INPUTS_PATCH_DIFF_SHA256}" "${ANTHROPIC_INPUTS_PATCH_FILE}" \
  "${QWEN_LANGUAGE_PATCH_DIFF_SHA256}" "${QWEN_LANGUAGE_PATCH_FILE}" \
  "${PNG_SOURCE_PATCH_DIFF_SHA256}" "${PNG_SOURCE_PATCH_FILE}" \
  "${KV_PHYSICAL_PATCH_DIFF_SHA256}" "${KV_PHYSICAL_PATCH_FILE}" \
  "${SINGLE_CALL_PATCH_DIFF_SHA256}" "${SINGLE_CALL_PATCH_FILE}" \
  "${RESPONSES_HISTORY_PATCH_DIFF_SHA256}" "${RESPONSES_HISTORY_PATCH_FILE}" \
  "${RESPONSES_IDENTITY_PATCH_DIFF_SHA256}" "${RESPONSES_IDENTITY_PATCH_FILE}" \
  "${ANTHROPIC_TERMINAL_PATCH_DIFF_SHA256}" "${ANTHROPIC_TERMINAL_PATCH_FILE}" \
  "${SAMPLING_RESOLUTION_PATCH_DIFF_SHA256}" "${SAMPLING_RESOLUTION_PATCH_FILE}" \
  "${PHASE_AWARE_PARSER_TERMINALS_PATCH_DIFF_SHA256}" "${PHASE_AWARE_PARSER_TERMINALS_PATCH_FILE}" \
  "${TOOL_OUTPUT_COMPLETION_PATCH_DIFF_SHA256}" "${TOOL_OUTPUT_COMPLETION_PATCH_FILE}" \
  "${ONE_WAY_THINKING_BOUNDARY_PATCH_DIFF_SHA256}" "${ONE_WAY_THINKING_BOUNDARY_PATCH_FILE}" \
  "${SCHEMA_FAITHFUL_XML_PATCH_DIFF_SHA256}" "${SCHEMA_FAITHFUL_XML_PATCH_FILE}" \
  "${TOKEN_TEXT_PROVENANCE_PATCH_DIFF_SHA256}" "${TOKEN_TEXT_PROVENANCE_PATCH_FILE}" \
  "${PRECISE_REQUEST_ERRORS_PATCH_DIFF_SHA256}" "${PRECISE_REQUEST_ERRORS_PATCH_FILE}" \
  "${INPUT_STREAM_AGENT_IDENTITY_PATCH_DIFF_SHA256}" "${INPUT_STREAM_AGENT_IDENTITY_PATCH_FILE}" \
  "${XML_TEXT_FIDELITY_PATCH_DIFF_SHA256}" "${XML_TEXT_FIDELITY_PATCH_FILE}" \
  "${RAW_IMAGE_TRANSPORT_PATCH_DIFF_SHA256}" "${RAW_IMAGE_TRANSPORT_PATCH_FILE}" \
  "${GENERATE_RESULT_PATCH_DIFF_SHA256}" "${GENERATE_RESULT_PATCH_FILE}" \
  "${SAMPLING_BOUNDARY_PATCH_DIFF_SHA256}" "${SAMPLING_BOUNDARY_PATCH_FILE}" \
  "${TOOL_TRUNCATION_PATCH_DIFF_SHA256}" "${TOOL_TRUNCATION_PATCH_FILE}" \
  "${VISION_RUNTIME_PATCH_DIFF_SHA256}" "${VISION_RUNTIME_PATCH_FILE}" \
  "${NUMERICAL_AUDITS_PATCH_DIFF_SHA256}" "${NUMERICAL_AUDITS_PATCH_FILE}" \
  "${TURBOQUANT_GUARDS_PATCH_DIFF_SHA256}" "${TURBOQUANT_GUARDS_PATCH_FILE}" \
  "${KV_OFFLOAD_PINNING_PATCH_DIFF_SHA256}" "${KV_OFFLOAD_PINNING_PATCH_FILE}" \
  "${SHARED_PREFIX_CACHE_PATCH_DIFF_SHA256}" "${SHARED_PREFIX_CACHE_PATCH_FILE}" \
  "${EXACT_REASONING_USAGE_PATCH_DIFF_SHA256}" "${EXACT_REASONING_USAGE_PATCH_FILE}" \
  "${CANONICAL_FRAMING_PATCH_DIFF_SHA256}" "${CANONICAL_FRAMING_PATCH_FILE}" \
  "${QWEN_OWNED_GRAMMAR_PATCH_DIFF_SHA256}" "${QWEN_OWNED_GRAMMAR_PATCH_FILE}" | \
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
  -m unittest -v \
  patches.source_patch_v1.test_framework \
  patches.source_patch_v1.test_materialise_live_tree \
  scripts.runtime_image_unit

# The served chat template is a landmark-aware transformation, and
# it is proved here on the same terms as the runtime source stages: reconstructed from
# the model's own template through named stages and refused if it does not
# reproduce the published bytes. Before this existed the template's differences
# from the model's were pinned but not derived -- the bytes were fixed and
# nothing could say what change produced them.
docker run --rm \
  --network none \
  --read-only \
  --user "$(id -u):$(id -g)" \
  --tmpfs /tmp:rw,nodev,nosuid,size=64m \
  --env PYTHONPYCACHEPREFIX=/tmp/pycache \
  --entrypoint python3 \
  --volume "${PROJECT_DIR}:/project:ro" \
  --workdir /project \
  "${BASE_IMAGE_TAG}" \
  scripts/derive-chat-template.py \
    --project /project \
    --model "/project/${MODEL_DIR_NAME}" \
    --check

# Prove that the landmark-aware transaction recreates this exact worktree from
# the pinned upstream commit. The reviewed diffs are independently hashed and
# parsed as review evidence, but they never select mutation locations. The
# private worktree is discarded on every failure and never becomes a runtime.
VERIFY_WORKTREE="$(mktemp -d "${TMPDIR:-/tmp}/qwen38-vllm-verify.XXXXXX")"
remove_verify_worktree() {
  if ! git -C "${VLLM_DIR}" worktree remove --force "${VERIFY_WORKTREE}"; then
    printf 'ERROR: failed to remove the exact disposable verification worktree: %s\n' \
      "${VERIFY_WORKTREE}" >&2
    return 1
  fi
}
# The verification worktree exists only inside this section, so its cleanup
# takes the EXIT trap over from the export directory's and hands it back
# afterwards: a trap that merely replaced the earlier one would leave the
# export directory -- and, in build mode, the runtime archive written into
# it -- behind on every run.
cleanup_verify_worktree() {
  local status=$?
  trap - EXIT
  if ! remove_verify_worktree; then
    status=1
  fi
  cleanup_build_export
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

# The reconstruction is proved at this point: every stage applied from the
# pinned commit under complete pre/post hashes, and its worktree state is
# exactly the reviewed one. Only here, and only when the operator asked for it
# by name, may it be written to the unmanaged live tree. A reconstruction that
# did not verify has already exited above and is never written anywhere.
if [[ "${MODE}" == "materialise" ]]; then
  # Every identity this repository has itself shipped for a path, read out of
  # its own history: the FINAL_FILES of each committed revision of the
  # generated stage data. This is what lets "provably stale" be a statement
  # about committed data rather than about the tree being examined.
  SHIPPED_IDENTITIES="${BUILD_EXPORT_DIR}/shipped-identities"
  : >"${SHIPPED_IDENTITIES}"
  while IFS= read -r revision; do
    git -C "${PROJECT_DIR}" show "${revision}:${GENERATED_STAGES_REL}" \
      | sed -n '/^FINAL_FILES = {/,$p' \
      | sed -n "s/.*'\([^']\{1,\}\)': '\([0-9a-f]\{64\}\)'.*/\2 \1 ${revision}/p" \
      >>"${SHIPPED_IDENTITIES}"
  done < <(git -C "${PROJECT_DIR}" log --format=%H -- "${GENERATED_STAGES_REL}")
  if [[ ! -s "${SHIPPED_IDENTITIES}" ]]; then
    echo "MATERIALISE REFUSED: no committed final identity was found for" \
      "${GENERATED_STAGES_REL}; a tree without that history cannot prove" \
      "staleness, and nothing was written." >&2
    exit 1
  fi

  docker run --rm \
    --network none \
    --read-only \
    --user "$(id -u):$(id -g)" \
    --tmpfs /tmp:rw,nodev,nosuid,size=512m \
    --env PYTHONPYCACHEPREFIX=/tmp/pycache \
    --entrypoint python3 \
    --volume "${PROJECT_DIR}:/project:ro" \
    --volume "${VERIFY_WORKTREE}:/reconstruction:ro" \
    --volume "${VLLM_DIR}:/live:rw" \
    --volume "${SHIPPED_IDENTITIES}:/shipped:ro" \
    --workdir /project \
    "${BASE_IMAGE_TAG}" \
    -m patches.source_patch_v1.materialise_live_tree \
    --reconstruction /reconstruction --live /live --shipped /shipped
  remove_verify_worktree
  trap cleanup_build_export EXIT
  exit 0
fi

while IFS= read -r status_line; do
  relative_path="${status_line:3}"
  if [[ "${status_line:0:2}" == " D" ]]; then
    # A deletion is reproduced only when the path is absent on BOTH sides;
    # cmp cannot say that, and a survivor on either side is drift.
    if [[ -e "${VERIFY_WORKTREE}/${relative_path}" \
       || -e "${VLLM_DIR}/${relative_path}" ]]; then
      echo "Reviewed patches do not reproduce deletion of ${relative_path}." >&2
      echo "The reconstruction is authoritative and the live tree is behind it." >&2
      echo "Next: ./scripts/build-vllm.sh materialise" >&2
      exit 1
    fi
    continue
  fi
  if ! cmp -s \
    "${VERIFY_WORKTREE}/${relative_path}" \
    "${VLLM_DIR}/${relative_path}"; then
    echo "Reviewed patches do not reproduce ${relative_path}." >&2
    echo "The reconstruction is authoritative and the live tree is behind it." >&2
    echo "Next: ./scripts/build-vllm.sh materialise" >&2
    exit 1
  fi
done <<<"${EXPECTED_STATUS}"
remove_verify_worktree
trap cleanup_build_export EXIT

printf '%s  %s\n' \
  "${TURBOQUANT_PATCHED_FILE_SHA256}" "${VLLM_DIR}/${TURBOQUANT_REL}" \
  "${TURBOQUANT_DECODE_PATCHED_FILE_SHA256}" "${VLLM_DIR}/vllm/v1/attention/ops/triton_turboquant_decode.py" \
  "${TURBOQUANT_STORE_PATCHED_FILE_SHA256}" "${VLLM_DIR}/vllm/v1/attention/ops/triton_turboquant_store.py" \
  "${TOOL_SCHEMA_PATCHED_FILE_SHA256}" "${VLLM_DIR}/${TOOL_SCHEMA_REL}" \
  "${TOOL_PARSER_ABSTRACT_PATCHED_FILE_SHA256}" "${VLLM_DIR}/vllm/tool_parsers/abstract_tool_parser.py" \
  "${MODEL_CONFIG_PATCHED_FILE_SHA256}" "${VLLM_DIR}/${MODEL_CONFIG_REL}" \
  "${ANTHROPIC_PROTOCOL_PATCHED_FILE_SHA256}" "${VLLM_DIR}/${ANTHROPIC_PROTOCOL_REL}" \
  "${ANTHROPIC_SERVING_PATCHED_FILE_SHA256}" "${VLLM_DIR}/${ANTHROPIC_SERVING_REL}" \
  "${ERROR_RESPONSE_PATCHED_FILE_SHA256}" "${VLLM_DIR}/vllm/entrypoints/serve/exception_handling/error_response.py" \
  "${EXCEPTION_EXCEPTION_HANDLER_PATCHED_FILE_SHA256}" "${VLLM_DIR}/vllm/entrypoints/serve/exception_handling/handlers/exception.py" \
  "${HTTP_EXCEPTION_HANDLER_PATCHED_FILE_SHA256}" "${VLLM_DIR}/vllm/entrypoints/serve/exception_handling/handlers/http.py" \
  "${VALIDATION_EXCEPTION_HANDLER_PATCHED_FILE_SHA256}" "${VLLM_DIR}/vllm/entrypoints/serve/exception_handling/handlers/validation.py" \
  "${VLLM_ERROR_EXCEPTION_HANDLER_PATCHED_FILE_SHA256}" "${VLLM_DIR}/vllm/entrypoints/serve/exception_handling/handlers/vllm_error.py" \
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

printf '%s  %s\n' \
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

printf '%s  %s\n' \
  "${CACHE_CONFIG_PATCHED_FILE_SHA256}" "${VLLM_DIR}/${CACHE_CONFIG_REL}" \
  "${VLLM_CONFIG_PATCHED_FILE_SHA256}" "${VLLM_DIR}/${VLLM_CONFIG_REL}" \
  "${ARG_UTILS_PATCHED_FILE_SHA256}" "${VLLM_DIR}/${ARG_UTILS_REL}" \
  "${LLM_ENTRYPOINT_PATCHED_FILE_SHA256}" "${VLLM_DIR}/${LLM_ENTRYPOINT_REL}" \
  "${KV_CACHE_UTILS_PATCHED_FILE_SHA256}" "${VLLM_DIR}/${KV_CACHE_UTILS_REL}" \
  "${GPU_WORKER_PATCHED_FILE_SHA256}" "${VLLM_DIR}/${GPU_WORKER_REL}" \
  "${STARTUP_PLAN_PATCHED_FILE_SHA256}" "${VLLM_DIR}/${STARTUP_PLAN_REL}" \
  "${KV_OFFLOAD_CONFIG_PATCHED_FILE_SHA256}" "${VLLM_DIR}/${KV_OFFLOAD_CONFIG_REL}" \
  "${KV_OFFLOAD_BASE_PATCHED_FILE_SHA256}" "${VLLM_DIR}/${KV_OFFLOAD_BASE_REL}" \
  "${KV_OFFLOAD_CPU_SPEC_PATCHED_FILE_SHA256}" "${VLLM_DIR}/${KV_OFFLOAD_CPU_SPEC_REL}" \
  "${KV_OFFLOAD_CPU_MANAGER_PATCHED_FILE_SHA256}" "${VLLM_DIR}/${KV_OFFLOAD_CPU_MANAGER_REL}" \
  "${KV_TIERING_SPEC_PATCHED_FILE_SHA256}" "${VLLM_DIR}/${KV_TIERING_SPEC_REL}" \
  "${KV_TIERING_MANAGER_PATCHED_FILE_SHA256}" "${VLLM_DIR}/${KV_TIERING_MANAGER_REL}" \
  "${OFFLOAD_CONNECTOR_CONFIG_PATCHED_FILE_SHA256}" "${VLLM_DIR}/${OFFLOAD_CONNECTOR_CONFIG_REL}" \
  "${OFFLOAD_CONNECTOR_SCHEDULER_PATCHED_FILE_SHA256}" "${VLLM_DIR}/${OFFLOAD_CONNECTOR_SCHEDULER_REL}" \
  "${COMPLETION_PROTOCOL_PATCHED_FILE_SHA256}" "${VLLM_DIR}/${COMPLETION_PROTOCOL_REL}" \
  "${GENERATE_API_ROUTER_PATCHED_FILE_SHA256}" "${VLLM_DIR}/${GENERATE_API_ROUTER_REL}" \
  "${CLI_ARGS_PATCHED_FILE_SHA256}" "${VLLM_DIR}/${CLI_ARGS_REL}" \
  "${TITOTO_PROTOCOL_PATCHED_FILE_SHA256}" "${VLLM_DIR}/${TITOTO_PROTOCOL_REL}" \
  "${TITOTO_SERVING_PATCHED_FILE_SHA256}" "${VLLM_DIR}/${TITOTO_SERVING_REL}" \
  "${ABSTRACT_PARSER_PATCHED_FILE_SHA256}" "${VLLM_DIR}/${ABSTRACT_PARSER_REL}" \
  "${PARSER_ADAPTERS_PATCHED_FILE_SHA256}" "${VLLM_DIR}/${PARSER_ADAPTERS_REL}" \
  "${PARSER_EVENTS_PATCHED_FILE_SHA256}" "${VLLM_DIR}/vllm/parser/engine/events.py" \
  "${PARSER_ENGINE_CONFIG_PATCHED_FILE_SHA256}" "${VLLM_DIR}/vllm/parser/engine/parser_engine_config.py" \
  "${STREAMING_PARSER_ENGINE_PATCHED_FILE_SHA256}" "${VLLM_DIR}/vllm/parser/engine/streaming_parser_engine.py" \
  "${TOKEN_ID_SCANNER_PATCHED_FILE_SHA256}" "${VLLM_DIR}/vllm/parser/engine/token_id_scanner.py" \
  "${ENGINE_PROTOCOL_PATCHED_FILE_SHA256}" "${VLLM_DIR}/${ENGINE_PROTOCOL_REL}" | \
  sha256sum --check --strict

printf '%s  %s\n' \
  "${OFFLOADING_CONNECTOR_PATCHED_FILE_SHA256}" "${VLLM_DIR}/vllm/distributed/kv_transfer/kv_connector/v1/offloading_connector.py" \
  "${BLOCK_POOL_PATCHED_FILE_SHA256}" "${VLLM_DIR}/vllm/v1/core/block_pool.py" \
  "${KV_CACHE_COORDINATOR_PATCHED_FILE_SHA256}" "${VLLM_DIR}/vllm/v1/core/kv_cache_coordinator.py" \
  "${KV_CACHE_MANAGER_PATCHED_FILE_SHA256}" "${VLLM_DIR}/vllm/v1/core/kv_cache_manager.py" \
  "${PREFIX_CACHE_PATCHED_FILE_SHA256}" "${VLLM_DIR}/vllm/v1/core/prefix_cache.py" \
  "${SINGLE_TYPE_KV_CACHE_MANAGER_PATCHED_FILE_SHA256}" "${VLLM_DIR}/vllm/v1/core/single_type_kv_cache_manager.py" \
  "${KV_OFFLOAD_CPU_COMMON_PATCHED_FILE_SHA256}" "${VLLM_DIR}/vllm/v1/kv_offload/cpu/common.py" \
  "${SIMPLE_KV_OFFLOAD_MANAGER_PATCHED_FILE_SHA256}" "${VLLM_DIR}/vllm/v1/simple_kv_offload/manager.py" \
  "${SHARED_PREFIX_CACHE_UNIT_SHA256}" "${PROJECT_DIR}/scripts/shared_prefix_cache_unit.py" \
  "${RAW_MEDIA_UNIT_SHA256}" "${PROJECT_DIR}/scripts/raw_media_unit.py" \
  "${GENERATE_RESULT_UNIT_SHA256}" "${PROJECT_DIR}/scripts/generate_result_unit.py" | sha256sum --check --strict

printf '%s  %s\n' \
  "${COMPLETION_SERVING_PATCHED_FILE_SHA256}" "${VLLM_DIR}/vllm/entrypoints/openai/completion/serving.py" \
  "${RENDER_SERVING_PATCHED_FILE_SHA256}" "${VLLM_DIR}/vllm/entrypoints/scale_out/render/serving.py" \
  | sha256sum --check --strict

printf '%s  %s\n' \
  "${ONLINE_DERENDERER_PATCHED_FILE_SHA256}" "${VLLM_DIR}/vllm/renderers/online_derenderer.py" \
  | sha256sum --check --strict

printf '%s  %s\n' \
  "${DERENDER_SERVING_PATCHED_FILE_SHA256}" "${VLLM_DIR}/vllm/entrypoints/scale_out/derender/serving.py" \
  "${MM_PROCESSOR_INPUTS_PATCHED_FILE_SHA256}" "${VLLM_DIR}/vllm/multimodal/processing/inputs.py" \
  "${MM_PROCESSOR_PATCHED_FILE_SHA256}" "${VLLM_DIR}/vllm/multimodal/processing/processor.py" \
  "${BASE_RENDERER_PATCHED_FILE_SHA256}" "${VLLM_DIR}/vllm/renderers/base.py" \
  | sha256sum --check --strict

printf '%s  %s\n' \
  "${DEEPSEEK_V32_PARSER_PATCHED_FILE_SHA256}" "${VLLM_DIR}/vllm/parser/deepseek_v32.py" \
  "${DEEPSEEK_V4_PARSER_PATCHED_FILE_SHA256}" "${VLLM_DIR}/vllm/parser/deepseek_v4.py" \
  "${INKLING_PARSER_PATCHED_FILE_SHA256}" "${VLLM_DIR}/vllm/parser/inkling.py" \
  "${KIMI_K2_PARSER_PATCHED_FILE_SHA256}" "${VLLM_DIR}/vllm/parser/kimi_k2.py" \
  | sha256sum --check --strict

printf '%s  %s\n' \
  "${GEMMA4_PARSER_PATCHED_FILE_SHA256}" "${VLLM_DIR}/vllm/parser/gemma4.py" \
  "${GLM47_PARSER_PATCHED_FILE_SHA256}" "${VLLM_DIR}/vllm/parser/glm47_moe.py" \
  "${MINIMAX_M2_PARSER_PATCHED_FILE_SHA256}" "${VLLM_DIR}/vllm/parser/minimax_m2.py" \
  "${MISTRAL_PARSER_PATCHED_FILE_SHA256}" "${VLLM_DIR}/vllm/parser/mistral.py" \
  | sha256sum --check --strict

printf '%s  %s\n' \
  "${ASYNC_LLM_PATCHED_FILE_SHA256}" "${VLLM_DIR}/vllm/v1/engine/async_llm.py" \
  | sha256sum --check --strict

printf '%s  %s\n' \
  "${V1_SCHEDULER_PATCHED_FILE_SHA256}" "${VLLM_DIR}/vllm/v1/core/sched/scheduler.py" \
  | sha256sum --check --strict

printf '%s  %s\n' \
  "${V1_DETOKENIZER_PATCHED_FILE_SHA256}" "${VLLM_DIR}/vllm/v1/engine/detokenizer.py" \
  | sha256sum --check --strict

printf '%s  %s\n' \
  "${V1_OUTPUT_PROCESSOR_PATCHED_FILE_SHA256}" "${VLLM_DIR}/vllm/v1/engine/output_processor.py" \
  | sha256sum --check --strict

printf '%s  %s\n' \
  "${STRUCTURED_OUTPUT_BACKEND_TYPES_PATCHED_FILE_SHA256}" "${VLLM_DIR}/vllm/v1/structured_output/backend_types.py" \
  | sha256sum --check --strict

printf '%s  %s\n' \
  "${XGRAMMAR_BACKEND_PATCHED_FILE_SHA256}" "${VLLM_DIR}/vllm/v1/structured_output/backend_xgrammar.py" \
  | sha256sum --check --strict

printf '%s  %s\n' \
  "${STRUCTURAL_TAG_STOP_CHECKER_PATCHED_FILE_SHA256}" "${VLLM_DIR}/vllm/v1/structured_output/stop_checker.py" \
  | sha256sum --check --strict

printf '%s  %s\n' \
  "${REASONING_CONFIG_PATCHED_FILE_SHA256}" "${VLLM_DIR}/vllm/config/reasoning.py" \
  | sha256sum --check --strict

printf '%s  %s\n' \
  "${BASE_REASONING_PARSER_PATCHED_FILE_SHA256}" "${VLLM_DIR}/vllm/reasoning/abs_reasoning_parsers.py" \
  | sha256sum --check --strict

printf '%s  %s\n' \
  "${V1_THINKING_BUDGET_PATCHED_FILE_SHA256}" "${VLLM_DIR}/vllm/v1/sample/thinking_budget_state.py" \
  | sha256sum --check --strict

printf '%s  %s\n' \
  "${TOOL_PARSER_UTILS_PATCHED_FILE_SHA256}" "${VLLM_DIR}/vllm/tool_parsers/utils.py" \
  | sha256sum --check --strict

printf '%s  %s\n' \
  "${DETOKENIZER_UTILS_PATCHED_FILE_SHA256}" "${VLLM_DIR}/vllm/tokenizers/detokenizer_utils.py" \
  | sha256sum --check --strict

printf '%s  %s\n' \
  "${EXCEPTION_REGISTRATION_PATCHED_FILE_SHA256}" "${VLLM_DIR}/vllm/entrypoints/serve/exception_handling/register.py" \
  | sha256sum --check --strict

printf '%s  %s\n' \
  "${HF_RENDERER_PATCHED_FILE_SHA256}" "${VLLM_DIR}/vllm/renderers/hf.py" \
  | sha256sum --check --strict

printf '%s  %s\n' \
  "${RUNTIME_DOCKERFILE_SHA256}" "${DOCKERFILE}" \
  "${DOCKERIGNORE_SHA256}" "${DOCKERIGNORE}" | \
  sha256sum --check --strict

printf '%s  %s\n' \
  "${TURBOQUANT_GUARD_UNIT_SHA256}" "${PROJECT_DIR}/scripts/turboquant_guard_unit.py" \
  "${TURBOQUANT_K8V4_UNIT_SHA256}" "${TURBOQUANT_K8V4_UNIT_FILE}" \
  "${QWEN38_CONTEXT_UNIT_SHA256}" "${QWEN38_CONTEXT_UNIT_FILE}" \
  "${CHAT_TEMPLATE_RETENTION_UNIT_SHA256}" "${CHAT_TEMPLATE_RETENTION_UNIT_FILE}" \
  "${NVFP4_KERNEL_UNIT_SHA256}" "${NVFP4_KERNEL_UNIT_FILE}" \
  "${TOOL_OUTPUT_PARSER_UNIT_SHA256}" "${TOOL_OUTPUT_PARSER_UNIT_FILE}" \
  "${REASONING_USAGE_UNIT_SHA256}" "${REASONING_USAGE_UNIT_FILE}" \
  "${QWEN_GRAMMAR_UNIT_SHA256}" "${QWEN_GRAMMAR_UNIT_FILE}" | \
  sha256sum --check --strict

docker run --rm --network none --read-only \
  --user "$(id -u):$(id -g)" \
  --tmpfs /tmp:rw,nodev,nosuid,size=128m \
  --env PYTHONPYCACHEPREFIX=/tmp/pycache \
  --env CUDA_VISIBLE_DEVICES= --env TRITON_INTERPRET=1 \
  --volume "${PROJECT_DIR}:/project:ro" \
  --volume "${VLLM_DIR}/vllm/v1/attention/ops/triton_turboquant_store.py:/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/ops/triton_turboquant_store.py:ro" \
  --volume "${VLLM_DIR}/vllm/v1/attention/ops/triton_turboquant_decode.py:/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/ops/triton_turboquant_decode.py:ro" \
  --entrypoint python3 "${BASE_IMAGE_TAG}" /project/scripts/turboquant_guard_unit.py

# Execute CPU contract units against the complete reviewed runtime overlay.
parser_unit_mounts=()
while IFS= read -r status_line; do
  case "${status_line}" in
    " M vllm/"*|"?? vllm/"*)
      path="${status_line:3}"
      parser_unit_mounts+=(--volume "${VLLM_DIR}/${path}:/usr/local/lib/python3.12/dist-packages/${path}:ro")
      ;;
  esac
done <<<"${EXPECTED_STATUS}"
for unit in chat_template_retention_unit tool_output_parser_unit vision_contract_unit reasoning_usage_unit shared_prefix_cache_unit phase_budget_unit generate_result_unit raw_media_unit qwen_grammar_unit; do
  docker run --rm --network none --read-only --user "$(id -u):$(id -g)" \
    --tmpfs /tmp:rw,nodev,nosuid,size=256m \
    --env PYTHONDONTWRITEBYTECODE=1 --env CUDA_VISIBLE_DEVICES= \
    --volume "${TEMPLATE_FILE}:/opt/qwen38/chat_template.jinja:ro" --volume "${PROJECT_DIR}:/project:ro" "${parser_unit_mounts[@]}" \
    --entrypoint python3 "${BASE_IMAGE_TAG}" "/project/scripts/${unit}.py"
done

git -C "${VLLM_DIR}" diff --check

if [[ "${MODE}" == "check" ]]; then
  # Every count below is derived from the objects this run just verified —
  # EXPECTED_STATUS and the deployment-input manifest — never restated by
  # hand: a hand count here is one more copy that can drift from the thing
  # it describes.
  modified_runtime_count="$(grep -c '^ M vllm/' <<<"${EXPECTED_STATUS}" || :)"
  new_runtime_count="$(grep -c '^?? vllm/' <<<"${EXPECTED_STATUS}" || :)"
  deleted_runtime_count="$(grep -c '^ D vllm/' <<<"${EXPECTED_STATUS}" || :)"
  modified_test_count="$(grep -c '^ M tests/' <<<"${EXPECTED_STATUS}" || :)"
  new_test_count="$(grep -c '^?? tests/' <<<"${EXPECTED_STATUS}" || :)"
  deleted_test_count="$(grep -c '^ D tests/' <<<"${EXPECTED_STATUS}" || :)"
  modified_doc_count="$(grep -c '^ M docs/' <<<"${EXPECTED_STATUS}" || :)"
  review_diff_count="$(grep -c '^[0-9a-f]\{64\}  patches/vllm-.*\.patch$' \
    "${DEPLOYMENT_INPUT_MANIFEST}" || :)"
  echo "Pinned base image, vLLM commit, transactional landmark patcher," \
    "${modified_runtime_count} reviewed modified runtime source files," \
    "${new_runtime_count} reviewed new runtime source files," \
    "${deleted_runtime_count} reviewed runtime source deletions," \
    "${modified_test_count} reviewed modified test files," \
    "${new_test_count} reviewed new test files," \
    "${deleted_test_count} reviewed test deletions," \
    "${modified_doc_count} reviewed modified documentation files," \
    "${review_diff_count} review diffs, agent template, numerical audit" \
    "units, and all build units are exact."
  exit 0
fi

docker buildx build --progress=plain \
  --builder default \
  --platform linux/amd64 \
  --network none \
  --pull=false \
  --provenance=false \
  --no-cache \
  --target runtime \
  --build-arg "BASE_IMAGE=${BASE_IMAGE_TAG}" \
  --build-arg "OFFLOADING_CONNECTOR_UPSTREAM_FILE_SHA256=${OFFLOADING_CONNECTOR_UPSTREAM_FILE_SHA256}" \
  --build-arg "OFFLOADING_CONNECTOR_PATCHED_FILE_SHA256=${OFFLOADING_CONNECTOR_PATCHED_FILE_SHA256}" \
  --build-arg "BLOCK_POOL_UPSTREAM_FILE_SHA256=${BLOCK_POOL_UPSTREAM_FILE_SHA256}" \
  --build-arg "BLOCK_POOL_PATCHED_FILE_SHA256=${BLOCK_POOL_PATCHED_FILE_SHA256}" \
  --build-arg "KV_CACHE_COORDINATOR_UPSTREAM_FILE_SHA256=${KV_CACHE_COORDINATOR_UPSTREAM_FILE_SHA256}" \
  --build-arg "KV_CACHE_COORDINATOR_PATCHED_FILE_SHA256=${KV_CACHE_COORDINATOR_PATCHED_FILE_SHA256}" \
  --build-arg "KV_CACHE_MANAGER_UPSTREAM_FILE_SHA256=${KV_CACHE_MANAGER_UPSTREAM_FILE_SHA256}" \
  --build-arg "KV_CACHE_MANAGER_PATCHED_FILE_SHA256=${KV_CACHE_MANAGER_PATCHED_FILE_SHA256}" \
  --build-arg "PREFIX_CACHE_PATCHED_FILE_SHA256=${PREFIX_CACHE_PATCHED_FILE_SHA256}" \
  --build-arg "SINGLE_TYPE_KV_CACHE_MANAGER_UPSTREAM_FILE_SHA256=${SINGLE_TYPE_KV_CACHE_MANAGER_UPSTREAM_FILE_SHA256}" \
  --build-arg "SINGLE_TYPE_KV_CACHE_MANAGER_PATCHED_FILE_SHA256=${SINGLE_TYPE_KV_CACHE_MANAGER_PATCHED_FILE_SHA256}" \
  --build-arg "KV_OFFLOAD_CPU_COMMON_UPSTREAM_FILE_SHA256=${KV_OFFLOAD_CPU_COMMON_UPSTREAM_FILE_SHA256}" \
  --build-arg "KV_OFFLOAD_CPU_COMMON_PATCHED_FILE_SHA256=${KV_OFFLOAD_CPU_COMMON_PATCHED_FILE_SHA256}" \
  --build-arg "SIMPLE_KV_OFFLOAD_MANAGER_UPSTREAM_FILE_SHA256=${SIMPLE_KV_OFFLOAD_MANAGER_UPSTREAM_FILE_SHA256}" \
  --build-arg "SIMPLE_KV_OFFLOAD_MANAGER_PATCHED_FILE_SHA256=${SIMPLE_KV_OFFLOAD_MANAGER_PATCHED_FILE_SHA256}" \
  --build-arg "SHARED_PREFIX_CACHE_UNIT_SHA256=${SHARED_PREFIX_CACHE_UNIT_SHA256}" \
  --build-arg "RAW_MEDIA_UNIT_SHA256=${RAW_MEDIA_UNIT_SHA256}" \
  --build-arg "GENERATE_RESULT_UNIT_SHA256=${GENERATE_RESULT_UNIT_SHA256}" \
  --build-arg "TURBOQUANT_UPSTREAM_FILE_SHA256=${TURBOQUANT_UPSTREAM_FILE_SHA256}" \
  --build-arg "TURBOQUANT_DECODE_UPSTREAM_FILE_SHA256=${TURBOQUANT_DECODE_UPSTREAM_FILE_SHA256}" \
  --build-arg "TURBOQUANT_STORE_UPSTREAM_FILE_SHA256=${TURBOQUANT_STORE_UPSTREAM_FILE_SHA256}" \
  --build-arg "TOOL_SCHEMA_UPSTREAM_FILE_SHA256=${TOOL_SCHEMA_UPSTREAM_FILE_SHA256}" \
  --build-arg "TOOL_PARSER_ABSTRACT_UPSTREAM_FILE_SHA256=${TOOL_PARSER_ABSTRACT_UPSTREAM_FILE_SHA256}" \
  --build-arg "TOOL_CALL_FILTER_UPSTREAM_FILE_SHA256=${TOOL_CALL_FILTER_UPSTREAM_FILE_SHA256}" \
  --build-arg "MODEL_CONFIG_UPSTREAM_FILE_SHA256=${MODEL_CONFIG_UPSTREAM_FILE_SHA256}" \
  --build-arg "ANTHROPIC_PROTOCOL_UPSTREAM_FILE_SHA256=${ANTHROPIC_PROTOCOL_UPSTREAM_FILE_SHA256}" \
  --build-arg "ANTHROPIC_SERVING_UPSTREAM_FILE_SHA256=${ANTHROPIC_SERVING_UPSTREAM_FILE_SHA256}" \
  --build-arg "ERROR_RESPONSE_UPSTREAM_FILE_SHA256=${ERROR_RESPONSE_UPSTREAM_FILE_SHA256}" \
  --build-arg "EXCEPTION_EXCEPTION_HANDLER_UPSTREAM_FILE_SHA256=${EXCEPTION_EXCEPTION_HANDLER_UPSTREAM_FILE_SHA256}" \
  --build-arg "HTTP_EXCEPTION_HANDLER_UPSTREAM_FILE_SHA256=${HTTP_EXCEPTION_HANDLER_UPSTREAM_FILE_SHA256}" \
  --build-arg "VALIDATION_EXCEPTION_HANDLER_UPSTREAM_FILE_SHA256=${VALIDATION_EXCEPTION_HANDLER_UPSTREAM_FILE_SHA256}" \
  --build-arg "VLLM_ERROR_EXCEPTION_HANDLER_UPSTREAM_FILE_SHA256=${VLLM_ERROR_EXCEPTION_HANDLER_UPSTREAM_FILE_SHA256}" \
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
  --build-arg "ABSTRACT_PARSER_UPSTREAM_FILE_SHA256=${ABSTRACT_PARSER_UPSTREAM_FILE_SHA256}" \
  --build-arg "PARSER_ADAPTERS_UPSTREAM_FILE_SHA256=${PARSER_ADAPTERS_UPSTREAM_FILE_SHA256}" \
  --build-arg "PARSER_EVENTS_UPSTREAM_FILE_SHA256=${PARSER_EVENTS_UPSTREAM_FILE_SHA256}" \
  --build-arg "PARSER_ENGINE_CONFIG_UPSTREAM_FILE_SHA256=${PARSER_ENGINE_CONFIG_UPSTREAM_FILE_SHA256}" \
  --build-arg "STREAMING_PARSER_ENGINE_UPSTREAM_FILE_SHA256=${STREAMING_PARSER_ENGINE_UPSTREAM_FILE_SHA256}" \
  --build-arg "TOKEN_ID_SCANNER_UPSTREAM_FILE_SHA256=${TOKEN_ID_SCANNER_UPSTREAM_FILE_SHA256}" \
  --build-arg "ENGINE_PROTOCOL_UPSTREAM_FILE_SHA256=${ENGINE_PROTOCOL_UPSTREAM_FILE_SHA256}" \
  --build-arg "TURBOQUANT_PATCHED_FILE_SHA256=${TURBOQUANT_PATCHED_FILE_SHA256}" \
  --build-arg "TURBOQUANT_DECODE_PATCHED_FILE_SHA256=${TURBOQUANT_DECODE_PATCHED_FILE_SHA256}" \
  --build-arg "TURBOQUANT_STORE_PATCHED_FILE_SHA256=${TURBOQUANT_STORE_PATCHED_FILE_SHA256}" \
  --build-arg "TOOL_SCHEMA_PATCHED_FILE_SHA256=${TOOL_SCHEMA_PATCHED_FILE_SHA256}" \
  --build-arg "TOOL_PARSER_ABSTRACT_PATCHED_FILE_SHA256=${TOOL_PARSER_ABSTRACT_PATCHED_FILE_SHA256}" \
  --build-arg "MODEL_CONFIG_PATCHED_FILE_SHA256=${MODEL_CONFIG_PATCHED_FILE_SHA256}" \
  --build-arg "ANTHROPIC_PROTOCOL_PATCHED_FILE_SHA256=${ANTHROPIC_PROTOCOL_PATCHED_FILE_SHA256}" \
  --build-arg "ANTHROPIC_SERVING_PATCHED_FILE_SHA256=${ANTHROPIC_SERVING_PATCHED_FILE_SHA256}" \
  --build-arg "ERROR_RESPONSE_PATCHED_FILE_SHA256=${ERROR_RESPONSE_PATCHED_FILE_SHA256}" \
  --build-arg "EXCEPTION_EXCEPTION_HANDLER_PATCHED_FILE_SHA256=${EXCEPTION_EXCEPTION_HANDLER_PATCHED_FILE_SHA256}" \
  --build-arg "HTTP_EXCEPTION_HANDLER_PATCHED_FILE_SHA256=${HTTP_EXCEPTION_HANDLER_PATCHED_FILE_SHA256}" \
  --build-arg "VALIDATION_EXCEPTION_HANDLER_PATCHED_FILE_SHA256=${VALIDATION_EXCEPTION_HANDLER_PATCHED_FILE_SHA256}" \
  --build-arg "VLLM_ERROR_EXCEPTION_HANDLER_PATCHED_FILE_SHA256=${VLLM_ERROR_EXCEPTION_HANDLER_PATCHED_FILE_SHA256}" \
  --build-arg "CHAT_PROTOCOL_PATCHED_FILE_SHA256=${CHAT_PROTOCOL_PATCHED_FILE_SHA256}" \
  --build-arg "SAMPLING_PARAMS_PATCHED_FILE_SHA256=${SAMPLING_PARAMS_PATCHED_FILE_SHA256}" \
  --build-arg "SCHED_UTILS_PATCHED_FILE_SHA256=${SCHED_UTILS_PATCHED_FILE_SHA256}" \
  --build-arg "INPUT_PROCESSOR_PATCHED_FILE_SHA256=${INPUT_PROCESSOR_PATCHED_FILE_SHA256}" \
  --build-arg "REQUEST_PATCHED_FILE_SHA256=${REQUEST_PATCHED_FILE_SHA256}" \
  --build-arg "QWEN3_PARSER_PATCHED_FILE_SHA256=${QWEN3_PARSER_PATCHED_FILE_SHA256}" \
  --build-arg "STRUCTURED_OUTPUT_PATCHED_FILE_SHA256=${STRUCTURED_OUTPUT_PATCHED_FILE_SHA256}" \
  --build-arg "HF_RENDERER_PATCHED_FILE_SHA256=${HF_RENDERER_PATCHED_FILE_SHA256}" \
  --build-arg "HF_RENDERER_UPSTREAM_FILE_SHA256=${HF_RENDERER_UPSTREAM_FILE_SHA256}" \
  --build-arg "EXCEPTION_REGISTRATION_PATCHED_FILE_SHA256=${EXCEPTION_REGISTRATION_PATCHED_FILE_SHA256}" \
  --build-arg "EXCEPTION_REGISTRATION_UPSTREAM_FILE_SHA256=${EXCEPTION_REGISTRATION_UPSTREAM_FILE_SHA256}" \
  --build-arg "DETOKENIZER_UTILS_PATCHED_FILE_SHA256=${DETOKENIZER_UTILS_PATCHED_FILE_SHA256}" \
  --build-arg "DETOKENIZER_UTILS_UPSTREAM_FILE_SHA256=${DETOKENIZER_UTILS_UPSTREAM_FILE_SHA256}" \
  --build-arg "TOOL_PARSER_UTILS_PATCHED_FILE_SHA256=${TOOL_PARSER_UTILS_PATCHED_FILE_SHA256}" \
  --build-arg "TOOL_PARSER_UTILS_UPSTREAM_FILE_SHA256=${TOOL_PARSER_UTILS_UPSTREAM_FILE_SHA256}" \
  --build-arg "V1_THINKING_BUDGET_PATCHED_FILE_SHA256=${V1_THINKING_BUDGET_PATCHED_FILE_SHA256}" \
  --build-arg "V1_THINKING_BUDGET_UPSTREAM_FILE_SHA256=${V1_THINKING_BUDGET_UPSTREAM_FILE_SHA256}" \
  --build-arg "BASE_REASONING_PARSER_PATCHED_FILE_SHA256=${BASE_REASONING_PARSER_PATCHED_FILE_SHA256}" \
  --build-arg "BASE_REASONING_PARSER_UPSTREAM_FILE_SHA256=${BASE_REASONING_PARSER_UPSTREAM_FILE_SHA256}" \
  --build-arg "REASONING_CONFIG_PATCHED_FILE_SHA256=${REASONING_CONFIG_PATCHED_FILE_SHA256}" \
  --build-arg "REASONING_CONFIG_UPSTREAM_FILE_SHA256=${REASONING_CONFIG_UPSTREAM_FILE_SHA256}" \
  --build-arg "STRUCTURAL_TAG_STOP_CHECKER_PATCHED_FILE_SHA256=${STRUCTURAL_TAG_STOP_CHECKER_PATCHED_FILE_SHA256}" \
  --build-arg "XGRAMMAR_BACKEND_PATCHED_FILE_SHA256=${XGRAMMAR_BACKEND_PATCHED_FILE_SHA256}" \
  --build-arg "XGRAMMAR_BACKEND_UPSTREAM_FILE_SHA256=${XGRAMMAR_BACKEND_UPSTREAM_FILE_SHA256}" \
  --build-arg "STRUCTURED_OUTPUT_BACKEND_TYPES_PATCHED_FILE_SHA256=${STRUCTURED_OUTPUT_BACKEND_TYPES_PATCHED_FILE_SHA256}" \
  --build-arg "STRUCTURED_OUTPUT_BACKEND_TYPES_UPSTREAM_FILE_SHA256=${STRUCTURED_OUTPUT_BACKEND_TYPES_UPSTREAM_FILE_SHA256}" \
  --build-arg "V1_OUTPUT_PROCESSOR_PATCHED_FILE_SHA256=${V1_OUTPUT_PROCESSOR_PATCHED_FILE_SHA256}" \
  --build-arg "V1_OUTPUT_PROCESSOR_UPSTREAM_FILE_SHA256=${V1_OUTPUT_PROCESSOR_UPSTREAM_FILE_SHA256}" \
  --build-arg "V1_DETOKENIZER_PATCHED_FILE_SHA256=${V1_DETOKENIZER_PATCHED_FILE_SHA256}" \
  --build-arg "V1_DETOKENIZER_UPSTREAM_FILE_SHA256=${V1_DETOKENIZER_UPSTREAM_FILE_SHA256}" \
  --build-arg "V1_SCHEDULER_PATCHED_FILE_SHA256=${V1_SCHEDULER_PATCHED_FILE_SHA256}" \
  --build-arg "V1_SCHEDULER_UPSTREAM_FILE_SHA256=${V1_SCHEDULER_UPSTREAM_FILE_SHA256}" \
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
  --build-arg "TURBOQUANT_GUARD_UNIT_SHA256=${TURBOQUANT_GUARD_UNIT_SHA256}" \
  --build-arg "TURBOQUANT_K8V4_UNIT_SHA256=${TURBOQUANT_K8V4_UNIT_SHA256}" \
  --build-arg "QWEN38_CONTEXT_UNIT_SHA256=${QWEN38_CONTEXT_UNIT_SHA256}" \
  --build-arg "CHAT_TEMPLATE_RETENTION_UNIT_SHA256=${CHAT_TEMPLATE_RETENTION_UNIT_SHA256}" \
  --build-arg "NVFP4_KERNEL_UNIT_SHA256=${NVFP4_KERNEL_UNIT_SHA256}" \
  --build-arg "REASONING_USAGE_UNIT_SHA256=${REASONING_USAGE_UNIT_SHA256}" \
  --build-arg "TOOL_OUTPUT_PARSER_UNIT_SHA256=${TOOL_OUTPUT_PARSER_UNIT_SHA256}" \
  --build-arg "QWEN_GRAMMAR_UNIT_SHA256=${QWEN_GRAMMAR_UNIT_SHA256}" \
  --build-arg "TURBOQUANT_PATCH_DIFF_SHA256=${TURBOQUANT_PATCH_DIFF_SHA256}" \
  --build-arg "TOOL_SCHEMA_PATCH_DIFF_SHA256=${TOOL_SCHEMA_PATCH_DIFF_SHA256}" \
  --build-arg "AGENT_DEFAULTS_PATCH_DIFF_SHA256=${AGENT_DEFAULTS_PATCH_DIFF_SHA256}" \
  --build-arg "PHASE_BUDGET_PATCH_DIFF_SHA256=${PHASE_BUDGET_PATCH_DIFF_SHA256}" \
  --build-arg "IMPLICIT_TOOL_GRAMMAR_PATCH_DIFF_SHA256=${IMPLICIT_TOOL_GRAMMAR_PATCH_DIFF_SHA256}" \
  --build-arg "ANTHROPIC_VALIDATION_PATCH_DIFF_SHA256=${ANTHROPIC_VALIDATION_PATCH_DIFF_SHA256}" \
  --build-arg "ANTHROPIC_INPUTS_PATCH_DIFF_SHA256=${ANTHROPIC_INPUTS_PATCH_DIFF_SHA256}" \
  --build-arg "QWEN_LANGUAGE_PATCH_DIFF_SHA256=${QWEN_LANGUAGE_PATCH_DIFF_SHA256}" \
  --build-arg "PNG_SOURCE_PATCH_DIFF_SHA256=${PNG_SOURCE_PATCH_DIFF_SHA256}" \
  --build-arg "KV_PHYSICAL_PATCH_DIFF_SHA256=${KV_PHYSICAL_PATCH_DIFF_SHA256}" \
  --build-arg "SINGLE_CALL_PATCH_DIFF_SHA256=${SINGLE_CALL_PATCH_DIFF_SHA256}" \
  --build-arg "RESPONSES_HISTORY_PATCH_DIFF_SHA256=${RESPONSES_HISTORY_PATCH_DIFF_SHA256}" \
  --build-arg "RESPONSES_IDENTITY_PATCH_DIFF_SHA256=${RESPONSES_IDENTITY_PATCH_DIFF_SHA256}" \
  --build-arg "ANTHROPIC_TERMINAL_PATCH_DIFF_SHA256=${ANTHROPIC_TERMINAL_PATCH_DIFF_SHA256}" \
  --build-arg "SAMPLING_RESOLUTION_PATCH_DIFF_SHA256=${SAMPLING_RESOLUTION_PATCH_DIFF_SHA256}" \
  --build-arg "GEMMA4_PARSER_UPSTREAM_FILE_SHA256=${GEMMA4_PARSER_UPSTREAM_FILE_SHA256}" \
  --build-arg "GEMMA4_PARSER_PATCHED_FILE_SHA256=${GEMMA4_PARSER_PATCHED_FILE_SHA256}" \
  --build-arg "GLM47_PARSER_UPSTREAM_FILE_SHA256=${GLM47_PARSER_UPSTREAM_FILE_SHA256}" \
  --build-arg "GLM47_PARSER_PATCHED_FILE_SHA256=${GLM47_PARSER_PATCHED_FILE_SHA256}" \
  --build-arg "MINIMAX_M2_PARSER_UPSTREAM_FILE_SHA256=${MINIMAX_M2_PARSER_UPSTREAM_FILE_SHA256}" \
  --build-arg "MINIMAX_M2_PARSER_PATCHED_FILE_SHA256=${MINIMAX_M2_PARSER_PATCHED_FILE_SHA256}" \
  --build-arg "MISTRAL_PARSER_UPSTREAM_FILE_SHA256=${MISTRAL_PARSER_UPSTREAM_FILE_SHA256}" \
  --build-arg "MISTRAL_PARSER_PATCHED_FILE_SHA256=${MISTRAL_PARSER_PATCHED_FILE_SHA256}" \
  --build-arg "PHASE_AWARE_PARSER_TERMINALS_PATCH_DIFF_SHA256=${PHASE_AWARE_PARSER_TERMINALS_PATCH_DIFF_SHA256}" \
  --build-arg "TOOL_OUTPUT_COMPLETION_PATCH_DIFF_SHA256=${TOOL_OUTPUT_COMPLETION_PATCH_DIFF_SHA256}" \
  --build-arg "ONE_WAY_THINKING_BOUNDARY_PATCH_DIFF_SHA256=${ONE_WAY_THINKING_BOUNDARY_PATCH_DIFF_SHA256}" \
  --build-arg "SCHEMA_FAITHFUL_XML_PATCH_DIFF_SHA256=${SCHEMA_FAITHFUL_XML_PATCH_DIFF_SHA256}" \
  --build-arg "TOKEN_TEXT_PROVENANCE_PATCH_DIFF_SHA256=${TOKEN_TEXT_PROVENANCE_PATCH_DIFF_SHA256}" \
  --build-arg "PRECISE_REQUEST_ERRORS_PATCH_DIFF_SHA256=${PRECISE_REQUEST_ERRORS_PATCH_DIFF_SHA256}" \
  --build-arg "INPUT_STREAM_AGENT_IDENTITY_PATCH_DIFF_SHA256=${INPUT_STREAM_AGENT_IDENTITY_PATCH_DIFF_SHA256}" \
  --build-arg "ASYNC_LLM_UPSTREAM_FILE_SHA256=${ASYNC_LLM_UPSTREAM_FILE_SHA256}" \
  --build-arg "ASYNC_LLM_PATCHED_FILE_SHA256=${ASYNC_LLM_PATCHED_FILE_SHA256}" \
  --build-arg "DEEPSEEK_V32_PARSER_UPSTREAM_FILE_SHA256=${DEEPSEEK_V32_PARSER_UPSTREAM_FILE_SHA256}" \
  --build-arg "DEEPSEEK_V32_PARSER_PATCHED_FILE_SHA256=${DEEPSEEK_V32_PARSER_PATCHED_FILE_SHA256}" \
  --build-arg "DEEPSEEK_V4_PARSER_UPSTREAM_FILE_SHA256=${DEEPSEEK_V4_PARSER_UPSTREAM_FILE_SHA256}" \
  --build-arg "DEEPSEEK_V4_PARSER_PATCHED_FILE_SHA256=${DEEPSEEK_V4_PARSER_PATCHED_FILE_SHA256}" \
  --build-arg "INKLING_PARSER_UPSTREAM_FILE_SHA256=${INKLING_PARSER_UPSTREAM_FILE_SHA256}" \
  --build-arg "INKLING_PARSER_PATCHED_FILE_SHA256=${INKLING_PARSER_PATCHED_FILE_SHA256}" \
  --build-arg "KIMI_K2_PARSER_UPSTREAM_FILE_SHA256=${KIMI_K2_PARSER_UPSTREAM_FILE_SHA256}" \
  --build-arg "KIMI_K2_PARSER_PATCHED_FILE_SHA256=${KIMI_K2_PARSER_PATCHED_FILE_SHA256}" \
  --build-arg "XML_TEXT_FIDELITY_PATCH_DIFF_SHA256=${XML_TEXT_FIDELITY_PATCH_DIFF_SHA256}" \
  --build-arg "DERENDER_SERVING_UPSTREAM_FILE_SHA256=${DERENDER_SERVING_UPSTREAM_FILE_SHA256}" \
  --build-arg "DERENDER_SERVING_PATCHED_FILE_SHA256=${DERENDER_SERVING_PATCHED_FILE_SHA256}" \
  --build-arg "MM_PROCESSOR_INPUTS_UPSTREAM_FILE_SHA256=${MM_PROCESSOR_INPUTS_UPSTREAM_FILE_SHA256}" \
  --build-arg "MM_PROCESSOR_INPUTS_PATCHED_FILE_SHA256=${MM_PROCESSOR_INPUTS_PATCHED_FILE_SHA256}" \
  --build-arg "MM_PROCESSOR_UPSTREAM_FILE_SHA256=${MM_PROCESSOR_UPSTREAM_FILE_SHA256}" \
  --build-arg "MM_PROCESSOR_PATCHED_FILE_SHA256=${MM_PROCESSOR_PATCHED_FILE_SHA256}" \
  --build-arg "BASE_RENDERER_UPSTREAM_FILE_SHA256=${BASE_RENDERER_UPSTREAM_FILE_SHA256}" \
  --build-arg "BASE_RENDERER_PATCHED_FILE_SHA256=${BASE_RENDERER_PATCHED_FILE_SHA256}" \
  --build-arg "RAW_IMAGE_TRANSPORT_PATCH_DIFF_SHA256=${RAW_IMAGE_TRANSPORT_PATCH_DIFF_SHA256}" \
  --build-arg "MM_SERDE_UPSTREAM_FILE_SHA256=${MM_SERDE_UPSTREAM_FILE_SHA256}" \
  --build-arg "ONLINE_DERENDERER_UPSTREAM_FILE_SHA256=${ONLINE_DERENDERER_UPSTREAM_FILE_SHA256}" \
  --build-arg "ONLINE_DERENDERER_PATCHED_FILE_SHA256=${ONLINE_DERENDERER_PATCHED_FILE_SHA256}" \
  --build-arg "GENERATE_RESULT_PATCH_DIFF_SHA256=${GENERATE_RESULT_PATCH_DIFF_SHA256}" \
  --build-arg "COMPLETION_SERVING_UPSTREAM_FILE_SHA256=${COMPLETION_SERVING_UPSTREAM_FILE_SHA256}" \
  --build-arg "COMPLETION_SERVING_PATCHED_FILE_SHA256=${COMPLETION_SERVING_PATCHED_FILE_SHA256}" \
  --build-arg "RENDER_SERVING_UPSTREAM_FILE_SHA256=${RENDER_SERVING_UPSTREAM_FILE_SHA256}" \
  --build-arg "RENDER_SERVING_PATCHED_FILE_SHA256=${RENDER_SERVING_PATCHED_FILE_SHA256}" \
  --build-arg "SAMPLING_BOUNDARY_PATCH_DIFF_SHA256=${SAMPLING_BOUNDARY_PATCH_DIFF_SHA256}" \
  --build-arg "TOOL_TRUNCATION_PATCH_DIFF_SHA256=${TOOL_TRUNCATION_PATCH_DIFF_SHA256}" \
  --build-arg "VISION_RUNTIME_PATCH_DIFF_SHA256=${VISION_RUNTIME_PATCH_DIFF_SHA256}" \
  --build-arg "NUMERICAL_AUDITS_PATCH_DIFF_SHA256=${NUMERICAL_AUDITS_PATCH_DIFF_SHA256}" \
  --build-arg "TURBOQUANT_GUARDS_PATCH_DIFF_SHA256=${TURBOQUANT_GUARDS_PATCH_DIFF_SHA256}" \
  --build-arg "KV_OFFLOAD_PINNING_PATCH_DIFF_SHA256=${KV_OFFLOAD_PINNING_PATCH_DIFF_SHA256}" \
  --build-arg "SHARED_PREFIX_CACHE_PATCH_DIFF_SHA256=${SHARED_PREFIX_CACHE_PATCH_DIFF_SHA256}" \
  --build-arg "EXACT_REASONING_USAGE_PATCH_DIFF_SHA256=${EXACT_REASONING_USAGE_PATCH_DIFF_SHA256}" \
  --build-arg "IMAGE_PROFILE_VERSION=${IMAGE_PROFILE_VERSION}" \
  --build-arg "CACHE_CONFIG_UPSTREAM_FILE_SHA256=${CACHE_CONFIG_UPSTREAM_FILE_SHA256}" \
  --build-arg "VLLM_CONFIG_UPSTREAM_FILE_SHA256=${VLLM_CONFIG_UPSTREAM_FILE_SHA256}" \
  --build-arg "ARG_UTILS_UPSTREAM_FILE_SHA256=${ARG_UTILS_UPSTREAM_FILE_SHA256}" \
  --build-arg "LLM_ENTRYPOINT_UPSTREAM_FILE_SHA256=${LLM_ENTRYPOINT_UPSTREAM_FILE_SHA256}" \
  --build-arg "KV_CACHE_UTILS_UPSTREAM_FILE_SHA256=${KV_CACHE_UTILS_UPSTREAM_FILE_SHA256}" \
  --build-arg "GPU_WORKER_UPSTREAM_FILE_SHA256=${GPU_WORKER_UPSTREAM_FILE_SHA256}" \
  --build-arg "STARTUP_PLAN_UPSTREAM_FILE_SHA256=${STARTUP_PLAN_UPSTREAM_FILE_SHA256}" \
  --build-arg "KV_OFFLOAD_CONFIG_UPSTREAM_FILE_SHA256=${KV_OFFLOAD_CONFIG_UPSTREAM_FILE_SHA256}" \
  --build-arg "KV_OFFLOAD_BASE_UPSTREAM_FILE_SHA256=${KV_OFFLOAD_BASE_UPSTREAM_FILE_SHA256}" \
  --build-arg "KV_OFFLOAD_CPU_SPEC_UPSTREAM_FILE_SHA256=${KV_OFFLOAD_CPU_SPEC_UPSTREAM_FILE_SHA256}" \
  --build-arg "KV_OFFLOAD_CPU_MANAGER_UPSTREAM_FILE_SHA256=${KV_OFFLOAD_CPU_MANAGER_UPSTREAM_FILE_SHA256}" \
  --build-arg "KV_TIERING_SPEC_UPSTREAM_FILE_SHA256=${KV_TIERING_SPEC_UPSTREAM_FILE_SHA256}" \
  --build-arg "KV_TIERING_MANAGER_UPSTREAM_FILE_SHA256=${KV_TIERING_MANAGER_UPSTREAM_FILE_SHA256}" \
  --build-arg "OFFLOAD_CONNECTOR_CONFIG_UPSTREAM_FILE_SHA256=${OFFLOAD_CONNECTOR_CONFIG_UPSTREAM_FILE_SHA256}" \
  --build-arg "OFFLOAD_CONNECTOR_SCHEDULER_UPSTREAM_FILE_SHA256=${OFFLOAD_CONNECTOR_SCHEDULER_UPSTREAM_FILE_SHA256}" \
  --build-arg "COMPLETION_PROTOCOL_UPSTREAM_FILE_SHA256=${COMPLETION_PROTOCOL_UPSTREAM_FILE_SHA256}" \
  --build-arg "GENERATE_API_ROUTER_UPSTREAM_FILE_SHA256=${GENERATE_API_ROUTER_UPSTREAM_FILE_SHA256}" \
  --build-arg "CLI_ARGS_UPSTREAM_FILE_SHA256=${CLI_ARGS_UPSTREAM_FILE_SHA256}" \
  --build-arg "TITOTO_PROTOCOL_UPSTREAM_FILE_SHA256=${TITOTO_PROTOCOL_UPSTREAM_FILE_SHA256}" \
  --build-arg "TITOTO_SERVING_UPSTREAM_FILE_SHA256=${TITOTO_SERVING_UPSTREAM_FILE_SHA256}" \
  --build-arg "POLICY_PKG_INIT_UPSTREAM_FILE_SHA256=${POLICY_PKG_INIT_UPSTREAM_FILE_SHA256}" \
  --build-arg "POLICY_BASE_UPSTREAM_FILE_SHA256=${POLICY_BASE_UPSTREAM_FILE_SHA256}" \
  --build-arg "POLICY_FACTORY_UPSTREAM_FILE_SHA256=${POLICY_FACTORY_UPSTREAM_FILE_SHA256}" \
  --build-arg "POLICY_LRU_UPSTREAM_FILE_SHA256=${POLICY_LRU_UPSTREAM_FILE_SHA256}" \
  --build-arg "POLICY_ARC_UPSTREAM_FILE_SHA256=${POLICY_ARC_UPSTREAM_FILE_SHA256}" \
  --build-arg "CACHE_CONFIG_PATCHED_FILE_SHA256=${CACHE_CONFIG_PATCHED_FILE_SHA256}" \
  --build-arg "VLLM_CONFIG_PATCHED_FILE_SHA256=${VLLM_CONFIG_PATCHED_FILE_SHA256}" \
  --build-arg "ARG_UTILS_PATCHED_FILE_SHA256=${ARG_UTILS_PATCHED_FILE_SHA256}" \
  --build-arg "LLM_ENTRYPOINT_PATCHED_FILE_SHA256=${LLM_ENTRYPOINT_PATCHED_FILE_SHA256}" \
  --build-arg "KV_CACHE_UTILS_PATCHED_FILE_SHA256=${KV_CACHE_UTILS_PATCHED_FILE_SHA256}" \
  --build-arg "GPU_WORKER_PATCHED_FILE_SHA256=${GPU_WORKER_PATCHED_FILE_SHA256}" \
  --build-arg "STARTUP_PLAN_PATCHED_FILE_SHA256=${STARTUP_PLAN_PATCHED_FILE_SHA256}" \
  --build-arg "KV_OFFLOAD_CONFIG_PATCHED_FILE_SHA256=${KV_OFFLOAD_CONFIG_PATCHED_FILE_SHA256}" \
  --build-arg "KV_OFFLOAD_BASE_PATCHED_FILE_SHA256=${KV_OFFLOAD_BASE_PATCHED_FILE_SHA256}" \
  --build-arg "KV_OFFLOAD_CPU_SPEC_PATCHED_FILE_SHA256=${KV_OFFLOAD_CPU_SPEC_PATCHED_FILE_SHA256}" \
  --build-arg "KV_OFFLOAD_CPU_MANAGER_PATCHED_FILE_SHA256=${KV_OFFLOAD_CPU_MANAGER_PATCHED_FILE_SHA256}" \
  --build-arg "KV_TIERING_SPEC_PATCHED_FILE_SHA256=${KV_TIERING_SPEC_PATCHED_FILE_SHA256}" \
  --build-arg "KV_TIERING_MANAGER_PATCHED_FILE_SHA256=${KV_TIERING_MANAGER_PATCHED_FILE_SHA256}" \
  --build-arg "OFFLOAD_CONNECTOR_CONFIG_PATCHED_FILE_SHA256=${OFFLOAD_CONNECTOR_CONFIG_PATCHED_FILE_SHA256}" \
  --build-arg "OFFLOAD_CONNECTOR_SCHEDULER_PATCHED_FILE_SHA256=${OFFLOAD_CONNECTOR_SCHEDULER_PATCHED_FILE_SHA256}" \
  --build-arg "COMPLETION_PROTOCOL_PATCHED_FILE_SHA256=${COMPLETION_PROTOCOL_PATCHED_FILE_SHA256}" \
  --build-arg "GENERATE_API_ROUTER_PATCHED_FILE_SHA256=${GENERATE_API_ROUTER_PATCHED_FILE_SHA256}" \
  --build-arg "CLI_ARGS_PATCHED_FILE_SHA256=${CLI_ARGS_PATCHED_FILE_SHA256}" \
  --build-arg "TITOTO_PROTOCOL_PATCHED_FILE_SHA256=${TITOTO_PROTOCOL_PATCHED_FILE_SHA256}" \
  --build-arg "TITOTO_SERVING_PATCHED_FILE_SHA256=${TITOTO_SERVING_PATCHED_FILE_SHA256}" \
  --build-arg "ABSTRACT_PARSER_PATCHED_FILE_SHA256=${ABSTRACT_PARSER_PATCHED_FILE_SHA256}" \
  --build-arg "PARSER_ADAPTERS_PATCHED_FILE_SHA256=${PARSER_ADAPTERS_PATCHED_FILE_SHA256}" \
  --build-arg "PARSER_EVENTS_PATCHED_FILE_SHA256=${PARSER_EVENTS_PATCHED_FILE_SHA256}" \
  --build-arg "PARSER_ENGINE_CONFIG_PATCHED_FILE_SHA256=${PARSER_ENGINE_CONFIG_PATCHED_FILE_SHA256}" \
  --build-arg "STREAMING_PARSER_ENGINE_PATCHED_FILE_SHA256=${STREAMING_PARSER_ENGINE_PATCHED_FILE_SHA256}" \
  --build-arg "TOKEN_ID_SCANNER_PATCHED_FILE_SHA256=${TOKEN_ID_SCANNER_PATCHED_FILE_SHA256}" \
  --build-arg "ENGINE_PROTOCOL_PATCHED_FILE_SHA256=${ENGINE_PROTOCOL_PATCHED_FILE_SHA256}" \
  --build-arg "SOURCE_DATE_EPOCH=${SOURCE_DATE_EPOCH}" \
  --output "type=docker,dest=${RUNTIME_ARCHIVE},name=${IMAGE_TAG},rewrite-timestamp=true" \
  --file "${DOCKERFILE}" \
  "${PROJECT_DIR}"
docker load --input "${RUNTIME_ARCHIVE}"

actual_image_id="$(docker image inspect --format '{{.Id}}' "${IMAGE_TAG}")"
actual_installed_report="$(
  docker run --rm --network none --entrypoint sha256sum "${IMAGE_TAG}" \
    /usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/turboquant_attn.py \
    /usr/local/lib/python3.12/dist-packages/vllm/v1/attention/ops/triton_turboquant_store.py \
    /usr/local/lib/python3.12/dist-packages/vllm/v1/attention/ops/triton_turboquant_decode.py \
    /usr/local/lib/python3.12/dist-packages/vllm/tool_parsers/structural_tag_registry.py \
    /usr/local/lib/python3.12/dist-packages/vllm/tool_parsers/abstract_tool_parser.py \
    /usr/local/lib/python3.12/dist-packages/vllm/config/model.py \
    /usr/local/lib/python3.12/dist-packages/vllm/entrypoints/anthropic/protocol.py \
    /usr/local/lib/python3.12/dist-packages/vllm/entrypoints/anthropic/serving.py \
    /usr/local/lib/python3.12/dist-packages/vllm/entrypoints/serve/exception_handling/error_response.py \
    /usr/local/lib/python3.12/dist-packages/vllm/entrypoints/serve/exception_handling/handlers/exception.py \
    /usr/local/lib/python3.12/dist-packages/vllm/entrypoints/serve/exception_handling/handlers/http.py \
    /usr/local/lib/python3.12/dist-packages/vllm/entrypoints/serve/exception_handling/handlers/validation.py \
    /usr/local/lib/python3.12/dist-packages/vllm/entrypoints/serve/exception_handling/handlers/vllm_error.py \
    /usr/local/lib/python3.12/dist-packages/vllm/entrypoints/openai/chat_completion/protocol.py \
    /usr/local/lib/python3.12/dist-packages/vllm/sampling_params.py \
    /usr/local/lib/python3.12/dist-packages/vllm/v1/core/sched/utils.py \
    /usr/local/lib/python3.12/dist-packages/vllm/v1/engine/input_processor.py \
    /usr/local/lib/python3.12/dist-packages/vllm/v1/request.py \
    /usr/local/lib/python3.12/dist-packages/vllm/parser/qwen3.py \
    /usr/local/lib/python3.12/dist-packages/vllm/v1/structured_output/__init__.py \
    /usr/local/lib/python3.12/dist-packages/vllm/entrypoints/anthropic/api_router.py \
    /usr/local/lib/python3.12/dist-packages/vllm/entrypoints/openai/chat_completion/serving.py \
    /usr/local/lib/python3.12/dist-packages/vllm/renderers/hf.py \
    /usr/local/lib/python3.12/dist-packages/vllm/entrypoints/serve/exception_handling/register.py \
    /usr/local/lib/python3.12/dist-packages/vllm/tokenizers/detokenizer_utils.py \
    /usr/local/lib/python3.12/dist-packages/vllm/tool_parsers/utils.py \
    /usr/local/lib/python3.12/dist-packages/vllm/v1/sample/thinking_budget_state.py \
    /usr/local/lib/python3.12/dist-packages/vllm/reasoning/abs_reasoning_parsers.py \
    /usr/local/lib/python3.12/dist-packages/vllm/config/reasoning.py \
    /usr/local/lib/python3.12/dist-packages/vllm/v1/structured_output/stop_checker.py \
    /usr/local/lib/python3.12/dist-packages/vllm/v1/structured_output/backend_xgrammar.py \
    /usr/local/lib/python3.12/dist-packages/vllm/v1/structured_output/backend_types.py \
    /usr/local/lib/python3.12/dist-packages/vllm/v1/engine/output_processor.py \
    /usr/local/lib/python3.12/dist-packages/vllm/v1/engine/detokenizer.py \
    /usr/local/lib/python3.12/dist-packages/vllm/v1/core/sched/scheduler.py \
    /usr/local/lib/python3.12/dist-packages/vllm/v1/engine/async_llm.py \
    /usr/local/lib/python3.12/dist-packages/vllm/parser/gemma4.py \
    /usr/local/lib/python3.12/dist-packages/vllm/parser/glm47_moe.py \
    /usr/local/lib/python3.12/dist-packages/vllm/parser/minimax_m2.py \
    /usr/local/lib/python3.12/dist-packages/vllm/parser/mistral.py \
    /usr/local/lib/python3.12/dist-packages/vllm/parser/deepseek_v32.py \
    /usr/local/lib/python3.12/dist-packages/vllm/parser/deepseek_v4.py \
    /usr/local/lib/python3.12/dist-packages/vllm/parser/inkling.py \
    /usr/local/lib/python3.12/dist-packages/vllm/parser/kimi_k2.py \
    /usr/local/lib/python3.12/dist-packages/vllm/entrypoints/scale_out/derender/serving.py \
    /usr/local/lib/python3.12/dist-packages/vllm/multimodal/processing/inputs.py \
    /usr/local/lib/python3.12/dist-packages/vllm/multimodal/processing/processor.py \
    /usr/local/lib/python3.12/dist-packages/vllm/renderers/base.py \
    /usr/local/lib/python3.12/dist-packages/vllm/renderers/online_derenderer.py \
    /usr/local/lib/python3.12/dist-packages/vllm/entrypoints/openai/completion/serving.py \
    /usr/local/lib/python3.12/dist-packages/vllm/entrypoints/scale_out/render/serving.py \
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
expected_installed_report="$(printf '%s  %s\n' \
  "${TURBOQUANT_PATCHED_FILE_SHA256}" /usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/turboquant_attn.py \
  "${TURBOQUANT_STORE_PATCHED_FILE_SHA256}" /usr/local/lib/python3.12/dist-packages/vllm/v1/attention/ops/triton_turboquant_store.py \
  "${TURBOQUANT_DECODE_PATCHED_FILE_SHA256}" /usr/local/lib/python3.12/dist-packages/vllm/v1/attention/ops/triton_turboquant_decode.py \
  "${TOOL_SCHEMA_PATCHED_FILE_SHA256}" /usr/local/lib/python3.12/dist-packages/vllm/tool_parsers/structural_tag_registry.py \
  "${TOOL_PARSER_ABSTRACT_PATCHED_FILE_SHA256}" /usr/local/lib/python3.12/dist-packages/vllm/tool_parsers/abstract_tool_parser.py \
  "${MODEL_CONFIG_PATCHED_FILE_SHA256}" /usr/local/lib/python3.12/dist-packages/vllm/config/model.py \
  "${ANTHROPIC_PROTOCOL_PATCHED_FILE_SHA256}" /usr/local/lib/python3.12/dist-packages/vllm/entrypoints/anthropic/protocol.py \
  "${ANTHROPIC_SERVING_PATCHED_FILE_SHA256}" /usr/local/lib/python3.12/dist-packages/vllm/entrypoints/anthropic/serving.py \
  "${ERROR_RESPONSE_PATCHED_FILE_SHA256}" /usr/local/lib/python3.12/dist-packages/vllm/entrypoints/serve/exception_handling/error_response.py \
  "${EXCEPTION_EXCEPTION_HANDLER_PATCHED_FILE_SHA256}" /usr/local/lib/python3.12/dist-packages/vllm/entrypoints/serve/exception_handling/handlers/exception.py \
  "${HTTP_EXCEPTION_HANDLER_PATCHED_FILE_SHA256}" /usr/local/lib/python3.12/dist-packages/vllm/entrypoints/serve/exception_handling/handlers/http.py \
  "${VALIDATION_EXCEPTION_HANDLER_PATCHED_FILE_SHA256}" /usr/local/lib/python3.12/dist-packages/vllm/entrypoints/serve/exception_handling/handlers/validation.py \
  "${VLLM_ERROR_EXCEPTION_HANDLER_PATCHED_FILE_SHA256}" /usr/local/lib/python3.12/dist-packages/vllm/entrypoints/serve/exception_handling/handlers/vllm_error.py \
  "${CHAT_PROTOCOL_PATCHED_FILE_SHA256}" /usr/local/lib/python3.12/dist-packages/vllm/entrypoints/openai/chat_completion/protocol.py \
  "${SAMPLING_PARAMS_PATCHED_FILE_SHA256}" /usr/local/lib/python3.12/dist-packages/vllm/sampling_params.py \
  "${SCHED_UTILS_PATCHED_FILE_SHA256}" /usr/local/lib/python3.12/dist-packages/vllm/v1/core/sched/utils.py \
  "${INPUT_PROCESSOR_PATCHED_FILE_SHA256}" /usr/local/lib/python3.12/dist-packages/vllm/v1/engine/input_processor.py \
  "${REQUEST_PATCHED_FILE_SHA256}" /usr/local/lib/python3.12/dist-packages/vllm/v1/request.py \
  "${QWEN3_PARSER_PATCHED_FILE_SHA256}" /usr/local/lib/python3.12/dist-packages/vllm/parser/qwen3.py \
  "${STRUCTURED_OUTPUT_PATCHED_FILE_SHA256}" /usr/local/lib/python3.12/dist-packages/vllm/v1/structured_output/__init__.py \
  "${ANTHROPIC_API_ROUTER_PATCHED_FILE_SHA256}" /usr/local/lib/python3.12/dist-packages/vllm/entrypoints/anthropic/api_router.py \
  "${CHAT_SERVING_PATCHED_FILE_SHA256}" /usr/local/lib/python3.12/dist-packages/vllm/entrypoints/openai/chat_completion/serving.py \
  "${HF_RENDERER_PATCHED_FILE_SHA256}" /usr/local/lib/python3.12/dist-packages/vllm/renderers/hf.py \
  "${EXCEPTION_REGISTRATION_PATCHED_FILE_SHA256}" /usr/local/lib/python3.12/dist-packages/vllm/entrypoints/serve/exception_handling/register.py \
  "${DETOKENIZER_UTILS_PATCHED_FILE_SHA256}" /usr/local/lib/python3.12/dist-packages/vllm/tokenizers/detokenizer_utils.py \
  "${TOOL_PARSER_UTILS_PATCHED_FILE_SHA256}" /usr/local/lib/python3.12/dist-packages/vllm/tool_parsers/utils.py \
  "${V1_THINKING_BUDGET_PATCHED_FILE_SHA256}" /usr/local/lib/python3.12/dist-packages/vllm/v1/sample/thinking_budget_state.py \
  "${BASE_REASONING_PARSER_PATCHED_FILE_SHA256}" /usr/local/lib/python3.12/dist-packages/vllm/reasoning/abs_reasoning_parsers.py \
  "${REASONING_CONFIG_PATCHED_FILE_SHA256}" /usr/local/lib/python3.12/dist-packages/vllm/config/reasoning.py \
  "${STRUCTURAL_TAG_STOP_CHECKER_PATCHED_FILE_SHA256}" /usr/local/lib/python3.12/dist-packages/vllm/v1/structured_output/stop_checker.py \
  "${XGRAMMAR_BACKEND_PATCHED_FILE_SHA256}" /usr/local/lib/python3.12/dist-packages/vllm/v1/structured_output/backend_xgrammar.py \
  "${STRUCTURED_OUTPUT_BACKEND_TYPES_PATCHED_FILE_SHA256}" /usr/local/lib/python3.12/dist-packages/vllm/v1/structured_output/backend_types.py \
  "${V1_OUTPUT_PROCESSOR_PATCHED_FILE_SHA256}" /usr/local/lib/python3.12/dist-packages/vllm/v1/engine/output_processor.py \
  "${V1_DETOKENIZER_PATCHED_FILE_SHA256}" /usr/local/lib/python3.12/dist-packages/vllm/v1/engine/detokenizer.py \
  "${V1_SCHEDULER_PATCHED_FILE_SHA256}" /usr/local/lib/python3.12/dist-packages/vllm/v1/core/sched/scheduler.py \
  "${ASYNC_LLM_PATCHED_FILE_SHA256}" /usr/local/lib/python3.12/dist-packages/vllm/v1/engine/async_llm.py \
  "${GEMMA4_PARSER_PATCHED_FILE_SHA256}" /usr/local/lib/python3.12/dist-packages/vllm/parser/gemma4.py \
  "${GLM47_PARSER_PATCHED_FILE_SHA256}" /usr/local/lib/python3.12/dist-packages/vllm/parser/glm47_moe.py \
  "${MINIMAX_M2_PARSER_PATCHED_FILE_SHA256}" /usr/local/lib/python3.12/dist-packages/vllm/parser/minimax_m2.py \
  "${MISTRAL_PARSER_PATCHED_FILE_SHA256}" /usr/local/lib/python3.12/dist-packages/vllm/parser/mistral.py \
  "${DEEPSEEK_V32_PARSER_PATCHED_FILE_SHA256}" /usr/local/lib/python3.12/dist-packages/vllm/parser/deepseek_v32.py \
  "${DEEPSEEK_V4_PARSER_PATCHED_FILE_SHA256}" /usr/local/lib/python3.12/dist-packages/vllm/parser/deepseek_v4.py \
  "${INKLING_PARSER_PATCHED_FILE_SHA256}" /usr/local/lib/python3.12/dist-packages/vllm/parser/inkling.py \
  "${KIMI_K2_PARSER_PATCHED_FILE_SHA256}" /usr/local/lib/python3.12/dist-packages/vllm/parser/kimi_k2.py \
  "${DERENDER_SERVING_PATCHED_FILE_SHA256}" /usr/local/lib/python3.12/dist-packages/vllm/entrypoints/scale_out/derender/serving.py \
  "${MM_PROCESSOR_INPUTS_PATCHED_FILE_SHA256}" /usr/local/lib/python3.12/dist-packages/vllm/multimodal/processing/inputs.py \
  "${MM_PROCESSOR_PATCHED_FILE_SHA256}" /usr/local/lib/python3.12/dist-packages/vllm/multimodal/processing/processor.py \
  "${BASE_RENDERER_PATCHED_FILE_SHA256}" /usr/local/lib/python3.12/dist-packages/vllm/renderers/base.py \
  "${ONLINE_DERENDERER_PATCHED_FILE_SHA256}" /usr/local/lib/python3.12/dist-packages/vllm/renderers/online_derenderer.py \
  "${COMPLETION_SERVING_PATCHED_FILE_SHA256}" /usr/local/lib/python3.12/dist-packages/vllm/entrypoints/openai/completion/serving.py \
  "${RENDER_SERVING_PATCHED_FILE_SHA256}" /usr/local/lib/python3.12/dist-packages/vllm/entrypoints/scale_out/render/serving.py \
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
    /opt/qwen38/turboquant_guard_unit.py \
    /opt/qwen38/qwen38_context_unit.py \
    /opt/qwen38/chat_template_retention_unit.py \
    /opt/qwen38/nvfp4_kernel_unit.py
)"
expected_additional_installed_report="$(printf '%s  %s\n' \
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
  "${TURBOQUANT_GUARD_UNIT_SHA256}" /opt/qwen38/turboquant_guard_unit.py \
  "${QWEN38_CONTEXT_UNIT_SHA256}" /opt/qwen38/qwen38_context_unit.py \
  "${CHAT_TEMPLATE_RETENTION_UNIT_SHA256}" /opt/qwen38/chat_template_retention_unit.py \
  "${NVFP4_KERNEL_UNIT_SHA256}" /opt/qwen38/nvfp4_kernel_unit.py)"
if [[ "${additional_installed_report}" != "${expected_additional_installed_report}" ]]; then
  echo "Built image contains unexpected vision/runtime bytes." >&2
  echo "Expected:" >&2
  printf '%s\n' "${expected_additional_installed_report}" >&2
  echo "Found:" >&2
  printf '%s\n' "${additional_installed_report}" >&2
  exit 1
fi

kv_users_installed_report="$(
  docker run --rm --network none --entrypoint sha256sum "${IMAGE_TAG}" \
    /usr/local/lib/python3.12/dist-packages/vllm/distributed/kv_transfer/kv_connector/v1/offloading_connector.py \
    /usr/local/lib/python3.12/dist-packages/vllm/v1/core/block_pool.py \
    /usr/local/lib/python3.12/dist-packages/vllm/v1/core/kv_cache_coordinator.py \
    /usr/local/lib/python3.12/dist-packages/vllm/v1/core/kv_cache_manager.py \
    /usr/local/lib/python3.12/dist-packages/vllm/v1/core/prefix_cache.py \
    /usr/local/lib/python3.12/dist-packages/vllm/v1/core/single_type_kv_cache_manager.py \
    /usr/local/lib/python3.12/dist-packages/vllm/v1/kv_offload/cpu/common.py \
    /usr/local/lib/python3.12/dist-packages/vllm/v1/simple_kv_offload/manager.py \
    /opt/qwen38/shared_prefix_cache_unit.py \
    /opt/qwen38/raw_media_unit.py \
    /opt/qwen38/generate_result_unit.py \
    /usr/local/lib/python3.12/dist-packages/vllm/config/cache.py \
    /usr/local/lib/python3.12/dist-packages/vllm/config/vllm.py \
    /usr/local/lib/python3.12/dist-packages/vllm/engine/arg_utils.py \
    /usr/local/lib/python3.12/dist-packages/vllm/entrypoints/llm.py \
    /usr/local/lib/python3.12/dist-packages/vllm/v1/core/kv_cache_utils.py \
    /usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu_worker.py \
    /usr/local/lib/python3.12/dist-packages/vllm/v1/worker/startup_plan.py \
    /usr/local/lib/python3.12/dist-packages/vllm/v1/kv_offload/config.py \
    /usr/local/lib/python3.12/dist-packages/vllm/v1/kv_offload/base.py \
    /usr/local/lib/python3.12/dist-packages/vllm/v1/kv_offload/cpu/spec.py \
    /usr/local/lib/python3.12/dist-packages/vllm/v1/kv_offload/cpu/manager.py \
    /usr/local/lib/python3.12/dist-packages/vllm/v1/kv_offload/tiering/spec.py \
    /usr/local/lib/python3.12/dist-packages/vllm/v1/kv_offload/tiering/manager.py \
    /usr/local/lib/python3.12/dist-packages/vllm/distributed/kv_transfer/kv_connector/v1/offloading/config.py \
    /usr/local/lib/python3.12/dist-packages/vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py \
    /usr/local/lib/python3.12/dist-packages/vllm/entrypoints/openai/completion/protocol.py \
    /usr/local/lib/python3.12/dist-packages/vllm/entrypoints/generate/api_router.py \
    /usr/local/lib/python3.12/dist-packages/vllm/entrypoints/openai/cli_args.py \
    /usr/local/lib/python3.12/dist-packages/vllm/entrypoints/scale_out/token_in_token_out/protocol.py \
    /usr/local/lib/python3.12/dist-packages/vllm/entrypoints/scale_out/token_in_token_out/serving.py
)"
expected_kv_users_installed_report="$(printf '%s  %s\n' \
  "${OFFLOADING_CONNECTOR_PATCHED_FILE_SHA256}" /usr/local/lib/python3.12/dist-packages/vllm/distributed/kv_transfer/kv_connector/v1/offloading_connector.py \
  "${BLOCK_POOL_PATCHED_FILE_SHA256}" /usr/local/lib/python3.12/dist-packages/vllm/v1/core/block_pool.py \
  "${KV_CACHE_COORDINATOR_PATCHED_FILE_SHA256}" /usr/local/lib/python3.12/dist-packages/vllm/v1/core/kv_cache_coordinator.py \
  "${KV_CACHE_MANAGER_PATCHED_FILE_SHA256}" /usr/local/lib/python3.12/dist-packages/vllm/v1/core/kv_cache_manager.py \
  "${PREFIX_CACHE_PATCHED_FILE_SHA256}" /usr/local/lib/python3.12/dist-packages/vllm/v1/core/prefix_cache.py \
  "${SINGLE_TYPE_KV_CACHE_MANAGER_PATCHED_FILE_SHA256}" /usr/local/lib/python3.12/dist-packages/vllm/v1/core/single_type_kv_cache_manager.py \
  "${KV_OFFLOAD_CPU_COMMON_PATCHED_FILE_SHA256}" /usr/local/lib/python3.12/dist-packages/vllm/v1/kv_offload/cpu/common.py \
  "${SIMPLE_KV_OFFLOAD_MANAGER_PATCHED_FILE_SHA256}" /usr/local/lib/python3.12/dist-packages/vllm/v1/simple_kv_offload/manager.py \
  "${SHARED_PREFIX_CACHE_UNIT_SHA256}" /opt/qwen38/shared_prefix_cache_unit.py \
  "${RAW_MEDIA_UNIT_SHA256}" /opt/qwen38/raw_media_unit.py \
  "${GENERATE_RESULT_UNIT_SHA256}" /opt/qwen38/generate_result_unit.py \
  "${CACHE_CONFIG_PATCHED_FILE_SHA256}" /usr/local/lib/python3.12/dist-packages/vllm/config/cache.py \
  "${VLLM_CONFIG_PATCHED_FILE_SHA256}" /usr/local/lib/python3.12/dist-packages/vllm/config/vllm.py \
  "${ARG_UTILS_PATCHED_FILE_SHA256}" /usr/local/lib/python3.12/dist-packages/vllm/engine/arg_utils.py \
  "${LLM_ENTRYPOINT_PATCHED_FILE_SHA256}" /usr/local/lib/python3.12/dist-packages/vllm/entrypoints/llm.py \
  "${KV_CACHE_UTILS_PATCHED_FILE_SHA256}" /usr/local/lib/python3.12/dist-packages/vllm/v1/core/kv_cache_utils.py \
  "${GPU_WORKER_PATCHED_FILE_SHA256}" /usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu_worker.py \
  "${STARTUP_PLAN_PATCHED_FILE_SHA256}" /usr/local/lib/python3.12/dist-packages/vllm/v1/worker/startup_plan.py \
  "${KV_OFFLOAD_CONFIG_PATCHED_FILE_SHA256}" /usr/local/lib/python3.12/dist-packages/vllm/v1/kv_offload/config.py \
  "${KV_OFFLOAD_BASE_PATCHED_FILE_SHA256}" /usr/local/lib/python3.12/dist-packages/vllm/v1/kv_offload/base.py \
  "${KV_OFFLOAD_CPU_SPEC_PATCHED_FILE_SHA256}" /usr/local/lib/python3.12/dist-packages/vllm/v1/kv_offload/cpu/spec.py \
  "${KV_OFFLOAD_CPU_MANAGER_PATCHED_FILE_SHA256}" /usr/local/lib/python3.12/dist-packages/vllm/v1/kv_offload/cpu/manager.py \
  "${KV_TIERING_SPEC_PATCHED_FILE_SHA256}" /usr/local/lib/python3.12/dist-packages/vllm/v1/kv_offload/tiering/spec.py \
  "${KV_TIERING_MANAGER_PATCHED_FILE_SHA256}" /usr/local/lib/python3.12/dist-packages/vllm/v1/kv_offload/tiering/manager.py \
  "${OFFLOAD_CONNECTOR_CONFIG_PATCHED_FILE_SHA256}" /usr/local/lib/python3.12/dist-packages/vllm/distributed/kv_transfer/kv_connector/v1/offloading/config.py \
  "${OFFLOAD_CONNECTOR_SCHEDULER_PATCHED_FILE_SHA256}" /usr/local/lib/python3.12/dist-packages/vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py \
  "${COMPLETION_PROTOCOL_PATCHED_FILE_SHA256}" /usr/local/lib/python3.12/dist-packages/vllm/entrypoints/openai/completion/protocol.py \
  "${GENERATE_API_ROUTER_PATCHED_FILE_SHA256}" /usr/local/lib/python3.12/dist-packages/vllm/entrypoints/generate/api_router.py \
  "${CLI_ARGS_PATCHED_FILE_SHA256}" /usr/local/lib/python3.12/dist-packages/vllm/entrypoints/openai/cli_args.py \
  "${TITOTO_PROTOCOL_PATCHED_FILE_SHA256}" /usr/local/lib/python3.12/dist-packages/vllm/entrypoints/scale_out/token_in_token_out/protocol.py \
  "${TITOTO_SERVING_PATCHED_FILE_SHA256}" /usr/local/lib/python3.12/dist-packages/vllm/entrypoints/scale_out/token_in_token_out/serving.py)"
if [[ "${kv_users_installed_report}" != "${expected_kv_users_installed_report}" ]]; then
  echo "Built image contains unexpected shared KV cache/capacity bytes." >&2
  echo "Expected:" >&2
  printf '%s\n' "${expected_kv_users_installed_report}" >&2
  echo "Found:" >&2
  printf '%s\n' "${kv_users_installed_report}" >&2
  exit 1
fi

reasoning_usage_installed_report="$(
  docker run --rm --network none --entrypoint sha256sum "${IMAGE_TAG}" \
    /usr/local/lib/python3.12/dist-packages/vllm/parser/abstract_parser.py \
    /usr/local/lib/python3.12/dist-packages/vllm/parser/engine/adapters.py \
    /usr/local/lib/python3.12/dist-packages/vllm/parser/engine/events.py \
    /usr/local/lib/python3.12/dist-packages/vllm/parser/engine/parser_engine_config.py \
    /usr/local/lib/python3.12/dist-packages/vllm/parser/engine/streaming_parser_engine.py \
    /usr/local/lib/python3.12/dist-packages/vllm/parser/engine/token_id_scanner.py \
    /usr/local/lib/python3.12/dist-packages/vllm/entrypoints/openai/engine/protocol.py \
    /opt/qwen38/tool_output_parser_unit.py \
    /opt/qwen38/reasoning_usage_unit.py
)"
expected_reasoning_usage_installed_report="$(printf '%s  %s\n' \
  "${ABSTRACT_PARSER_PATCHED_FILE_SHA256}" /usr/local/lib/python3.12/dist-packages/vllm/parser/abstract_parser.py \
  "${PARSER_ADAPTERS_PATCHED_FILE_SHA256}" /usr/local/lib/python3.12/dist-packages/vllm/parser/engine/adapters.py \
  "${PARSER_EVENTS_PATCHED_FILE_SHA256}" /usr/local/lib/python3.12/dist-packages/vllm/parser/engine/events.py \
  "${PARSER_ENGINE_CONFIG_PATCHED_FILE_SHA256}" /usr/local/lib/python3.12/dist-packages/vllm/parser/engine/parser_engine_config.py \
  "${STREAMING_PARSER_ENGINE_PATCHED_FILE_SHA256}" /usr/local/lib/python3.12/dist-packages/vllm/parser/engine/streaming_parser_engine.py \
  "${TOKEN_ID_SCANNER_PATCHED_FILE_SHA256}" /usr/local/lib/python3.12/dist-packages/vllm/parser/engine/token_id_scanner.py \
  "${ENGINE_PROTOCOL_PATCHED_FILE_SHA256}" /usr/local/lib/python3.12/dist-packages/vllm/entrypoints/openai/engine/protocol.py \
  "${TOOL_OUTPUT_PARSER_UNIT_SHA256}" /opt/qwen38/tool_output_parser_unit.py \
  "${REASONING_USAGE_UNIT_SHA256}" /opt/qwen38/reasoning_usage_unit.py)"
if [[ "${reasoning_usage_installed_report}" != "${expected_reasoning_usage_installed_report}" ]]; then
  echo "Built image contains unexpected reasoning-usage bytes." >&2
  echo "Expected:" >&2
  printf '%s\n' "${expected_reasoning_usage_installed_report}" >&2
  echo "Found:" >&2
  printf '%s\n' "${reasoning_usage_installed_report}" >&2
  exit 1
fi

# Superseded runtime code must be absent from the shipped image.
for obsolete in \
  /usr/local/lib/python3.12/dist-packages/vllm/v1/kv_offload/cpu/policies \
  /usr/local/lib/python3.12/dist-packages/vllm/entrypoints/serve/utils/tool_calls_utils.py \
  /usr/local/lib/python3.12/dist-packages/vllm/entrypoints/scale_out/token_in_token_out/mm_serde.py; do
  if ! docker run --rm --network none --entrypoint test "${IMAGE_TAG}" '!' -e "${obsolete}"; then
    printf 'Built image still contains superseded runtime code: %s\n' "${obsolete}" >&2
    exit 1
  fi
done

actual_profile_label="$(
  docker image inspect --format '{{index .Config.Labels "qwen38.runtime.profile"}}' \
    "${IMAGE_TAG}"
)"
if [[ "${actual_profile_label}" != "${IMAGE_PROFILE_VERSION}" ]]; then
  echo "Built image carries the wrong runtime profile label." >&2
  echo "Expected: ${IMAGE_PROFILE_VERSION}" >&2
  echo "Found:    ${actual_profile_label:-missing}" >&2
  exit 1
fi

if [[ "${actual_image_id}" != "${EXPECTED_IMAGE_ID}" ]]; then
  echo "Reproducible build ID mismatch." >&2
  echo "Expected: ${EXPECTED_IMAGE_ID}" >&2
  echo "Found:    ${actual_image_id}" >&2
  exit 1
fi

# Only a build that reproduced the pin names the pinned image, so this is where
# it takes its identity tag. One that already names another image is refused
# rather than moved: the tag carries the image's own ID, so that can only have
# been done by hand.
identity_tag_id="$(docker image inspect --format '{{.Id}}' "${IMAGE_IDENTITY_TAG}" 2>/dev/null || true)"
if [[ -n "${identity_tag_id}" && "${identity_tag_id}" != "${EXPECTED_IMAGE_ID}" ]]; then
  echo "The identity tag ${IMAGE_IDENTITY_TAG} already names ${identity_tag_id}; it was not moved." >&2
  exit 1
fi
docker tag "${EXPECTED_IMAGE_ID}" "${IMAGE_IDENTITY_TAG}"

echo "Built ${IMAGE_TAG} with no build-time network access."
echo "Verified reproducible image ID: ${actual_image_id}"
echo "Identity tag: ${IMAGE_IDENTITY_TAG}"
