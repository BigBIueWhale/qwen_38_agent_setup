#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'USAGE'
Usage: build-vllm.sh {build|check|serve-check|materialise DIRECTORY}
  build        verify every pinned input, build the runtime image from the
               verified reconstruction, run the units that need the GPU in it
               on this host's GPU, and pin it
  check        verify every pinned input and stop; nothing is written
  serve-check  check, then refuse unless the pinned image was built from exactly
               these inputs; start.sh and status.sh run it before serving
  materialise  write the verified reconstruction to DIRECTORY, a new tree in
               which to author a stage; nothing reads it
USAGE
  exit 2
}

# The mode is required. It defaulted to `build`, which made the longest and
# only image-producing action the one you got by typing nothing -- a second
# mode wearing the first one's clothes.
(($# >= 1)) || usage
MODE="$1"
case "${MODE}" in
  build|check|serve-check)
    (($# == 1)) || usage
    ;;
  materialise)
    (($# == 2)) || usage
    ;;
  *)
    usage
    ;;
esac
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=runtime-common.sh
source "${SCRIPT_DIR}/runtime-common.sh"
# A build pins only an image whose GPU units passed in it on this host
# (run_gpu_release_units, below), so a host that cannot run them is refused
# before anything is built.
if [[ "${MODE}" == build ]]; then
  check_host_prerequisites
fi
RUNTIME_LOCK="${PROJECT_DIR}/config/runtime-v1.sh"
VLLM_DIR="${PROJECT_DIR}/vllm"
DOCKERFILE="${PROJECT_DIR}/containers/Dockerfile.runtime"
SOURCE_PATCH_DIR="${PROJECT_DIR}/patches/source_patch_v1"
SOURCE_PATCH_MANIFEST="${SOURCE_PATCH_DIR}/manifest.sha256"
README_FILE="${PROJECT_DIR}/README.md"
MODEL_MANIFEST_DIR="${PROJECT_DIR}/manifests"

# The authoring tree is a directory this run creates. The verb never writes
# into one that exists, so it cannot write over anybody's work, and it needs
# no proof about what such a tree holds.
if [[ "${MODE}" == "materialise" ]]; then
  if ! authoring_parent="$(cd -- "$(dirname -- "$2")" 2>/dev/null && pwd)"; then
    echo "The directory that would hold the authoring tree does not exist: $(dirname -- "$2")" >&2
    echo "Next: create it, or name a new directory in one that exists." >&2
    exit 1
  fi
  AUTHORING_TREE="${authoring_parent}/$(basename -- "$2")"
  if [[ -e "${AUTHORING_TREE}" || -L "${AUTHORING_TREE}" ]]; then
    echo "Refusing to materialise into ${AUTHORING_TREE}: it already exists." >&2
    echo "This verb writes only a new tree, so it never writes over a tree that holds work." >&2
    echo "Next: name a directory that does not exist yet." >&2
    exit 1
  fi
  readonly AUTHORING_TREE
fi

# The image is exported to a tarball and loaded, rather than tagged directly,
# so that `rewrite-timestamp=true` can clamp every layer timestamp to
# SOURCE_DATE_EPOCH: files a RUN step writes carry the moment of the build
# otherwise. The build context gets its own timestamps from the assembly below
# rather than from any checkout, because COPY keeps them and the .pyc files
# the build's units write embed them.
BUILD_EXPORT_DIR="$(mktemp -d "${TMPDIR:-/tmp}/qwen38-vllm-build.XXXXXX")"
case "${BUILD_EXPORT_DIR}" in
  "${TMPDIR:-/tmp}"/qwen38-vllm-build.*) ;;
  *) echo "Unexpected temporary build-export directory: ${BUILD_EXPORT_DIR}" >&2; exit 1 ;;
esac
readonly BUILD_EXPORT_DIR
# The reconstruction is a worktree registered in the submodule's repository,
# so it is removed through git, never by deleting its directory alone.
if [[ "${MODE}" == "materialise" ]]; then
  RECONSTRUCTION="${AUTHORING_TREE}"
else
  RECONSTRUCTION="${BUILD_EXPORT_DIR}/reconstruction"
fi
readonly RECONSTRUCTION
reconstruction_held=false
remove_reconstruction() {
  if ! git -C "${VLLM_DIR}" worktree remove --force "${RECONSTRUCTION}"; then
    printf 'ERROR: failed to remove the reconstruction worktree: %s\n' \
      "${RECONSTRUCTION}" >&2
    return 1
  fi
  reconstruction_held=false
}
cleanup() {
  local status=$?
  trap - EXIT
  if [[ "${reconstruction_held}" == true && -e "${RECONSTRUCTION}" ]] &&
      ! remove_reconstruction; then
    status=1
  fi
  rm -rf -- "${BUILD_EXPORT_DIR}"
  exit "${status}"
}
trap cleanup EXIT
RUNTIME_ARCHIVE="${BUILD_EXPORT_DIR}/runtime.tar"
readonly RUNTIME_ARCHIVE
# The build is loaded under a name of this run's own, never under IMAGE_TAG.
# IMAGE_TAG names the pinned image, and a build loaded under it would take it
# from that image before anything had compared the build with the pin: a build
# that does not reproduce the pin would leave the pinned image untagged, looking
# exactly like a failed build, and the tag on an image no lock pinned. The name
# does not enter the image, so the ID is the same under any name. Once loaded,
# the build is known by its ID and named by its identity tag, and this name is
# removed.
BUILD_LOAD_TAG="${IMAGE_TAG%%:*}:build-${BUILD_EXPORT_DIR##*.}"
readonly BUILD_LOAD_TAG

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
GENERATION_AGENT_ID_PATCH_FILE="${PROJECT_DIR}/patches/vllm-generation-requires-agent-id.patch"
ATTENTION_PREFIX_HASH_PATCH_FILE="${PROJECT_DIR}/patches/vllm-attention-growth-keeps-prefix-hash.patch"
GROUPED_SPEC_GEOMETRY_PATCH_FILE="${PROJECT_DIR}/patches/vllm-grouped-kv-specs-use-layer-geometry.patch"
AGENT_OFFLOAD_RETENTION_PATCH_FILE="${PROJECT_DIR}/patches/vllm-agent-grouped-offload-retention.patch"
AGENTLESS_ROUTES_PATCH_FILE="${PROJECT_DIR}/patches/vllm-agentless-generation-routes-unmounted.patch"
KV_DECLARED_USERS_PATCH_FILE="${PROJECT_DIR}/patches/vllm-kv-capacity-in-declared-users.patch"
KV_PHYSICAL_BOUND_PATCH_FILE="${PROJECT_DIR}/patches/vllm-kv-declaration-within-physical-bound.patch"
EXACT_REASONING_USAGE_PATCH_FILE="${PROJECT_DIR}/patches/vllm-exact-reasoning-usage.patch"
CANONICAL_FRAMING_PATCH_FILE="${PROJECT_DIR}/patches/vllm-qwen-canonical-parameter-framing.patch"
QWEN_OWNED_GRAMMAR_PATCH_FILE="${PROJECT_DIR}/patches/vllm-qwen-owned-tool-grammar.patch"
QWEN_UNIQUE_PARAMETERS_PATCH_FILE="${PROJECT_DIR}/patches/vllm-qwen-unique-tool-parameters.patch"
GENERATION_ADMISSION_PATCH_FILE="${PROJECT_DIR}/patches/vllm-generation-admission-before-response.patch"
KV_SCOPE_SINGLE_FLIGHT_PATCH_FILE="${PROJECT_DIR}/patches/vllm-kv-scope-single-flight.patch"
TEMPLATE_AUTHORED_CONTROL_TOKENS_PATCH_FILE="${PROJECT_DIR}/patches/vllm-template-authored-control-tokens.patch"
NVFP4_NATIVE_KERNEL_PATCH_FILE="${PROJECT_DIR}/patches/vllm-nvfp4-native-kernel-required.patch"
QWEN_ARGUMENTS_READ_BY_GRAMMAR_PATCH_FILE="${PROJECT_DIR}/patches/vllm-qwen-arguments-read-by-grammar.patch"
STARTUP_PLAN_BOUND_PATCH_FILE="${PROJECT_DIR}/patches/vllm-startup-plan-admission-bound.patch"
TEMPLATE_REFUSAL_PARAMETER_PATCH_FILE="${PROJECT_DIR}/patches/vllm-template-refusals-name-their-parameter.patch"
QWEN_REPEATED_PARAMETER_REFUSAL_PATCH_FILE="${PROJECT_DIR}/patches/vllm-qwen-repeated-parameter-refusal.patch"
GENERATED_TOKENS_SURVIVE_PARSING_PATCH_FILE="${PROJECT_DIR}/patches/vllm-generated-tokens-survive-parsing.patch"
INCLUDE_REASONING_SHAPES_RESPONSE_PATCH_FILE="${PROJECT_DIR}/patches/vllm-include-reasoning-shapes-the-response.patch"
UNSPECIFIED_TOOL_CHOICE_PATCH_FILE="${PROJECT_DIR}/patches/vllm-unspecified-tool-choice-is-the-default.patch"
CALL_ONLY_ANSWER_BLANK_LINE_PATCH_FILE="${PROJECT_DIR}/patches/vllm-call-only-answer-keeps-the-blank-line.patch"
RESPONSES_ONE_FUNCTION_LIST_PATCH_FILE="${PROJECT_DIR}/patches/vllm-responses-tools-are-one-function-list.patch"
BATCH_PARSE_FROM_PROMPT_PATCH_FILE="${PROJECT_DIR}/patches/vllm-batch-parse-starts-where-the-prompt-leaves.patch"
DERENDER_STOP_TOKEN_TEXT_PATCH_FILE="${PROJECT_DIR}/patches/vllm-derender-text-is-the-detokenizers.patch"
OUTPUT_CONSTRAINT_BESIDE_TOOLS_PATCH_FILE="${PROJECT_DIR}/patches/vllm-output-constraints-refused-beside-tool-calls.patch"
BATCH_INVARIANT_NATIVE_FP4_PATCH_FILE="${PROJECT_DIR}/patches/vllm-batch-invariance-substitutes-no-nvfp4-kernel.patch"
RENDER_EVERY_IMAGE_PATCH_FILE="${PROJECT_DIR}/patches/vllm-render-carries-every-image-chat-renders.patch"
RENDERED_PROMPT_NOT_TRUNCATED_PATCH_FILE="${PROJECT_DIR}/patches/vllm-rendered-prompts-are-never-truncated.patch"
KV_TRANSFER_PARAMS_DECLARED_PATCH_FILE="${PROJECT_DIR}/patches/vllm-kv-transfer-params-are-declared.patch"
RESPONSES_TOOLS_NEVER_GIVEN_PATCH_FILE="${PROJECT_DIR}/patches/vllm-responses-refuses-tools-the-template-is-never-given.patch"
CHAT_STREAM_LOGPROBS_PATCH_FILE="${PROJECT_DIR}/patches/vllm-chat-stream-carries-every-token-logprob.patch"
CHAT_MESSAGES_ONE_RULE_PATCH_FILE="${PROJECT_DIR}/patches/vllm-chat-messages-read-by-one-rule.patch"
RESPONSES_REFUSES_UNHONOURED_PATCH_FILE="${PROJECT_DIR}/patches/vllm-responses-refuses-what-it-cannot-honour.patch"
REVIEWED_TESTS_DECLARE_NEEDS_PATCH_FILE="${PROJECT_DIR}/patches/vllm-reviewed-tests-declare-what-they-need.patch"
PRIORITY_REFUSED_PATCH_FILE="${PROJECT_DIR}/patches/vllm-priority-is-refused-where-nothing-orders-by-it.patch"
FORCED_CHOICE_THROUGH_REASONING_PATCH_FILE="${PROJECT_DIR}/patches/vllm-forced-tool-choice-is-held-through-reasoning.patch"
QWEN_ONE_ENGINE_PATCH_FILE="${PROJECT_DIR}/patches/vllm-qwen-format-is-served-as-one-engine.patch"

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
  --tmpfs /tmp:rw,nodev,nosuid,size=16m \
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

# The footprint the reviewed patch set leaves on a checkout of VLLM_COMMIT, as
# git reports it. It is derived from the committed stage data, whose every
# identity the transaction below proves, so it can describe no other patch set.
REVIEWED_STATUS="$(
  docker run --rm \
    --network none \
    --read-only \
    --user "$(id -u):$(id -g)" \
    --tmpfs /tmp:rw,nodev,nosuid,size=256m \
    --env PYTHONPYCACHEPREFIX=/tmp/pycache \
    --entrypoint python3 \
    --volume "${PROJECT_DIR}:/project:ro" \
    --workdir /project \
    "${BASE_IMAGE_TAG}" \
    -c 'from patches.source_patch_v1.apply_vllm_patchset import build_patchset
print("\n".join(build_patchset().worktree_status()))'
)"
readonly REVIEWED_STATUS

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
  "${GENERATION_AGENT_ID_PATCH_DIFF_SHA256}" "${GENERATION_AGENT_ID_PATCH_FILE}" \
  "${ATTENTION_PREFIX_HASH_PATCH_DIFF_SHA256}" "${ATTENTION_PREFIX_HASH_PATCH_FILE}" \
  "${GROUPED_SPEC_GEOMETRY_PATCH_DIFF_SHA256}" "${GROUPED_SPEC_GEOMETRY_PATCH_FILE}" \
  "${AGENT_OFFLOAD_RETENTION_PATCH_DIFF_SHA256}" "${AGENT_OFFLOAD_RETENTION_PATCH_FILE}" \
  "${AGENTLESS_ROUTES_PATCH_DIFF_SHA256}" "${AGENTLESS_ROUTES_PATCH_FILE}" \
  "${KV_DECLARED_USERS_PATCH_DIFF_SHA256}" "${KV_DECLARED_USERS_PATCH_FILE}" \
  "${KV_PHYSICAL_BOUND_PATCH_DIFF_SHA256}" "${KV_PHYSICAL_BOUND_PATCH_FILE}" \
  "${EXACT_REASONING_USAGE_PATCH_DIFF_SHA256}" "${EXACT_REASONING_USAGE_PATCH_FILE}" \
  "${CANONICAL_FRAMING_PATCH_DIFF_SHA256}" "${CANONICAL_FRAMING_PATCH_FILE}" \
  "${QWEN_OWNED_GRAMMAR_PATCH_DIFF_SHA256}" "${QWEN_OWNED_GRAMMAR_PATCH_FILE}" \
  "${QWEN_UNIQUE_PARAMETERS_PATCH_DIFF_SHA256}" "${QWEN_UNIQUE_PARAMETERS_PATCH_FILE}" \
  "${GENERATION_ADMISSION_PATCH_DIFF_SHA256}" "${GENERATION_ADMISSION_PATCH_FILE}" \
  "${KV_SCOPE_SINGLE_FLIGHT_PATCH_DIFF_SHA256}" "${KV_SCOPE_SINGLE_FLIGHT_PATCH_FILE}" \
  "${TEMPLATE_AUTHORED_CONTROL_TOKENS_PATCH_DIFF_SHA256}" "${TEMPLATE_AUTHORED_CONTROL_TOKENS_PATCH_FILE}" \
  "${NVFP4_NATIVE_KERNEL_PATCH_DIFF_SHA256}" "${NVFP4_NATIVE_KERNEL_PATCH_FILE}" \
  "${QWEN_ARGUMENTS_READ_BY_GRAMMAR_PATCH_DIFF_SHA256}" "${QWEN_ARGUMENTS_READ_BY_GRAMMAR_PATCH_FILE}" \
  "${STARTUP_PLAN_BOUND_PATCH_DIFF_SHA256}" "${STARTUP_PLAN_BOUND_PATCH_FILE}" \
  "${TEMPLATE_REFUSAL_PARAMETER_PATCH_DIFF_SHA256}" "${TEMPLATE_REFUSAL_PARAMETER_PATCH_FILE}" \
  "${QWEN_REPEATED_PARAMETER_REFUSAL_PATCH_DIFF_SHA256}" "${QWEN_REPEATED_PARAMETER_REFUSAL_PATCH_FILE}" \
  "${GENERATED_TOKENS_SURVIVE_PARSING_PATCH_DIFF_SHA256}" "${GENERATED_TOKENS_SURVIVE_PARSING_PATCH_FILE}" \
  "${INCLUDE_REASONING_SHAPES_RESPONSE_PATCH_DIFF_SHA256}" "${INCLUDE_REASONING_SHAPES_RESPONSE_PATCH_FILE}" \
  "${UNSPECIFIED_TOOL_CHOICE_PATCH_DIFF_SHA256}" "${UNSPECIFIED_TOOL_CHOICE_PATCH_FILE}" \
  "${CALL_ONLY_ANSWER_BLANK_LINE_PATCH_DIFF_SHA256}" "${CALL_ONLY_ANSWER_BLANK_LINE_PATCH_FILE}" \
  "${RESPONSES_ONE_FUNCTION_LIST_PATCH_DIFF_SHA256}" "${RESPONSES_ONE_FUNCTION_LIST_PATCH_FILE}" \
  "${BATCH_PARSE_FROM_PROMPT_PATCH_DIFF_SHA256}" "${BATCH_PARSE_FROM_PROMPT_PATCH_FILE}" \
  "${DERENDER_STOP_TOKEN_TEXT_PATCH_DIFF_SHA256}" "${DERENDER_STOP_TOKEN_TEXT_PATCH_FILE}" \
  "${OUTPUT_CONSTRAINT_BESIDE_TOOLS_PATCH_DIFF_SHA256}" "${OUTPUT_CONSTRAINT_BESIDE_TOOLS_PATCH_FILE}" \
  "${BATCH_INVARIANT_NATIVE_FP4_PATCH_DIFF_SHA256}" "${BATCH_INVARIANT_NATIVE_FP4_PATCH_FILE}" \
  "${RENDER_EVERY_IMAGE_PATCH_DIFF_SHA256}" "${RENDER_EVERY_IMAGE_PATCH_FILE}" \
  "${RENDERED_PROMPT_NOT_TRUNCATED_PATCH_DIFF_SHA256}" "${RENDERED_PROMPT_NOT_TRUNCATED_PATCH_FILE}" \
  "${KV_TRANSFER_PARAMS_DECLARED_PATCH_DIFF_SHA256}" "${KV_TRANSFER_PARAMS_DECLARED_PATCH_FILE}" \
  "${RESPONSES_TOOLS_NEVER_GIVEN_PATCH_DIFF_SHA256}" "${RESPONSES_TOOLS_NEVER_GIVEN_PATCH_FILE}" \
  "${CHAT_STREAM_LOGPROBS_PATCH_DIFF_SHA256}" "${CHAT_STREAM_LOGPROBS_PATCH_FILE}" \
  "${CHAT_MESSAGES_ONE_RULE_PATCH_DIFF_SHA256}" "${CHAT_MESSAGES_ONE_RULE_PATCH_FILE}" \
  "${RESPONSES_REFUSES_UNHONOURED_PATCH_DIFF_SHA256}" "${RESPONSES_REFUSES_UNHONOURED_PATCH_FILE}" \
  "${REVIEWED_TESTS_DECLARE_NEEDS_PATCH_DIFF_SHA256}" "${REVIEWED_TESTS_DECLARE_NEEDS_PATCH_FILE}" \
  "${PRIORITY_REFUSED_PATCH_DIFF_SHA256}" "${PRIORITY_REFUSED_PATCH_FILE}" \
  "${FORCED_CHOICE_THROUGH_REASONING_PATCH_DIFF_SHA256}" "${FORCED_CHOICE_THROUGH_REASONING_PATCH_FILE}" \
  "${QWEN_ONE_ENGINE_PATCH_DIFF_SHA256}" "${QWEN_ONE_ENGINE_PATCH_FILE}" | \
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

# The image is built from the reconstruction, and from nothing else under
# vllm/: the landmark-aware transaction recreates the reviewed tree from the
# pinned upstream commit in a worktree of this run's own, and the build context
# is copied out of it. The vllm/ checkout is reached only through git, for the
# commit's objects. The reviewed diffs are independently hashed and parsed as
# review evidence, but they never select mutation locations. A reconstruction
# that fails any step is removed and reaches no context and no author.
reconstruction_held=true
git -C "${VLLM_DIR}" worktree add --detach "${RECONSTRUCTION}" \
  "${VLLM_COMMIT}" >/dev/null
docker run --rm \
  --network none \
  --read-only \
  --user "$(id -u):$(id -g)" \
  --tmpfs /tmp:rw,nodev,nosuid,size=512m \
  --env PYTHONPYCACHEPREFIX=/tmp/pycache \
  --entrypoint python3 \
  --volume "${PROJECT_DIR}:/project:ro" \
  --volume "${RECONSTRUCTION}:/source:rw" \
  --workdir /project \
  "${BASE_IMAGE_TAG}" \
  -m patches.source_patch_v1.apply_vllm_patchset /source /project
# The transaction proved every path its stages name. git proves the rest: the
# reconstruction differs from the pinned commit in exactly those paths.
reconstruction_status="$(
  git -C "${RECONSTRUCTION}" status --short --untracked-files=all
)"
if [[ "${reconstruction_status}" != "${REVIEWED_STATUS}" ]]; then
  echo "The reviewed patch set left a footprint on the reconstruction other than its own." >&2
  echo "Derived from the committed stage data:" >&2
  printf '%s\n' "${REVIEWED_STATUS}" | sed 's/^/  /' >&2
  echo "Found by git in the reconstruction:" >&2
  printf '%s\n' "${reconstruction_status}" | sed 's/^/  /' >&2
  echo "The transaction writes only the paths its stages name, so this is a defect" \
    "in the patch framework or the stage data, not in any tree you hold." >&2
  echo "Nothing was built or written." >&2
  exit 1
fi
git -C "${RECONSTRUCTION}" diff --check

if [[ "${MODE}" == "materialise" ]]; then
  # The tree's index records the reconstruction, so `git diff` there shows
  # exactly what an author changes on top of it.
  git -C "${RECONSTRUCTION}" add --all
  reconstruction_held=false
  printf 'Materialised the verified reconstruction at %s:\n' "${RECONSTRUCTION}"
  printf 'vLLM %s with every reviewed stage applied, and recorded in its index.\n' \
    "${VLLM_COMMIT}"
  printf 'Nothing reads this tree. Author a stage there: git -C %q diff is then\n' \
    "${RECONSTRUCTION}"
  printf 'exactly your change, the review diff patches/source_patch_v1/compile_review_diff.py\n'
  printf 'compiles (run git add --intent-to-add on a new file first).\n'
  printf 'Remove it with: git -C %q worktree remove --force %q\n' \
    "${PROJECT_DIR}/vllm" "${RECONSTRUCTION}"
  exit 0
fi

# The build context is exactly what the runtime Dockerfile copies -- each file
# from the reconstruction when it lies under vllm/, from this repository
# otherwise -- and the Dockerfile itself. Every file is 0644, every directory
# 0755 and every timestamp SOURCE_DATE_EPOCH, so the context records the
# committed inputs and nothing of the machine or the moment that assembled it:
# COPY keeps these timestamps, and the .pyc files the build's units write
# embed them.
BUILD_CONTEXT="${BUILD_EXPORT_DIR}/context"
readonly BUILD_CONTEXT
context_sources="$(
  sed -e ':joined' -e '/\\$/{N;s/\\\n/ /;b joined' -e '}' "${DOCKERFILE}" |
    awk '$1 == "COPY" || $1 == "ADD" {
      if ($1 == "COPY" && NF == 4 && $2 == "--chmod=0644") print $3
      else print "UNREADABLE " $0
    }'
)"
if grep -q '^UNREADABLE ' <<<"${context_sources}"; then
  echo "The runtime Dockerfile reads its context in a form the context assembly does not:" >&2
  sed -n 's/^UNREADABLE /  /p' <<<"${context_sources}" >&2
  echo "Every context file is copied by one 'COPY --chmod=0644 SOURCE DESTINATION'." >&2
  echo "Next: write the instruction in that form, or extend this assembly to the new one." >&2
  exit 1
fi
install -d -m 0755 "${BUILD_CONTEXT}"
while IFS= read -r context_source; do
  case "${context_source}" in
    vllm/*) context_origin="${RECONSTRUCTION}/${context_source#vllm/}" ;;
    *) context_origin="${PROJECT_DIR}/${context_source}" ;;
  esac
  if [[ ! -f "${context_origin}" || -L "${context_origin}" ]]; then
    echo "The runtime Dockerfile copies ${context_source}, and ${context_origin}" \
      "is not a regular file." >&2
    echo "Next: copy only files the reviewed patch set or this repository holds." >&2
    exit 1
  fi
  install -D -m 0644 -- "${context_origin}" "${BUILD_CONTEXT}/${context_source}"
done <<<"${context_sources}"
install -D -m 0644 -- "${DOCKERFILE}" "${BUILD_CONTEXT}/containers/Dockerfile.runtime"
find "${BUILD_CONTEXT}" -mindepth 1 -type d -exec chmod 0755 {} +
find "${BUILD_CONTEXT}" -exec touch --no-dereference --date="@${SOURCE_DATE_EPOCH}" {} +
# The reconstruction stays until the reviewed tests below have run from it.

# Everything that determines the image, and nothing that does not: the base
# image, every option and argument the build is given -- the options below
# are passed exactly as listed -- and every entry of the context with its
# type, mode, mtime and bytes. The digest of this manifest is the identity of
# the build's inputs, so a change to any of them is a change of inputs and
# nothing else is. The progress format, the context's temporary path and the
# archive's are not inputs: they name where this run works, not what it builds.
image_build_options=(
  --builder default
  --platform linux/amd64
  --network none
  --pull=false
  --provenance=false
  --no-cache
  --target runtime
  --build-arg "BASE_IMAGE=${BASE_IMAGE_TAG}"
  --build-arg "BLOCK_POOL_UPSTREAM_FILE_SHA256=${BLOCK_POOL_UPSTREAM_FILE_SHA256}"
  --build-arg "BLOCK_POOL_PATCHED_FILE_SHA256=${BLOCK_POOL_PATCHED_FILE_SHA256}"
  --build-arg "SINGLE_TYPE_KV_CACHE_MANAGER_UPSTREAM_FILE_SHA256=${SINGLE_TYPE_KV_CACHE_MANAGER_UPSTREAM_FILE_SHA256}"
  --build-arg "SINGLE_TYPE_KV_CACHE_MANAGER_PATCHED_FILE_SHA256=${SINGLE_TYPE_KV_CACHE_MANAGER_PATCHED_FILE_SHA256}"
  --build-arg "KV_OFFLOAD_CPU_COMMON_UPSTREAM_FILE_SHA256=${KV_OFFLOAD_CPU_COMMON_UPSTREAM_FILE_SHA256}"
  --build-arg "KV_OFFLOAD_CPU_COMMON_PATCHED_FILE_SHA256=${KV_OFFLOAD_CPU_COMMON_PATCHED_FILE_SHA256}"
  --build-arg "SHARED_PREFIX_CACHE_UNIT_SHA256=${SHARED_PREFIX_CACHE_UNIT_SHA256}"
  --build-arg "RAW_MEDIA_UNIT_SHA256=${RAW_MEDIA_UNIT_SHA256}"
  --build-arg "GENERATE_RESULT_UNIT_SHA256=${GENERATE_RESULT_UNIT_SHA256}"
  --build-arg "TURBOQUANT_UPSTREAM_FILE_SHA256=${TURBOQUANT_UPSTREAM_FILE_SHA256}"
  --build-arg "TURBOQUANT_DECODE_UPSTREAM_FILE_SHA256=${TURBOQUANT_DECODE_UPSTREAM_FILE_SHA256}"
  --build-arg "TURBOQUANT_STORE_UPSTREAM_FILE_SHA256=${TURBOQUANT_STORE_UPSTREAM_FILE_SHA256}"
  --build-arg "TOOL_SCHEMA_UPSTREAM_FILE_SHA256=${TOOL_SCHEMA_UPSTREAM_FILE_SHA256}"
  --build-arg "TOOL_PARSER_ABSTRACT_UPSTREAM_FILE_SHA256=${TOOL_PARSER_ABSTRACT_UPSTREAM_FILE_SHA256}"
  --build-arg "TOOL_CALL_FILTER_UPSTREAM_FILE_SHA256=${TOOL_CALL_FILTER_UPSTREAM_FILE_SHA256}"
  --build-arg "MODEL_CONFIG_UPSTREAM_FILE_SHA256=${MODEL_CONFIG_UPSTREAM_FILE_SHA256}"
  --build-arg "ANTHROPIC_PROTOCOL_UPSTREAM_FILE_SHA256=${ANTHROPIC_PROTOCOL_UPSTREAM_FILE_SHA256}"
  --build-arg "ANTHROPIC_SERVING_UPSTREAM_FILE_SHA256=${ANTHROPIC_SERVING_UPSTREAM_FILE_SHA256}"
  --build-arg "ERROR_RESPONSE_UPSTREAM_FILE_SHA256=${ERROR_RESPONSE_UPSTREAM_FILE_SHA256}"
  --build-arg "EXCEPTION_EXCEPTION_HANDLER_UPSTREAM_FILE_SHA256=${EXCEPTION_EXCEPTION_HANDLER_UPSTREAM_FILE_SHA256}"
  --build-arg "HTTP_EXCEPTION_HANDLER_UPSTREAM_FILE_SHA256=${HTTP_EXCEPTION_HANDLER_UPSTREAM_FILE_SHA256}"
  --build-arg "VALIDATION_EXCEPTION_HANDLER_UPSTREAM_FILE_SHA256=${VALIDATION_EXCEPTION_HANDLER_UPSTREAM_FILE_SHA256}"
  --build-arg "VLLM_ERROR_EXCEPTION_HANDLER_UPSTREAM_FILE_SHA256=${VLLM_ERROR_EXCEPTION_HANDLER_UPSTREAM_FILE_SHA256}"
  --build-arg "CHAT_PROTOCOL_UPSTREAM_FILE_SHA256=${CHAT_PROTOCOL_UPSTREAM_FILE_SHA256}"
  --build-arg "SAMPLING_PARAMS_UPSTREAM_FILE_SHA256=${SAMPLING_PARAMS_UPSTREAM_FILE_SHA256}"
  --build-arg "SCHED_UTILS_UPSTREAM_FILE_SHA256=${SCHED_UTILS_UPSTREAM_FILE_SHA256}"
  --build-arg "INPUT_PROCESSOR_UPSTREAM_FILE_SHA256=${INPUT_PROCESSOR_UPSTREAM_FILE_SHA256}"
  --build-arg "REQUEST_UPSTREAM_FILE_SHA256=${REQUEST_UPSTREAM_FILE_SHA256}"
  --build-arg "QWEN3_PARSER_UPSTREAM_FILE_SHA256=${QWEN3_PARSER_UPSTREAM_FILE_SHA256}"
  --build-arg "STRUCTURED_OUTPUT_UPSTREAM_FILE_SHA256=${STRUCTURED_OUTPUT_UPSTREAM_FILE_SHA256}"
  --build-arg "ANTHROPIC_API_ROUTER_UPSTREAM_FILE_SHA256=${ANTHROPIC_API_ROUTER_UPSTREAM_FILE_SHA256}"
  --build-arg "CHAT_SERVING_UPSTREAM_FILE_SHA256=${CHAT_SERVING_UPSTREAM_FILE_SHA256}"
  --build-arg "RESPONSES_CONTEXT_UPSTREAM_FILE_SHA256=${RESPONSES_CONTEXT_UPSTREAM_FILE_SHA256}"
  --build-arg "RESPONSES_PROTOCOL_UPSTREAM_FILE_SHA256=${RESPONSES_PROTOCOL_UPSTREAM_FILE_SHA256}"
  --build-arg "RESPONSES_SERVING_UPSTREAM_FILE_SHA256=${RESPONSES_SERVING_UPSTREAM_FILE_SHA256}"
  --build-arg "RESPONSES_STREAMING_UPSTREAM_FILE_SHA256=${RESPONSES_STREAMING_UPSTREAM_FILE_SHA256}"
  --build-arg "RESPONSES_UTILS_UPSTREAM_FILE_SHA256=${RESPONSES_UTILS_UPSTREAM_FILE_SHA256}"
  --build-arg "PARSER_ENGINE_UPSTREAM_FILE_SHA256=${PARSER_ENGINE_UPSTREAM_FILE_SHA256}"
  --build-arg "KV_OFFLOAD_WORKER_UPSTREAM_FILE_SHA256=${KV_OFFLOAD_WORKER_UPSTREAM_FILE_SHA256}"
  --build-arg "WORKSPACE_UPSTREAM_FILE_SHA256=${WORKSPACE_UPSTREAM_FILE_SHA256}"
  --build-arg "GPU_MODEL_RUNNER_UPSTREAM_FILE_SHA256=${GPU_MODEL_RUNNER_UPSTREAM_FILE_SHA256}"
  --build-arg "API_UTILS_UPSTREAM_FILE_SHA256=${API_UTILS_UPSTREAM_FILE_SHA256}"
  --build-arg "ENVS_UPSTREAM_FILE_SHA256=${ENVS_UPSTREAM_FILE_SHA256}"
  --build-arg "CHAT_UTILS_UPSTREAM_FILE_SHA256=${CHAT_UTILS_UPSTREAM_FILE_SHA256}"
  --build-arg "MEDIA_CONNECTOR_UPSTREAM_FILE_SHA256=${MEDIA_CONNECTOR_UPSTREAM_FILE_SHA256}"
  --build-arg "IMAGE_MEDIA_UPSTREAM_FILE_SHA256=${IMAGE_MEDIA_UPSTREAM_FILE_SHA256}"
  --build-arg "RENDER_PARAMS_UPSTREAM_FILE_SHA256=${RENDER_PARAMS_UPSTREAM_FILE_SHA256}"
  --build-arg "QWEN3_VL_MODEL_UPSTREAM_FILE_SHA256=${QWEN3_VL_MODEL_UPSTREAM_FILE_SHA256}"
  --build-arg "ABSTRACT_PARSER_UPSTREAM_FILE_SHA256=${ABSTRACT_PARSER_UPSTREAM_FILE_SHA256}"
  --build-arg "PARSER_ADAPTERS_UPSTREAM_FILE_SHA256=${PARSER_ADAPTERS_UPSTREAM_FILE_SHA256}"
  --build-arg "PARSER_EVENTS_UPSTREAM_FILE_SHA256=${PARSER_EVENTS_UPSTREAM_FILE_SHA256}"
  --build-arg "PARSER_ENGINE_CONFIG_UPSTREAM_FILE_SHA256=${PARSER_ENGINE_CONFIG_UPSTREAM_FILE_SHA256}"
  --build-arg "STREAMING_PARSER_ENGINE_UPSTREAM_FILE_SHA256=${STREAMING_PARSER_ENGINE_UPSTREAM_FILE_SHA256}"
  --build-arg "TOKEN_ID_SCANNER_UPSTREAM_FILE_SHA256=${TOKEN_ID_SCANNER_UPSTREAM_FILE_SHA256}"
  --build-arg "ENGINE_PROTOCOL_UPSTREAM_FILE_SHA256=${ENGINE_PROTOCOL_UPSTREAM_FILE_SHA256}"
  --build-arg "TURBOQUANT_PATCHED_FILE_SHA256=${TURBOQUANT_PATCHED_FILE_SHA256}"
  --build-arg "TURBOQUANT_DECODE_PATCHED_FILE_SHA256=${TURBOQUANT_DECODE_PATCHED_FILE_SHA256}"
  --build-arg "TURBOQUANT_STORE_PATCHED_FILE_SHA256=${TURBOQUANT_STORE_PATCHED_FILE_SHA256}"
  --build-arg "TOOL_SCHEMA_PATCHED_FILE_SHA256=${TOOL_SCHEMA_PATCHED_FILE_SHA256}"
  --build-arg "TOOL_PARSER_ABSTRACT_PATCHED_FILE_SHA256=${TOOL_PARSER_ABSTRACT_PATCHED_FILE_SHA256}"
  --build-arg "MODEL_CONFIG_PATCHED_FILE_SHA256=${MODEL_CONFIG_PATCHED_FILE_SHA256}"
  --build-arg "ANTHROPIC_PROTOCOL_PATCHED_FILE_SHA256=${ANTHROPIC_PROTOCOL_PATCHED_FILE_SHA256}"
  --build-arg "ANTHROPIC_SERVING_PATCHED_FILE_SHA256=${ANTHROPIC_SERVING_PATCHED_FILE_SHA256}"
  --build-arg "ERROR_RESPONSE_PATCHED_FILE_SHA256=${ERROR_RESPONSE_PATCHED_FILE_SHA256}"
  --build-arg "EXCEPTION_EXCEPTION_HANDLER_PATCHED_FILE_SHA256=${EXCEPTION_EXCEPTION_HANDLER_PATCHED_FILE_SHA256}"
  --build-arg "HTTP_EXCEPTION_HANDLER_PATCHED_FILE_SHA256=${HTTP_EXCEPTION_HANDLER_PATCHED_FILE_SHA256}"
  --build-arg "VALIDATION_EXCEPTION_HANDLER_PATCHED_FILE_SHA256=${VALIDATION_EXCEPTION_HANDLER_PATCHED_FILE_SHA256}"
  --build-arg "VLLM_ERROR_EXCEPTION_HANDLER_PATCHED_FILE_SHA256=${VLLM_ERROR_EXCEPTION_HANDLER_PATCHED_FILE_SHA256}"
  --build-arg "CHAT_PROTOCOL_PATCHED_FILE_SHA256=${CHAT_PROTOCOL_PATCHED_FILE_SHA256}"
  --build-arg "SAMPLING_PARAMS_PATCHED_FILE_SHA256=${SAMPLING_PARAMS_PATCHED_FILE_SHA256}"
  --build-arg "SCHED_UTILS_PATCHED_FILE_SHA256=${SCHED_UTILS_PATCHED_FILE_SHA256}"
  --build-arg "INPUT_PROCESSOR_PATCHED_FILE_SHA256=${INPUT_PROCESSOR_PATCHED_FILE_SHA256}"
  --build-arg "REQUEST_PATCHED_FILE_SHA256=${REQUEST_PATCHED_FILE_SHA256}"
  --build-arg "QWEN3_PARSER_PATCHED_FILE_SHA256=${QWEN3_PARSER_PATCHED_FILE_SHA256}"
  --build-arg "STRUCTURED_OUTPUT_PATCHED_FILE_SHA256=${STRUCTURED_OUTPUT_PATCHED_FILE_SHA256}"
  --build-arg "HF_RENDERER_PATCHED_FILE_SHA256=${HF_RENDERER_PATCHED_FILE_SHA256}"
  --build-arg "HF_RENDERER_UPSTREAM_FILE_SHA256=${HF_RENDERER_UPSTREAM_FILE_SHA256}"
  --build-arg "TEMPLATE_AUTHORSHIP_PATCHED_FILE_SHA256=${TEMPLATE_AUTHORSHIP_PATCHED_FILE_SHA256}"
  --build-arg "ONLINE_RENDERER_UPSTREAM_FILE_SHA256=${ONLINE_RENDERER_UPSTREAM_FILE_SHA256}"
  --build-arg "ONLINE_RENDERER_PATCHED_FILE_SHA256=${ONLINE_RENDERER_PATCHED_FILE_SHA256}"
  --build-arg "SCORING_IO_PROCESSOR_PATCHED_FILE_SHA256=${SCORING_IO_PROCESSOR_PATCHED_FILE_SHA256}"
  --build-arg "SCORING_IO_PROCESSOR_UPSTREAM_FILE_SHA256=${SCORING_IO_PROCESSOR_UPSTREAM_FILE_SHA256}"
  --build-arg "LINEAR_KERNELS_PATCHED_FILE_SHA256=${LINEAR_KERNELS_PATCHED_FILE_SHA256}"
  --build-arg "LINEAR_KERNELS_UPSTREAM_FILE_SHA256=${LINEAR_KERNELS_UPSTREAM_FILE_SHA256}"
  --build-arg "TOKENIZE_PROTOCOL_PATCHED_FILE_SHA256=${TOKENIZE_PROTOCOL_PATCHED_FILE_SHA256}"
  --build-arg "TOKENIZE_PROTOCOL_UPSTREAM_FILE_SHA256=${TOKENIZE_PROTOCOL_UPSTREAM_FILE_SHA256}"
  --build-arg "EXCEPTION_REGISTRATION_PATCHED_FILE_SHA256=${EXCEPTION_REGISTRATION_PATCHED_FILE_SHA256}"
  --build-arg "EXCEPTION_REGISTRATION_UPSTREAM_FILE_SHA256=${EXCEPTION_REGISTRATION_UPSTREAM_FILE_SHA256}"
  --build-arg "DETOKENIZER_UTILS_PATCHED_FILE_SHA256=${DETOKENIZER_UTILS_PATCHED_FILE_SHA256}"
  --build-arg "DETOKENIZER_UTILS_UPSTREAM_FILE_SHA256=${DETOKENIZER_UTILS_UPSTREAM_FILE_SHA256}"
  --build-arg "TOOL_PARSER_UTILS_PATCHED_FILE_SHA256=${TOOL_PARSER_UTILS_PATCHED_FILE_SHA256}"
  --build-arg "TOOL_PARSER_UTILS_UPSTREAM_FILE_SHA256=${TOOL_PARSER_UTILS_UPSTREAM_FILE_SHA256}"
  --build-arg "V1_THINKING_BUDGET_PATCHED_FILE_SHA256=${V1_THINKING_BUDGET_PATCHED_FILE_SHA256}"
  --build-arg "V1_THINKING_BUDGET_UPSTREAM_FILE_SHA256=${V1_THINKING_BUDGET_UPSTREAM_FILE_SHA256}"
  --build-arg "BASE_REASONING_PARSER_PATCHED_FILE_SHA256=${BASE_REASONING_PARSER_PATCHED_FILE_SHA256}"
  --build-arg "BASE_REASONING_PARSER_UPSTREAM_FILE_SHA256=${BASE_REASONING_PARSER_UPSTREAM_FILE_SHA256}"
  --build-arg "REASONING_CONFIG_PATCHED_FILE_SHA256=${REASONING_CONFIG_PATCHED_FILE_SHA256}"
  --build-arg "REASONING_CONFIG_UPSTREAM_FILE_SHA256=${REASONING_CONFIG_UPSTREAM_FILE_SHA256}"
  --build-arg "STRUCTURAL_TAG_STOP_CHECKER_PATCHED_FILE_SHA256=${STRUCTURAL_TAG_STOP_CHECKER_PATCHED_FILE_SHA256}"
  --build-arg "XGRAMMAR_BACKEND_PATCHED_FILE_SHA256=${XGRAMMAR_BACKEND_PATCHED_FILE_SHA256}"
  --build-arg "XGRAMMAR_BACKEND_UPSTREAM_FILE_SHA256=${XGRAMMAR_BACKEND_UPSTREAM_FILE_SHA256}"
  --build-arg "STRUCTURED_OUTPUT_BACKEND_TYPES_PATCHED_FILE_SHA256=${STRUCTURED_OUTPUT_BACKEND_TYPES_PATCHED_FILE_SHA256}"
  --build-arg "STRUCTURED_OUTPUT_BACKEND_TYPES_UPSTREAM_FILE_SHA256=${STRUCTURED_OUTPUT_BACKEND_TYPES_UPSTREAM_FILE_SHA256}"
  --build-arg "V1_OUTPUT_PROCESSOR_PATCHED_FILE_SHA256=${V1_OUTPUT_PROCESSOR_PATCHED_FILE_SHA256}"
  --build-arg "V1_OUTPUT_PROCESSOR_UPSTREAM_FILE_SHA256=${V1_OUTPUT_PROCESSOR_UPSTREAM_FILE_SHA256}"
  --build-arg "V1_DETOKENIZER_PATCHED_FILE_SHA256=${V1_DETOKENIZER_PATCHED_FILE_SHA256}"
  --build-arg "V1_DETOKENIZER_UPSTREAM_FILE_SHA256=${V1_DETOKENIZER_UPSTREAM_FILE_SHA256}"
  --build-arg "V1_SCHEDULER_PATCHED_FILE_SHA256=${V1_SCHEDULER_PATCHED_FILE_SHA256}"
  --build-arg "V1_SCHEDULER_UPSTREAM_FILE_SHA256=${V1_SCHEDULER_UPSTREAM_FILE_SHA256}"
  --build-arg "ANTHROPIC_API_ROUTER_PATCHED_FILE_SHA256=${ANTHROPIC_API_ROUTER_PATCHED_FILE_SHA256}"
  --build-arg "CHAT_SERVING_PATCHED_FILE_SHA256=${CHAT_SERVING_PATCHED_FILE_SHA256}"
  --build-arg "RESPONSES_CONTEXT_PATCHED_FILE_SHA256=${RESPONSES_CONTEXT_PATCHED_FILE_SHA256}"
  --build-arg "RESPONSES_PROTOCOL_PATCHED_FILE_SHA256=${RESPONSES_PROTOCOL_PATCHED_FILE_SHA256}"
  --build-arg "RESPONSES_SERVING_PATCHED_FILE_SHA256=${RESPONSES_SERVING_PATCHED_FILE_SHA256}"
  --build-arg "RESPONSES_STREAMING_PATCHED_FILE_SHA256=${RESPONSES_STREAMING_PATCHED_FILE_SHA256}"
  --build-arg "RESPONSES_UTILS_PATCHED_FILE_SHA256=${RESPONSES_UTILS_PATCHED_FILE_SHA256}"
  --build-arg "PARSER_ENGINE_PATCHED_FILE_SHA256=${PARSER_ENGINE_PATCHED_FILE_SHA256}"
  --build-arg "KV_OFFLOAD_WORKER_PATCHED_FILE_SHA256=${KV_OFFLOAD_WORKER_PATCHED_FILE_SHA256}"
  --build-arg "WORKSPACE_PATCHED_FILE_SHA256=${WORKSPACE_PATCHED_FILE_SHA256}"
  --build-arg "GPU_MODEL_RUNNER_PATCHED_FILE_SHA256=${GPU_MODEL_RUNNER_PATCHED_FILE_SHA256}"
  --build-arg "API_UTILS_PATCHED_FILE_SHA256=${API_UTILS_PATCHED_FILE_SHA256}"
  --build-arg "ENVS_PATCHED_FILE_SHA256=${ENVS_PATCHED_FILE_SHA256}"
  --build-arg "CHAT_UTILS_PATCHED_FILE_SHA256=${CHAT_UTILS_PATCHED_FILE_SHA256}"
  --build-arg "MEDIA_CONNECTOR_PATCHED_FILE_SHA256=${MEDIA_CONNECTOR_PATCHED_FILE_SHA256}"
  --build-arg "IMAGE_MEDIA_PATCHED_FILE_SHA256=${IMAGE_MEDIA_PATCHED_FILE_SHA256}"
  --build-arg "RENDER_PARAMS_PATCHED_FILE_SHA256=${RENDER_PARAMS_PATCHED_FILE_SHA256}"
  --build-arg "QWEN3_VL_MODEL_PATCHED_FILE_SHA256=${QWEN3_VL_MODEL_PATCHED_FILE_SHA256}"
  --build-arg "AGENT_CHAT_TEMPLATE_SHA256=${AGENT_CHAT_TEMPLATE_SHA256}"
  --build-arg "PHASE_BUDGET_UNIT_SHA256=${PHASE_BUDGET_UNIT_SHA256}"
  --build-arg "VISION_WORKSPACE_UNIT_SHA256=${VISION_WORKSPACE_UNIT_SHA256}"
  --build-arg "VISION_CONTRACT_UNIT_SHA256=${VISION_CONTRACT_UNIT_SHA256}"
  --build-arg "VISION_MLP_UNIT_SHA256=${VISION_MLP_UNIT_SHA256}"
  --build-arg "TURBOQUANT_GUARD_UNIT_SHA256=${TURBOQUANT_GUARD_UNIT_SHA256}"
  --build-arg "TURBOQUANT_K8V4_UNIT_SHA256=${TURBOQUANT_K8V4_UNIT_SHA256}"
  --build-arg "QWEN38_CONTEXT_UNIT_SHA256=${QWEN38_CONTEXT_UNIT_SHA256}"
  --build-arg "CHAT_TEMPLATE_RETENTION_UNIT_SHA256=${CHAT_TEMPLATE_RETENTION_UNIT_SHA256}"
  --build-arg "NVFP4_KERNEL_UNIT_SHA256=${NVFP4_KERNEL_UNIT_SHA256}"
  --build-arg "REASONING_USAGE_UNIT_SHA256=${REASONING_USAGE_UNIT_SHA256}"
  --build-arg "TOOL_OUTPUT_PARSER_UNIT_SHA256=${TOOL_OUTPUT_PARSER_UNIT_SHA256}"
  --build-arg "RESPONSES_STREAM_RULE_SHA256=${RESPONSES_STREAM_RULE_SHA256}"
  --build-arg "QWEN_GRAMMAR_UNIT_SHA256=${QWEN_GRAMMAR_UNIT_SHA256}"
  --build-arg "TEMPLATE_AUTHORSHIP_UNIT_SHA256=${TEMPLATE_AUTHORSHIP_UNIT_SHA256}"
  --build-arg "NATIVE_FP4_SELECTION_UNIT_SHA256=${NATIVE_FP4_SELECTION_UNIT_SHA256}"
  --build-arg "TURBOQUANT_PATCH_DIFF_SHA256=${TURBOQUANT_PATCH_DIFF_SHA256}"
  --build-arg "TOOL_SCHEMA_PATCH_DIFF_SHA256=${TOOL_SCHEMA_PATCH_DIFF_SHA256}"
  --build-arg "AGENT_DEFAULTS_PATCH_DIFF_SHA256=${AGENT_DEFAULTS_PATCH_DIFF_SHA256}"
  --build-arg "PHASE_BUDGET_PATCH_DIFF_SHA256=${PHASE_BUDGET_PATCH_DIFF_SHA256}"
  --build-arg "IMPLICIT_TOOL_GRAMMAR_PATCH_DIFF_SHA256=${IMPLICIT_TOOL_GRAMMAR_PATCH_DIFF_SHA256}"
  --build-arg "ANTHROPIC_VALIDATION_PATCH_DIFF_SHA256=${ANTHROPIC_VALIDATION_PATCH_DIFF_SHA256}"
  --build-arg "ANTHROPIC_INPUTS_PATCH_DIFF_SHA256=${ANTHROPIC_INPUTS_PATCH_DIFF_SHA256}"
  --build-arg "QWEN_LANGUAGE_PATCH_DIFF_SHA256=${QWEN_LANGUAGE_PATCH_DIFF_SHA256}"
  --build-arg "PNG_SOURCE_PATCH_DIFF_SHA256=${PNG_SOURCE_PATCH_DIFF_SHA256}"
  --build-arg "KV_PHYSICAL_PATCH_DIFF_SHA256=${KV_PHYSICAL_PATCH_DIFF_SHA256}"
  --build-arg "SINGLE_CALL_PATCH_DIFF_SHA256=${SINGLE_CALL_PATCH_DIFF_SHA256}"
  --build-arg "RESPONSES_HISTORY_PATCH_DIFF_SHA256=${RESPONSES_HISTORY_PATCH_DIFF_SHA256}"
  --build-arg "RESPONSES_IDENTITY_PATCH_DIFF_SHA256=${RESPONSES_IDENTITY_PATCH_DIFF_SHA256}"
  --build-arg "ANTHROPIC_TERMINAL_PATCH_DIFF_SHA256=${ANTHROPIC_TERMINAL_PATCH_DIFF_SHA256}"
  --build-arg "SAMPLING_RESOLUTION_PATCH_DIFF_SHA256=${SAMPLING_RESOLUTION_PATCH_DIFF_SHA256}"
  --build-arg "GEMMA4_PARSER_UPSTREAM_FILE_SHA256=${GEMMA4_PARSER_UPSTREAM_FILE_SHA256}"
  --build-arg "GEMMA4_PARSER_PATCHED_FILE_SHA256=${GEMMA4_PARSER_PATCHED_FILE_SHA256}"
  --build-arg "GLM47_PARSER_UPSTREAM_FILE_SHA256=${GLM47_PARSER_UPSTREAM_FILE_SHA256}"
  --build-arg "GLM47_PARSER_PATCHED_FILE_SHA256=${GLM47_PARSER_PATCHED_FILE_SHA256}"
  --build-arg "MINIMAX_M2_PARSER_UPSTREAM_FILE_SHA256=${MINIMAX_M2_PARSER_UPSTREAM_FILE_SHA256}"
  --build-arg "MINIMAX_M2_PARSER_PATCHED_FILE_SHA256=${MINIMAX_M2_PARSER_PATCHED_FILE_SHA256}"
  --build-arg "MISTRAL_PARSER_UPSTREAM_FILE_SHA256=${MISTRAL_PARSER_UPSTREAM_FILE_SHA256}"
  --build-arg "HARMONY_PARSER_UPSTREAM_FILE_SHA256=${HARMONY_PARSER_UPSTREAM_FILE_SHA256}"
  --build-arg "MISTRAL_PARSER_PATCHED_FILE_SHA256=${MISTRAL_PARSER_PATCHED_FILE_SHA256}"
  --build-arg "HARMONY_PARSER_PATCHED_FILE_SHA256=${HARMONY_PARSER_PATCHED_FILE_SHA256}"
  --build-arg "PHASE_AWARE_PARSER_TERMINALS_PATCH_DIFF_SHA256=${PHASE_AWARE_PARSER_TERMINALS_PATCH_DIFF_SHA256}"
  --build-arg "TOOL_OUTPUT_COMPLETION_PATCH_DIFF_SHA256=${TOOL_OUTPUT_COMPLETION_PATCH_DIFF_SHA256}"
  --build-arg "ONE_WAY_THINKING_BOUNDARY_PATCH_DIFF_SHA256=${ONE_WAY_THINKING_BOUNDARY_PATCH_DIFF_SHA256}"
  --build-arg "SCHEMA_FAITHFUL_XML_PATCH_DIFF_SHA256=${SCHEMA_FAITHFUL_XML_PATCH_DIFF_SHA256}"
  --build-arg "TOKEN_TEXT_PROVENANCE_PATCH_DIFF_SHA256=${TOKEN_TEXT_PROVENANCE_PATCH_DIFF_SHA256}"
  --build-arg "PRECISE_REQUEST_ERRORS_PATCH_DIFF_SHA256=${PRECISE_REQUEST_ERRORS_PATCH_DIFF_SHA256}"
  --build-arg "INPUT_STREAM_AGENT_IDENTITY_PATCH_DIFF_SHA256=${INPUT_STREAM_AGENT_IDENTITY_PATCH_DIFF_SHA256}"
  --build-arg "ASYNC_LLM_UPSTREAM_FILE_SHA256=${ASYNC_LLM_UPSTREAM_FILE_SHA256}"
  --build-arg "ASYNC_LLM_PATCHED_FILE_SHA256=${ASYNC_LLM_PATCHED_FILE_SHA256}"
  --build-arg "DEEPSEEK_V32_PARSER_UPSTREAM_FILE_SHA256=${DEEPSEEK_V32_PARSER_UPSTREAM_FILE_SHA256}"
  --build-arg "DEEPSEEK_V32_PARSER_PATCHED_FILE_SHA256=${DEEPSEEK_V32_PARSER_PATCHED_FILE_SHA256}"
  --build-arg "DEEPSEEK_V4_PARSER_UPSTREAM_FILE_SHA256=${DEEPSEEK_V4_PARSER_UPSTREAM_FILE_SHA256}"
  --build-arg "DEEPSEEK_V4_PARSER_PATCHED_FILE_SHA256=${DEEPSEEK_V4_PARSER_PATCHED_FILE_SHA256}"
  --build-arg "INKLING_PARSER_UPSTREAM_FILE_SHA256=${INKLING_PARSER_UPSTREAM_FILE_SHA256}"
  --build-arg "INKLING_PARSER_PATCHED_FILE_SHA256=${INKLING_PARSER_PATCHED_FILE_SHA256}"
  --build-arg "KIMI_K2_PARSER_UPSTREAM_FILE_SHA256=${KIMI_K2_PARSER_UPSTREAM_FILE_SHA256}"
  --build-arg "KIMI_K2_PARSER_PATCHED_FILE_SHA256=${KIMI_K2_PARSER_PATCHED_FILE_SHA256}"
  --build-arg "KIMI_K3_PARSER_UPSTREAM_FILE_SHA256=${KIMI_K3_PARSER_UPSTREAM_FILE_SHA256}"
  --build-arg "KIMI_K3_PARSER_PATCHED_FILE_SHA256=${KIMI_K3_PARSER_PATCHED_FILE_SHA256}"
  --build-arg "XML_TEXT_FIDELITY_PATCH_DIFF_SHA256=${XML_TEXT_FIDELITY_PATCH_DIFF_SHA256}"
  --build-arg "DERENDER_SERVING_UPSTREAM_FILE_SHA256=${DERENDER_SERVING_UPSTREAM_FILE_SHA256}"
  --build-arg "DERENDER_SERVING_PATCHED_FILE_SHA256=${DERENDER_SERVING_PATCHED_FILE_SHA256}"
  --build-arg "MM_PROCESSOR_INPUTS_UPSTREAM_FILE_SHA256=${MM_PROCESSOR_INPUTS_UPSTREAM_FILE_SHA256}"
  --build-arg "MM_PROCESSOR_INPUTS_PATCHED_FILE_SHA256=${MM_PROCESSOR_INPUTS_PATCHED_FILE_SHA256}"
  --build-arg "MM_PROCESSOR_UPSTREAM_FILE_SHA256=${MM_PROCESSOR_UPSTREAM_FILE_SHA256}"
  --build-arg "MM_PROCESSOR_PATCHED_FILE_SHA256=${MM_PROCESSOR_PATCHED_FILE_SHA256}"
  --build-arg "BASE_RENDERER_UPSTREAM_FILE_SHA256=${BASE_RENDERER_UPSTREAM_FILE_SHA256}"
  --build-arg "BASE_RENDERER_PATCHED_FILE_SHA256=${BASE_RENDERER_PATCHED_FILE_SHA256}"
  --build-arg "RAW_IMAGE_TRANSPORT_PATCH_DIFF_SHA256=${RAW_IMAGE_TRANSPORT_PATCH_DIFF_SHA256}"
  --build-arg "MM_SERDE_UPSTREAM_FILE_SHA256=${MM_SERDE_UPSTREAM_FILE_SHA256}"
  --build-arg "ONLINE_DERENDERER_UPSTREAM_FILE_SHA256=${ONLINE_DERENDERER_UPSTREAM_FILE_SHA256}"
  --build-arg "ONLINE_DERENDERER_PATCHED_FILE_SHA256=${ONLINE_DERENDERER_PATCHED_FILE_SHA256}"
  --build-arg "GENERATE_RESULT_PATCH_DIFF_SHA256=${GENERATE_RESULT_PATCH_DIFF_SHA256}"
  --build-arg "COMPLETION_SERVING_UPSTREAM_FILE_SHA256=${COMPLETION_SERVING_UPSTREAM_FILE_SHA256}"
  --build-arg "COMPLETION_SERVING_PATCHED_FILE_SHA256=${COMPLETION_SERVING_PATCHED_FILE_SHA256}"
  --build-arg "CHAT_API_ROUTER_UPSTREAM_FILE_SHA256=${CHAT_API_ROUTER_UPSTREAM_FILE_SHA256}"
  --build-arg "CHAT_API_ROUTER_PATCHED_FILE_SHA256=${CHAT_API_ROUTER_PATCHED_FILE_SHA256}"
  --build-arg "CHAT_BATCH_SERVING_UPSTREAM_FILE_SHA256=${CHAT_BATCH_SERVING_UPSTREAM_FILE_SHA256}"
  --build-arg "CHAT_BATCH_SERVING_PATCHED_FILE_SHA256=${CHAT_BATCH_SERVING_PATCHED_FILE_SHA256}"
  --build-arg "COMPLETION_API_ROUTER_UPSTREAM_FILE_SHA256=${COMPLETION_API_ROUTER_UPSTREAM_FILE_SHA256}"
  --build-arg "COMPLETION_API_ROUTER_PATCHED_FILE_SHA256=${COMPLETION_API_ROUTER_PATCHED_FILE_SHA256}"
  --build-arg "TITOTO_API_ROUTER_UPSTREAM_FILE_SHA256=${TITOTO_API_ROUTER_UPSTREAM_FILE_SHA256}"
  --build-arg "TITOTO_API_ROUTER_PATCHED_FILE_SHA256=${TITOTO_API_ROUTER_PATCHED_FILE_SHA256}"
  --build-arg "RUN_BATCH_UPSTREAM_FILE_SHA256=${RUN_BATCH_UPSTREAM_FILE_SHA256}"
  --build-arg "RUN_BATCH_PATCHED_FILE_SHA256=${RUN_BATCH_PATCHED_FILE_SHA256}"
  --build-arg "ENGINE_CLIENT_PROTOCOL_UPSTREAM_FILE_SHA256=${ENGINE_CLIENT_PROTOCOL_UPSTREAM_FILE_SHA256}"
  --build-arg "ENGINE_CLIENT_PROTOCOL_PATCHED_FILE_SHA256=${ENGINE_CLIENT_PROTOCOL_PATCHED_FILE_SHA256}"
  --build-arg "RENDER_SERVING_UPSTREAM_FILE_SHA256=${RENDER_SERVING_UPSTREAM_FILE_SHA256}"
  --build-arg "RENDER_SERVING_PATCHED_FILE_SHA256=${RENDER_SERVING_PATCHED_FILE_SHA256}"
  --build-arg "SAMPLING_BOUNDARY_PATCH_DIFF_SHA256=${SAMPLING_BOUNDARY_PATCH_DIFF_SHA256}"
  --build-arg "TOOL_TRUNCATION_PATCH_DIFF_SHA256=${TOOL_TRUNCATION_PATCH_DIFF_SHA256}"
  --build-arg "VISION_RUNTIME_PATCH_DIFF_SHA256=${VISION_RUNTIME_PATCH_DIFF_SHA256}"
  --build-arg "NUMERICAL_AUDITS_PATCH_DIFF_SHA256=${NUMERICAL_AUDITS_PATCH_DIFF_SHA256}"
  --build-arg "TURBOQUANT_GUARDS_PATCH_DIFF_SHA256=${TURBOQUANT_GUARDS_PATCH_DIFF_SHA256}"
  --build-arg "KV_OFFLOAD_PINNING_PATCH_DIFF_SHA256=${KV_OFFLOAD_PINNING_PATCH_DIFF_SHA256}"
  --build-arg "GENERATION_AGENT_ID_PATCH_DIFF_SHA256=${GENERATION_AGENT_ID_PATCH_DIFF_SHA256}"
  --build-arg "ATTENTION_PREFIX_HASH_PATCH_DIFF_SHA256=${ATTENTION_PREFIX_HASH_PATCH_DIFF_SHA256}"
  --build-arg "GROUPED_SPEC_GEOMETRY_PATCH_DIFF_SHA256=${GROUPED_SPEC_GEOMETRY_PATCH_DIFF_SHA256}"
  --build-arg "AGENT_OFFLOAD_RETENTION_PATCH_DIFF_SHA256=${AGENT_OFFLOAD_RETENTION_PATCH_DIFF_SHA256}"
  --build-arg "AGENTLESS_ROUTES_PATCH_DIFF_SHA256=${AGENTLESS_ROUTES_PATCH_DIFF_SHA256}"
  --build-arg "KV_DECLARED_USERS_PATCH_DIFF_SHA256=${KV_DECLARED_USERS_PATCH_DIFF_SHA256}"
  --build-arg "KV_PHYSICAL_BOUND_PATCH_DIFF_SHA256=${KV_PHYSICAL_BOUND_PATCH_DIFF_SHA256}"
  --build-arg "EXACT_REASONING_USAGE_PATCH_DIFF_SHA256=${EXACT_REASONING_USAGE_PATCH_DIFF_SHA256}"
  --build-arg "IMAGE_PROFILE_VERSION=${IMAGE_PROFILE_VERSION}"
  --build-arg "CACHE_CONFIG_UPSTREAM_FILE_SHA256=${CACHE_CONFIG_UPSTREAM_FILE_SHA256}"
  --build-arg "VLLM_CONFIG_UPSTREAM_FILE_SHA256=${VLLM_CONFIG_UPSTREAM_FILE_SHA256}"
  --build-arg "ARG_UTILS_UPSTREAM_FILE_SHA256=${ARG_UTILS_UPSTREAM_FILE_SHA256}"
  --build-arg "LLM_ENTRYPOINT_UPSTREAM_FILE_SHA256=${LLM_ENTRYPOINT_UPSTREAM_FILE_SHA256}"
  --build-arg "KV_CACHE_UTILS_UPSTREAM_FILE_SHA256=${KV_CACHE_UTILS_UPSTREAM_FILE_SHA256}"
  --build-arg "GPU_WORKER_UPSTREAM_FILE_SHA256=${GPU_WORKER_UPSTREAM_FILE_SHA256}"
  --build-arg "STARTUP_PLAN_UPSTREAM_FILE_SHA256=${STARTUP_PLAN_UPSTREAM_FILE_SHA256}"
  --build-arg "KV_OFFLOAD_CONFIG_UPSTREAM_FILE_SHA256=${KV_OFFLOAD_CONFIG_UPSTREAM_FILE_SHA256}"
  --build-arg "KV_OFFLOAD_BASE_UPSTREAM_FILE_SHA256=${KV_OFFLOAD_BASE_UPSTREAM_FILE_SHA256}"
  --build-arg "KV_OFFLOAD_CPU_SPEC_UPSTREAM_FILE_SHA256=${KV_OFFLOAD_CPU_SPEC_UPSTREAM_FILE_SHA256}"
  --build-arg "KV_OFFLOAD_CPU_MANAGER_UPSTREAM_FILE_SHA256=${KV_OFFLOAD_CPU_MANAGER_UPSTREAM_FILE_SHA256}"
  --build-arg "KV_TIERING_SPEC_UPSTREAM_FILE_SHA256=${KV_TIERING_SPEC_UPSTREAM_FILE_SHA256}"
  --build-arg "KV_TIERING_MANAGER_UPSTREAM_FILE_SHA256=${KV_TIERING_MANAGER_UPSTREAM_FILE_SHA256}"
  --build-arg "OFFLOAD_CONNECTOR_CONFIG_UPSTREAM_FILE_SHA256=${OFFLOAD_CONNECTOR_CONFIG_UPSTREAM_FILE_SHA256}"
  --build-arg "OFFLOAD_CONNECTOR_SCHEDULER_UPSTREAM_FILE_SHA256=${OFFLOAD_CONNECTOR_SCHEDULER_UPSTREAM_FILE_SHA256}"
  --build-arg "COMPLETION_PROTOCOL_UPSTREAM_FILE_SHA256=${COMPLETION_PROTOCOL_UPSTREAM_FILE_SHA256}"
  --build-arg "GENERATE_API_ROUTER_UPSTREAM_FILE_SHA256=${GENERATE_API_ROUTER_UPSTREAM_FILE_SHA256}"
  --build-arg "CLI_ARGS_UPSTREAM_FILE_SHA256=${CLI_ARGS_UPSTREAM_FILE_SHA256}"
  --build-arg "TITOTO_PROTOCOL_UPSTREAM_FILE_SHA256=${TITOTO_PROTOCOL_UPSTREAM_FILE_SHA256}"
  --build-arg "TITOTO_SERVING_UPSTREAM_FILE_SHA256=${TITOTO_SERVING_UPSTREAM_FILE_SHA256}"
  --build-arg "POLICY_PKG_INIT_UPSTREAM_FILE_SHA256=${POLICY_PKG_INIT_UPSTREAM_FILE_SHA256}"
  --build-arg "POLICY_BASE_UPSTREAM_FILE_SHA256=${POLICY_BASE_UPSTREAM_FILE_SHA256}"
  --build-arg "POLICY_FACTORY_UPSTREAM_FILE_SHA256=${POLICY_FACTORY_UPSTREAM_FILE_SHA256}"
  --build-arg "POLICY_LRU_UPSTREAM_FILE_SHA256=${POLICY_LRU_UPSTREAM_FILE_SHA256}"
  --build-arg "POLICY_ARC_UPSTREAM_FILE_SHA256=${POLICY_ARC_UPSTREAM_FILE_SHA256}"
  --build-arg "CACHE_CONFIG_PATCHED_FILE_SHA256=${CACHE_CONFIG_PATCHED_FILE_SHA256}"
  --build-arg "VLLM_CONFIG_PATCHED_FILE_SHA256=${VLLM_CONFIG_PATCHED_FILE_SHA256}"
  --build-arg "ARG_UTILS_PATCHED_FILE_SHA256=${ARG_UTILS_PATCHED_FILE_SHA256}"
  --build-arg "LLM_ENTRYPOINT_PATCHED_FILE_SHA256=${LLM_ENTRYPOINT_PATCHED_FILE_SHA256}"
  --build-arg "KV_CACHE_UTILS_PATCHED_FILE_SHA256=${KV_CACHE_UTILS_PATCHED_FILE_SHA256}"
  --build-arg "GPU_WORKER_PATCHED_FILE_SHA256=${GPU_WORKER_PATCHED_FILE_SHA256}"
  --build-arg "STARTUP_PLAN_PATCHED_FILE_SHA256=${STARTUP_PLAN_PATCHED_FILE_SHA256}"
  --build-arg "KV_OFFLOAD_CONFIG_PATCHED_FILE_SHA256=${KV_OFFLOAD_CONFIG_PATCHED_FILE_SHA256}"
  --build-arg "KV_OFFLOAD_BASE_PATCHED_FILE_SHA256=${KV_OFFLOAD_BASE_PATCHED_FILE_SHA256}"
  --build-arg "KV_OFFLOAD_CPU_SPEC_PATCHED_FILE_SHA256=${KV_OFFLOAD_CPU_SPEC_PATCHED_FILE_SHA256}"
  --build-arg "KV_OFFLOAD_CPU_MANAGER_PATCHED_FILE_SHA256=${KV_OFFLOAD_CPU_MANAGER_PATCHED_FILE_SHA256}"
  --build-arg "KV_TIERING_SPEC_PATCHED_FILE_SHA256=${KV_TIERING_SPEC_PATCHED_FILE_SHA256}"
  --build-arg "KV_TIERING_MANAGER_PATCHED_FILE_SHA256=${KV_TIERING_MANAGER_PATCHED_FILE_SHA256}"
  --build-arg "OFFLOAD_CONNECTOR_CONFIG_PATCHED_FILE_SHA256=${OFFLOAD_CONNECTOR_CONFIG_PATCHED_FILE_SHA256}"
  --build-arg "OFFLOAD_CONNECTOR_SCHEDULER_PATCHED_FILE_SHA256=${OFFLOAD_CONNECTOR_SCHEDULER_PATCHED_FILE_SHA256}"
  --build-arg "COMPLETION_PROTOCOL_PATCHED_FILE_SHA256=${COMPLETION_PROTOCOL_PATCHED_FILE_SHA256}"
  --build-arg "GENERATE_API_ROUTER_PATCHED_FILE_SHA256=${GENERATE_API_ROUTER_PATCHED_FILE_SHA256}"
  --build-arg "CLI_ARGS_PATCHED_FILE_SHA256=${CLI_ARGS_PATCHED_FILE_SHA256}"
  --build-arg "TITOTO_PROTOCOL_PATCHED_FILE_SHA256=${TITOTO_PROTOCOL_PATCHED_FILE_SHA256}"
  --build-arg "TITOTO_SERVING_PATCHED_FILE_SHA256=${TITOTO_SERVING_PATCHED_FILE_SHA256}"
  --build-arg "ABSTRACT_PARSER_PATCHED_FILE_SHA256=${ABSTRACT_PARSER_PATCHED_FILE_SHA256}"
  --build-arg "PARSER_ADAPTERS_PATCHED_FILE_SHA256=${PARSER_ADAPTERS_PATCHED_FILE_SHA256}"
  --build-arg "PARSER_EVENTS_PATCHED_FILE_SHA256=${PARSER_EVENTS_PATCHED_FILE_SHA256}"
  --build-arg "PARSER_ENGINE_CONFIG_PATCHED_FILE_SHA256=${PARSER_ENGINE_CONFIG_PATCHED_FILE_SHA256}"
  --build-arg "STREAMING_PARSER_ENGINE_PATCHED_FILE_SHA256=${STREAMING_PARSER_ENGINE_PATCHED_FILE_SHA256}"
  --build-arg "TOKEN_ID_SCANNER_PATCHED_FILE_SHA256=${TOKEN_ID_SCANNER_PATCHED_FILE_SHA256}"
  --build-arg "ENGINE_PROTOCOL_PATCHED_FILE_SHA256=${ENGINE_PROTOCOL_PATCHED_FILE_SHA256}"
  --build-arg "CPU_MODEL_RUNNER_PATCHED_FILE_SHA256=${CPU_MODEL_RUNNER_PATCHED_FILE_SHA256}"
  --build-arg "CPU_MODEL_RUNNER_UPSTREAM_FILE_SHA256=${CPU_MODEL_RUNNER_UPSTREAM_FILE_SHA256}"
  --build-arg "CPU_TRITON_UTILS_PATCHED_FILE_SHA256=${CPU_TRITON_UTILS_PATCHED_FILE_SHA256}"
  --build-arg "CPU_TRITON_UTILS_UPSTREAM_FILE_SHA256=${CPU_TRITON_UTILS_UPSTREAM_FILE_SHA256}"
  --build-arg "SEED_OSS_ENGINE_TOOL_PARSER_PATCHED_FILE_SHA256=${SEED_OSS_ENGINE_TOOL_PARSER_PATCHED_FILE_SHA256}"
  --build-arg "SEED_OSS_ENGINE_TOOL_PARSER_UPSTREAM_FILE_SHA256=${SEED_OSS_ENGINE_TOOL_PARSER_UPSTREAM_FILE_SHA256}"
  --build-arg "SEED_OSS_PARSER_PATCHED_FILE_SHA256=${SEED_OSS_PARSER_PATCHED_FILE_SHA256}"
  --build-arg "SEED_OSS_PARSER_UPSTREAM_FILE_SHA256=${SEED_OSS_PARSER_UPSTREAM_FILE_SHA256}"
  --build-arg "MISTRAL_TOOL_PARSER_PATCHED_FILE_SHA256=${MISTRAL_TOOL_PARSER_PATCHED_FILE_SHA256}"
  --build-arg "MISTRAL_TOOL_PARSER_UPSTREAM_FILE_SHA256=${MISTRAL_TOOL_PARSER_UPSTREAM_FILE_SHA256}"
  --build-arg "KIMI_K2_TOOL_PARSER_PATCHED_FILE_SHA256=${KIMI_K2_TOOL_PARSER_PATCHED_FILE_SHA256}"
  --build-arg "KIMI_K2_TOOL_PARSER_UPSTREAM_FILE_SHA256=${KIMI_K2_TOOL_PARSER_UPSTREAM_FILE_SHA256}"
  --build-arg "POOLING_BASE_PROTOCOL_PATCHED_FILE_SHA256=${POOLING_BASE_PROTOCOL_PATCHED_FILE_SHA256}"
  --build-arg "POOLING_BASE_PROTOCOL_UPSTREAM_FILE_SHA256=${POOLING_BASE_PROTOCOL_UPSTREAM_FILE_SHA256}"
  --build-arg "ENGINE_CORE_PATCHED_FILE_SHA256=${ENGINE_CORE_PATCHED_FILE_SHA256}"
  --build-arg "ENGINE_CORE_UPSTREAM_FILE_SHA256=${ENGINE_CORE_UPSTREAM_FILE_SHA256}"
  --build-arg "QWEN3_ENGINE_TOOL_PARSER_PATCHED_FILE_SHA256=${QWEN3_ENGINE_TOOL_PARSER_PATCHED_FILE_SHA256}"
  --build-arg "QWEN3_ENGINE_TOOL_PARSER_UPSTREAM_FILE_SHA256=${QWEN3_ENGINE_TOOL_PARSER_UPSTREAM_FILE_SHA256}"
  --build-arg "PARSER_MANAGER_PATCHED_FILE_SHA256=${PARSER_MANAGER_PATCHED_FILE_SHA256}"
  --build-arg "PARSER_MANAGER_UPSTREAM_FILE_SHA256=${PARSER_MANAGER_UPSTREAM_FILE_SHA256}"
  --build-arg "POOLSIDE_V1_TOOL_PARSER_PATCHED_FILE_SHA256=${POOLSIDE_V1_TOOL_PARSER_PATCHED_FILE_SHA256}"
  --build-arg "POOLSIDE_V1_TOOL_PARSER_UPSTREAM_FILE_SHA256=${POOLSIDE_V1_TOOL_PARSER_UPSTREAM_FILE_SHA256}"
  --build-arg "MINIMAX_M3_REASONING_PARSER_PATCHED_FILE_SHA256=${MINIMAX_M3_REASONING_PARSER_PATCHED_FILE_SHA256}"
  --build-arg "MINIMAX_M3_REASONING_PARSER_UPSTREAM_FILE_SHA256=${MINIMAX_M3_REASONING_PARSER_UPSTREAM_FILE_SHA256}"
  --build-arg "BASIC_REASONING_PARSERS_PATCHED_FILE_SHA256=${BASIC_REASONING_PARSERS_PATCHED_FILE_SHA256}"
  --build-arg "BASIC_REASONING_PARSERS_UPSTREAM_FILE_SHA256=${BASIC_REASONING_PARSERS_UPSTREAM_FILE_SHA256}"
  --build-arg "KV_TIERING_P2P_MANAGER_PATCHED_FILE_SHA256=${KV_TIERING_P2P_MANAGER_PATCHED_FILE_SHA256}"
  --build-arg "KV_TIERING_P2P_MANAGER_UPSTREAM_FILE_SHA256=${KV_TIERING_P2P_MANAGER_UPSTREAM_FILE_SHA256}"
  --build-arg "KV_TIERING_BASE_PATCHED_FILE_SHA256=${KV_TIERING_BASE_PATCHED_FILE_SHA256}"
  --build-arg "KV_TIERING_BASE_UPSTREAM_FILE_SHA256=${KV_TIERING_BASE_UPSTREAM_FILE_SHA256}"
  --build-arg "HIDDEN_STATES_CONNECTOR_PATCHED_FILE_SHA256=${HIDDEN_STATES_CONNECTOR_PATCHED_FILE_SHA256}"
  --build-arg "HIDDEN_STATES_CONNECTOR_UPSTREAM_FILE_SHA256=${HIDDEN_STATES_CONNECTOR_UPSTREAM_FILE_SHA256}"
  --build-arg "LMCACHE_MP_CONNECTOR_PATCHED_FILE_SHA256=${LMCACHE_MP_CONNECTOR_PATCHED_FILE_SHA256}"
  --build-arg "LMCACHE_MP_CONNECTOR_UPSTREAM_FILE_SHA256=${LMCACHE_MP_CONNECTOR_UPSTREAM_FILE_SHA256}"
  --build-arg "LMCACHE_CONNECTOR_PATCHED_FILE_SHA256=${LMCACHE_CONNECTOR_PATCHED_FILE_SHA256}"
  --build-arg "LMCACHE_CONNECTOR_UPSTREAM_FILE_SHA256=${LMCACHE_CONNECTOR_UPSTREAM_FILE_SHA256}"
  --build-arg "MORIIO_CONNECTOR_PATCHED_FILE_SHA256=${MORIIO_CONNECTOR_PATCHED_FILE_SHA256}"
  --build-arg "MORIIO_CONNECTOR_UPSTREAM_FILE_SHA256=${MORIIO_CONNECTOR_UPSTREAM_FILE_SHA256}"
  --build-arg "MOONCAKE_CONNECTOR_PATCHED_FILE_SHA256=${MOONCAKE_CONNECTOR_PATCHED_FILE_SHA256}"
  --build-arg "MOONCAKE_CONNECTOR_UPSTREAM_FILE_SHA256=${MOONCAKE_CONNECTOR_UPSTREAM_FILE_SHA256}"
  --build-arg "NIXL_CONNECTOR_PATCHED_FILE_SHA256=${NIXL_CONNECTOR_PATCHED_FILE_SHA256}"
  --build-arg "NIXL_CONNECTOR_UPSTREAM_FILE_SHA256=${NIXL_CONNECTOR_UPSTREAM_FILE_SHA256}"
  --build-arg "OFFLOAD_CONNECTOR_PATCHED_FILE_SHA256=${OFFLOAD_CONNECTOR_PATCHED_FILE_SHA256}"
  --build-arg "OFFLOAD_CONNECTOR_UPSTREAM_FILE_SHA256=${OFFLOAD_CONNECTOR_UPSTREAM_FILE_SHA256}"
  --build-arg "MULTI_CONNECTOR_PATCHED_FILE_SHA256=${MULTI_CONNECTOR_PATCHED_FILE_SHA256}"
  --build-arg "MULTI_CONNECTOR_UPSTREAM_FILE_SHA256=${MULTI_CONNECTOR_UPSTREAM_FILE_SHA256}"
  --build-arg "KV_CONNECTOR_BASE_PATCHED_FILE_SHA256=${KV_CONNECTOR_BASE_PATCHED_FILE_SHA256}"
  --build-arg "KV_CONNECTOR_BASE_UPSTREAM_FILE_SHA256=${KV_CONNECTOR_BASE_UPSTREAM_FILE_SHA256}"
  --build-arg "KV_CONNECTOR_FACTORY_PATCHED_FILE_SHA256=${KV_CONNECTOR_FACTORY_PATCHED_FILE_SHA256}"
  --build-arg "KV_CONNECTOR_FACTORY_UPSTREAM_FILE_SHA256=${KV_CONNECTOR_FACTORY_UPSTREAM_FILE_SHA256}"
  --build-arg "NEMOTRON_V3_PARSER_PATCHED_FILE_SHA256=${NEMOTRON_V3_PARSER_PATCHED_FILE_SHA256}"
  --build-arg "NEMOTRON_V3_PARSER_UPSTREAM_FILE_SHA256=${NEMOTRON_V3_PARSER_UPSTREAM_FILE_SHA256}"
  --build-arg "GENERATE_INVOCATION_TYPES_PATCHED_FILE_SHA256=${GENERATE_INVOCATION_TYPES_PATCHED_FILE_SHA256}"
  --build-arg "GENERATE_INVOCATION_TYPES_UPSTREAM_FILE_SHA256=${GENERATE_INVOCATION_TYPES_UPSTREAM_FILE_SHA256}"
  --build-arg "GENERATE_BASE_SERVING_PATCHED_FILE_SHA256=${GENERATE_BASE_SERVING_PATCHED_FILE_SHA256}"
  --build-arg "GENERATE_BASE_SERVING_UPSTREAM_FILE_SHA256=${GENERATE_BASE_SERVING_UPSTREAM_FILE_SHA256}"
  --build-arg "ENGINE_CORE_REQUEST_PATCHED_FILE_SHA256=${ENGINE_CORE_REQUEST_PATCHED_FILE_SHA256}"
  --build-arg "ENGINE_CORE_REQUEST_UPSTREAM_FILE_SHA256=${ENGINE_CORE_REQUEST_UPSTREAM_FILE_SHA256}"
  --build-arg "SOURCE_DATE_EPOCH=${SOURCE_DATE_EPOCH}"
)
readonly -a image_build_options
readonly IMAGE_OUTPUT_OPTIONS="type=docker,rewrite-timestamp=true"
IMAGE_INPUTS_MANIFEST="${BUILD_EXPORT_DIR}/image-inputs"
readonly IMAGE_INPUTS_MANIFEST
{
  printf 'base-image %s\n' "${EXPECTED_BASE_IMAGE_ID}"
  printf 'option %s\n' "${image_build_options[@]}"
  printf 'output %s\n' "${IMAGE_OUTPUT_OPTIONS}"
  (
    cd "${BUILD_CONTEXT}"
    find . -mindepth 1 -printf 'entry %y %m %T@ %P\n' | LC_ALL=C sort -k5
    find . -type f -printf '%P\0' | LC_ALL=C sort -z | xargs -0 sha256sum -- |
      sed 's/^/file /'
  )
} >"${IMAGE_INPUTS_MANIFEST}"
image_inputs_sha256="$(sha256sum <"${IMAGE_INPUTS_MANIFEST}")"
image_inputs_sha256="${image_inputs_sha256%% *}"
readonly image_inputs_sha256
context_file_count="$(find "${BUILD_CONTEXT}" -type f | wc -l)"
readonly context_file_count

# The lock pins every file of this repository the image carries; each pin must
# describe the copy in the context. The reviewed vLLM files need no line here:
# scripts/runtime_image_unit.py proves every pin the Dockerfile names equals
# the stage data's final identity, and the transaction proved the
# reconstruction holds exactly those.
printf '%s  %s\n' \
  "${AGENT_CHAT_TEMPLATE_SHA256}" "${BUILD_CONTEXT}/chat_template.jinja" \
  "${PHASE_BUDGET_UNIT_SHA256}" "${BUILD_CONTEXT}/scripts/phase_budget_unit.py" \
  "${VISION_WORKSPACE_UNIT_SHA256}" "${BUILD_CONTEXT}/scripts/vision_workspace_unit.py" \
  "${VISION_CONTRACT_UNIT_SHA256}" "${BUILD_CONTEXT}/scripts/vision_contract_unit.py" \
  "${VISION_MLP_UNIT_SHA256}" "${BUILD_CONTEXT}/scripts/vision_mlp_unit.py" \
  "${SHARED_PREFIX_CACHE_UNIT_SHA256}" "${BUILD_CONTEXT}/scripts/shared_prefix_cache_unit.py" \
  "${RAW_MEDIA_UNIT_SHA256}" "${BUILD_CONTEXT}/scripts/raw_media_unit.py" \
  "${GENERATE_RESULT_UNIT_SHA256}" "${BUILD_CONTEXT}/scripts/generate_result_unit.py" \
  "${TURBOQUANT_GUARD_UNIT_SHA256}" "${BUILD_CONTEXT}/scripts/turboquant_guard_unit.py" \
  "${TURBOQUANT_K8V4_UNIT_SHA256}" "${BUILD_CONTEXT}/scripts/turboquant_k8v4_unit.py" \
  "${QWEN38_CONTEXT_UNIT_SHA256}" "${BUILD_CONTEXT}/scripts/qwen38_context_unit.py" \
  "${CHAT_TEMPLATE_RETENTION_UNIT_SHA256}" "${BUILD_CONTEXT}/scripts/chat_template_retention_unit.py" \
  "${NVFP4_KERNEL_UNIT_SHA256}" "${BUILD_CONTEXT}/scripts/nvfp4_kernel_unit.py" \
  "${TOOL_OUTPUT_PARSER_UNIT_SHA256}" "${BUILD_CONTEXT}/scripts/tool_output_parser_unit.py" \
  "${RESPONSES_STREAM_RULE_SHA256}" "${BUILD_CONTEXT}/scripts/responses_stream_rule.py" \
  "${REASONING_USAGE_UNIT_SHA256}" "${BUILD_CONTEXT}/scripts/reasoning_usage_unit.py" \
  "${QWEN_GRAMMAR_UNIT_SHA256}" "${BUILD_CONTEXT}/scripts/qwen_grammar_unit.py" \
  "${TEMPLATE_AUTHORSHIP_UNIT_SHA256}" "${BUILD_CONTEXT}/scripts/template_authorship_unit.py" \
  "${NATIVE_FP4_SELECTION_UNIT_SHA256}" "${BUILD_CONTEXT}/scripts/native_fp4_selection_unit.py" \
  "${RUNTIME_DOCKERFILE_SHA256}" "${BUILD_CONTEXT}/containers/Dockerfile.runtime" | \
  sha256sum --check --strict

# The units below run the context's own copies: the bytes they prove are the
# bytes the image carries.
docker run --rm --network none --read-only \
  --user "$(id -u):$(id -g)" \
  --tmpfs /tmp:rw,nodev,nosuid,size=128m \
  --env PYTHONPYCACHEPREFIX=/tmp/pycache \
  --env CUDA_VISIBLE_DEVICES= --env TRITON_INTERPRET=1 \
  --volume "${BUILD_CONTEXT}:/context:ro" \
  --volume "${BUILD_CONTEXT}/vllm/vllm/v1/attention/ops/triton_turboquant_store.py:/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/ops/triton_turboquant_store.py:ro" \
  --volume "${BUILD_CONTEXT}/vllm/vllm/v1/attention/ops/triton_turboquant_decode.py:/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/ops/triton_turboquant_decode.py:ro" \
  --entrypoint python3 "${BASE_IMAGE_TAG}" /context/scripts/turboquant_guard_unit.py

# Execute CPU contract units against the complete reviewed runtime overlay.
# They parse as serving does: with the parsers and template arguments the
# launch selects, on the served model's tokenizer and generation files, each
# checked against the model manifest before it is mounted.
served_model_files=(tokenizer.json tokenizer_config.json vocab.json config.json generation_config.json)
printf '%s  %s\n' "${MODEL_MANIFEST_SHA256}" "${MODEL_MANIFEST}" | \
  sha256sum --check --strict >/dev/null
served_model_mounts=()
for served_file in "${served_model_files[@]}"; do
  served_line="$(grep -E "^[0-9a-f]{64}  ${served_file//./\\.}\$" "${MODEL_MANIFEST}" || true)"
  [[ -n "${served_line}" && "$(wc -l <<<"${served_line}")" == 1 ]] || \
    die "The model manifest must pin ${served_file} exactly once." \
      "Manifest: ${MODEL_MANIFEST}"
  (cd -- "${MODEL_DIR}" && sha256sum --check --strict --quiet <<<"${served_line}") || \
    die "The served model's ${served_file} differs from its manifest." \
      "Directory: ${MODEL_DIR}" "Restore the pinned model files before checking."
  served_model_mounts+=(--volume "${MODEL_DIR}/${served_file}:/served-model/${served_file}:ro")
done
served_parser_env=(
  --env SERVED_MODEL=/served-model
  --env "SERVED_REASONING_PARSER=$(launch_arg_value --reasoning-parser)"
  --env "SERVED_TOOL_CALL_PARSER=$(launch_arg_value --tool-call-parser)"
  --env "SERVED_CHAT_TEMPLATE_KWARGS=$(launch_arg_value --default-chat-template-kwargs)"
)
parser_unit_mounts=()
while IFS= read -r status_line; do
  case "${status_line}" in
    " M vllm/"*|"?? vllm/"*)
      path="${status_line:3}"
      parser_unit_mounts+=(--volume "${BUILD_CONTEXT}/vllm/${path}:/usr/local/lib/python3.12/dist-packages/${path}:ro")
      ;;
  esac
done <<<"${REVIEWED_STATUS}"
for unit in chat_template_retention_unit tool_output_parser_unit vision_contract_unit vision_workspace_unit reasoning_usage_unit shared_prefix_cache_unit phase_budget_unit generate_result_unit raw_media_unit qwen_grammar_unit template_authorship_unit native_fp4_selection_unit qwen38_context_unit; do
  docker run --rm --network none --read-only --user "$(id -u):$(id -g)" \
    --tmpfs /tmp:rw,nodev,nosuid,size=256m \
    --env PYTHONDONTWRITEBYTECODE=1 --env CUDA_VISIBLE_DEVICES= "${served_parser_env[@]}" \
    --volume "${BUILD_CONTEXT}/chat_template.jinja:/opt/qwen38/chat_template.jinja:ro" --volume "${BUILD_CONTEXT}:/context:ro" "${parser_unit_mounts[@]}" "${served_model_mounts[@]}" \
    --entrypoint python3 "${BASE_IMAGE_TAG}" "/context/scripts/${unit}.py"
done

# The probes run only against the live backend, and the image does not carry
# them, so this is where a probe that cannot build the requests it sends fails:
# every probe builds each kind of request offline, through the builders its
# live run sends through, and the API server's own app, built from the
# reviewed runtime, validates each body with its route's request model.
docker run --rm --network none --read-only --user "$(id -u):$(id -g)" \
  --tmpfs /tmp:rw,nodev,nosuid,size=256m \
  --env PYTHONDONTWRITEBYTECODE=1 --env CUDA_VISIBLE_DEVICES= \
  --volume "${PROJECT_DIR}/scripts:/probes:ro" "${parser_unit_mounts[@]}" \
  --entrypoint python3 "${BASE_IMAGE_TAG}" /probes/probe_requests_unit.py

# The reviewed tests: every file under tests/ the patch set changes or adds, read
# from REVIEWED_STATUS, never listed by hand -- a stage that adds a test adds it
# here. The runner is pinned like every other input: the wheels in test-runner/
# (pytest and the plugins vLLM's tests use, at the versions vLLM's
# requirements/test/cuda.txt pins at VLLM_COMMIT) are deployment inputs, verified
# above, and are unpacked inside the container from exactly the entries the
# deployment-input manifest names. Where a test runs is the test's own
# declaration: it carries the gpu or network mark the tree's tests/conftest.py
# registers, on itself, its class or its module, when it needs what a machine may
# lack. scripts/reviewed_tests_plugin.py collects these files and nothing else,
# runs the selected set and reports, on lines prefixed REVIEWED_TESTS, what it did
# and did not execute.
reviewed_test_files=()
reviewed_test_support_files=()
reviewed_test_scripts=()
deleted_runtime_files=()
while IFS= read -r status_line; do
  status_path="${status_line:3}"
  case "${status_line}" in
    " M tests/"*|"?? tests/"*)
      case "${status_path##*/}" in
        test_*.py) reviewed_test_files+=("${status_path}") ;;
        *.py) reviewed_test_support_files+=("${status_path}") ;;
        *) reviewed_test_scripts+=("${status_path}") ;;
      esac
      ;;
    " D vllm/"*) deleted_runtime_files+=("${status_path}") ;;
  esac
done <<<"${REVIEWED_STATUS}"
readonly -a reviewed_test_files reviewed_test_support_files reviewed_test_scripts \
  deleted_runtime_files
mapfile -t test_runner_wheels < <(
  sed -n 's#^[0-9a-f]\{64\}  test-runner/\([^/]*\.whl\)$#/test-runner/\1#p' \
    "${DEPLOYMENT_INPUT_MANIFEST}"
)
((${#test_runner_wheels[@]} > 0)) || \
  die "The deployment-input manifest pins no test-runner wheel."
readonly -a test_runner_wheels

# offline: the reviewed tests that declare no need, on the base image with
# exactly the context's reviewed runtime files overlaid -- the bytes the image
# carries, as the units above run them -- with no GPU and no network.
# release: every reviewed test, in the image the build made, on this host's GPU,
# with the network; a gated Hugging Face repository is read with the HF_TOKEN of
# the environment build runs in, and what the tests download is kept under this
# run's export directory and removed with it.
# collection: every reviewed test file, imported and its tests built as the
# release set collects them, and none run, on the offline machine. A release run
# that meets a collection error runs nothing, so build could pin no image; this
# is where such an error is found. The plugin names what only build can collect:
# a module that skips itself at import without a GPU, and one that declares a
# GPU and cannot be imported without one -- whose imports and declaration must
# still run here.
# Both read the served model's tokenizer files, checked against the model manifest
# above, where a test asks for the pinned local tokenizer (QWEN_TOKENIZER_PATH).
# The tree's own vllm/ is masked, so a test imports the installed package, and a
# runtime module the patch set deletes cannot be imported, as in the image.
# check runs the offline set in this many shards at once: pytest processes of
# about 1 GB each, each given every twelfth reviewed file. Measured on a 24-CPU
# host, one process takes 16.5 minutes, six take 5.8 and twelve 4.0; the set's
# result does not depend on the count.
readonly OFFLINE_REVIEWED_TEST_SHARDS=12
# Every reviewed test runs under this timeout, setup and teardown included
# (pytest-timeout, pinned with the runner), so a test that hangs ends its run
# naming itself instead of holding check or build forever. It is a declared
# policy, not a measurement: the longest offline test measured 34 seconds, and
# a release test's first use of a server pays for a model download and for
# upstream's own wait of up to 480 seconds for that server to come up, so half
# an hour is past anything a test legitimately does and ends a hung build
# within the hour.
readonly REVIEWED_TEST_TIMEOUT_SECONDS=1800
run_reviewed_tests() {
  local reviewed_set="$1" image="$2" shards="$3" image_note="" shard failed=false
  local log_dir="${BUILD_EXPORT_DIR}/reviewed-tests-${reviewed_set}"
  local -a machine pids=() logs=() verbosity=(-q)
  case "${reviewed_set}" in
    offline)
      machine=(--network none --env CUDA_VISIBLE_DEVICES= --env HF_HUB_OFFLINE=1
        "${parser_unit_mounts[@]}")
      ;;
    collection)
      machine=(--network none --env CUDA_VISIBLE_DEVICES= --env HF_HUB_OFFLINE=1
        "${parser_unit_mounts[@]}")
      # Collected, not run: pytest names each file's count, not each test.
      verbosity=(-qq)
      ;;
    release)
      image_note="; ${image} was not pinned"
      install -d -m 0700 "${BUILD_EXPORT_DIR}/reviewed-tests-cache"
      machine=(--gpus all --network bridge --env HF_TOKEN
        --volume "${BUILD_EXPORT_DIR}/reviewed-tests-cache:/reviewed-tests-cache"
        --env HF_HOME=/reviewed-tests-cache/huggingface
        --env VLLM_CACHE_ROOT=/reviewed-tests-cache/vllm
        --env XDG_CACHE_HOME=/reviewed-tests-cache/xdg)
      ;;
  esac
  install -d -m 0700 "${log_dir}"
  reviewed_tests_shard() {
    docker run --rm --read-only --user "$(id -u):$(id -g)" \
      --cap-drop ALL --security-opt no-new-privileges:true \
      --tmpfs /tmp:rw,nodev,nosuid,size=4g \
      --env HOME=/tmp --env PYTHONDONTWRITEBYTECODE=1 \
      --env "REVIEWED_TEST_FILES=${reviewed_test_files[*]}" \
      --env "REVIEWED_TEST_SUPPORT_FILES=${reviewed_test_support_files[*]}" \
      --env "REVIEWED_DELETED_RUNTIME_FILES=${deleted_runtime_files[*]}" \
      --volume "${RECONSTRUCTION}:/reviewed:ro" --tmpfs /reviewed/vllm:ro,size=4k \
      --volume "${PROJECT_DIR}/test-runner:/test-runner:ro" \
      --volume "${PROJECT_DIR}/scripts/reviewed_tests_plugin.py:/test-runner-plugin/reviewed_tests_plugin.py:ro" \
      "${served_model_mounts[@]}" --env QWEN_TOKENIZER_PATH=/served-model \
      "${machine[@]}" --workdir /reviewed --entrypoint bash "${image}" -c '
        set -euo pipefail
        reviewed_set="$1" shard="$2" timeout="$3" verbosity="$4"; shift 4
        for wheel in "$@"; do python3 -m zipfile -e "${wheel}" /tmp/test-runner; done
        PYTHONPATH=/tmp/test-runner:/test-runner-plugin exec python3 -m pytest \
          -p no:cacheprovider -p reviewed_tests_plugin --reviewed-set "${reviewed_set}" \
          --reviewed-shard "${shard}" --timeout "${timeout}" "${verbosity}" -rfE \
          --tb=short --disable-warnings tests' \
      run-reviewed-tests "${reviewed_set}" "$1" "${REVIEWED_TEST_TIMEOUT_SECONDS}" \
      "${verbosity[@]}" "${test_runner_wheels[@]}"
  }
  if ((shards == 1)); then
    logs=("${log_dir}/shard-1.log")
    # A collection names each file's count; the merge below says what they add up to.
    reviewed_tests_shard 1/1 2>&1 | tee "${logs[0]}" |
      sed -e '/^REVIEWED_TESTS_JSON /d' -e '/^tests\/[^ ]*\.py: [0-9][0-9]*$/d' ||
      failed=true
  else
    # The shards run at once, each its own pytest process in its own container,
    # and are reported in order once all have ended.
    for ((shard = 1; shard <= shards; shard++)); do
      logs+=("${log_dir}/shard-${shard}.log")
      reviewed_tests_shard "${shard}/${shards}" >"${logs[-1]}" 2>&1 &
      pids+=("$!")
    done
    for shard in "${!pids[@]}"; do
      wait "${pids[shard]}" || failed=true
      printf 'Reviewed tests, %s set, shard %d of %d:\n' \
        "${reviewed_set}" "$((shard + 1))" "${shards}"
      sed -e '/^REVIEWED_TESTS_JSON /d' -e 's/^/  /' "${logs[shard]}"
    done
  fi
  docker run --rm --network none --read-only --user "$(id -u):$(id -g)" \
    --volume "${log_dir}:/reviewed-tests:ro" \
    --volume "${PROJECT_DIR}/scripts/reviewed_tests_plugin.py:/test-runner-plugin/reviewed_tests_plugin.py:ro" \
    --entrypoint python3 "${BASE_IMAGE_TAG}" \
    /test-runner-plugin/reviewed_tests_plugin.py merge "${logs[@]/#"${log_dir}"//reviewed-tests}" \
    >"${log_dir}/merged.log" || failed=true
  sed '/^REVIEWED_TESTS_SUMMARY /d' "${log_dir}/merged.log"
  [[ "${failed}" == false ]] || \
    die "The reviewed tests failed (the ${reviewed_set} set)${image_note}." \
      "Their output is above; the lines prefixed REVIEWED_TESTS say what ran and what did not." \
      "Next: fix what failed. A test that needs a GPU or the network this set does not give it says so with the gpu or network mark."
  reviewed_tests_summary="$(sed -n 's/^REVIEWED_TESTS_SUMMARY //p' "${log_dir}/merged.log")"
}
# What no gate runs, said beside what the runner ran: the shell scripts under
# tests/ the patch set changes, and the test files it leaves as upstream has them.
upstream_test_file_count="$(
  find "${RECONSTRUCTION}/tests" -type f -name 'test_*.py' | wc -l
)"
upstream_test_file_count=$((upstream_test_file_count - ${#reviewed_test_files[@]}))
readonly upstream_test_file_count
if [[ "${MODE}" == "serve-check" ]]; then
  reviewed_tests_summary="serve-check does not run the reviewed tests; check runs every one that needs neither a GPU nor the network, and build runs them all before it pins an image."
else
  run_reviewed_tests collection "${BASE_IMAGE_TAG}" 1
  reviewed_tests_collection="${reviewed_tests_summary}"
  run_reviewed_tests offline "${BASE_IMAGE_TAG}" "${OFFLINE_REVIEWED_TEST_SHARDS}"
  reviewed_tests_summary="${reviewed_tests_collection}"$'\n'"${reviewed_tests_summary}"
fi

if [[ "${MODE}" == "check" || "${MODE}" == "serve-check" ]]; then
  remove_reconstruction
  # Every count below is derived from the objects this run just verified —
  # REVIEWED_STATUS and the deployment-input manifest — never restated by
  # hand: a hand count here is one more copy that can drift from the thing
  # it describes.
  modified_runtime_count="$(grep -c '^ M vllm/' <<<"${REVIEWED_STATUS}" || :)"
  new_runtime_count="$(grep -c '^?? vllm/' <<<"${REVIEWED_STATUS}" || :)"
  deleted_runtime_count="$(grep -c '^ D vllm/' <<<"${REVIEWED_STATUS}" || :)"
  modified_test_count="$(grep -c '^ M tests/' <<<"${REVIEWED_STATUS}" || :)"
  new_test_count="$(grep -c '^?? tests/' <<<"${REVIEWED_STATUS}" || :)"
  deleted_test_count="$(grep -c '^ D tests/' <<<"${REVIEWED_STATUS}" || :)"
  modified_doc_count="$(grep -c '^ M docs/' <<<"${REVIEWED_STATUS}" || :)"
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
    "units, the test runner, and all build units are exact."
  echo "${reviewed_tests_summary}"
  echo "Not executed by any gate: the ${#reviewed_test_scripts[@]} reviewed" \
    "shell scripts under tests/ (${reviewed_test_scripts[*]}), which start prefill" \
    "and decode servers that exchange KV through NIXL or the P2P tier; the" \
    "${upstream_test_file_count} test files the patch set leaves as upstream has them;" \
    "and every reviewed test whose own condition holds on no machine this deployment" \
    "has -- an optional test (neither set passes --optional), a ROCm-only test, one" \
    "that needs more than the one GPU build requires, or one that imports a package" \
    "the image does not carry -- which build skips as check does and names among its skips."
  if [[ "${MODE}" == "serve-check" ]]; then
    require_image_built_from_inputs "${image_inputs_sha256}" "${context_file_count}"
  fi
  if [[ "${image_inputs_sha256}" == "${IMAGE_BUILD_INPUTS_SHA256}" ]]; then
    echo "Image inputs ${image_inputs_sha256} (${context_file_count} context files):" \
      "the pinned image ${EXPECTED_IMAGE_ID} was built from exactly these, and a build" \
      "must reproduce it."
  else
    echo "Image inputs ${image_inputs_sha256} (${context_file_count} context files):" \
      "the pinned image ${EXPECTED_IMAGE_ID} was not built from these; a build writes" \
      "the ID they produce into config/runtime-v1.sh."
  fi
  exit 0
fi

printf 'Building from image inputs %s:\n' "${image_inputs_sha256}"
sed 's/^/  /' "${IMAGE_INPUTS_MANIFEST}"
docker buildx build --progress=plain \
  "${image_build_options[@]}" \
  --output "${IMAGE_OUTPUT_OPTIONS},dest=${RUNTIME_ARCHIVE},name=${BUILD_LOAD_TAG}" \
  --file "${BUILD_CONTEXT}/containers/Dockerfile.runtime" \
  "${BUILD_CONTEXT}"
docker load --input "${RUNTIME_ARCHIVE}"

actual_image_id="$(docker image inspect --format '{{.Id}}' "${BUILD_LOAD_TAG}")"
# The build takes the identity tag of its own ID before anything is checked, so
# whatever the checks find, it keeps a name that says which image it is -- and
# IMAGE_TAG, which the checks never touch, still names the pinned image. An
# identity tag that already names another image is refused, never moved: it
# carries that image's own ID, so that can only have been done by hand.
actual_identity_tag="${IMAGE_IDENTITY_TAG_PREFIX}${actual_image_id#sha256:}"
identity_tag_id="$(docker image inspect --format '{{.Id}}' "${actual_identity_tag}" 2>/dev/null || true)"
if [[ -n "${identity_tag_id}" && "${identity_tag_id}" != "${actual_image_id}" ]]; then
  echo "The identity tag ${actual_identity_tag} already names ${identity_tag_id}; it was not moved." >&2
  exit 1
fi
docker tag "${actual_image_id}" "${actual_identity_tag}"
docker image rm "${BUILD_LOAD_TAG}" >/dev/null
actual_installed_report="$(
  docker run --rm --network none --entrypoint sha256sum "${actual_image_id}" \
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
    /usr/local/lib/python3.12/dist-packages/vllm/renderers/template_authorship.py \
    /usr/local/lib/python3.12/dist-packages/vllm/renderers/online_renderer.py \
    /usr/local/lib/python3.12/dist-packages/vllm/entrypoints/pooling/scoring/io_processor.py \
    /usr/local/lib/python3.12/dist-packages/vllm/model_executor/kernels/linear/__init__.py \
    /usr/local/lib/python3.12/dist-packages/vllm/entrypoints/serve/tokenize/protocol.py \
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
    /usr/local/lib/python3.12/dist-packages/vllm/v1/worker/cpu_model_runner.py \
    /usr/local/lib/python3.12/dist-packages/vllm/utils/cpu_triton_utils.py \
    /usr/local/lib/python3.12/dist-packages/vllm/tool_parsers/seed_oss_engine_tool_parser.py \
    /usr/local/lib/python3.12/dist-packages/vllm/parser/seed_oss.py \
    /usr/local/lib/python3.12/dist-packages/vllm/tool_parsers/mistral_tool_parser.py \
    /usr/local/lib/python3.12/dist-packages/vllm/tool_parsers/kimi_k2_tool_parser.py \
    /usr/local/lib/python3.12/dist-packages/vllm/entrypoints/pooling/base/protocol.py \
    /usr/local/lib/python3.12/dist-packages/vllm/v1/engine/core.py \
    /usr/local/lib/python3.12/dist-packages/vllm/tool_parsers/qwen3_engine_tool_parser.py \
    /usr/local/lib/python3.12/dist-packages/vllm/parser/parser_manager.py \
    /usr/local/lib/python3.12/dist-packages/vllm/tool_parsers/poolside_v1_tool_parser.py \
    /usr/local/lib/python3.12/dist-packages/vllm/reasoning/minimax_m3_reasoning_parser.py \
    /usr/local/lib/python3.12/dist-packages/vllm/reasoning/basic_parsers.py \
    /usr/local/lib/python3.12/dist-packages/vllm/v1/kv_offload/tiering/p2p/manager.py \
    /usr/local/lib/python3.12/dist-packages/vllm/v1/kv_offload/tiering/base.py \
    /usr/local/lib/python3.12/dist-packages/vllm/distributed/kv_transfer/kv_connector/v1/example_hidden_states_connector.py \
    /usr/local/lib/python3.12/dist-packages/vllm/distributed/kv_transfer/kv_connector/v1/lmcache_mp_connector.py \
    /usr/local/lib/python3.12/dist-packages/vllm/distributed/kv_transfer/kv_connector/v1/lmcache_connector.py \
    /usr/local/lib/python3.12/dist-packages/vllm/distributed/kv_transfer/kv_connector/v1/moriio/moriio_connector.py \
    /usr/local/lib/python3.12/dist-packages/vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_connector.py \
    /usr/local/lib/python3.12/dist-packages/vllm/distributed/kv_transfer/kv_connector/v1/nixl/connector.py \
    /usr/local/lib/python3.12/dist-packages/vllm/distributed/kv_transfer/kv_connector/v1/offloading_connector.py \
    /usr/local/lib/python3.12/dist-packages/vllm/distributed/kv_transfer/kv_connector/v1/multi_connector.py \
    /usr/local/lib/python3.12/dist-packages/vllm/distributed/kv_transfer/kv_connector/v1/base.py \
    /usr/local/lib/python3.12/dist-packages/vllm/distributed/kv_transfer/kv_connector/factory.py \
    /usr/local/lib/python3.12/dist-packages/vllm/parser/nemotron_v3.py \
    /usr/local/lib/python3.12/dist-packages/vllm/entrypoints/generate/factories.py \
    /usr/local/lib/python3.12/dist-packages/vllm/entrypoints/generate/base/serving.py \
    /usr/local/lib/python3.12/dist-packages/vllm/v1/engine/__init__.py \
    /usr/local/lib/python3.12/dist-packages/vllm/parser/gemma4.py \
    /usr/local/lib/python3.12/dist-packages/vllm/parser/glm47_moe.py \
    /usr/local/lib/python3.12/dist-packages/vllm/parser/minimax_m2.py \
    /usr/local/lib/python3.12/dist-packages/vllm/parser/mistral.py \
    /usr/local/lib/python3.12/dist-packages/vllm/parser/harmony.py \
    /usr/local/lib/python3.12/dist-packages/vllm/parser/deepseek_v32.py \
    /usr/local/lib/python3.12/dist-packages/vllm/parser/deepseek_v4.py \
    /usr/local/lib/python3.12/dist-packages/vllm/parser/inkling.py \
    /usr/local/lib/python3.12/dist-packages/vllm/parser/kimi_k2.py \
    /usr/local/lib/python3.12/dist-packages/vllm/parser/kimi_k3.py \
    /usr/local/lib/python3.12/dist-packages/vllm/entrypoints/scale_out/derender/serving.py \
    /usr/local/lib/python3.12/dist-packages/vllm/multimodal/processing/inputs.py \
    /usr/local/lib/python3.12/dist-packages/vllm/multimodal/processing/processor.py \
    /usr/local/lib/python3.12/dist-packages/vllm/renderers/base.py \
    /usr/local/lib/python3.12/dist-packages/vllm/renderers/online_derenderer.py \
    /usr/local/lib/python3.12/dist-packages/vllm/entrypoints/openai/completion/serving.py \
    /usr/local/lib/python3.12/dist-packages/vllm/entrypoints/openai/chat_completion/api_router.py \
    /usr/local/lib/python3.12/dist-packages/vllm/entrypoints/openai/chat_completion/batch_serving.py \
    /usr/local/lib/python3.12/dist-packages/vllm/entrypoints/openai/completion/api_router.py \
    /usr/local/lib/python3.12/dist-packages/vllm/entrypoints/scale_out/token_in_token_out/api_router.py \
    /usr/local/lib/python3.12/dist-packages/vllm/entrypoints/openai/run_batch.py \
    /usr/local/lib/python3.12/dist-packages/vllm/engine/protocol.py \
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
  "${TEMPLATE_AUTHORSHIP_PATCHED_FILE_SHA256}" /usr/local/lib/python3.12/dist-packages/vllm/renderers/template_authorship.py \
  "${ONLINE_RENDERER_PATCHED_FILE_SHA256}" /usr/local/lib/python3.12/dist-packages/vllm/renderers/online_renderer.py \
  "${SCORING_IO_PROCESSOR_PATCHED_FILE_SHA256}" /usr/local/lib/python3.12/dist-packages/vllm/entrypoints/pooling/scoring/io_processor.py \
  "${LINEAR_KERNELS_PATCHED_FILE_SHA256}" /usr/local/lib/python3.12/dist-packages/vllm/model_executor/kernels/linear/__init__.py \
  "${TOKENIZE_PROTOCOL_PATCHED_FILE_SHA256}" /usr/local/lib/python3.12/dist-packages/vllm/entrypoints/serve/tokenize/protocol.py \
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
  "${CPU_MODEL_RUNNER_PATCHED_FILE_SHA256}" /usr/local/lib/python3.12/dist-packages/vllm/v1/worker/cpu_model_runner.py \
  "${CPU_TRITON_UTILS_PATCHED_FILE_SHA256}" /usr/local/lib/python3.12/dist-packages/vllm/utils/cpu_triton_utils.py \
  "${SEED_OSS_ENGINE_TOOL_PARSER_PATCHED_FILE_SHA256}" /usr/local/lib/python3.12/dist-packages/vllm/tool_parsers/seed_oss_engine_tool_parser.py \
  "${SEED_OSS_PARSER_PATCHED_FILE_SHA256}" /usr/local/lib/python3.12/dist-packages/vllm/parser/seed_oss.py \
  "${MISTRAL_TOOL_PARSER_PATCHED_FILE_SHA256}" /usr/local/lib/python3.12/dist-packages/vllm/tool_parsers/mistral_tool_parser.py \
  "${KIMI_K2_TOOL_PARSER_PATCHED_FILE_SHA256}" /usr/local/lib/python3.12/dist-packages/vllm/tool_parsers/kimi_k2_tool_parser.py \
  "${POOLING_BASE_PROTOCOL_PATCHED_FILE_SHA256}" /usr/local/lib/python3.12/dist-packages/vllm/entrypoints/pooling/base/protocol.py \
  "${ENGINE_CORE_PATCHED_FILE_SHA256}" /usr/local/lib/python3.12/dist-packages/vllm/v1/engine/core.py \
  "${QWEN3_ENGINE_TOOL_PARSER_PATCHED_FILE_SHA256}" /usr/local/lib/python3.12/dist-packages/vllm/tool_parsers/qwen3_engine_tool_parser.py \
  "${PARSER_MANAGER_PATCHED_FILE_SHA256}" /usr/local/lib/python3.12/dist-packages/vllm/parser/parser_manager.py \
  "${POOLSIDE_V1_TOOL_PARSER_PATCHED_FILE_SHA256}" /usr/local/lib/python3.12/dist-packages/vllm/tool_parsers/poolside_v1_tool_parser.py \
  "${MINIMAX_M3_REASONING_PARSER_PATCHED_FILE_SHA256}" /usr/local/lib/python3.12/dist-packages/vllm/reasoning/minimax_m3_reasoning_parser.py \
  "${BASIC_REASONING_PARSERS_PATCHED_FILE_SHA256}" /usr/local/lib/python3.12/dist-packages/vllm/reasoning/basic_parsers.py \
  "${KV_TIERING_P2P_MANAGER_PATCHED_FILE_SHA256}" /usr/local/lib/python3.12/dist-packages/vllm/v1/kv_offload/tiering/p2p/manager.py \
  "${KV_TIERING_BASE_PATCHED_FILE_SHA256}" /usr/local/lib/python3.12/dist-packages/vllm/v1/kv_offload/tiering/base.py \
  "${HIDDEN_STATES_CONNECTOR_PATCHED_FILE_SHA256}" /usr/local/lib/python3.12/dist-packages/vllm/distributed/kv_transfer/kv_connector/v1/example_hidden_states_connector.py \
  "${LMCACHE_MP_CONNECTOR_PATCHED_FILE_SHA256}" /usr/local/lib/python3.12/dist-packages/vllm/distributed/kv_transfer/kv_connector/v1/lmcache_mp_connector.py \
  "${LMCACHE_CONNECTOR_PATCHED_FILE_SHA256}" /usr/local/lib/python3.12/dist-packages/vllm/distributed/kv_transfer/kv_connector/v1/lmcache_connector.py \
  "${MORIIO_CONNECTOR_PATCHED_FILE_SHA256}" /usr/local/lib/python3.12/dist-packages/vllm/distributed/kv_transfer/kv_connector/v1/moriio/moriio_connector.py \
  "${MOONCAKE_CONNECTOR_PATCHED_FILE_SHA256}" /usr/local/lib/python3.12/dist-packages/vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_connector.py \
  "${NIXL_CONNECTOR_PATCHED_FILE_SHA256}" /usr/local/lib/python3.12/dist-packages/vllm/distributed/kv_transfer/kv_connector/v1/nixl/connector.py \
  "${OFFLOAD_CONNECTOR_PATCHED_FILE_SHA256}" /usr/local/lib/python3.12/dist-packages/vllm/distributed/kv_transfer/kv_connector/v1/offloading_connector.py \
  "${MULTI_CONNECTOR_PATCHED_FILE_SHA256}" /usr/local/lib/python3.12/dist-packages/vllm/distributed/kv_transfer/kv_connector/v1/multi_connector.py \
  "${KV_CONNECTOR_BASE_PATCHED_FILE_SHA256}" /usr/local/lib/python3.12/dist-packages/vllm/distributed/kv_transfer/kv_connector/v1/base.py \
  "${KV_CONNECTOR_FACTORY_PATCHED_FILE_SHA256}" /usr/local/lib/python3.12/dist-packages/vllm/distributed/kv_transfer/kv_connector/factory.py \
  "${NEMOTRON_V3_PARSER_PATCHED_FILE_SHA256}" /usr/local/lib/python3.12/dist-packages/vllm/parser/nemotron_v3.py \
  "${GENERATE_INVOCATION_TYPES_PATCHED_FILE_SHA256}" /usr/local/lib/python3.12/dist-packages/vllm/entrypoints/generate/factories.py \
  "${GENERATE_BASE_SERVING_PATCHED_FILE_SHA256}" /usr/local/lib/python3.12/dist-packages/vllm/entrypoints/generate/base/serving.py \
  "${ENGINE_CORE_REQUEST_PATCHED_FILE_SHA256}" /usr/local/lib/python3.12/dist-packages/vllm/v1/engine/__init__.py \
  "${GEMMA4_PARSER_PATCHED_FILE_SHA256}" /usr/local/lib/python3.12/dist-packages/vllm/parser/gemma4.py \
  "${GLM47_PARSER_PATCHED_FILE_SHA256}" /usr/local/lib/python3.12/dist-packages/vllm/parser/glm47_moe.py \
  "${MINIMAX_M2_PARSER_PATCHED_FILE_SHA256}" /usr/local/lib/python3.12/dist-packages/vllm/parser/minimax_m2.py \
  "${MISTRAL_PARSER_PATCHED_FILE_SHA256}" /usr/local/lib/python3.12/dist-packages/vllm/parser/mistral.py \
  "${HARMONY_PARSER_PATCHED_FILE_SHA256}" /usr/local/lib/python3.12/dist-packages/vllm/parser/harmony.py \
  "${DEEPSEEK_V32_PARSER_PATCHED_FILE_SHA256}" /usr/local/lib/python3.12/dist-packages/vllm/parser/deepseek_v32.py \
  "${DEEPSEEK_V4_PARSER_PATCHED_FILE_SHA256}" /usr/local/lib/python3.12/dist-packages/vllm/parser/deepseek_v4.py \
  "${INKLING_PARSER_PATCHED_FILE_SHA256}" /usr/local/lib/python3.12/dist-packages/vllm/parser/inkling.py \
  "${KIMI_K2_PARSER_PATCHED_FILE_SHA256}" /usr/local/lib/python3.12/dist-packages/vllm/parser/kimi_k2.py \
  "${KIMI_K3_PARSER_PATCHED_FILE_SHA256}" /usr/local/lib/python3.12/dist-packages/vllm/parser/kimi_k3.py \
  "${DERENDER_SERVING_PATCHED_FILE_SHA256}" /usr/local/lib/python3.12/dist-packages/vllm/entrypoints/scale_out/derender/serving.py \
  "${MM_PROCESSOR_INPUTS_PATCHED_FILE_SHA256}" /usr/local/lib/python3.12/dist-packages/vllm/multimodal/processing/inputs.py \
  "${MM_PROCESSOR_PATCHED_FILE_SHA256}" /usr/local/lib/python3.12/dist-packages/vllm/multimodal/processing/processor.py \
  "${BASE_RENDERER_PATCHED_FILE_SHA256}" /usr/local/lib/python3.12/dist-packages/vllm/renderers/base.py \
  "${ONLINE_DERENDERER_PATCHED_FILE_SHA256}" /usr/local/lib/python3.12/dist-packages/vllm/renderers/online_derenderer.py \
  "${COMPLETION_SERVING_PATCHED_FILE_SHA256}" /usr/local/lib/python3.12/dist-packages/vllm/entrypoints/openai/completion/serving.py \
  "${CHAT_API_ROUTER_PATCHED_FILE_SHA256}" /usr/local/lib/python3.12/dist-packages/vllm/entrypoints/openai/chat_completion/api_router.py \
  "${CHAT_BATCH_SERVING_PATCHED_FILE_SHA256}" /usr/local/lib/python3.12/dist-packages/vllm/entrypoints/openai/chat_completion/batch_serving.py \
  "${COMPLETION_API_ROUTER_PATCHED_FILE_SHA256}" /usr/local/lib/python3.12/dist-packages/vllm/entrypoints/openai/completion/api_router.py \
  "${TITOTO_API_ROUTER_PATCHED_FILE_SHA256}" /usr/local/lib/python3.12/dist-packages/vllm/entrypoints/scale_out/token_in_token_out/api_router.py \
  "${RUN_BATCH_PATCHED_FILE_SHA256}" /usr/local/lib/python3.12/dist-packages/vllm/entrypoints/openai/run_batch.py \
  "${ENGINE_CLIENT_PROTOCOL_PATCHED_FILE_SHA256}" /usr/local/lib/python3.12/dist-packages/vllm/engine/protocol.py \
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
  docker run --rm --network none --entrypoint sha256sum "${actual_image_id}" \
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
    /opt/qwen38/template_authorship_unit.py \
    /opt/qwen38/native_fp4_selection_unit.py \
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
  "${TEMPLATE_AUTHORSHIP_UNIT_SHA256}" /opt/qwen38/template_authorship_unit.py \
  "${NATIVE_FP4_SELECTION_UNIT_SHA256}" /opt/qwen38/native_fp4_selection_unit.py \
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
  docker run --rm --network none --entrypoint sha256sum "${actual_image_id}" \
    /usr/local/lib/python3.12/dist-packages/vllm/v1/core/block_pool.py \
    /usr/local/lib/python3.12/dist-packages/vllm/v1/core/single_type_kv_cache_manager.py \
    /usr/local/lib/python3.12/dist-packages/vllm/v1/kv_offload/cpu/common.py \
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
  "${BLOCK_POOL_PATCHED_FILE_SHA256}" /usr/local/lib/python3.12/dist-packages/vllm/v1/core/block_pool.py \
  "${SINGLE_TYPE_KV_CACHE_MANAGER_PATCHED_FILE_SHA256}" /usr/local/lib/python3.12/dist-packages/vllm/v1/core/single_type_kv_cache_manager.py \
  "${KV_OFFLOAD_CPU_COMMON_PATCHED_FILE_SHA256}" /usr/local/lib/python3.12/dist-packages/vllm/v1/kv_offload/cpu/common.py \
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
  docker run --rm --network none --entrypoint sha256sum "${actual_image_id}" \
    /usr/local/lib/python3.12/dist-packages/vllm/parser/abstract_parser.py \
    /usr/local/lib/python3.12/dist-packages/vllm/parser/engine/adapters.py \
    /usr/local/lib/python3.12/dist-packages/vllm/parser/engine/events.py \
    /usr/local/lib/python3.12/dist-packages/vllm/parser/engine/parser_engine_config.py \
    /usr/local/lib/python3.12/dist-packages/vllm/parser/engine/streaming_parser_engine.py \
    /usr/local/lib/python3.12/dist-packages/vllm/parser/engine/token_id_scanner.py \
    /usr/local/lib/python3.12/dist-packages/vllm/entrypoints/openai/engine/protocol.py \
    /opt/qwen38/tool_output_parser_unit.py \
    /opt/qwen38/responses_stream_rule.py \
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
  "${RESPONSES_STREAM_RULE_SHA256}" /opt/qwen38/responses_stream_rule.py \
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
  if ! docker run --rm --network none --entrypoint test "${actual_image_id}" '!' -e "${obsolete}"; then
    printf 'Built image still contains superseded runtime code: %s\n' "${obsolete}" >&2
    exit 1
  fi
done

actual_profile_label="$(
  docker image inspect --format '{{index .Config.Labels "qwen38.runtime.profile"}}' \
    "${actual_image_id}"
)"
if [[ "${actual_profile_label}" != "${IMAGE_PROFILE_VERSION}" ]]; then
  echo "Built image carries the wrong runtime profile label." >&2
  echo "Expected: ${IMAGE_PROFILE_VERSION}" >&2
  echo "Found:    ${actual_profile_label:-missing}" >&2
  exit 1
fi

# The units that need the GPU run in this image, on this host's GPU, before
# anything pins it or moves IMAGE_TAG to it.
run_gpu_release_units "${actual_image_id}"
# Every reviewed test, in this image, on this host's GPU and with the network,
# beside the units above and on the same terms: the image is pinned only once
# they pass.
run_reviewed_tests release "${actual_image_id}" 1
echo "${reviewed_tests_summary}"
remove_reconstruction

# The build is the only witness of the image it made, so it writes the pin
# itself. A build of the inputs the pin was made from must reproduce it and is
# refused otherwise; a build of other inputs writes its own ID and inputs.
settle_produced_identity pin_outcome "${RUNTIME_LOCK}" \
  EXPECTED_IMAGE_ID IMAGE_BUILD_INPUTS_SHA256 \
  "${actual_image_id}" "${image_inputs_sha256}" \
  "Reproducible build ID mismatch: this build's inputs are the ones the pinned image was built from, and they built a different image." \
  "The inputs are listed at the start of this build: base image ${EXPECTED_BASE_IMAGE_ID}, every build option, and ${context_file_count} context files." \
  "The build is kept as ${actual_identity_tag}; ${IMAGE_TAG} was not moved." \
  "Next: on any host but the one that built the pin, restore the pinned image with ./scripts/restore-images.sh -- images reproduce per host, not across hosts. On that host the builder now makes a different image of the same inputs: find out why before releasing, and release a different image only as a new release (IMAGE_PROFILE_VERSION, IMAGE_TAG and IMAGE_ARCHIVE_NAME), whose inputs differ."

# The build is the pinned image now, and this is the one place IMAGE_TAG is
# moved.
docker tag "${actual_image_id}" "${IMAGE_TAG}"

echo "Built ${IMAGE_TAG} with no build-time network access."
if [[ "${pin_outcome}" == written ]]; then
  "${PROJECT_DIR}/scripts/generate-deployment-input-manifest.sh"
  echo "Wrote the pin into config/runtime-v1.sh: EXPECTED_IMAGE_ID=${actual_image_id}," \
    "IMAGE_BUILD_INPUTS_SHA256=${image_inputs_sha256}; config/deployment-inputs.sha256 is regenerated."
  echo "Next: build again, which must report \"Verified reproducible image ID\";" \
    "then ./scripts/save-images.sh, which writes the archive pin, and commit both pins together."
else
  echo "Verified reproducible image ID: ${actual_image_id}"
fi
echo "Identity tag: ${actual_identity_tag}"
