#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
readonly PROJECT_DIR
readonly EXPECTED_INPUT_COUNT="71"

if (($# != 0)); then
  printf 'ERROR: no arguments are supported. Usage: ./scripts/generate-deployment-input-manifest.sh\n' >&2
  exit 2
fi

output="${PROJECT_DIR}/config/deployment-inputs.sha256"
temporary="$(mktemp "${PROJECT_DIR}/config/.deployment-inputs.sha256.XXXXXX")"
case "${temporary}" in
  "${PROJECT_DIR}"/config/.deployment-inputs.sha256.*) ;;
  *)
    printf 'ERROR: unexpected temporary manifest path: %s\n' "${temporary}" >&2
    exit 1
    ;;
esac
cleanup() {
  rm -f -- "${temporary}"
}
trap cleanup EXIT

count=0
found_generator=false
found_runtime_lock=false
found_build_verifier=false
found_runtime_validator=false
found_start=false
found_status=false
found_stop=false
while IFS= read -r -d '' path; do
  case "${path}" in
    *$'\n'*|*$'\r'*)
      printf 'ERROR: deployment input path contains a line break: %q\n' "${path}" >&2
      exit 1
      ;;
  esac
  [[ -f "${PROJECT_DIR}/${path}" && ! -L "${PROJECT_DIR}/${path}" ]] || {
    printf 'ERROR: tracked deployment input is not a regular non-symlink file: %s\n' "${path}" >&2
    exit 1
  }
  case "${path}" in
    scripts/generate-deployment-input-manifest.sh) found_generator=true ;;
    config/runtime-v1.sh) found_runtime_lock=true ;;
    scripts/build-vllm.sh) found_build_verifier=true ;;
    scripts/runtime-common.sh) found_runtime_validator=true ;;
    start.sh) found_start=true ;;
    status.sh) found_status=true ;;
    stop.sh) found_stop=true ;;
  esac
  (
    cd "${PROJECT_DIR}"
    sha256sum -- "${path}"
  ) >>"${temporary}"
  count=$((count + 1))
done < <(
  git -C "${PROJECT_DIR}" ls-files -z -- \
    .dockerignore \
    chat_template.jinja \
    config/runtime-v1.sh \
    containers/Dockerfile.runtime \
    'manifests/*.sha256' \
    'patches/*.patch' \
    'patches/source_patch_v1/*' \
    'scripts/*.py' \
    'scripts/*.sh' \
    start.sh \
    status.sh \
    stop.sh | LC_ALL=C sort -z
)

[[ "${count}" == "${EXPECTED_INPUT_COUNT}" ]] || {
  printf 'ERROR: deployment-input allowlist contains %s files; expected exactly %s\n' \
    "${count}" "${EXPECTED_INPUT_COUNT}" >&2
  exit 1
}
for required in \
  found_generator \
  found_runtime_lock \
  found_build_verifier \
  found_runtime_validator \
  found_start \
  found_status \
  found_stop; do
  [[ "${!required}" == true ]] || {
    printf 'ERROR: required deployment input was not selected: %s\n' "${required#found_}" >&2
    exit 1
  }
done

chmod 0644 "${temporary}"
(
  cd "${PROJECT_DIR}"
  sha256sum --check --strict "${temporary}"
) >/dev/null
mv -- "${temporary}" "${output}"
trap - EXIT
printf 'WROTE %s entries=%s sha256=%s\n' \
  "${output}" "${count}" "$(sha256sum -- "${output}" | awk '{print $1}')"
