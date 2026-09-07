#!/usr/bin/env bash
# Run one probe against the live backend, as one agent.
#
# Every probe is a caller of the sole deployment and runs inside the serving
# container, where the real tokenizer, the installed vLLM sources, and the
# server's private loopback live. The suite is staged whole into the
# container's bounded scratch tmpfs and the named probe is run by file, so it
# can import its siblings and can name itself: scripts/probe_scope.py mints the
# kv_scope every generative request carries from the file the interpreter was
# given and a fresh run id. The container's own shell owns the staging
# directory for its whole life, so nothing persists whatever the probe's
# status and however this launcher ends.
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=../config/runtime-v1.sh
source "${PROJECT_DIR}/config/runtime-v1.sh"

if (($# < 1)); then
  printf 'Usage: %s <name>_probe.py [probe arguments...]\n' "$0" >&2
  exit 2
fi
probe="$1"
shift
if [[ "${probe}" != *_probe.py || "${probe}" == */* || ! -f "${SCRIPT_DIR}/${probe}" ]]; then
  printf 'ERROR: %s is not a probe in %s\n' "${probe}" "${SCRIPT_DIR}" >&2
  exit 2
fi

# A probe's evidence describes the pinned deployment or nothing: the container
# it runs in must be the locked name running the locked image.
running_image="$(docker inspect --format '{{.Image}}' "${CONTAINER_NAME}" 2>/dev/null || true)"
if [[ "${running_image}" != "${EXPECTED_IMAGE_ID}" ]]; then
  printf 'ERROR: %s is not running the pinned runtime image.\n' "${CONTAINER_NAME}" >&2
  printf 'Expected: %s\nFound:    %s\n' "${EXPECTED_IMAGE_ID}" "${running_image:-nothing}" >&2
  exit 1
fi

probes=("${SCRIPT_DIR}"/*_probe.py)
inside="$(cat <<'INNER'
set -eu
stage="$(mktemp -d /tmp/probe.XXXXXX)"
trap 'rm -rf -- "${stage}"' EXIT
tar --extract --file - --directory "${stage}"
probe="$1"
shift
python3 -u "${stage}/${probe}" "$@"
INNER
)"
tar --create --file - --directory "${SCRIPT_DIR}" probe_scope.py "${probes[@]##*/}" |
  docker exec --interactive "${CONTAINER_NAME}" sh -c "${inside}" probe "${probe}" "$@"
