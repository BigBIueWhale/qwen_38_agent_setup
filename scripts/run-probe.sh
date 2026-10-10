#!/usr/bin/env bash
# Run one probe against the live backend.
#
# Every probe is a caller of the sole deployment and runs inside the serving
# container, where the real tokenizer, the installed vLLM sources, and the
# server's private loopback live. The suite is staged whole into the
# container's bounded scratch tmpfs and the named probe is run by file, so it
# can import its siblings and can name itself: scripts/probe_scope.py mints a
# new kv_scope for each conversation the probe holds from the file the
# interpreter was given, a fresh run id and a sequence number. The
# container's own shell owns the staging
# directory for its whole life, so nothing persists whatever the probe's
# status and however this launcher ends.
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=runtime-common.sh
source "${SCRIPT_DIR}/runtime-common.sh"

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
# A probe that parses locally builds the parser the launch serves
# (scripts/probe_parser.py), named here from the launch itself.
tar --create --file - --directory "${SCRIPT_DIR}" probe_scope.py probe_parser.py \
    responses_stream_rule.py \
    "${probes[@]##*/}" |
  docker exec --interactive \
    --env "SERVED_REASONING_PARSER=$(launch_arg_value --reasoning-parser)" \
    --env "SERVED_TOOL_CALL_PARSER=$(launch_arg_value --tool-call-parser)" \
    --env "SERVED_CHAT_TEMPLATE_KWARGS=$(launch_arg_value --default-chat-template-kwargs)" \
    "${CONTAINER_NAME}" sh -c "${inside}" probe "${probe}" "$@"
