#!/usr/bin/env python3
"""Derive this deployment's chat template from the model's own template.

The served template is not the model's. It rejects `enable_thinking: false`,
accepts only `xhigh` reasoning effort, and renders every assistant turn in the
history with its reasoning, refusing any `preserve_thinking` other than `true`.
Those differences were previously carried as an edited file whose bytes were
pinned but whose derivation existed nowhere, so nothing could answer "what did
we change, and is the served template still exactly that change applied to the
model's?".

This script answers it the way `repair-offset-norms.py` answers the same
question for the checkpoint: a verified source, a sequence of named changes
anchored on landmarks that must appear exactly once, verification of the result
against a pinned hash, and an atomic publish. The model's template is the source
and is never edited. It uses only the standard library and runs inside the
pinned project image.

A stage whose landmark is missing or ambiguous is a refusal, not a no-op: it
means the source moved under a change that was written against a different
source, which is the case where silently producing something is worst.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import sys
from pathlib import Path
from typing import NoReturn


SOURCE_MANIFEST = "manifests/model-snapshot-16b6615a.sha256"
SOURCE_MANIFEST_SHA256 = (
    "6d979221939858d8f98c7e615028e1e468cffb3ff2d501f943646c1e12ef2cdc"
)
SOURCE_TEMPLATE_NAME = "chat_template.jinja"
SOURCE_TEMPLATE_SHA256 = (
    "12827f24b742ea4e80cdc12dbcf9622227056b9f797252a3149263d4f9aaadce"
)
DERIVED_TEMPLATE_SHA256 = (
    "07f545cd8ed9232f2b24d79010fad187f92e5b25b532448eb9017c0f8b8c2088"
)

# A shedding rule was built here and removed. It worked exactly as designed --
# the reasoning of the last assistant turn kept, the cut immovable by injected
# reminders, no empty think blocks -- and was still no fix: an invented rule
# renders a history this model was never trained on. Stage B below enforces the
# opposite; the README's agent defaults give the reasons.

MANIFEST_LINE = re.compile(r"^([0-9a-f]{64})  ([^\0]+)$")


def fail(message: str) -> NoReturn:
    sys.exit(f"derive-chat-template: {message}")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_manifest(path: Path) -> dict[str, str]:
    if not path.is_file() or path.is_symlink():
        fail(f"source manifest is missing or not a regular file: {path}")
    entries: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        matched = MANIFEST_LINE.match(line)
        if matched is None:
            fail(f"source manifest line is not one canonical sha256 record: {line!r}")
        digest, name = matched.groups()
        if name in entries:
            fail(f"source manifest names {name} more than once")
        entries[name] = digest
    if not entries:
        fail("source manifest records nothing")
    return entries


def verified_source(project: Path, model: Path) -> str:
    """The model's own template, proved against the pinned snapshot manifest."""
    manifest_path = project / SOURCE_MANIFEST
    observed_manifest = sha256_file(manifest_path)
    if observed_manifest != SOURCE_MANIFEST_SHA256:
        fail(
            f"{SOURCE_MANIFEST} is not the pinned manifest: expected "
            f"{SOURCE_MANIFEST_SHA256}, observed {observed_manifest}"
        )
    manifest = read_manifest(manifest_path)
    pinned = manifest.get(SOURCE_TEMPLATE_NAME)
    if pinned is None:
        fail(f"{SOURCE_MANIFEST} does not record {SOURCE_TEMPLATE_NAME}")
    if pinned != SOURCE_TEMPLATE_SHA256:
        fail(
            "the snapshot manifest and this script disagree about the source "
            f"template: manifest {pinned}, expected {SOURCE_TEMPLATE_SHA256}"
        )
    source_path = model / SOURCE_TEMPLATE_NAME
    if not source_path.is_file() or source_path.is_symlink():
        fail(f"source template is missing or not a regular file: {source_path}")
    observed = sha256_file(source_path)
    if observed != SOURCE_TEMPLATE_SHA256:
        fail(
            f"source template is not the pinned bytes: expected "
            f"{SOURCE_TEMPLATE_SHA256}, observed {observed}"
        )
    return source_path.read_text(encoding="utf-8")


def apply_stage(text: str, name: str, before: str, after: str) -> str:
    occurrences = text.count(before)
    if occurrences != 1:
        fail(
            f"stage {name!r} does not anchor: its landmark appears {occurrences} "
            "times in the source, and a change that cannot be placed exactly "
            "once is not applied"
        )
    return text.replace(before, after, 1)


# --- Stage A: the correctness-first thinking profile -------------------------
#
# This deployment serves one thinking mode. A request that asks for less is a
# request this profile cannot honour, so it is refused loudly rather than served
# a weaker rendering: `enable_thinking: false` raises, `high` and `max` resolve
# to the one supported effort, and anything else raises. These were already the
# served behaviour; what is new is that they are derived here rather than
# carried as an edited file.

STAGE_A_THINKING_GATE = (
    "thinking-cannot-be-disabled",
    "{%- if enable_thinking is undefined or enable_thinking is true %}\n",
    "{%- if enable_thinking is defined and enable_thinking is false %}\n"
    "    {{- raise_exception('Thinking cannot be disabled in this correctness-first profile.') }}\n"
    "{%- else %}\n",
)

STAGE_A_EFFORT_ALIASES = (
    "high-and-max-are-xhigh",
    "    {%- if resolved_reasoning_effort == 'high' %}\n",
    "    {%- if resolved_reasoning_effort == 'high' or resolved_reasoning_effort == 'max' %}\n",
)

STAGE_A_EFFORT_IS_XHIGH_ONLY = (
    "xhigh-is-the-only-effort",
    "    {%- if resolved_reasoning_effort not in ('xhigh', 'medium', 'low') %}\n"
    "        {{- raise_exception('Unexpected reasoning effort ' ~ reasoning_effort ~ '. Supported types are xhigh (default), medium, and low.') }}\n"
    "    {%- endif %}\n"
    "    {%- if resolved_reasoning_effort == 'xhigh' %}\n"
    "        {%- set reasoning_instructions = 'Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.' %}\n"
    "    {%- elif resolved_reasoning_effort == 'low' %}\n"
    "        {%- set reasoning_instructions = 'Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the conclusion without unnecessary elaboration.' %}\n"
    "    {%- endif %}\n",
    "    {%- if resolved_reasoning_effort != 'xhigh' %}\n"
    "        {{- raise_exception('Unexpected reasoning effort ' ~ reasoning_effort ~ '. This correctness-first profile accepts only xhigh; high and max are aliases for xhigh.') }}\n"
    "    {%- endif %}\n"
    "    {%- set reasoning_instructions = 'Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.' %}\n",
)

# With thinking never disabled, the empty-block branch of the generation prompt
# is unreachable. It is removed rather than left as a path nothing can take.
STAGE_A_GENERATION_PROMPT = (
    "generation-prompt-always-opens-thinking",
    "    {%- if enable_thinking is defined and enable_thinking is false %}\n"
    "        {{- '<think>\\n\\n</think>\\n\\n' }}\n"
    "    {%- else %}\n"
    "        {{- '<think>\\n' }}\n"
    "    {%- endif %}\n",
    "    {{- '<think>\\n' }}\n",
)


# --- Stage B: reasoning in history is not optional --------------------------
#
# The model's template renders a historical assistant turn's reasoning unless a
# request asks otherwise, and this profile does not accept that request. The
# condition is removed rather than defaulted: a guarantee that depends on a
# value being set correctly is one deletion away from being lost. With the
# clause gone there is nothing to set.
#
# The refusal is the same discipline stage A applies to `enable_thinking` and to
# reasoning effort: a request this profile cannot honour is an error, never a
# quietly different rendering. It is enforced in the served artifact, so it
# holds for every caller of this model -- including one running their own client
# against a differently launched stack -- rather than for whoever happens to
# have read our launch configuration.

STAGE_B_REFUSE_DISCARD = (
    "historical-thinking-cannot-be-discarded",
    "{%- set reasoning_instructions = '' %}\n",
    "{%- set reasoning_instructions = '' %}\n"
    "{%- if preserve_thinking is defined and preserve_thinking is not true %}\n"
    "    {{- raise_exception('Historical thinking cannot be discarded in this "
    "correctness-first profile. Every assistant turn in the conversation is "
    "rendered with its reasoning, and preserve_thinking accepts only true.') }}\n"
    "{%- endif %}\n",
)

# Unconditional: the reasoning of every historical assistant turn is rendered.
# `ns.last_query_index` no longer decides anything here.
STAGE_B_ALWAYS_RENDER_REASONING = (
    "history-always-carries-its-reasoning",
    "        {%- if preserve_thinking is undefined or preserve_thinking is true or loop.index0 > ns.last_query_index %}\n"
    "            {{- '<|im_start|>' + message.role + '\\n<think>\\n' + reasoning_content + '\\n</think>\\n\\n' + content }}\n"
    "        {%- else %}\n"
    "            {{- '<|im_start|>' + message.role + '\\n' + content }}\n"
    "        {%- endif %}\n",
    "        {{- '<|im_start|>' + message.role + '\\n<think>\\n' + reasoning_content + '\\n</think>\\n\\n' + content }}\n",
)

STAGE_B = (STAGE_B_REFUSE_DISCARD, STAGE_B_ALWAYS_RENDER_REASONING)


STAGE_A = (
    STAGE_A_THINKING_GATE,
    STAGE_A_EFFORT_ALIASES,
    STAGE_A_EFFORT_IS_XHIGH_ONLY,
    STAGE_A_GENERATION_PROMPT,
)

TRAILER = (
    "name-the-derivation",
    "{#- Unsloth fixes - developer role, merged system messages, tool calling #}",
    "{#- Project-pinned agent template: developer roles, merged system messages,"
    " xhigh-only effort, and fail-loud historical tool-call validation. #}\n",
)


def derive(source: str) -> str:
    text = source
    for name, before, after in (*STAGE_A, *STAGE_B, TRAILER):
        text = apply_stage(text, name, before, after)
    return text


def write_atomic(path: Path, content: str) -> None:
    temporary = path.with_name(path.name + ".deriving")
    descriptor = os.open(
        temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC, 0o644
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        os.unlink(temporary)
        raise
    os.replace(temporary, path)
    directory = os.open(path.parent, os.O_RDONLY | os.O_CLOEXEC)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Derive the served chat template from the model's own."
    )
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="the verified model snapshot holding the source template",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="reconstruct and compare against the published template; write nothing",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project = args.project.resolve()
    derived = derive(verified_source(project, args.model.resolve()))
    digest = sha256_text(derived)

    if DERIVED_TEMPLATE_SHA256 and digest != DERIVED_TEMPLATE_SHA256:
        fail(
            "the derivation no longer reproduces its pinned result: expected "
            f"{DERIVED_TEMPLATE_SHA256}, produced {digest}. Either a stage or "
            "the source changed; neither is adopted automatically."
        )

    published = project / SOURCE_TEMPLATE_NAME
    if args.check:
        if not published.is_file() or published.is_symlink():
            fail(f"published template is missing or not a regular file: {published}")
        observed = sha256_file(published)
        if observed != digest:
            fail(
                "the published template is not what this derivation produces: "
                f"published {observed}, derived {digest}. Regenerate it with "
                "scripts/derive-chat-template.py; it is not edited by hand."
            )
        print(f"CHAT_TEMPLATE_DERIVATION_OK sha256={digest}")
        return

    write_atomic(published, derived)
    print(f"CHAT_TEMPLATE_DERIVED sha256={digest} -> {published}")


if __name__ == "__main__":
    main()
