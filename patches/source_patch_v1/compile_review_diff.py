#!/usr/bin/env python3
"""Compile reviewed unified diffs into redundant Python landmark data.

This is a maintenance tool, not the production application mechanism.  Its
output is reviewed and committed.  Production patchers independently verify
that the committed Python blocks and the hashed review diffs remain identical.
"""

from __future__ import annotations

import argparse
import pprint
import re
from dataclasses import dataclass
from pathlib import Path

from framework import PatchRefusedError, _parse_review_diff, sha256_bytes, sha256_text


@dataclass(frozen=True)
class StageArgument:
    name: str
    review_path: str


def parse_stage(raw: str) -> StageArgument:
    parts = raw.split("=", 1)
    if len(parts) != 2 or not all(parts):
        raise argparse.ArgumentTypeError("stage must be NAME=REVIEW_PATCH")
    return StageArgument(parts[0], parts[1])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--identity", action="append", default=[])
    parser.add_argument("--stage", action="append", type=parse_stage, required=True)
    return parser.parse_args()


def read_text(path: Path) -> str:
    data = path.read_bytes()
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise PatchRefusedError(f"{path} is not UTF-8: {exc}") from exc
    if "\r" in text or not text.endswith("\n"):
        raise PatchRefusedError(f"{path} violates LF/terminal-newline contract")
    return text


def emit_python(value: object) -> str:
    return pprint.pformat(value, width=88, sort_dicts=False)


def line_offset(text: str, one_based_line: int) -> int:
    if one_based_line < 1:
        raise PatchRefusedError(f"invalid one-based line {one_based_line}")
    lines = text.splitlines(keepends=True)
    if one_based_line > len(lines) + 1:
        raise PatchRefusedError(
            f"line {one_based_line} exceeds source length {len(lines)}"
        )
    return sum(len(line) for line in lines[: one_based_line - 1])


def occurrence_offsets(text: str, needle: str) -> list[int]:
    if needle == "":
        return [0] if text == "" else []
    offsets: list[int] = []
    start = 0
    while True:
        found = text.find(needle, start)
        if found < 0:
            return offsets
        offsets.append(found)
        start = found + 1


def expand_to_unique_landmark(
    current: str,
    review_before: str,
    review_after: str,
    old_start: int,
) -> tuple[str, str]:
    offsets = occurrence_offsets(current, review_before)
    if len(offsets) == 1 and current.count(review_after) == 0:
        return review_before, review_after
    if review_before == "":
        raise PatchRefusedError("new-file review hunk unexpectedly has source text")

    expected_offset = line_offset(current, old_start)
    selected = min(offsets, key=lambda offset: abs(offset - expected_offset))
    if offsets.count(selected) != 1:
        raise PatchRefusedError("cannot select a unique review-hunk occurrence")

    line_starts = [0]
    for match in re.finditer("\n", current):
        line_starts.append(match.end())
    start_line = max(i for i, offset in enumerate(line_starts) if offset <= selected)
    review_end = selected + len(review_before)
    end_line = max(i for i, offset in enumerate(line_starts) if offset < review_end)
    source_lines = current.splitlines(keepends=True)

    for radius in range(1, max(len(source_lines), 2)):
        expanded_start = max(0, start_line - radius)
        expanded_end = min(len(source_lines), end_line + radius + 1)
        prefix = "".join(source_lines[expanded_start:start_line])
        suffix_start = review_end
        suffix_end = sum(len(line) for line in source_lines[:expanded_end])
        suffix = current[suffix_start:suffix_end]
        expanded_before = prefix + review_before + suffix
        expanded_after = prefix + review_after + suffix
        if current.count(expanded_before) == 1 and current.count(expanded_after) == 0:
            return expanded_before, expanded_after
    raise PatchRefusedError("could not expand an ambiguous hunk to unique landmarks")


def main() -> None:
    args = parse_args()
    source = args.source.resolve(strict=True)
    artifact_root = args.artifact_root.resolve(strict=True)
    state: dict[str, str] = {}
    exists_state: dict[str, bool] = {}

    identity_files: dict[str, str] = {}
    for relative in args.identity:
        text = read_text(source / relative)
        state[relative] = text
        exists_state[relative] = True
        identity_files[relative] = sha256_text(text)

    stages: list[dict[str, object]] = []
    all_first: dict[str, str | None] = {}
    for stage_arg in args.stage:
        review_file = artifact_root / stage_arg.review_path
        review_data = review_file.read_bytes()
        parsed = _parse_review_diff(review_data, label=stage_arg.name)
        touched = tuple(dict.fromkeys(edit.path for edit in parsed))
        files: list[dict[str, str | None]] = []
        before_by_path: dict[str, str | None] = {}
        for relative in touched:
            if relative not in state:
                candidate = source / relative
                exists_state[relative] = candidate.exists()
                state[relative] = read_text(candidate) if candidate.exists() else ""
            before = state[relative]
            before_digest = sha256_text(before) if exists_state[relative] else None
            before_by_path[relative] = before_digest
            all_first.setdefault(relative, before_digest)

        edits: list[dict[str, str]] = []
        per_path_index: dict[str, int] = {}
        for parsed_edit in parsed:
            index = per_path_index.get(parsed_edit.path, 0) + 1
            per_path_index[parsed_edit.path] = index
            current = state[parsed_edit.path]
            landmark_before, landmark_after = expand_to_unique_landmark(
                current,
                parsed_edit.before,
                parsed_edit.after,
                parsed_edit.old_start,
            )
            before_count = current.count(landmark_before)
            after_count = current.count(landmark_after)
            if before_count != 1 or after_count != 0:
                raise PatchRefusedError(
                    f"{stage_arg.name}:{parsed_edit.path}:landmark-{index}: "
                    f"before_count={before_count}, after_count={after_count}"
                )
            state[parsed_edit.path] = current.replace(
                landmark_before, landmark_after, 1
            )
            exists_state[parsed_edit.path] = True
            edits.append(
                {
                    "name": f"{parsed_edit.path}:landmark-{index}",
                    "path": parsed_edit.path,
                    "before": landmark_before,
                    "after": landmark_after,
                    "review_before": parsed_edit.before,
                    "review_after": parsed_edit.after,
                }
            )

        for relative in touched:
            files.append(
                {
                    "path": relative,
                    "before_sha256": before_by_path[relative],
                    "after_sha256": sha256_text(state[relative]),
                }
            )
        stages.append(
            {
                "name": stage_arg.name,
                "review_patch": stage_arg.review_path,
                "review_sha256": sha256_bytes(review_data),
                "files": tuple(files),
                "edits": tuple(edits),
            }
        )

    final_files = {path: sha256_text(state[path]) for path in sorted(all_first)}
    output = (
        '"""Generated redundant landmark blocks; see compile_review_diff.py."""\n\n'
        f"SOURCE_REVISION = {args.source_revision!r}\n\n"
        f"IDENTITY_FILES = {emit_python(identity_files)}\n\n"
        f"GENERATED_STAGES = {emit_python(tuple(stages))}\n\n"
        f"FINAL_FILES = {emit_python(final_files)}\n"
    )
    args.output.write_text(output, encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
