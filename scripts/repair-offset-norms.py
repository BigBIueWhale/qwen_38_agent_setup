#!/usr/bin/env python3
"""Create the deployable Qwen3.8 NVFP4 snapshot with exact official norms.

The pinned Unsloth source snapshot was exported through llm-compressor's
offset-norm calibration context. Qwen3.5 stores an offset ``w`` and applies
``1 + w`` at inference, but the exporter temporarily stores ``1 + w`` in BF16
and later subtracts one. That round trip loses low bits from all 161 ordinary
Qwen RMSNorm tensors.

This script does not reinterpret or reserialize the 23 GB checkpoint. It copies
the fully verified source snapshot, replaces only those 161 same-size BF16 byte
ranges with bytes from the fully verified official checkpoint, verifies the
result against a pinned corrected manifest, and publishes it with one atomic
directory rename. It uses only the Python standard library and is intended to
run inside the pinned project image through ``repair-model.sh``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import struct
import sys
from pathlib import Path
from typing import Any


SOURCE_MANIFEST_SHA256 = (
    "6d979221939858d8f98c7e615028e1e468cffb3ff2d501f943646c1e12ef2cdc"
)
OFFICIAL_MANIFEST_SHA256 = (
    "2b50c4c6f4544fc73a891c460b0731c2d7b1824fe7c92d534ecc1193c4bad067"
)
EXPECTED_NORMS = 161
FULL_ATTENTION_LAYERS = tuple(range(3, 64, 4))
MANIFEST_LINE = re.compile(r"^([0-9a-f]{64})  ([^\0]+)$")


def fail(message: str) -> "NoReturn":
    raise AssertionError(message)


def sha256_file(path: Path, chunk_bytes: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(chunk_bytes):
            digest.update(block)
    return digest.hexdigest()


def read_manifest(path: Path, expected_sha256: str | None) -> list[tuple[str, str]]:
    if not path.is_file() or path.is_symlink():
        fail(f"Manifest is missing or is not a regular file: {path}")
    actual_manifest_sha256 = sha256_file(path)
    if expected_sha256 is not None and actual_manifest_sha256 != expected_sha256:
        fail(
            f"Manifest identity mismatch for {path}: expected {expected_sha256}, "
            f"got {actual_manifest_sha256}"
        )
    entries: list[tuple[str, str]] = []
    seen: set[str] = set()
    for line_number, raw in enumerate(path.read_text().splitlines(), start=1):
        if not raw or raw.startswith("#"):
            continue
        match = MANIFEST_LINE.fullmatch(raw)
        if match is None:
            fail(f"Malformed manifest line {path}:{line_number}: {raw!r}")
        digest, relative = match.groups()
        relative_path = Path(relative)
        if relative in seen:
            fail(f"Duplicate manifest path in {path}: {relative}")
        if relative_path.is_absolute() or ".." in relative_path.parts:
            fail(f"Unsafe manifest path in {path}: {relative}")
        seen.add(relative)
        entries.append((digest, relative))
    if not entries:
        fail(f"Manifest has no entries: {path}")
    return entries


def verify_entries(
    root: Path,
    entries: list[tuple[str, str]],
    label: str,
) -> None:
    for index, (expected, relative) in enumerate(entries, start=1):
        path = root / relative
        if not path.is_file() or path.is_symlink():
            fail(f"{label} file is missing or is not regular: {path}")
        actual = sha256_file(path)
        if actual != expected:
            fail(
                f"{label} hash mismatch for {path}: expected {expected}, got {actual}"
            )
        print(
            f"verify {label}: {index}/{len(entries)} {relative}",
            file=sys.stderr,
            flush=True,
        )


def expected_norm_names() -> set[str]:
    names = {"model.language_model.norm.weight"}
    for layer in range(64):
        prefix = f"model.language_model.layers.{layer}"
        names.add(f"{prefix}.input_layernorm.weight")
        names.add(f"{prefix}.post_attention_layernorm.weight")
    for layer in FULL_ATTENTION_LAYERS:
        prefix = f"model.language_model.layers.{layer}.self_attn"
        names.add(f"{prefix}.q_norm.weight")
        names.add(f"{prefix}.k_norm.weight")
    if len(names) != EXPECTED_NORMS:
        fail(f"Internal norm accounting error: {len(names)}")
    return names


def read_index(path: Path) -> dict[str, str]:
    data = json.loads(path.read_text())
    weight_map = data.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        fail(f"Invalid safetensors index: {path}")
    if any(not isinstance(k, str) or not isinstance(v, str) for k, v in weight_map.items()):
        fail(f"Non-string safetensors index entry: {path}")
    return weight_map


def read_safetensors_header(path: Path) -> tuple[int, dict[str, Any]]:
    with path.open("rb") as source:
        raw_length = source.read(8)
        if len(raw_length) != 8:
            fail(f"Truncated safetensors length: {path}")
        (header_length,) = struct.unpack("<Q", raw_length)
        if header_length <= 0 or header_length > path.stat().st_size - 8:
            fail(f"Invalid safetensors header length in {path}: {header_length}")
        raw_header = source.read(header_length)
        if len(raw_header) != header_length:
            fail(f"Truncated safetensors header: {path}")
    try:
        header = json.loads(raw_header)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        fail(f"Invalid safetensors JSON header in {path}: {error}")
    if not isinstance(header, dict):
        fail(f"Safetensors header is not an object: {path}")
    return 8 + header_length, header


def tensor_region(
    path: Path,
    header: dict[str, Any],
    data_base: int,
    name: str,
) -> tuple[int, int, list[int], str]:
    metadata = header.get(name)
    if not isinstance(metadata, dict):
        fail(f"Tensor absent from safetensors header {path}: {name}")
    dtype = metadata.get("dtype")
    shape = metadata.get("shape")
    offsets = metadata.get("data_offsets")
    if dtype != "BF16" or not isinstance(shape, list):
        fail(f"Unexpected norm metadata in {path}: {name}: {metadata}")
    if (
        not isinstance(offsets, list)
        or len(offsets) != 2
        or not all(isinstance(value, int) for value in offsets)
        or offsets[0] < 0
        or offsets[1] <= offsets[0]
    ):
        fail(f"Invalid tensor offsets in {path}: {name}: {offsets}")
    start = data_base + offsets[0]
    end = data_base + offsets[1]
    if end > path.stat().st_size:
        fail(f"Tensor extends beyond safetensors file {path}: {name}")
    elements = 1
    for dimension in shape:
        if not isinstance(dimension, int) or dimension <= 0:
            fail(f"Invalid tensor shape in {path}: {name}: {shape}")
        elements *= dimension
    if end - start != elements * 2:
        fail(f"BF16 tensor byte count mismatch in {path}: {name}")
    return start, end, shape, dtype


def copy_exact_range(source_fd: int, target_fd: int, source_at: int, target_at: int, size: int) -> None:
    remaining = size
    while remaining:
        request = min(remaining, 1024 * 1024)
        block = os.pread(source_fd, request, source_at)
        if len(block) != request:
            fail(f"Short read while copying norm bytes at source offset {source_at}")
        written = 0
        while written < len(block):
            count = os.pwrite(target_fd, block[written:], target_at + written)
            if count <= 0:
                fail(f"Short write while copying norm bytes at target offset {target_at}")
            written += count
        source_at += request
        target_at += request
        remaining -= request


def read_exact_range(source_fd: int, source_at: int, size: int) -> bytes:
    blocks: list[bytes] = []
    remaining = size
    while remaining:
        request = min(remaining, 1024 * 1024)
        block = os.pread(source_fd, request, source_at)
        if len(block) != request:
            fail(f"Short read at offset {source_at}: wanted {request}, got {len(block)}")
        blocks.append(block)
        source_at += request
        remaining -= request
    return b"".join(blocks)


def expected_corrected_model_sha256(project: Path, source: Path, target: Path) -> str:
    """Hash source bytes with exactly the 161 official norm ranges substituted."""

    official_index = read_index(project / "model.safetensors.index.json")
    converted_index = read_index(source / "model.safetensors.index.json")
    names = expected_norm_names()
    if any(converted_index.get(name) != "model.safetensors" for name in names):
        fail("Not all source offset norms map to model.safetensors")

    source_model = source / "model.safetensors"
    target_model = target / "model.safetensors"
    if source_model.stat().st_size != target_model.stat().st_size:
        fail("Source/corrected model file sizes differ")
    source_base, source_header = read_safetensors_header(source_model)
    target_base, target_header = read_safetensors_header(target_model)
    if source_base != target_base or source_header != target_header:
        fail("Source/corrected safetensors headers differ")

    official_files: dict[str, tuple[Path, int, dict[str, Any], int]] = {}
    replacements: list[tuple[int, int, bytes, str]] = []
    source_fd = os.open(source_model, os.O_RDONLY)
    try:
        for name in sorted(names):
            source_start, source_end, source_shape, source_dtype = tensor_region(
                source_model, source_header, source_base, name
            )
            official_filename = official_index.get(name)
            if not isinstance(official_filename, str):
                fail(f"Official index does not map norm tensor: {name}")
            if official_filename not in official_files:
                official_path = project / official_filename
                official_base, official_header = read_safetensors_header(official_path)
                official_files[official_filename] = (
                    official_path,
                    official_base,
                    official_header,
                    os.open(official_path, os.O_RDONLY),
                )
            official_path, official_base, official_header, official_fd = official_files[
                official_filename
            ]
            official_start, official_end, official_shape, official_dtype = tensor_region(
                official_path, official_header, official_base, name
            )
            if source_shape != official_shape or source_dtype != official_dtype:
                fail(f"Source/official norm metadata mismatch for {name}")
            size = source_end - source_start
            if size != official_end - official_start:
                fail(f"Source/official norm byte count mismatch for {name}")
            official_bytes = read_exact_range(official_fd, official_start, size)
            replacements.append((source_start, source_end, official_bytes, name))

        replacements.sort()
        previous_end = 0
        digest = hashlib.sha256()
        for start, end, official_bytes, name in replacements:
            if start < previous_end:
                fail(f"Overlapping norm tensor byte ranges near {name}")
            source_at = previous_end
            remaining = start - previous_end
            while remaining:
                request = min(remaining, 16 * 1024 * 1024)
                digest.update(read_exact_range(source_fd, source_at, request))
                source_at += request
                remaining -= request
            digest.update(official_bytes)
            previous_end = end
        source_at = previous_end
        remaining = source_model.stat().st_size - previous_end
        while remaining:
            request = min(remaining, 16 * 1024 * 1024)
            digest.update(read_exact_range(source_fd, source_at, request))
            source_at += request
            remaining -= request
        return digest.hexdigest()
    finally:
        os.close(source_fd)
        for _, _, _, official_fd in official_files.values():
            os.close(official_fd)


def verify_semantic_correction(
    project: Path,
    source: Path,
    target: Path,
    corrected_entries: list[tuple[str, str]],
) -> None:
    manifest_map = {relative: digest for digest, relative in corrected_entries}
    actual = manifest_map.get("model.safetensors")
    if actual is None:
        fail("Corrected manifest does not contain model.safetensors")
    expected = expected_corrected_model_sha256(project, source, target)
    if actual != expected:
        fail(
            "Corrected model contains changes outside the exact official norm "
            f"substitutions: expected {expected}, manifest has {actual}"
        )
    print(
        "verified semantic correction: only 161 official norm ranges differ "
        f"(model sha256 {actual})",
        file=sys.stderr,
        flush=True,
    )


def patch_norms(project: Path, target: Path) -> None:
    official_index = read_index(project / "model.safetensors.index.json")
    converted_index = read_index(target / "model.safetensors.index.json")
    names = expected_norm_names()
    if any(converted_index.get(name) != "model.safetensors" for name in names):
        fail("Not all source offset norms map to model.safetensors")

    target_model = target / "model.safetensors"
    target_base, target_header = read_safetensors_header(target_model)
    official_files: dict[str, tuple[Path, int, dict[str, Any], int]] = {}
    target_fd = os.open(target_model, os.O_RDWR)
    try:
        for index, name in enumerate(sorted(names), start=1):
            official_filename = official_index.get(name)
            if not isinstance(official_filename, str):
                fail(f"Official index does not map norm tensor: {name}")
            if official_filename not in official_files:
                official_path = project / official_filename
                official_base, official_header = read_safetensors_header(official_path)
                official_fd = os.open(official_path, os.O_RDONLY)
                official_files[official_filename] = (
                    official_path,
                    official_base,
                    official_header,
                    official_fd,
                )
            official_path, official_base, official_header, official_fd = official_files[
                official_filename
            ]
            source_start, source_end, source_shape, source_dtype = tensor_region(
                official_path, official_header, official_base, name
            )
            target_start, target_end, target_shape, target_dtype = tensor_region(
                target_model, target_header, target_base, name
            )
            if source_shape != target_shape or source_dtype != target_dtype:
                fail(
                    f"Official/converted norm metadata mismatch for {name}: "
                    f"{source_dtype}{source_shape} != {target_dtype}{target_shape}"
                )
            size = source_end - source_start
            if size != target_end - target_start:
                fail(f"Official/converted norm byte count mismatch for {name}")
            copy_exact_range(official_fd, target_fd, source_start, target_start, size)
            print(
                f"restore official norm: {index}/{len(names)} {name}",
                file=sys.stderr,
                flush=True,
            )
        os.fsync(target_fd)
    finally:
        os.close(target_fd)
        for _, _, _, source_fd in official_files.values():
            os.close(source_fd)


def manifest_text(entries: list[tuple[str, str]], root: Path) -> str:
    lines = [
        "# Corrected deployment snapshot: pinned Unsloth source with the 161",
        "# Qwen3.5 offset-RMSNorm tensors restored byte-for-byte from official BF16.",
    ]
    for _, relative in entries:
        lines.append(f"{sha256_file(root / relative)}  {relative}")
    return "\n".join(lines) + "\n"


def write_atomic(path: Path, content: str) -> None:
    temporary = path.with_name(f".{path.name}.part.{os.getpid()}")
    if temporary.exists():
        fail(f"Unexpected manifest temporary path already exists: {temporary}")
    try:
        with temporary.open("x", encoding="utf-8") as output:
            output.write(content)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, default=Path("/project"))
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("/project/Qwen3.8-27B-NVFP4-Unsloth"),
    )
    parser.add_argument(
        "--target",
        type=Path,
        default=Path("/project/Qwen3.8-27B-NVFP4-Corrected"),
    )
    parser.add_argument(
        "--source-manifest",
        type=Path,
        default=Path("/project/manifests/model-snapshot-16b6615a.sha256"),
    )
    parser.add_argument(
        "--official-manifest",
        type=Path,
        default=Path("/project/manifests/official-qwen38-27b-1d4bf0f2.sha256"),
    )
    parser.add_argument(
        "--corrected-manifest",
        type=Path,
        default=Path("/project/manifests/model-corrected-16b6615a-norms-1d4bf0f2.sha256"),
    )
    parser.add_argument(
        "--bootstrap-manifest",
        action="store_true",
        help="write the first corrected manifest; forbidden once it exists",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project = args.project.resolve()
    source = args.source.resolve()
    target = args.target.resolve()
    corrected_manifest = args.corrected_manifest.resolve()
    if source.parent != project or target.parent != project:
        fail("Source and target must be direct children of the project directory")
    if source == target or target.name != "Qwen3.8-27B-NVFP4-Corrected":
        fail(f"Refusing unexpected correction target: {target}")

    source_entries = read_manifest(args.source_manifest, SOURCE_MANIFEST_SHA256)
    official_entries = read_manifest(args.official_manifest, OFFICIAL_MANIFEST_SHA256)
    verify_entries(source, source_entries, "pinned Unsloth source")
    verify_entries(project, official_entries, "pinned official BF16")

    if corrected_manifest.exists():
        if args.bootstrap_manifest:
            fail(f"Corrected manifest already exists; bootstrap is forbidden: {corrected_manifest}")
        corrected_entries = read_manifest(corrected_manifest, None)
        if [relative for _, relative in corrected_entries] != [
            relative for _, relative in source_entries
        ]:
            fail("Corrected/source manifest path sets or ordering differ")
        if target.exists():
            verify_entries(target, corrected_entries, "corrected deployment snapshot")
            verify_semantic_correction(project, source, target, corrected_entries)
            print(f"Corrected deployment snapshot already verified: {target}")
            return
    elif not args.bootstrap_manifest:
        fail(
            f"Pinned corrected manifest is missing: {corrected_manifest}. "
            "Bootstrap is only permitted when intentionally creating its first revision."
        )

    if target.exists():
        fail(f"Correction target exists but was not verified; refusing overwrite: {target}")
    temporary = target.with_name(f".{target.name}.part.{os.getpid()}")
    if temporary.exists():
        fail(f"Unexpected correction temporary path already exists: {temporary}")

    try:
        temporary.mkdir(mode=0o755)
        for index, (_, relative) in enumerate(source_entries, start=1):
            source_path = source / relative
            target_path = temporary / relative
            target_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, target_path, follow_symlinks=False)
            print(
                f"copy source snapshot: {index}/{len(source_entries)} {relative}",
                file=sys.stderr,
                flush=True,
            )
        patch_norms(project, temporary)
        generated = manifest_text(source_entries, temporary)
        if corrected_manifest.exists():
            expected = corrected_manifest.read_text()
            if generated != expected:
                fail("Generated corrected snapshot does not match its pinned manifest")
        else:
            write_atomic(corrected_manifest, generated)
        corrected_entries = read_manifest(corrected_manifest, None)
        verify_entries(temporary, corrected_entries, "new corrected deployment snapshot")
        verify_semantic_correction(project, source, temporary, corrected_entries)
        os.replace(temporary, target)
        directory_fd = os.open(project, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise

    print(f"Created and verified corrected deployment snapshot: {target}")


if __name__ == "__main__":
    main()
