#!/usr/bin/env python3
"""Audit every Qwen3.8 NVFP4/FP8 tensor against the official BF16 release.

This is deliberately a whole-checkpoint audit rather than a tensor sample. It:

* verifies both immutable snapshot manifests before opening any tensor;
* accounts for every tensor in the official and converted checkpoints;
* compares every reference-precision tensor, distinguishing byte identity from
  the exact lossy BF16 offset-norm calibration round trip in the source export;
* dequantizes every FP8 projection using its per-output-channel scale;
* dequantizes every packed NVFP4 projection from first principles using the
  low-nibble-first E2M1 layout and the checkpoint's FP8 block/global scales;
* validates every otherwise-unused static K/V calibration scalar; and
* emits machine-readable, element-weighted error statistics.

Run this only inside the pinned project image. It needs torch and safetensors,
but it does not need a GPU and never imports publisher conversion code.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open


OFFICIAL_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
CONVERTED_REVISION = "16b6615af3548b88e2d8e382457bc705b00479cf"
EXPECTED_OFFICIAL_TENSORS = 1199
EXPECTED_CONVERTED_TENSORS = 1968
EXPECTED_REFERENCE_PRECISION_TENSORS = 798
EXPECTED_FP8_WEIGHTS = 233
EXPECTED_NVFP4_WEIGHTS = 168
EXPECTED_KV_SCALE_TENSORS = 32
EXPECTED_OFFSET_NORM_TENSORS = 161
FULL_ATTENTION_LAYERS = tuple(range(3, 64, 4))
GROUP_SIZE = 16
FP8_MAX = 448.0
NVFP4_MAX = 6.0

MANIFEST_LINE = re.compile(r"^([0-9a-f]{64})  ([^\0]+)$")
E2M1 = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
     -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)


def expected_offset_norm_names() -> set[str]:
    names = {"model.language_model.norm.weight"}
    for layer in range(64):
        prefix = f"model.language_model.layers.{layer}"
        names.add(f"{prefix}.input_layernorm.weight")
        names.add(f"{prefix}.post_attention_layernorm.weight")
    for layer in FULL_ATTENTION_LAYERS:
        prefix = f"model.language_model.layers.{layer}.self_attn"
        names.add(f"{prefix}.q_norm.weight")
        names.add(f"{prefix}.k_norm.weight")
    if len(names) != EXPECTED_OFFSET_NORM_TENSORS:
        fail(f"Internal offset-norm accounting error: {len(names)}")
    return names


def fail(message: str) -> "NoReturn":
    raise AssertionError(message)


def sha256_file(path: Path, chunk_bytes: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(chunk_bytes):
            digest.update(block)
    return digest.hexdigest()


def verify_manifest(manifest: Path, base: Path) -> dict[str, Any]:
    entries: list[tuple[str, str]] = []
    seen: set[str] = set()
    for line_number, raw in enumerate(manifest.read_text().splitlines(), start=1):
        if not raw or raw.startswith("#"):
            continue
        match = MANIFEST_LINE.fullmatch(raw)
        if match is None:
            fail(f"Malformed manifest line {manifest}:{line_number}: {raw!r}")
        expected, relative = match.groups()
        if relative in seen:
            fail(f"Duplicate manifest path in {manifest}: {relative}")
        if Path(relative).is_absolute() or ".." in Path(relative).parts:
            fail(f"Unsafe manifest path in {manifest}: {relative}")
        seen.add(relative)
        entries.append((expected, relative))

    if not entries:
        fail(f"Manifest has no entries: {manifest}")

    total_bytes = 0
    for index, (expected, relative) in enumerate(entries, start=1):
        path = base / relative
        if not path.is_file():
            fail(f"Manifest file is missing: {path}")
        actual = sha256_file(path)
        if actual != expected:
            fail(f"Manifest hash mismatch for {path}: expected {expected}, got {actual}")
        total_bytes += path.stat().st_size
        print(
            f"manifest {manifest.name}: {index}/{len(entries)} {relative}",
            file=sys.stderr,
            flush=True,
        )

    return {
        "manifest": str(manifest),
        "entries": len(entries),
        "bytes": total_bytes,
        "sha256": sha256_file(manifest),
    }


class TensorStore:
    def __init__(
        self,
        root: Path,
        index_path: Path,
        stack: contextlib.ExitStack,
    ) -> None:
        index = json.loads(index_path.read_text())
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            fail(f"Invalid safetensors index: {index_path}")
        if any(not isinstance(k, str) or not isinstance(v, str) for k, v in weight_map.items()):
            fail(f"Non-string safetensors mapping in {index_path}")

        self.root = root
        self.weight_map: dict[str, str] = weight_map
        self.handles: dict[str, Any] = {}
        for filename in sorted(set(weight_map.values())):
            path = root / filename
            if not path.is_file():
                fail(f"Indexed safetensors file is missing: {path}")
            self.handles[filename] = stack.enter_context(
                safe_open(path, framework="pt", device="cpu")
            )

        for name, filename in weight_map.items():
            if name not in self.handles[filename].keys():
                fail(f"Index maps absent tensor {name} to {filename}")

    @property
    def names(self) -> set[str]:
        return set(self.weight_map)

    def handle(self, name: str):
        try:
            return self.handles[self.weight_map[name]]
        except KeyError as error:
            fail(f"Unknown tensor: {name}")
            raise error

    def shape(self, name: str) -> tuple[int, ...]:
        return tuple(self.handle(name).get_slice(name).get_shape())

    def dtype(self, name: str) -> str:
        return self.handle(name).get_slice(name).get_dtype()

    def tensor(self, name: str) -> torch.Tensor:
        return self.handle(name).get_tensor(name)

    def rows(self, name: str, start: int, end: int) -> torch.Tensor:
        shape = self.shape(name)
        if not shape:
            fail(f"Cannot row-slice scalar tensor: {name}")
        return self.handle(name).get_slice(name)[start:end]


@dataclass
class ErrorAccumulator:
    elements: int = 0
    reference_square_sum: float = 0.0
    approximation_square_sum: float = 0.0
    error_square_sum: float = 0.0
    dot_sum: float = 0.0
    absolute_error_sum: float = 0.0
    max_absolute_error: float = 0.0

    def update(self, approximation: torch.Tensor, reference: torch.Tensor) -> None:
        if approximation.shape != reference.shape:
            fail(f"Metric shape mismatch: {approximation.shape} != {reference.shape}")
        if not torch.isfinite(approximation).all():
            fail("Dequantized tensor contains non-finite values")
        if not torch.isfinite(reference).all():
            fail("Official reference tensor contains non-finite values")

        approximation = approximation.float()
        reference = reference.float()
        error = approximation - reference
        self.elements += reference.numel()
        self.reference_square_sum += torch.sum(
            reference * reference, dtype=torch.float64
        ).item()
        self.approximation_square_sum += torch.sum(
            approximation * approximation, dtype=torch.float64
        ).item()
        self.error_square_sum += torch.sum(error * error, dtype=torch.float64).item()
        self.dot_sum += torch.sum(approximation * reference, dtype=torch.float64).item()
        absolute = error.abs()
        self.absolute_error_sum += torch.sum(absolute, dtype=torch.float64).item()
        self.max_absolute_error = max(
            self.max_absolute_error,
            absolute.max().item() if absolute.numel() else 0.0,
        )

    def merge(self, other: "ErrorAccumulator") -> None:
        self.elements += other.elements
        self.reference_square_sum += other.reference_square_sum
        self.approximation_square_sum += other.approximation_square_sum
        self.error_square_sum += other.error_square_sum
        self.dot_sum += other.dot_sum
        self.absolute_error_sum += other.absolute_error_sum
        self.max_absolute_error = max(
            self.max_absolute_error, other.max_absolute_error
        )

    def report(self) -> dict[str, float | int]:
        if self.elements == 0 or self.reference_square_sum <= 0.0:
            fail("Cannot report empty or zero-norm error metrics")
        denominator = math.sqrt(
            self.reference_square_sum * self.approximation_square_sum
        )
        if denominator <= 0.0:
            fail("Cannot report cosine for a zero-norm approximation")
        return {
            "elements": self.elements,
            "relative_l2_error": math.sqrt(
                self.error_square_sum / self.reference_square_sum
            ),
            "cosine_similarity": self.dot_sum / denominator,
            "mean_absolute_error": self.absolute_error_sum / self.elements,
            "max_absolute_error": self.max_absolute_error,
            "reference_rms": math.sqrt(
                self.reference_square_sum / self.elements
            ),
            "approximation_rms": math.sqrt(
                self.approximation_square_sum / self.elements
            ),
        }


def compare_reference_precision(
    converted: TensorStore,
    official: TensorStore,
    names: list[str],
    chunk_rows: int,
) -> dict[str, Any]:
    total_elements = 0
    total_bytes = 0
    identical_tensors = 0
    differing_reports: list[dict[str, Any]] = []
    differing_total = ErrorAccumulator()
    effective_gain_total = ErrorAccumulator()
    expected_norms = expected_offset_norm_names()
    observed_norms: set[str] = set()
    for index, name in enumerate(names, start=1):
        converted_shape = converted.shape(name)
        official_shape = official.shape(name)
        converted_dtype = converted.dtype(name)
        official_dtype = official.dtype(name)
        if converted_shape != official_shape or converted_dtype != official_dtype:
            fail(
                f"Untouched tensor metadata mismatch for {name}: "
                f"{converted_dtype}{converted_shape} != {official_dtype}{official_shape}"
            )

        if not converted_shape:
            left = converted.tensor(name)
            right = official.tensor(name)
            equal = torch.equal(left.view(torch.uint8), right.view(torch.uint8))
            elements = 1
            bytes_count = left.element_size()
        else:
            equal = True
            elements = math.prod(converted_shape)
            bytes_count = elements * converted.rows(name, 0, 1).element_size()
            for start in range(0, converted_shape[0], chunk_rows):
                end = min(start + chunk_rows, converted_shape[0])
                left = converted.rows(name, start, end).contiguous().view(torch.uint8)
                right = official.rows(name, start, end).contiguous().view(torch.uint8)
                if not torch.equal(left, right):
                    equal = False
                    break
        if equal:
            identical_tensors += 1
        else:
            accumulator = ErrorAccumulator()
            if not converted_shape:
                accumulator.update(converted.tensor(name), official.tensor(name))
            else:
                for start in range(0, converted_shape[0], chunk_rows):
                    end = min(start + chunk_rows, converted_shape[0])
                    accumulator.update(
                        converted.rows(name, start, end),
                        official.rows(name, start, end),
                    )
            metrics = accumulator.report()
            if name not in expected_norms:
                fail(
                    "Unexpected modified reference-precision tensor: "
                    f"{name}: {metrics}"
                )
            official_tensor = official.tensor(name)
            converted_tensor = converted.tensor(name)
            if official_tensor.dtype != torch.bfloat16:
                fail(f"Offset norm is not BF16 in the official checkpoint: {name}")
            # llm-compressor's offset-norm calibration context converts Qwen's
            # stored offset w to a standard gain in BF16, then restores it:
            #   gain = BF16(1 + FP32(w)); restored = BF16(FP32(gain) - 1)
            # This is lossy because BF16 spacing near one is much coarser than
            # spacing near the small stored offset. Require the source snapshot
            # to match that transformation exactly rather than accepting a broad
            # similarity threshold that could hide an unrelated mutation.
            expected_roundtrip = (
                (official_tensor.float() + 1.0).to(torch.bfloat16).float() - 1.0
            ).to(torch.bfloat16)
            if not torch.equal(converted_tensor, expected_roundtrip):
                mismatch = torch.count_nonzero(
                    converted_tensor != expected_roundtrip
                ).item()
                fail(
                    "Modified offset norm does not exactly match the known BF16 "
                    f"calibration round trip: {name} ({mismatch} elements differ)"
                )
            observed_norms.add(name)
            effective_gain_total.update(
                converted_tensor.float() + 1.0,
                official_tensor.float() + 1.0,
            )
            differing_total.merge(accumulator)
            differing_reports.append({"name": name, "metrics": metrics})
        total_elements += elements
        total_bytes += bytes_count
        if not equal or index % 100 == 0 or index == len(names):
            print(
                f"reference-precision: {index}/{len(names)} {name} "
                f"byte_identical={equal}",
                file=sys.stderr,
                flush=True,
            )
    if observed_norms != expected_norms:
        missing = sorted(expected_norms - observed_norms)
        extra = sorted(observed_norms - expected_norms)
        fail(
            "Offset-norm mutation set mismatch: "
            f"missing={missing}, extra={extra}"
        )
    report: dict[str, Any] = {
        "tensors": len(names),
        "byte_identical_tensors": identical_tensors,
        "numerically_modified_tensors": len(differing_reports),
        "elements": total_elements,
        "bytes": total_bytes,
        "modified_tensor_names": [item["name"] for item in differing_reports],
        "worst_modified_tensors": worst_reports(differing_reports),
    }
    if differing_reports:
        report["modified_aggregate"] = differing_total.report()
        report["modification_classification"] = (
            "exact lossy BF16 offset-norm calibration round trip"
        )
        report["effective_gain_aggregate"] = effective_gain_total.report()
    return report


def audit_fp8_tensor(
    converted: TensorStore,
    official: TensorStore,
    name: str,
    chunk_rows: int,
) -> tuple[ErrorAccumulator, dict[str, int]]:
    scale_name = name.removesuffix("weight") + "weight_scale"
    shape = converted.shape(name)
    if len(shape) != 2 or official.shape(name) != shape:
        fail(f"Invalid FP8 weight/reference shape for {name}: {shape}")
    if converted.dtype(name) != "F8_E4M3" or official.dtype(name) != "BF16":
        fail(
            f"Invalid FP8/reference dtype for {name}: "
            f"{converted.dtype(name)}/{official.dtype(name)}"
        )
    # Compressed Tensors leaves scale_dtype unspecified for this group, so this
    # particular checkpoint serializes the channel scales in BF16. vLLM loads
    # them without changing their meaning; other valid exports may use F32.
    if converted.dtype(scale_name) not in {"BF16", "F32"} or converted.shape(
        scale_name
    ) != (shape[0], 1):
        fail(f"Invalid FP8 channel scale for {name}")

    accumulator = ErrorAccumulator()
    saturation = 0
    for start in range(0, shape[0], chunk_rows):
        end = min(start + chunk_rows, shape[0])
        quantized = converted.rows(name, start, end).float()
        scale = converted.rows(scale_name, start, end).float()
        if not torch.isfinite(scale).all() or not torch.all(scale > 0):
            fail(f"FP8 scale is not finite and positive: {scale_name}")
        approximation = quantized * scale
        reference = official.rows(name, start, end).float()
        accumulator.update(approximation, reference)
        saturation += torch.count_nonzero(quantized.abs() == FP8_MAX).item()
    return accumulator, {"saturated_values": saturation}


def audit_nvfp4_tensor(
    converted: TensorStore,
    official: TensorStore,
    packed_name: str,
    chunk_rows: int,
) -> tuple[ErrorAccumulator, dict[str, Any]]:
    stem = packed_name.removesuffix("weight_packed")
    official_name = stem + "weight"
    scale_name = stem + "weight_scale"
    global_name = stem + "weight_global_scale"
    input_name = stem + "input_global_scale"

    packed_shape = converted.shape(packed_name)
    official_shape = official.shape(official_name)
    if len(packed_shape) != 2 or official_shape != (packed_shape[0], packed_shape[1] * 2):
        fail(f"Invalid packed/reference NVFP4 shapes for {packed_name}")
    if converted.dtype(packed_name) != "U8" or official.dtype(official_name) != "BF16":
        fail(f"Invalid packed/reference NVFP4 dtypes for {packed_name}")
    expected_scale_shape = (official_shape[0], official_shape[1] // GROUP_SIZE)
    if converted.dtype(scale_name) != "F8_E4M3" or converted.shape(scale_name) != expected_scale_shape:
        fail(f"Invalid NVFP4 block scale for {packed_name}")
    if converted.dtype(global_name) != "F32" or converted.shape(global_name) != (1,):
        fail(f"Invalid NVFP4 weight global scale for {packed_name}")
    if converted.dtype(input_name) != "F32" or converted.shape(input_name) != (1,):
        fail(f"Invalid NVFP4 input global scale for {packed_name}")

    global_divisor = converted.tensor(global_name).float()
    input_divisor = converted.tensor(input_name).float()
    if (
        not torch.isfinite(global_divisor).all()
        or not torch.all(global_divisor > 0)
        or not torch.isfinite(input_divisor).all()
        or not torch.all(input_divisor > 0)
    ):
        fail(f"NVFP4 global scales are not finite and positive: {stem}")

    accumulator = ErrorAccumulator()
    nibble_histogram = torch.zeros(16, dtype=torch.int64)
    saturated_values = 0
    scale_saturation = 0
    for start in range(0, packed_shape[0], chunk_rows):
        end = min(start + chunk_rows, packed_shape[0])
        packed = converted.rows(packed_name, start, end)
        low = packed & 0x0F
        high = (packed >> 4) & 0x0F
        nibble_histogram += torch.bincount(low.flatten().long(), minlength=16)
        nibble_histogram += torch.bincount(high.flatten().long(), minlength=16)
        saturated_values += torch.count_nonzero((low & 0x07) == 7).item()
        saturated_values += torch.count_nonzero((high & 0x07) == 7).item()

        decoded = torch.empty(
            (end - start, official_shape[1]), dtype=torch.float32
        )
        decoded[:, 0::2] = E2M1[low.long()]
        decoded[:, 1::2] = E2M1[high.long()]

        scale = converted.rows(scale_name, start, end).float()
        if not torch.isfinite(scale).all() or not torch.all(scale > 0):
            fail(f"NVFP4 block scale is not finite and positive: {scale_name}")
        scale_saturation += torch.count_nonzero(scale.abs() == FP8_MAX).item()
        approximation = (
            decoded.reshape(end - start, -1, GROUP_SIZE)
            * (scale / global_divisor).unsqueeze(-1)
        ).reshape(end - start, official_shape[1])
        reference = official.rows(official_name, start, end).float()
        accumulator.update(approximation, reference)

    return accumulator, {
        "saturated_values": saturated_values,
        "scale_saturated_values": scale_saturation,
        "nibble_histogram": nibble_histogram.tolist(),
        "checkpoint_weight_global_divisor": global_divisor.item(),
        "checkpoint_input_global_divisor": input_divisor.item(),
    }


def audit_kv_scales(converted: TensorStore, names: list[str]) -> dict[str, Any]:
    values: list[float] = []
    dtypes: dict[str, int] = {}
    for name in names:
        dtype = converted.dtype(name)
        if dtype not in {"BF16", "F32"} or converted.shape(name) != (1,):
            fail(f"Invalid static K/V calibration metadata: {name}")
        value = converted.tensor(name).item()
        if not math.isfinite(value) or value <= 0:
            fail(f"Static K/V calibration value is not finite and positive: {name}")
        dtypes[dtype] = dtypes.get(dtype, 0) + 1
        values.append(value)
    return {
        "tensors": len(names),
        "storage_dtypes": dtypes,
        "minimum": min(values),
        "maximum": max(values),
        "runtime_use": "none: turboquant_k8v4 quantizes live K/V independently",
    }


def worst_reports(reports: list[dict[str, Any]], count: int = 20) -> list[dict[str, Any]]:
    return sorted(
        reports,
        key=lambda item: item["metrics"]["relative_l2_error"],
        reverse=True,
    )[:count]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, default=Path("/project"))
    parser.add_argument(
        "--converted", type=Path, default=Path("/project/Qwen3.8-27B-NVFP4-Unsloth")
    )
    parser.add_argument("--chunk-rows", type=int, default=512)
    parser.add_argument("--skip-hashes", action="store_true")
    args = parser.parse_args()
    if args.chunk_rows <= 0:
        parser.error("--chunk-rows must be positive")
    return args


def main() -> None:
    args = parse_args()
    project = args.project.resolve()
    converted_root = args.converted.resolve()

    provenance: dict[str, Any] = {
        "official_repository": "Qwen/Qwen3.8-27B",
        "official_revision": OFFICIAL_REVISION,
        "converted_repository": "unsloth/Qwen3.8-27B-NVFP4",
        "converted_revision": CONVERTED_REVISION,
    }
    if not args.skip_hashes:
        provenance["official_manifest"] = verify_manifest(
            project / "manifests/official-qwen38-27b-1d4bf0f2.sha256",
            project,
        )
        provenance["converted_manifest"] = verify_manifest(
            project / "manifests/model-snapshot-16b6615a.sha256",
            converted_root,
        )

    with contextlib.ExitStack() as stack:
        official = TensorStore(project, project / "model.safetensors.index.json", stack)
        converted = TensorStore(
            converted_root,
            converted_root / "model.safetensors.index.json",
            stack,
        )

        if len(official.names) != EXPECTED_OFFICIAL_TENSORS:
            fail(f"Unexpected official tensor count: {len(official.names)}")
        if len(converted.names) != EXPECTED_CONVERTED_TENSORS:
            fail(f"Unexpected converted tensor count: {len(converted.names)}")

        nvfp4_packed = sorted(
            name for name in converted.names if name.endswith(".weight_packed")
        )
        fp8_weights = sorted(
            name
            for name in converted.names
            if name.endswith(".weight") and converted.dtype(name) == "F8_E4M3"
        )
        kv_scales = sorted(
            name
            for name in converted.names
            if name.endswith(".k_scale") or name.endswith(".v_scale")
        )

        if len(nvfp4_packed) != EXPECTED_NVFP4_WEIGHTS:
            fail(f"Unexpected NVFP4 tensor count: {len(nvfp4_packed)}")
        if len(fp8_weights) != EXPECTED_FP8_WEIGHTS:
            fail(f"Unexpected FP8 tensor count: {len(fp8_weights)}")
        if len(kv_scales) != EXPECTED_KV_SCALE_TENSORS:
            fail(f"Unexpected K/V scale count: {len(kv_scales)}")

        nvfp4_official = {
            name.removesuffix("weight_packed") + "weight" for name in nvfp4_packed
        }
        reference_precision_names = sorted(
            official.names - set(fp8_weights) - nvfp4_official
        )
        if len(reference_precision_names) != EXPECTED_REFERENCE_PRECISION_TENSORS:
            fail(
                "Unexpected reference-precision tensor count: "
                f"{len(reference_precision_names)}"
            )
        if not set(reference_precision_names).issubset(converted.names):
            missing = sorted(set(reference_precision_names) - converted.names)
            fail(f"Converted checkpoint omitted official tensors: {missing}")

        nvfp4_metadata = {
            name.removesuffix("weight_packed") + suffix
            for name in nvfp4_packed
            for suffix in (
                "weight_scale",
                "weight_global_scale",
                "input_global_scale",
            )
        }
        fp8_metadata = {
            name.removesuffix("weight") + "weight_scale" for name in fp8_weights
        }
        accounted_converted = (
            set(reference_precision_names)
            | set(fp8_weights)
            | set(nvfp4_packed)
            | nvfp4_metadata
            | fp8_metadata
            | set(kv_scales)
        )
        if accounted_converted != converted.names:
            fail(
                "Converted tensor accounting mismatch; unaccounted="
                f"{sorted(converted.names - accounted_converted)}, duplicated/absent="
                f"{sorted(accounted_converted - converted.names)}"
            )

        reference_precision_report = compare_reference_precision(
            converted, official, reference_precision_names, args.chunk_rows
        )

        modified_names = set(reference_precision_report["modified_tensor_names"])
        forbidden_modified = sorted(
            name
            for name in modified_names
            if name.startswith("model.visual.") or name.startswith("mtp.")
        )
        if forbidden_modified:
            fail(
                "The checkpoint claims to exclude the complete vision/MTP payload "
                f"from conversion, but these tensors differ: {forbidden_modified}"
            )

        fp8_total = ErrorAccumulator()
        fp8_reports: list[dict[str, Any]] = []
        fp8_saturated = 0
        for index, name in enumerate(fp8_weights, start=1):
            accumulator, metadata = audit_fp8_tensor(
                converted, official, name, args.chunk_rows
            )
            metrics = accumulator.report()
            if metrics["cosine_similarity"] < 0.99 or metrics["relative_l2_error"] > 0.15:
                fail(f"FP8 conversion is not credibly aligned with official tensor {name}: {metrics}")
            fp8_total.merge(accumulator)
            fp8_saturated += metadata["saturated_values"]
            fp8_reports.append({"name": name, "metrics": metrics, **metadata})
            print(
                f"fp8: {index}/{len(fp8_weights)} {name} "
                f"rel_l2={metrics['relative_l2_error']:.8f}",
                file=sys.stderr,
                flush=True,
            )

        nvfp4_total = ErrorAccumulator()
        nvfp4_reports: list[dict[str, Any]] = []
        nvfp4_saturated = 0
        nvfp4_scale_saturated = 0
        nvfp4_histogram = [0] * 16
        for index, name in enumerate(nvfp4_packed, start=1):
            accumulator, metadata = audit_nvfp4_tensor(
                converted, official, name, args.chunk_rows
            )
            metrics = accumulator.report()
            if metrics["cosine_similarity"] < 0.90 or metrics["relative_l2_error"] > 0.50:
                fail(f"NVFP4 conversion is not credibly aligned with official tensor {name}: {metrics}")
            nvfp4_total.merge(accumulator)
            nvfp4_saturated += metadata["saturated_values"]
            nvfp4_scale_saturated += metadata["scale_saturated_values"]
            for nibble, count in enumerate(metadata["nibble_histogram"]):
                nvfp4_histogram[nibble] += count
            nvfp4_reports.append({"name": name, "metrics": metrics, **metadata})
            print(
                f"nvfp4: {index}/{len(nvfp4_packed)} {name} "
                f"rel_l2={metrics['relative_l2_error']:.8f}",
                file=sys.stderr,
                flush=True,
            )

        result = {
            "status": "PASS",
            "provenance": provenance,
            "accounting": {
                "official_tensors": len(official.names),
                "converted_tensors": len(converted.names),
                "reference_precision_tensors": len(reference_precision_names),
                "fp8_weight_tensors": len(fp8_weights),
                "fp8_scale_tensors": len(fp8_metadata),
                "nvfp4_weight_tensors": len(nvfp4_packed),
                "nvfp4_metadata_tensors": len(nvfp4_metadata),
                "kv_calibration_tensors": len(kv_scales),
            },
            "reference_precision": reference_precision_report,
            "fp8": {
                "semantics": (
                    "E4M3 value * per-output-channel floating scale "
                    "(BF16 in this snapshot)"
                ),
                "aggregate": fp8_total.report(),
                "saturated_values": fp8_saturated,
                "saturation_fraction": (
                    fp8_saturated / fp8_total.elements
                ),
                "worst_tensors": worst_reports(fp8_reports),
            },
            "nvfp4": {
                "semantics": (
                    "low-nibble-first E2M1 value * FP8 block scale / "
                    "checkpoint weight_global_scale divisor"
                ),
                "group_size": GROUP_SIZE,
                "aggregate": nvfp4_total.report(),
                "saturated_values": nvfp4_saturated,
                "saturation_fraction": (
                    nvfp4_saturated / nvfp4_total.elements
                ),
                "scale_saturated_values": nvfp4_scale_saturated,
                "scale_saturation_fraction": (
                    nvfp4_scale_saturated
                    / (nvfp4_total.elements // GROUP_SIZE)
                ),
                "nibble_histogram": nvfp4_histogram,
                "worst_tensors": worst_reports(nvfp4_reports),
            },
            "kv_calibration": audit_kv_scales(converted, kv_scales),
        }
        print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
