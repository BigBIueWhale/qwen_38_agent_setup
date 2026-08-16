#!/usr/bin/env python3
"""Actual-checkpoint NVFP4 kernel acceptance for Qwen3.8-27B.

This loads one complete, deliberately worst-error NVFP4 layer from the pinned
checkpoint, selects vLLM's production kernel, and compares its BF16 outputs to
an independent E2M1/FP8-block-scale dequantization and BF16 matmul.
"""

from __future__ import annotations

import math

import torch
from safetensors import safe_open

from vllm import _custom_ops as ops
from vllm.model_executor.kernels.linear import init_nvfp4_linear_kernel

MODEL_FILE = "/model/model.safetensors"
PREFIX = "model.language_model.layers.0.mlp.down_proj"
EXPECTED_KERNEL = "FlashInferCutlassNvFp4LinearKernel"
E2M1_MAGNITUDES = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def unpack_e2m1(packed: torch.Tensor) -> torch.Tensor:
    require(packed.dtype == torch.uint8, "NVFP4 payload must be uint8")
    low = packed & 0x0F
    high = (packed >> 4) & 0x0F
    codes = torch.stack((low, high), dim=-1).reshape(*packed.shape[:-1], -1)
    signs = torch.where(codes & 0x08 != 0, -1.0, 1.0)
    magnitudes = E2M1_MAGNITUDES.to(packed.device)[(codes & 0x07).long()]
    return signs * magnitudes


def unswizzle_128x4(
    swizzled: torch.Tensor,
    rows: int,
    elements_per_row: int,
    block_size: int = 16,
) -> torch.Tensor:
    row_tiles = math.ceil(rows / 128)
    tile_width = block_size * 4
    column_tiles = math.ceil(elements_per_row / tile_width)
    reshaped = swizzled.reshape(1, row_tiles, column_tiles, 32, 4, 4)
    linear = reshaped.permute(0, 1, 4, 3, 2, 5).reshape(
        row_tiles * 128, column_tiles * 4
    )
    return linear[:rows, : elements_per_row // block_size]


def dequantize_linear_nvfp4(
    packed: torch.Tensor,
    block_scales: torch.Tensor,
    global_divisor: torch.Tensor,
) -> torch.Tensor:
    rows, packed_columns = packed.shape
    columns = packed_columns * 2
    require(columns % 16 == 0, "NVFP4 input width must be a multiple of 16")
    values = unpack_e2m1(packed).reshape(rows, columns // 16, 16)
    scales = block_scales.to(torch.float32) / global_divisor.to(torch.float32)
    require(
        scales.shape == (rows, columns // 16),
        f"unexpected linear scale shape: {tuple(scales.shape)}",
    )
    return (values * scales.unsqueeze(-1)).reshape(rows, columns).to(torch.bfloat16)


def dequantize_swizzled_nvfp4(
    packed: torch.Tensor,
    block_scales: torch.Tensor,
    global_divisor: torch.Tensor,
) -> torch.Tensor:
    rows, packed_columns = packed.shape
    columns = packed_columns * 2
    linear_scales = unswizzle_128x4(block_scales, rows, columns)
    return dequantize_linear_nvfp4(packed, linear_scales, global_divisor)


class AuditLayer(torch.nn.Module):
    pass


@torch.inference_mode()
def main() -> None:
    require(torch.cuda.is_available(), "NVFP4 kernel acceptance requires CUDA")
    require(torch.cuda.get_device_capability(0) == (12, 0), "expected SM 12.0 GPU")
    device = torch.device("cuda:0")

    names = {
        "weight": f"{PREFIX}.weight_packed",
        "weight_scale": f"{PREFIX}.weight_scale",
        "weight_global": f"{PREFIX}.weight_global_scale",
        "input_global": f"{PREFIX}.input_global_scale",
    }
    with safe_open(MODEL_FILE, framework="pt", device="cpu") as handle:
        tensors = {name: handle.get_tensor(key) for name, key in names.items()}

    raw_weight = tensors["weight"].to(device)
    raw_weight_scale = tensors["weight_scale"].to(device)
    weight_divisor = tensors["weight_global"].max().to(device, torch.float32)
    input_divisor = tensors["input_global"].max().to(device, torch.float32)
    output_width, packed_input_width = raw_weight.shape
    input_width = packed_input_width * 2

    require(raw_weight.dtype == torch.uint8, "checkpoint weight is not packed uint8")
    require(
        raw_weight_scale.dtype == torch.float8_e4m3fn,
        "checkpoint block scale is not E4M3",
    )
    require(
        raw_weight_scale.shape == (output_width, input_width // 16),
        "checkpoint NVFP4 weight/scale geometry differs",
    )
    require((output_width, input_width) == (5120, 17408), "unexpected down-proj shape")
    require(weight_divisor.item() == 2752.0, "unexpected checkpoint weight divisor")
    require(input_divisor.item() == 161.0, "unexpected checkpoint input divisor")

    kernel = init_nvfp4_linear_kernel(use_a16=False)
    require(
        type(kernel).__name__ == EXPECTED_KERNEL,
        f"unexpected production NVFP4 kernel: {type(kernel).__name__}",
    )
    layer = AuditLayer().to(device)
    layer.output_size_per_partition = output_width
    layer.weight = torch.nn.Parameter(raw_weight, requires_grad=False)
    layer.weight_scale = torch.nn.Parameter(raw_weight_scale, requires_grad=False)
    layer.input_global_scale_inv = torch.nn.Parameter(
        input_divisor, requires_grad=False
    )
    layer.alpha = torch.nn.Parameter(
        1.0 / (input_divisor * weight_divisor), requires_grad=False
    )
    kernel.process_weights_after_loading(layer)
    require(getattr(layer, "weights_padding_cols", None) == 0, "unexpected K padding")
    require(layer.weight.shape == raw_weight.shape, "unexpected weight padding")

    dequant_weight = dequantize_linear_nvfp4(
        raw_weight, raw_weight_scale, weight_divisor
    )
    torch.manual_seed(3827)
    results = []
    for rows in (1, 17, 129):
        activation = (
            torch.randn(rows, input_width, device=device, dtype=torch.bfloat16) * 3.0
        )
        activation_packed, activation_scales = ops.scaled_fp4_quant(
            activation,
            input_divisor,
            is_sf_swizzled_layout=True,
            backend="flashinfer-cutlass",
            padded_n=input_width,
        )
        dequant_activation = dequantize_swizzled_nvfp4(
            activation_packed, activation_scales, input_divisor
        )
        reference = dequant_activation @ dequant_weight.T
        actual = kernel.apply_weights(layer, activation)
        torch.cuda.synchronize()

        actual_f32 = actual.float()
        reference_f32 = reference.float()
        difference = actual_f32 - reference_f32
        max_abs = difference.abs().max().item()
        relative_l2 = (
            difference.norm() / reference_f32.norm().clamp_min(1e-12)
        ).item()
        cosine = torch.nn.functional.cosine_similarity(
            actual_f32.reshape(1, -1), reference_f32.reshape(1, -1)
        ).item()
        close_mask = torch.isclose(
            actual_f32, reference_f32, atol=0.1, rtol=0.1
        )
        mismatch_fraction = 1.0 - close_mask.float().mean().item()
        require(actual.dtype == torch.bfloat16, "NVFP4 kernel output is not BF16")
        require(max_abs <= 1.0, f"NVFP4 max absolute error too large for M={rows}")
        require(relative_l2 < 0.005, f"NVFP4 relative L2 too large for M={rows}")
        require(cosine > 0.99998, f"NVFP4 cosine too small for M={rows}")
        require(
            mismatch_fraction < 1e-5,
            f"NVFP4 elementwise outlier fraction too large for M={rows}: "
            f"{mismatch_fraction:.9f}",
        )
        results.append(
            f"M={rows}:max_abs={max_abs:.8f},rel_l2={relative_l2:.9f},"
            f"cos={cosine:.9f},mismatch_fraction={mismatch_fraction:.9f}"
        )

    print(
        "PASS nvfp4_kernel: "
        f"layer={PREFIX} shape={output_width}x{input_width} "
        f"kernel={type(kernel).__name__} "
        + " ".join(results)
    )


if __name__ == "__main__":
    main()
