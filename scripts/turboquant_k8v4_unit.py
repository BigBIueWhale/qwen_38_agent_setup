#!/usr/bin/env python3
"""Dependency-free GPU acceptance for Qwen3.8-27B TurboQuant K8V4.

This verifies the installed runtime, not a reimplementation: the Triton store
and fused decode launchers are imported from vLLM.  A separate PyTorch path
constructs every expected cache byte and the complete GQA attention result.
"""

from __future__ import annotations

import math

import torch

from vllm.model_executor.layers.quantization.turboquant.config import (
    TurboQuantConfig,
)
from vllm.v1.attention.ops.triton_turboquant_decode import (
    triton_turboquant_decode_attention,
)
from vllm.v1.attention.ops.triton_turboquant_store import (
    triton_turboquant_store,
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    require(torch.cuda.is_available(), "TurboQuant acceptance requires CUDA")
    device = torch.device("cuda:0")
    preset = "turboquant_k8v4"
    head_dim = 256
    num_kv_heads = 4
    num_query_heads = 24
    num_tokens = 37
    block_size = 16
    num_blocks = math.ceil(num_tokens / block_size)
    cfg = TurboQuantConfig.from_cache_dtype(preset, head_dim=head_dim)

    require(cfg.key_fp8, "K8V4 must use FP8 keys")
    require(cfg.key_packed_size == 256, "K8V4 D=256 key must occupy 256 bytes")
    require(cfg.value_packed_size == 132, "V4 D=256 must occupy 128+2+2 bytes")
    require(cfg.slot_size == 388, "K8V4 D=256 slot must occupy exactly 388 bytes")
    require(cfg.slot_size_aligned == 388, "K8V4 slot must need no hidden padding")

    torch.manual_seed(38027)
    key = torch.randn(
        num_tokens,
        num_kv_heads,
        head_dim,
        device=device,
        dtype=torch.bfloat16,
    )
    value = torch.randn_like(key)
    query = torch.randn(
        1,
        num_query_heads,
        head_dim,
        device=device,
        dtype=torch.bfloat16,
    )
    kv_cache = torch.zeros(
        num_blocks,
        block_size,
        num_kv_heads,
        cfg.slot_size_aligned,
        device=device,
        dtype=torch.uint8,
    )
    slots = torch.arange(num_tokens, device=device, dtype=torch.int32)
    identity = torch.eye(head_dim, device=device, dtype=torch.float32)
    centroids = torch.arange(16, device=device, dtype=torch.float32)
    midpoints = torch.arange(15, device=device, dtype=torch.float32) + 0.5

    triton_turboquant_store(
        key,
        value,
        kv_cache,
        slots,
        identity,
        midpoints,
        mse_bits=cfg.key_mse_bits,
        key_packed_size=cfg.key_packed_size,
        value_quant_bits=cfg.effective_value_quant_bits,
        key_fp8=cfg.key_fp8,
    )
    torch.cuda.synchronize()

    # Independent exact encoder for the K8V4 cache representation.
    key_fp8 = key.float().to(torch.float8_e4m3fn)
    expected_key_bytes = key_fp8.view(torch.uint8).cpu()
    value_fp32 = value.float()
    value_min = value_fp32.amin(dim=-1)
    value_max = value_fp32.amax(dim=-1)
    quant_scale = torch.clamp((value_max - value_min) / 15.0, min=1e-8)
    value_codes = torch.clamp(
        (
            (value_fp32 - value_min.unsqueeze(-1)) / quant_scale.unsqueeze(-1)
            + 0.5
        ).to(torch.int32),
        0,
        15,
    )
    packed_values = (
        value_codes[..., 0::2] | (value_codes[..., 1::2] << 4)
    ).to(torch.uint8).cpu()
    stored_scales = quant_scale.to(torch.float16)
    stored_minima = value_min.to(torch.float16)
    scale_bytes = (
        stored_scales.view(torch.uint8)
        .reshape(num_tokens, num_kv_heads, 2)
        .cpu()
    )
    minimum_bytes = (
        stored_minima.view(torch.uint8)
        .reshape(num_tokens, num_kv_heads, 2)
        .cpu()
    )
    expected_slots = torch.cat(
        [expected_key_bytes, packed_values, scale_bytes, minimum_bytes], dim=-1
    )
    actual_slots = torch.empty_like(expected_slots)
    cache_cpu = kv_cache.cpu()
    for token_idx in range(num_tokens):
        block_idx, block_offset = divmod(token_idx, block_size)
        actual_slots[token_idx] = cache_cpu[block_idx, block_offset]
    value_start = cfg.key_packed_size
    value_end = value_start + head_dim // 2
    require(
        torch.equal(actual_slots[..., :value_start], expected_key_bytes),
        "Triton E4M3 key bytes differ from the independent FP8 encoder",
    )
    require(
        torch.equal(actual_slots[..., value_end:386], scale_bytes),
        "Triton value-scale bytes differ from independent FP16 encoding",
    )
    require(
        torch.equal(actual_slots[..., 386:388], minimum_bytes),
        "Triton value-minimum bytes differ from independent FP16 encoding",
    )

    # Algebraically identical FP32 expressions may land immediately to either
    # side of an integer boundary after compiler reassociation.  Decode the
    # actual nibbles and require exact agreement everywhere except within a
    # small, explicit boundary interval, where only one code step is allowed.
    actual_packed = actual_slots[..., value_start:value_end]
    actual_codes_cpu = torch.empty(
        num_tokens, num_kv_heads, head_dim, dtype=torch.int32
    )
    actual_codes_cpu[..., 0::2] = (actual_packed & 0x0F).to(torch.int32)
    actual_codes_cpu[..., 1::2] = (actual_packed >> 4).to(torch.int32)
    actual_codes = actual_codes_cpu.to(device)
    rounded_inputs = (
        (value_fp32 - value_min.unsqueeze(-1)) / quant_scale.unsqueeze(-1) + 0.5
    )
    boundary_distance = (rounded_inputs - rounded_inputs.round()).abs()
    code_delta = (actual_codes - value_codes).abs()
    boundary_epsilon = 2e-5
    stable = boundary_distance > boundary_epsilon
    require(
        bool((code_delta[stable] == 0).all()),
        "Triton V4 code differs from the reference away from a rounding boundary",
    )
    require(
        bool((code_delta[~stable] <= 1).all()),
        "Triton V4 boundary code differs by more than one quantization step",
    )
    num_boundary_choices = int((code_delta != 0).sum().item())

    block_table = torch.arange(
        num_blocks, device=device, dtype=torch.int32
    ).unsqueeze(0)
    seq_lens = torch.tensor([num_tokens], device=device, dtype=torch.int32)
    output = triton_turboquant_decode_attention(
        query=query,
        kv_cache=kv_cache,
        block_table=block_table,
        seq_lens=seq_lens,
        Pi=identity,
        centroids=centroids,
        scale=1.0 / math.sqrt(head_dim),
        mse_bits=cfg.key_mse_bits,
        key_packed_size=cfg.key_packed_size,
        value_quant_bits=cfg.effective_value_quant_bits,
        key_fp8=True,
        norm_correction=False,
        PiT=identity,
        max_num_kv_splits=4,
    )
    torch.cuda.synchronize()

    dequant_key = key_fp8.float()
    dequant_value = (
        actual_codes.float() * stored_scales.float().unsqueeze(-1)
        + stored_minima.float().unsqueeze(-1)
    )
    reference = torch.empty_like(query, dtype=torch.float32)
    group_size = num_query_heads // num_kv_heads
    for query_head in range(num_query_heads):
        kv_head = query_head // group_size
        scores = (
            query[0, query_head].float() @ dequant_key[:, kv_head].T
        ) / math.sqrt(head_dim)
        probabilities = torch.softmax(scores, dim=-1)
        reference[0, query_head] = probabilities @ dequant_value[:, kv_head]

    output_fp32 = output.float()
    max_abs = (output_fp32 - reference).abs().max().item()
    require(output.dtype == torch.bfloat16, f"unexpected output dtype: {output.dtype}")
    require(
        torch.allclose(output_fp32, reference, atol=2e-2, rtol=2e-2),
        f"fused K8V4 output differs from explicit reference: max_abs={max_abs:.8f}",
    )
    similarities = torch.nn.functional.cosine_similarity(
        output_fp32, reference, dim=-1
    )
    min_cosine = similarities.min().item()
    require(min_cosine > 0.999, f"fused/reference min cosine={min_cosine:.9f}")
    print(
        "PASS turboquant_k8v4: "
        f"D={head_dim} Hq={num_query_heads} Hkv={num_kv_heads} "
        f"tokens={num_tokens} slot={cfg.slot_size} "
        f"boundary_choices={num_boundary_choices} "
        f"max_abs={max_abs:.8f} min_cosine={min_cosine:.9f}"
    )


if __name__ == "__main__":
    main()
