#!/usr/bin/env python3
"""First-principles context/MRoPE acceptance for the pinned Qwen3.8 model.

The test compares vLLM's actual Qwen3.5 image-position builder with an
independent implementation of Transformers 5.15's released Qwen3.5 algorithm.
It also freezes the native text/RoPE/layer geometry used in the VRAM proof.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from transformers import AutoConfig

from vllm.model_executor.layers.rotary_embedding.mrope import MRotaryEmbedding
from vllm.model_executor.models.qwen3_5 import Qwen3_5ForConditionalGeneration
from vllm.multimodal.inputs import (
    MultiModalFeatureSpec,
    MultiModalFieldElem,
    MultiModalKwargsItem,
    PlaceholderRange,
)

MODEL_PATH = "/model"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


@dataclass(frozen=True)
class ImagePlacement:
    offset: int
    grid_h: int
    grid_w: int
    token_count: int


def official_image_reference(
    sequence_length: int,
    placements: list[ImagePlacement],
) -> tuple[torch.Tensor, int]:
    """Released Qwen3.5 get_rope_index algorithm, specialized to images.

    Vision start/end markers remain text tokens. Image placeholder tokens form
    modality runs whose T/H/W coordinates begin at the current text position;
    the following text begins after max(grid_h, grid_w), not token_count.
    """
    chunks: list[torch.Tensor] = []
    cursor = 0
    current_position = 0
    for placement in sorted(placements, key=lambda item: item.offset):
        text_length = placement.offset - cursor
        require(text_length >= 0, "overlapping or out-of-order image placements")
        if text_length:
            text = torch.arange(text_length, dtype=torch.long).expand(3, -1)
            chunks.append(text + current_position)
            current_position += text_length

        temporal = torch.full(
            (placement.grid_h * placement.grid_w,),
            current_position,
            dtype=torch.long,
        )
        height = (
            torch.arange(placement.grid_h, dtype=torch.long)
            .view(-1, 1)
            .expand(-1, placement.grid_w)
            .reshape(-1)
            + current_position
        )
        width = (
            torch.arange(placement.grid_w, dtype=torch.long)
            .view(1, -1)
            .expand(placement.grid_h, -1)
            .reshape(-1)
            + current_position
        )
        chunks.append(torch.stack((temporal, height, width)))
        current_position += max(placement.grid_h, placement.grid_w)
        cursor = placement.offset + placement.token_count

    trailing_text = sequence_length - cursor
    require(trailing_text >= 0, "image placement extends past the prompt")
    if trailing_text:
        text = torch.arange(trailing_text, dtype=torch.long).expand(3, -1)
        chunks.append(text + current_position)
    positions = torch.cat(chunks, dim=1)
    require(positions.shape == (3, sequence_length), "reference length mismatch")
    delta = int(positions.max().item() + 1 - sequence_length)
    return positions, delta


def build_interleaved_prompt(config) -> tuple[
    list[int], list[MultiModalFeatureSpec], list[ImagePlacement]
]:
    tokens: list[int] = []
    features: list[MultiModalFeatureSpec] = []
    placements: list[ImagePlacement] = []
    merge = config.vision_config.spatial_merge_size
    image_specs = (
        (7, 8, 10),
        (13, 14, 6),
        (5, 4, 18),
    )
    for image_index, (text_length, patch_h, patch_w) in enumerate(image_specs):
        tokens.extend([100 + image_index] * text_length)
        tokens.append(config.vision_start_token_id)
        offset = len(tokens)
        grid_h = patch_h // merge
        grid_w = patch_w // merge
        token_count = grid_h * grid_w
        tokens.extend([config.image_token_id] * token_count)
        tokens.append(config.vision_end_token_id)
        placements.append(ImagePlacement(offset, grid_h, grid_w, token_count))
        features.append(
            MultiModalFeatureSpec(
                data=MultiModalKwargsItem(
                    {
                        "image_grid_thw": MultiModalFieldElem(
                            data=torch.tensor([1, patch_h, patch_w]),
                            field=None,
                        )
                    }
                ),
                modality="image",
                identifier=f"image-{image_index}",
                mm_position=PlaceholderRange(offset=offset, length=token_count),
            )
        )
    tokens.extend([199] * 9)
    # Deliberately reverse the transport list. Position offsets, rather than
    # request-container order, must determine chronology across agent turns.
    return tokens, list(reversed(features)), placements


def main() -> None:
    config = AutoConfig.from_pretrained(MODEL_PATH, local_files_only=True)
    text = config.text_config
    rope = text.rope_parameters

    require(
        config.architectures == ["Qwen3_5ForConditionalGeneration"],
        f"unexpected architecture: {config.architectures}",
    )
    require(text.max_position_embeddings == 262_144, "native context is not 262144")
    require(text.head_dim == 256, "unexpected attention head dimension")
    require(text.num_attention_heads == 24, "unexpected query-head count")
    require(text.num_key_value_heads == 4, "unexpected KV-head count")
    require(text.num_hidden_layers == 64, "unexpected language-layer count")
    require(text.full_attention_interval == 4, "unexpected full-attention interval")
    expected_full_layers = list(range(3, 64, 4))
    actual_full_layers = [
        index
        for index, layer_type in enumerate(text.layer_types)
        if layer_type == "full_attention"
    ]
    require(actual_full_layers == expected_full_layers, "full-attention pattern drift")
    require(len(actual_full_layers) == 16, "Qwen3.8 must have 16 KV-cached layers")

    require(rope["rope_type"] == "default", "native profile must not use RoPE scaling")
    require(rope["rope_theta"] == 10_000_000, "unexpected RoPE theta")
    require(rope["partial_rotary_factor"] == 0.25, "unexpected partial RoPE factor")
    require(rope["mrope_section"] == [11, 11, 10], "unexpected MRoPE sections")
    require(rope["mrope_interleaved"] is True, "MRoPE must be interleaved")
    rotary_dim = int(text.head_dim * rope["partial_rotary_factor"])
    require(rotary_dim == 64, "Qwen3.8 rotary dimension must be 64")
    require(sum(rope["mrope_section"]) == rotary_dim // 2, "MRoPE sections do not fit")

    # Transformers 5.15 overwrites temporal frequency slots with H/W at these
    # interleaved indices. vLLM's Triton predicates must describe the same map.
    official_axes = [0] * (rotary_dim // 2)
    for index in range(1, rope["mrope_section"][1] * 3, 3):
        official_axes[index] = 1
    for index in range(2, rope["mrope_section"][2] * 3, 3):
        official_axes[index] = 2
    vllm_axes = []
    for index in range(rotary_dim // 2):
        is_height = index % 3 == 1 and index <= 3 * rope["mrope_section"][1]
        is_width = index % 3 == 2 and index <= 3 * rope["mrope_section"][2]
        vllm_axes.append(1 if is_height else 2 if is_width else 0)
    require(vllm_axes == official_axes, "vLLM/Transformers interleaved axis map differs")
    require([vllm_axes.count(axis) for axis in range(3)] == [11, 11, 10], "axis counts drift")

    input_tokens, features, placements = build_interleaved_prompt(config)
    actual_positions, actual_delta = (
        Qwen3_5ForConditionalGeneration._get_mrope_input_positions(
            input_tokens=input_tokens,
            mm_features=features,
            config=config,
        )
    )
    expected_positions, expected_delta = official_image_reference(
        len(input_tokens), placements
    )
    require(
        torch.equal(actual_positions, expected_positions),
        "vLLM image MRoPE positions differ from released Qwen3.5 semantics",
    )
    require(actual_delta == expected_delta, "vLLM image MRoPE delta differs")

    text_only_length = 257
    text_positions, text_delta = (
        Qwen3_5ForConditionalGeneration._get_mrope_input_positions(
            input_tokens=[42] * text_only_length,
            mm_features=[],
            config=config,
        )
    )
    expected_text = torch.arange(text_only_length, dtype=torch.long).expand(3, -1)
    require(torch.equal(text_positions, expected_text), "text-only positions are not 1D")
    require(text_delta == 0, "text-only MRoPE delta must be zero")

    continuation = np.empty((3, 17), dtype=np.int64)
    MRotaryEmbedding.get_next_input_positions_tensor(
        out=continuation,
        out_offset=0,
        mrope_position_delta=actual_delta,
        context_len=len(input_tokens),
        num_new_tokens=17,
    )
    expected_start = int(actual_positions.max().item() + 1)
    require(
        np.array_equal(
            continuation,
            np.arange(expected_start, expected_start + 17, dtype=np.int64)[None, :]
            .repeat(3, axis=0),
        ),
        "generated-token positions do not continue after the multimodal prompt",
    )

    slot_bytes = 388
    raw_k8v4_bytes = (
        slot_bytes
        * text.num_key_value_heads
        * len(actual_full_layers)
        * text.max_position_embeddings
    )
    require(raw_k8v4_bytes == 6_509_559_808, "native K8V4 byte derivation drift")
    require(math.isclose(raw_k8v4_bytes / 2**30, 6.0625), "native K8V4 GiB drift")
    print(
        "PASS qwen38_context: native=262144 layers=64 full_layers=16 "
        "head_dim=256 rotary_dim=64 mrope=[11,11,10]/interleaved "
        f"images=3 prompt_tokens={len(input_tokens)} delta={actual_delta} "
        "raw_k8v4=6509559808B/6.0625GiB"
    )


if __name__ == "__main__":
    main()
