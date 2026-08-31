#!/usr/bin/env python3
"""Prove high-pixel Qwen image processing across supported aspect ratios."""

from __future__ import annotations

import base64
import io
import json
import time
from dataclasses import dataclass
from typing import Any

from PIL import Image, ImageDraw, ImageFont
from transformers import AutoTokenizer

from vision_quality_probe import IMAGE_PIXELS, MODEL, post_json


FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"
SHAPES = (
    # Three fresh, independent trials in each orientation at the final proven
    # boundary.  Repeated 31:1 and 32:1 trials both produced genuine far-end
    # transcription errors, while 30:1 passed symmetrically.
    (22_080, 736),        # exact 30:1, 15,870 visual tokens
    (736, 22_080),
    (22_080, 736),
    (736, 22_080),
    (22_080, 736),
    (736, 22_080),
)
FIRST_CODES = (
    "27",
    "29",
    "74",
    "52",
    "47",
    "95",
)
SECOND_CODES = (
    "94",
    "72",
    "25",
    "49",
    "92",
    "57",
)
COLORS = (
    (232, 240, 250),
    (246, 235, 213),
    (226, 244, 229),
    (244, 224, 232),
    (234, 226, 246),
    (224, 242, 242),
    (248, 232, 218),
    (226, 232, 248),
    (240, 226, 238),
) * 2


@dataclass(frozen=True)
class AspectImage:
    width: int
    height: int
    first: str
    second: str
    data_url: str
    png_bytes: int


def font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(FONT_PATH, size=size)


def draw_centered(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    text: str,
    text_font: ImageFont.FreeTypeFont,
    fill: tuple[int, int, int],
) -> None:
    left, top, right, bottom = draw.textbbox((0, 0), text, font=text_font)
    text_width = right - left
    text_height = bottom - top
    box_left, box_top, box_right, box_bottom = box
    x = box_left + (box_right - box_left - text_width) // 2
    y = box_top + (box_bottom - box_top - text_height) // 2 - top
    draw.text((x, y), text, font=text_font, fill=fill)


def make_aspect_image(index: int, width: int, height: int) -> AspectImage:
    if width * height > IMAGE_PIXELS:
        raise AssertionError(f"shape {width}x{height} exceeds the pixel budget")
    if width % 32 or height % 32:
        raise AssertionError(f"shape {width}x{height} is not on Qwen's 32px grid")

    first = FIRST_CODES[index]
    second = SECOND_CODES[index]
    image = Image.new("RGB", (width, height), COLORS[index])
    draw = ImageDraw.Draw(image)
    short_edge = min(width, height)
    border = max(8, short_edge // 32)
    draw.rectangle(
        (border, border, width - border - 1, height - border - 1),
        outline=(12, 20, 35),
        width=border,
    )

    if width >= height:
        panel_width = min(width // 3, max(1_200, height * 3))
        text_font = font(min(240, max(112, height // 3)))
        left_box = (border * 3, border * 3, panel_width, height - border * 3)
        right_box = (
            width - panel_width,
            border * 3,
            width - border * 3,
            height - border * 3,
        )
        draw.rectangle(left_box, fill=(12, 20, 35))
        draw.rectangle(right_box, fill=(12, 20, 35))
        draw_centered(draw, left_box, first, text_font, (255, 238, 74))
        draw_centered(draw, right_box, second, text_font, (92, 244, 255))
    else:
        panel_height = min(height // 3, max(1_200, width * 16))
        top_box = (border * 3, border * 3, width - border * 3, panel_height)
        bottom_box = (
            border * 3,
            height - panel_height,
            width - border * 3,
            height - border * 3,
        )
        draw.rectangle(top_box, fill=(12, 20, 35))
        draw.rectangle(bottom_box, fill=(12, 20, 35))
        text_font = font(min(220, max(112, width // 3)))
        draw_centered(draw, top_box, first, text_font, (255, 238, 74))
        draw_centered(draw, bottom_box, second, text_font, (92, 244, 255))

    index_font = font(min(320, max(128, short_edge // 2)))
    center_box = (
        width // 3,
        height // 3,
        width - width // 3,
        height - height // 3,
    )
    draw_centered(draw, center_box, str(index + 1), index_font, (20, 30, 50))

    output = io.BytesIO()
    image.save(output, format="PNG", compress_level=1, optimize=False)
    png = output.getvalue()
    return AspectImage(
        width=width,
        height=height,
        first=first,
        second=second,
        data_url="data:image/png;base64," + base64.b64encode(png).decode("ascii"),
        png_bytes=len(png),
    )


def messages_for(images: list[AspectImage]) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = []
    for index, image in enumerate(images, start=1):
        content.extend(
            [
                {
                    "type": "text",
                    "text": (
                        f"Full-pixel image {index}, source shape "
                        f"{image.width}x{image.height}:"
                    ),
                },
                {"type": "image_url", "image_url": {"url": image.data_url}},
            ]
        )
    content.append(
        {
            "type": "text",
            "text": (
                "Inspect both distant dark panels in every image. Return one line "
                "per image as NN|FIRST|SECOND. Copy the codes exactly."
            ),
        }
    )
    return [
        {
            "role": "developer",
            "content": (
                "Use the image pixels, including both far ends of extreme aspect "
                "ratios. Never infer one image's code from another image."
            ),
        },
        {"role": "user", "content": content},
    ]


def serialized_count(tokenizer: Any, messages: list[dict[str, Any]]) -> int:
    encoded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=True,
        reasoning_effort="xhigh",
    )
    if hasattr(encoded, "keys"):
        return len(encoded["input_ids"])
    return len(encoded)


def live_count(messages: list[dict[str, Any]]) -> int:
    status, response = post_json(
        "/tokenize", {"model": MODEL, "messages": messages}, timeout=900
    )
    if status != 200:
        raise AssertionError(f"/tokenize returned HTTP {status}: {response}")
    return int(response["count"])


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(
        "/model", local_files_only=True, trust_remote_code=False
    )
    images = [
        make_aspect_image(index, width, height)
        for index, (width, height) in enumerate(SHAPES)
    ]

    token_proofs = []
    for image in images:
        one_image_messages = messages_for([image])
        serialized = serialized_count(tokenizer, one_image_messages)
        live = live_count(one_image_messages)
        image_tokens = live - serialized + 1
        expected_image_tokens = image.width * image.height // (32 * 32)
        if image_tokens != expected_image_tokens:
            raise AssertionError(
                f"{image.width}x{image.height} used {image_tokens} image tokens; "
                f"expected {expected_image_tokens} from the exact processor grid"
            )
        token_proofs.append(
            {
                "width": image.width,
                "height": image.height,
                "pixels": image.width * image.height,
                "pixel_budget_fraction": round(
                    image.width * image.height / IMAGE_PIXELS, 6
                ),
                "image_tokens": image_tokens,
            }
        )

    over_ratio = make_aspect_image(0, 21_824, 704)
    over_status, over_response = post_json(
        "/tokenize",
        {"model": MODEL, "messages": messages_for([over_ratio])},
        timeout=900,
    )
    if over_status != 400:
        raise AssertionError(
            f"aspect ratio above 30 returned HTTP {over_status}: {over_response}"
        )
    over_error = str((over_response.get("error") or {}).get("message") or "")
    if "aspect ratio" not in over_error.lower() or "30" not in over_error:
        raise AssertionError(f"aspect rejection was not explicit: {over_response}")

    inference_proofs = []
    started_all = time.monotonic()
    for index, image in enumerate(images, start=1):
        # Test each shape independently.  A single giant multi-image answer can
        # confound aspect-ratio perception with long cross-image enumeration.
        # Sampling remains Alibaba's server-side default; no deterministic-output
        # assumption is made.  The asserted invariant is direct pixel retrieval.
        payload = {
            "model": MODEL,
            "messages": messages_for([image]),
            # This is a vision-quality probe, not an output-budget probe.  A
            # 4,096-token test cap allowed one correct-looking xhigh reasoning
            # path to exhaust the budget before emitting its final answer.
            "max_tokens": 16_384,
            "cache_salt": (
                f"vision-v10-aspect-final-{index}-"
                f"{image.width}x{image.height}"
            ),
            "stream": False,
        }
        started = time.monotonic()
        status, response = post_json(
            "/v1/chat/completions", payload, timeout=3_600
        )
        elapsed = time.monotonic() - started
        if status != 200:
            raise AssertionError(
                f"aspect inference for {image.width}x{image.height} returned "
                f"HTTP {status}: {response}"
            )

        answer = response["choices"][0]["message"].get("content") or ""
        missing = [
            code for code in (image.first, image.second) if code not in answer
        ]
        inference_proofs.append(
            {
                "width": image.width,
                "height": image.height,
                "aspect_ratio": round(
                    max(image.width, image.height) / min(image.width, image.height),
                    6,
                ),
                "expected": [image.first, image.second],
                "passed": not missing,
                "missing": missing,
                "elapsed_seconds": round(elapsed, 3),
                "usage": response["usage"],
                "answer": answer,
            }
        )

    failed = [proof for proof in inference_proofs if not proof["passed"]]
    if failed:
        print(
            json.dumps(
                {
                    "status": "failed",
                    "source_transport": "lossless PNG data URLs",
                    "token_proofs": token_proofs,
                    "over_30_ratio_rejection": over_error,
                    "inference_proofs": inference_proofs,
                    "elapsed_seconds": round(time.monotonic() - started_all, 3),
                },
                indent=2,
            )
        )
        raise AssertionError(
            "one or more exact-grid aspect ratios failed direct far-end pixel "
            "retrieval; see the structured evidence above"
        )

    print(
        json.dumps(
            {
                "status": "passed",
                "source_transport": "lossless PNG data URLs",
                "token_proofs": token_proofs,
                "over_30_ratio_rejection": over_error,
                "png_bytes": [image.png_bytes for image in images],
                "inference_proofs": inference_proofs,
                "elapsed_seconds": round(time.monotonic() - started_all, 3),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
