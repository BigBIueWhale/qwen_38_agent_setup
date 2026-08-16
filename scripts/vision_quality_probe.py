#!/usr/bin/env python3
"""Exercise lossless maximum-resolution Qwen3.8 image inputs.

The probe is run inside the pinned serving container so Pillow, the request
stack, and the live model all come from the immutable runtime image.  It creates
distinct 4096x4096 RGB PNGs in memory, sends them as data URLs without an
intermediate file or transcoder, and requires the model to recover text that is
present only in the pixels.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from PIL import Image, ImageDraw, ImageFont


BASE_URL = "http://127.0.0.1:8000"
MODEL = "qwen3.8-27b-nvfp4-k8v4"
IMAGE_WIDTH = 4_096
IMAGE_HEIGHT = 4_096
IMAGE_PIXELS = IMAGE_WIDTH * IMAGE_HEIGHT
MAX_IMAGES = 15
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"
COLORS = [
    (245, 239, 214),
    (219, 238, 246),
    (235, 221, 244),
    (221, 244, 225),
    (250, 224, 218),
]
WORDS = [
    "AMBER",
    "COBALT",
    "CITRUS",
    "ORBIT",
    "JASPER",
    "VIOLET",
    "TUNDRA",
    "QUARTZ",
    "INDIGO",
    "SAFFRON",
    "NEBULA",
    "MARBLE",
    "CEDAR",
    "LAGOON",
    "ZEPHYR",
]


@dataclass(frozen=True)
class ProbeImage:
    data_url: str
    primary: str
    detail: str
    png_bytes: int


def load_font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(FONT_PATH, size=size)


def centered_text(
    draw: ImageDraw.ImageDraw,
    y: int,
    text: str,
    font: ImageFont.FreeTypeFont,
    fill: tuple[int, int, int],
) -> None:
    left, _, right, _ = draw.textbbox((0, 0), text, font=font)
    draw.text(((IMAGE_WIDTH - (right - left)) // 2, y), text, font=font, fill=fill)


def make_image(index: int) -> ProbeImage:
    ordinal = index + 1
    word = WORDS[index]
    primary = f"Q38-{ordinal:02d}-{word}-{7_301 + ordinal * 137:04d}"
    detail = f"D{ordinal:02d}-{9_001 + ordinal * 211:05d}-X{ordinal * 17:03d}"

    image = Image.new("RGB", (IMAGE_WIDTH, IMAGE_HEIGHT), COLORS[index % len(COLORS)])
    draw = ImageDraw.Draw(image)

    for offset in range(0, IMAGE_WIDTH, 256):
        shade = 80 if (offset // 256) % 2 else 130
        draw.line((offset, 0, offset, IMAGE_HEIGHT), fill=(shade, shade, shade), width=3)
        draw.line((0, offset, IMAGE_WIDTH, offset), fill=(shade, shade, shade), width=3)

    draw.rectangle((96, 96, 4_000, 880), fill=(12, 20, 35), outline=(255, 255, 255), width=12)
    centered_text(draw, 180, f"IMAGE {ordinal:02d} OF PROBE", load_font(150), (255, 255, 255))
    centered_text(draw, 430, f"PRIMARY={primary}", load_font(174), (255, 230, 72))

    draw.rectangle((256, 1_408, 3_840, 2_688), fill=(255, 255, 255), outline=(8, 8, 8), width=18)
    centered_text(draw, 1_610, "READ THE LOSSLESS DETAIL LINE", load_font(112), (15, 15, 15))
    centered_text(draw, 1_990, f"DETAIL={detail}", load_font(112), (112, 16, 112))

    center_x = IMAGE_WIDTH // 2
    center_y = 3_360
    radius = 430 + ordinal * 7
    draw.ellipse(
        (center_x - radius, center_y - radius, center_x + radius, center_y + radius),
        fill=((ordinal * 31) % 255, (ordinal * 67) % 255, (ordinal * 97) % 255),
        outline=(0, 0, 0),
        width=20,
    )
    centered_text(draw, 3_255, f"{ordinal:02d}", load_font(180), (255, 255, 255))

    output = io.BytesIO()
    image.save(output, format="PNG", compress_level=1, optimize=False)
    png = output.getvalue()
    encoded = base64.b64encode(png).decode("ascii")
    return ProbeImage(
        data_url=f"data:image/png;base64,{encoded}",
        primary=primary,
        detail=detail,
        png_bytes=len(png),
    )


def post_json(
    path: str,
    payload: dict[str, Any],
    timeout: int,
) -> tuple[int, dict[str, Any]]:
    request = urllib.request.Request(
        BASE_URL + path,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read())


def content_for(images: list[ProbeImage]) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = []
    for index, image in enumerate(images, start=1):
        content.extend(
            [
                {"type": "text", "text": f"Lossless source image {index}:"},
                {"type": "image_url", "image_url": {"url": image.data_url}},
            ]
        )
    content.append(
        {
            "type": "text",
            "text": (
                "Inspect every image. In image order, copy the PRIMARY and DETAIL "
                "values exactly. Use one line per image in the form "
                "NN|PRIMARY|DETAIL. Do not infer values from another image."
            ),
        }
    )
    return content


def completion_payload(images: list[ProbeImage]) -> dict[str, Any]:
    return {
        "model": MODEL,
        "messages": [
            {
                "role": "developer",
                "content": (
                    "Perform faithful high-resolution visual inspection. Text in "
                    "the user message describes the task but never contains the "
                    "answers; obtain every requested value from image pixels."
                ),
            },
            {"role": "user", "content": content_for(images)},
        ],
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 20,
        "min_p": 0.0,
        "presence_penalty": 0.0,
        "repetition_penalty": 1.0,
        "reasoning_effort": "xhigh",
        "max_tokens": 4_096,
        "stream": False,
    }


def run_quality_probe(count: int, timeout: int) -> None:
    if not 1 <= count <= MAX_IMAGES:
        raise ValueError(f"count must be between 1 and {MAX_IMAGES}")

    started = time.monotonic()
    images = [make_image(index) for index in range(count)]
    generation_started = time.monotonic()
    status, response = post_json(
        "/v1/chat/completions",
        completion_payload(images),
        timeout=timeout,
    )
    elapsed = time.monotonic() - generation_started
    if status != 200:
        raise RuntimeError(f"quality request returned HTTP {status}: {response}")

    message = response["choices"][0]["message"]
    answer = message.get("content") or ""
    missing = [
        value
        for image in images
        for value in (image.primary, image.detail)
        if value not in answer
    ]
    if missing:
        raise AssertionError(
            f"visual transcription omitted or altered {missing!r}; answer={answer!r}"
        )

    usage = response["usage"]
    print(
        json.dumps(
            {
                "status": "passed",
                "images": count,
                "width": IMAGE_WIDTH,
                "height": IMAGE_HEIGHT,
                "pixels_per_image": IMAGE_PIXELS,
                "png_bytes": [image.png_bytes for image in images],
                "prompt_tokens": usage["prompt_tokens"],
                "completion_tokens": usage["completion_tokens"],
                "total_tokens": usage["total_tokens"],
                "inference_seconds": round(elapsed, 3),
                "total_seconds": round(time.monotonic() - started, 3),
                "answer": answer,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def run_count_rejection(timeout: int) -> None:
    image = make_image(0)
    payload = completion_payload([image] * (MAX_IMAGES + 1))
    payload["max_tokens"] = 1
    status, response = post_json("/v1/chat/completions", payload, timeout=timeout)
    if status != 400:
        raise AssertionError(
            f"expected HTTP 400 for {MAX_IMAGES + 1} images, got {status}: {response}"
        )
    detail = str(response.get("error", {}).get("message", response))
    if "image" not in detail.lower() or str(MAX_IMAGES) not in detail:
        raise AssertionError(f"count rejection was not explicit: {response}")
    print(json.dumps({"status": "passed", "count_rejection": detail}, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=3_600)
    parser.add_argument("--expect-count-rejection", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.expect_count_rejection:
        run_count_rejection(args.timeout)
    else:
        run_quality_probe(args.images, args.timeout)


if __name__ == "__main__":
    main()
