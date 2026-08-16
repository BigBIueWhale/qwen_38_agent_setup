#!/usr/bin/env python3
"""Dependency-free assertions for the pinned Qwen3.8 media contract.

This runs inside the immutable Docker build and deliberately uses the installed
vLLM/Pillow/Jinja dependencies. It does not use a host Python environment.
"""

from __future__ import annotations

import asyncio
import os
from io import BytesIO
from types import SimpleNamespace

os.environ["VLLM_QWEN38_STRICT_IMAGE_CONTRACT"] = "1"
os.environ["VLLM_MAX_IMAGE_PIXELS"] = "16777216"

from PIL import Image

from vllm.entrypoints.anthropic.serving import AnthropicServingMessages
from vllm.entrypoints.chat_utils import _parse_chat_message_content_mm_part
from vllm.exceptions import VLLMUnprocessableEntityError, VLLMValidationError
from vllm.multimodal.media import ImageMediaIO, MediaConnector
from vllm.renderers.params import ChatParams


def expect(error_type, text: str, fn) -> None:
    try:
        fn()
    except error_type as error:
        assert text in str(error), (text, str(error))
    else:
        raise AssertionError(f"expected {error_type.__name__} containing {text!r}")


def png_bytes(
    mode: str = "RGB",
    size: tuple[int, int] = (8, 8),
    color=None,
) -> bytes:
    if color is None:
        color = (10, 20, 30, 255) if mode == "RGBA" else (10, 20, 30)
    with BytesIO() as output:
        Image.new(mode, size, color).save(output, format="PNG")
        return output.getvalue()


def test_decoder() -> str:
    image_io = ImageMediaIO()
    rgb_bytes = png_bytes()
    assert image_io.load_bytes(rgb_bytes).media.getpixel((0, 0)) == (10, 20, 30)
    rgba = image_io.load_bytes(png_bytes("RGBA", color=(0, 0, 0, 0))).media
    assert rgba.mode == "RGB"
    assert rgba.getpixel((0, 0)) == (255, 255, 255)

    expect(ValueError, "requires image_mode='RGB'", lambda: ImageMediaIO(None))
    expect(
        ValueError,
        "pinned white",
        lambda: ImageMediaIO(rgba_background_color=(0, 0, 0)),
    )
    expect(
        ValueError,
        "media type 'image/png'",
        lambda: image_io.load_base64("image/jpeg", "AA=="),
    )

    with BytesIO() as output:
        Image.new("RGB", (8, 8)).save(output, format="JPEG")
        jpeg_bytes = output.getvalue()
    expect(ValueError, "only decoded PNG", lambda: image_io.load_bytes(jpeg_bytes))
    expect(
        ValueError,
        "only 8-bit RGB or RGBA",
        lambda: image_io.load_bytes(png_bytes("L", color=128)),
    )
    assert image_io.load_bytes(png_bytes(size=(30, 1))).media.size == (30, 1)
    assert image_io.load_bytes(png_bytes(size=(1, 30))).media.size == (1, 30)
    expect(
        ValueError,
        "aspect ratio <= 30:1",
        lambda: image_io.load_bytes(png_bytes(size=(31, 1))),
    )
    expect(
        ValueError,
        "aspect ratio <= 30:1",
        lambda: image_io.load_bytes(png_bytes(size=(1, 31))),
    )

    with BytesIO() as output:
        first = Image.new("RGB", (8, 8), (1, 2, 3))
        second = Image.new("RGB", (8, 8), (3, 2, 1))
        first.save(
            output,
            format="PNG",
            save_all=True,
            append_images=[second],
            duration=100,
            loop=0,
        )
        animated = output.getvalue()
    expect(ValueError, "single-frame PNG", lambda: image_io.load_bytes(animated))
    return rgb_bytes.hex()


async def test_connector(rgb_hex: str) -> None:
    import base64

    encoded = base64.b64encode(bytes.fromhex(rgb_hex)).decode("ascii")
    canonical = f"data:image/png;base64,{encoded}"
    connector = MediaConnector()
    assert connector.fetch_image(canonical).size == (8, 8)
    assert (await connector.fetch_image_async(canonical)).size == (8, 8)

    for forbidden in (
        "https://example.test/image.png",
        "file:///tmp/image.png",
        f"data:image/png;charset=utf-8;base64,{encoded}",
        f"data:image/jpeg;base64,{encoded}",
    ):
        expect(
            VLLMUnprocessableEntityError,
            "inline, lossless PNG",
            lambda forbidden=forbidden: connector.fetch_image(forbidden),
        )
        try:
            await connector.fetch_image_async(forbidden)
        except VLLMUnprocessableEntityError as error:
            assert "inline, lossless PNG" in str(error)
        else:
            raise AssertionError(f"async connector accepted {forbidden!r}")


def test_request_surfaces() -> None:
    canonical = {
        "type": "image_url",
        "image_url": {
            "url": "data:image/png;base64,AAAA",
            "detail": "high",
        },
    }
    assert _parse_chat_message_content_mm_part(canonical) == (
        "image_url",
        "data:image/png;base64,AAAA",
    )
    expect(
        VLLMValidationError,
        "explicit content-part",
        lambda: _parse_chat_message_content_mm_part(
            {"image_url": "data:image/png;base64,AAAA"}
        ),
    )
    expect(
        VLLMValidationError,
        "UUIDs are forbidden",
        lambda: _parse_chat_message_content_mm_part(canonical | {"uuid": "trusted"}),
    )
    low = {
        "type": "image_url",
        "image_url": {
            "url": "data:image/png;base64,AAAA",
            "detail": "low",
        },
    }
    expect(
        VLLMValidationError,
        "'low'",
        lambda: _parse_chat_message_content_mm_part(low),
    )
    expect(
        VLLMValidationError,
        "video, audio",
        lambda: _parse_chat_message_content_mm_part(
            {"type": "video_url", "video_url": {"url": "https://example.test"}}
        ),
    )

    expect(
        VLLMValidationError,
        "media_io_kwargs",
        lambda: ChatParams(
            media_io_kwargs={"image": {"image_mode": None}}
        ).with_defaults(),
    )
    expect(
        VLLMValidationError,
        "mm_processor_kwargs",
        lambda: ChatParams(
            mm_processor_kwargs={"size": {"longest_edge": 1}}
        ).with_defaults(),
    )
    expect(
        VLLMValidationError,
        "add_vision_id",
        lambda: ChatParams(
            chat_template_kwargs={"add_vision_id": True}
        ).with_defaults(),
    )
    assert (
        ChatParams(
            chat_template_kwargs={"add_vision_id": False}
        ).with_defaults().chat_template_kwargs
        == {"add_vision_id": False}
    )


def test_anthropic_tool_image_interleave() -> None:
    block = SimpleNamespace(
        tool_use_id="call_001",
        content=[
            {"type": "text", "text": "before"},
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": "AAAA",
                },
            },
            {"type": "text", "text": "after"},
        ],
    )
    messages: list[dict] = []
    AnthropicServingMessages._convert_user_tool_result(block, messages)
    assert messages == [
        {
            "role": "tool",
            "tool_call_id": "call_001",
            "content": [
                {"type": "text", "text": "before"},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,AAAA"},
                },
                {"type": "text", "text": "after"},
            ],
        }
    ]


def main() -> None:
    rgb_hex = test_decoder()
    asyncio.run(test_connector(rgb_hex))
    test_request_surfaces()
    test_anthropic_tool_image_interleave()
    print("vision-contract-unit: PASS")


if __name__ == "__main__":
    main()
