#!/usr/bin/env python3
"""Prove chronological tool-image rendering and practical cache reuse.

The probe runs inside the pinned runtime image.  It uses a unique lossless
4096x4096 PNG, the real tokenizer, the live render and inference endpoints,
both OpenAI Chat and Anthropic Messages, and authoritative vLLM counters.
No output-byte determinism is assumed; the semantic invariant is retrieval of
an exact value that exists only in the historical tool-result image.
"""

from __future__ import annotations

import base64
import io
import json
import time
import urllib.error
import urllib.request
import uuid
from collections.abc import Iterator
from typing import Any

from PIL import Image, ImageDraw, ImageFont
from transformers import AutoTokenizer

from vllm.entrypoints.anthropic.protocol import AnthropicMessagesRequest
from vllm.entrypoints.anthropic.serving import AnthropicServingMessages


BASE_URL = "http://127.0.0.1:8000"
MODEL = "qwen3.8-27b-nvfp4-k8v4"
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"
IMAGE_SIZE = 4096
OLD_REASONING = "OLD_HIDDEN_TRACE_MUST_NOT_BE_REPLAYED_7B91"
BEFORE = "BEGIN_ORIGINATING_TOOL_IMAGE_5C17"
AFTER = "END_ORIGINATING_TOOL_IMAGE_8A42"
ACK = "The tool result has been received; I will use its pixels when asked."
QUESTION = (
    "Copy the VISUAL-CODE from the screenshot returned by the earlier tool. "
    "Return the code exactly."
)
ANTHROPIC_HEADERS = {"anthropic-version": "2023-06-01"}


def make_image(
    code: str, color: tuple[int, int, int], nonce: str
) -> tuple[str, str]:
    image = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), color)
    # Make the encoded image bytes unique on every probe run without turning
    # the semantic assertion into a long random-string OCR benchmark.
    image.putpixel((0, 0), tuple(bytes.fromhex(nonce[:6])))
    draw = ImageDraw.Draw(image)
    draw.rectangle((96, 96, 4000, 4000), outline=(10, 18, 32), width=32)
    draw.rectangle((224, 1050, 3872, 3000), fill=(10, 18, 32))
    heading = ImageFont.truetype(FONT_PATH, 154)
    value_font = ImageFont.truetype(FONT_PATH, 250)
    draw.text((380, 450), "ORIGINATING TOOL RESULT", font=heading, fill=(20, 25, 35))
    label = f"VISUAL-CODE={code}"
    box = draw.textbbox((0, 0), label, font=value_font)
    draw.text(
        ((IMAGE_SIZE - (box[2] - box[0])) // 2, 1850),
        label,
        font=value_font,
        fill=(255, 235, 70),
    )
    draw.text(
        (680, 3300),
        "READ PIXELS; DO NOT INFER",
        font=heading,
        fill=(70, 15, 100),
    )
    output = io.BytesIO()
    image.save(output, format="PNG", compress_level=1, optimize=False)
    raw = output.getvalue()
    return "data:image/png;base64," + base64.b64encode(raw).decode("ascii"), str(
        len(raw)
    )


def tool_call() -> dict[str, Any]:
    return {
        "id": "call_origin_image_001",
        "type": "function",
        "function": {"name": "capture_screen", "arguments": "{}"},
    }


def openai_messages(data_url: str, *, moved: bool = False) -> list[dict[str, Any]]:
    image_part = {"type": "image_url", "image_url": {"url": data_url}}
    tool_content: str | list[dict[str, Any]]
    final_content: str | list[dict[str, Any]]
    if moved:
        tool_content = f"{BEFORE}\n{AFTER}"
        final_content = [
            {
                "type": "text",
                "text": "A client incorrectly moved this old tool image here:",
            },
            image_part,
            {"type": "text", "text": QUESTION},
        ]
    else:
        tool_content = [
            {"type": "text", "text": BEFORE},
            image_part,
            {"type": "text", "text": AFTER},
        ]
        final_content = QUESTION
    return [
        {
            "role": "system",
            "content": (
                "Images returned by tools are evidence owned by that exact tool "
                "response. Read their pixels when a later turn refers to them."
            ),
        },
        {"role": "user", "content": "Capture the synthetic screen now."},
        {
            "role": "assistant",
            "content": None,
            "reasoning": OLD_REASONING,
            "tool_calls": [tool_call()],
        },
        {
            "role": "tool",
            "tool_call_id": tool_call()["id"],
            "content": tool_content,
        },
        {"role": "assistant", "content": ACK},
        {"role": "user", "content": final_content},
    ]


def anthropic_payload(data_url: str, *, stream: bool, salt: str) -> dict[str, Any]:
    encoded = data_url.split(",", 1)[1]
    return {
        "model": MODEL,
        "system": (
            "Images returned by tools are evidence owned by that exact tool "
            "response. Read their pixels when a later turn refers to them."
        ),
        "messages": [
            {"role": "user", "content": "Capture the synthetic screen now."},
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": OLD_REASONING},
                    {
                        "type": "tool_use",
                        "id": tool_call()["id"],
                        "name": "capture_screen",
                        "input": {},
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_call()["id"],
                        "content": [
                            {"type": "text", "text": BEFORE},
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": encoded,
                                },
                            },
                            {"type": "text", "text": AFTER},
                        ],
                    }
                ],
            },
            {"role": "assistant", "content": ACK},
            {"role": "user", "content": QUESTION},
        ],
        "max_tokens": 16_384,
        "cache_salt": salt,
        "stream": stream,
    }


def openai_payload(
    data_url: str, *, stream: bool, salt: str, moved: bool = False
) -> dict[str, Any]:
    payload = {
        "model": MODEL,
        "messages": openai_messages(data_url, moved=moved),
        "max_tokens": 16_384,
        "cache_salt": salt,
        "stream": stream,
    }
    if stream:
        payload["stream_options"] = {"include_usage": True}
    return payload


def request_json(
    path: str,
    payload: dict[str, Any] | None = None,
    *,
    headers: dict[str, str] | None = None,
    timeout: int = 900,
) -> dict[str, Any] | str:
    request = urllib.request.Request(
        BASE_URL + path,
        data=(
            json.dumps(payload, ensure_ascii=False).encode("utf-8")
            if payload is not None
            else None
        ),
        headers={"Content-Type": "application/json", **(headers or {})},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {error.code} from {path}: {body}") from error
    return json.loads(body) if payload is not None else body


def iter_sse(
    path: str,
    payload: dict[str, Any],
    headers: dict[str, str] | None = None,
) -> Iterator[tuple[float, str | None, dict[str, Any] | str]]:
    request = urllib.request.Request(
        BASE_URL + path,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            **(headers or {}),
        },
    )
    started = time.monotonic()
    try:
        response = urllib.request.urlopen(request, timeout=900)
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {error.code} from {path}: {body}") from error
    with response:
        event_name: str | None = None
        data_lines: list[str] = []
        for raw_line in response:
            line = raw_line.decode("utf-8").rstrip("\r\n")
            if not line:
                if data_lines:
                    data = "\n".join(data_lines)
                    yield (
                        time.monotonic() - started,
                        event_name,
                        data if data == "[DONE]" else json.loads(data),
                    )
                event_name = None
                data_lines = []
                continue
            if line.startswith(":"):
                continue
            if line.startswith("event:"):
                event_name = line[6:].strip()
            elif line.startswith("data:"):
                data_lines.append(line[5:].lstrip())
        if data_lines:
            data = "\n".join(data_lines)
            yield (
                time.monotonic() - started,
                event_name,
                data if data == "[DONE]" else json.loads(data),
            )


def metric_snapshot() -> dict[str, int]:
    body = request_json("/metrics")
    assert isinstance(body, str)
    wanted = {
        "prefix_queries": "vllm:prefix_cache_queries_total",
        "prefix_hits": "vllm:prefix_cache_hits_total",
        "computed": "vllm:prompt_tokens_by_source_total",
        "mm_queries": "vllm:mm_cache_queries_total",
        "mm_hits": "vllm:mm_cache_hits_total",
    }
    values: dict[str, int] = {}
    for line in body.splitlines():
        for key, metric in wanted.items():
            if not line.startswith(metric + "{"):
                continue
            if f'model_name="{MODEL}"' not in line:
                continue
            if key == "computed" and 'source="local_compute"' not in line:
                continue
            values[key] = int(float(line.rsplit(" ", 1)[1]))
    if set(values) != set(wanted):
        raise AssertionError(f"Required cache metrics are missing: {values}")
    return values


def metric_delta(before: dict[str, int], after: dict[str, int]) -> dict[str, int]:
    return {key: after[key] - before[key] for key in before}


def validate_delta(delta: dict[str, int], prompt_tokens: int) -> None:
    if delta["prefix_queries"] != prompt_tokens:
        raise AssertionError(
            f"Concurrent or missing metric attribution: {delta}, prompt={prompt_tokens}"
        )
    if delta["prefix_hits"] + delta["computed"] != prompt_tokens:
        raise AssertionError(f"Prompt source accounting does not close: {delta}")
    if delta["mm_queries"] != 1:
        raise AssertionError(f"Expected exactly one image cache query: {delta}")


def render_proof(
    tokenizer: Any, data_url: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    openai_render = request_json(
        "/v1/chat/completions/render",
        {
            "model": MODEL,
            "messages": openai_messages(data_url),
            "max_tokens": 1,
        },
    )
    assert isinstance(openai_render, dict)

    anthropic_request = AnthropicMessagesRequest.model_validate(
        anthropic_payload(data_url, stream=False, salt="render-only")
    )
    converted = AnthropicServingMessages._convert_anthropic_to_openai_request(
        anthropic_request, merge_inline_system=True
    )
    anthropic_render = request_json(
        "/v1/chat/completions/render",
        converted.model_dump(mode="json", by_alias=True, exclude_none=True)
        | {"max_tokens": 1},
    )
    assert isinstance(anthropic_render, dict)
    if openai_render["token_ids"] != anthropic_render["token_ids"]:
        raise AssertionError(
            "Equivalent OpenAI and Anthropic image histories rendered different IDs"
        )

    decoded = tokenizer.decode(
        openai_render["token_ids"],
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    vision = "<|vision_start|>"
    if decoded.count(vision) != 1:
        raise AssertionError(f"Expected one vision marker, got: {decoded.count(vision)}")
    positions = {
        "tool_start": decoded.index("<tool_response>"),
        "before": decoded.index(BEFORE),
        "vision": decoded.index(vision),
        "after": decoded.index(AFTER),
        "tool_end": decoded.index("</tool_response>"),
        "ack": decoded.index(ACK),
        "question": decoded.index(QUESTION),
    }
    if list(positions.values()) != sorted(positions.values()):
        raise AssertionError(f"Image/history chronology changed: {positions}")
    if OLD_REASONING in decoded:
        raise AssertionError("Default preserve_thinking=false replayed old reasoning")
    if "Reasoning effort is set to xhigh." not in decoded:
        raise AssertionError("Omitted reasoning controls did not resolve to xhigh")

    moved_render = request_json(
        "/v1/chat/completions/render",
        {
            "model": MODEL,
            "messages": openai_messages(data_url, moved=True),
            "max_tokens": 1,
        },
    )
    assert isinstance(moved_render, dict)
    if moved_render["token_ids"] == openai_render["token_ids"]:
        raise AssertionError("Moving tool media to the latest user turn changed no IDs")
    moved_decoded = tokenizer.decode(
        moved_render["token_ids"],
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    if moved_decoded.index(vision) < moved_decoded.index("</tool_response>"):
        raise AssertionError("Moved-media negative control unexpectedly stayed in tool result")
    return (
        {
            "prompt_tokens": len(openai_render["token_ids"]),
            "openai_anthropic_token_ids_equal": True,
            "historical_reasoning_omitted_by_default": True,
            "xhigh_resolved_by_default": True,
            "chronological_positions": positions,
        },
        {
            "prompt_tokens": len(moved_render["token_ids"]),
            "token_ids_differ": True,
            "vision_after_tool_response": True,
        },
    )


def collect_openai_stream(payload: dict[str, Any], code: str) -> dict[str, Any]:
    content = ""
    reasoning = ""
    finish_reason = None
    usage = None
    first_semantic = None
    terminal = None
    saw_done = False
    for elapsed, _, event in iter_sse("/v1/chat/completions", payload):
        terminal = elapsed
        if event == "[DONE]":
            saw_done = True
            continue
        assert isinstance(event, dict)
        usage = event.get("usage") or usage
        for choice in event.get("choices", []):
            delta = choice.get("delta") or {}
            semantic = (delta.get("reasoning") or "") + (delta.get("content") or "")
            if semantic and first_semantic is None:
                first_semantic = elapsed
            reasoning += delta.get("reasoning") or ""
            content += delta.get("content") or ""
            finish_reason = choice.get("finish_reason") or finish_reason
    if not saw_done or finish_reason != "stop" or code not in content or not usage:
        raise AssertionError(
            f"OpenAI stream failed: done={saw_done}, finish={finish_reason}, "
            f"code={code in content}, usage={usage}, answer={content!r}"
        )
    return {
        "ttft_seconds": first_semantic,
        "terminal_seconds": terminal,
        "prompt_tokens": int(usage["prompt_tokens"]),
        "completion_tokens": int(usage["completion_tokens"]),
        "reasoning_present": bool(reasoning),
        "answer": content,
    }


def collect_anthropic_stream(payload: dict[str, Any], code: str) -> dict[str, Any]:
    text = ""
    thinking = ""
    stop_reason = None
    first_semantic = None
    terminal = None
    input_tokens = None
    output_tokens = None
    saw_stop = False
    for elapsed, event_name, event in iter_sse(
        "/v1/messages", payload, ANTHROPIC_HEADERS
    ):
        terminal = elapsed
        if event == "[DONE]":
            continue
        assert isinstance(event, dict)
        event_type = event.get("type") or event_name
        if event_type == "error":
            raise AssertionError(f"Anthropic stream error: {event}")
        if event_type == "message_start":
            input_tokens = ((event.get("message") or {}).get("usage") or {}).get(
                "input_tokens"
            )
        elif event_type == "content_block_delta":
            delta = event.get("delta") or {}
            piece = delta.get("thinking") or delta.get("text") or ""
            if piece and first_semantic is None:
                first_semantic = elapsed
            thinking += delta.get("thinking") or ""
            text += delta.get("text") or ""
        elif event_type == "message_delta":
            stop_reason = (event.get("delta") or {}).get("stop_reason") or stop_reason
            output_tokens = (event.get("usage") or {}).get("output_tokens")
        elif event_type == "message_stop":
            saw_stop = True
    if not saw_stop or stop_reason != "end_turn" or code not in text:
        raise AssertionError(
            f"Anthropic stream failed: stop={saw_stop}, reason={stop_reason}, "
            f"code={code in text}, answer={text!r}"
        )
    return {
        "ttft_seconds": first_semantic,
        "terminal_seconds": terminal,
        "prompt_tokens": int(input_tokens),
        "completion_tokens": int(output_tokens),
        "reasoning_present": bool(thinking),
        "answer": text,
    }


def nonstream_openai(payload: dict[str, Any], code: str) -> dict[str, Any]:
    started = time.monotonic()
    response = request_json("/v1/chat/completions", payload)
    elapsed = time.monotonic() - started
    assert isinstance(response, dict)
    choice = response["choices"][0]
    answer = choice["message"].get("content") or ""
    if choice.get("finish_reason") != "stop" or code not in answer:
        raise AssertionError(f"OpenAI non-stream failed: {response}")
    return {
        "elapsed_seconds": elapsed,
        "prompt_tokens": int(response["usage"]["prompt_tokens"]),
        "completion_tokens": int(response["usage"]["completion_tokens"]),
        "reasoning_present": bool(choice["message"].get("reasoning")),
        "answer": answer,
    }


def nonstream_anthropic(payload: dict[str, Any], code: str) -> dict[str, Any]:
    started = time.monotonic()
    response = request_json(
        "/v1/messages", payload, headers=ANTHROPIC_HEADERS
    )
    elapsed = time.monotonic() - started
    assert isinstance(response, dict)
    answer = "".join(
        block.get("text", "")
        for block in response["content"]
        if block.get("type") == "text"
    )
    if response.get("stop_reason") != "end_turn" or code not in answer:
        raise AssertionError(f"Anthropic non-stream failed: {response}")
    return {
        "elapsed_seconds": elapsed,
        "prompt_tokens": int(response["usage"]["input_tokens"]),
        "completion_tokens": int(response["usage"]["output_tokens"]),
        "reasoning_present": any(
            block.get("type") == "thinking" for block in response["content"]
        ),
        "answer": answer,
    }


def measured(fn, before: dict[str, int]) -> tuple[dict[str, Any], dict[str, int]]:
    result = fn()
    after = metric_snapshot()
    delta = metric_delta(before, after)
    validate_delta(delta, result["prompt_tokens"])
    result["cache_delta"] = delta
    return result, after


def main() -> None:
    run = uuid.uuid4().hex
    code = "VX-2749"
    changed_code = "CY-7247"
    data_url, png_bytes = make_image(code, (228, 239, 248), run)
    changed_url, changed_png_bytes = make_image(
        changed_code, (245, 229, 218), run[6:]
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "/model", local_files_only=True, trust_remote_code=False
    )

    render, moved_render = render_proof(tokenizer, data_url)
    salt = "vision-history-cache-" + uuid.uuid4().hex

    snapshot = metric_snapshot()
    cold_stream, snapshot = measured(
        lambda: collect_openai_stream(
            openai_payload(data_url, stream=True, salt=salt), code
        ),
        snapshot,
    )
    warm_nonstream, snapshot = measured(
        lambda: nonstream_openai(
            openai_payload(data_url, stream=False, salt=salt), code
        ),
        snapshot,
    )
    cross_protocol_stream, snapshot = measured(
        lambda: collect_anthropic_stream(
            anthropic_payload(data_url, stream=True, salt=salt), code
        ),
        snapshot,
    )
    cross_protocol_nonstream, snapshot = measured(
        lambda: nonstream_anthropic(
            anthropic_payload(data_url, stream=False, salt=salt), code
        ),
        snapshot,
    )
    changed_control, snapshot = measured(
        lambda: collect_openai_stream(
            openai_payload(
                changed_url,
                stream=True,
                salt=salt + "-changed-bytes",
            ),
            changed_code,
        ),
        snapshot,
    )
    moved_control, _ = measured(
        lambda: collect_openai_stream(
            openai_payload(data_url, stream=True, salt=salt, moved=True), code
        ),
        snapshot,
    )

    if cold_stream["cache_delta"]["prefix_hits"] != 0:
        raise AssertionError(f"Salted cold prompt unexpectedly hit KV: {cold_stream}")
    if cold_stream["cache_delta"]["mm_hits"] != 0:
        raise AssertionError(f"Unique cold image unexpectedly hit MM cache: {cold_stream}")
    for name, result in (
        ("warm OpenAI non-stream", warm_nonstream),
        ("cross-protocol Anthropic stream", cross_protocol_stream),
        ("cross-protocol Anthropic non-stream", cross_protocol_nonstream),
    ):
        if result["cache_delta"]["prefix_hits"] <= 0:
            raise AssertionError(f"{name} had no real prefix-cache hit: {result}")
        if result["cache_delta"]["mm_hits"] != 1:
            raise AssertionError(f"{name} had no image preprocessing-cache hit: {result}")
    if changed_control["cache_delta"]["mm_hits"] != 0:
        raise AssertionError("Changed image bytes incorrectly reused an MM cache entry")
    if moved_control["cache_delta"]["mm_hits"] != 1:
        raise AssertionError("Same moved image bytes did not reuse preprocessing cache")
    if moved_control["cache_delta"]["prefix_hits"] >= moved_control["prompt_tokens"]:
        raise AssertionError("Moved image incorrectly reused the complete prompt prefix")

    cold_ttft = float(cold_stream["ttft_seconds"])
    warm_ttft = float(cross_protocol_stream["ttft_seconds"])
    if warm_ttft >= cold_ttft:
        raise AssertionError(
            f"Cached chronological history had no TTFT improvement: "
            f"cold={cold_ttft}, warm={warm_ttft}"
        )

    print(
        json.dumps(
            {
                "status": "passed",
                "code": code,
                "png_bytes": int(png_bytes),
                "changed_png_bytes": int(changed_png_bytes),
                "render": render,
                "moved_render_control": moved_render,
                "openai_stream_cold": cold_stream,
                "openai_nonstream_warm": warm_nonstream,
                "anthropic_stream_warm": cross_protocol_stream,
                "anthropic_nonstream_warm": cross_protocol_nonstream,
                "changed_image_control": changed_control,
                "moved_image_control": moved_control,
                "cross_protocol_ttft_speedup": round(cold_ttft / warm_ttft, 3),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
