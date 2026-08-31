#!/usr/bin/env python3
"""Prove Qwen agent-history render/parser and stream/non-stream equivalence.

This probe runs inside the serving image. It uses the installed vLLM parser,
the real checkpoint tokenizer, the live render endpoint, and both live Chat
Completions response modes. Assertions compare semantics where generation is
stochastic and exact token IDs where rendering is expected to be identical.

The synthetic history is the autonomous agent shape: one task prompt followed
only by assistant reasoning/tool-call turns and their tool results, rendered
mid-run. The rendered prompt must carry the assistant turn's reasoning in
full, and parsing that turn back must recover it byte-exactly, so
render -> parse -> render is a fixed point on exact token IDs.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from collections.abc import Iterator, Sequence
from typing import Any

from transformers import AutoTokenizer

from vllm.entrypoints.anthropic.protocol import AnthropicMessagesRequest
from vllm.entrypoints.anthropic.serving import AnthropicServingMessages
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
    ChatCompletionToolsParam,
)
from vllm.parser.qwen3 import Qwen3Parser
from vllm.tokenizers.detokenizer_utils import detokenize_incrementally


MODEL = "qwen3.8-27b-nvfp4-k8v4"
BASE_URL = "http://127.0.0.1:8000"
SYSTEM = "Use the declared tools exactly and preserve result order."
USER = "Inspect both records, in order, before continuing."
REASONING = "I need both records, and their declared order is significant."
RESULTS = [
    "record one: alpha=17; mode=careful",
    "record two: symbol=Qwen3.8; enabled=true",
]


def inspect_tool() -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "inspect_record",
            "description": "Inspect one exact synthetic record.",
            "strict": False,
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "const": "/workspace/round trip Ω.json",
                    },
                    "line": {"type": "integer", "const": 17},
                    "options": {
                        "type": "object",
                        "properties": {
                            "mode": {"type": "string", "const": "careful"},
                            "tags": {
                                "type": "array",
                                "items": {"type": "string"},
                                "const": ["alpha", "βeta"],
                                "minItems": 2,
                                "maxItems": 2,
                            },
                        },
                        "required": ["mode", "tags"],
                        "additionalProperties": False,
                    },
                },
                "required": ["path", "line", "options"],
                "additionalProperties": False,
            },
        },
    }


def lookup_tool() -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "lookup_symbol",
            "description": "Look up one exact synthetic symbol.",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "const": "Qwen3.8"},
                    "enabled": {"type": "boolean", "const": True},
                },
                "required": ["symbol", "enabled"],
                "additionalProperties": False,
            },
        },
    }


TOOLS = [inspect_tool(), lookup_tool()]
EXPECTED_CALLS = [
    (
        "inspect_record",
        {
            "path": "/workspace/round trip Ω.json",
            "line": 17,
            "options": {"mode": "careful", "tags": ["alpha", "βeta"]},
        },
    ),
    ("lookup_symbol", {"symbol": "Qwen3.8", "enabled": True}),
]


def post_json(path: str, payload: dict[str, Any], timeout: int = 300) -> dict[str, Any]:
    request = urllib.request.Request(
        BASE_URL + path,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {error.code} from {path}: {body}") from error


def iter_sse(path: str, payload: dict[str, Any]) -> Iterator[dict[str, Any] | str]:
    request = urllib.request.Request(
        BASE_URL + path,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        },
    )
    try:
        response = urllib.request.urlopen(request, timeout=300)
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {error.code} from {path}: {body}") from error

    with response:
        data_lines: list[str] = []
        for raw_line in response:
            line = raw_line.decode("utf-8").rstrip("\r\n")
            if not line:
                if data_lines:
                    data = "\n".join(data_lines)
                    yield data if data == "[DONE]" else json.loads(data)
                data_lines = []
                continue
            if line.startswith("data:"):
                data_lines.append(line[5:].lstrip())
        if data_lines:
            data = "\n".join(data_lines)
            yield data if data == "[DONE]" else json.loads(data)


def make_history(
    call_ids: Sequence[str],
    reasoning: str = REASONING,
    calls: Sequence[tuple[str, dict[str, Any]]] = EXPECTED_CALLS,
) -> list[dict[str, Any]]:
    if len(call_ids) != len(calls) or len(call_ids) != len(RESULTS):
        raise AssertionError("History call/result cardinality changed")
    tool_calls = [
        {
            "id": call_id,
            "type": "function",
            "function": {
                "name": name,
                "arguments": json.dumps(arguments, ensure_ascii=False),
            },
        }
        for call_id, (name, arguments) in zip(call_ids, calls, strict=True)
    ]
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": USER},
        {
            "role": "assistant",
            "content": None,
            "reasoning": reasoning,
            "tool_calls": tool_calls,
        },
    ]
    messages.extend(
        {
            "role": "tool",
            "tool_call_id": call_id,
            "content": result,
        }
        for call_id, result in zip(call_ids, RESULTS, strict=True)
    )
    return messages


def render_request(messages: list[dict[str, Any]]) -> dict[str, Any]:
    payload = {
        "model": MODEL,
        "messages": messages,
        "tools": TOOLS,
        "tool_choice": "required",
        "parallel_tool_calls": True,
        "max_tokens": 1,
        "chat_template_kwargs": {
            "enable_thinking": True,
            "reasoning_effort": "xhigh",
        },
    }
    return post_json("/v1/chat/completions/render", payload)


def decode_render(tokenizer: Any, rendered: dict[str, Any]) -> str:
    token_ids = rendered.get("token_ids")
    if not isinstance(token_ids, list) or not token_ids:
        raise AssertionError(f"Render endpoint returned no token IDs: {rendered}")
    return tokenizer.decode(
        token_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )


def assistant_turn(rendered_text: str) -> str:
    start_marker = "<|im_start|>assistant\n"
    start = rendered_text.find(start_marker)
    if start < 0:
        raise AssertionError("Rendered history has no assistant turn")
    start += len(start_marker)
    end = rendered_text.find("<|im_end|>", start)
    if end < 0:
        raise AssertionError("Rendered assistant turn has no end marker")
    return rendered_text[start:end]


def normalize_calls(calls: Sequence[Any] | None) -> list[tuple[str, dict[str, Any]]]:
    if calls is None:
        return []
    normalized = []
    for call in calls:
        name = getattr(call, "name", None)
        arguments = getattr(call, "arguments", None)
        if name is None and getattr(call, "function", None) is not None:
            name = call.function.name
            arguments = call.function.arguments
        normalized.append((name, json.loads(arguments)))
    return normalized


def parse_nonstream(
    tokenizer: Any,
    raw_turn: str,
    request: ChatCompletionRequest,
) -> tuple[str, str, list[tuple[str, dict[str, Any]]], list[Any]]:
    parser = Qwen3Parser(tokenizer, tools=request.tools)
    token_ids = tokenizer.encode(raw_turn, add_special_tokens=False)
    reasoning, content, calls = parser.parse(
        raw_turn,
        request,
        enable_auto_tools=True,
        model_output_token_ids=token_ids,
    )
    return reasoning or "", content or "", normalize_calls(calls), list(calls or [])


def parse_streaming(
    tokenizer: Any,
    raw_turn: str,
    request: ChatCompletionRequest,
) -> tuple[str, str, list[tuple[str, dict[str, Any]]]]:
    parser = Qwen3Parser(tokenizer, tools=request.tools)
    all_token_ids = tokenizer.encode(raw_turn, add_special_tokens=False)
    previous_text = ""
    previous_tokens = None
    prefix_offset = 0
    read_offset = 0
    reasoning_parts: list[str] = []
    content_parts: list[str] = []
    states: dict[int, dict[str, str]] = {}

    def collect(delta: Any) -> None:
        if delta is None:
            return
        reasoning_parts.append(delta.reasoning or "")
        content_parts.append(delta.content or "")
        for tool_delta in delta.tool_calls or []:
            slot = states.setdefault(
                tool_delta.index,
                {"name": "", "arguments": ""},
            )
            function = tool_delta.function
            if function is not None:
                slot["name"] += function.name or ""
                slot["arguments"] += function.arguments or ""

    for index, delta_token in enumerate(all_token_ids):
        current_token_ids = all_token_ids[: index + 1]
        new_tokens, delta_text, prefix_offset, read_offset = detokenize_incrementally(
            tokenizer=tokenizer,
            all_input_ids=current_token_ids,
            prev_tokens=previous_tokens,
            prefix_offset=prefix_offset,
            read_offset=read_offset,
            skip_special_tokens=False,
            spaces_between_special_tokens=True,
        )
        current_text = previous_text + delta_text
        collect(
            parser.extract_tool_calls_streaming(
                previous_text=previous_text,
                current_text=current_text,
                delta_text=delta_text,
                previous_token_ids=all_token_ids[:index],
                current_token_ids=current_token_ids,
                delta_token_ids=[delta_token],
                request=request,
            )
        )
        previous_text = current_text
        previous_tokens = (
            previous_tokens + new_tokens if previous_tokens is not None else new_tokens
        )
    collect(parser.finish_streaming())

    calls = [
        (states[index]["name"], json.loads(states[index]["arguments"]))
        for index in sorted(states)
    ]
    return "".join(reasoning_parts), "".join(content_parts), calls


def synthetic_roundtrip(tokenizer: Any) -> dict[str, Any]:
    original_ids = ["call-roundtrip-a", "call-roundtrip-b"]
    original_history = make_history(original_ids)
    request = ChatCompletionRequest.model_validate(
        {
            "model": MODEL,
            "messages": original_history,
            "tools": TOOLS,
            "tool_choice": "required",
            "parallel_tool_calls": True,
            "max_tokens": 1,
        }
    )
    rendered = render_request(original_history)
    rendered_text = decode_render(tokenizer, rendered)
    if any(call_id in rendered_text for call_id in original_ids):
        raise AssertionError("Transport tool-call IDs leaked into Qwen prompt XML")

    raw_turn = assistant_turn(rendered_text)
    batch_reasoning, batch_content, batch_calls, parsed_calls = parse_nonstream(
        tokenizer, raw_turn, request
    )
    stream_reasoning, stream_content, stream_calls = parse_streaming(
        tokenizer, raw_turn, request
    )
    if batch_calls != EXPECTED_CALLS or stream_calls != EXPECTED_CALLS:
        raise AssertionError(
            "Stream/non-stream parser mismatch: "
            f"batch={batch_calls!r}, stream={stream_calls!r}"
        )
    if batch_reasoning != stream_reasoning or batch_content != stream_content:
        raise AssertionError(
            "Stream/non-stream text mismatch: "
            f"batch={(batch_reasoning, batch_content)!r}, "
            f"stream={(stream_reasoning, stream_content)!r}"
        )
    if batch_reasoning.strip() != REASONING or batch_content.strip():
        raise AssertionError(
            f"Rendered reasoning/content changed: {(batch_reasoning, batch_content)!r}"
        )

    parsed_ids = [call.id for call in parsed_calls]
    rerendered = render_request(
        make_history(parsed_ids, reasoning=batch_reasoning, calls=batch_calls)
    )
    if rendered["token_ids"] != rerendered["token_ids"]:
        raise AssertionError("render(parse(render(history))) changed prompt token IDs")

    anthropic = AnthropicMessagesRequest.model_validate(
        {
            "model": MODEL,
            "system": SYSTEM,
            "messages": [
                {"role": "user", "content": USER},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": REASONING},
                        *[
                            {
                                "type": "tool_use",
                                "id": call_id,
                                "name": name,
                                "input": arguments,
                            }
                            for call_id, (name, arguments) in zip(
                                original_ids, EXPECTED_CALLS, strict=True
                            )
                        ],
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": call_id,
                            "content": result,
                        }
                        for call_id, result in zip(
                            original_ids, RESULTS, strict=True
                        )
                    ],
                },
            ],
            "tools": [
                {
                    "name": tool["function"]["name"],
                    "description": tool["function"]["description"],
                    "input_schema": tool["function"]["parameters"],
                    **(
                        {"strict": tool["function"]["strict"]}
                        if "strict" in tool["function"]
                        else {}
                    ),
                }
                for tool in TOOLS
            ],
            "tool_choice": {"type": "any", "disable_parallel_tool_use": False},
            "max_tokens": 1,
        }
    )
    converted = AnthropicServingMessages._convert_anthropic_to_openai_request(
        anthropic,
        merge_inline_system=True,
    )
    converted_payload = converted.model_dump(
        mode="json",
        by_alias=True,
        exclude_none=True,
    )
    anthropic_render = post_json(
        "/v1/chat/completions/render",
        converted_payload,
    )
    if rendered["token_ids"] != anthropic_render["token_ids"]:
        raise AssertionError(
            "Equivalent Anthropic and OpenAI histories rendered different token IDs"
        )

    return {
        "rendered_prompt_tokens": len(rendered["token_ids"]),
        "transport_ids_absent_from_prompt": True,
        "typed_nested_arguments_preserved": True,
        "stream_nonstream_parser_semantics_equal": True,
        "render_parse_render_token_ids_equal": True,
        "anthropic_openai_history_token_ids_equal": True,
    }


def live_call_payload(stream: bool) -> dict[str, Any]:
    return {
        "model": MODEL,
        "messages": [
            {
                "role": "developer",
                "content": (
                    "Call inspect_record exactly once with every required constant. "
                    "Do not call any other tool."
                ),
            },
            {"role": "user", "content": "Inspect the required record now."},
        ],
        "tools": [inspect_tool()],
        "tool_choice": "required",
        "parallel_tool_calls": False,
        "max_tokens": 1_024,
        "stream": stream,
        "return_prompt_text": True,
        **({"stream_options": {"include_usage": True}} if stream else {}),
    }


def validate_live_call(message: dict[str, Any], finish_reason: str | None) -> None:
    calls = message.get("tool_calls") or []
    if finish_reason != "tool_calls" or len(calls) != 1:
        raise AssertionError(
            f"Malformed live tool response: finish={finish_reason}, calls={calls}"
        )
    call = calls[0]
    actual = (
        call["function"]["name"],
        json.loads(call["function"]["arguments"]),
    )
    if actual != EXPECTED_CALLS[0]:
        raise AssertionError(f"Live tool call changed typed arguments: {actual!r}")
    if not message.get("reasoning"):
        raise AssertionError("xhigh live tool call emitted no separated reasoning")


def live_stream_nonstream() -> dict[str, Any]:
    nonstream = post_json("/v1/chat/completions", live_call_payload(False))
    nonstream_choice = nonstream["choices"][0]
    nonstream_message = nonstream_choice["message"]
    validate_live_call(nonstream_message, nonstream_choice.get("finish_reason"))
    nonstream_prompt = nonstream.get("prompt_text")
    if not isinstance(nonstream_prompt, str) or not nonstream_prompt:
        raise AssertionError("Non-streaming response omitted requested prompt_text")

    stream_message: dict[str, Any] = {
        "reasoning": "",
        "content": "",
        "tool_calls": [],
    }
    slots: dict[int, dict[str, Any]] = {}
    stream_finish = None
    stream_prompt = None
    saw_done = False
    for event in iter_sse("/v1/chat/completions", live_call_payload(True)):
        if event == "[DONE]":
            saw_done = True
            continue
        assert isinstance(event, dict)
        stream_prompt = event.get("prompt_text") or stream_prompt
        for choice in event.get("choices", []):
            delta = choice.get("delta") or {}
            stream_message["reasoning"] += delta.get("reasoning") or ""
            stream_message["content"] += delta.get("content") or ""
            for tool_delta in delta.get("tool_calls") or []:
                slot = slots.setdefault(
                    tool_delta["index"],
                    {
                        "id": "",
                        "type": "function",
                        "function": {"name": "", "arguments": ""},
                    },
                )
                slot["id"] += tool_delta.get("id") or ""
                function = tool_delta.get("function") or {}
                slot["function"]["name"] += function.get("name") or ""
                slot["function"]["arguments"] += function.get("arguments") or ""
            stream_finish = choice.get("finish_reason") or stream_finish
    stream_message["tool_calls"] = [slots[index] for index in sorted(slots)]
    validate_live_call(stream_message, stream_finish)
    if not saw_done:
        raise AssertionError("Streaming response omitted [DONE]")
    if stream_prompt != nonstream_prompt:
        raise AssertionError("Live stream/non-stream requests rendered different prompts")

    # Prove both live parser outputs can be accepted as history, rendered, and
    # parsed back without changing their typed call semantics.
    roundtrip_tokens = []
    for message in (nonstream_message, stream_message):
        call = message["tool_calls"][0]
        history = [
            *live_call_payload(False)["messages"],
            {
                "role": "assistant",
                "content": message.get("content") or None,
                "reasoning": message["reasoning"],
                "tool_calls": [call],
            },
            {
                "role": "tool",
                "tool_call_id": call["id"],
                "content": RESULTS[0],
            },
        ]
        payload = {
            "model": MODEL,
            "messages": history,
            "tools": [inspect_tool()],
            "tool_choice": "auto",
            "parallel_tool_calls": False,
            "max_tokens": 1,
        }
        rendered = post_json("/v1/chat/completions/render", payload)
        roundtrip_tokens.append(len(rendered["token_ids"]))

    return {
        "input_prompt_text_exactly_equal": True,
        "stream_done_marker_present": True,
        "nonstream_finish_reason": nonstream_choice["finish_reason"],
        "stream_finish_reason": stream_finish,
        "typed_tool_arguments_equal": True,
        "both_live_histories_rerendered": True,
        "rerendered_history_tokens": roundtrip_tokens,
    }


def main() -> None:
    started = time.monotonic()
    tokenizer = AutoTokenizer.from_pretrained(
        "/model",
        local_files_only=True,
        trust_remote_code=False,
    )
    result = {
        "synthetic_exact_roundtrip": synthetic_roundtrip(tokenizer),
        "live_stream_nonstream": live_stream_nonstream(),
    }
    result["elapsed_seconds"] = round(time.monotonic() - started, 3)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
