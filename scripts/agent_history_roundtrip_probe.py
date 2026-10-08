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

import argparse
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
from vllm.tokenizers.detokenizer_utils import NativeDecodeStream

from probe_parser import served_parser
from probe_scope import new_conversation


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


# One dict for both renders compared below, so that comparison tests the
# Anthropic/OpenAI conversion rather than the coincidence of two requests
# carrying different kwargs. Retention is absent deliberately: the served
# template enforces it and accepts no other value, so a probe that named it
# would be restating a guarantee rather than testing one.
TEMPLATE_KWARGS: dict[str, Any] = {
    "enable_thinking": True,
    "reasoning_effort": "xhigh",
}


def render_payload(messages: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "model": MODEL,
        "messages": messages,
        "tools": TOOLS,
        "tool_choice": "required",
        "parallel_tool_calls": True,
        "max_tokens": 1,
        "chat_template_kwargs": dict(TEMPLATE_KWARGS),
    }


def render_request(messages: list[dict[str, Any]]) -> dict[str, Any]:
    return post_json("/v1/chat/completions/render", render_payload(messages))


def generated_turn(
    tokenizer: Any,
    prompt: dict[str, Any],
    rendered: dict[str, Any],
) -> tuple[list[int], list[int], list[str]]:
    """The assistant turn as serving hands it to the parser.

    *prompt* renders the history the turn answers, with the generation prompt;
    the full history must begin with exactly those ids. The turn is the ids
    after them through the end-of-turn token, which the model generates and
    the detokenizer gives no text; the rest is decoded incrementally from the
    prompt, as the served detokenizer does.
    """
    prompt_ids, history_ids = prompt["token_ids"], rendered["token_ids"]
    if history_ids[: len(prompt_ids)] != prompt_ids:
        raise AssertionError(
            "The rendered history does not begin with the prompt its assistant "
            "turn was generated after"
        )
    end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    turn = history_ids[len(prompt_ids) :]
    if end not in turn:
        raise AssertionError("Rendered assistant turn has no end-of-turn token")
    turn = turn[: turn.index(end) + 1]
    decoder = NativeDecodeStream(
        tokenizer.backend_tokenizer, ids=list(prompt_ids), skip_special_tokens=False
    )
    texts = [decoder.step(token) or "" for token in turn[:-1]] + [""]
    return list(prompt_ids), turn, texts


def decode_render(tokenizer: Any, rendered: dict[str, Any]) -> str:
    token_ids = rendered.get("token_ids")
    if not isinstance(token_ids, list) or not token_ids:
        raise AssertionError(f"Render endpoint returned no token IDs: {rendered}")
    return tokenizer.decode(
        token_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )


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
    prompt_ids: list[int],
    turn_ids: list[int],
    texts: list[str],
    request: ChatCompletionRequest,
) -> tuple[str, str, list[tuple[str, dict[str, Any]]], list[Any]]:
    """The served batch path: parse_output on the generated text and ids,
    starting where the prompt leaves reasoning."""
    parser = served_parser(tokenizer, request.tools)
    reasoning, content, calls = parser.parse_output(
        "".join(texts),
        request,
        prompt_token_ids=prompt_ids,
        finish_reason="stop",
        stop_reason=None,
        enable_auto_tools=True,
        model_output_token_ids=turn_ids,
    )
    return reasoning or "", content or "", normalize_calls(calls), list(calls or [])


def parse_streaming(
    tokenizer: Any,
    prompt_ids: list[int],
    turn_ids: list[int],
    texts: list[str],
    request: ChatCompletionRequest,
    chunk_size: int,
) -> tuple[str, str, list[tuple[str, dict[str, Any]]]]:
    """The served stream path: parse_output_delta per engine output, given the
    prompt, with the finish reason on the last delta."""
    parser = served_parser(tokenizer, request.tools)
    reasoning_parts: list[str] = []
    content_parts: list[str] = []
    states: dict[int, dict[str, str]] = {}
    for start in range(0, len(turn_ids), chunk_size):
        last = start + chunk_size >= len(turn_ids)
        delta = parser.parse_output_delta(
            delta_text="".join(texts[start : start + chunk_size]),
            delta_token_ids=turn_ids[start : start + chunk_size],
            request=request,
            prompt_token_ids=prompt_ids,
            finish_reason="stop" if last else None,
            stop_reason=None,
        )
        if delta is None:
            continue
        reasoning_parts.append(delta.reasoning or "")
        content_parts.append(delta.content or "")
        for tool_delta in delta.tool_calls or []:
            slot = states.setdefault(tool_delta.index, {"name": "", "arguments": ""})
            if tool_delta.function is not None:
                slot["name"] += tool_delta.function.name or ""
                slot["arguments"] += tool_delta.function.arguments or ""
    calls = [
        (states[index]["name"], json.loads(states[index]["arguments"]))
        for index in sorted(states)
    ]
    return "".join(reasoning_parts), "".join(content_parts), calls


def anthropic_render_payload(call_ids: Sequence[str]) -> dict[str, Any]:
    """The history as an Anthropic Messages request, converted as the Messages
    route converts it, as a render request."""
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
                                call_ids, EXPECTED_CALLS, strict=True
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
                            call_ids, RESULTS, strict=True
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
            # A Messages request is a generation request and names its line of
            # work; the render it is converted into allocates no KV and names
            # none.
            "kv_scope": new_conversation("anthropic-history-render"),
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
        exclude={"kv_scope"},
    )
    # The Anthropic request carries no template kwargs of its own, so without
    # this the converted side would inherit the deployment default while the
    # OpenAI side states one, and the round trip's token-ID comparison would
    # be measuring that difference instead of the conversion.
    converted_payload["chat_template_kwargs"] = dict(TEMPLATE_KWARGS)
    return converted_payload


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

    # The turn is parsed as the model generated it: after the prompt that
    # renders the history before it, which ends with the template's opener.
    prompt_ids, turn_ids, texts = generated_turn(
        tokenizer, render_request(original_history[:2]), rendered
    )
    batch_reasoning, batch_content, batch_calls, parsed_calls = parse_nonstream(
        tokenizer, prompt_ids, turn_ids, texts, request
    )
    if batch_calls != EXPECTED_CALLS:
        raise AssertionError(f"Batch parse changed the calls: {batch_calls!r}")
    # Every engine chunking must give the batch answer, byte for byte.
    for chunk_size in (1, 2, 3, 7, len(turn_ids)):
        streamed = parse_streaming(
            tokenizer, prompt_ids, turn_ids, texts, request, chunk_size
        )
        if streamed != (batch_reasoning, batch_content, batch_calls):
            raise AssertionError(
                f"Stream (chunks of {chunk_size}) and batch parses differ: "
                f"batch={(batch_reasoning, batch_content, batch_calls)!r}, "
                f"stream={streamed!r}"
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

    converted_payload = anthropic_render_payload(original_ids)
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


LIVE_CALL_MESSAGES = [
    {
        "role": "developer",
        "content": (
            "Call inspect_record exactly once with every required constant. "
            "Do not call any other tool."
        ),
    },
    {"role": "user", "content": "Inspect the required record now."},
]


def live_call_payload(stream: bool, kv_scope: str) -> dict[str, Any]:
    return {
        "model": MODEL,
        "messages": LIVE_CALL_MESSAGES,
        "tools": [inspect_tool()],
        "tool_choice": "required",
        "parallel_tool_calls": False,
        "max_tokens": 1_024,
        "stream": stream,
        "return_prompt_text": True,
        "kv_scope": kv_scope,
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


def live_history_render_payload(message: dict[str, Any]) -> dict[str, Any]:
    """The live turn ``message`` as history, through its call's result, as a
    render request."""
    call = message["tool_calls"][0]
    history = [
        *LIVE_CALL_MESSAGES,
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
    return {
        "model": MODEL,
        "messages": history,
        "tools": [inspect_tool()],
        "tool_choice": "auto",
        "parallel_tool_calls": False,
        "max_tokens": 1,
    }


def live_stream_nonstream() -> dict[str, Any]:
    # The streamed request redraws the same turn, so both are one conversation.
    scope = new_conversation("live-tool-call")
    nonstream = post_json("/v1/chat/completions", live_call_payload(False, scope))
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
    for event in iter_sse("/v1/chat/completions", live_call_payload(True, scope)):
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
        rendered = post_json(
            "/v1/chat/completions/render", live_history_render_payload(message)
        )
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


def offline_requests() -> list[tuple[str, dict[str, Any] | None, str | None]]:
    """Each kind of request main() sends, built without a server."""
    call_ids = ["call-roundtrip-a", "call-roundtrip-b"]
    history = make_history(call_ids)
    scope = new_conversation("offline-live-tool-call")
    # The live history carries a parsed turn, of the shape the route returns.
    name, arguments = EXPECTED_CALLS[0]
    message = {
        "content": None,
        "reasoning": REASONING,
        "tool_calls": [
            {
                "id": "call_offline",
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(arguments)},
            }
        ],
    }
    return [
        ("/v1/chat/completions/render", render_payload(history), None),
        ("/v1/chat/completions/render", render_payload(history[:2]), None),
        ("/v1/chat/completions/render", anthropic_render_payload(call_ids), None),
        ("/v1/chat/completions", live_call_payload(False, scope), None),
        ("/v1/chat/completions", live_call_payload(True, scope), None),
        ("/v1/chat/completions/render", live_history_render_payload(message), None),
    ]


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
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
