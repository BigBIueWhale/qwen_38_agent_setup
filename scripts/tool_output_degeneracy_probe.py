#!/usr/bin/env python3
"""Fail-closed stream/non-stream checks for degenerate Qwen tool output.

This runs inside the pinned serving container so every parser check uses the
real checkpoint tokenizer and the exact installed vLLM code.  It covers both
controlled token replay and live max-token truncation across Chat Completions,
Anthropic Messages, and OpenAI Responses.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from collections.abc import Iterator, Sequence
from typing import Any

from transformers import AutoTokenizer

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
    ChatCompletionToolsParam,
)
from vllm.entrypoints.openai.responses.streaming_events import (
    SimpleStreamingEventProcessor,
    _StateType,
)
from vllm.parser.qwen3 import Qwen3Parser
from vllm.tokenizers.detokenizer_utils import detokenize_incrementally


BASE_URL = "http://127.0.0.1:8000"
MODEL = "qwen3.8-27b-nvfp4-k8v4"
ANTHROPIC_HEADERS = {"anthropic-version": "2023-06-01"}
LONG_VALUE = "".join(f"{index:04x}" for index in range(1_024))
BUDGET_CANDIDATES = (384, 512, 640, 768, 1_024)
MIN_CALIBRATION_TOOL_PREFIX_TOKENS = 192
FAULT_INJECTION_THINKING_BUDGET = 128


def post_json(
    path: str,
    payload: dict[str, Any],
    headers: dict[str, str] | None = None,
    *,
    timeout: int = 300,
) -> dict[str, Any]:
    request = urllib.request.Request(
        BASE_URL + path,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json", **(headers or {})},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {error.code} from {path}: {body}") from error


def iter_sse(
    path: str,
    payload: dict[str, Any],
    headers: dict[str, str] | None = None,
    *,
    timeout: int = 300,
) -> Iterator[tuple[str | None, dict[str, Any] | str]]:
    request = urllib.request.Request(
        BASE_URL + path,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            **(headers or {}),
        },
    )
    try:
        response = urllib.request.urlopen(request, timeout=timeout)
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
                    yield event_name, data if data == "[DONE]" else json.loads(data)
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
            yield event_name, data if data == "[DONE]" else json.loads(data)


def parser_tool() -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "configure",
            "description": "Apply one typed configuration.",
            "strict": False,
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "options": {
                        "type": "object",
                        "properties": {
                            "safe": {"type": "boolean"},
                            "retries": {"type": "integer"},
                        },
                        "required": ["safe", "retries"],
                        "additionalProperties": False,
                    },
                },
                "required": ["path", "options"],
                "additionalProperties": False,
            },
        },
    }


def normalize_calls(calls: Sequence[Any] | None) -> list[tuple[str, str]]:
    normalized: list[tuple[str, str]] = []
    for call in calls or []:
        name = getattr(call, "name", None)
        arguments = getattr(call, "arguments", None)
        function = getattr(call, "function", None)
        if name is None and function is not None:
            name = function.name
            arguments = function.arguments
        normalized.append((name or "", arguments or ""))
    return normalized


def parse_nonstream(
    tokenizer: Any,
    raw: str,
    token_ids: list[int],
    request: ChatCompletionRequest,
) -> tuple[str, str, list[tuple[str, str]]]:
    parser = Qwen3Parser(tokenizer, tools=request.tools)
    reasoning, content, calls = parser.parse(
        raw,
        request,
        enable_auto_tools=True,
        model_output_token_ids=token_ids,
    )
    return reasoning or "", content or "", normalize_calls(calls)


def parse_stream(
    tokenizer: Any,
    token_ids: list[int],
    request: ChatCompletionRequest,
) -> tuple[str, str, list[tuple[str, str]]]:
    parser = Qwen3Parser(tokenizer, tools=request.tools)
    previous_text = ""
    previous_tokens = None
    prefix_offset = 0
    read_offset = 0
    reasoning_parts: list[str] = []
    content_parts: list[str] = []
    slots: dict[int, dict[str, str]] = {}

    def collect(delta: Any) -> None:
        if delta is None:
            return
        reasoning_parts.append(delta.reasoning or "")
        content_parts.append(delta.content or "")
        for tool_delta in delta.tool_calls or []:
            slot = slots.setdefault(tool_delta.index, {"name": "", "arguments": ""})
            if tool_delta.function is not None:
                slot["name"] += tool_delta.function.name or ""
                slot["arguments"] += tool_delta.function.arguments or ""

    if not token_ids:
        collect(
            parser.parse_delta(
                delta_text="",
                delta_token_ids=[],
                request=request,
                prompt_token_ids=[],
                finished=True,
            )
        )
    for index, token_id in enumerate(token_ids):
        current_ids = token_ids[: index + 1]
        new_tokens, delta_text, prefix_offset, read_offset = detokenize_incrementally(
            tokenizer=tokenizer,
            all_input_ids=current_ids,
            prev_tokens=previous_tokens,
            prefix_offset=prefix_offset,
            read_offset=read_offset,
            skip_special_tokens=False,
            spaces_between_special_tokens=True,
        )
        collect(
            parser.parse_delta(
                delta_text=delta_text,
                delta_token_ids=[token_id],
                request=request,
                prompt_token_ids=[] if index == 0 else None,
                finished=index == len(token_ids) - 1,
            )
        )
        previous_text += delta_text
        previous_tokens = (
            previous_tokens + new_tokens if previous_tokens is not None else new_tokens
        )

    calls = [
        (slots[index]["name"], slots[index]["arguments"])
        for index in sorted(slots)
    ]
    return "".join(reasoning_parts), "".join(content_parts), calls


def controlled_parser_replay(tokenizer: Any) -> dict[str, Any]:
    tool = ChatCompletionToolsParam.model_validate(parser_tool())
    request = ChatCompletionRequest.model_validate(
        {
            "model": MODEL,
            "messages": [{"role": "user", "content": "Configure it."}],
            "tools": [parser_tool()],
            "tool_choice": "auto",
            "max_tokens": 16,
            "include_reasoning": True,
        }
    )
    valid = (
        "<think>inspect schema</think>"
        "<tool_call>\n<function=configure>\n"
        "<parameter=path>/workspace/a b.txt</parameter>\n"
        '<parameter=options>{"safe":true,"retries":2}</parameter>\n'
        "</function>\n</tool_call>"
    )
    second = valid.replace("/workspace/a b.txt", "/workspace/second.txt")
    cases = {
        "reasoning_without_close": "<think>still reasoning and then truncation",
        "partial_tool_trigger": "<think>done</think><tool_call>",
        "partial_function_name": "<tool_call>\n<function=config",
        "function_without_arguments": "<tool_call>\n<function=configure>",
        "partial_parameter": (
            "<tool_call>\n<function=configure>\n"
            "<parameter=path>/workspace/part"
        ),
        "missing_parameter_closer": (
            "<tool_call>\n<function=configure>\n"
            "<parameter=path>/workspace/a b.txt\n</function>"
        ),
        "complete_valid": valid,
        "unknown_tool": valid.replace("function=configure", "function=erase_disk"),
        "wrong_nested_type": valid.replace('"retries":2', '"retries":"two"'),
        "extra_nested_property": valid.replace(
            '"retries":2}', '"retries":2,"fallback":true}'
        ),
        "complete_then_trailing_content": valid + "AFTER_TOOL_TEXT",
        "complete_then_partial_second": valid + second[: second.index("=configure") + 5],
    }

    structural_ids = {
        tokenizer.get_vocab()[marker]
        for marker in ("<think>", "</think>", "<tool_call>", "</tool_call>")
    }
    checked_prefixes = 0
    for case_name, raw_case in cases.items():
        all_ids = tokenizer.encode(raw_case, add_special_tokens=False)
        # The corpus already names explicit malformed/truncated endpoints.  In
        # addition, the complete control is cut at evenly spaced positions and
        # immediately around each Qwen structural special token.  This covers
        # parser states without retesting every ordinary character token.
        prefix_lengths = {len(all_ids)}
        if case_name == "complete_valid":
            prefix_lengths.update(
                round(step * len(all_ids) / 12) for step in range(13)
            )
            for index, token_id in enumerate(all_ids):
                if token_id in structural_ids:
                    prefix_lengths.update(
                        (index, index + 1, min(index + 2, len(all_ids)))
                    )
        for prefix_length in sorted(prefix_lengths):
            prefix_ids = all_ids[:prefix_length]
            prefix_text = tokenizer.decode(
                prefix_ids,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            batch = parse_nonstream(tokenizer, prefix_text, prefix_ids, request)
            streamed = parse_stream(tokenizer, prefix_ids, request)
            if batch != streamed:
                raise AssertionError(
                    "Parser stream/non-stream mismatch for "
                    f"{case_name} at token prefix {prefix_length}/{len(all_ids)}: "
                    f"batch={batch!r}, stream={streamed!r}"
                )
            checked_prefixes += 1

    # Ensure the happy-path control did not pass merely because every parser
    # case collapsed to text.
    valid_ids = tokenizer.encode(valid, add_special_tokens=False)
    _, _, valid_calls = parse_nonstream(tokenizer, valid, valid_ids, request)
    if len(valid_calls) != 1 or valid_calls[0][0] != tool.function.name:
        raise AssertionError(f"Valid replay did not produce configure: {valid_calls}")
    if json.loads(valid_calls[0][1]) != {
        "path": "/workspace/a b.txt",
        "options": {"safe": True, "retries": 2},
    }:
        raise AssertionError(f"Valid replay changed typed arguments: {valid_calls}")

    return {
        "cases": list(cases),
        "real_token_prefixes_compared": checked_prefixes,
        "stream_nonstream_semantics_equal": True,
        "valid_nested_types_preserved": True,
    }


def long_openai_tool() -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "write_exact",
            "description": "Write the exact schema-provided value.",
            "strict": False,
            "parameters": {
                "type": "object",
                "properties": {
                    "value": {"type": "string", "const": LONG_VALUE},
                },
                "required": ["value"],
                "additionalProperties": False,
            },
        },
    }


def long_anthropic_tool() -> dict[str, Any]:
    function = long_openai_tool()["function"]
    return {
        "name": function["name"],
        "description": function["description"],
        "input_schema": function["parameters"],
        "strict": function["strict"],
    }


def long_responses_tool() -> dict[str, Any]:
    function = long_openai_tool()["function"]
    return {
        "type": "function",
        "name": function["name"],
        "description": function["description"],
        "parameters": function["parameters"],
        "strict": function["strict"],
    }


def chat_payload(tool_choice: str, budget: int, *, stream: bool) -> dict[str, Any]:
    return {
        "model": MODEL,
        "messages": [
            {
                "role": "developer",
                "content": "Call write_exact immediately. Do not explain.",
            },
            {"role": "user", "content": "Do it."},
        ],
        "tools": [long_openai_tool()],
        "tool_choice": tool_choice,
        "parallel_tool_calls": False,
        "max_tokens": budget,
        "temperature": 0,
        "reasoning_effort": "xhigh",
        "thinking_token_budget": FAULT_INJECTION_THINKING_BUDGET,
        "return_token_ids": True,
        "stream": stream,
        **({"stream_options": {"include_usage": True}} if stream else {}),
    }


def collect_chat_stream(payload: dict[str, Any]) -> dict[str, Any]:
    token_ids: list[int] = []
    calls: dict[int, dict[str, str]] = {}
    finish_reason = None
    usage: dict[str, Any] = {}
    saw_done = False
    for _, event in iter_sse("/v1/chat/completions", payload):
        if event == "[DONE]":
            saw_done = True
            continue
        assert isinstance(event, dict)
        usage = event.get("usage") or usage
        for choice in event.get("choices", []):
            token_ids.extend(choice.get("token_ids") or [])
            finish_reason = choice.get("finish_reason") or finish_reason
            for delta in (choice.get("delta") or {}).get("tool_calls") or []:
                slot = calls.setdefault(delta["index"], {"name": "", "arguments": ""})
                function = delta.get("function") or {}
                slot["name"] += function.get("name") or ""
                slot["arguments"] += function.get("arguments") or ""
    return {
        "finish_reason": finish_reason,
        "token_ids": token_ids,
        "calls": [calls[index] for index in sorted(calls)],
        "usage": usage,
        "done": saw_done,
    }


def assert_chat_truncation(
    tokenizer: Any,
    result: dict[str, Any],
    budget: int,
    label: str,
) -> dict[str, Any]:
    raw = tokenizer.decode(
        result["token_ids"],
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    calls = result["calls"]
    completion_tokens = result["usage"].get("completion_tokens")
    if completion_tokens != budget:
        raise AssertionError(
            f"{label} did not exhaust the controlled budget: "
            f"{completion_tokens} != {budget}"
        )
    if "<tool_call>" not in raw or "</tool_call>" in raw:
        raise AssertionError(
            f"{label} was not truncated inside a tool call; raw tail={raw[-200:]!r}"
        )
    if result["finish_reason"] != "length":
        raise AssertionError(
            f"{label} upgraded a truncated tool prefix to "
            f"{result['finish_reason']!r}"
        )
    if len(calls) != 1 or calls[0]["name"] != "write_exact":
        raise AssertionError(f"{label} lost its recoverable tool prefix: {calls}")
    parsed = json.loads(calls[0]["arguments"])
    if parsed.get("value") == LONG_VALUE:
        raise AssertionError(f"{label} unexpectedly reconstructed the entire constant")
    return {
        "finish_reason": result["finish_reason"],
        "completion_tokens": completion_tokens,
        "raw_tool_wrapper_incomplete": True,
        "recoverable_prefix_argument_chars": len(parsed.get("value", "")),
    }


def find_truncated_chat_budget(tokenizer: Any) -> tuple[int, dict[str, Any]]:
    diagnostics: list[str] = []
    for budget in BUDGET_CANDIDATES:
        response = post_json(
            "/v1/chat/completions",
            chat_payload("required", budget, stream=False),
        )
        choice = response["choices"][0]
        result = {
            "finish_reason": choice.get("finish_reason"),
            "token_ids": choice.get("token_ids") or [],
            "calls": [
                {
                    "name": call["function"]["name"],
                    "arguments": call["function"]["arguments"],
                }
                for call in choice["message"].get("tool_calls") or []
            ],
            "usage": response["usage"],
        }
        raw = tokenizer.decode(
            result["token_ids"],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        tool_prefix_tokens = 0
        if "<tool_call>" in raw:
            tool_prefix_tokens = len(
                tokenizer.encode(
                    raw[raw.index("<tool_call>") :],
                    add_special_tokens=False,
                )
            )
        if (
            response["usage"].get("completion_tokens") == budget
            and "<tool_call>" in raw
            and "</tool_call>" not in raw
            and result["calls"]
            and tool_prefix_tokens >= MIN_CALIBRATION_TOOL_PREFIX_TOKENS
        ):
            return budget, assert_chat_truncation(
                tokenizer, result, budget, "OpenAI required non-stream"
            )
        diagnostics.append(
            f"{budget}:finish={choice.get('finish_reason')},"
            f"tokens={response['usage'].get('completion_tokens')},"
            f"tool_start={'<tool_call>' in raw},tool_end={'</tool_call>' in raw},"
            f"calls={len(result['calls'])},tool_prefix_tokens={tool_prefix_tokens}"
        )
    raise AssertionError(
        "Could not establish a live in-tool truncation boundary: " + "; ".join(diagnostics)
    )


def live_chat_truncation(tokenizer: Any) -> tuple[int, dict[str, Any]]:
    budget, baseline = find_truncated_chat_budget(tokenizer)
    results: dict[str, Any] = {"required_nonstream": baseline}
    for tool_choice, stream in (
        ("required", True),
        ("auto", False),
        ("auto", True),
    ):
        label = f"OpenAI {tool_choice} {'stream' if stream else 'non-stream'}"
        if stream:
            result = collect_chat_stream(chat_payload(tool_choice, budget, stream=True))
            if not result["done"]:
                raise AssertionError(f"{label} omitted [DONE]")
        else:
            response = post_json(
                "/v1/chat/completions",
                chat_payload(tool_choice, budget, stream=False),
            )
            choice = response["choices"][0]
            result = {
                "finish_reason": choice.get("finish_reason"),
                "token_ids": choice.get("token_ids") or [],
                "calls": [
                    {
                        "name": call["function"]["name"],
                        "arguments": call["function"]["arguments"],
                    }
                    for call in choice["message"].get("tool_calls") or []
                ],
                "usage": response["usage"],
            }
        key = f"{tool_choice}_{'stream' if stream else 'nonstream'}"
        results[key] = assert_chat_truncation(tokenizer, result, budget, label)
    results["independent_generations_not_byte_compared"] = True
    results["all_terminal_reasons_length"] = True
    return budget, results


def anthropic_payload(budget: int, *, stream: bool) -> dict[str, Any]:
    return {
        "model": MODEL,
        "system": "Call write_exact immediately. Do not explain.",
        "messages": [{"role": "user", "content": "Do it."}],
        "tools": [long_anthropic_tool()],
        "tool_choice": {"type": "any", "disable_parallel_tool_use": True},
        "max_tokens": budget,
        "temperature": 0,
        "output_config": {"effort": "xhigh"},
        "thinking": {
            "type": "enabled",
            "budget_tokens": FAULT_INJECTION_THINKING_BUDGET,
        },
        "stream": stream,
    }


def live_anthropic_truncation(budget: int) -> dict[str, Any]:
    response = post_json(
        "/v1/messages",
        anthropic_payload(budget, stream=False),
        ANTHROPIC_HEADERS,
    )
    uses = [block for block in response["content"] if block.get("type") == "tool_use"]
    if response.get("stop_reason") != "max_tokens" or response["usage"].get(
        "output_tokens"
    ) != budget:
        raise AssertionError(f"Anthropic non-stream hid truncation: {response}")
    if len(uses) != 1 or uses[0].get("name") != "write_exact":
        raise AssertionError(f"Anthropic non-stream lost tool prefix: {response}")
    if uses[0].get("input", {}).get("value") == LONG_VALUE:
        raise AssertionError("Anthropic non-stream unexpectedly completed the long value")

    stop_reason = None
    output_tokens = None
    block_types: list[str] = []
    partial_json = ""
    saw_message_stop = False
    for event_name, event in iter_sse(
        "/v1/messages",
        anthropic_payload(budget, stream=True),
        ANTHROPIC_HEADERS,
    ):
        if event == "[DONE]":
            continue
        assert isinstance(event, dict)
        event_type = event.get("type") or event_name
        if event_type == "error":
            raise AssertionError(f"Anthropic stream error: {event}")
        if event_type == "content_block_start":
            block_types.append(event["content_block"]["type"])
        elif event_type == "content_block_delta":
            delta = event.get("delta") or {}
            if delta.get("type") == "input_json_delta":
                partial_json += delta.get("partial_json") or ""
        elif event_type == "message_delta":
            stop_reason = (event.get("delta") or {}).get("stop_reason") or stop_reason
            output_tokens = (event.get("usage") or {}).get("output_tokens", output_tokens)
        elif event_type == "message_stop":
            saw_message_stop = True
    if (
        stop_reason != "max_tokens"
        or output_tokens != budget
        or "tool_use" not in block_types
        or not saw_message_stop
    ):
        raise AssertionError(
            "Anthropic stream hid or malformed truncation: "
            f"stop={stop_reason}, tokens={output_tokens}, blocks={block_types}, "
            f"message_stop={saw_message_stop}"
        )
    parsed = json.loads(partial_json)
    if parsed.get("value") == LONG_VALUE:
        raise AssertionError("Anthropic stream unexpectedly completed the long value")
    return {
        "nonstream_stop_reason": response["stop_reason"],
        "stream_stop_reason": stop_reason,
        "output_tokens": budget,
        "tool_prefix_exposed_but_not_terminally_executable": True,
        "stream_message_stop": True,
    }


def responses_payload(budget: int, *, stream: bool) -> dict[str, Any]:
    return {
        "model": MODEL,
        "instructions": "Call write_exact immediately. Do not explain.",
        "input": "Do it.",
        "tools": [long_responses_tool()],
        "tool_choice": "required",
        "parallel_tool_calls": False,
        "max_output_tokens": budget,
        "temperature": 0,
        "stream": stream,
        "store": False,
        "reasoning": {"effort": "xhigh"},
        "thinking_token_budget": FAULT_INJECTION_THINKING_BUDGET,
"chat_template_kwargs": {
            "enable_thinking": True,
            "reasoning_effort": "xhigh",
        },
    }


def response_calls(response: dict[str, Any]) -> list[dict[str, Any]]:
    return [item for item in response.get("output", []) if item.get("type") == "function_call"]


def assert_incomplete_response(
    response: dict[str, Any], budget: int, label: str
) -> list[dict[str, Any]]:
    calls = response_calls(response)
    if response.get("status") != "incomplete":
        raise AssertionError(f"{label} status is not incomplete: {response}")
    if (response.get("incomplete_details") or {}).get("reason") != "max_output_tokens":
        raise AssertionError(f"{label} omitted max_output_tokens details: {response}")
    if (response.get("usage") or {}).get("output_tokens") != budget:
        raise AssertionError(f"{label} output-token accounting changed: {response}")
    if len(calls) != 1 or calls[0].get("name") != "write_exact":
        raise AssertionError(f"{label} lost the tool prefix: {calls}")
    if calls[0].get("status") != "incomplete":
        raise AssertionError(f"{label} marked a truncated call completed: {calls[0]}")
    if json.loads(calls[0]["arguments"]).get("value") == LONG_VALUE:
        raise AssertionError(f"{label} unexpectedly completed the long value")
    return calls


def live_responses_truncation(budget: int) -> dict[str, Any]:
    nonstream = post_json("/v1/responses", responses_payload(budget, stream=False))
    assert_incomplete_response(nonstream, budget, "Responses non-stream")

    event_types: list[str] = []
    done_statuses: list[str | None] = []
    final_response = None
    for event_name, event in iter_sse(
        "/v1/responses", responses_payload(budget, stream=True)
    ):
        if event == "[DONE]":
            continue
        assert isinstance(event, dict)
        event_type = event.get("type") or event_name or ""
        event_types.append(event_type)
        if event_type == "response.output_item.done":
            item = event.get("item") or {}
            if item.get("type") == "function_call":
                done_statuses.append(item.get("status"))
        if event_type == "response.incomplete":
            final_response = event["response"]
    if "response.completed" in event_types:
        raise AssertionError("Responses stream emitted response.completed on truncation")
    if "response.function_call_arguments.done" in event_types:
        raise AssertionError(
            "Responses stream emitted executable arguments.done on truncation"
        )
    if final_response is None:
        raise AssertionError(f"Responses stream omitted response.incomplete: {event_types}")
    assert_incomplete_response(final_response, budget, "Responses stream final")
    if done_statuses != ["incomplete"]:
        raise AssertionError(
            f"Responses output_item.done did not carry incomplete: {done_statuses}"
        )
    return {
        "nonstream_status": nonstream["status"],
        "stream_terminal_event": "response.incomplete",
        "function_call_arguments_done_absent": True,
        "output_item_done_status": done_statuses[0],
        "response_completed_absent": True,
        "output_tokens": budget,
    }


def responses_unit_boundary() -> dict[str, Any]:
    processor = SimpleStreamingEventProcessor()
    processor.state.current_state = _StateType.TOOL_CALL
    processor.state.current_item_id = "fc_unit"
    processor.state.tool_call_id = "call_unit"
    processor.state.tool_call_name = "write_exact"
    processor.state.tool_call_index = 0
    processor.state.accumulated_text = '{"value":"partial"}'
    processor.state.has_emitted_tool_call_delta = True
    events = processor.close_current(incomplete=True)
    if [event.type for event in events] != ["response.output_item.done"]:
        raise AssertionError(f"Incomplete unit boundary emitted unsafe events: {events}")
    if events[0].item.status != "incomplete":
        raise AssertionError(f"Incomplete unit boundary status changed: {events[0]}")
    return {
        "arguments_done_suppressed": True,
        "output_item_status": events[0].item.status,
    }


def main() -> None:
    started = time.monotonic()
    tokenizer = AutoTokenizer.from_pretrained(
        "/model", local_files_only=True, trust_remote_code=False
    )
    parser_result = controlled_parser_replay(tokenizer)
    unit_result = responses_unit_boundary()
    budget, chat_result = live_chat_truncation(tokenizer)
    result = {
        "controlled_real_token_replay": parser_result,
        "responses_execution_boundary_unit": unit_result,
        "live_truncation_budget_tokens": budget,
        "chat_completions": chat_result,
        "anthropic_messages": live_anthropic_truncation(budget),
        "openai_responses": live_responses_truncation(budget),
        "contract": (
            "Partial tool deltas may be observable, but only a normal tool-call "
            "terminal is executable; every max-token interruption stays explicit."
        ),
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
