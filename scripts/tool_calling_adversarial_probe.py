#!/usr/bin/env python3
"""Adversarial, invariant-based tool-calling checks for the pinned Qwen runtime.

Run this inside the serving container. It intentionally uses the real checkpoint
tokenizer and installed XGrammar/vLLM sources; no character/token estimates and
no host Python packages are involved.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from typing import Any

import xgrammar as xgr
from transformers import AutoTokenizer

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
    ChatCompletionToolsParam,
)
from vllm.parser.qwen3 import Qwen3Parser
from vllm.tool_parsers.structural_tag_registry import get_model_structural_tag
from vllm.v1.structured_output import StructuredOutputManager


MODEL = "qwen3.8-27b-nvfp4-k8v4"
BASE_URL = "http://127.0.0.1:8000"
PATH = "/workspace/README.md"
HEADING = "Qwen3.8-27B NVFP4 — correctness-first local agent server"
TOOL_RESULT = f"# {HEADING}\n\nVerified synthetic tool result."
ANTHROPIC_HEADERS = {"anthropic-version": "2023-06-01"}


def post_json(
    path: str, payload: dict[str, Any], headers: dict[str, str] | None = None
) -> dict[str, Any]:
    request = urllib.request.Request(
        BASE_URL + path,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json", **(headers or {})},
    )
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {error.code} from {path}: {body}") from error


def expect_http_400(
    path: str, payload: dict[str, Any], headers: dict[str, str] | None = None
) -> dict[str, Any]:
    request = urllib.request.Request(
        BASE_URL + path,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json", **(headers or {})},
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            body = response.read().decode("utf-8", errors="replace")
            raise AssertionError(f"Expected HTTP 400 from {path}, got {response.status}: {body}")
    except urllib.error.HTTPError as error:
        body = json.loads(error.read())
        if error.code != 400:
            raise AssertionError(f"Expected HTTP 400 from {path}, got {error.code}: {body}")
        return body


def iter_sse(
    path: str, payload: dict[str, Any], headers: dict[str, str] | None = None
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
        response = urllib.request.urlopen(request, timeout=300)
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


def read_file_tool(strict: bool | None = None) -> dict[str, Any]:
    function: dict[str, Any] = {
        "name": "read_file",
        "description": "Read the one permitted synthetic workspace file.",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string", "const": PATH}},
            "required": ["path"],
            "additionalProperties": False,
        },
    }
    if strict is not None:
        function["strict"] = strict
    return {"type": "function", "function": function}


def anthropic_read_file_tool(strict: bool | None = None) -> dict[str, Any]:
    tool: dict[str, Any] = {
        "name": "read_file",
        "description": "Read the one permitted synthetic workspace file.",
        "input_schema": read_file_tool()["function"]["parameters"],
    }
    if strict is not None:
        tool["strict"] = strict
    return tool


def xgrammar_accepts(
    compiled: xgr.CompiledGrammar, tokenizer: Any, text: str
) -> tuple[bool, int | None]:
    matcher = xgr.GrammarMatcher(compiled, terminate_without_stop_token=True)
    for index, token_id in enumerate(tokenizer.encode(text, add_special_tokens=False)):
        if not matcher.accept_token(token_id):
            return False, index
    return matcher.is_completed(), None


def real_tokenizer_and_grammar_probe() -> dict[str, Any]:
    tokenizer = AutoTokenizer.from_pretrained(
        "/model", local_files_only=True, trust_remote_code=False
    )
    expected_markers = {
        "<think>": [248068],
        "</think>": [248069],
        "<tool_call>": [248058],
        "</tool_call>": [248059],
    }
    actual_markers = {
        marker: tokenizer.encode(marker, add_special_tokens=False)
        for marker in expected_markers
    }
    if actual_markers != expected_markers:
        raise AssertionError(
            f"Pinned tokenizer marker IDs changed: expected {expected_markers}, "
            f"got {actual_markers}"
        )

    protocol_tool = ChatCompletionToolsParam.model_validate(read_file_tool(False))
    for strict_value in (None, False, True):
        candidate = protocol_tool.model_copy(deep=True)
        candidate.function.strict = strict_value
        tag = get_model_structural_tag("qwen_3_coder", [candidate], "auto", False)
        if tag is None:
            raise AssertionError(f"Qwen auto schema missing for strict={strict_value!r}")
        schema = tag.model_dump()["format"]["tags"][0]["content"]["json_schema"]
        if schema != candidate.function.parameters:
            raise AssertionError(
                f"Qwen schema changed for strict={strict_value!r}: {schema}"
            )
    for other_model in ("llama", "qwen_3_5", "deepseek_v4"):
        if get_model_structural_tag(other_model, [protocol_tool], "auto", False):
            raise AssertionError(f"Non-Qwen strict policy was broadened to {other_model}")
    if get_model_structural_tag("qwen_3_coder", [protocol_tool], "none", False):
        raise AssertionError("tool_choice=none unexpectedly activated Qwen grammar")

    nested_tool = ChatCompletionToolsParam.model_validate(
        {
            "type": "function",
            "function": {
                "name": "configure",
                "strict": False,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "options": {
                            "type": "object",
                            "properties": {
                                "mode": {"type": "string", "enum": ["safe"]},
                                "retries": {"type": "integer", "minimum": 2, "maximum": 2},
                            },
                            "required": ["mode", "retries"],
                            "additionalProperties": False,
                        },
                        "paths": {
                            "type": "array",
                            "items": {"type": "string"},
                            "minItems": 2,
                            "maxItems": 2,
                        },
                    },
                    "required": ["options", "paths"],
                    "additionalProperties": False,
                },
            },
        }
    )
    tag = get_model_structural_tag("qwen_3_coder", [nested_tool], "auto", False)
    if tag is None:
        raise AssertionError("Nested Qwen schema did not produce a structural tag")
    compiler = xgr.GrammarCompiler(xgr.TokenizerInfo.from_huggingface(tokenizer))
    compiled = compiler.compile_structural_tag(tag)
    valid = (
        "<tool_call>\n<function=configure>\n"
        '<parameter=options>{"mode":"safe","retries":2}</parameter>\n'
        '<parameter=paths>["a","b"]</parameter>\n'
        "</function>\n</tool_call>"
    )
    unknown = valid.replace("function=configure", "function=delete_everything", 1)
    wrong_type = valid.replace('"retries":2', '"retries":"2"', 1)
    extra_property = valid.replace(
        '"retries":2}', '"retries":2,"fallback":true}', 1
    )
    valid_ok, valid_reject = xgrammar_accepts(compiled, tokenizer, valid)
    unknown_ok, unknown_reject = xgrammar_accepts(compiled, tokenizer, unknown)
    wrong_type_ok, wrong_type_reject = xgrammar_accepts(compiled, tokenizer, wrong_type)
    extra_ok, extra_reject = xgrammar_accepts(compiled, tokenizer, extra_property)
    if not valid_ok or valid_reject is not None:
        raise AssertionError(f"Valid nested grammar rejected at {valid_reject}")
    if unknown_ok or unknown_reject is None:
        raise AssertionError("Unknown tool name was not rejected by XGrammar")
    if wrong_type_ok or wrong_type_reject is None:
        raise AssertionError("Schema-invalid nested type was not rejected by XGrammar")
    if extra_ok or extra_reject is None:
        raise AssertionError("additionalProperties:false was not enforced by XGrammar")

    # Sensitivity control for the exact scheduler bug: without the structural
    # trigger token, triggered-tag grammar treats the remaining XML as ordinary
    # text and accepts the unknown function. The fixed trim must retain it.
    unknown_ids = tokenizer.encode(unknown, add_special_tokens=False)
    if unknown_ids[0] != expected_markers["<tool_call>"][0]:
        raise AssertionError("Unknown-call control does not begin with <tool_call>")
    bypass = xgr.GrammarMatcher(compiled, terminate_without_stop_token=True)
    if not all(bypass.accept_token(token_id) for token_id in unknown_ids[1:]):
        raise AssertionError("Dropped-trigger sensitivity control unexpectedly rejected")

    boundary_parser = object.__new__(Qwen3Parser)
    boundary_parser._reasoning_start_token_id = expected_markers["<think>"][0]
    boundary_parser._reasoning_end_token_id = expected_markers["</think>"][0]
    boundary_parser._tool_call_token_id = expected_markers["<tool_call>"][0]
    boundary_parser._tool_call_end_token_id = expected_markers["</tool_call>"][0]
    implicit = [expected_markers["<think>"][0], 11, expected_markers["<tool_call>"][0]]
    implicit_index = StructuredOutputManager._find_reasoning_end_index(
        boundary_parser, implicit, 2
    )
    if implicit_index != 1:
        raise AssertionError(f"Implicit boundary index is {implicit_index}, expected 1")
    request_stub = type("Request", (), {})()
    request_stub.all_token_ids = implicit
    request_stub.structured_output_request = type("Structured", (), {})()
    request_stub.structured_output_request.reasoning_end_token_index = implicit_index
    retained = StructuredOutputManager.trim_reasoning_for_advance(
        object(), request_stub, [expected_markers["<tool_call>"][0]]
    )
    if retained != expected_markers["<tool_call>"]:
        raise AssertionError(f"Implicit Qwen grammar trigger was trimmed: {retained}")

    parser = Qwen3Parser(tokenizer, tools=[protocol_tool])
    request = ChatCompletionRequest(
        model=MODEL,
        messages=[{"role": "user", "content": "test"}],
        tools=[protocol_tool],
        tool_choice="auto",
    )
    raw = (
        "reason first"
        f"<tool_call>\n<function=read_file>\n<parameter=path>{PATH}</parameter>\n"
        "</function>\n</tool_call>"
    )
    parsed = parser.extract_tool_calls(raw, request)
    if not parsed.tools_called or len(parsed.tool_calls) != 1:
        raise AssertionError(f"Unified Qwen parser lost implicit tool call: {parsed}")
    parsed_args = json.loads(parsed.tool_calls[0].function.arguments)
    if parsed.tool_calls[0].function.name != "read_file" or parsed_args != {"path": PATH}:
        raise AssertionError(f"Unified Qwen parser reconstructed wrong call: {parsed}")

    return {
        "real_tokenizer_marker_ids": actual_markers,
        "qwen_auto_schema_strict_none_false_true": True,
        "non_qwen_policy_unchanged": True,
        "nested_schema_valid_accepted": True,
        "unknown_tool_rejected_at_token": unknown_reject,
        "wrong_nested_type_rejected_at_token": wrong_type_reject,
        "extra_property_rejected_at_token": extra_reject,
        "dropped_trigger_control_accepts_unknown_call": True,
        "fixed_trim_retains_trigger": True,
        "unified_parser_implicit_tool_call": True,
    }


def openai_stream_call() -> dict[str, Any]:
    initial_messages = [
        {
            "role": "developer",
            "content": "You MUST call read_file exactly once before answering.",
        },
        {"role": "user", "content": f"Read {PATH}."},
    ]
    payload = {
        "model": MODEL,
        "messages": initial_messages,
        "tools": [read_file_tool(False)],
        "tool_choice": "required",
        "parallel_tool_calls": False,
        "max_tokens": 1_024,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    calls: dict[int, dict[str, str]] = {}
    reasoning = ""
    content = ""
    finish_reason = None
    saw_done = False
    for _, event in iter_sse("/v1/chat/completions", payload):
        if event == "[DONE]":
            saw_done = True
            continue
        assert isinstance(event, dict)
        for choice in event.get("choices", []):
            delta = choice.get("delta") or {}
            reasoning += delta.get("reasoning") or ""
            content += delta.get("content") or ""
            for tool_delta in delta.get("tool_calls") or []:
                slot = calls.setdefault(
                    tool_delta["index"], {"id": "", "name": "", "arguments": ""}
                )
                slot["id"] += tool_delta.get("id") or ""
                function = tool_delta.get("function") or {}
                slot["name"] += function.get("name") or ""
                slot["arguments"] += function.get("arguments") or ""
            if choice.get("finish_reason") is not None:
                finish_reason = choice["finish_reason"]
    if not saw_done or finish_reason != "tool_calls" or sorted(calls) != [0]:
        raise AssertionError(
            f"Malformed OpenAI stream termination: done={saw_done}, "
            f"finish={finish_reason}, calls={calls}"
        )
    call = calls[0]
    if call["name"] != "read_file" or json.loads(call["arguments"]) != {"path": PATH}:
        raise AssertionError(f"Incorrect reconstructed OpenAI stream call: {call}")

    history_call = {
        "id": call["id"],
        "type": "function",
        "function": {"name": call["name"], "arguments": call["arguments"]},
    }
    history_assistant: dict[str, Any] = {
        "role": "assistant",
        "content": content or None,
        "tool_calls": [history_call],
    }
    if reasoning:
        history_assistant["reasoning"] = reasoning
    continuation_payload = {
        "model": MODEL,
        "messages": initial_messages
        + [
            history_assistant,
            {"role": "tool", "tool_call_id": call["id"], "content": TOOL_RESULT},
        ],
        "tools": [read_file_tool(False)],
        "tool_choice": "auto",
        "max_tokens": 1_024,
        "stream": True,
    }
    answer = ""
    continuation_finish = None
    for _, event in iter_sse("/v1/chat/completions", continuation_payload):
        if event == "[DONE]":
            continue
        assert isinstance(event, dict)
        for choice in event.get("choices", []):
            delta = choice.get("delta") or {}
            answer += delta.get("content") or ""
            if choice.get("finish_reason") is not None:
                continuation_finish = choice["finish_reason"]
    if continuation_finish != "stop" or HEADING not in answer:
        raise AssertionError(
            f"Incorrect OpenAI streamed continuation: finish={continuation_finish}, "
            f"answer={answer!r}"
        )
    return {
        "initial_finish_reason": finish_reason,
        "tool_name": call["name"],
        "tool_arguments": json.loads(call["arguments"]),
        "reasoning_streamed": bool(reasoning),
        "continuation_finish_reason": continuation_finish,
        "heading_present": True,
    }


def collect_anthropic_stream(payload: dict[str, Any]) -> dict[str, Any]:
    blocks: dict[int, dict[str, Any]] = {}
    stop_reason = None
    saw_message_stop = False
    for event_name, event in iter_sse("/v1/messages", payload, ANTHROPIC_HEADERS):
        if event == "[DONE]":
            continue
        assert isinstance(event, dict)
        event_type = event.get("type") or event_name
        if event_type == "error":
            raise AssertionError(f"Anthropic stream error: {event}")
        if event_type == "content_block_start":
            block = dict(event["content_block"])
            if block.get("type") == "tool_use":
                block["partial_json"] = ""
            blocks[event["index"]] = block
        elif event_type == "content_block_delta":
            block = blocks[event["index"]]
            delta = event["delta"]
            delta_type = delta.get("type")
            if delta_type == "thinking_delta":
                block["thinking"] = (block.get("thinking") or "") + (
                    delta.get("thinking") or ""
                )
            elif delta_type == "signature_delta":
                block["signature"] = (block.get("signature") or "") + (
                    delta.get("signature") or ""
                )
            elif delta_type == "text_delta":
                block["text"] = (block.get("text") or "") + (delta.get("text") or "")
            elif delta_type == "input_json_delta":
                block["partial_json"] += delta.get("partial_json") or ""
        elif event_type == "message_delta":
            stop_reason = (event.get("delta") or {}).get("stop_reason") or stop_reason
        elif event_type == "message_stop":
            saw_message_stop = True

    ordered = []
    for index in sorted(blocks):
        block = blocks[index]
        if block.get("type") == "tool_use":
            block["input"] = json.loads(block.pop("partial_json") or "{}")
        ordered.append(block)
    return {
        "content": ordered,
        "stop_reason": stop_reason,
        "message_stop": saw_message_stop,
    }


def anthropic_stream_call() -> dict[str, Any]:
    first_user = {"role": "user", "content": f"Read {PATH}."}
    payload = {
        "model": MODEL,
        "system": "You MUST call read_file exactly once before answering.",
        "messages": [first_user],
        "tools": [anthropic_read_file_tool(False)],
        "tool_choice": {"type": "any", "disable_parallel_tool_use": True},
        "max_tokens": 1_024,
        "stream": True,
    }
    first = collect_anthropic_stream(payload)
    uses = [block for block in first["content"] if block.get("type") == "tool_use"]
    if not first["message_stop"] or first["stop_reason"] != "tool_use" or len(uses) != 1:
        raise AssertionError(f"Malformed Anthropic tool stream: {first}")
    use = uses[0]
    if use.get("name") != "read_file" or use.get("input") != {"path": PATH}:
        raise AssertionError(f"Incorrect Anthropic stream tool call: {use}")

    continuation = collect_anthropic_stream(
        {
            "model": MODEL,
            "system": payload["system"],
            "messages": [
                first_user,
                {"role": "assistant", "content": first["content"]},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": use["id"],
                            "content": TOOL_RESULT,
                        }
                    ],
                },
            ],
            "tools": [anthropic_read_file_tool(False)],
            "tool_choice": {"type": "auto", "disable_parallel_tool_use": True},
            "max_tokens": 1_024,
            "stream": True,
        }
    )
    answer = "".join(
        block.get("text", "")
        for block in continuation["content"]
        if block.get("type") == "text"
    )
    if (
        not continuation["message_stop"]
        or continuation["stop_reason"] != "end_turn"
        or HEADING not in answer
    ):
        raise AssertionError(f"Incorrect Anthropic streamed continuation: {continuation}")
    return {
        "initial_stop_reason": first["stop_reason"],
        "tool_name": use["name"],
        "tool_input": use["input"],
        "thinking_streamed": any(
            block.get("type") == "thinking" for block in first["content"]
        ),
        "continuation_stop_reason": continuation["stop_reason"],
        "heading_present": True,
    }


def live_policy_probe() -> dict[str, Any]:
    no_tool = post_json(
        "/v1/chat/completions",
        {
            "model": MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": "Do not call a tool. Briefly say that tools are disabled.",
                }
            ],
            "tools": [read_file_tool(False)],
            "tool_choice": "none",
            "max_tokens": 512,
        },
    )["choices"][0]
    if no_tool["message"].get("tool_calls"):
        raise AssertionError(f"tool_choice=none returned a tool call: {no_tool}")

    parallel = post_json(
        "/v1/chat/completions",
        {
            "model": MODEL,
            "messages": [
                {
                    "role": "developer",
                    "content": "You must use read_file. Never answer directly.",
                },
                {
                    "role": "user",
                    "content": f"Call read_file twice for {PATH}, as two parallel calls.",
                },
            ],
            "tools": [read_file_tool(False)],
            "tool_choice": "required",
            "parallel_tool_calls": False,
            "max_tokens": 1_024,
        },
    )["choices"][0]
    parallel_calls = parallel["message"].get("tool_calls") or []
    if parallel.get("finish_reason") != "tool_calls" or len(parallel_calls) != 1:
        raise AssertionError(f"parallel_tool_calls=false did not expose one call: {parallel}")

    malformed_openai = {
        "model": MODEL,
        "messages": [
            {"role": "user", "content": "test"},
            {"role": "tool", "tool_call_id": "orphan", "content": "bad"},
        ],
        "max_tokens": 16,
    }
    openai_error = expect_http_400("/v1/chat/completions", malformed_openai)
    malformed_anthropic = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "orphan", "content": "bad"}
                ],
            }
        ],
        "max_tokens": 16,
    }
    anthropic_error = expect_http_400(
        "/v1/messages", malformed_anthropic, ANTHROPIC_HEADERS
    )
    if anthropic_error.get("error", {}).get("type") != "invalid_request_error":
        raise AssertionError(f"Anthropic validation error type is wrong: {anthropic_error}")

    return {
        "tool_choice_none_no_calls": True,
        "parallel_tool_calls_false_exposed_calls": len(parallel_calls),
        "openai_orphan_result_http_400": openai_error["error"]["code"],
        "anthropic_orphan_result_http_400": 400,
        "anthropic_error_type": anthropic_error["error"]["type"],
    }


def main() -> None:
    started = time.monotonic()
    result = {
        "real_tokenizer_and_grammar": real_tokenizer_and_grammar_probe(),
        "openai_stream": openai_stream_call(),
        "anthropic_stream": anthropic_stream_call(),
        "live_policy": live_policy_probe(),
    }
    result["elapsed_seconds"] = round(time.monotonic() - started, 3)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
