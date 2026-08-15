#!/usr/bin/env python3
"""Repeated OpenAI and Anthropic tool-call protocol checks for the local server."""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request


MODEL = "qwen3.8-27b-nvfp4-k8v4"
PATH = "/workspace/README.md"
HEADING = "Qwen3.8-27B NVFP4 — correctness-first local agent server"
TOOL_RESULT = f"# {HEADING}\n\nMeasured configuration details follow."


def post_json(url: str, payload: dict, headers: dict | None = None) -> dict:
    request_headers = {"Content-Type": "application/json"}
    request_headers.update(headers or {})
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers=request_headers,
    )
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {error.code} from {url}: {body}") from error


def openai_trial(base_url: str) -> dict:
    tool = {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a UTF-8 text file from the local workspace.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
                "additionalProperties": False,
            },
        },
    }
    initial = [
        {
            "role": "developer",
            "content": (
                "You are a local coding agent. You MUST call read_file before "
                "answering and must never invent file contents."
            ),
        },
        {"role": "user", "content": f"Read {PATH} and report its first heading."},
    ]
    first = post_json(
        f"{base_url}/v1/chat/completions",
        {
            "model": MODEL,
            "messages": initial,
            "tools": [tool],
            "tool_choice": "auto",
            "max_tokens": 1_024,
        },
    )
    choice = first["choices"][0]
    assistant = choice["message"]
    calls = assistant.get("tool_calls") or []
    if choice.get("finish_reason") != "tool_calls" or len(calls) != 1:
        raise RuntimeError(f"Malformed OpenAI tool choice: {choice}")
    call = calls[0]
    arguments = json.loads(call["function"]["arguments"])
    if call["function"]["name"] != "read_file" or arguments != {"path": PATH}:
        raise RuntimeError(f"Incorrect OpenAI tool call: {call}")

    messages = initial + [
        {
            "role": "assistant",
            "content": assistant.get("content"),
            "tool_calls": [call],
        },
        {
            "role": "tool",
            "tool_call_id": call["id"],
            "content": TOOL_RESULT,
        },
    ]
    second = post_json(
        f"{base_url}/v1/chat/completions",
        {
            "model": MODEL,
            "messages": messages,
            "tools": [tool],
            "tool_choice": "auto",
            "max_tokens": 1_024,
        },
    )
    continuation = second["choices"][0]
    answer = continuation["message"].get("content") or ""
    if continuation.get("finish_reason") != "stop" or HEADING not in answer:
        raise RuntimeError(f"Incorrect OpenAI continuation: {continuation}")
    return {
        "tool_name": call["function"]["name"],
        "tool_arguments": arguments,
        "tool_finish_reason": choice.get("finish_reason"),
        "continuation_finish_reason": continuation.get("finish_reason"),
        "heading_present": True,
    }


def anthropic_trial(base_url: str) -> dict:
    headers = {"anthropic-version": "2023-06-01"}
    tool = {
        "name": "read_file",
        "description": "Read a UTF-8 text file from the local workspace.",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
            "additionalProperties": False,
        },
    }
    system = (
        "You are a local coding agent. You MUST call read_file before answering "
        "and must never invent file contents."
    )
    first_user = {
        "role": "user",
        "content": f"Read {PATH} and report its first heading.",
    }
    first = post_json(
        f"{base_url}/v1/messages",
        {
            "model": MODEL,
            "system": system,
            "messages": [first_user],
            "tools": [tool],
            "max_tokens": 1_024,
        },
        headers,
    )
    tool_blocks = [block for block in first["content"] if block["type"] == "tool_use"]
    if first.get("stop_reason") != "tool_use" or len(tool_blocks) != 1:
        raise RuntimeError(f"Malformed Anthropic tool choice: {first}")
    use = tool_blocks[0]
    if use["name"] != "read_file" or use["input"] != {"path": PATH}:
        raise RuntimeError(f"Incorrect Anthropic tool call: {use}")

    messages = [
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
    ]
    second = post_json(
        f"{base_url}/v1/messages",
        {
            "model": MODEL,
            "system": system,
            "messages": messages,
            "tools": [tool],
            "max_tokens": 1_024,
        },
        headers,
    )
    answer = "".join(
        block.get("text", "")
        for block in second["content"]
        if block["type"] == "text"
    )
    if second.get("stop_reason") != "end_turn" or HEADING not in answer:
        raise RuntimeError(f"Incorrect Anthropic continuation: {second}")
    return {
        "tool_name": use["name"],
        "tool_input": use["input"],
        "tool_stop_reason": first.get("stop_reason"),
        "continuation_stop_reason": second.get("stop_reason"),
        "thinking_present": any(
            block["type"] == "thinking" for block in first["content"]
        ),
        "heading_present": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--trials", type=int, default=3)
    args = parser.parse_args()
    if args.trials < 1:
        raise SystemExit("--trials must be positive")

    base_url = args.url.rstrip("/")
    started = time.monotonic()
    openai_results = [openai_trial(base_url) for _ in range(args.trials)]
    anthropic_results = [anthropic_trial(base_url) for _ in range(args.trials)]
    print(
        json.dumps(
            {
                "trials_per_protocol": args.trials,
                "openai_passed": len(openai_results),
                "anthropic_passed": len(anthropic_results),
                "elapsed_seconds": round(time.monotonic() - started, 3),
                "openai": openai_results,
                "anthropic": anthropic_results,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
