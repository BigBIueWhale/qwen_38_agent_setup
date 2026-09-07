#!/usr/bin/env python3
"""Repeated OpenAI and Anthropic tool-call protocol checks for the local server."""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request


MODEL = "qwen3.8-27b-nvfp4-k8v4"
# Every generative request names the agent whose KV cache it belongs to;
# the backend refuses one that does not.
KV_SCOPE = "protocol-probe"
PATH = "/workspace/README.md"
HEADING = "Qwen3.8-27B NVFP4 — correctness-first local agent server"
TOOL_RESULT = f"# {HEADING}\n\nMeasured configuration details follow."
REASONING_END = "</think>"
TOOL_CALL_START = "<tool_call>"
END_OF_TURN = "<|im_end|>"


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


def post_sse(url: str, payload: dict) -> list[dict]:
    """POST a streaming request and return every JSON event before [DONE]."""
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    events: list[dict] = []
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8").rstrip("\r\n")
                if not line.startswith("data: "):
                    continue
                data = line[len("data: ") :]
                if data == "[DONE]":
                    break
                events.append(json.loads(data))
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {error.code} from {url}: {body}") from error
    return events


def marker_token_ids(base_url: str) -> dict[str, int]:
    """Resolve the single ids of the markers the served count is defined by.

    This is a vocabulary lookup of three fixed strings, made once so the
    probe's oracle -- the position of the first boundary id in the generated
    ids -- is independent of the server's own count. Nothing generated is
    ever re-tokenised.
    """
    ids: dict[str, int] = {}
    for text in (REASONING_END, TOOL_CALL_START, END_OF_TURN):
        response = post_json(
            f"{base_url}/tokenize",
            {"model": MODEL, "prompt": text, "add_special_tokens": False},
        )
        tokens = response.get("tokens")
        if not isinstance(tokens, list) or len(tokens) != 1:
            raise RuntimeError(f"{text!r} is not a single token: {response}")
        ids[text] = tokens[0]
    return ids


def require_exact_usage(
    usage: dict,
    token_ids: list[int],
    markers: dict[str, int],
    label: str,
) -> int:
    """Hold a served usage to the generated ids it describes.

    ``completion_tokens`` must count every generated id, and
    ``completion_tokens_details.reasoning_tokens`` must be the number of ids
    before the first reasoning-end id -- ``</think>`` or Qwen's implicit
    ``<tool_call>`` -- or all of them when reasoning never ends. The cached
    prompt count must be served too: the deployment enables it, and a
    client cannot tell an absent field from a zero it invented.
    """
    details = usage.get("completion_tokens_details")
    if not isinstance(details, dict) or not isinstance(
        details.get("reasoning_tokens"), int
    ):
        raise RuntimeError(f"{label}: no served reasoning_tokens in usage {usage}")
    boundary = {markers[REASONING_END], markers[TOOL_CALL_START]}
    expected = next(
        (index for index, token_id in enumerate(token_ids) if token_id in boundary),
        len(token_ids),
    )
    if details["reasoning_tokens"] != expected:
        raise RuntimeError(
            f"{label}: served reasoning_tokens={details['reasoning_tokens']} but "
            f"the boundary id sits at position {expected} of {len(token_ids)}"
        )
    if usage.get("completion_tokens") != len(token_ids):
        raise RuntimeError(
            f"{label}: completion_tokens={usage.get('completion_tokens')} but "
            f"{len(token_ids)} ids were generated"
        )
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
    if not isinstance(cached, int):
        raise RuntimeError(f"{label}: no served cached_tokens in usage {usage}")
    return expected


def openai_trial(base_url: str, markers: dict[str, int]) -> dict:
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
    first_request = {
        "model": MODEL,
        "messages": initial,
        "tools": [tool],
        "tool_choice": "auto",
        "max_tokens": 1_024,
        "kv_scope": KV_SCOPE,
        "return_token_ids": True,
    }
    first = post_json(f"{base_url}/v1/chat/completions", first_request)
    choice = first["choices"][0]
    assistant = choice["message"]
    calls = assistant.get("tool_calls") or []
    if choice.get("finish_reason") != "tool_calls" or len(calls) != 1:
        raise RuntimeError(f"Malformed OpenAI tool choice: {choice}")
    call = calls[0]
    arguments = json.loads(call["function"]["arguments"])
    if call["function"]["name"] != "read_file" or arguments != {"path": PATH}:
        raise RuntimeError(f"Incorrect OpenAI tool call: {call}")
    first_ids = choice["token_ids"]
    first_reasoning_tokens = require_exact_usage(
        first["usage"], first_ids, markers, "OpenAI tool call"
    )

    # The same request streamed: the final usage chunk must describe the
    # ids the chunks carried, by the same boundary.
    streamed = post_sse(
        f"{base_url}/v1/chat/completions",
        {
            **first_request,
            "stream": True,
            "stream_options": {"include_usage": True},
        },
    )
    streamed_ids: list[int] = []
    for event in streamed:
        for streamed_choice in event.get("choices") or []:
            streamed_ids.extend(streamed_choice.get("token_ids") or [])
    usage_chunks = [event for event in streamed if not event.get("choices")]
    if len(usage_chunks) != 1 or "usage" not in usage_chunks[0]:
        raise RuntimeError(f"Streaming response lacks one final usage chunk: {streamed}")
    streamed_reasoning_tokens = require_exact_usage(
        usage_chunks[0]["usage"], streamed_ids, markers, "OpenAI streamed tool call"
    )

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
            "kv_scope": KV_SCOPE,
            "return_token_ids": True,
        },
    )
    continuation = second["choices"][0]
    answer = continuation["message"].get("content") or ""
    if continuation.get("finish_reason") != "stop" or HEADING not in answer:
        raise RuntimeError(f"Incorrect OpenAI continuation: {continuation}")
    continuation_reasoning_tokens = require_exact_usage(
        second["usage"], continuation["token_ids"], markers, "OpenAI continuation"
    )
    return {
        "tool_name": call["function"]["name"],
        "tool_arguments": arguments,
        "tool_finish_reason": choice.get("finish_reason"),
        "continuation_finish_reason": continuation.get("finish_reason"),
        "heading_present": True,
        "tool_call_reasoning_tokens": first_reasoning_tokens,
        "tool_call_completion_tokens": len(first_ids),
        "streamed_reasoning_tokens": streamed_reasoning_tokens,
        "streamed_completion_tokens": len(streamed_ids),
        "continuation_reasoning_tokens": continuation_reasoning_tokens,
        "continuation_completion_tokens": len(continuation["token_ids"]),
        "cached_prompt_tokens": (
            first["usage"]["prompt_tokens_details"]["cached_tokens"],
            second["usage"]["prompt_tokens_details"]["cached_tokens"],
        ),
        # Whether a tool_calls finish carries the end-of-turn id as its last
        # generated id: reported, not asserted, so the identity
        # completion = reasoning + content + markers can be read exactly.
        "tool_call_ends_with_end_of_turn": first_ids[-1] == markers[END_OF_TURN],
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
            "kv_scope": KV_SCOPE,
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
            "kv_scope": KV_SCOPE,
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
    markers = marker_token_ids(base_url)
    openai_results = [openai_trial(base_url, markers) for _ in range(args.trials)]
    anthropic_results = [anthropic_trial(base_url) for _ in range(args.trials)]
    print(
        json.dumps(
            {
                "trials_per_protocol": args.trials,
                "openai_passed": len(openai_results),
                "anthropic_passed": len(anthropic_results),
                "elapsed_seconds": round(time.monotonic() - started, 3),
                "marker_token_ids": markers,
                "openai": openai_results,
                "anthropic": anthropic_results,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
