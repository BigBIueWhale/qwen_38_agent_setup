#!/usr/bin/env python3
"""Validate the Responses API tool loop used by current OpenAI Codex clients."""

from __future__ import annotations

import json
import urllib.error
import urllib.request


BASE_URL = "http://127.0.0.1:8000"
MODEL = "qwen3.8-27b-nvfp4-k8v4"
PATH = "/workspace/README.md"
HEADING = "Qwen3.8-27B NVFP4 — correctness-first local agent server"
TOOL_RESULT = f"# {HEADING}\n\nMeasured configuration details follow."

TOOL = {
    "type": "function",
    "name": "read_file",
    "description": "Read one UTF-8 text file from the local workspace.",
    "parameters": {
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ["path"],
        "additionalProperties": False,
    },
    "strict": True,
}


def request(payload: dict, *, stream: bool = False):
    req = urllib.request.Request(
        f"{BASE_URL}/v1/responses",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        response = urllib.request.urlopen(req, timeout=300)
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {error.code}: {body}") from error
    if not stream:
        with response:
            return json.load(response)
    return response


def base_payload(*, stream: bool) -> dict:
    return {
        "model": MODEL,
        "instructions": (
            "You are a local coding agent. You MUST call read_file before "
            "answering and must never invent file contents."
        ),
        "input": f"Read {PATH} and report its first heading.",
        "tools": [TOOL],
        "tool_choice": "auto",
        "max_output_tokens": 1_024,
        "stream": stream,
        "store": False,
        "reasoning": {"effort": "xhigh"},
        "chat_template_kwargs": {
            "enable_thinking": True,
            "reasoning_effort": "xhigh",
        },
    }


def function_calls(output: list[dict]) -> list[dict]:
    return [item for item in output if item.get("type") == "function_call"]


def output_text(output: list[dict]) -> str:
    return "".join(
        content.get("text", "")
        for item in output
        if item.get("type") == "message"
        for content in item.get("content", [])
        if content.get("type") == "output_text"
    )


def validate_call(call: dict) -> None:
    arguments = json.loads(call["arguments"])
    if call.get("name") != "read_file" or arguments != {"path": PATH}:
        raise RuntimeError(f"Incorrect Responses function call: {call}")


def continuation_payload(first: dict, *, stream: bool) -> dict:
    payload = base_payload(stream=stream)
    payload["input"] = [
        {"role": "user", "content": base_payload(stream=False)["input"]},
        *first["output"],
        {
            "type": "function_call_output",
            "call_id": function_calls(first["output"])[0]["call_id"],
            "output": TOOL_RESULT,
        },
    ]
    return payload


def read_sse(response) -> tuple[list[dict], dict]:
    events: list[dict] = []
    completed: dict | None = None
    with response:
        for raw_line in response:
            line = raw_line.decode("utf-8").strip()
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                continue
            event = json.loads(data)
            events.append(event)
            if event.get("type") == "response.completed":
                completed = event["response"]
    if completed is None:
        raise RuntimeError("Responses stream ended without response.completed")
    return events, completed


def semantic_summary(response: dict) -> dict:
    calls = function_calls(response["output"])
    return {
        "status": response.get("status"),
        "calls": [
            {
                "name": call.get("name"),
                "arguments": json.loads(call["arguments"]),
            }
            for call in calls
        ],
        "text": output_text(response["output"]),
    }


def assert_equivalent_final_responses(nonstream: dict, streamed: dict) -> None:
    """Compare contract semantics, never independently sampled prose bytes."""
    for transport, response in (("non-stream", nonstream), ("stream", streamed)):
        if response.get("status") != "completed":
            raise RuntimeError(f"{transport} continuation did not complete: {response}")
        if function_calls(response["output"]):
            raise RuntimeError(
                f"{transport} continuation unexpectedly called another tool: {response}"
            )
        if HEADING not in output_text(response["output"]):
            raise RuntimeError(
                f"{transport} continuation omitted expected heading: {response}"
            )


def main() -> None:
    nonstream_first = request(base_payload(stream=False))
    nonstream_calls = function_calls(nonstream_first["output"])
    if len(nonstream_calls) != 1:
        raise RuntimeError(f"Expected one non-stream function call: {nonstream_first}")
    validate_call(nonstream_calls[0])

    stream_events, stream_first = read_sse(request(base_payload(stream=True), stream=True))
    stream_calls = function_calls(stream_first["output"])
    if len(stream_calls) != 1:
        raise RuntimeError(f"Expected one streamed function call: {stream_first}")
    validate_call(stream_calls[0])

    first_nonstream_semantics = semantic_summary(nonstream_first)
    first_stream_semantics = semantic_summary(stream_first)
    if first_nonstream_semantics != first_stream_semantics:
        raise RuntimeError(
            "Responses stream/non-stream tool semantics differ: "
            f"{first_nonstream_semantics!r} != {first_stream_semantics!r}"
        )

    nonstream_final = request(continuation_payload(nonstream_first, stream=False))
    _, stream_final = read_sse(
        request(continuation_payload(nonstream_first, stream=True), stream=True)
    )
    final_nonstream_semantics = semantic_summary(nonstream_final)
    final_stream_semantics = semantic_summary(stream_final)
    assert_equivalent_final_responses(nonstream_final, stream_final)

    print(
        json.dumps(
            {
                "responses_api": "passed",
                "tool_call": first_nonstream_semantics,
                "continuation": final_nonstream_semantics,
                "streamed_continuation": final_stream_semantics,
                "stream_event_types": sorted(
                    {event.get("type", "") for event in stream_events}
                ),
                "stream_nonstream_tool_semantics_equal": True,
                "final_transport_contracts_passed": True,
                "explicit_reasoning_effort": "xhigh",
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
