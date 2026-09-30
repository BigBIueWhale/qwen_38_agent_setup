#!/usr/bin/env python3
"""Repeated OpenAI and Anthropic tool-call protocol checks for the local server.

The probe also proves the rules every caller lives under: generation names
the agent that owns its KV cache, and a request naming none is refused with
HTTP 400 on every mounted identity surface, streaming or not, before any
status line; and one agent has at most one request in flight, so a request
overlapping its own unfinished predecessor is refused the same way.

Each trial is one conversation under its own agent ID: the tool-call turn,
its streamed redraw where there is one, and the tool-result continuation.
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from http.client import HTTPResponse

from probe_scope import new_conversation


MODEL = "qwen3.8-27b-nvfp4-k8v4"
ANTHROPIC_HEADERS = {"anthropic-version": "2023-06-01"}
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


def post_status(
    url: str, payload: dict, headers: dict | None = None
) -> tuple[int, dict]:
    """POST and return the status with the parsed body, refusal or not."""
    request_headers = {"Content-Type": "application/json"}
    request_headers.update(headers or {})
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers=request_headers,
    )
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read().decode("utf-8", errors="replace"))


def post_stream_status(
    url: str, payload: dict, headers: dict | None = None
) -> tuple[int, dict | str]:
    """POST a streaming request and return its status with its body.

    A refusal comes back parsed. A stream that was admitted instead is read
    to its end and returned as raw text, so no admitted request outlives the
    check that reports it.
    """
    request_headers = {
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
    }
    request_headers.update(headers or {})
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers=request_headers,
    )
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            return response.status, response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as error:
        raw = error.read().decode("utf-8", errors="replace")
        try:
            return error.code, json.loads(raw)
        except json.JSONDecodeError:
            return error.code, raw


def require_kv_scope_refusal(
    path: str, what: str, status: int, body: dict | str
) -> dict:
    """Hold a refusal to HTTP 400 naming ``kv_scope``, and describe it.

    The field is named as ``error.param`` on the OpenAI-shaped surfaces, and
    as an ``invalid_request_error`` whose message names it on the Anthropic
    surface.
    """
    error = body.get("error") if isinstance(body, dict) else None
    if status != 400 or not isinstance(error, dict):
        raise RuntimeError(
            f"{path}: {what} was not refused with HTTP 400: {status} {body}"
        )
    if path == "/v1/messages":
        named = error.get("type") == "invalid_request_error" and "kv_scope" in str(
            error.get("message")
        )
    else:
        named = error.get("param") == "kv_scope"
    if not named:
        raise RuntimeError(f"{path}: the refusal does not name kv_scope: {body}")
    return {"http_status": status, "error_type": error.get("type")}


def unscoped_refusals(base_url: str) -> tuple[dict[str, dict], dict[str, dict]]:
    """Prove that generation names its agent or does not happen.

    Every mounted route that reaches the engine's generative path is sent a
    request that is complete except for ``kv_scope``: the same one-line turn
    everywhere, pre-rendered for the token-in-token-out surface as the
    scale-out layer would send it. The refusal must be HTTP 400 naming the
    field -- never a 500, and never a generation. Every route that streams is
    then sent the same request with ``stream: true``: admission precedes the
    status line, so the refusal must again be HTTP 400 naming the field, never
    a 200 stream that carries the error inside it. The batch route does not
    stream, so it is checked unstreamed only. The render and tokenize
    endpoints allocate nothing and need no agent, so they are not here.
    """
    turn = [{"role": "user", "content": "Reply with one word."}]
    rendered = post_json(
        f"{base_url}/tokenize", {"model": MODEL, "messages": turn}
    )["tokens"]
    surfaces: dict[str, tuple[dict, dict, bool]] = {
        "/v1/chat/completions": (
            {"model": MODEL, "messages": turn, "max_tokens": 1},
            {},
            True,
        ),
        "/v1/chat/completions/batch": (
            {"model": MODEL, "messages": [turn], "max_tokens": 1},
            {},
            False,
        ),
        "/v1/completions": (
            {"model": MODEL, "prompt": turn[0]["content"], "max_tokens": 1},
            {},
            True,
        ),
        "/v1/responses": (
            {
                "model": MODEL,
                "input": turn[0]["content"],
                "max_output_tokens": 1,
                "store": False,
            },
            {},
            True,
        ),
        "/v1/messages": (
            {"model": MODEL, "messages": turn, "max_tokens": 1},
            ANTHROPIC_HEADERS,
            True,
        ),
        "/inference/v1/generate": (
            {
                "model": MODEL,
                "token_ids": rendered,
                "sampling_params": {"max_tokens": 1},
            },
            {},
            True,
        ),
    }
    refusals: dict[str, dict] = {}
    stream_refusals: dict[str, dict] = {}
    for path, (payload, headers, streams) in surfaces.items():
        status, body = post_status(f"{base_url}{path}", payload, headers)
        refusals[path] = require_kv_scope_refusal(
            path, "unscoped generation", status, body
        )
        if not streams:
            continue
        status, body = post_stream_status(
            f"{base_url}{path}", {**payload, "stream": True}, headers
        )
        stream_refusals[path] = require_kv_scope_refusal(
            path, "unscoped streaming generation", status, body
        )
    return refusals, stream_refusals


def iter_sse_events(response: HTTPResponse) -> Iterator[dict | str]:
    """Yield each JSON event of an open SSE response, and ``[DONE]``."""
    for raw_line in response:
        line = raw_line.decode("utf-8").rstrip("\r\n")
        if not line.startswith("data: "):
            continue
        data = line[len("data: ") :]
        yield data if data == "[DONE]" else json.loads(data)


def overlap_refusals(base_url: str) -> dict:
    """Prove that one agent has at most one request in flight.

    A streaming request under a new ID is admitted -- admission precedes the
    status line -- and its first event shows it generating toward a 256-token
    budget. While it is still in flight, a non-streaming and a streaming
    request under the same ID must each be refused with HTTP 400 naming
    ``kv_scope``, before any status line. The first request is then read to its
    own end: the refusals must leave it intact, and it finishes before the
    probe moves on. The ID is never sent again. Each refusal message must name
    the ID it refused.
    """
    path = "/v1/chat/completions"
    scope = new_conversation("overlap")
    turn = [
        {
            "role": "user",
            "content": (
                "List forty different colors, one per line, with a note on each."
            ),
        }
    ]
    request = urllib.request.Request(
        f"{base_url}{path}",
        data=json.dumps(
            {
                "model": MODEL,
                "messages": turn,
                "max_tokens": 256,
                "stream": True,
                "kv_scope": scope,
            },
            ensure_ascii=False,
        ).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
    )
    try:
        first = urllib.request.urlopen(request, timeout=300)
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"HTTP {error.code} from {path} for the request to overlap: {body}"
        ) from error
    refusals: dict[str, dict] = {}
    with first:
        events = iter_sse_events(first)
        opening = next(events, None)
        if not isinstance(opening, dict) or "error" in opening:
            raise RuntimeError(
                f"The request to overlap did not start generating: {opening}"
            )
        overlapping = {
            "model": MODEL,
            "messages": turn,
            "max_tokens": 1,
            "kv_scope": scope,
        }
        for label, stream in (("non_streaming", False), ("streaming", True)):
            if stream:
                status, body = post_stream_status(
                    f"{base_url}{path}", {**overlapping, "stream": True}
                )
            else:
                status, body = post_status(f"{base_url}{path}", overlapping)
            refusal = require_kv_scope_refusal(
                path,
                f"a {label.replace('_', '-')} request overlapping its agent's "
                "unfinished request",
                status,
                body,
            )
            if scope not in str(body["error"].get("message")):
                raise RuntimeError(
                    f"{path}: the overlap refusal does not name {scope!r}: {body}"
                )
            refusals[label] = refusal
        finish_reason = None
        saw_done = False
        for event in itertools.chain([opening], events):
            if event == "[DONE]":
                saw_done = True
                continue
            if "error" in event:
                raise RuntimeError(
                    f"The overlapped request failed after the refusals: {event}"
                )
            for choice in event.get("choices") or []:
                finish_reason = choice.get("finish_reason") or finish_reason
        if not saw_done or finish_reason is None:
            raise RuntimeError(
                "The overlapped request did not finish intact after the refusals: "
                f"done={saw_done}, finish_reason={finish_reason}"
            )
    return {
        "overlapping_requests_refused": refusals,
        "overlapped_request_finish_reason": finish_reason,
    }


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
    # The tool call, its streamed redraw and the continuation are one
    # conversation.
    scope = new_conversation("openai-trial")
    first_request = {
        "model": MODEL,
        "messages": initial,
        "tools": [tool],
        "tool_choice": "auto",
        "max_tokens": 1_024,
        "kv_scope": scope,
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
            "kv_scope": scope,
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
        "kv_scope": scope,
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
    # The tool call and its continuation are one conversation.
    scope = new_conversation("anthropic-trial")
    first = post_json(
        f"{base_url}/v1/messages",
        {
            "model": MODEL,
            "system": system,
            "messages": [first_user],
            "tools": [tool],
            "max_tokens": 1_024,
            "kv_scope": scope,
        },
        ANTHROPIC_HEADERS,
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
            "kv_scope": scope,
        },
        ANTHROPIC_HEADERS,
    )
    answer = "".join(
        block.get("text", "")
        for block in second["content"]
        if block["type"] == "text"
    )
    if second.get("stop_reason") != "end_turn" or HEADING not in answer:
        raise RuntimeError(f"Incorrect Anthropic continuation: {second}")
    return {
        "kv_scope": scope,
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
    refusals, stream_refusals = unscoped_refusals(base_url)
    overlap = overlap_refusals(base_url)
    openai_results = [openai_trial(base_url, markers) for _ in range(args.trials)]
    anthropic_results = [anthropic_trial(base_url) for _ in range(args.trials)]
    print(
        json.dumps(
            {
                "trials_per_protocol": args.trials,
                "openai_passed": len(openai_results),
                "anthropic_passed": len(anthropic_results),
                "elapsed_seconds": round(time.monotonic() - started, 3),
                "unscoped_generation_refused": refusals,
                "unscoped_streaming_generation_refused": stream_refusals,
                "overlap_in_one_agent_refused": overlap,
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
