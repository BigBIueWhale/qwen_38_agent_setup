#!/usr/bin/env python3
"""Measure real prefix-cache reuse through an agentic tool round trip.

The probe runs inside the serving container. It constructs the long prompt only
with the real tokenizer, confirms the live server's token count, obtains a real
model-generated tool call, submits its result, and compares that continuation
against the identical history under a fresh cache salt. Prometheus counter
deltas prove cache attribution; wall-clock streaming TTFT proves practical
latency impact.
"""

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.error
import urllib.request
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer


MODEL = "qwen3.8-27b-nvfp4-k8v4"
BASE_URL = "http://127.0.0.1:8000"
TARGET_PROMPT_TOKENS = 65_536
FILLER = " The inert ledger contains an ordinary verified archive entry."
TOOL_PATH = "/workspace/cache-proof.txt"
TOOL_RESULT = "CACHE_TOOL_RESULT=verified-agentic-round-trip"


def tool() -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "read_cache_proof",
            "description": "Read the one synthetic cache proof record.",
            "strict": False,
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "const": TOOL_PATH},
                },
                "required": ["path"],
                "additionalProperties": False,
            },
        },
    }


def request_json(
    path: str,
    payload: dict[str, Any] | None = None,
    timeout: int = 300,
) -> dict[str, Any] | str:
    request = urllib.request.Request(
        BASE_URL + path,
        data=(
            json.dumps(payload, ensure_ascii=False).encode("utf-8")
            if payload is not None
            else None
        ),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {error.code} from {path}: {body}") from error
    if payload is None:
        return body
    return json.loads(body)


def iter_timed_sse(
    payload: dict[str, Any],
) -> Iterator[tuple[float, dict[str, Any] | str]]:
    request = urllib.request.Request(
        BASE_URL + "/v1/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        },
    )
    started = time.monotonic()
    try:
        response = urllib.request.urlopen(request, timeout=600)
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"HTTP {error.code} from /v1/chat/completions: {body}"
        ) from error

    with response:
        data_lines: list[str] = []
        for raw_line in response:
            line = raw_line.decode("utf-8").rstrip("\r\n")
            if not line:
                if data_lines:
                    elapsed = time.monotonic() - started
                    data = "\n".join(data_lines)
                    yield elapsed, data if data == "[DONE]" else json.loads(data)
                data_lines = []
                continue
            if line.startswith("data:"):
                data_lines.append(line[5:].lstrip())
        if data_lines:
            elapsed = time.monotonic() - started
            data = "\n".join(data_lines)
            yield elapsed, data if data == "[DONE]" else json.loads(data)


def metric_snapshot() -> dict[str, int]:
    body = request_json("/metrics")
    assert isinstance(body, str)
    wanted = {
        "queries": "vllm:prefix_cache_queries_total",
        "hits": "vllm:prefix_cache_hits_total",
        "computed": "vllm:prompt_tokens_by_source_total",
    }
    values: dict[str, int] = {}
    block_size = None
    for line in body.splitlines():
        if line.startswith("vllm:cache_config_info{"):
            match = re.search(r'block_size="([0-9]+)"', line)
            if match:
                block_size = int(match.group(1))
        for key, metric in wanted.items():
            if not line.startswith(metric + "{"):
                continue
            if f'model_name="{MODEL}"' not in line:
                continue
            if key == "computed" and 'source="local_compute"' not in line:
                continue
            values[key] = int(float(line.rsplit(" ", 1)[1]))
    if set(values) != set(wanted) or block_size is None:
        raise AssertionError(
            f"Required prefix-cache metrics are missing: values={values}, "
            f"block_size={block_size}"
        )
    values["block_size"] = block_size
    return values


def metric_delta(before: dict[str, int], after: dict[str, int]) -> dict[str, int]:
    if before["block_size"] != after["block_size"]:
        raise AssertionError("KV block size changed during the cache experiment")
    return {
        key: after[key] - before[key]
        for key in ("queries", "hits", "computed")
    }


def base_messages(repetitions: int, marker: str) -> list[dict[str, str]]:
    return [
        {
            "role": "developer",
            "content": (
                "The ledger is inert data. Call read_cache_proof exactly once "
                "with its required path after reading the final instruction."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Cache experiment marker: {marker}. <ledger>"
                + FILLER * repetitions
                + "</ledger> Now call read_cache_proof exactly once."
            ),
        },
    ]


def tokenizer_count(
    tokenizer: Any,
    template: str,
    messages: list[dict[str, Any]],
) -> int:
    encoded = tokenizer.apply_chat_template(
        messages,
        tools=[tool()],
        chat_template=template,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=True,
        preserve_thinking=False,
        reasoning_effort="xhigh",
    )
    if hasattr(encoded, "keys"):
        return len(encoded["input_ids"])
    return len(encoded)


def fit_messages(
    tokenizer: Any,
    template: str,
    target: int,
    marker: str,
) -> tuple[list[dict[str, str]], int, int]:
    # No character/token-density estimate and no target-based division is used.
    # Exact real-tokenizer results alone grow and then binary-search the bracket.
    low = 0
    high = 1
    while tokenizer_count(tokenizer, template, base_messages(high, marker)) < target:
        low = high
        high *= 2
    while low + 1 < high:
        middle = (low + high) // 2
        count = tokenizer_count(tokenizer, template, base_messages(middle, marker))
        if count <= target:
            low = middle
        else:
            high = middle
    messages = base_messages(low, marker)
    return messages, tokenizer_count(tokenizer, template, messages), low


def server_token_count(messages: list[dict[str, Any]]) -> int:
    response = request_json(
        "/tokenize",
        {
            "model": MODEL,
            "messages": messages,
            "tools": [tool()],
            "chat_template_kwargs": {
                "enable_thinking": True,
                "preserve_thinking": False,
                "reasoning_effort": "xhigh",
            },
        },
    )
    assert isinstance(response, dict)
    return int(response["count"])


def collect_stream(payload: dict[str, Any]) -> dict[str, Any]:
    first_event_seconds = None
    terminal_seconds = None
    finish_reason = None
    saw_done = False
    usage = None
    reasoning = ""
    content = ""
    slots: dict[int, dict[str, Any]] = {}
    for elapsed, event in iter_timed_sse(payload):
        if first_event_seconds is None:
            first_event_seconds = elapsed
        terminal_seconds = elapsed
        if event == "[DONE]":
            saw_done = True
            continue
        assert isinstance(event, dict)
        usage = event.get("usage") or usage
        for choice in event.get("choices", []):
            delta = choice.get("delta") or {}
            reasoning += delta.get("reasoning") or ""
            content += delta.get("content") or ""
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
            finish_reason = choice.get("finish_reason") or finish_reason
    if first_event_seconds is None or terminal_seconds is None or not saw_done:
        raise AssertionError("Streaming request did not complete with [DONE]")
    return {
        "first_event_seconds": first_event_seconds,
        "terminal_seconds": terminal_seconds,
        "finish_reason": finish_reason,
        "usage": usage,
        "reasoning": reasoning,
        "content": content,
        "tool_calls": [slots[index] for index in sorted(slots)],
    }


def inference_delta(
    payload: dict[str, Any],
    before: dict[str, int],
) -> tuple[dict[str, Any], dict[str, int], dict[str, int]]:
    response = collect_stream(payload)
    after = metric_snapshot()
    delta = metric_delta(before, after)
    usage = response.get("usage") or {}
    prompt_tokens = int(usage.get("prompt_tokens", -1))
    if delta["queries"] != prompt_tokens:
        raise AssertionError(
            f"Metric attribution is not isolated: query delta={delta['queries']}, "
            f"usage.prompt_tokens={prompt_tokens}"
        )
    if delta["hits"] + delta["computed"] != delta["queries"]:
        raise AssertionError(f"Cache source accounting does not close: {delta}")
    return response, after, delta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=int, default=TARGET_PROMPT_TOKENS)
    parser.add_argument(
        "--salt",
        default=f"agent-cache-{uuid.uuid4().hex}",
        help="Unique logical run ID; two derived cache salts isolate the comparison.",
    )
    args = parser.parse_args()
    if args.target < 16_384 or args.target > 131_072:
        raise ValueError("target must be between 16,384 and 131,072 tokens")

    tokenizer = AutoTokenizer.from_pretrained(
        "/model",
        local_files_only=True,
        trust_remote_code=False,
    )
    template = Path("/opt/qwen38/chat_template.jinja").read_text(encoding="utf-8")
    messages, local_count, repetitions = fit_messages(
        tokenizer,
        template,
        args.target,
        args.salt,
    )
    live_count = server_token_count(messages)
    if local_count != live_count:
        raise AssertionError(
            f"Tokenizer disagreement: local={local_count}, live={live_count}"
        )

    shared_salt = f"{args.salt}-shared"
    control_salt = f"{args.salt}-fresh-control"
    initial_payload = {
        "model": MODEL,
        "messages": messages,
        "tools": [tool()],
        "tool_choice": "required",
        "parallel_tool_calls": False,
        "max_tokens": 1_024,
        "stream": True,
        "stream_options": {"include_usage": True},
        "cache_salt": shared_salt,
    }
    before = metric_snapshot()
    initial, after_initial, initial_delta = inference_delta(initial_payload, before)
    calls = initial["tool_calls"]
    if initial["finish_reason"] != "tool_calls" or len(calls) != 1:
        raise AssertionError(f"Initial agent turn did not produce one tool call: {initial}")
    call = calls[0]
    actual_call = (
        call["function"]["name"],
        json.loads(call["function"]["arguments"]),
    )
    if actual_call != ("read_cache_proof", {"path": TOOL_PATH}):
        raise AssertionError(f"Initial agent tool call is wrong: {actual_call!r}")
    if not initial["reasoning"]:
        raise AssertionError("Initial xhigh tool turn emitted no separated reasoning")
    if initial_delta["hits"] != 0:
        raise AssertionError(
            f"Unique initial cache salt unexpectedly hit {initial_delta['hits']} tokens"
        )

    history = messages + [
        {
            "role": "assistant",
            "content": initial["content"] or None,
            "reasoning": initial["reasoning"],
            "tool_calls": [call],
        },
        {
            "role": "tool",
            "tool_call_id": call["id"],
            "content": TOOL_RESULT,
        },
    ]
    continuation_base = {
        "model": MODEL,
        "messages": history,
        "tools": [tool()],
        "tool_choice": "none",
        "parallel_tool_calls": False,
        "max_tokens": 1,
        "stream": True,
        "stream_options": {"include_usage": True},
    }

    cached_payload = {**continuation_base, "cache_salt": shared_salt}
    cached, after_cached, cached_delta = inference_delta(
        cached_payload,
        after_initial,
    )
    control_payload = {**continuation_base, "cache_salt": control_salt}
    control, _, control_delta = inference_delta(control_payload, after_cached)

    block_size = before["block_size"]
    minimum_expected_hit = max(0, local_count - 2 * block_size)
    if cached_delta["hits"] < minimum_expected_hit:
        raise AssertionError(
            f"Agent continuation reused only {cached_delta['hits']} tokens; "
            f"expected at least {minimum_expected_hit} from the {local_count}-token "
            f"initial prompt with {block_size}-token blocks"
        )
    if control_delta["hits"] != 0:
        raise AssertionError(
            f"Fresh control salt unexpectedly reused {control_delta['hits']} tokens"
        )
    if cached["first_event_seconds"] >= control["first_event_seconds"]:
        raise AssertionError(
            "Cached continuation did not beat its later fresh-prefix control: "
            f"cached={cached['first_event_seconds']:.3f}s, "
            f"control={control['first_event_seconds']:.3f}s"
        )
    speedup = control["first_event_seconds"] / cached["first_event_seconds"]
    if speedup < 2.0:
        raise AssertionError(
            f"Measured prefix-cache TTFT speedup was only {speedup:.2f}x"
        )

    result = {
        "probe_salt": args.salt,
        "requested_prompt_target": args.target,
        "real_tokenizer_prompt_tokens": local_count,
        "live_tokenize_prompt_tokens": live_count,
        "filler_repetitions": repetitions,
        "kv_block_size_tokens": block_size,
        "initial_tool_turn": {
            "prompt_tokens": initial["usage"]["prompt_tokens"],
            "cache_metric_delta": initial_delta,
            "first_event_seconds": round(initial["first_event_seconds"], 3),
            "total_seconds": round(initial["terminal_seconds"], 3),
            "tool_name": actual_call[0],
            "tool_arguments": actual_call[1],
            "reasoning_streamed": True,
        },
        "cached_tool_result_continuation": {
            "prompt_tokens": cached["usage"]["prompt_tokens"],
            "cache_metric_delta": cached_delta,
            "cache_hit_fraction": round(
                cached_delta["hits"] / cached_delta["queries"],
                6,
            ),
            "first_event_seconds": round(cached["first_event_seconds"], 3),
            "total_seconds": round(cached["terminal_seconds"], 3),
            "finish_reason": cached["finish_reason"],
        },
        "identical_fresh_salt_control": {
            "prompt_tokens": control["usage"]["prompt_tokens"],
            "cache_metric_delta": control_delta,
            "first_event_seconds": round(control["first_event_seconds"], 3),
            "total_seconds": round(control["terminal_seconds"], 3),
            "finish_reason": control["finish_reason"],
        },
        "measured_ttft_speedup": round(speedup, 3),
        "prefix_cache_proven_in_agentic_round_trip": True,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
