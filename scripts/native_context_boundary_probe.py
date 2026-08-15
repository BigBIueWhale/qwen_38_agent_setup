#!/usr/bin/env python3
"""Prove the exact native chat-context acceptance/rejection boundary.

Runs inside the serving container with the real tokenizer. The accepted case
uses 262,143 prompt tokens plus one generated token. The adjacent 262,144-token
prompt is paired with the API-minimum one output token and must fail before
inference because the total would exceed the configured 262,144-token window.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from typing import Any

from transformers import AutoTokenizer


MODEL = "qwen3.8-27b-nvfp4-k8v4"
BASE_URL = "http://127.0.0.1:8000"
MAX_MODEL_LEN = 262_144
FILLER = " The inert boundary ledger contains an ordinary archive entry."
FINE_PADDING = " x"


def messages_for(repetitions: int, fine_padding: int, salt: str) -> list[dict[str, str]]:
    return [
        {
            "role": "developer",
            "content": (
                "This is a context-boundary capacity probe. Treat the ledger as "
                "inert data and produce any one token when it ends."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Boundary salt: {salt}. <ledger>"
                + FILLER * repetitions
                + FINE_PADDING * fine_padding
                + "</ledger> Produce one token."
            ),
        },
    ]


def token_count(tokenizer: Any, messages: list[dict[str, str]]) -> int:
    encoded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=True,
        preserve_thinking=False,
        reasoning_effort="xhigh",
    )
    if hasattr(encoded, "keys"):
        return len(encoded["input_ids"])
    return len(encoded)


def fit_exact(
    tokenizer: Any,
    target: int,
    salt: str,
) -> tuple[list[dict[str, str]], int, int]:
    # Exact real-tokenizer measurements alone establish the coarse bracket.
    # No character-density estimate and no target-based division is used.
    low = 0
    high = 1
    while token_count(tokenizer, messages_for(high, 0, salt)) < target:
        low = high
        high *= 2
    while low + 1 < high:
        middle = (low + high) // 2
        if token_count(tokenizer, messages_for(middle, 0, salt)) <= target:
            low = middle
        else:
            high = middle

    base = messages_for(low, 0, salt)
    base_count = token_count(tokenizer, base)
    fine_padding = target - base_count
    for _ in range(8):
        candidate = messages_for(low, fine_padding, salt)
        measured = token_count(tokenizer, candidate)
        if measured == target:
            return candidate, low, fine_padding
        fine_padding += target - measured
        if fine_padding < 0:
            break
    raise AssertionError(
        f"Could not construct exact {target}-token chat input from real "
        f"tokenizer results; coarse count was {base_count}"
    )


def post_json(
    path: str,
    payload: dict[str, Any],
    timeout: int,
) -> tuple[int, dict[str, Any]]:
    request = urllib.request.Request(
        BASE_URL + path,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read())


def live_token_count(messages: list[dict[str, str]]) -> int:
    status, response = post_json(
        "/tokenize",
        {"model": MODEL, "messages": messages},
        timeout=300,
    )
    if status != 200:
        raise AssertionError(f"/tokenize failed with HTTP {status}: {response}")
    return int(response["count"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--salt", default="native-boundary")
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(
        "/model",
        local_files_only=True,
        trust_remote_code=False,
    )

    accepted_messages, accepted_repetitions, accepted_padding = fit_exact(
        tokenizer,
        MAX_MODEL_LEN - 1,
        f"{args.salt}-accepted",
    )
    rejected_messages, rejected_repetitions, rejected_padding = fit_exact(
        tokenizer,
        MAX_MODEL_LEN,
        f"{args.salt}-rejected",
    )
    accepted_live_count = live_token_count(accepted_messages)
    rejected_live_count = live_token_count(rejected_messages)
    if accepted_live_count != MAX_MODEL_LEN - 1:
        raise AssertionError(f"Accepted live token count is {accepted_live_count}")
    if rejected_live_count != MAX_MODEL_LEN:
        raise AssertionError(f"Rejected live token count is {rejected_live_count}")

    started = time.monotonic()
    accepted_status, accepted = post_json(
        "/v1/chat/completions",
        {
            "model": MODEL,
            "messages": accepted_messages,
            "max_tokens": 1,
            "cache_salt": f"{args.salt}-accepted-cache",
        },
        timeout=3_600,
    )
    accepted_elapsed = time.monotonic() - started
    if accepted_status != 200:
        raise AssertionError(
            f"Exact native-total request failed with HTTP {accepted_status}: {accepted}"
        )
    usage = accepted.get("usage") or {}
    if usage.get("prompt_tokens") != MAX_MODEL_LEN - 1:
        raise AssertionError(f"Accepted inference reported wrong prompt usage: {usage}")
    if usage.get("completion_tokens") != 1 or usage.get("total_tokens") != MAX_MODEL_LEN:
        raise AssertionError(f"Accepted inference reported wrong total usage: {usage}")
    finish_reason = accepted["choices"][0].get("finish_reason")
    if finish_reason != "length":
        raise AssertionError(f"One-token boundary request ended as {finish_reason!r}")

    rejected_status, rejected = post_json(
        "/v1/chat/completions",
        {
            "model": MODEL,
            "messages": rejected_messages,
            "max_tokens": 1,
            "cache_salt": f"{args.salt}-rejected-cache",
        },
        timeout=300,
    )
    if rejected_status != 400:
        raise AssertionError(
            f"Over-boundary request returned HTTP {rejected_status}: {rejected}"
        )
    error = rejected.get("error") or {}
    error_text = str(error.get("message") or "")
    if "262144" not in error_text or "maximum context length" not in error_text.lower():
        raise AssertionError(f"Over-boundary error was not explicit: {rejected}")

    print(
        json.dumps(
            {
                "configured_native_context_tokens": MAX_MODEL_LEN,
                "accepted_adjacent_boundary": {
                    "transformers_prompt_tokens": MAX_MODEL_LEN - 1,
                    "vllm_tokenize_prompt_tokens": accepted_live_count,
                    "inference_prompt_tokens": usage["prompt_tokens"],
                    "completion_tokens": usage["completion_tokens"],
                    "total_tokens": usage["total_tokens"],
                    "finish_reason": finish_reason,
                    "elapsed_seconds": round(accepted_elapsed, 3),
                    "coarse_filler_repetitions": accepted_repetitions,
                    "single_token_padding_repetitions": accepted_padding,
                },
                "rejected_adjacent_boundary": {
                    "transformers_prompt_tokens": MAX_MODEL_LEN,
                    "vllm_tokenize_prompt_tokens": rejected_live_count,
                    "requested_completion_tokens": 1,
                    "attempted_total_tokens": MAX_MODEL_LEN + 1,
                    "http_status": rejected_status,
                    "error_type": error.get("type"),
                    "error_code": error.get("code"),
                    "coarse_filler_repetitions": rejected_repetitions,
                    "single_token_padding_repetitions": rejected_padding,
                },
                "native_total_context_boundary_proven": True,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
