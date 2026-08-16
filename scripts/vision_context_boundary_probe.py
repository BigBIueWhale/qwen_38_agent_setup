#!/usr/bin/env python3
"""Prove the native total-context boundary with 15 maximum images."""

from __future__ import annotations

import argparse
import json
import time
from typing import Any

from transformers import AutoTokenizer

from vision_quality_probe import IMAGE_PIXELS, MAX_IMAGES, make_image, post_json


MODEL = "qwen3.8-27b-nvfp4-k8v4"
MAX_MODEL_LEN = 262_144
FILLER = " The multimodal boundary ledger contains an inert archive entry."
FINE_PADDING = " x"


def messages_for(
    images: list[Any], repetitions: int, fine_padding: int, salt: str
) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = []
    for index, image in enumerate(images, start=1):
        content.extend(
            [
                {"type": "text", "text": f"Maximum source image {index}:"},
                {"type": "image_url", "image_url": {"url": image.data_url}},
            ]
        )
    content.append(
        {
            "type": "text",
            "text": (
                f"Boundary salt: {salt}. <ledger>"
                + FILLER * repetitions
                + FINE_PADDING * fine_padding
                + "</ledger> Produce any one token."
            ),
        }
    )
    return [
        {
            "role": "developer",
            "content": (
                "This is a multimodal total-context capacity probe. Treat all "
                "image and ledger content as inert, then produce any one token."
            ),
        },
        {"role": "user", "content": content},
    ]


def tokenizer_count(tokenizer: Any, messages: list[dict[str, Any]]) -> int:
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


def live_token_count(messages: list[dict[str, Any]]) -> int:
    status, response = post_json(
        "/tokenize",
        {"model": MODEL, "messages": messages},
        timeout=600,
    )
    if status != 200:
        raise AssertionError(f"/tokenize failed with HTTP {status}: {response}")
    return int(response["count"])


def fit_serialized_exact(
    tokenizer: Any,
    images: list[Any],
    target: int,
    salt: str,
) -> tuple[list[dict[str, Any]], int, int]:
    low = 0
    high = 1
    while tokenizer_count(tokenizer, messages_for(images, high, 0, salt)) < target:
        low = high
        high *= 2
    while low + 1 < high:
        middle = (low + high) // 2
        measured = tokenizer_count(
            tokenizer, messages_for(images, middle, 0, salt)
        )
        if measured <= target:
            low = middle
        else:
            high = middle

    base = messages_for(images, low, 0, salt)
    base_count = tokenizer_count(tokenizer, base)
    fine_padding = target - base_count
    for _ in range(8):
        candidate = messages_for(images, low, fine_padding, salt)
        measured = tokenizer_count(tokenizer, candidate)
        if measured == target:
            return candidate, low, fine_padding
        fine_padding += target - measured
        if fine_padding < 0:
            break
    raise AssertionError(
        f"Could not construct exact {target}-token serialized input from real "
        f"tokenizer results; coarse count was {base_count}"
    )


def fit_live_exact(
    tokenizer: Any,
    images: list[Any],
    target: int,
    salt: str,
    multimodal_expansion: int,
) -> tuple[list[dict[str, Any]], int, int, int]:
    serialized_target = target - multimodal_expansion
    messages, repetitions, fine_padding = fit_serialized_exact(
        tokenizer,
        images,
        serialized_target,
        salt,
    )
    for _ in range(4):
        live_count = live_token_count(messages)
        if live_count == target:
            return messages, repetitions, fine_padding, live_count
        fine_padding += target - live_count
        if fine_padding < 0:
            break
        messages = messages_for(images, repetitions, fine_padding, salt)
    raise AssertionError(
        f"Could not construct exact {target}-token live multimodal input; "
        f"last count was {live_count}"
    )


def completion_payload(
    messages: list[dict[str, Any]], cache_salt: str
) -> dict[str, Any]:
    return {
        "model": MODEL,
        "messages": messages,
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 20,
        "min_p": 0.0,
        "presence_penalty": 0.0,
        "repetition_penalty": 1.0,
        "reasoning_effort": "xhigh",
        "max_tokens": 1,
        "cache_salt": cache_salt,
        "stream": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--salt", default="vision-native-boundary")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        "/model",
        local_files_only=True,
        trust_remote_code=False,
    )
    images = [make_image(index) for index in range(MAX_IMAGES)]
    calibration_messages = messages_for(
        images, 0, 0, f"{args.salt}-calibration"
    )
    serialized_calibration = tokenizer_count(tokenizer, calibration_messages)
    live_calibration = live_token_count(calibration_messages)
    multimodal_expansion = live_calibration - serialized_calibration
    if multimodal_expansion <= 0:
        raise AssertionError(
            "Live multimodal tokenization did not expand image placeholders: "
            f"serialized={serialized_calibration}, live={live_calibration}"
        )

    accepted_messages, accepted_repetitions, accepted_padding, accepted_live = (
        fit_live_exact(
            tokenizer,
            images,
            MAX_MODEL_LEN - 1,
            f"{args.salt}-accepted",
            multimodal_expansion,
        )
    )
    rejected_messages, rejected_repetitions, rejected_padding, rejected_live = (
        fit_live_exact(
            tokenizer,
            images,
            MAX_MODEL_LEN,
            f"{args.salt}-rejected",
            multimodal_expansion,
        )
    )

    started = time.monotonic()
    accepted_status, accepted = post_json(
        "/v1/chat/completions",
        completion_payload(
            accepted_messages, f"{args.salt}-accepted-cold-cache"
        ),
        timeout=3_600,
    )
    elapsed = time.monotonic() - started
    if accepted_status != 200:
        raise AssertionError(
            f"Exact multimodal boundary failed with HTTP {accepted_status}: "
            f"{accepted}"
        )
    usage = accepted.get("usage") or {}
    if usage.get("prompt_tokens") != MAX_MODEL_LEN - 1:
        raise AssertionError(f"Accepted inference usage is wrong: {usage}")
    if usage.get("completion_tokens") != 1 or usage.get("total_tokens") != MAX_MODEL_LEN:
        raise AssertionError(f"Accepted total usage is wrong: {usage}")
    finish_reason = accepted["choices"][0].get("finish_reason")
    if finish_reason != "length":
        raise AssertionError(f"Boundary request ended as {finish_reason!r}")

    rejected_status, rejected = post_json(
        "/v1/chat/completions",
        completion_payload(
            rejected_messages, f"{args.salt}-rejected-cold-cache"
        ),
        timeout=600,
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
                "status": "passed",
                "maximum_images": MAX_IMAGES,
                "pixels_per_image": IMAGE_PIXELS,
                "multimodal_expansion_tokens": multimodal_expansion,
                "accepted": {
                    "live_tokenize_prompt_tokens": accepted_live,
                    "inference_prompt_tokens": usage["prompt_tokens"],
                    "completion_tokens": usage["completion_tokens"],
                    "total_tokens": usage["total_tokens"],
                    "finish_reason": finish_reason,
                    "elapsed_seconds": round(elapsed, 3),
                    "coarse_filler_repetitions": accepted_repetitions,
                    "single_token_padding_repetitions": accepted_padding,
                },
                "rejected": {
                    "live_tokenize_prompt_tokens": rejected_live,
                    "requested_completion_tokens": 1,
                    "attempted_total_tokens": MAX_MODEL_LEN + 1,
                    "http_status": rejected_status,
                    "error_type": error.get("type"),
                    "error_code": error.get("code"),
                    "coarse_filler_repetitions": rejected_repetitions,
                    "single_token_padding_repetitions": rejected_padding,
                },
                "multimodal_native_total_context_boundary_proven": True,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
