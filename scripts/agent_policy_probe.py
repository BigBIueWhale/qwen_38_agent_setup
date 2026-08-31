#!/usr/bin/env python3
"""Live fail-closed checks for the one Qwen3.8 agent policy.

The history fixtures here deliberately contain a second interactive user turn
(user -> assistant-with-reasoning -> user), so they exercise rendering across
a completed conversational episode: historical reasoning must appear in the
rendered prompt, and legacy `reasoning_content` on Chat Completions ingress
must normalize to the canonical `reasoning` field with identical token
accounting. Effort aliasing, weaker-mode rejection, and both phase-budget
gates are asserted on single-turn requests.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request


BASE_URL = "http://127.0.0.1:8000"
MODEL = "qwen3.8-27b-nvfp4-k8v4"


def post(path: str, payload: dict, headers: dict | None = None) -> tuple[int, dict]:
    request_headers = {"Content-Type": "application/json"}
    request_headers.update(headers or {})
    request = urllib.request.Request(
        BASE_URL + path,
        data=json.dumps(payload).encode("utf-8"),
        headers=request_headers,
    )
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        try:
            parsed = json.loads(body)
        except json.JSONDecodeError:
            parsed = {"raw": body}
        return error.code, parsed


def require_rejected(path: str, payload: dict, needle: str) -> int:
    status, body = post(path, payload, {"anthropic-version": "2023-06-01"})
    rendered = json.dumps(body, ensure_ascii=False)
    if status < 400 or needle.lower() not in rendered.lower():
        raise RuntimeError(
            f"unsafe request was not rejected as expected: status={status}, body={body}"
        )
    return status


def tokenize(messages: list[dict], **template_kwargs) -> dict:
    payload: dict = {"model": MODEL, "messages": messages}
    if template_kwargs:
        payload["chat_template_kwargs"] = template_kwargs
    status, response = post("/tokenize", payload)
    if status != 200:
        raise RuntimeError(f"tokenize failed: status={status}, response={response}")
    return response


def chat_prompt_tokens(messages: list[dict], **template_kwargs) -> int:
    payload: dict = {
        "model": MODEL,
        "messages": messages,
        "max_tokens": 1,
    }
    if template_kwargs:
        payload["chat_template_kwargs"] = template_kwargs
    status, response = post("/v1/chat/completions", payload)
    if status != 200:
        raise RuntimeError(
            f"one-token Chat Completions probe failed: {status}, {response}"
        )
    return int(response["usage"]["prompt_tokens"])


def main() -> None:
    hidden_marker = "HISTORICAL_HIDDEN_REASONING_" * 256
    history = [
        {"role": "user", "content": "first turn"},
        {
            "role": "assistant",
            "content": "visible result",
            "reasoning": hidden_marker,
        },
        {"role": "user", "content": "second turn"},
    ]
    stripped_history = [
        history[0],
        {"role": "assistant", "content": "visible result"},
        history[2],
    ]

    default_render = tokenize(history)
    stripped_render = tokenize(stripped_history)
    if default_render["count"] <= stripped_render["count"] + 256:
        raise RuntimeError(
            "historical reasoning was not rendered into the prompt by the "
            f"required margin: with={default_render['count']}, "
            f"without={stripped_render['count']}"
        )

    # /tokenize bypasses the Chat Completions ingress normalizer, so the
    # legacy-alias check must go through a real Chat Completions request.
    legacy_history = [
        history[0],
        {
            "role": "assistant",
            "content": "visible result",
            "reasoning_content": hidden_marker,
        },
        history[2],
    ]
    canonical_tokens = chat_prompt_tokens(history)
    legacy_tokens = chat_prompt_tokens(legacy_history)
    stripped_tokens = chat_prompt_tokens(stripped_history)
    if legacy_tokens != canonical_tokens:
        raise RuntimeError(
            "Chat Completions did not normalize legacy reasoning_content to "
            f"the canonical reasoning field: legacy={legacy_tokens}, "
            f"canonical={canonical_tokens}"
        )
    if legacy_tokens <= stripped_tokens + 256:
        raise RuntimeError(
            "legacy reasoning_content ingress did not render the historical "
            f"trace: legacy={legacy_tokens}, without={stripped_tokens}"
        )

    simple = [{"role": "user", "content": "Return POLICY_OK."}]
    xhigh = tokenize(simple, reasoning_effort="xhigh")
    high = tokenize(simple, reasoning_effort="high")
    maximum = tokenize(simple, reasoning_effort="max")
    if not (xhigh["tokens"] == high["tokens"] == maximum["tokens"]):
        raise RuntimeError("high/max are not exact aliases for Qwen xhigh")

    openai_base = {"model": MODEL, "messages": simple, "max_tokens": 128}
    openai_low = dict(openai_base, reasoning_effort="low")
    openai_disabled = dict(
        openai_base,
        chat_template_kwargs={"enable_thinking": False},
    )
    openai_low_status = require_rejected(
        "/v1/chat/completions", openai_low, "only xhigh"
    )
    openai_disabled_status = require_rejected(
        "/v1/chat/completions", openai_disabled, "cannot be disabled"
    )

    anthropic_base = {
        "model": MODEL,
        "messages": simple,
        "max_tokens": 128,
    }
    anthropic_disabled_status = require_rejected(
        "/v1/messages",
        dict(anthropic_base, thinking={"type": "disabled"}),
        "cannot be disabled",
    )
    anthropic_low_status = require_rejected(
        "/v1/messages",
        dict(anthropic_base, output_config={"effort": "low"}),
        "only high",
    )

    status, adaptive = post(
        "/v1/messages",
        {
            "model": MODEL,
            "messages": simple,
            "max_tokens": 512,
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": "max"},
        },
        {"anthropic-version": "2023-06-01"},
    )
    if status != 200:
        raise RuntimeError(f"Anthropic adaptive/max failed: {status}, {adaptive}")
    if not any(block.get("type") == "thinking" for block in adaptive["content"]):
        raise RuntimeError(f"Anthropic adaptive/max omitted thinking: {adaptive}")

    status, capped = post(
        "/v1/messages",
        {
            "model": MODEL,
            "messages": simple,
            "max_tokens": 256,
            "thinking": {"type": "enabled", "budget_tokens": 32},
            "output_config": {"effort": "xhigh"},
        },
        {"anthropic-version": "2023-06-01"},
    )
    if status != 200:
        raise RuntimeError(f"Anthropic explicit thinking budget failed: {status}, {capped}")
    thinking_text = "".join(
        block.get("thinking", "")
        for block in capped["content"]
        if block.get("type") == "thinking"
    )
    if not thinking_text:
        raise RuntimeError(f"explicit thinking budget produced no thinking block: {capped}")

    status, phase_capped = post(
        "/v1/chat/completions",
        {
            "model": MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Think briefly, then write a numbered list of twenty "
                        "different colors with one color per item. Do not use tools."
                    ),
                }
            ],
            "max_tokens": 1024,
            "final_response_token_budget": 5,
        },
    )
    if status != 200:
        raise RuntimeError(
            f"separate final-response budget failed: {status}, {phase_capped}"
        )
    phase_choice = phase_capped["choices"][0]
    if phase_choice.get("finish_reason") != "length" or phase_choice.get(
        "stop_reason"
    ) != "final_response_token_budget":
        raise RuntimeError(
            "final-response phase did not stop on its dedicated budget: "
            f"{phase_choice}"
        )
    phase_message = phase_choice["message"]
    if not phase_message.get("reasoning"):
        raise RuntimeError(
            f"phase-budget probe produced no separated reasoning: {phase_message}"
        )
    status, final_tokenization = post(
        "/tokenize",
        {
            "model": MODEL,
            "prompt": phase_message.get("content") or "",
            "add_special_tokens": False,
        },
    )
    if status != 200 or final_tokenization.get("count") != 5:
        raise RuntimeError(
            "final-response phase was not exactly five real tokenizer tokens: "
            f"status={status}, tokenization={final_tokenization}, "
            f"message={phase_message}"
        )

    print(
        json.dumps(
            {
                "historical_reasoning_tokens_rendered": default_render["count"]
                - stripped_render["count"],
                "legacy_reasoning_content_chat_ingress_normalized": True,
                "xhigh_high_max_tokenization_identical": True,
                "openai_low_rejected_http": openai_low_status,
                "openai_thinking_disabled_rejected_http": openai_disabled_status,
                "anthropic_low_rejected_http": anthropic_low_status,
                "anthropic_thinking_disabled_rejected_http": anthropic_disabled_status,
                "anthropic_adaptive_max_thinking_present": True,
                "anthropic_explicit_budget": 32,
                "anthropic_explicit_budget_thinking_characters": len(thinking_text),
                "separate_final_response_budget_tokens": 5,
                "separate_final_response_stop_reason": phase_choice["stop_reason"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
